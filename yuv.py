"""Work on a clip's own YUV planes, so assembly never converts.

WHY
---
Assembly stream-COPIES most frames and re-encodes only the ones it
changes. Any difference between the two treatments shows up as a step at
the boundary between them -- which is exactly where a repair happens, so
the repair draws attention to the seam it was fixing.

The old bridge path went MP4(yuv420p) -> RGB -> MP4(yuv420p). That loses,
and not for a fixable reason: limited-range 8-bit Y has 219 levels and
full-range RGB has 256, so a round trip through RGB cannot be the
identity and its rounding lands low. Measured at -0.4 luma, or 0.33%,
identically at crf 23, 16 and 8 -- crf-independent, therefore conversion
and not compression. swscale's `accurate_rnd` changed nothing (the
rounding was never the problem), and full-range yuvj420p made it worse.

Doing the work on the planes themselves removes both conversions.
Measured: YUV -> YUV is +0.039 luma at crf 18 and 0.000 at crf 0, i.e.
nothing beyond the compression the copied frames also carry.

It also drops the chroma resample (4:2:0 -> RGB -> 4:2:0), and needs 8x
less memory than the RGB path: 1.5 bytes per pixel against 12 for
float32 RGB.

WHAT WE CAN AND CANNOT EXPRESS HERE
-----------------------------------
A brightness gain g maps exactly: Y and the chroma planes each scale
about their own anchor (16 for Y, 128 for chroma), because if R, G and B
all scale by g then so do Y, U and V. THREE DIFFERENT per-channel gains
do not map -- Y mixes all three -- so the caller checks how far apart
they are and falls back to the RGB path when they diverge. On real
joins they agree to about 0.1%.

A crossfade maps exactly: the transform is linear, so blending planes
gives the same result as blending RGB.

Standalone: numpy + ffmpeg, no torch and no comfy imports.
"""

import logging
import os
import subprocess

from . import frames as fr

_LOG = logging.getLogger("obvpm.h3")

PIX = "yuv420p"
# limited range ("tv"): black sits at 16, chroma is neutral at 128
Y_ANCHOR = 16
C_ANCHOR = 128
# How far apart the per-channel gains may be before a single luma gain
# stops describing them and we hand back to the RGB path.
GAIN_SPREAD_LIMIT = 0.01
# BT.709, matching the luma the rest of the pack measures
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def probe(path):
    """(pix_fmt, width, height) of a clip's video stream, or None."""
    try:
        import av
        with av.open(path) as c:
            s = c.streams.video[0]
            cc = s.codec_context
            return (cc.pix_fmt, int(cc.width), int(cc.height))
    except Exception:
        _LOG.debug("obvpm.h3: could not probe %s", path, exc_info=True)
        return None


def usable(path):
    """Can this clip be worked on as planes without any conversion?

    Only when it ALREADY is yuv420p. Anything else -- a dropped-in H.265
    clip, 4:2:2, 10-bit -- would have to be converted to reach the
    planes, which is the very thing this module exists to avoid, and the
    RGB path handles it no worse.
    """
    got = probe(path)
    return bool(got and got[0] == PIX and got[1] % 2 == 0 and got[2] % 2 == 0)


def frame_bytes(w, h):
    return w * h + 2 * ((w // 2) * (h // 2))


def decode(path, upto=None):
    """Planes of a yuv420p clip: (Y, U, V) uint8 arrays, or None.

    Y is (n, h, w); U and V are (n, h/2, w/2). No conversion happens --
    ffmpeg writes the decoded planes straight out, so the values are the
    ones the file stores.
    """
    import numpy as np
    from .nodes_save import _ffmpeg_exe

    got = probe(path)
    if not got or got[0] != PIX:
        return None
    _pf, w, h = got
    args = [_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path]
    if upto:
        args += ["-frames:v", str(int(upto))]
    args += ["-f", "rawvideo", "-pix_fmt", PIX, "-"]
    try:
        raw = subprocess.run(args, capture_output=True,
                             check=True).stdout
    except Exception:
        _LOG.debug("obvpm.h3: plane decode failed for %s", path,
                   exc_info=True)
        return None
    fsz = frame_bytes(w, h)
    n = len(raw) // fsz
    if n <= 0:
        return None
    ysz, csz = w * h, (w // 2) * (h // 2)
    buf = np.frombuffer(raw[:n * fsz], np.uint8).reshape(n, fsz)
    y = buf[:, :ysz].reshape(n, h, w)
    u = buf[:, ysz:ysz + csz].reshape(n, h // 2, w // 2)
    v = buf[:, ysz + csz:].reshape(n, h // 2, w // 2)
    return y, u, v


def encode(path, planes, crf, fps=None):
    """Encode planes as H.264, with the pack's exact encoder settings.

    The argument list has to stay in step with nodes_save's ffmpeg call:
    assembly stream-copies between the pieces this writes and the ones
    that come straight off disk, and `_mux_pieces` refuses to splice
    pieces whose codec extradata differs.
    """
    import numpy as np
    from .nodes_save import _ffmpeg_exe

    y, u, v = planes
    n, h, w = y.shape
    args = [_ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", PIX, "-s", "%dx%d" % (w, h),
            "-r", str(fps or fr.FPS), "-i", "-",
            "-c:v", "libx264", "-crf", str(int(crf)),
            "-pix_fmt", PIX, path]
    p = subprocess.Popen(args, stdin=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    try:
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(y[i]).tobytes())
            p.stdin.write(np.ascontiguousarray(u[i]).tobytes())
            p.stdin.write(np.ascontiguousarray(v[i]).tobytes())
    except (BrokenPipeError, OSError):
        pass  # ffmpeg died early; its stderr says why
    finally:
        try:
            p.stdin.close()
        except OSError:
            pass
    err = p.stderr.read().decode("utf-8", "replace").strip()
    p.stderr.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg exited %d: %s" % (p.returncode, err[:400]))
    if err:
        _LOG.warning("obvpm.h3: ffmpeg: %s", err[:400])


def slice_planes(planes, a, b):
    y, u, v = planes
    return y[a:b], u[a:b], v[a:b]


def concat(parts):
    import numpy as np
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def luma_gain(rgb_gains):
    """One gain from three per-channel ones, or None if too far apart.

    Y is a weighted mix of R, G and B, so a single scale of Y can only
    stand in for three separate scales when those three agree. Returning
    None is the caller's signal to use the RGB path rather than quietly
    apply an average that is not what was asked for.
    """
    g = [float(x) for x in rgb_gains]
    if max(g) - min(g) > GAIN_SPREAD_LIMIT:
        return None
    return sum(w * x for w, x in zip(LUMA_WEIGHTS, g))


def apply_gain(planes, gains):
    """Scale frames about the limited-range anchors. `gains` is per frame.

    A gain on (Y - 16) is exactly a gain on the RGB values, because
    Y - 16 is proportional to their luma. Chroma scales about 128 for the
    same reason: U and V are differences of the same scaled quantities.
    """
    import numpy as np
    y, u, v = (p.copy() for p in planes)
    n = min(len(gains), y.shape[0])
    for t in range(n):
        g = float(gains[t])
        if abs(g - 1.0) < 1e-6:
            continue
        y[t] = np.clip(
            np.rint((y[t].astype(np.float32) - Y_ANCHOR) * g + Y_ANCHOR),
            0, 255).astype(np.uint8)
        for c in (u, v):
            c[t] = np.clip(
                np.rint((c[t].astype(np.float32) - C_ANCHOR) * g + C_ANCHOR),
                0, 255).astype(np.uint8)
    return y, u, v


def plane_means(planes):
    """Per-frame mean of each plane, as floats."""
    y, u, v = planes
    return (y.mean(axis=(1, 2)), u.mean(axis=(1, 2)), v.mean(axis=(1, 2)))


# The gain field is coarse ON PURPOSE. It corrects lighting-scale
# differences between two renderings of the same moment; going finer
# starts correcting CONTENT, which would pull the incoming frames toward
# the outgoing ones and erase the very rendering we are fading to.
# Measured on a real join (independent 9x14 grid): worst block error
# 13.9 global -> 4.8 at 12x16, frame-to-frame block flicker 0.57 ->
# 0.26, while the residual difference from the parent stayed at 3.7 --
# i.e. the child's rendering survived.
MATCH_GRID = (12, 16)
MATCH_LIMIT = 0.25
# How far the handover ramp may carry a fade away from the level it was
# matched to. It expresses the disagreement between two renderings of one
# moment -- a few percent on real joins (measured 3.9%) -- so a demand
# beyond this is a bad measurement rather than a bad join, and clamping
# is the safe response.
HANDOVER_LIMIT = 0.20


def luma_level(y):
    """Mean luma of a Y plane, anchored -- 0 is black, not 16.

    Ratios have to be taken on the anchored value: Y=126 is not 3%
    brighter than Y=122, it is 110 against 106, which is 3.9%. Scaling
    happens about the anchor too (see apply_gain), so the two agree.
    """
    import numpy as np
    return float(np.asarray(y).astype(np.float32).mean()) - Y_ANCHOR


def opening_level(y, k=4):
    """Where a clip's first delivered frame sits, read off its trend.

    A single frame carries the model's frame-to-frame noise (0.18 luma
    on a real take), and averaging the first few instead would undershoot
    -- an opening is DECAYING back toward its parent, so its mean is not
    its start. Fitting the first k frames and taking the intercept keeps
    the start while dividing the noise down.
    """
    import numpy as np
    n = min(int(k), int(y.shape[0]))
    if n <= 0:
        return 0.0
    lv = y[:n].astype(np.float32).mean(axis=(1, 2)) - Y_ANCHOR
    if n < 2:
        return float(lv[0])
    return float(np.polyfit(np.arange(n, dtype=np.float64),
                            lv.astype(np.float64), 1)[1])


def block_means(a, gh, gw):
    """Anchored block means of a Y plane, as (n, gh, gw).

    Clamps the grid to what the plane can carry, so the caller gets the
    grid it actually got rather than the one it asked for.
    """
    import numpy as np
    n, h, w = a.shape
    gh, gw = max(1, min(int(gh), h)), max(1, min(int(gw), w))
    bh, bw = h // gh, w // gw
    if bh < 2 or bw < 2:
        gh = gw = 1
        bh, bw = h, w
    return (a[:, :gh * bh, :gw * bw].astype(np.float32)
            .reshape(n, gh, bh, gw, bw).mean(axis=(2, 4))) - Y_ANCHOR


def balance_field(shape, y):
    """Rescale a shape field so it moves no whole-frame level.

    A field whose blocks average to 1 does NOT leave the frame mean
    alone: the gain multiplies (Y - 16), so a bright block carries more
    of the frame's level than a dark one and a flat block average lets
    the mean drift. Measured before this: 0.64 luma, against a step the
    level lock had just taken to 0.11 -- the spatial fix was quietly
    undoing part of the level fix.

    Weighting by each block's own anchored level makes the correction
    purely redistributive, which is the whole point: the lock owns the
    level, this owns only how it is spread.
    """
    import numpy as np
    n = min(int(shape.shape[0]), int(y.shape[0]))
    b = block_means(y[:n], shape.shape[1], shape.shape[2])
    if b.shape[1:] != shape.shape[1:]:
        return shape[:n]
    num = (b * shape[:n]).mean(axis=(1, 2))
    den = b.mean(axis=(1, 2))
    k = np.where(np.abs(num) > 1e-6, den / num, 1.0)
    return shape[:n] * k.reshape(n, 1, 1).astype(np.float32)


def _gain_field(src, ref, grid, limit):
    """Per-block gain mapping src onto ref, as (n, gh, gw)."""
    import numpy as np
    gh, gw = grid
    s = block_means(src, gh, gw)
    r = block_means(ref, gh, gw)
    # a near-black block carries no level information; leave it alone
    g = np.where(np.abs(s) > 1.0, r / np.where(np.abs(s) > 1.0, s, 1.0), 1.0)
    return np.clip(g, 1.0 - limit, 1.0 + limit)


def _upsample(field, h, w):
    """Bilinear, from block centres. Nearest-neighbour would leave the
    grid visible as a patchwork of level steps."""
    import numpy as np
    n, gh, gw = field.shape
    if gh == 1 and gw == 1:
        return np.repeat(np.repeat(field, h, axis=1), w, axis=2)
    ys = (np.arange(h) + 0.5) * gh / h - 0.5
    xs = (np.arange(w) + 0.5) * gw / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, gh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, gw - 1)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = np.clip(ys - y0, 0, 1).astype(np.float32)[None, :, None]
    wx = np.clip(xs - x0, 0, 1).astype(np.float32)[None, None, :]
    top = field[:, y0][:, :, x0] * (1 - wx) + field[:, y0][:, :, x1] * wx
    bot = field[:, y1][:, :, x0] * (1 - wx) + field[:, y1][:, :, x1] * wx
    return top * (1 - wy) + bot * wy


def match(src, ref, grid=MATCH_GRID, limit=MATCH_LIMIT, ramp=None):
    """Scale `src` onto `ref`'s level, LOCALLY, about the anchors.

    The two cover the same moment, so this is a registered match rather
    than an estimate -- see crossfade.match_levels for why the fade is
    wrong without it.

    Local rather than one gain per frame, because the incoming clip's
    drift is not spatially uniform. A global match holds the frame
    average exactly while leaving parts of the picture several luma out,
    and since that leftover pattern changes frame to frame it reads as a
    region flickering -- which is worse than the step being fixed, and
    was reported as exactly that.

    `ramp` is an optional per-frame scalar applied ON TOP of the match,
    so the fade lands somewhere other than `ref`'s level -- see
    crossfade.handover_ramp. It multiplies the field after the field's
    own clamp, and carries its own, because the two limits guard
    different things: the field's bounds a per-block estimate, the
    ramp's a whole-frame handover that is deliberately not 1.
    """
    import numpy as np
    n = min(src[0].shape[0], ref[0].shape[0])
    if n <= 0:
        return src
    field = _gain_field(src[0][:n], ref[0][:n], grid, limit)
    if ramp is not None:
        k = np.clip(np.asarray(ramp, np.float32).reshape(-1)[:n],
                    1.0 - HANDOVER_LIMIT, 1.0 + HANDOVER_LIMIT)
        if k.shape[0] < n:
            k = np.concatenate([k, np.full(n - k.shape[0], k[-1]
                                           if k.size else 1.0, np.float32)])
        field = field * k.reshape(n, 1, 1)
    return apply_field(src, field)


# How far the carried field may pull any one block once the mean is
# taken out. The field it comes from is already bounded by MATCH_LIMIT;
# this bounds the SPATIAL part alone, which on real joins is a couple of
# percent. It is deliberately tight: past the overlap the field is an
# extrapolation, not a measurement.
CARRY_LIMIT = 0.10


def carry_field(src, ref, grid=MATCH_GRID, limit=MATCH_LIMIT):
    """The last registered frame's gain field, with its mean divided out.

    What is left is pure spatial SHAPE: "this corner runs 2% low against
    that one", with the whole-frame level removed because the level lock
    owns that and would otherwise apply it twice.

    Only meaningful where src and ref cover the same moment -- i.e.
    inside the overlap. The caller carries it into the frames after,
    where no reference exists; see preview_route for the measured decay
    that justifies it.
    """
    import numpy as np
    n = min(src[0].shape[0], ref[0].shape[0])
    if n <= 0:
        return None
    f = _gain_field(src[0][:n], ref[0][:n], grid, limit)[-1]
    m = float(f.mean())
    if not np.isfinite(m) or abs(m) < 1e-6:
        return None
    return np.clip(f / m, 1.0 - CARRY_LIMIT, 1.0 + CARRY_LIMIT)


def apply_field(planes, field):
    """Scale planes by a per-frame, per-block gain field, about the anchors.

    `field` is (n, gh, gw); it is upsampled bilinearly to each plane's
    size, so the chroma planes get the same gain at the same picture
    position despite being half resolution.
    """
    import numpy as np
    n = min(int(planes[0].shape[0]), int(field.shape[0]))
    out = []
    for plane, anchor in ((planes[0], Y_ANCHOR), (planes[1], C_ANCHOR),
                          (planes[2], C_ANCHOR)):
        p = plane[:n].astype(np.float32)
        g = _upsample(field[:n], p.shape[1], p.shape[2])
        out.append(np.clip(np.rint((p - anchor) * g + anchor),
                           0, 255).astype(np.uint8))
    return tuple(out)


def blend(a, b, weights):
    """Crossfade a into b, plane by plane.

    The YUV<->RGB transform is linear, so this is exactly the result of
    blending in RGB -- without either conversion.
    """
    import numpy as np
    out = []
    n = min(a[0].shape[0], b[0].shape[0], len(weights))
    for k in range(3):
        pa = a[k][:n].astype(np.float32)
        pb = b[k][:n].astype(np.float32)
        w = np.asarray(weights[:n], np.float32).reshape(n, 1, 1)
        out.append(np.clip(np.rint(pa * (1.0 - w) + pb * w),
                           0, 255).astype(np.uint8))
    return tuple(out)
