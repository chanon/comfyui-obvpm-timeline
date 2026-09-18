"""Crossfade a join across the overlap both takes rendered.

WHAT THE OVERLAP IS
-------------------
An extend is handed its parent's latent tail, re-renders exactly that
window as its `pinned_head`, then continues past it. Assembly keeps the
PARENT's pixels for the window and delivers the child from after it, so
the join lands where the parent's rendering of a moment meets the
child's continuation of ITS rendering of the same moment. The two
renderings disagree on level -- measured at a median of 2.87 luma
against a median frame-to-frame noise of 0.44 -- and that difference is
the visible step.

Both renderings of that window exist: the parent's in its delivered
frames, the child's in the `overlap` blob its sidecar carries (see
nodes_save._overlap_bytes). They cover the SAME MOMENT, so a fade
between them is invisible as content and turns the level step into a
gradual handover. At the end of the fade the picture is entirely the
child's rendering, which is exactly what its delivered frames continue
from -- so nothing is discontinuous afterwards.

Duration does not change: the overlap replaces the parent's tail frames
in place rather than being inserted.

WHY THE WHOLE WINDOW
--------------------
The child's disagreement with its context is nearly as large at the
window's START as at its end (measured: it grows only 1.26x across it).
So there is no short fade that hides the step -- fading over 4 frames,
as other suites do, ramps a quarter of it and cuts the rest. The default
is the entire overlap.

WHAT IT DOES NOT FIX
--------------------
Flicker. The oscillation seen on some joins lives in the delivered
frames AFTER the overlap, where the model has left its conditioning
behind. The fade ends exactly where that begins. Use the level lock's
swing straightening for it -- the two compose, and `preview_route` runs
both.
"""

import logging
import os
import tempfile

from . import frames as fr
from . import mctx

_LOG = logging.getLogger("obvpm.h3")

# Equal-power keeps perceived brightness (and loudness) flat through the
# handover. A linear fade dips in the middle wherever the two sources
# differ, which is precisely the case we are here for.
EQUAL_POWER = "equal_power"
LINEAR = "linear"

_CACHE = {}
_CACHE_CAP = 32


def weights(n, shape=EQUAL_POWER):
    """Fade weights for the incoming side: 0 -> 1 over n frames."""
    import numpy as np
    if n <= 1:
        return np.ones(max(n, 0))
    t = np.linspace(0.0, 1.0, n)
    if shape == LINEAR:
        return t
    return np.sin(t * np.pi / 2.0) ** 2


def available(clip_path, tail=False):
    """Frames of overlap this take can contribute, 0 if none.

    `tail` asks for its re-render of the pinned TAIL -- what a prepend
    kept, covering the clip it arrives at -- rather than of the pinned
    head, which is what an extend kept.
    """
    try:
        header = mctx.read_header(mctx.sidecar_path(clip_path))
    except (OSError, ValueError):
        return 0
    n = int(header.get(
        "overlap_tail_frames" if tail else "overlap_frames", 0) or 0)
    # older takes predate the blob; the header alone is not proof
    if n <= 0:
        return 0
    return n


def load(clip_path, tail=False):
    """(images, audio) of a take's own re-render of its join, or None.

    images is an IMAGE tensor (n,h,w,3) 0..1; audio the usual dict.
    """
    key = os.path.abspath(clip_path)
    try:
        # the side is part of the key: one take can carry both
        key = (key, os.stat(clip_path).st_mtime_ns, bool(tail))
    except OSError:
        return None
    if key in _CACHE:
        return _CACHE[key]

    from .nodes_save import read_overlap
    raw = read_overlap(clip_path, tail=tail)
    if not raw:
        return None
    out = None
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "overlap.mp4")
            with open(p, "wb") as f:
                f.write(raw)
            from .preview_route import _decode_audio, _decode_images
            out = (_decode_images(p), _decode_audio(p))
    except Exception:
        _LOG.exception("obvpm.h3: could not read the overlap stored with %s",
                       clip_path)
        return None
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = out
    return out


_PLANES = {}


def load_planes(clip_path, tail=False):
    """The overlap as YUV planes, or None if it cannot be read that way.

    Same material as load(), without the RGB conversion -- so a fade
    built from it costs nothing beyond the re-encode itself.
    """
    from . import yuv
    key = os.path.abspath(clip_path)
    try:
        key = (key, os.stat(clip_path).st_mtime_ns, bool(tail))
    except OSError:
        return None
    if key in _PLANES:
        return _PLANES[key]
    from .nodes_save import read_overlap
    raw = read_overlap(clip_path, tail=tail)
    if not raw:
        return None
    out = None
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "overlap.mp4")
            with open(p, "wb") as f:
                f.write(raw)
            out = yuv.decode(p)
    except Exception:
        _LOG.exception("obvpm.h3: could not read the overlap planes of %s",
                       clip_path)
        return None
    if out is None:
        return None
    if len(_PLANES) >= _CACHE_CAP:
        _PLANES.clear()
    _PLANES[key] = out
    return out


def blend_planes(parent_tail, child_head, shape=EQUAL_POWER, match=True,
                 ramp=None):
    """blend_images, on planes. See that function for the reasoning."""
    from . import yuv
    n = min(int(parent_tail[0].shape[0]), int(child_head[0].shape[0]))
    if n <= 0:
        return parent_tail
    incoming = (yuv.match(yuv.slice_planes(child_head, 0, n),
                          yuv.slice_planes(parent_tail, 0, n), ramp=ramp)
                if match else yuv.slice_planes(child_head, 0, n))
    return yuv.blend(yuv.slice_planes(parent_tail, 0, n), incoming,
                     weights(n, shape))


def handover_ramp(n, ratio, shape=EQUAL_POWER):
    """Per-frame scalar carrying a matched fade onto the CHILD's level.

    Matching pins the fade's far end to the parent, which is right when
    the level lock is going to correct the delivered frames afterwards:
    the fade hands over content, the lock hands over level, and neither
    does the other's job twice.

    With no lock coming, that pin is the worst possible choice. The step
    is measured FROM the parent's level, so ending there guarantees it
    survives whole -- measured 4.19 across the fade against 4.15 with no
    fade at all. The fade spends its whole length relocating the step
    rather than removing it.

    This ramps the match target from the parent's level to the child's
    own, along the same curve as the blend, so the fade hands over both
    at once and the step lands at 0.00.

    Why a ramp and not simply skipping the match -- which also ends on
    the child. Because the incoming rendering's own level path is not
    smooth: measured 5.1 luma peak-to-peak of wobble around its trend,
    against 0.18 of ordinary frame-to-frame noise in the parent, with
    single-frame jumps up to 2.99. Not matching lets all of that
    through, in the middle of the fade, as a flash. Matching and then
    ramping divides that path out and substitutes one we choose: worst
    jump 0.49, i.e. the parent's own noise floor. The endpoint is the
    same; what happens over the 22 frames between is not.

    It is still second best. The ramp follows the child UP to its
    opening level, and the child then decays back down on its own over
    the second that follows -- stepless, but a swell. The level lock
    pulls the child DOWN to where the parent was heading instead, so
    nothing moves at all (0.11 with both on). This is what to do when
    the lock is off, not a reason to turn it off.
    """
    import numpy as np
    return 1.0 + (float(ratio) - 1.0) * weights(n, shape)


def forget():
    _CACHE.clear()
    _PLANES.clear()


def _pixel_pin(header, place):
    """Was this take's `place` window encoded from a clip's pixels?"""
    for p in mctx.parse_pins(header or {}):
        if (p.get("place") == place
                and p.get("source_kind") == "clip_pixels"):
            return True
    return False


def usable(left, right, max_frames=None):
    """How many frames this join can crossfade over. 0 = it cannot.

    Requires a genuine extend at its NATURAL join: a cut on either side
    means the two clips no longer meet where the overlap was rendered,
    and fading there would blend two different moments.
    """
    from . import levellock
    if left.get("gap") or right.get("gap"):
        return 0
    if levellock._linked(left.get("header"), right.get("header")) != "extends":
        return 0
    # the parent must play to its own end, and the child start at its own
    # beginning -- otherwise the overlap is not where the join now is
    if right.get("enter"):
        return 0
    lh = left.get("header") or {}
    delivered = int(lh.get("delivered_frames", 0) or 0)
    exit_f = left.get("exit")
    if exit_f is not None and delivered and int(exit_f) != delivered:
        # A window encoded from PIXELS may sit anywhere in its source --
        # footage with no sidecar is cut wherever the eye wants, and the
        # pixel route takes any frame. There the natural place is the
        # frame the lineage derives, exactly as usable_tail reads it: the
        # overlap's last frame is the one before that exit. Latent-grade
        # takes keep the stricter rule they always had.
        if not _pixel_pin(right.get("header"), "before"):
            return 0
        from . import nodes_assemble as na
        try:
            want_exit, _, _ = na._derive_seam(left, right)
        except Exception:
            return 0
        if want_exit is None or int(exit_f) != int(want_exit):
            return 0
        delivered = int(exit_f)
    n = available(right["path"])
    if n <= 0:
        return 0
    n = min(n, delivered or n)
    if max_frames:
        n = min(n, int(max_frames))
    return max(0, int(n))


def usable_tail(left, right, max_frames=None):
    """How many frames a PREPEND join can crossfade over. 0 = it cannot.

    The mirror of `usable`. There the child kept its re-render of the
    parent's tail; here the take kept its re-render of the clip it
    arrives at, so the blend lands on the RIGHT clip's opening and the
    overlap comes off the LEFT one.

    "Natural place" means something different on this side. An extend
    requires the child to start at frame 0; a prepend's target is
    EXPECTED to enter partway in -- at exactly the frame the lineage
    derives. Anything else is a manual cut, which moves the join away
    from the moment the overlap was rendered for.
    """
    from . import levellock, nodes_assemble as na
    if left.get("gap") or right.get("gap"):
        return 0
    if levellock._linked(left.get("header"),
                         right.get("header")) != "prepends":
        return 0
    # the take must play to its own end -- a cut there moves the join
    lh = left.get("header") or {}
    delivered = int(lh.get("delivered_frames", 0) or 0)
    exit_f = left.get("exit")
    if exit_f is not None and delivered and int(exit_f) != delivered:
        return 0
    # ...and the target must enter exactly where the lineage says
    try:
        _, want_enter, _ = na._derive_seam(left, right)
    except Exception:
        return 0
    if int(right.get("enter") or 0) != int(want_enter or 0):
        return 0
    n = available(left["path"], tail=True)
    if n <= 0:
        return 0
    # it cannot cover more of the target than the target plays
    rh = right.get("header") or {}
    r_delivered = int(rh.get("delivered_frames", 0) or 0)
    hi = right.get("exit")
    playable = ((int(hi) if hi is not None else r_delivered)
                - int(right.get("enter") or 0))
    if playable > 0:
        n = min(n, playable)
    if max_frames:
        n = min(n, int(max_frames))
    return max(0, int(n))


def match_levels(src, ref, max_gain=1.35, ramp=None):
    """Scale `src` onto `ref`'s level, per frame and per channel.

    NOTE this is the GLOBAL match, and it is the fallback path only.
    yuv.match does it locally, which matters: a per-frame gain holds the
    frame average exactly while leaving regions several luma out, and
    that leftover pattern shifts frame to frame as a visible local
    flicker. This path survives for sources that are not yuv420p -- and
    since a fade needs an overlap, which only our own (always yuv420p)
    takes carry, it is reached only when the OUTGOING clip is foreign.

    The two cover the SAME MOMENT -- that is the whole point of the
    overlap -- so this is an exact registered match, not an estimate.

    It is not optional. Measured on a real join, the model's re-render
    STARTS at its conditioning's level and drifts steadily brighter
    across the window (123.0 -> 130.3 against a parent sitting at 122).
    Fading into it therefore walks the picture UP by 7 luma over a
    second and the delivered frames then decay back down -- the fade
    introduces a brightness swell instead of removing a step. Matching
    first makes the fade level-neutral, so it does the one job it is
    actually good at: handing over CONTENT.
    """
    import torch
    n = min(int(src.shape[0]), int(ref.shape[0]))
    if n <= 0:
        return src
    s = src[:n].mean(dim=(1, 2))                 # (n, 3)
    r = ref[:n].mean(dim=(1, 2))
    g = torch.where(s.abs() > 1e-6, r / s, torch.ones_like(s))
    g = g.clamp(1.0 / max_gain, max_gain)
    if ramp is not None:
        from .yuv import HANDOVER_LIMIT
        k = torch.as_tensor(ramp, dtype=g.dtype).reshape(-1)[:n]
        if int(k.shape[0]) < n:
            k = torch.cat([k, k[-1:].expand(n - int(k.shape[0]))])
        g = g * k.clamp(1.0 - HANDOVER_LIMIT,
                        1.0 + HANDOVER_LIMIT).view(n, 1)
    return (src[:n] * g.view(n, 1, 1, 3)).clamp_(0.0, 1.0)


def blend_images(parent_tail, child_head, shape=EQUAL_POWER, match=True,
                 ramp=None):
    """Fade parent_tail into child_head. Both (n,h,w,3); returns (n,h,w,3).

    `match` levels the incoming side onto the outgoing one first -- see
    match_levels for why that is the difference between a fade that
    removes a seam and one that adds a swell.
    """
    import torch
    n = min(int(parent_tail.shape[0]), int(child_head.shape[0]))
    if n <= 0:
        return parent_tail
    incoming = (match_levels(child_head[:n], parent_tail[:n], ramp=ramp)
                if match else child_head[:n])
    w = weights(n, shape)
    wt = torch.from_numpy(w).to(parent_tail.dtype).view(n, 1, 1, 1)
    out = parent_tail[:n] * (1.0 - wt) + incoming * wt
    return out.clamp_(0.0, 1.0)


def blend_audio(parent_wave, child_wave, rate, shape=EQUAL_POWER):
    """Same handover for sound, over whatever both sides can cover."""
    import torch
    if parent_wave is None or child_wave is None:
        return parent_wave
    n = min(int(parent_wave.shape[-1]), int(child_wave.shape[-1]))
    if n <= 0:
        return parent_wave
    import numpy as np
    t = np.linspace(0.0, 1.0, n)
    w = t if shape == LINEAR else np.sin(t * np.pi / 2.0) ** 2
    wt = torch.from_numpy(w).to(parent_wave.dtype)
    return parent_wave[..., :n] * (1.0 - wt) + child_wave[..., :n] * wt


# 5ms per side, NOT the 15ms an overlapping crossfade uses. The two are
# not the same operation: a crossfade sums two runs, so its length costs
# nothing audibly, while this has no second run to fade into and simply
# dips to silence. Tapering both sides at 15ms would leave a 30ms hole,
# which in sustained ambience is audible as a soft blip -- one artefact
# traded for another. 5ms out + 5ms in is 10ms, under the threshold at
# which the ear hears a gap in continuous sound, and still far longer
# than the single-sample edge that causes the click.
DECLICK_MS = 5.0


def taper_audio(wave, rate, ms=DECLICK_MS, head=False, tail=False):
    """Fade a run in and/or out over `ms`: a join with no overlap.

    Not a crossfade -- there is no second source to fade into -- but it
    removes the sample-level discontinuity that clicks. Kept short
    because it IS an artefact of its own, just a much smaller one.

    BOTH sides of a boundary have to be tapered. Fading only the
    outgoing run moves the discontinuity rather than removing it: the
    step is then between silence and the incoming run's first sample,
    which on a loud entry is a bigger jump than the original. So the
    caller tapers the tail of one part and the head of the next.
    """
    import numpy as np
    import torch
    if wave is None or rate <= 0 or not (head or tail):
        return wave
    n = int(round(ms / 1000.0 * rate))
    n = min(n, int(wave.shape[-1]) // 2)
    if n <= 1:
        return wave
    ramp = np.cos(np.linspace(0.0, np.pi / 2.0, n)) ** 2
    out = wave.clone()
    if tail:
        out[..., -n:] = out[..., -n:] * torch.from_numpy(ramp).to(out.dtype)
    if head:
        out[..., :n] = out[..., :n] * torch.from_numpy(
            ramp[::-1].copy()).to(out.dtype)
    return out


def audio_frames_samples(frames, rate):
    """Samples covering `frames` video frames, the pack's rounding."""
    return int(round(frames / float(fr.FPS) * rate))
