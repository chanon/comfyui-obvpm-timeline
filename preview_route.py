"""Stateless preview + export endpoints for the Timeline node.

POST /obvpm/h3/preview_cut  {"sequence": "<the Timeline node's text>",
                             "crf": 19, "base_folder": "h3",
                             "preview_filename": "obvpm_h3_preview",
                             "probe": false}
->  {"filename", "subfolder", "type": "output", "starts": [f...],
     "total", "cached", "v"}

POST /obvpm/h3/export  {"sequence", "crf", "base_folder",
                        "preview_filename",
                        "export_filename_prefix": "full"}
->  {"path": "<output-relative path of the written MP4>", "cached"}

The timeline widget plays the preview as ONE continuous stream -- the
only way a preview can be truly gapless (two <video> elements can never
be sample-locked). For all-copyable sequences this is a packet copy,
sub-second; seams needing bridges decode only the bridged clips.

The preview is a SINGLE file per Timeline node, at
output/<base_folder>/<preview_filename>.mp4, overwritten on every
build; a meta JSON beside it records the content key so staleness
survives page reloads AND ComfyUI restarts (it deliberately does NOT
live in temp, which is wiped at startup). Because the URL never changes
per build, players must cache-bust with the returned "v". Export =
promote: ensure the preview is current (rebuild if stale), then copy it
beside itself under export_filename_prefix with the normal counter --
the export is byte-identical to the preview by construction.

These are stateless utility routes, NOT review gates: they drive no
generation, hold no state, and keep the loop-less design intact
(DESIGN.md section 1).
"""

import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile

import folder_paths

from . import condstore
from . import crossfade as xf
from . import frames as fr
from . import levellock
from . import mctx
from . import nodes_assemble as na
from . import yuv
from .nodes_save import H3SaveVideoWithMCtx

_LOG = logging.getLogger("obvpm.h3")

DEFAULT_PREVIEW_NAME = "obvpm_h3_preview"

# Bumped whenever the BUILD PATH changes what it writes for an unchanged
# sequence. The cache key covers the inputs (clips, cuts, crf, repair
# settings) but cannot see the code, so without this a preview built by
# an older version is served as current -- and a fix looks like it did
# nothing. Raise it with any change to how frames are decoded, corrected
# or encoded.
BUILD_VERSION = 10


def _decode_audio(path):
    """Audio-only decode: (waveform [1,C,T] float tensor, rate) or None."""
    import av
    import torch
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        s = c.streams.audio[0]
        sr = int(s.codec_context.sample_rate or 0)
        if not sr:
            return None
        ch = int(s.codec_context.channels or 2)
        layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(ch, "stereo")
        res = av.audio.resampler.AudioResampler(
            format="fltp", layout=layout, rate=sr)
        chunks = []
        for frame in c.decode(s):
            for rf in res.resample(frame):
                chunks.append(torch.from_numpy(rf.to_ndarray()))
        for rf in res.resample(None):  # flush the resampler tail
            chunks.append(torch.from_numpy(rf.to_ndarray()))
        if not chunks:
            return None
        return {"waveform": torch.cat(chunks, dim=1).unsqueeze(0),
                "sample_rate": sr}


def _decode_images(path):
    from comfy_api.input_impl import VideoFromFile
    return VideoFromFile(path).get_components().images


def _gap_size(entries):
    """(width, height) for black filler, taken from a real clip."""
    for e in entries:
        if e.get("gap"):
            continue
        header = e.get("header") or {}
        if header.get("width") and header.get("height"):
            return int(header["width"]), int(header["height"])
        import av
        with av.open(e["path"]) as c:
            st = c.streams.video[0]
            return int(st.codec_context.width), int(st.codec_context.height)
    raise ValueError(
        "preview: the sequence is only empty space -- there is no clip to "
        "take the frame size from. Add a clip, or shorten the gap away.")


def _clip_frames(e):
    if e.get("gap"):
        return int(e["gap"])
    if e["header"]:
        d = int(e["header"].get("delivered_frames", 0) or 0)
        if d:
            return d
    return na._scan_keyframes(e["path"])[1]


def _preview_paths(base_folder, preview_filename):
    """(mp4_path, meta_path, filename, subfolder) under the OUTPUT root.

    The preview lives beside the clips it cuts (base_folder), not in
    temp -- findable, and it survives restarts together with its meta.
    Refused if the combination escapes the output root (same traversal
    rule as the sequence paths).
    """
    name = str(preview_filename or DEFAULT_PREVIEW_NAME).strip().strip("/")
    sub = str(base_folder or "").strip().strip("/")
    rel = "%s/%s" % (sub, name) if sub else name
    root = os.path.abspath(folder_paths.get_output_directory())
    path = os.path.abspath(os.path.join(root, rel + ".mp4"))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(
            "preview location escapes the output folder: %r" % rel)
    sub, name = os.path.split(os.path.relpath(path, root))
    return path, path + ".json", name, sub.replace(os.sep, "/")


def _measure_encode_gain(frames, crf, tmpdir):
    """Per-channel gain that cancels ONE encode generation's loss.

    Decoding to RGB and encoding back to yuv420p is not lossless even at
    crf 0: the range conversion rounds, and it rounds DOWN on average.
    Measured across two clips at crf 23/16/8 the result was the same
    every time -- a multiplicative loss of about 0.33% (-0.40 luma on a
    122-luma clip, -0.33 on a 108-luma one), independent of quality.

    That matters because assembly stream-COPIES most frames and
    re-encodes only the ones it changes. The re-encoded run therefore
    sits ~0.4 luma below the copied frames on either side of it, and on
    footage whose own frame-to-frame movement is 0.20 that reads as a
    visible darkening of exactly the repaired stretch -- the seam repair
    drawing attention to the seam.

    Calibrated per build on the real frames rather than hardcoded: it is
    a property of this ffmpeg build's conversion, not of our maths.
    """
    import torch
    n = min(4, int(frames.shape[0]))
    if n <= 0:
        return None
    probe = frames[:n]
    p = os.path.join(tmpdir, "calibrate.mp4")
    try:
        H3SaveVideoWithMCtx._encode_mp4(p, probe, None, crf)
        back = _decode_images(p)
    except Exception:
        _LOG.debug("obvpm.h3: encode calibration failed", exc_info=True)
        return None
    m = min(int(back.shape[0]), n)
    if m <= 0:
        return None
    src = probe[:m].mean(dim=(0, 1, 2))
    got = back[:m].mean(dim=(0, 1, 2))
    g = torch.where(got > 1e-6, src / got, torch.ones_like(got))
    # a real generation loss is a fraction of a percent; anything larger
    # is a measurement gone wrong, not something to act on
    return g.clamp(0.98, 1.02)


class _Compensator:
    """Applies the measured gain to everything this build re-encodes.

    Measured once at the build's crf and reused for every bridge, even
    though bridges are now written at each clip's OWN crf (see
    nodes_assemble.source_crf). That is deliberate rather than sloppy:
    the RGB round trip's loss is crf-INDEPENDENT -- measured -0.402,
    -0.401 and -0.419 luma at crf 23, 16 and 8 -- because it comes from
    limited-range Y having 219 levels against RGB's 256, which is a
    property of the conversion and not of the compression. Re-measuring
    per crf would cost an encode each to land on the same number.
    """

    def __init__(self, crf, tmpdir):
        self.crf = crf
        self.tmpdir = tmpdir
        self.gain = None
        self.tried = False

    def __call__(self, seg):
        if seg is None or int(seg.shape[0]) == 0:
            return seg
        if not self.tried:
            self.tried = True
            self.gain = _measure_encode_gain(seg, self.crf, self.tmpdir)
            if self.gain is not None:
                _LOG.debug("obvpm.h3: encode generation gain %s",
                           [round(float(x), 5) for x in self.gain])
        if self.gain is None:
            return seg
        return (seg * self.gain).clamp(0.0, 1.0)


def _short(entry):
    """A clip's bare filename, for progress labels."""
    c = str(entry.get("clip") or "")
    return c.rsplit("/", 1)[-1] or "clip"


def _progress_sink(target):
    """Send build progress to the browser, tagged with WHICH preview.

    One websocket message per step. Every Timeline and Result Preview
    hears all of them, so the payload names the preview file it belongs
    to -- two nodes building at once must not drive each other's bar.
    """
    try:
        from server import PromptServer
    except Exception:
        return None

    def sink(stage, frac):
        # send_sync hops to the event loop thread internally, which is
        # what makes this safe from the worker thread build_preview runs
        # on (the route hands it to asyncio.to_thread).
        PromptServer.instance.send_sync("obvpm.h3.build_progress", {
            "target": target, "stage": stage,
            "progress": round(float(frac), 4),
        })
    return sink


def _read_meta(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# The timeline's own values, as build_preview receives them. A join may
# override any of them from its own sequence line -- see
# nodes_assemble.split_seam_opts.
def _seam_plan(entries, base, loop=False):
    """The settings in force at each join, indexed by the CHILD entry.

    A join's overrides ride on the clip to its RIGHT, so entry i's
    settings describe the join between i-1 and i -- the same index the
    corrections themselves are keyed by. Absent keys inherit `base`.
    In a looping cut entry 0 is the child of the wrap join, so its
    settings describe the join out of the last entry.
    """
    out = []
    for i, e in enumerate(entries):
        one = dict(base)
        prev = entries[i - 1] if i else (entries[-1] if loop else None)
        if prev is not None and na.masked_continuation(
                prev.get("header"), e.get("header")):
            # a masked join is latent-identical: repairs would correct
            # noise (measured to INTRODUCE flicker). Defaults off; the
            # join's own seam_opts below still override.
            one["level_lock"] = False
            one["crossfade"] = False
        one.update(e.get("seam_opts") or {})
        out.append(one)
    return out


def _join_fixes(entries, plan, loop=False):
    """({index: opening gains}, {index: fade frames}, {ramps},
    {index: closing gains}) for a sequence.

    The two repairs COMPOSE -- they do not overlap, and neither replaces
    the other:

      the crossfade ends on the parent's level (the incoming frames are
      matched onto it before blending), so it removes the switch between
      two renderings of the same moment;

      the level lock then corrects the child's DELIVERED frames, which
      still open off-level -- the model's drift does not stop at the end
      of the pinned window, it peaks there and decays over the second
      that follows.

    An earlier version dropped the step correction wherever a fade ran,
    on the theory that the fade "already handed the level over". It does
    not: it hands over to frames that are themselves drifted, so
    suppressing the lock left the whole opening step in place and the
    fade merely walked into it.

    The third return says where a fade must carry the level over ITSELF
    -- see crossfade.handover_ramp. That needs BOTH conditions:

      the level lock is off, so nothing else is going to do it; AND
      the join actually measured a step worth carrying.

    The second is not redundant, and getting it wrong is a regression on
    the joins that were already fine. levellock.plan returns no entry
    for a join whose step and swing sit inside the frame-to-frame noise
    -- there is nothing to correct. Reading "no entry" as "nobody is
    fixing this, so ramp it" therefore fires the ramp on every clean
    join in the sequence and moves a level that measured flat. The lock
    being OFF is what means "nobody is fixing this"; the measurement is
    what means "there is something to fix".

    The same plan() call answers the second question, so a clean join
    means exactly the same thing whichever mode you are in.
    """
    def opt(i, key, fallback):
        return plan[i].get(key, fallback) if i < len(plan) else fallback

    fades, tfades = {}, {}
    for i in range(1, len(entries)):
        if not opt(i, "crossfade", True):
            continue
        cap = opt(i, "crossfade_frames", 0) or None
        n = xf.usable(entries[i - 1], entries[i], max_frames=cap)
        if n > 0:
            fades[i] = n
            continue
        # the same join read backwards: the take on the LEFT kept the
        # re-render, and the blend lands on the RIGHT clip's opening
        n = xf.usable_tail(entries[i - 1], entries[i], max_frames=cap)
        if n > 0:
            tfades[i] = n

    # Locked joins and ramped ones are disjoint by construction: a join
    # whose lock is ON gets the correction, one whose lock is OFF may
    # get the fade's handover ramp instead. Both are decided per join
    # now, so one sequence can carry a mix.
    def lock_opts(i):
        if not opt(i, "level_lock", True):
            return None
        return {"frames": opt(i, "level_lock_frames", levellock.FRAMES),
                "fix_step": True,
                "fix_swing": bool(opt(i, "level_lock_flicker", True))}

    # two directions: `locks` corrects a child's OPENING (an extend),
    # `tail_locks` corrects a take's CLOSING (a prepend, working
    # backwards from the join). Both are keyed by the entry they touch.
    # the wrap join of a loop is planned like any other: its opening
    # correction lands on entry 0, its closing one on the last entry.
    # No crossfade crosses the wrap (the blend would have to straddle
    # the file's end); bridge joins default it off regardless.
    locks, tail_locks = levellock.plan(entries, opts_for=lock_opts,
                                       loop=loop)

    # Where the lock is off but a fade runs, the fade has to carry the
    # level itself -- and only where the join measured a step worth
    # carrying. fix_swing off: a fade cannot straighten a wobble.
    def ramp_opts(i):
        if opt(i, "level_lock", True) or i not in fades:
            return None
        return {"frames": opt(i, "level_lock_frames", levellock.FRAMES),
                "fix_step": True, "fix_swing": False}

    ramps = set(levellock.plan(entries, opts_for=ramp_opts,
                               loop=loop)[0]) & set(fades)
    return locks, fades, ramps, tail_locks, tfades


def _prepend_fields(entries, tfades, local):
    """{take index: carried field} for prepend fades, measured up front.

    The extend case can record its field mid-build because the clip that
    needs it is patched later. A prepend inverts that: the fade lands on
    the clip being arrived at, while the lock that wants the field runs
    on the take before it. Measuring here keeps one direction of data
    flow instead of two passes over the whole build.
    """
    out = {}
    for i, n in (tfades or {}).items():
        if (i - 1) not in local or n <= 0:
            continue
        prev = entries[i - 1]
        enter = int(entries[i].get("enter") or 0)
        try:
            over = xf.load_planes(prev["path"], tail=True)
            if over is None:
                continue
            got = yuv.decode(entries[i]["path"], upto=enter + n)
            if got is None or got[0].shape[0] < enter + n:
                continue
            field = yuv.carry_field(
                yuv.match(yuv.slice_planes(over, 0, n),
                          yuv.slice_planes(got, enter, enter + n)),
                yuv.slice_planes(got, enter, enter + n))
        except Exception:
            _LOG.exception("obvpm.h3: could not measure the carried field "
                           "for the prepend at %s", prev.get("clip"))
            continue
        if field is not None:
            out[i - 1] = field
    return out


def _merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _rgb_luma(images):
    """Mean BT.709 luma of an IMAGE tensor, on its own 0..1 scale.

    No anchor: RGB is full range, so black really is 0 and a ratio taken
    here means what it says. The YUV path anchors at 16 instead --
    yuv.luma_level.
    """
    w = yuv.LUMA_WEIGHTS
    return float((images[..., 0] * w[0] + images[..., 1] * w[1]
                  + images[..., 2] * w[2]).mean())


def _handover(child, ramps, i, n, tail_level, read_open,
              floor=1.0):
    """The fade's level target: the parent, or a ramp onto the child.

    Returns None for "match to the parent and stop there", which is
    right both when the level lock is going to correct this child's
    delivered frames -- the fade then owns content, the lock owns level
    -- and when the join measured clean, where there is no level to
    carry and moving one would be inventing a drift.

    `ramps` holds the joins where neither is true, decided in
    _join_fixes. Deciding it here from `locks` alone was WRONG: an
    absent lock entry also means "this join needed nothing", so the
    ramp fired on every clean join in the sequence.
    """
    if n <= 0 or (i + 1) not in ramps:
        return None
    # a near-black tail carries no level information, so the ratio
    # would be noise over noise. `floor` is in the caller's units --
    # anchored luma for planes, 0..1 for RGB.
    if tail_level is None or tail_level <= floor:
        return None
    try:
        open_level = read_open(child["path"])
    except Exception:
        _LOG.exception("obvpm.h3: could not read the opening level of %s",
                       child.get("clip"))
        return None
    if open_level is None:
        return None
    return xf.handover_ramp(n, open_level / tail_level)


def _carry_gains(gains, carry, n, y=None):
    """Fold a carried spatial field into the lock's per-frame gains.

    Returns an (n, gh, gw) field. The lock's gain sets each frame's
    level; the carried field only says how that level is DISTRIBUTED,
    and it fades to flat on the same ramp the lock uses -- so by the end
    of the window the correction is exactly what it was before.

    Measured on the one join whose parent and child both carry the data:
    a field measured inside the overlap removes 69-80% of the per-block
    discrepancy one frame later, 51-64% four frames later, and is still
    reducing it at twelve. It never made a frame worse at any of the six
    measurement points tried. The field's self-correlation decays 0.91
    -> 0.64 across those twelve frames, which is why the ramp is the
    lock's length rather than longer.
    """
    import numpy as np
    gh, gw = carry.shape
    ramp = 1.0 - np.arange(float(n)) / float(n)
    shape = 1.0 + (carry.reshape(1, gh, gw) - 1.0) * ramp.reshape(n, 1, 1)
    if y is not None:
        # purely redistributive, or it eats into the lock's own work
        shape = yuv.balance_field(shape, y)
    return np.asarray(gains[:n], np.float32).reshape(n, 1, 1) * shape


def _patch_clip_planes(entries, i, enter, hi, locks, fades,
                       ramps=(), carry=None, local=(), tail_locks=None,
                       tfades=None):
    """(planes, changed ranges) working entirely in YUV, or None.

    None means "this clip cannot be done losslessly" -- it is not
    yuv420p, or the level lock's three per-channel gains diverge too far
    for one luma gain to stand in for them. The caller then uses the RGB
    path, which is no worse than it was before.
    """
    g = locks.get(i)
    # A prepend corrects THIS clip's CLOSING instead of the next clip's
    # opening: same join, same measurement, other end.
    gt = (tail_locks or {}).get(i)
    fade = fades.get(i + 1, 0)
    # A prepend fade blends this clip's OPENING with the re-render the
    # take before it kept -- the same operation as the extend fade
    # below, on the other end and with the sides swapped.
    tfade = (tfades or {}).get(i, 0)
    if g is None and gt is None and not fade and not tfade:
        return None, []
    path = entries[i]["path"]
    if not yuv.usable(path):
        return None, None            # None ranges = "fall back to RGB"

    gains = tgains = None
    for src, name in ((g, "gains"), (gt, "tgains")):
        if src is None:
            continue
        rows = [yuv.luma_gain(row) for row in src]
        if any(x is None for x in rows):
            _LOG.info("obvpm.h3: %s needs per-channel correction; using "
                      "the RGB path for it", entries[i].get("clip"))
            return None, None
        if name == "gains":
            gains = rows
        else:
            tgains = rows

    child = entries[i + 1] if fade else None
    head = xf.load_planes(child["path"]) if fade else None
    avail = xf.available(child["path"]) if fade else 0
    if fade and (head is None or not avail):
        return None, None            # overlap unreadable as planes
    prev = entries[i - 1] if tfade else None
    thead = xf.load_planes(prev["path"], tail=True) if tfade else None
    tavail = xf.available(prev["path"], tail=True) if tfade else 0
    if tfade and (thead is None or not tavail):
        return None, None            # tail overlap unreadable as planes
    fade_n = min(fade, hi - enter, avail, head[0].shape[0]) if fade else 0
    tfade_n = (min(tfade, hi - enter, tavail, thead[0].shape[0])
               if tfade else 0)
    lock_n = min(len(gains), hi - enter) if gains else 0
    tlock_n = min(len(tgains), hi - enter) if tgains else 0
    if lock_n <= 0 and tlock_n <= 0 and fade_n <= 0 and tfade_n <= 0:
        return None, []

    planes = yuv.decode(path, upto=hi)
    if planes is None or planes[0].shape[0] < hi:
        return None, None
    y, u, v = (p.copy() for p in planes)
    ranges = []
    if tfade_n > 0:
        # The prepend fade, in planes. Align at the JOIN: the take's
        # overlap re-renders this clip's opening window, so its FIRST n
        # frames cover this clip's first n from `enter` -- the mirror of
        # the extend fade, where the overlap's LAST n cover the parent's
        # last n.
        inc = yuv.slice_planes(thead, 0, tfade_n)
        seg_in = yuv.slice_planes((y, u, v), enter, enter + tfade_n)
        # The reference is THIS clip: it already exists, possibly with
        # children of its own, so the arriving take is what yields --
        # the same direction the closing lock corrects in. blend_planes'
        # own `match` levels the incoming onto the outgoing, which is
        # the wrong way round here, so the match is done explicitly and
        # the blend told not to repeat it.
        inc = yuv.match(inc, seg_in)
        # the field for the take's CLOSING lock was measured up front
        # (see _prepend_fields) -- this direction of the dependency runs
        # backwards through a forward pass
        seg = xf.blend_planes(inc, seg_in, match=False)
        y[enter:enter + tfade_n], u[enter:enter + tfade_n],             v[enter:enter + tfade_n] = seg
        ranges.append((enter, enter + tfade_n))
    if lock_n > 0:
        seg_in = yuv.slice_planes((y, u, v), enter, enter + lock_n)
        # the field the fade measured at THIS clip's own opening, if the
        # option is on and the join in front of us actually faded
        # `local` is the set of CHILD indices that asked for it, so a
        # sequence can mix per join. This clip is the child here.
        cf = carry.get(i) if (carry and i in local) else None
        seg = (yuv.apply_field(seg_in,
                               _carry_gains(gains, cf, lock_n, seg_in[0]))
               if cf is not None
               else yuv.apply_gain(seg_in, gains[:lock_n]))
        y[enter:enter + lock_n], u[enter:enter + lock_n], \
            v[enter:enter + lock_n] = seg
        ranges.append((enter, enter + lock_n))
    if tlock_n > 0:
        # the window sits AT the join, so when it has to be shortened it
        # is the far end that goes -- keep the frames nearest the join.
        # Runs before the fade so that a fade over the same tail wins,
        # exactly as the opening lock yields to it.
        seg_in = yuv.slice_planes((y, u, v), hi - tlock_n, hi)
        rows = tgains[len(tgains) - tlock_n:]
        # the spatial field the prepend fade measured at the join this
        # take arrives at -- same use as the opening lock's, other end
        cf = carry.get(i) if (carry and i in local) else None
        seg = (yuv.apply_field(seg_in,
                               _carry_gains(rows, cf, tlock_n, seg_in[0]))
               if cf is not None else yuv.apply_gain(seg_in, rows))
        y[hi - tlock_n:hi], u[hi - tlock_n:hi], v[hi - tlock_n:hi] = seg
        ranges.append((hi - tlock_n, hi))
    if fade_n > 0:
        # align the tails: the overlap's LAST n frames cover the
        # parent's last n
        inc = yuv.slice_planes(head, avail - fade_n, avail)

        def _open(path):
            first = yuv.decode(path, upto=4)
            return (None if first is None or first[0].shape[0] < 1
                    else yuv.opening_level(first[0]))

        ramp = _handover(entries[i + 1], ramps, i, fade_n,
                         yuv.luma_level(y[hi - 1:hi]), _open)
        tail = yuv.slice_planes((y, u, v), hi - fade_n, hi)
        # ...and here it is the PARENT, recording for the child in front
        if carry is not None and (i + 1) in local:
            # measured against the parent's own pixels BEFORE the blend
            # overwrites them -- afterwards the tail is partly the child
            # already and the field would read as flat
            got = yuv.carry_field(inc, tail)
            if got is not None:
                carry[i + 1] = got
        seg = xf.blend_planes(tail, inc, ramp=ramp)
        y[hi - fade_n:hi], u[hi - fade_n:hi], v[hi - fade_n:hi] = seg
        ranges.append((hi - fade_n, hi))
    return (y, u, v), _merge(ranges)


def _patch_clip(entries, i, enter, hi, locks, fades, images=None,
                ramps=(), tail_locks=None, tfades=None):
    """(patched images or None, changed ranges) for one entry.

    Decodes nothing when nothing changes, so a sequence with the options
    off costs exactly what it did before.

    A correction forces the WHOLE clip to be re-encoded, not just the
    frames it touches: our takes are written with a single keyframe, so
    there is no later IDR to resume stream-copying from. Correcting 12
    frames therefore costs one decode and one encode of that clip. The
    alternative -- keyframes every second -- would cost size on every
    take ever saved to make this one operation cheaper, which is the
    wrong trade.

    The decode goes through _decode_images like every other frame in
    this module. A hand-rolled PyAV conversion is NOT interchangeable:
    it disagreed by up to 0.17 (of 0..1) on the same frames, enough to
    make a corrected segment visibly mismatch the segments around it.
    """
    g = locks.get(i)
    # a prepend corrects THIS clip's closing -- see _patch_clip_planes
    gt = (tail_locks or {}).get(i)
    # this clip is the PARENT of the join in front of it
    fade = fades.get(i + 1, 0)
    # ...and the TARGET of the join behind it, when that one is a
    # prepend: the blend lands on this clip's opening, from the re-render
    # the take on the left kept
    tfade = (tfades or {}).get(i, 0)
    if g is None and gt is None and not fade and not tfade:
        return None, []

    child = entries[i + 1] if fade else None
    got = xf.load(child["path"]) if fade else None
    avail = xf.available(child["path"]) if fade else 0
    if fade and not (got and got[0] is not None and avail):
        _LOG.warning("obvpm.h3: %s advertises an overlap but it could not "
                     "be read; not crossfading", child.get("clip"))
        fade = 0
    fade_n = min(fade, hi - enter, avail) if fade else 0
    lock_n = min(len(g), hi - enter) if g is not None else 0
    tlock_n = min(len(gt), hi - enter) if gt is not None else 0
    if lock_n <= 0 and tlock_n <= 0 and fade_n <= 0:
        return None, []

    tgot = xf.load(entries[i - 1]["path"], tail=True) if tfade else None
    tavail = (xf.available(entries[i - 1]["path"], tail=True)
              if tfade else 0)
    if tfade and not (tgot and tgot[0] is not None and tavail):
        _LOG.warning("obvpm.h3: %s advertises a tail overlap but it could "
                     "not be read; not crossfading",
                     entries[i - 1].get("clip"))
        tfade = 0
    tfade_n = min(tfade, hi - enter, tavail) if tfade else 0

    src = images if images is not None else _decode_images(entries[i]["path"])
    if src is None or int(src.shape[0]) == 0:
        return None, []
    out = src.clone()
    ranges = []
    if tfade_n > 0:
        # the take's re-render of THIS clip's opening, levelled onto it
        # -- the established clip is the reference, so the new take
        # yields, which is the same direction the closing lock works in
        inc = tgot[0][:tfade_n]
        head = out[enter:enter + tfade_n]
        blended = xf.blend_images(xf.match_levels(inc, head), head,
                                  match=False)
        out[enter:enter + tfade_n] = blended
        ranges.append((enter, enter + tfade_n))
    if lock_n > 0:
        out[enter:enter + lock_n] = levellock.apply_gains(
            out[enter:enter + lock_n], g[:lock_n])
        ranges.append((enter, enter + lock_n))
    if tlock_n > 0:
        # shortened from the FAR end: the frames nearest the join are
        # the ones that carry the correction. Before the fade, so a fade
        # over the same tail still wins.
        out[hi - tlock_n:hi] = levellock.apply_gains(
            out[hi - tlock_n:hi], gt[len(gt) - tlock_n:])
        ranges.append((hi - tlock_n, hi))
    if fade_n > 0:
        # align the tails: the overlap's LAST n frames are the ones
        # covering the parent's last n. The fade runs last so that on a
        # clip short enough for the two to overlap it wins there -- it
        # is the correction that actually spans the join.
        head = got[0][avail - fade_n:avail]

        def _open(path):
            img = _decode_images(path)
            return (None if img is None or int(img.shape[0]) == 0
                    else _rgb_luma(img[:1]))

        ramp = _handover(entries[i + 1], ramps, i, fade_n,
                         _rgb_luma(out[hi - 1:hi]), _open, floor=1.0 / 255)
        out[hi - fade_n:hi] = xf.blend_images(out[hi - fade_n:hi], head,
                                              ramp=ramp)
        ranges.append((hi - fade_n, hi))
    return out, _merge(ranges)


class _Progress:
    """Weighted progress over a build, reported to whoever asked for it.

    Steps are NOT equal: copying a clip's packets is near-instant while a
    corrected clip is decoded and re-encoded whole (single keyframe --
    there is no later IDR to resume copying from). Weighting by that
    distinction is the difference between a bar that crawls then jumps
    and one that means something.
    """

    COPY, WORK, MUX = 1.0, 12.0, 2.0

    def __init__(self, sink):
        self.sink = sink
        self.total = 0.0
        self.done = 0.0

    def plan(self, entries, locks, fades, tail_locks=(), tfades=()):
        for i, e in enumerate(entries):
            if e.get("gap"):
                self.total += self.WORK
                continue
            heavy = (i in locks or i in tail_locks or (i + 1) in fades
                     or i in tfades
                     or e.get("enter") or e.get("exit") is not None)
            self.total += self.WORK if heavy else self.COPY
        self.total += self.MUX
        return self

    def step(self, weight, stage):
        self.done += weight
        self.emit(stage)

    def emit(self, stage):
        if not self.sink or self.total <= 0:
            return
        try:
            self.sink(stage, max(0.0, min(1.0, self.done / self.total)))
        except Exception:
            _LOG.debug("obvpm.h3: progress sink failed", exc_info=True)


def build_preview(sequence, crf, probe=False, base_folder="",
                  preview_filename=None, level_lock=True,
                  level_lock_frames=levellock.FRAMES,
                  level_lock_flicker=True, crossfade=True,
                  crossfade_frames=0, audio_declick=True,
                  on_progress=None, level_lock_local=True):
    """Returns (filename, subfolder, starts, total, cached, key).

    One file per node at output/<base_folder>/<preview_filename>.mp4,
    overwritten on rebuild; the meta JSON beside it holds the content
    key, so `cached` means "the file on disk was built from exactly this
    sequence/crf/source state". probe=True never builds.

    The seam-repair options all fold into the cache key, so toggling one
    rebuilds rather than serving a preview made under the old setting.
    """
    import torch

    entries = na.resolve_sequence(sequence)
    loop = na.sequence_loops(sequence)

    played = []
    for e in entries:
        n = _clip_frames(e)
        hi = n if e["exit"] is None else min(e["exit"], n)
        if hi <= e["enter"]:
            raise ValueError(
                "preview: the cuts leave nothing of %s"
                % (e["clip"] or "the empty space"))
        e["frames"] = n
        played.append(hi - e["enter"])
    starts, total = [], 0
    for p in played:
        starts.append(total)
        total += p

    key_src = [["~gap", e["gap"], 0, 0, None] if e.get("gap")
               else [e["clip"], os.stat(e["path"]).st_size,
                     os.stat(e["path"]).st_mtime_ns, e["enter"], e["exit"]]
               for e in entries]
    base_opts = {
        "level_lock": bool(level_lock),
        "level_lock_frames": int(level_lock_frames),
        "level_lock_flicker": bool(level_lock_flicker),
        "level_lock_local": bool(level_lock_local),
        "crossfade": bool(crossfade),
        "crossfade_frames": int(crossfade_frames),
        "audio_declick": bool(audio_declick),
    }
    seam_plan = _seam_plan(entries, base_opts, loop)
    # The RESOLVED settings go in the key, not the timeline's defaults:
    # a per-join override changes the output, so a preview built under
    # one must not be served for another. `loop` too: the same lines
    # build a different file once they wrap.
    fixes = [[one[k] for k in sorted(base_opts)] for one in seam_plan]
    key = hashlib.sha256(
        json.dumps([key_src, int(crf), fixes, bool(loop),
                    BUILD_VERSION]).encode()).hexdigest()[:16]
    out_path, meta_path, name, sub = _preview_paths(base_folder,
                                                    preview_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    current = (os.path.isfile(out_path)
               and _read_meta(meta_path).get("key") == key)
    if current or probe:
        return name, sub, starts, total, current, key

    (locks, fades, ramps, tail_locks,
     tfades) = _join_fixes(entries, seam_plan, loop)
    # the joins whose child asked for the carried spatial field (entry
    # 0 is a child too when the cut loops)
    locals_ = {i for i in range(0 if loop else 1, len(entries))
               if seam_plan[i].get("level_lock_local", True)}
    if locks or fades:
        overrides = sum(1 for e in entries if e.get("seam_opts"))
        _LOG.info("obvpm.h3: seam repair -- level lock on %d join(s), "
                  "crossfade on %d%s", len(locks), len(fades),
                  ", %d join(s) with their own settings" % overrides
                  if overrides else "")
    prog = _Progress(on_progress).plan(entries, locks, fades,
                                       tail_locks=tail_locks,
                                       tfades=tfades)
    prog.emit("reading audio")

    audio_parts, have_audio, sample_rate = [], True, None
    # which joins the AUDIO crossfade actually covered, so the declick
    # can taper the incoming side of every boundary it did not
    _faded_audio = {}
    # gaps are placeholders, so their silence is sized after the real
    # clips have established the rate and channel count
    for i, e in enumerate(entries):
        if e.get("gap"):
            audio_parts.append({"silence": int(e["gap"])})
            continue
        a = _decode_audio(e["path"]) if have_audio else None
        if a is None:
            have_audio = False
            continue
        if sample_rate is None:
            sample_rate = a["sample_rate"]
        elif a["sample_rate"] != sample_rate:
            have_audio = False
            continue
        hi = e["frames"] if e["exit"] is None else e["exit"]
        lo_s = round(e["enter"] / fr.FPS * sample_rate)
        hi_s = round(hi / fr.FPS * sample_rate)
        wav = a["waveform"]
        part = wav[..., lo_s:min(hi_s, wav.shape[-1])]
        # sound gets the same handover the picture does: the fade runs
        # over the same span, so a level change and a timbre change
        # arrive together instead of a frame apart
        fade = fades.get(i + 1, 0)
        faded_here = False
        if fade and sample_rate:
            got = xf.load(entries[i + 1]["path"])
            avail = xf.available(entries[i + 1]["path"])
            child_a = got[1] if got else None
            if child_a is not None and avail:
                n = min(fade, avail)
                ns = xf.audio_frames_samples(n, sample_rate)
                cw = child_a["waveform"]
                ca = cw[..., max(0, cw.shape[-1] - ns):]
                ns = min(ns, int(part.shape[-1]), int(ca.shape[-1]))
                if ns > 1:
                    part = part.clone()
                    part[..., -ns:] = xf.blend_audio(
                        part[..., -ns:], ca[..., -ns:], sample_rate)
                    faded_here = True
                    _faded_audio[i + 1] = True
        # per boundary: the setting on the clip AFTER this one owns the
        # join in front of it, which is the boundary being tapered
        # _seam_plan seeds every entry from base_opts, so the
        # fallback here can only fire on a malformed plan -- and when it
        # does it must still agree with the timeline's own setting. A
        # literal here would be a second, quieter default.
        declick_here = (seam_plan[i + 1].get("audio_declick",
                                             base_opts["audio_declick"])
                        if i + 1 < len(seam_plan)
                        else seam_plan[0].get("audio_declick",
                                              base_opts["audio_declick"])
                        if loop and seam_plan
                        else base_opts["audio_declick"])
        if declick_here and sample_rate:
            # No overlap to fade into: soften the discontinuity instead,
            # on BOTH sides of it. A boundary is tapered when the join
            # there carries no crossfade -- including the edges of empty
            # space, where sound drops to digital silence and comes back
            # from it, which is the most click-prone boundary of all.
            # The cut's own start and end are left alone: that is not a
            # join, and fading them would quietly alter the top and tail
            # of every export. UNLESS the cut loops: then the end runs
            # straight into the start on every repeat, and that IS a
            # join -- the most exposed one in the file.
            # keyed on whether a fade ACTUALLY happened, not on whether
            # one was planned: an overlap that carries no audio would
            # otherwise leave the boundary with neither treatment
            tail = (i + 1 < len(entries) or loop) and not faded_here
            head = (i > 0 or loop) and not _faded_audio.get(i, False)
            if head or tail:
                part = xf.taper_audio(part, sample_rate,
                                      head=head, tail=tail)
        audio_parts.append(part)
    audio = None
    real = [w for w in audio_parts if not isinstance(w, dict)]
    if have_audio and real and sample_rate:
        channels = int(real[0].shape[-2])
        filled = [torch.zeros(1, channels,
                              int(round(w["silence"] / fr.FPS * sample_rate)))
                  if isinstance(w, dict) else w
                  for w in audio_parts]
        audio = {"waveform": torch.cat(filled, dim=-1),
                 "sample_rate": sample_rate}

    tmp = out_path + ".tmp.mp4"
    try:
        with tempfile.TemporaryDirectory() as btd:
            pieces = []
            gap_w = gap_h = None
            # every re-encoded run has to land at the level of the
            # copied runs it sits between
            compensate = _Compensator(crf, btd)
            # filled by each clip's fade, read by the NEXT clip's lock;
            # the build walks entries in order, so the parent is always
            # patched before the child that needs its field.
            #
            # A PREPEND runs the other way: the field is measured on the
            # clip being arrived at, and wanted by the take BEFORE it --
            # which a forward pass has already patched. So those are
            # measured up front. It costs one short decode per prepend
            # fade (the target's opening only) and nothing at all when
            # there are none.
            carry = _prepend_fields(entries, tfades, locals_)
            for i, e in enumerate(entries):
                if e.get("gap"):
                    # black filler, encoded through THE encode site so it
                    # splices with the real clips instead of forcing the
                    # whole preview into the re-encode fallback
                    prog.emit("building empty space")
                    if gap_w is None:
                        gap_w, gap_h = _gap_size(entries)
                    bp = os.path.join(btd, "gap_%d.mp4" % i)
                    H3SaveVideoWithMCtx._encode_mp4(
                        bp, torch.zeros(int(e["gap"]), gap_h, gap_w, 3),
                        None, crf)
                    pieces.append({"path": bp, "from": 0,
                                   "to": int(e["gap"])})
                    prog.step(prog.WORK, "building empty space")
                    continue
                enter = e["enter"]
                hi = e["frames"] if e["exit"] is None else e["exit"]
                heavy = (i in locks or i in tail_locks
                         or (i + 1) in fades or i in tfades)
                # A clip only leaves the copy path for one of two
                # reasons, and the label should say which. A repair
                # rewrites its pixels, so those frames no longer match
                # the bytes on disk; a cut starts it somewhere without a
                # keyframe, so the run up to the next one has to be
                # re-encoded. Everything else is copied packet for
                # packet and never decoded at all.
                trimmed = bool(enter) or e["exit"] is not None
                work = ("improving the seam at %s" if heavy
                        else "trimming %s" if trimmed
                        else "copying %s")
                prog.emit(work % _short(e))
                # planes first: no conversion, so a repaired run lands at
                # the same level as the copied runs it splices between
                planes, changed = _patch_clip_planes(
                    entries, i, enter, hi, locks, fades,
                    tail_locks=tail_locks, tfades=tfades, ramps=ramps,
                    carry=carry, local=locals_)
                patched = None
                if changed is None:          # not plane-workable
                    planes = None
                    if i in locals_ and carry.get(i) is not None:
                        _LOG.info("obvpm.h3: %s is not plane-workable, so "
                                  "the carried field cannot be applied to "
                                  "it; its level lock stays whole-frame",
                                  _short(e))
                    patched, changed = _patch_clip(
                        entries, i, enter, hi, locks, fades, ramps=ramps,
                        tail_locks=tail_locks, tfades=tfades)
                if not changed and enter == 0 and e["exit"] is None:
                    pieces.append({"path": e["path"], "from": 0,
                                   "to": e["frames"]})
                    prog.step(prog.COPY, "copying %s" % _short(e))
                    continue
                kfs, n_total = na._scan_keyframes(e["path"])
                imgs = patched
                # A corrected range can never be copied: those frames no
                # longer match the bytes on disk. Where the clip HAS
                # later keyframes the rest still splices losslessly;
                # where it does not (our own takes, single IDR) the
                # planner bridges the remainder, which is the honest
                # cost of touching a frame at all.
                spans, cursor = [], enter
                for ra, rb in changed:
                    if ra > cursor:
                        spans.append((cursor, ra, False))
                    spans.append((max(ra, enter), min(rb, hi), True))
                    cursor = min(rb, hi)
                if cursor < hi:
                    spans.append((cursor, hi, False))
                if not spans:
                    spans = [(enter, hi, False)]
                plan = []
                for sa, sb, forced in spans:
                    if sb <= sa:
                        continue
                    if forced:
                        plan.append(("bridge", sa, sb))
                    else:
                        plan.extend(na._plan_pieces(
                            sa, None if sb >= n_total else sb, kfs, n_total))
                for kind, a, b in plan:
                    if kind == "copy":
                        pieces.append({"path": e["path"],
                                       "from": a, "to": b})
                        continue
                    bp = os.path.join(btd, "bridge_%d_%d.mp4" % (i, a))
                    # the CLIP's crf, not the caller's: a bridge is
                    # spliced between stream-copied packets of this same
                    # clip, and pic_init_qp lives in the PPS, so a
                    # different crf means different extradata and
                    # _mux_pieces refuses the whole splice. See
                    # nodes_assemble.source_crf.
                    bcrf = na.source_crf(e["path"], crf)
                    if planes is not None:
                        # straight from planes to the encoder: nothing to
                        # compensate, because nothing was converted
                        yuv.encode(bp, yuv.slice_planes(planes, a, b), bcrf)
                    else:
                        if imgs is None:  # decode only clips that bridge
                            imgs = _decode_images(e["path"])
                        H3SaveVideoWithMCtx._encode_mp4(
                            bp, compensate(imgs[a:b]), None, bcrf)
                    pieces.append({"path": bp, "from": 0, "to": b - a})
                prog.step(prog.WORK, work % _short(e))
            prog.emit("assembling")
            na._mux_pieces(pieces, tmp, audio)
            prog.step(prog.MUX, "assembling")
            na.replace_when_free(tmp, out_path)
        _LOG.debug("obvpm.h3: preview built at %s (%d clips, %d frames)",
                  out_path, len(entries), total)
    except na._StreamMismatch as why:
        _LOG.warning("obvpm.h3: preview smart-cut not possible (%s); "
                     "re-encoding", why)
        try:
            os.remove(tmp)
        except OSError:
            pass
        # the fallback re-encodes everything, so the earlier weighting no
        # longer describes the work -- restate it as one step per entry
        prog.done = 0.0
        prog.total = float(len(entries) + 1)
        frames = []
        gap_w = gap_h = None
        for i, e in enumerate(entries):
            if e.get("gap"):
                if gap_w is None:
                    gap_w, gap_h = _gap_size(entries)
                frames.append(torch.zeros(int(e["gap"]), gap_h, gap_w, 3))
                prog.step(1.0, "building empty space")
                continue
            prog.emit("re-encoding %s" % _short(e))
            imgs = _decode_images(e["path"])
            hi = imgs.shape[0] if e["exit"] is None else e["exit"]
            patched, changed = _patch_clip(entries, i, e["enter"], hi,
                                           locks, fades, images=imgs,
                                           ramps=ramps)
            if changed:
                imgs = patched
            frames.append(imgs[e["enter"]:hi])
            prog.step(1.0, "re-encoding %s" % _short(e))
        prog.emit("assembling")
        H3SaveVideoWithMCtx._encode_mp4(
            out_path, torch.cat(frames, dim=0), audio, crf)
        prog.step(1.0, "assembling")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"key": key, "starts": starts, "total": total}, f)
    return name, sub, starts, total, True, key


def seam_levels(sequence, level_lock_frames=levellock.FRAMES,
                with_gains=True):
    """Per-join measurements for the timeline's seam badges.

    Reports what is happening at every join whether or not the repair
    options are on: knowing a seam is a fixable step rather than a real
    content mismatch is worth having before deciding to re-roll a take.
    """
    entries = na.resolve_sequence(sequence)
    for e in entries:
        if not e.get("gap"):
            e["frames"] = _clip_frames(e)
    loop = na.sequence_loops(sequence)
    found = levellock.report(entries, frames=level_lock_frames,
                             with_gains=with_gains, loop=loop)
    # The gains drive quick playback's brightness filter, so they follow
    # the same plan the build does: a latent-identical join is not
    # corrected there, and must not be corrected here either. The
    # MEASUREMENT stays -- the badge reports every join.
    plan = _seam_plan(entries, {"level_lock": True}, loop)
    for i, one in found.items():
        if (isinstance(one, dict) and "gains" in one and i < len(plan)
                and not plan[i].get("level_lock", True)):
            one["gains"] = None
    return found


def export_cut(sequence, crf, base_folder, preview_filename,
               export_filename_prefix, **fixes):
    """Promote the preview beside itself: (relative_path, was_cached).

    Rebuilds first when missing or stale, so the exported file is always
    the current sequence -- and byte-identical to the preview. Exports
    land in base_folder under export_filename_prefix with the usual
    counter.
    """
    name, sub, _, _, cached, _ = build_preview(
        sequence, crf, probe=False, base_folder=base_folder,
        preview_filename=preview_filename, **fixes)
    src = os.path.join(folder_paths.get_output_directory(), sub, name)
    prefix = str(export_filename_prefix or "full").strip().strip("/")
    folder = str(base_folder or "").strip().strip("/")
    if folder:
        prefix = "%s/%s" % (folder, prefix)
    full_folder, base, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory())
    out_name = "%s_%05d.mp4" % (base, counter)
    out_path = os.path.join(full_folder, out_name)
    shutil.copy2(src, out_path)
    rel = os.path.join(subfolder, out_name).replace(os.sep, "/").lstrip("/")
    _LOG.info("obvpm.h3: exported %s (%s)", rel,
              "promoted cached preview" if cached else "built fresh")
    return rel, cached


def plain_clip_meta(clip):
    """Identity + length for a clip with NO trustworthy sidecar.

    The browser reads a real sidecar itself, with a ranged request for
    the safetensors header -- that is what keeps scanning a folder cheap
    and is deliberately not routed through here. But a plain video has
    no header to read, and hashing it in the browser would mean pulling
    the whole file across. So the one thing the browser cannot work out
    for itself is the one thing this route answers.

    Returns None when the clip HAS a usable sidecar (the browser already
    has the better answer) or cannot be read at all.
    """
    from . import nodes_load
    path = nodes_load.resolve_clip_path(clip)
    side = mctx.sidecar_path(path)
    if os.path.isfile(side):
        try:
            if mctx.read_header(side).get("self_id") ==                     nodes_load._cached_hash(path):
                return None
        except Exception:
            pass
    return nodes_load.synthetic_header(path)


def video_workflow(path):
    """The workflow embedded in a video, for clips with no mctx sidecar.

    Container tags are read here rather than in the browser: parsing
    isobmff boxes client-side would reimplement what PyAV already does,
    and the same call transparently covers VideoHelperSuite's `comment`
    blob, which ComfyUI's own reader does not understand.
    """
    from .nodes_save import read_video_workflow
    root = os.path.abspath(folder_paths.get_output_directory())
    ap = os.path.abspath(os.path.join(root, str(path or "").strip()))
    if os.path.commonpath([root, ap]) != root:
        raise ValueError("video_workflow: %r escapes the output folder"
                         % path)
    if not os.path.isfile(ap):
        raise ValueError("video_workflow: not found: %r" % path)
    return read_video_workflow(ap)


def clip_arrival(path):
    """The Result Preview's `arrival` reading for a take, on demand.

    The reading normally rides in the node's payload, but that payload is
    saved INTO the workflow and restored without running anything -- so a
    take shown before the reading existed would never get one. Same
    function, same numbers; None when the take arrives nowhere or is too
    short to measure.
    """
    from .seam_report import arrival_frame, measure_arrival
    root = os.path.abspath(folder_paths.get_output_directory())
    ap = os.path.abspath(os.path.join(root, str(path or "").strip()))
    if os.path.commonpath([root, ap]) != root:
        raise ValueError("arrival: %r escapes the output folder" % path)
    if not ap.lower().endswith(".mp4") or not os.path.isfile(ap):
        raise ValueError("arrival: not found: %r" % path)
    side = mctx.sidecar_path(ap)
    if not os.path.isfile(side):
        return None
    at = arrival_frame(mctx.read_header(side))
    return measure_arrival(ap, at) if at is not None else None


def delete_take(path):
    """Delete a take and everything filed under its name.

    Exists for the Result Preview's delete button: rejecting a take
    right where its measured seam verdict appeared. Removes the MP4,
    its mctx sidecar and its conditioning cache (.cond.safetensors) --
    per-take files that would otherwise sit orphaned, or worse, get
    adopted by the NEXT take when the save counter reuses the freed
    number. Returns the output-relative names removed.
    """
    root = os.path.abspath(folder_paths.get_output_directory())
    ap = os.path.abspath(os.path.join(root, str(path or "").strip()))
    if os.path.commonpath([root, ap]) != root:
        raise ValueError("delete_take: %r escapes the output folder"
                         % path)
    if not ap.lower().endswith(".mp4"):
        raise ValueError("delete_take: only takes (.mp4) can be deleted")
    if not os.path.isfile(ap):
        raise ValueError("delete_take: not found: %r" % path)
    removed = []
    for f in (ap, mctx.sidecar_path(ap), condstore.cond_path(ap)):
        if os.path.isfile(f):
            os.remove(f)
            removed.append(os.path.relpath(f, root).replace(os.sep, "/"))
    _LOG.info("obvpm.h3: deleted take %s", ", ".join(removed))
    return removed


def _fix_opts(data):
    """The seam-repair options as build_preview keywords.

    Absent keys keep the defaults rather than reading as False, so an
    older UI (or a direct API caller) does not silently turn the repairs
    off by not mentioning them.
    """
    out = {}
    if "level_lock" in data:
        out["level_lock"] = bool(data.get("level_lock"))
    if "level_lock_frames" in data:
        out["level_lock_frames"] = max(2, min(120, int(
            data.get("level_lock_frames") or levellock.FRAMES)))
    if "level_lock_flicker" in data:
        out["level_lock_flicker"] = bool(data.get("level_lock_flicker"))
    if "crossfade" in data:
        out["crossfade"] = bool(data.get("crossfade"))
    if "crossfade_frames" in data:
        out["crossfade_frames"] = max(0, min(240, int(
            data.get("crossfade_frames") or 0)))
    if "audio_declick" in data:
        out["audio_declick"] = bool(data.get("audio_declick"))
    if "level_lock_local" in data:
        out["level_lock_local"] = bool(data.get("level_lock_local"))
    return out


def register():
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/obvpm/h3/preview_cut")
    async def _preview_cut(request):
        import asyncio
        try:
            data = await request.json()
            target = str(data.get("preview_filename", "")
                         or DEFAULT_PREVIEW_NAME)
            name, sub, starts, total, cached, key = await asyncio.to_thread(
                functools.partial(
                    build_preview, str(data.get("sequence", "")),
                    int(data.get("crf", 19)), bool(data.get("probe", False)),
                    str(data.get("base_folder", "")), target,
                    on_progress=(None if data.get("probe")
                                 else _progress_sink(target)),
                    **_fix_opts(data)))
            return web.json_response({
                "filename": name, "subfolder": sub, "type": "output",
                "starts": starts, "total": total, "cached": cached,
                "v": key,
            })
        except na.EmptySequence:
            # an empty strip asks about itself on every repaint: a
            # state, not a failure. A traceback per repaint buries
            # the errors that do mean something.
            _LOG.debug("obvpm.h3: preview_cut on an empty sequence")
            return web.json_response(
                {"error": "the sequence is empty"}, status=400)
        except Exception as exc:
            _LOG.exception("obvpm.h3: preview_cut failed: %s: %s",
                           type(exc).__name__, exc)
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/obvpm/h3/export")
    async def _export(request):
        import asyncio
        try:
            data = await request.json()
            target = str(data.get("preview_filename", "")
                         or DEFAULT_PREVIEW_NAME)
            rel, cached = await asyncio.to_thread(
                functools.partial(
                    export_cut, str(data.get("sequence", "")),
                    int(data.get("crf", 19)),
                    str(data.get("base_folder", "")), target,
                    str(data.get("export_filename_prefix", "") or "full"),
                    on_progress=_progress_sink(target),
                    **_fix_opts(data)))
            return web.json_response({"path": rel, "cached": cached})
        except na.EmptySequence:
            # an empty strip asks about itself on every repaint: a
            # state, not a failure. A traceback per repaint buries
            # the errors that do mean something.
            _LOG.debug("obvpm.h3: export on an empty sequence")
            return web.json_response(
                {"error": "the sequence is empty"}, status=400)
        except Exception as exc:
            _LOG.exception("obvpm.h3: export failed: %s: %s",
                           type(exc).__name__, exc)
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/obvpm/h3/seam_levels")
    async def _seam_levels(request):
        import asyncio
        try:
            data = await request.json()
            found = await asyncio.to_thread(
                seam_levels, str(data.get("sequence", "")),
                max(2, min(120, int(data.get("level_lock_frames")
                                    or levellock.FRAMES))))
            return web.json_response(
                {str(k): v for k, v in found.items()})
        except na.EmptySequence:
            # an empty strip asks about itself on every repaint: a
            # state, not a failure. A traceback per repaint buries
            # the errors that do mean something.
            _LOG.debug("obvpm.h3: seam_levels on an empty sequence")
            return web.json_response(
                {"error": "the sequence is empty"}, status=400)
        except Exception as exc:
            _LOG.exception("obvpm.h3: seam_levels failed: %s: %s",
                           type(exc).__name__, exc)
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/obvpm/h3/video_workflow")
    async def _video_workflow(request):
        import asyncio
        try:
            data = await request.json()
            found = await asyncio.to_thread(
                video_workflow, str(data.get("path", "")))
            return web.json_response(found or {})
        except Exception as exc:
            _LOG.exception("obvpm.h3: video_workflow failed: %s: %s",
                           type(exc).__name__, exc)
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/obvpm/h3/clip_meta")
    async def _clip_meta(request):
        import asyncio
        try:
            data = await request.json()
            clip = str(data.get("clip", ""))
            found = await asyncio.to_thread(plain_clip_meta, clip)
            return web.json_response(found or {})
        except Exception as exc:
            # info, not exception: asking about a clip that has since been
            # moved is ordinary, and the caller degrades to "no meta"
            _LOG.info("obvpm.h3: clip_meta failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/obvpm/h3/arrival")
    async def _arrival(request):
        import asyncio
        try:
            data = await request.json()
            found = await asyncio.to_thread(
                clip_arrival, str(data.get("clip", "")))
            return web.json_response({"arrival": found})
        except Exception as exc:
            _LOG.info("obvpm.h3: arrival failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=400)

    @PromptServer.instance.routes.post("/obvpm/h3/delete_take")
    async def _delete_take(request):
        import asyncio
        try:
            data = await request.json()
            removed = await asyncio.to_thread(
                delete_take, str(data.get("path", "")))
            return web.json_response({"removed": removed})
        except Exception as exc:
            _LOG.exception("obvpm.h3: delete_take failed: %s: %s",
                           type(exc).__name__, exc)
            return web.json_response({"error": str(exc)}, status=400)

    _LOG.info("obvpm.h3: preview/export/delete/workflow routes registered "
              "(/obvpm/h3/preview_cut, /obvpm/h3/export, "
              "/obvpm/h3/delete_take, /obvpm/h3/video_workflow, "
              "/obvpm/h3/seam_levels, /obvpm/h3/clip_meta)")
