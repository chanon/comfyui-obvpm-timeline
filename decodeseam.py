"""Swap the frames a decode EDGE got wrong at a masked join.

WHAT GOES WRONG
---------------
H3's video VAE decodes in chunks of 5 latent steps and lets each chunk
look 2 steps ahead (comfy/ldm/minimax/vae.py: tokens_chunk_size,
token_overlap). A frame in the middle of a file is decoded with context
on both sides. The frames at the very EDGES of a file are not:

  the last latent step (4 frames) has nothing after it, and
  the first two latent steps (5 frames) have nothing before them.

Measured on a masked extend (007-test2 clip 3 -> 4, two decodes of the
very same latents): frames 90..119 of the parent are bit-identical in
both decodes, while the parent's own last four decode differently from
the child's -- fine-detail coherence 0.99, 0.96, 0.91, 0.84 -- and the
child's first five differ from the parent's middle the same way (0.88
.. 0.99). Both VAEs (int8, fp16) behave identically.

WHERE IT SHOWS
--------------
A masked join is latent-identical, so the only thing that can differ
across it is the rendering. At an extend the parent's last frames are
its decode edge, and they meet the child's continuation, decoded from
the middle of the child's own file. That accounts for about a third of
the texture change seen at a high-resolution join (0.56 -> 0.70); the
rest is the model re-drawing texture, which no decode can undo.

The newer take holds the shared window in the MIDDLE of its own decode,
so its copy of those frames is the right one, and it already decoded
them -- the trim node just used to throw them away. So the take keeps a
few of them (nodes_save) and assembly swaps them in here:

  extend (masked/both "before" pin, hard hold): the parent's LAST
  END_SWAP frames, from the end of the child's head window -- only when
  the parent's delivered end IS its raw end (no pinned tail trimmed);

  prepend arriving with a hard hold (mask ramp 0): the target's FIRST
  START_SWAP frames, from the start of the take's tail window -- only
  when the target's delivered start IS its raw start. The default soft
  hold makes the take deliver those frames itself and the target enter
  past them, so this side rarely fires.

A bridge is one of each; a loop's wrap is whichever of them it is.
Anything that moves the join off the window's edge (a cut, a soft
handover) already sits mid-decode and is left alone.

VERIFY FRAMES. The take keeps VERIFY more frames than it swaps -- the
ones next to the swapped run, which must decode identically to the older
clip's own (they are mid-decode in both). Assembly compares them first
and skips the swap if they disagree: frames out of step with the older
clip, a re-rendered older clip, or a pixel-grade window would all show
up there. A different VAE between the two takes does not, and need not:
two H3 VAEs decode the same latents within a fraction of a luma level.

The joint render (upscale) never needs this: it decodes one latent line
in windows with a chunk of run-in and the decoder's lookahead
(nodes_render.window_plan), so none of its shown frames is an edge.
"""

import logging
import os
import tempfile

from . import mctx

_LOG = logging.getLogger("obvpm.h3")

END_SWAP = 4        # the last latent step
START_SWAP = 5      # the first two latent steps
VERIFY = 4
HEAD_BLOB = "decode_seam"
TAIL_BLOB = "decode_seam_tail"
HEAD_KEY = "decode_seam_frames"
TAIL_KEY = "decode_seam_tail_frames"
# mean |difference| of the verify frames, 0..255 luma (Y as stored).
# Measured 2026-10-01 at 1376x768, crf 19, foliage: frames identical
# before encoding land at ~1.65 after the two separate H.264 encodes; the
# same frames one frame out of step at ~8-9. The int8 and fp16 VAEs differ
# by only ~0.25 on the same latents -- far below the edge effect itself
# (0.9-3.3) -- so a changed VAE passes, and that is harmless.
MAX_DIFF = 4.0


# ---------------------------------------------------------------- save side

def capture_sides(pins):
    """{"before", "after"}: the window edges this take should keep.

    Only masked-hold windows (masked, or "both", which masks too) held
    HARD at the join: a soft handover delivers the ramped frames itself,
    so the older clip is cut short of its decode edge and needs nothing.
    A window encoded from pixels is a second rendering everywhere, which
    is the crossfade's case (nodes_save.rerendered_sides), not this one.
    """
    from . import nodes_masked
    out = set()
    for p in pins or []:
        spec = p.get("spec") or {}
        if spec.get("mode", "guide") not in ("masked", "both"):
            continue
        if spec.get("source_kind") == "clip_pixels":
            continue
        place = p.get("place") or spec.get("place")
        covered = int(p.get("covered", 0) or 0)
        shape = (int(spec.get("mask_ramp_frames", 0) or 0),
                 float(spec.get("mask_ramp_edge", 0.0) or 0.0),
                 float(spec.get("mask_hold", 0.0) or 0.0))
        if place == "before" and covered >= END_SWAP + VERIFY:
            if nodes_masked.handover_frames(covered, "before", *shape) == covered:
                out.add("before")
        elif place == "after" and covered >= START_SWAP + VERIFY:
            if nodes_masked.handover_frames(covered, "after", *shape) == 0:
                out.add("after")
    return out


def head_frames(untrimmed, head):
    """The take's decode of the END of its head window, or None."""
    n = END_SWAP + VERIFY
    if untrimmed is None or head < n or int(untrimmed.shape[0]) < head:
        return None
    return untrimmed[head - n:head]


def tail_frames(untrimmed, tail):
    """The take's decode of the START of its tail window, or None."""
    n = START_SWAP + VERIFY
    total = 0 if untrimmed is None else int(untrimmed.shape[0])
    if untrimmed is None or tail < n or total < tail:
        return None
    return untrimmed[total - tail:total - tail + n]


# ------------------------------------------------------------ assembly side

def _stored(path, tail):
    try:
        header = mctx.read_header(mctx.sidecar_path(path))
    except (OSError, ValueError):
        return 0
    return int(header.get(TAIL_KEY if tail else HEAD_KEY, 0) or 0)


def plan(entries, loop=False):
    """{entry index: [(kind, n, source path)]} -- the swaps a sequence gets.

    kind "end" replaces the entry's last n played frames, "start" its
    first n. `entries` are nodes_assemble.resolve_sequence's.
    """
    from . import nodes_assemble as na
    from .levellock import _linked
    from .crossfade import _pixel_pin
    out = {}
    pairs = [(i - 1, i) for i in range(1, len(entries))]
    if loop and len(entries) > 1:
        pairs.append((len(entries) - 1, 0))
    for li, ri in pairs:
        left, right = entries[li], entries[ri]
        if left.get("gap") or right.get("gap"):
            continue
        lh, rh = left.get("header") or {}, right.get("header") or {}
        if not na.masked_continuation(lh, rh):
            continue
        rel = _linked(lh, rh)
        try:
            want_exit, want_enter, _ = na._derive_seam(left, right)
        except Exception:
            continue
        l_delivered = int(lh.get("delivered_frames", 0) or 0)
        l_exit = left.get("exit")
        plays_to_end = l_exit is None or (l_delivered and int(l_exit) == l_delivered)
        enters_at_0 = not int(right.get("enter") or 0)
        if rel == "extends":
            # the parent must play to its delivered end, that end must be
            # its raw end, and the child must open at its own frame 0
            if (want_exit is not None or not plays_to_end or not enters_at_0
                    or int(lh.get("pinned_tail_frames", 0) or 0)
                    or _pixel_pin(rh, "before")
                    or _stored(right["path"], False) < END_SWAP + VERIFY):
                continue
            out.setdefault(li, []).append(("end", END_SWAP, right["path"]))
        elif rel == "prepends":
            # the target opens at its raw frame 0 and the take plays to
            # its own end -- a hard hold; a soft one enters past the edge
            if (int(want_enter or 0) or not enters_at_0 or not plays_to_end
                    or int(rh.get("pinned_head_frames", 0) or 0)
                    or _pixel_pin(lh, "after")
                    or _stored(left["path"], True) < START_SWAP + VERIFY):
                continue
            out.setdefault(ri, []).append(("start", START_SWAP, left["path"]))
    return out


_CACHE = {}
_CACHE_CAP = 32


def _blob_file(path, tail, reader):
    """reader(mp4 path) over the stored frames, cached per take."""
    try:
        key = (os.path.abspath(path), os.stat(path).st_mtime_ns, bool(tail),
               reader.__name__)
    except OSError:
        return None
    if key in _CACHE:
        return _CACHE[key]
    raw = mctx.read_bytes_blob(mctx.sidecar_path(path),
                               TAIL_BLOB if tail else HEAD_BLOB)
    if not raw:
        return None
    out = None
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "seam.mp4")
            with open(p, "wb") as f:
                f.write(raw)
            out = reader(p)
    except Exception:
        _LOG.exception("obvpm.h3: could not read the decode seam stored "
                       "with %s", path)
        return None
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def forget():
    _CACHE.clear()


def _runs(stored_n, kind, n, enter, hi):
    """((src_a, src_b), (dst_a, dst_b), (vsrc_a, vsrc_b), (vdst_a, vdst_b)):
    the swapped run and the verify run, in stored and clip frames."""
    if kind == "end":
        return ((stored_n - n, stored_n), (hi - n, hi),
                (stored_n - n - VERIFY, stored_n - n), (hi - n - VERIFY, hi - n))
    return ((0, n), (enter, enter + n),
            (n, n + VERIFY), (enter + n, enter + n + VERIFY))


def apply_planes(planes, swaps, enter, hi, label=""):
    """Swap into (y, u, v) uint8 planes in place. Returns changed ranges."""
    import numpy as np
    from . import yuv
    y, u, v = planes
    ranges = []
    for kind, n, src in swaps:
        got = _blob_file(src, kind == "start", yuv.decode)
        if got is None:
            continue
        sn = int(got[0].shape[0])
        (sa, sb), (da, db), (va, vb), (wa, wb) = _runs(sn, kind, n, enter, hi)
        if (min(sa, va, da, wa) < 0 or max(sb, vb) > sn or max(db, wb) > hi
                or min(da, wa) < enter or got[0].shape[1:] != y.shape[1:]):
            _LOG.info("obvpm.h3: decode seam for %s does not fit (%s); "
                      "left as it is", label, kind)
            continue
        diff = float(np.abs(got[0][va:vb].astype(np.float32)
                            - y[wa:wb].astype(np.float32)).mean())
        if diff > MAX_DIFF:
            _LOG.info("obvpm.h3: decode seam for %s skipped (%s): its verify "
                      "frames differ by %.2f, so they are not the same "
                      "rendering", label, kind, diff)
            continue
        y[da:db], u[da:db], v[da:db] = (got[0][sa:sb], got[1][sa:sb],
                                         got[2][sa:sb])
        ranges.append((da, db))
        _LOG.info("obvpm.h3: decode seam -- %s's %s %d frames from %s "
                  "(verify %.2f)", label, "last" if kind == "end" else "first",
                  n, os.path.basename(src), diff)
    return ranges


def apply_images(images, swaps, enter, hi, decode, label=""):
    """The RGB path's version: images (n, h, w, 3) 0..1, swapped in place."""
    ranges = []
    for kind, n, src in swaps:
        got = _blob_file(src, kind == "start", decode)
        if got is None:
            continue
        sn = int(got.shape[0])
        (sa, sb), (da, db), (va, vb), (wa, wb) = _runs(sn, kind, n, enter, hi)
        if (min(sa, va, da, wa) < 0 or max(sb, vb) > sn or max(db, wb) > hi
                or min(da, wa) < enter or tuple(got.shape[1:]) != tuple(images.shape[1:])):
            _LOG.info("obvpm.h3: decode seam for %s does not fit (%s); "
                      "left as it is", label, kind)
            continue
        diff = float((got[va:vb].float() - images[wa:wb].float()).abs().mean()) * 255.0
        if diff > MAX_DIFF:
            _LOG.info("obvpm.h3: decode seam for %s skipped (%s): its verify "
                      "frames differ by %.2f", label, kind, diff)
            continue
        images[da:db] = got[sa:sb].to(images.dtype)
        ranges.append((da, db))
    return ranges
