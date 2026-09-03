"""Seam-quality report for mctx lineage clips. Standalone, no ComfyUI.

For every sidecar with an extends/prepends relation, measures two things
straight from the stored latents (no VAE, no decode):

  fidelity  cosine similarity between the pinned region of this clip and
            the parent slice it was pinned to (parent must be in the same
            folder). ~0.99+ means the conditioning was delivered and
            obeyed; low values mean the mechanism failed.
  seam      worst L2 adjacent-latent-step difference from the boundary
            through the next ~1s of GENERATED content, as a ratio of the
            same-phase median. ~1x means motion flows through the seam;
            >>1x is a hard cut -- the model planted the anchor but did
            not steer toward it. The scan past the boundary catches runs
            that hold the pinned window and only then break away; the
            per-phase baseline is required because a group's five latents
            cover unequal frame spans, so step size is structurally
            periodic (phase 4 measures ~1.2x phase 1) and a single median
            would flag that rhythm as a cut 0.4s after every seam.
            "@+0.4s" marks where the worst point sat when not at the join.

Usage:
    python -s seam_report.py <folder-with-clips> [more folders...]
"""

import os
import sys

if __package__ in (None, ""):
    # Run as a script: the pack folder's name has hyphens, so it cannot
    # be imported by name. Register it as a package under an alias
    # instead (without running __init__, which needs ComfyUI).
    import types
    _here = os.path.dirname(os.path.abspath(__file__))
    _pkg = types.ModuleType("obvpm_h3")
    _pkg.__path__ = [_here]
    sys.modules.setdefault("obvpm_h3", _pkg)
    from obvpm_h3 import frames as fr, mctx
else:
    from . import frames as fr, mctx


def _stepdiffs(video):
    d = video[:, :, 1:] - video[:, :, :-1]
    return ((d ** 2).mean(dim=(0, 1, 3, 4))) ** 0.5


# How far past the boundary to keep looking, in delivered frames. A run
# that reproduces its pinned window faithfully and only then breaks away
# reads as perfect at the boundary itself, so measuring one step there
# misses exactly the failure worth catching. ~1s of slack covers it.
SEAM_SCAN_FRAMES = 24


# Verdict thresholds for the scanned metric, deliberately above the old
# boundary-only pair (1.2/1.8): that compared ONE step with the clip
# median, this takes the worst of ~7 steps against a phase-matched
# baseline, so the same join reads a little higher. Calibrated against 24
# real extends where the clean ones land 1.07-1.35, ambiguous ones
# 1.55-1.75, and the ones that visibly cut 1.99-2.77.
SEAMLESS_BELOW = 1.45
SOFT_BUMP_BELOW = 1.9


def verdict_for(ratio):
    return ("seamless" if ratio < SEAMLESS_BELOW else
            "soft bump" if ratio < SOFT_BUMP_BELOW else "hard cut")


# ---- pixel-domain cut detection -------------------------------------
# The latent metric above cannot see a jump cut: one latent step spans
# 3.4 frames, so a single-frame jump is averaged away inside it. Measured
# on a take that visibly cuts twice in its first 1.5s, the latent metric
# said 1.31x ("seamless") while the decoded frames showed an 18x spike.
# Cuts are therefore found in the DELIVERED frames, which is what the
# viewer actually watches.
CUT_SIZE = (128, 72)     # diffs are about layout, not detail
# Across every extend in two folders each spike >= SPIKE_CUT lands
# between 0.12s and 1.12s of the join, so 1.0s would miss a real one and
# nothing appeared between 1.5s and 4s. 2s keeps margin over that sample
# of seven; the only cost is a little more decoding.
CUT_SCAN_SECONDS = 2.0
CUT_NEIGHBOURS = 12      # +-frames forming a step's local baseline

# A cut is a LOCAL anomaly. This footage swings between regimes (a fast
# walking pan, then calm sipping), so one median for the clip is
# meaningless -- calm content would flag and busy content would hide.
SPIKE_CUT, SPIKE_BUMP = 4.0, 2.8
# A run that lurches away for half a second instead of jumping in one
# frame: sustained motion far above what the rest of the clip does.
BURST_CUT, BURST_BUMP = 4.0, 2.5


def _frame_diffs(path, count, size=CUT_SIZE, tail=False):
    """Consecutive-frame differences over the first (or last) `count`.

    Scaling happens in libswscale via reformat() rather than PIL: the
    decode is cheap, converting every full-size frame is not. For the
    tail the whole file is still decoded (our takes carry a single
    keyframe, so seeking cannot skip anything) but only the frames in
    range are converted, which is where the time actually goes.
    """
    import av
    import numpy as np
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        total = int(stream.frames or 0)
        first = max(0, total - count) if (tail and total) else 0
        for i, frame in enumerate(container.decode(video=0)):
            if i < first:
                continue
            frames.append(
                frame.reformat(width=size[0], height=size[1], format="rgb24")
                .to_ndarray().astype(np.float32) / 255.0)
            if not tail and len(frames) >= count:
                break
    if len(frames) < 2:
        return None, None
    diffs = np.array([np.abs(frames[i + 1] - frames[i]).mean()
                      for i in range(len(frames) - 1)])
    return diffs, frames


def _frames_window(path, lo, hi, size=CUT_SIZE):
    """Delivered frames [lo, hi) of a clip, at the diff resolution."""
    import av
    import numpy as np
    out = []
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= hi:
                break
            if i < lo:
                continue
            out.append(
                frame.reformat(width=size[0], height=size[1], format="rgb24")
                .to_ndarray().astype(np.float32) / 255.0)
    return out


def measure_cuts(clip_path, where="start", scan_seconds=CUT_SCAN_SECONDS,
                 context=None):
    """Find hard cuts in the seconds either side of a take's join.

    `where` picks which end carries the join: "start" for an extend (the
    take begins at the parent) and "end" for a prepend (the take runs
    INTO its target, so the failure shows up in its closing seconds).
    Returns {"ratio", "verdict", "at", "kind"} where `kind` is "cut" (a
    single-frame jump), "burst" (a sustained lurch) or "clean", and `at`
    is seconds from the joining end, or None if unmeasurable.
    """
    import numpy as np
    tail = where == "end"
    scan = int(scan_seconds * fr.FPS)
    diffs, frames_ = _frame_diffs(
        clip_path, scan + CUT_NEIGHBOURS * 3, tail=tail)
    if diffs is None or len(diffs) < 6:
        return None

    # `j` = the diff that SPANS the join. Without context the take is
    # all we have, so the join is the outermost diff and the take's own
    # later content has to serve as the baseline. With the neighbour's
    # frames in hand the join sits INSIDE one continuous strip, and the
    # question becomes the honest one: does motion change where the two
    # clips actually meet?
    j = 0 if not tail else len(diffs) - 1
    if context:
        cpath, cupto = context
        want = CUT_NEIGHBOURS * 2
        try:
            ctx = (_frames_window(cpath, max(0, int(cupto) - want),
                                  int(cupto)) if not tail
                   else _frames_window(cpath, int(cupto), int(cupto) + want))
        except Exception:
            ctx = []
        if len(ctx) >= 4:
            joined = (ctx + frames_) if not tail else (frames_ + ctx)
            diffs = np.array([np.abs(joined[i + 1] - joined[i]).mean()
                              for i in range(len(joined) - 1)])
            j = (len(ctx) - 1) if not tail else (len(frames_) - 1)

    # scan from the join, INTO the take
    n = len(diffs)
    order = (range(j, max(-1, j - scan), -1) if tail
             else range(j, min(j + scan, n)))
    spike, spike_at = 0.0, 0
    for rank, i in enumerate(order):
        lo = max(0, i - CUT_NEIGHBOURS)
        hi = min(n, i + CUT_NEIGHBOURS + 1)
        near = np.concatenate([diffs[lo:i], diffs[i + 1:hi]])
        med = float(np.median(near)) if near.size else 0.0
        if med and diffs[i] / med > spike:
            spike, spike_at = float(diffs[i] / med), rank

    # A take that BEGINS in an active moment and then calms down is not
    # a lurch, but measured against its own later content it reads as
    # one -- and extending from a trim point selects for exactly that,
    # because people cut where something is happening. When the
    # neighbour's frames are available the baseline is ITS motion
    # approaching the join: the thing the take is meant to continue.
    burst = 0.0
    if context and 0 < j < len(diffs) - 1:
        # symmetric and LOCAL, in both directions: a cut is a local
        # anomaly, and motion that ACCELERATES through the join is not
        # one. Weighing the take's opening against the whole context
        # window instead of the frames immediately before the join just
        # moves the false positive rather than removing it.
        edge = diffs[j:j + 12] if not tail else diffs[max(0, j - 11):j + 1]
        rest = (diffs[max(0, j - 12):j] if not tail
                else diffs[j + 1:j + 13])
    else:
        edge = diffs[-12:] if tail else diffs[:12]
        rest = diffs[:-12] if tail else diffs[12:]
    if edge.size and rest.size:
        rest_med = float(np.median(rest))
        if rest_med:
            burst = float(np.median(edge) / rest_med)

    if spike >= SPIKE_CUT or burst >= BURST_CUT:
        verdict, kind = "hard cut", ("cut" if spike >= SPIKE_CUT else "burst")
    elif spike >= SPIKE_BUMP or burst >= BURST_BUMP:
        verdict, kind = "soft bump", ("cut" if spike >= SPIKE_BUMP else "burst")
    else:
        verdict, kind = "seamless", "clean"
    return {
        "ratio": round(max(spike, burst), 1),
        "verdict": verdict,
        "kind": kind,
        "at": round(spike_at / float(fr.FPS), 2) if kind == "cut" else 0.0,
    }


def scan_steps(frames=SEAM_SCAN_FRAMES):
    """Latent steps spanning `frames` delivered frames (17 frames = 5)."""
    return max(1, int(round(
        frames * fr.LATENTS_PER_GROUP / float(fr.FRAMES_PER_GROUP))))


def measure_seam(video, seam, into, scan_frames=SEAM_SCAN_FRAMES):
    """Worst step discontinuity from the seam into the GENERATED side.

    THE seam metric -- both the CLI report and H3ResultPreview call this,
    so their numbers cannot drift apart. `seam` is the step index whose
    difference spans the boundary; `into` is +1 when generated content
    follows the seam (extend) and -1 when it precedes it (prepend).

    Returns {"ratio", "at_frames", "boundary"} or None when not
    measurable: `ratio` is the worst step difference over the scanned
    span as a multiple of the clip's median step difference, `at_frames`
    where that worst point sits relative to the boundary, and `boundary`
    the boundary-only ratio (the pre-scan metric, kept for comparison).
    """
    import torch
    d = _stepdiffs(video)
    n = int(d.shape[0])
    if not 0 <= seam < n:
        return None
    span = scan_steps(scan_frames)
    if into >= 0:
        lo, hi = seam, min(n - 1, seam + span)
    else:
        lo, hi = max(0, seam - span), seam

    # Baseline PER PHASE, not one median for the clip. The five latents of
    # a group cover unequal frame spans (FRAME_PER_TOKEN), so step
    # differences are structurally periodic: phase 4 measures ~1.2x phase 1
    # in every clip here. Against a single median, scanning past the
    # boundary just finds the next phase-4 step every time -- an artefact
    # that reads as a cut ~0.4s after every seam. Comparing each step with
    # steps of ITS OWN phase removes that and leaves real anomalies.
    # The stretch under examination is excluded from its own baseline.
    idx = torch.arange(n)
    ratios = torch.zeros(n)
    outside = (idx < lo) | (idx > hi)
    for phase in range(fr.LATENTS_PER_GROUP):
        same = idx % fr.LATENTS_PER_GROUP == phase
        if not bool(same.any()):
            continue
        rest = d[same & outside]
        base = (rest.median() if rest.numel() else d[same].median()).item()
        if not base:
            return None
        ratios[same] = d[same] / base

    window = ratios[lo:hi + 1]
    k = int(window.argmax().item())
    at_steps = (lo + k) - seam
    return {
        "ratio": window[k].item(),
        "at_frames": at_steps * fr.FRAMES_PER_GROUP / float(
            fr.LATENTS_PER_GROUP),
        "boundary": ratios[seam].item(),
    }


def _cos(x, y):
    import torch
    return torch.nn.functional.cosine_similarity(
        x.flatten(), y.flatten(), dim=0).item()


def _parent_slice(parent_video, spec):
    start = int(spec.get("source_start", 0))
    frames = int(spec.get("source_frames", 0))
    if start % fr.FRAMES_PER_GROUP != 0:
        return None
    k = start // fr.FRAMES_PER_GROUP * fr.LATENTS_PER_GROUP
    return parent_video[:, :, k:k + fr.frames_to_latents(frames)]


def report(folder):
    import torch  # noqa: F401  (import check before any work)

    names = sorted(n for n in os.listdir(folder)
                   if n.endswith(mctx.SIDECAR_SUFFIX))
    by_id = {}
    for n in names:
        try:
            by_id[mctx.read_header(os.path.join(folder, n))["self_id"]] = n
        except Exception:
            pass

    print(f"{'clip':16} {'relation':9} {'pin':>4} {'fidelity':>9} "
          f"{'seam':>7} {'verdict'}")
    for n in names:
        try:
            meta = mctx.read_header(os.path.join(folder, n))
        except Exception as e:
            print(f"{n:16} unreadable: {e}")
            continue
        rel = meta.get("relation")
        if rel not in ("extends", "prepends"):
            continue
        video, _audio, meta = mctx.load_sidecar(os.path.join(folder, n))
        video = video.float()
        # the PINNED window, not the trim: a soft hold delivers part of
        # its window, so the two differ and only the window marks where
        # generated content ends (see nodes_result._pinned_window)
        def _window(place, fallback):
            n = sum(int(s.get("source_frames", 0) or 0)
                    for s in mctx.parse_pins(meta)
                    if s.get("place") == place)
            return n or int(meta.get(fallback, 0))
        head = _window("before", "pinned_head_frames")
        tail = _window("after", "pinned_tail_frames")
        lt = video.shape[2]
        if rel == "extends":
            seam = fr.frames_to_latents(head) - 1
            pinned = video[:, :, :fr.frames_to_latents(head)]
        else:
            seam = lt - fr.frames_to_latents(tail) - 1
            pinned = video[:, :, lt - fr.frames_to_latents(tail):]

        m = measure_seam(video, seam, 1 if rel == "extends" else -1)
        ratio = m["ratio"] if m else float("nan")

        fidelity = ""
        pins = mctx.parse_pins(meta)
        parent_name = by_id.get(meta.get("parent_id"))
        if parent_name and len(pins) == 1:
            pv, _pa, _pm = mctx.load_sidecar(os.path.join(folder, parent_name))
            sl = _parent_slice(pv.float(), pins[0])
            if sl is not None and sl.shape == pinned.shape:
                fidelity = "%.4f" % _cos(pinned, sl)

        verdict = verdict_for(ratio)
        if verdict == "hard cut":
            verdict = "HARD CUT"
        # where the worst point sat: at the join itself, or later, once
        # the run had already left the pinned window behind
        where = ""
        if m and abs(m["at_frames"]) >= 1:
            where = " @%+.1fs" % (m["at_frames"] / float(fr.FPS))
        # The decoded clip has the final say: the latent number cannot see
        # a jump cut (one step spans 3.4 frames, so a one-frame jump is
        # averaged away inside it). Only run it when the mp4 is there.
        cuts = ""
        mp4 = os.path.join(folder, n[:-len(mctx.SIDECAR_SUFFIX)] + ".mp4")
        if os.path.isfile(mp4):
            try:
                c = measure_cuts(
                    mp4, where="end" if rel == "prepends" else "start")
            except Exception as exc:
                c = None
                cuts = "  cuts: unreadable (%s)" % type(exc).__name__
            if c:
                cuts = "  cuts: %.1fx %s" % (
                    c["ratio"],
                    c["verdict"].upper() if c["verdict"] == "hard cut"
                    else c["verdict"])
                if c["kind"] == "cut" and c["at"]:
                    cuts += " @%.2fs" % c["at"]
                elif c["kind"] == "burst":
                    cuts += " (lurch)"

        parent = (parent_name or "<absent>").split(".")[0]
        print(f"{n.split('.')[0]:16} {rel:9} {head or tail:>4} "
              f"{fidelity or '-':>9} {ratio:>6.2f}x {verdict}{where}"
              f"  (parent {parent}){cuts}")


if __name__ == "__main__":
    folders = sys.argv[1:] or ["."]
    for f in folders:
        print(f"== {f}")
        report(f)
