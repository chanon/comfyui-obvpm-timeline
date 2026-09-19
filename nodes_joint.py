"""Joint refine: sample the whole timeline in one pass, in sliding windows.

A per-clip refine re-invents fine texture from its own prior and noise,
and no amount of held neighbour makes it copy the neighbour's invention
(measured 2026-09-04: a refine moves 74% of the high-pass texture away
from its upscaled prior, per clip, per seed -- exact holds, matched noise
levels, one noise field, keyframe references, continuous upscaled
latents and a 27-step hold all left the refined seam where it was).

A single long clip refines seamlessly because every row is sampled in
one pass with every other row in view. This module gives a chain the
same treatment: the clips' raw latents are laid onto ONE timeline latent
at their true positions, upscaled as one, and sampled as one -- in
context windows the size of a clip, overlapping by about a quarter, with
the model's predictions blended across the overlap at every step
(MultiDiffusion along time). Each window samples under the conditioning
of the clip that owns most of it, so a timeline of several prompts stays
several prompts. Texture decisions are shared across every join. Per
step the model runs once per window, so a chain costs one pass over its
length plus the overlaps; memory is one window's worth.

The sampled timeline IS the cut -- every row is the clip the cut shows
there -- so it is decoded straight into one finished video by H3 Joint
VAE Decode and Save (nodes_render.py), a few seconds at a time, and
saved as a take of its own that can go back on a timeline and be
refined again.

Nodes, in wiring order: H3 Join Latents (timeline -> one latent), H3
Joint Conditioning (every clip's conditioning, for the window handler),
H3 Joint Audio Mask (hold the finished soundtrack), H3 Context Windowing
(the model patch), and H3 Joint VAE Decode and Save (the sampled timeline -> MP4).
"""

import contextlib
import logging
import os
import re
import time

import torch

from . import avpack
from . import condload
from . import frames as fr
from . import mctx
from . import nodes_assemble as na
from . import nodes_load
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

# The key under which the sampler's CONDITIONING carries every clip's own
# conditioning for the window handler. Core copies every key of a
# conditioning entry's extras into the cond dict and hands only
# `model_conds` to the model, so the table rides the one wire the
# sampler already has and never reaches the model itself.
COND_KEY = "obvpm_h3_joint"


# ---------------------------------------------------------------------------
# the timeline text
# ---------------------------------------------------------------------------

# a sequence line's tail: ` @ enter`, ` @ enter..exit`, ` @ ..exit`
_MARKER = re.compile(r"^(\s*@\s*)(\d*)(\.\.)?(\d*)(?=\s|\[|$)")


def parse_sequence(text):
    """The timeline's `sequence` widget -> clip paths, in delivery order.

    The same shape H3Assemble reads: one output-relative clip per line,
    `# ...` and the `loop` directive ignored, and an optional ` @ N` cut
    marker which is about where a clip ENTERS the cut and is not part of
    its identity.
    """
    clips = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or na.LOOP_RE.match(line):
            continue
        clips.append(line.split("@")[0].strip())
    return [c for c in clips if c]


def parse_sequence_lines(text):
    """The same lines `parse_sequence` reads, markers still attached.

    Aligned with it by construction -- same filter, same order -- so
    position N in one is position N in the other.
    """
    out = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or na.LOOP_RE.match(line):
            continue
        if line.split("@")[0].strip():
            out.append(line)
    return out

# ---------------------------------------------------------------------------
# the timeline as one latent
# ---------------------------------------------------------------------------

def clip_headers(clips):
    """Every clip's sidecar header, in delivery order."""
    out = []
    for clip in clips:
        path = nodes_load.resolve_clip_path(clip)
        out.append(mctx.read_header(mctx.sidecar_path(path)))
    return out


def marker_starts(lines, headers):
    """Each clip's RAW-latent start on the timeline from the cut alone.

    Delivered frame f of clip i sits at T_i + (f - enter_i), where T_i is
    the delivered duration of everything before it; raw frame r is
    delivered frame r - head_i. So the raw latent starts at
    T_i - enter_i - head_i -- negative for a first clip whose pinned head
    precedes the timeline, which is fine: it is a coordinate.

    This is only right when every clip is shown in full up to its
    extension: a parent CUT before extending (clip 5 shown to frame 158,
    extended from there) makes T_i too large by the discarded tail, so
    pinned clips are re-placed by their pin in `raw_starts`.
    """
    starts, t = [], 0
    for (enter, exit_), header in zip(shown_cuts(lines, headers), headers):
        enter = enter or 0
        delivered = int(header.get("delivered_frames", 0) or 0)
        head = int(header.get("pinned_head_frames", 0) or 0)
        starts.append(t - enter - head)
        end = exit_ if exit_ is not None else delivered
        t += max(0, end - enter)
    return starts


def shown_cuts(lines, headers):
    """[[enter, exit]] per clip -- the delivered frames the cut SHOWS,
    None where neither the text nor the lineage says (0 / the clip's end).
    """
    # What the TEXT says, None where it says nothing...
    cuts = []
    for line in lines:
        suffix = line[len(line.split("@")[0].rstrip()):]
        m = _MARKER.match(suffix)
        cuts.append([int(m.group(2)) if m and m.group(2) else None,
                     int(m.group(4)) if m and m.group(3) and m.group(4)
                     else None])
    # ...and where it says nothing, what the LINEAGE derives -- the same
    # seam the timeline, the preview and the export all play. The widget
    # writes a marker only for a cut made by hand, so a parent cut before
    # it was extended, and the frame a bridge hands its target over at,
    # exist nowhere but in the takes' recipes. raw_starts re-places a
    # single-pin take by its pin and that covers the first case; a
    # BRIDGE has two pins and is placed by the cut alone, and so is the
    # target it arrives at when that target's own parent is not on the
    # timeline. Both landed off the latent grid (seen 2026-09-19: a
    # bridge leaving a cut at frame 107 of 115 put at frame 76, not 68).
    for i in range(1, len(lines)):
        if cuts[i - 1][1] is not None and cuts[i][0] is not None:
            continue
        try:
            exit_f, enter_f, _ = na._derive_seam(
                {"clip": lines[i - 1], "header": headers[i - 1]},
                {"clip": lines[i], "header": headers[i]})
        except Exception:
            continue
        if cuts[i - 1][1] is None and exit_f is not None:
            cuts[i - 1][1] = int(exit_f)
        if cuts[i][0] is None and enter_f:
            cuts[i][0] = int(enter_f)
    return cuts


def timeline_pin(header, ids):
    """How a clip hangs on another clip OF THE TIMELINE, from its recipe.

    Returns (parent index, offset, junction, kind) or None: the clip's raw
    frame 0 sits `offset` raw frames into the parent, and the cut changes
    hands at the parent's raw frame `junction` (the sidecar's
    parent_join_frame, ramps included). kind is "extends" (parent shown
    before the junction, this clip after) or "prepends" (the reverse).
    A single before-pin covers this clip's first frames, so offset is
    the pin's source_start; an after-pin covers its LAST frames, so the
    clip starts source_start minus everything before the pin. Pins to
    clips outside the timeline, pixel sources and multi-pin recipes
    yield None: such a clip is placed by the cut alone.
    """
    pins = mctx.parse_pins(header)
    if len(pins) != 1:
        return None
    p = pins[0]
    j = ids.get(p.get("source_id"))
    if j is None or p.get("source_kind") not in mctx.LINEAGE_KINDS:
        return None
    start = int(p.get("source_start", 0) or 0)
    covered = int(p.get("source_frames", 0) or 0)
    raw = int(header.get("raw_frames", 0) or 0)
    join = int(header.get("parent_join_frame", 0) or 0)
    if p.get("place") == "before":
        return (j, start, join, "extends")
    if p.get("place") == "after":
        return (j, start - (raw - covered), join, "prepends")
    return None


def raw_starts(lines, headers):
    """Each clip's RAW-latent start on the timeline, in frames.

    A pinned clip sits where its pin says: its held frames ARE the
    parent's frames at source_start, whatever the cut shows of either.
    Clips without a pin into the timeline sit where the cut puts them
    (`marker_starts`). Measured 2026-09-07: placing clip 7 by the cut put
    it 34 frames after its pin because clip 5 was cut at 158 before
    extending, so the held head sat on the wrong rows (rel diff 1.25 on
    what must be an exact hold) and the refine flickered at the seam.
    """
    starts = marker_starts(lines, headers)
    ids = {h.get("self_id"): i for i, h in enumerate(headers) if h.get("self_id")}
    rel = [timeline_pin(h, ids) for h in headers]
    for _ in range(len(headers)):
        moved = False
        for i, r in enumerate(rel):
            if r is None or r[0] == i:
                continue
            want = starts[r[0]] + r[1]
            if starts[i] != want:
                starts[i] = want
                moved = True
        if not moved:
            break
    return starts


def owners(spans, junctions, total):
    """Who owns each unit of the timeline: the clip the cut shows there.

    `spans` are (start, length) per clip, `junctions` (clip, kind, at):
    an "extends" clip owns its rows from `at` on, a "prepends" clip its
    rows before `at`; the rest is first come. A parent cut before its
    extension keeps its discarded tail out of the joint this way, and a
    child's held head stays the parent's rows (identical for a hard
    hold, the parent's for a ramp -- what the cut shows).
    """
    owner = [-1] * total
    for i, (s, n) in enumerate(spans):
        for t in range(max(0, s), min(total, s + n)):
            if owner[t] < 0:
                owner[t] = i
    for i, kind, at in junctions:
        s, n = spans[i]
        lo, hi = (max(s, at), s + n) if kind == "extends" else (s, min(at, s + n))
        for t in range(max(0, lo), min(total, hi)):
            owner[t] = i
    return owner


def timeline_layout(lines, headers, audio_ticks):
    """Where each clip's raw latent sits on one joint latent.

    `lines` are the sequence lines (markers kept), `headers` the clips'
    sidecar headers, `audio_ticks` each clip's audio length in ticks.
    Returns {"steps": [(start, len)], "ticks": [(start, len)],
    "total_steps", "total_frames", "total_ticks"}.
    """
    starts_frames = raw_starts(lines, headers)
    first = min(starts_frames)
    steps, ticks = [], []
    for header, start, a_len in zip(headers, starts_frames, audio_ticks):
        rel = start - first
        s = fr.steps_for_frames(rel)
        if s is None:
            raise ValueError(
                "H3 Join Latents: a clip starts %d frames into the timeline, "
                "which is not on the latent grid; the joint latent needs "
                "every clip on one grid (masked/both chains are)." % rel)
        n = fr.frames_to_latents(int(header.get("raw_frames", 0) or 0))
        steps.append((s, n))
        ticks.append((fr.audio_total(rel), int(a_len)))
    total_steps = max(s + n for s, n in steps)
    total_frames = fr.pixel_frames(total_steps)
    total_ticks = max(fr.audio_total(total_frames), max(t + n for t, n in ticks))
    # where the cut changes hands between a clip and its parent, so the
    # joint carries what the cut shows and not a cut-away tail
    ids = {h.get("self_id"): i for i, h in enumerate(headers) if h.get("self_id")}
    j_steps, j_ticks = [], []
    for i, header in enumerate(headers):
        r = timeline_pin(header, ids)
        if r is None or r[0] == i:
            continue
        j, _, join, kind = r
        at = starts_frames[j] - first + join
        js = fr.steps_for_frames(at)
        if js is None:
            _LOG.warning("obvpm.h3 joint: clip %d's junction at frame %d is off "
                         "the latent grid; its overlap stays first come", i, at)
            continue
        j_steps.append((i, kind, js))
        j_ticks.append((i, kind, fr.audio_total(at)))
    owner_steps = owners(steps, j_steps, int(total_steps))
    owner_ticks = owners(ticks, j_ticks, int(total_ticks))
    # ...and then, over that, what the cut SHOWS. A junction is only
    # known for a single-pin take whose source is on the timeline; a
    # bridge has two pins, and a clip whose parent is elsewhere has none
    # that count, so both were left "first come" -- which handed a cut
    # parent's discarded tail the rows its bridge plays (seen 2026-09-19:
    # ten steps of a cut-away tail in place of the bridge's opening).
    # Every clip plays delivered frames [enter, exit), and those frames
    # sit at known joint positions, so the rows under them are its own
    # whatever the recipe looks like. On a single-pin chain this writes
    # what the junctions already wrote.
    for i, ((enter, exit_), header) in enumerate(
            zip(shown_cuts(lines, headers), headers)):
        base = starts_frames[i] - first + int(
            header.get("pinned_head_frames", 0) or 0)
        lo_f = base + int(enter or 0)
        hi_f = base + int(exit_ if exit_ is not None
                          else int(header.get("delivered_frames", 0) or 0))
        lo_s, hi_s = fr.steps_for_frames(lo_f) if lo_f > 0 else 0,             fr.steps_for_frames(hi_f)
        if lo_s is None or hi_s is None or hi_f <= lo_f:
            continue
        for t in range(max(0, lo_s), min(int(total_steps), hi_s)):
            owner_steps[t] = i
        for t in range(max(0, fr.audio_total(lo_f)),
                       min(int(total_ticks), fr.audio_total(hi_f))):
            owner_ticks[t] = i
    # what each clip holds at its head, and whether that hold is exact
    # (a masked pin without a ramp), for the placement check in assemble
    heads, hard = [], []
    for header in headers:
        pins = mctx.parse_pins(header)
        heads.append(int(header.get("pinned_head_frames", 0) or 0))
        # only a hold whose SOURCE is on this timeline sits on rows it
        # must match; a head held from a clip that is elsewhere lies over
        # whatever the cut shows there, and differing from it is no error
        hard.append(len(pins) == 1 and pins[0].get("place") == "before"
                    and pins[0].get("source_id") in ids
                    and pins[0].get("mode") == "masked"
                    and not pins[0].get("mask_ramp_frames")
                    and not pins[0].get("mask_hold"))
    return {"steps": steps, "ticks": ticks,
            # each clip's RAW frame 0 on the joint, in frames -- what a
            # delivered frame of a clip maps through (see wrap_keep_frames)
            "starts": [int(s - first) for s in starts_frames],
            "total_steps": int(total_steps), "total_frames": int(total_frames),
            "total_ticks": int(total_ticks),
            "owner_steps": owner_steps,
            "owner_ticks": owner_ticks,
            "heads": heads, "hard_hold": hard}


def fit_keyframes(keyframes, target_shape, who="clip"):
    """Keyframes a window at `target_shape` can actually carry.

    Core lays a keyframe's rows out on the TARGET spatial grid (vt x the
    target's patch rows) but fills them from the keyframe's own latent,
    so a keyframe recorded at another resolution -- a take's 34x60
    anchor under a 2x upscaled 68x120 refine -- allocates four rows for
    every one it supplies and the forward dies on a shape mismatch.
    Such a keyframe is dropped here, with its audio twin (the entry that
    follows it, as apply() writes them), and said so once per clip.
    Takes saved before pin keyframes were tagged as lineage carry one of
    these in their .cond; this is what lets them still be refined.
    """
    if not keyframes:
        return keyframes
    try:
        want = (int(target_shape[3]), int(target_shape[4]))
    except (TypeError, IndexError):
        return keyframes
    out, dropped, skip_audio = [], 0, False
    for kf in keyframes:
        z = kf.get("latent") if isinstance(kf, dict) else None
        if z is not None:
            have = (int(z.shape[3]), int(z.shape[4]))
            if have != want:
                dropped += 1
                skip_audio = True
                continue
            skip_audio = False
        elif skip_audio and isinstance(kf, dict) and kf.get("audio_latent") is not None:
            skip_audio = False
            continue
        out.append(kf)
    if dropped:
        _LOG.warning("obvpm.h3 joint: %s carries %d keyframe(s) at another "
                     "resolution than this refine's %dx%d latent grid; "
                     "dropped (a lineage anchor that leaked into its "
                     ".cond, most likely -- the refine does not need it)",
                     who, dropped, want[0], want[1])
    return out


def wrap_keep_frames(layout, entries, headers):
    """[lo, hi): the joint frames the cut SHOWS when it loops.

    The joint carries every clip's whole raw latent, and that is right
    for refining: the take that closes a loop ends on rows reproducing
    the first clip's head, and the sampler should see them as context.
    But the rendered file must not: played on repeat it would show that
    opening twice. So the render is cropped to the cut's own wrap --
    the first entry's derived `enter` (past the frames the loop take
    delivers itself) to the last entry's `exit` -- mapped into joint
    coordinates: a clip's delivered frame f sits at its raw start plus
    its pinned head plus f. `entries` are resolve_sequence's, so manual
    cuts are already in.
    """
    starts, heads = layout["starts"], layout["heads"]
    first, last = entries[0], entries[-1]
    lo = starts[0] + heads[0] + int(first.get("enter") or 0)
    exit_ = last.get("exit")
    if exit_ is None:
        exit_ = int(headers[-1].get("delivered_frames", 0) or 0)
    hi = starts[-1] + heads[-1] + int(exit_)
    lo, hi = max(0, lo), min(int(layout["total_frames"]), hi)
    if hi <= lo:
        raise ValueError(
            "H3 Join Latents: the loop's wrap leaves nothing to render "
            "(frames %d..%d of %d)." % (lo, hi, layout["total_frames"]))
    return [int(lo), int(hi)]


def wrap_tie(layout, headers):
    """Where a looping cut's opening sits TWICE on the joint, or None.

    The joint is a line, and a loop is not. The take that closes a loop
    holds a window of the first clip at its tail, so the opening is on
    the joint twice: copy A, the first clip's own rows at the start, and
    copy B, the take's held tail at the very end. Between ordinary
    neighbours a held window is laid onto its source's rows and there is
    one copy; the wrap is the one join where a line cannot do that.

    Left alone the two are refined as strangers -- their own noise,
    their own window, B under the take's conditioning and A with no past
    at all -- and they come out the same content with different detail.
    The take's closing frames are refined to flow into B, the file then
    shows A, and the detail jumps exactly at the wrap (seen live
    2026-09-18). The render crop cannot help: it only chooses which copy
    is shown.

    So the window handler keeps the copies IDENTICAL instead -- see
    H3WindowHandler._tie. This returns what it needs, in video steps and
    audio ticks: where each copy starts, how long they are, and the
    HANDOVER inside the window. The handover matters because the copies
    are not shown the same way round: a softly held window's first
    frames are re-drawn by the take and the take delivers them, so
    before the handover B is what the cut shows and A is cropped away;
    from it on, A is shown and B is cropped.

    None when the cut's last entry does not hold a window of its first
    (a loop made by hand, say), or the copies are off the latent grid.
    """
    if len(headers) < 1:
        return None
    take, first_id = headers[-1], headers[0].get("self_id")
    pin = next((p for p in mctx.parse_pins(take)
                if p.get("place") == "after" and p.get("source_id")
                and p.get("source_id") == first_id
                and p.get("source_kind") in mctx.LINEAGE_KINDS), None)
    if pin is None:
        return None
    covered = int(pin.get("source_frames", 0) or 0)
    raw = int(take.get("raw_frames", 0) or 0)
    if covered <= 0 or raw < covered:
        return None
    a_f = int(layout["starts"][0]) + int(pin.get("source_start", 0) or 0)
    b_f = int(layout["starts"][-1]) + raw - covered
    hand_f = int(na._handover(pin, "after"))
    if a_f < 0 or b_f <= a_f:
        return None
    marks = [fr.steps_for_frames(f) if f else 0
             for f in (a_f, a_f + covered, b_f, b_f + covered, a_f + hand_f)]
    if any(m is None for m in marks):
        _LOG.warning("obvpm.h3 joint: the loop's held opening is off the "
                     "latent grid (frames %d and %d); the wrap is refined "
                     "untied", a_f, b_f)
        return None
    a0, a1, b0, b1, ah = marks
    n = min(a1 - a0, b1 - b0, int(layout["total_steps"]) - b0)
    if n <= 0 or a0 + n > b0:
        return None
    ta, tb = fr.audio_total(a_f), fr.audio_total(b_f)
    tn = min(fr.audio_span(a_f, a_f + covered),
             fr.audio_span(b_f, b_f + covered),
             int(layout["total_ticks"]) - tb)
    return {"a": int(a0), "b": int(b0), "n": int(n),
            "hand": int(max(0, min(n, ah - a0))),
            "ta": int(ta), "tb": int(tb), "tn": int(max(0, tn)),
            "thand": int(max(0, min(max(0, tn),
                                    fr.audio_span(a_f, a_f + hand_f))))}


def crop_blocks(blocks, lo, hi):
    """The frame blocks of a decode, cut to joint frames [lo, hi).

    Every block is still drawn from the decoder -- the windows past
    `hi` are its last ones and it must run to the end to report -- but
    only frames inside the range are yielded.
    """
    pos = 0
    for block in blocks:
        n = int(block.shape[0])
        a, b = max(lo, pos), min(hi, pos + n)
        if b > a:
            yield block[a - pos:b - pos]
        pos += n


def assemble(videos, audios, layout):
    """Lay the clips' raw latents onto one joint (video, audio) pair.

    Every row goes to the clip the cut shows there (layout["owner_steps"],
    see `owners`): a held head stays the parent's rows, a parent cut
    before its extension loses its cut-away tail to the child. Where a
    clip covers rows it does not own, its copy is compared and logged:
    an exact hold must match to the bit, so a difference over a masked
    pin's head is a placement error and is warned about.
    """
    v0, a0 = videos[0], audios[0]
    T, A = layout["total_steps"], layout["total_ticks"]
    video = torch.zeros(v0.shape[:2] + (T,) + v0.shape[3:], dtype=v0.dtype)
    audio = torch.zeros(a0.shape[:-1] + (A,), dtype=a0.dtype)
    v_owner = torch.tensor(layout["owner_steps"], dtype=torch.long)
    a_owner = torch.tensor(layout["owner_ticks"], dtype=torch.long)
    v_written = v_owner >= 0
    a_written = a_owner >= 0
    for i, (v, a) in enumerate(zip(videos, audios)):
        s, n = layout["steps"][i]
        if v.shape[2] != n:
            raise ValueError("H3 Join Latents: clip %d's latent has %d steps "
                             "but its header says %d" % (i, v.shape[2], n))
        if tuple(v.shape[3:]) != tuple(v0.shape[3:]):
            raise ValueError("H3 Join Latents: clip %d is %s, clip 0 is %s; "
                             "one timeline needs one size"
                             % (i, tuple(v.shape[3:]), tuple(v0.shape[3:])))
        mine = v_owner[s:s + n] == i
        idx = torch.nonzero(mine).flatten()
        video[:, :, s + idx] = v[:, :, idx].to(video.dtype)
        t, m = layout["ticks"][i]
        m = min(m, a.shape[-1], A - t)
        aidx = torch.nonzero(a_owner[t:t + m] == i).flatten()
        audio[..., t + aidx] = a[..., aidx].to(audio.dtype)
    for i, v in enumerate(videos):
        s, n = layout["steps"][i]
        theirs = torch.nonzero(v_owner[s:s + n] != i).flatten()
        if len(theirs) == 0:
            continue
        have = video[:, :, s + theirs].float()
        new = v[:, :, theirs].float()
        rel = float(((have - new) ** 2).mean().sqrt() / (have ** 2).mean().sqrt().clamp(min=1e-8))
        head = int(fr.frames_to_latents(int(layout.get("heads", [0] * len(videos))[i])))
        held = int((theirs < head).sum()) if head else 0
        if held and layout.get("hard_hold", [False] * len(videos))[i]:
            hv = video[:, :, s + theirs[:held]].float()
            hn = v[:, :, theirs[:held]].float()
            hrel = float(((hv - hn) ** 2).mean().sqrt() / (hv ** 2).mean().sqrt().clamp(min=1e-8))
            if hrel > 1e-3:
                _LOG.warning("obvpm.h3 joint: clip %d's held head (%d step(s)) differs "
                             "from the parent rows it sits on (rel diff %.4f); an exact "
                             "hold must match -- the clip is placed wrong or its parent "
                             "was regenerated", i, held, hrel)
        _LOG.info("obvpm.h3 joint: clip %d covers %d step(s) the sequence shows from "
                  "another clip; kept theirs (rel diff %.4f)", i, len(theirs), rel)
    if not v_written.all():
        holes = torch.nonzero(~v_written).flatten().tolist()
        raise ValueError("H3 Join Latents: the timeline has gaps at latent "
                         "step(s) %s -- clips that do not touch cannot be "
                         "refined as one. Refine such a timeline in runs, "
                         "or bridge the gap." % holes[:12])
    if not a_written.all():
        # the audio grid's last tick can fall past every clip's last tick;
        # carry the previous tick rather than leave silence
        for j in torch.nonzero(~a_written).flatten().tolist():
            if j > 0:
                audio[..., j] = audio[..., j - 1]
    return video, audio


class H3JointLatent:
    """The timeline's clips, laid onto one latent at their true positions."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "build"
    RETURN_TYPES = ("LATENT", wt.JOINT)
    RETURN_NAMES = ("latent", "layout")
    DESCRIPTION = (
        "Lays every clip of the timeline onto ONE raw AV latent at its true "
        "position (held windows coincide), for the joint refine: upscale it "
        "as one, sample it as one under H3 Context Windowing, and render it "
        "as one finished video with H3 Joint VAE Decode and Save. Texture is then "
        "decided across the joins, not per clip."
    )
    OUTPUT_TOOLTIPS = (
        "The joint raw AV latent (video + audio), for the upscaler -- and "
        "for H3 Joint VAE Decode and Save's source_audio when the sound is re-sampled.",
        "Where each clip sits on it. Wire to H3 Joint Conditioning and H3 "
        "Joint VAE Decode and Save.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The timeline, one output-relative clip per "
                               "line -- the Timeline's sequence output."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # the latent is read from the clips ON DISK; a take regenerated
        # under the same name must not be served from an earlier prompt
        return float("nan")

    def build(self, sequence):
        clips = parse_sequence(sequence)
        lines = parse_sequence_lines(sequence)
        if not clips:
            raise ValueError("H3 Join Latents: the sequence is empty.")
        headers = clip_headers(clips)
        videos, audios = [], []
        for clip in clips:
            v, a, _ = mctx.load_sidecar(mctx.sidecar_path(
                nodes_load.resolve_clip_path(clip)))
            videos.append(v)
            audios.append(a)
        layout = timeline_layout(lines, headers, [a.shape[-1] for a in audios])
        video, audio = assemble(videos, audios, layout)
        record = dict(layout, clips=list(clips), lines=list(lines))
        if na.sequence_loops(sequence):
            # the cut's own wrap, hash-verified and with manual cuts in,
            # as the frames the RENDER keeps; the joint itself is whole
            entries = na.resolve_sequence(sequence)
            record["keep_frames"] = wrap_keep_frames(layout, entries, headers)
            record["wrap_tie"] = wrap_tie(layout, headers)
            tie = record["wrap_tie"]
            if tie:
                _LOG.info("obvpm.h3 joint: the opening is on the joint twice "
                          "-- steps %d..%d and %d..%d -- and the refine will "
                          "keep the two identical (the take's copy leads for "
                          "%d step(s), the first clip's after)", tie["a"],
                          tie["a"] + tie["n"], tie["b"], tie["b"] + tie["n"],
                          tie["hand"])
            else:
                _LOG.info("obvpm.h3 joint: the sequence loops but its last clip "
                          "holds no window of its first; the wrap is "
                          "refined untied")
            lo, hi = record["keep_frames"]
            _LOG.info("obvpm.h3 joint: the sequence loops -- the render keeps "
                      "frames %d..%d of %d (%d off the head, %d off the "
                      "tail, which reproduce the opening)", lo, hi,
                      layout["total_frames"], lo, layout["total_frames"] - hi)
        info = "\n".join(
            "%s: steps %d..%d, ticks %d..%d"
            % (clip, s, s + n, t, t + m)
            for clip, (s, n), (t, m) in zip(clips, layout["steps"], layout["ticks"]))
        _LOG.info("obvpm.h3 joint: %d clip(s) -> %d steps (%d frames), %d "
                  "audio ticks\n%s", len(clips), layout["total_steps"],
                  layout["total_frames"], layout["total_ticks"], info)
        return (avpack.pack_av(video, audio, name="joint"), record)


# ---------------------------------------------------------------------------
# every clip's conditioning, on the sampler's one CONDITIONING wire
# ---------------------------------------------------------------------------

class H3JointConditioning:
    """The clips' own conditionings, packed for the window handler."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "build"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    DESCRIPTION = (
        "Loads the conditioning each clip of the timeline was generated "
        "with (its saved .cond) and hands them to the sampler as one "
        "CONDITIONING: H3 Context Windowing "
        "samples every window under the conditioning of the clip that owns "
        "most of it, so a timeline of several prompts and reference sets "
        "stays several. Wire to the guider/sampler."
    )
    OUTPUT_TOOLTIPS = ("For the sampler's conditioning input.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (wt.JOINT, {"tooltip": "From H3 Join Latents."}),
            },
        }

    def build(self, layout):
        table = build_table(layout, condload.load_for_clip)
        # the first clip's entries carry the table; without a window
        # handler the sampler simply runs under the first clip
        out = []
        for entry in table["conds"][0]:
            extras = dict(entry[1])
            extras[COND_KEY] = table
            out.append([entry[0], extras])
        _LOG.info("obvpm.h3 joint: conditioning for %d clip(s) packed",
                  len(table["clips"]))
        return (out,)


def _without_table(cond):
    """A conditioning's entries with any joint table of their own removed."""
    out = []
    for entry in cond:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 \
                and isinstance(entry[1], dict) and COND_KEY in entry[1]:
            extras = dict(entry[1])
            extras.pop(COND_KEY, None)
            out.append([entry[0], extras])
        else:
            out.append(entry)
    return out


def build_table(layout, load):
    """The window handler's table for a joint layout: {clips, spans, conds,
    owners}, one row per conditioning.

    A clip whose saved conditioning carries a table of its own -- a cut
    rendered by H3 Joint VAE Decode and Save, put back on a timeline to be refined
    again -- is SPLICED IN rather than treated as one clip: its inner
    rows are offset to where the clip sits on this joint, and the steps
    this clip owns are handed to whichever inner clip owned them there.
    So a second refine pass still samples each stretch of the rendered
    cut under the prompt and references it was made with, instead of
    running the whole cut under its first clip's.
    """
    clips, spans, conds = [], [], []
    # `outer` is read, `owners` written: the new indices must never be
    # mistaken for outer ones on a later pass
    outer = [int(o) for o in (layout.get("owner_steps") or [])]
    owners = list(outer)
    for i, name in enumerate(layout["clips"]):
        cond = load(nodes_load.resolve_clip_path(name))
        if not cond:
            raise ValueError("H3 Joint Conditioning: %s has an empty "
                             "conditioning" % name)
        s, n = int(layout["steps"][i][0]), int(layout["steps"][i][1])
        inner = None
        try:
            inner = cond[0][1].get(COND_KEY)
        except (IndexError, TypeError, AttributeError):
            inner = None
        if not (inner and inner.get("conds") and inner.get("spans")):
            index = len(clips)
            clips.append(name)
            spans.append([s, n])
            conds.append(_without_table(cond))
            for t, o in enumerate(outer):
                if o == i:
                    owners[t] = index
            continue
        base = len(clips)
        inner_names = list(inner.get("clips") or [])
        for j, (c2, (s2, n2)) in enumerate(zip(inner["conds"], inner["spans"])):
            clips.append("%s: %s" % (name, inner_names[j] if j < len(inner_names) else j))
            spans.append([s + int(s2), int(n2)])
            conds.append(_without_table(c2))
        inner_owners = [int(o) for o in (inner.get("owners") or [])]
        for t, o in enumerate(outer):
            if o != i:
                continue
            rel = t - s
            o2 = inner_owners[rel] if 0 <= rel < len(inner_owners) else -1
            # a row the inner cut left unowned goes to the inner clip
            # whose span holds it, else to the first inner clip
            if o2 < 0:
                o2 = next((j for j, (s2, n2) in enumerate(inner["spans"])
                           if int(s2) <= rel < int(s2) + int(n2)), 0)
            owners[t] = base + o2
        _LOG.info("obvpm.h3 joint: %s carries a joint conditioning of its own "
                  "(%d clip(s)); spliced in at step %d", name, len(inner["conds"]), s)
    return {"clips": clips, "spans": spans, "conds": conds,
            # who the cut shows at each step: the window handler
            # anchors its windows per clip on this
            "owners": owners,
            # a looping cut's opening, which sits on the joint twice
            "wrap_tie": layout.get("wrap_tie")}


def window_owner(spans, s, e):
    """Index of the clip with the most rows in window [s, e); ties -> earlier."""
    best, owner = -1, 0
    for j, (cs, cn) in enumerate(spans):
        overlap = min(e, cs + cn) - max(s, cs)
        if overlap > best:
            best, owner = overlap, j
    return owner


# ---------------------------------------------------------------------------
# hold the soundtrack
# ---------------------------------------------------------------------------

def hold_audio(latent, audio_denoise=0.0):
    """The latent with a noise mask that holds its audio at `audio_denoise`.

    The sampler denoises the nested AV pair together, so a refine would
    re-render sound that is already finished. 0 keeps it exactly; a
    value like 0.5 lets it re-sample alongside the picture (the model
    re-derives lip sync from it), and the render then takes the SOURCE
    audio back (H3 Joint VAE Decode and Save's source_audio). Video is fully open.
    """
    import comfy.nested_tensor
    if latent.get("noise_mask") is not None:
        raise ValueError("H3 Joint Audio Mask: the latent already carries a "
                         "noise mask; wire the upscaled AV latent directly.")
    video, audio = avpack.unpack_av(latent, name="latent")
    fill = min(1.0, max(0.0, float(audio_denoise)))
    video_mask = torch.ones((1, 1) + tuple(video.shape[2:]),
                            device=video.device, dtype=torch.float32)
    audio_mask = torch.full((1, 1) + tuple(audio.shape[2:]), fill,
                            device=audio.device, dtype=torch.float32)
    out = dict(latent)
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    _LOG.info("obvpm.h3 joint: audio held at denoise %.2f over %d tick(s); "
              "picture fully open", fill, int(audio.shape[-1]))
    return out


class H3JointAudioMask:
    CATEGORY = "obvpm/h3"
    FUNCTION = "mask"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Holds the joint latent's soundtrack while the picture is refined: "
        "the sampler denoises video and audio together, so without a mask "
        "a refine re-renders sound that was already finished. 0 keeps the "
        "audio exactly; ~0.5 re-samples it alongside the picture for lip "
        "sync, in which case render the SOURCE audio (wire H3 Joint "
        "Latent's latent to H3 Joint VAE Decode and Save's source_audio) rather than "
        "the resample. Wire the sampler's latent_image from here."
    )
    OUTPUT_TOOLTIPS = ("The same latent with the audio hold. Wire to the "
                       "sampler's latent_image.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The upscaled joint AV latent (video + audio)."}),
                "audio_denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = the audio comes out exactly as it went "
                               "in. Higher lets it re-sample with the "
                               "picture (0.5 for lip sync); save the source "
                               "audio then."}),
            },
        }

    def mask(self, latent, audio_denoise=0.0):
        return (hold_audio(latent, audio_denoise),)


# ---------------------------------------------------------------------------
# context windows on an H3 model
#
# Core's IndexListContextHandler assumes every modality keeps time on the
# same dim as the primary one. H3's audio latent is [B, 32, 2, ticks] --
# time LAST, and dim 2 is the channel pair -- so core's per-modality
# windows would slice the pair, not the ticks, and its proportional
# index mapping reads the pair's size as the audio length. Rather than
# transpose the audio through the whole sampling stack, this is a small
# handler of the same shape as core's, written for H3's two streams:
# video windows in latent steps, audio windows in ticks on the shared
# AV grid (fr.audio_total of the window's frame edges -- exact, never a
# proportion), conds sliced on the dim each one keeps time on, the
# window's conditioning swapped for its owning clip's, and the two
# streams' predictions blended with pyramid weights along their own
# time axes.
# ---------------------------------------------------------------------------

def clip_shaped(length):
    """The nearest window length that looks like a clip latent: 5j+2 steps.

    H3's latent grid runs in cycles of five steps covering (1, 4, 4, 4, 4)
    frames, and a clip's latent is 5k+2 steps starting at cycle phase 0.
    The model learned its statistics on exactly that shape, so a window
    is given the same one.
    """
    j = max(1, int(round((int(length) - fr.LATENT_BASE) / float(fr.LATENTS_PER_GROUP))))
    return j * fr.LATENTS_PER_GROUP + fr.LATENT_BASE


def static_windows(total, length, overlap):
    """Fixed windows of about `length`, as few as keep at least `overlap`
    shared steps, spread evenly, every start on the latent grid's 5-step
    cycle and the last window running to the end.

    Every window must start at cycle phase 0 (a multiple of 5 steps): a
    window that starts off the cycle shows the model a latent whose frame
    cycle is shifted from anything it was trained on, and it paints a
    periodic artefact -- measured 2026-09-06 as a flicker every 17 frames
    (one 5-step cycle) exactly where a window starting at step 356
    contributed, and absent under windows starting at 0 and 70. Clips
    themselves always start on the cycle (the timeline layout puts them
    there), so aligned windows see clip-shaped latents.

    Core's static schedule steps by length-overlap and pulls the final
    window back to the end, which for 162/87/21 makes three windows with
    the last two nearly coincident. Here the starts are multiples of the
    cycle spread as evenly as the cycle allows, every gap at most
    length-overlap so no overlap is ever shorter than asked, and the last
    window is stretched to `total` (up to four extra steps) rather than
    started off the cycle. For a two-clip chain, 92 steps with overlap
    22 is exactly two windows [0:92],[70:162].
    """
    import math
    g = fr.LATENTS_PER_GROUP
    if total <= length:
        return [(0, total)]
    last = ((total - length) // g) * g            # last start, on the cycle
    stride = max(g, ((length - overlap) // g) * g)   # largest gap that keeps the overlap
    units = last // g
    gaps = math.ceil(units / (stride // g))
    if gaps == 0:
        return [(0, total)]
    base, extra = divmod(units, gaps)             # gap sizes in cycles, as even as they get
    starts = [0]
    for i in range(gaps):
        starts.append(starts[-1] + (base + (1 if i < extra else 0)) * g)
    windows = [(st, st + length) for st in starts]
    windows[-1] = (starts[-1], total)
    return windows


def anchored_windows(total, length, overlap, spans, owners):
    """Windows anchored per clip, so conditioning changes hands only over
    a held head. Returns [(start, end, clip index)].

    Measured 2026-09-07 (joint_all3 vs joint_single A/B): windows laid on
    one grid across the whole timeline blend two clips' conditionings
    wherever adjacent windows happen to belong to different clips, and
    two of five such blends painted artefacts (glyphs on dark hair, marks
    on a face); sampling every window under one conditioning removed
    them. So each clip gets its own run of windows over the rows the cut
    shows from it, from its raw start (on the 5-step cycle) to the step
    where the next clip takes over (its junction). Adjacent runs overlap
    exactly over the next clip's held head -- rows both clips agree on --
    and that is the only place two conditionings blend. Inside a run the
    windows are `static_windows` of `length`/`overlap`. A run shorter
    than `overlap` steps at the hand-over (no held head) is stretched to
    keep at least that much blend, clip-shaped, within the next clip.
    """
    order = sorted(range(len(spans)), key=lambda i: (int(spans[i][0]), i))
    first_owned = {}
    for t, j in enumerate(owners):
        if j >= 0 and j not in first_owned:
            first_owned[j] = t
    out = []
    for pos, i in enumerate(order):
        a = int(spans[i][0])
        later = [first_owned[j] for j in order[pos + 1:]
                 if j in first_owned and first_owned[j] > a]
        b = min(later) if later else total
        nxt = order[pos + 1] if pos + 1 < len(order) else None
        if nxt is not None and b - int(spans[nxt][0]) < overlap:
            k = max(overlap, b - int(spans[nxt][0]))
            while (int(spans[nxt][0]) + k - a) % fr.LATENTS_PER_GROUP != fr.LATENT_BASE:
                k += 1
            b = min(int(spans[nxt][0]) + k, int(spans[nxt][0]) + int(spans[nxt][1]), total)
        b = min(b, total)
        if b <= a:
            continue
        for s, e in static_windows(b - a, length, overlap):
            out.append((a + s, a + e, i))
    return out


def layout_windows(total, length, overlap, table):
    """The run's windows with their conditioning owner: anchored per clip
    when the table carries the cut's ownership, else one grid over the
    timeline with each window owned by the clip holding most of it (or
    None without a table)."""
    owners = (table or {}).get("owners")
    if table and owners and len(owners) == total and table.get("spans"):
        return anchored_windows(total, length, overlap, table["spans"], owners)
    grid = static_windows(total, length, overlap)
    if not table:
        return [(s, e, None) for s, e in grid]
    return [(s, e, window_owner(table["spans"], s, e)) for s, e in grid]


def pyramid(n):
    """Triangular weights over n positions (FreeNoise's weighted average)."""
    if n % 2 == 0:
        half = n // 2
        seq = list(range(1, half + 1)) + list(range(half, 0, -1))
    else:
        half = (n + 1) // 2
        seq = list(range(1, half)) + [half] + list(range(half - 1, 0, -1))
    return torch.tensor(seq, dtype=torch.float32)


def audio_span(step_start, step_end):
    """Audio ticks [a, b) covering video steps [step_start, step_end)."""
    return (fr.audio_total(fr.frame_at_latent(step_start)),
            fr.audio_total(fr.frame_at_latent(step_end)))


# what a clip's own conditioning contributes per window; the rest of the
# window's model_conds (latent_shapes, the sliced masks) stay the base's
_PER_CLIP_SKIP = ("latent_shapes", "denoise_mask", "audio_denoise_mask")


class _TqdmHandler(logging.Handler):
    """A console handler's twin that prints through tqdm.write."""

    def __init__(self, orig):
        super().__init__(orig.level)
        self.setFormatter(orig.formatter)
        for f in orig.filters:
            self.addFilter(f)

    def emit(self, record):
        try:
            from tqdm import tqdm
            import sys
            # the live sys.stderr, whatever wraps it right now: that is
            # the stream the bars write to, so tqdm can clear them first
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def _log_under_bars():
    """Context: console log lines go through tqdm.write.

    A tqdm bar (the sampler's step bar, or ours) leaves the cursor at the
    end of its line; a log line written then starts there. tqdm.write
    clears every live bar, prints at column 0 and redraws the bars.

    Done by hand rather than with tqdm.contrib.logging: that one only
    recognises a console handler whose stream IS sys.stderr, and under
    ComfyUI the manager wraps sys.stderr again after core's handler was
    made, so it ADDED a bare second handler (every line printed twice,
    the first glued to the sampler's bar). Plain StreamHandlers on the
    root are the console; FileHandlers and other packs' handlers stay.
    """
    root = logging.getLogger()
    orig = list(root.handlers)
    console = [h for h in orig if type(h) is logging.StreamHandler]
    if not console:
        yield
        return
    root.handlers = [h for h in orig if h not in console] + [_TqdmHandler(h) for h in console]
    try:
        yield
    finally:
        root.handlers = orig


def _hms(seconds):
    seconds = max(0, int(round(seconds)))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def _vram_probe(device, reset=False):
    """Bytes allocated on the card now (None off-CUDA); resets the peak."""
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)
    except Exception:
        # no context on that device yet; the log line just goes without
        return None


def _vram_stats(device, before):
    """What one window cost on the card, in MB; None off-CUDA.

    peak = allocations above what was resident when the window started
    (the window's activations); reserved = the allocator's high-water
    mark, the honest total. Under the dynamic loader the weights it
    manages are not counted in either, so `resident` is latents and
    buffers only. `spill` flags the card's limit: Windows never reports
    out of memory there, it pages the excess to system RAM and the
    window runs ten times slower.
    """
    if before is None:
        return None
    mb = 1024.0 * 1024.0
    try:
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    return {"peak": (peak - before) / mb, "resident": before / mb,
            "reserved": reserved / mb, "total": total / mb,
            "spill": reserved > total * 0.97}


def _vram_report(stats):
    """', peak +X MB ...' for the log."""
    if not stats:
        return ""
    text = (", peak +%.0f MB over %.0f MB resident (reserved %.0f of %.0f MB)"
            % (stats["peak"], stats["resident"], stats["reserved"], stats["total"]))
    if stats["spill"]:
        text += " -- at the card's limit, a spill is likely"
    return text


class H3WindowHandler:
    """Windowed calc_cond_batch for H3's packed video+audio latent."""

    def __init__(self, length, overlap, fuse="pyramid"):
        self.context_length = clip_shaped(length)
        self.context_overlap = min(int(overlap), self.context_length - 1)
        self.fuse = fuse
        self._announced = False
        # per-clip model_conds by (clip, window) -- built once per run,
        # not once per step; keyed on the conditioning table's identity so
        # a cached MODEL output reused by a later run starts clean
        self._table_id = None
        self._cache = {}
        self._durations = []      # window seconds this run, for the time left

    @staticmethod
    def _step_position(model_options, sigma):
        """(step index, total steps) from the sampler's own sigma schedule.

        Core puts the schedule in transformer_options["sample_sigmas"]; the
        current step is the schedule entry nearest this call's sigma. (None,
        None) when the sampler did not say (a bare calc_cond_batch).
        """
        try:
            sigmas = (model_options.get("transformer_options") or {}).get("sample_sigmas")
            if sigmas is None or len(sigmas) < 2:
                return None, None
            vals = [float(v) for v in sigmas.flatten().tolist()]
            steps = len(vals) - 1                   # the last entry is the end sigma
            idx = min(range(steps), key=lambda i: abs(vals[i] - sigma))
            return idx, steps
        except Exception:
            return None, None

    # -- what core calls ----------------------------------------------------

    @staticmethod
    def _shapes(conds):
        for cond_list in conds:
            for cond in cond_list or []:
                shapes = (cond.get("model_conds") or {}).get("latent_shapes")
                if shapes is not None and len(shapes.cond) > 1:
                    return list(shapes.cond)
        return None

    @staticmethod
    def _table(conds):
        for cond_list in conds:
            for cond in cond_list or []:
                table = cond.get(COND_KEY)
                if table:
                    return table
        return None

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        shapes = self._shapes(conds)
        if shapes is None:
            return False
        total = int(shapes[0][2])
        use = total > self.context_length
        table = self._table(conds)
        if id(table) != self._table_id:
            self._table_id = id(table)
            self._cache = {}
            self._announced = False
        if not self._announced:
            self._announced = True
            windows = (layout_windows(total, self.context_length, self.context_overlap, table)
                       if use else [(0, total, None)])
            with _log_under_bars():
                self._announce(total, use, windows, table)
        return use

    def _announce(self, total, use, windows, table):
        """The run's windows and their owners, once per conditioning table."""
        anchored = bool(table and table.get("owners"))
        _LOG.info("obvpm.h3 context windows: %d video steps, window %d, "
                  "overlap %d -> %s", total, self.context_length,
                  self.context_overlap,
                  ("%d windows%s %s" % (len(windows),
                                        " anchored per clip (conditioning changes"
                                        " hands only over a held head):" if anchored else ":",
                                        [(s, e) if j is None else (s, e, j) for s, e, j in windows]))
                  if use else "one window, plain sampling")
        if table is None:
            _LOG.info("obvpm.h3 context windows: no per-clip conditioning "
                      "on the wire; every window samples under the "
                      "conditioning given (wire H3 Joint Conditioning "
                      "for per-clip prompts)")
        else:
            for k, (s, e, j) in enumerate(windows):
                _LOG.debug("obvpm.h3 context windows: window %d/%d steps "
                          "%d..%d conditioned by clip %d (%s)", k + 1,
                          len(windows), s, e, j, table["clips"][j])

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        import comfy.utils
        shapes = self._shapes(conds)
        video, audio = comfy.utils.unpack_latents(x_in, shapes)[:2]
        T, A = int(video.shape[2]), int(audio.shape[-1])
        tie = self._tie(self._table(conds), T, A)
        if tie:
            # what the windows SEE: one noisy state for both copies, the
            # shown copy's. The sampler's own state is left alone -- its
            # two copies carry different noise, but every prediction they
            # are stepped toward is shared (below), and the last step
            # lands on the prediction alone.
            video, audio = video.clone(), audio.clone()
            self._mirror(video, 2, tie["a"], tie["b"], tie["n"], tie["hand"])
            self._mirror(audio, -1, tie["ta"], tie["tb"], tie["tn"],
                         tie["thand"])
        windows = layout_windows(T, self.context_length, self.context_overlap,
                                 self._table(conds))
        # the blended prediction accumulates in system RAM: only one
        # window's slice crosses the bus per window, and the card keeps
        # the full-timeline copies' worth of room for the window's
        # activations (a spilled window runs ten times slower)
        acc_v = [torch.zeros(video.shape, dtype=torch.float32, device="cpu") for _ in conds]
        acc_a = [torch.zeros(audio.shape, dtype=torch.float32, device="cpu") for _ in conds]
        cnt_v = torch.zeros(T, dtype=torch.float32)
        cnt_a = torch.zeros(A, dtype=torch.float32)
        sigma = float(timestep.flatten()[0])
        table = self._table(conds)
        # one console gauge per step for the windows, nested under the
        # sampler's own step gauge; the per-window detail goes to DEBUG
        # and one INFO line sums the step up
        from tqdm.auto import tqdm
        step_idx, n_steps = self._step_position(model_options, sigma)
        if step_idx == 0:
            self._durations = []
        show = comfy.utils.PROGRESS_BAR_ENABLED
        bar = tqdm(total=len(windows), desc="obvpm.h3 windows σ%.3f" % sigma,
                   leave=False, dynamic_ncols=True, disable=not show,
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}, {rate_inv_fmt}]")
        # a second line under the bar: this window, and the time left
        # for the whole sampling (all steps' windows at this run's pace)
        line = tqdm(total=0, leave=False, dynamic_ncols=True, disable=not show,
                    bar_format="{desc}")
        def status(text, k):
            left = ""
            if self._durations and n_steps:
                todo = (n_steps - step_idx - 1) * len(windows) + (len(windows) - k)
                eta = todo * sum(self._durations) / len(self._durations)
                left = " | step %d/%d, ~%s left" % (step_idx + 1, n_steps, _hms(eta))
            elif n_steps:
                left = " | step %d/%d" % (step_idx + 1, n_steps)
            line.set_description_str(text + left, refresh=True)
        t_step, worst, spilled = time.time(), None, False
        # while the bar is up, log lines from anywhere go through tqdm.write
        # so they land at column 0 and the bar is redrawn under them
        try:
            with _log_under_bars():
                for k, (s, e, owner) in enumerate(windows):
                    t0 = time.time()
                    ta, tb = audio_span(s, e)
                    tb = min(tb, A)
                    if e == T:
                        tb = A          # the last window owns the grid's last ticks
                    v_win, a_win = video[:, :, s:e], audio[..., ta:tb]
                    sub_x, sub_shapes = comfy.utils.pack_latents([v_win, a_win])
                    sub_conds = [self._window_conds(model, c, s, e, ta, tb, sub_shapes,
                                                    x_in.device, owner)
                                 for c in conds]
                    status("steps %d..%d%s" % (
                        s, e, "" if owner is None else ", clip %d" % owner), k)
                    _LOG.debug("obvpm.h3 context windows: sigma %.3f window %d/%d "
                               "steps %d..%d ticks %d..%d", sigma, k + 1, len(windows),
                               s, e, ta, tb)
                    before = _vram_probe(x_in.device, reset=True)
                    outs = calc_cond_batch(model, sub_conds, sub_x, timestep, model_options)
                    stats = _vram_stats(x_in.device, before)
                    took = time.time() - t0
                    _LOG.debug("obvpm.h3 context windows: window %d/%d done in %.0fs%s",
                               k + 1, len(windows), took, _vram_report(stats))
                    if stats:
                        worst = stats if worst is None or stats["peak"] > worst["peak"] else worst
                        spilled = spilled or stats["spill"]
                    self._durations.append(took)
                    bar.update(1)
                    status("steps %d..%d%s, %.0fs%s" % (
                        s, e, "" if owner is None else ", clip %d" % owner, took,
                        "" if not stats else ", +%.1f GB" % (stats["peak"] / 1024.0)), k + 1)
                    w_v = self._weights(e - s)
                    w_a = self._weights(tb - ta)
                    for i, out in enumerate(outs):
                        if out is None:
                            continue
                        ov, oa = comfy.utils.unpack_latents(out, sub_shapes)[:2]
                        acc_v[i][:, :, s:e] += ov.float().cpu() * w_v.view(1, 1, -1, 1, 1)
                        acc_a[i][..., ta:tb] += oa.float().cpu() * w_a.view(1, 1, 1, -1)
                    cnt_v[s:e] += w_v
                    cnt_a[ta:tb] += w_a
                    del outs, sub_x, sub_conds
                    # the window's activations are gone; hand the cached blocks
                    # back so the next window does not push the allocator past
                    # the card (Windows spills silently to system RAM, and then
                    # every step crawls)
                    import comfy.model_management
                    comfy.model_management.soft_empty_cache()
        finally:
            line.close()
            bar.close()
        # the sampler's own step bar is still on the line: write under it
        with _log_under_bars():
            _LOG.info("obvpm.h3 context windows: sigma %.3f: %d window(s) in %.0fs%s%s",
                      sigma, len(windows), time.time() - t_step,
                      "" if worst is None else ", peak +%.0f MB over %.0f MB resident "
                      "(reserved %.0f of %.0f MB)" % (worst["peak"], worst["resident"],
                                                      worst["reserved"], worst["total"]),
                      " -- at the card's limit, a spill is likely" if spilled else "")
        cnt_v = cnt_v.clamp(min=1e-6).view(1, 1, -1, 1, 1)
        cnt_a = cnt_a.clamp(min=1e-6).view(1, 1, 1, -1)
        results = []
        for i in range(len(conds)):
            if conds[i] is None:
                # core hands back zeros for an absent cond (cfg 1 still
                # computes uncond + (cond - uncond) with it), never None
                results.append(torch.zeros_like(x_in))
                continue
            v, a = acc_v[i] / cnt_v, acc_a[i] / cnt_a
            if tie:
                # what they PREDICT: the mean of the two, written to both.
                # Copy A gains the past it never had (B was predicted with
                # the whole take in front of it) and B the future.
                self._share(v, 2, tie["a"], tie["b"], tie["n"])
                self._share(a, -1, tie["ta"], tie["tb"], tie["tn"])
            v = v.to(device=x_in.device, dtype=video.dtype)
            a = a.to(device=x_in.device, dtype=audio.dtype)
            results.append(comfy.utils.pack_latents([v, a])[0])
        return results

    # -- the wrap of a looping cut ----------------------------------------------

    def _tie(self, table, T, A):
        """The table's wrap tie, if it fits THIS latent; else None."""
        tie = (table or {}).get("wrap_tie")
        if not tie or int(tie.get("n", 0)) <= 0:
            return None
        if int(tie["b"]) + int(tie["n"]) > T or int(tie["a"]) < 0:
            if not getattr(self, "_tie_warned", False):
                self._tie_warned = True
                _LOG.warning("obvpm.h3 context windows: the loop's wrap tie "
                             "does not fit this latent (%d steps); refining "
                             "untied", T)
            return None
        out = {k: int(v) for k, v in tie.items()}
        out["tn"] = max(0, min(out.get("tn", 0), A - out.get("tb", 0)))
        out["thand"] = min(out.get("thand", 0), out["tn"])
        if not getattr(self, "_tie_said", False):
            self._tie_said = True
            _LOG.info("obvpm.h3 context windows: the sequence loops -- steps "
                      "%d..%d and %d..%d are kept identical through the "
                      "refine", out["a"], out["a"] + out["n"], out["b"],
                      out["b"] + out["n"])
        return out

    @staticmethod
    def _span(x, dim, start, n):
        return x.narrow(dim if dim >= 0 else x.dim() + dim, start, n)

    @classmethod
    def _mirror(cls, x, dim, a, b, n, hand):
        """Both copies take the SHOWN one's rows: B's up to the handover
        (the take re-drew those and delivers them), A's from it on."""
        if n <= 0:
            return
        hand = max(0, min(n, hand))
        if hand:
            cls._span(x, dim, a, hand).copy_(cls._span(x, dim, b, hand))
        if n - hand:
            cls._span(x, dim, b + hand, n - hand).copy_(
                cls._span(x, dim, a + hand, n - hand))

    @classmethod
    def _share(cls, x, dim, a, b, n):
        if n <= 0:
            return
        mean = (cls._span(x, dim, a, n) + cls._span(x, dim, b, n)) * 0.5
        cls._span(x, dim, a, n).copy_(mean)
        cls._span(x, dim, b, n).copy_(mean)

    # -- pieces -----------------------------------------------------------------

    def _weights(self, n):
        if self.fuse == "flat":
            return torch.ones(n)
        return pyramid(n)

    def _window_conds(self, model, cond_list, s, e, ta, tb, sub_shapes, device, owner=None):
        """One window's cond list: masks sliced, conditioning its owner's."""
        import comfy.conds
        if cond_list is None:
            return None
        out = []
        for cond in cond_list:
            new = dict(cond)
            mc = dict(cond.get("model_conds") or {})
            for key, value in list(mc.items()):
                tensor = getattr(value, "cond", None)
                if key == "latent_shapes":
                    mc[key] = comfy.conds.CONDConstant(list(sub_shapes))
                elif key == "denoise_mask" and isinstance(tensor, torch.Tensor):
                    mc[key] = value._copy_with(tensor[:, :, s:e])
                elif key == "audio_denoise_mask" and isinstance(tensor, torch.Tensor):
                    mc[key] = value._copy_with(tensor[..., ta:tb])
            table = cond.get(COND_KEY)
            if table:
                mc.update(self._clip_conds(model, table, s, e, sub_shapes,
                                           device, mc, owner))
                new.pop(COND_KEY, None)
            new["model_conds"] = mc
            out.append(new)
        return out

    def _build_conds(self, model, key, cross_attn, extras, clip_start, s, e,
                     sub_shapes, device, base_mc):
        """model_conds for window [s, e) from one clip's stored conditioning.

        Runs the model's own `extra_conds` at the window's shapes -- the
        same call the sampler made for the base conditioning -- so the
        text embedding, the reference blocks and the packed layout are
        exactly what that clip generated under. Content keyframes recorded
        against the clip's own frame 0 are moved to where the clip sits in
        the window.
        """
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        params = {k: v for k, v in extras.items()
                  if k not in (COND_KEY, "model_conds", "strength")}
        params["cross_attn"] = cross_attn
        params["device"] = device
        params["latent_shapes"] = list(sub_shapes)
        payload = base_mc.get("minimax_payload")
        params["seed"] = (payload.cond or {}).get("seed", 0) if payload is not None else 0
        keyframes = params.get("minimax_keyframes")
        if keyframes:
            shift = fr.frame_at_latent(clip_start) - fr.frame_at_latent(s)
            keyframes = [
                dict(kf, resolved_frame_index=kf.get("resolved_frame_index", 0) + shift)
                if isinstance(kf, dict) else kf for kf in keyframes]
            keyframes = fit_keyframes(keyframes, sub_shapes[0],
                                      "clip %s" % (key[0],))
        if keyframes is not None:
            params["minimax_keyframes"] = keyframes
        built = model.extra_conds(**params)
        keep = {k: v for k, v in built.items() if k not in _PER_CLIP_SKIP}
        self._cache[key] = keep
        return keep

    def _clip_conds(self, model, table, s, e, sub_shapes, device, base_mc, owner=None):
        """The owning clip's model_conds for window [s, e), cached per run."""
        j = owner if owner is not None else window_owner(table["spans"], s, e)
        entry = table["conds"][j][0]
        keep = self._build_conds(model, (j, s, e), entry[0], entry[1],
                                 int(table["spans"][j][0]), s, e, sub_shapes,
                                 device, base_mc)
        _LOG.debug("obvpm.h3 context windows: steps %d..%d -> conditioning "
                   "of clip %d (%s)", s, e, j, table["clips"][j])
        return keep


class H3ContextWindows:
    """Sliding-window sampling for an H3 model, in latent steps."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Samples a long AV latent in windows of window_seconds that overlap "
        "by overlap_seconds, blending the model's predictions across the "
        "overlap every step (MultiDiffusion along time, for H3's video+audio "
        "latent). Each window samples under the conditioning of the clip "
        "that owns most of it when H3 Joint Conditioning is on the sampler. "
        "One window's worth of memory, one model call per window per step; "
        "the work per step is the same at any window size, the memory is "
        "not. For the joint refine: the longest window the card holds."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "window_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 600.0, "step": 0.25,
                    "tooltip": "Window length in seconds of picture, rounded "
                               "to whole latent steps (about 3.4 frames "
                               "each; the log says what it became). Tokens "
                               "per window are what attention and memory pay "
                               "for: at 1920x1088, 5 s needs about 7 GB of "
                               "activations, 13 s (a clip and a bit) more "
                               "than 17 GB. Any length is valid; the work per "
                               "step is the same, only the memory changes."}),
                "overlap_seconds": ("FLOAT", {
                    "default": 1.25, "min": 0.0, "max": 600.0, "step": 0.25,
                    "tooltip": "Seconds shared by neighbouring windows; the "
                               "blend happens here. About a quarter of the "
                               "window."}),
            },
        }

    # `fuse_method` is no longer a widget: pyramid (triangular weights
    # over each window) was the working choice throughout, and flat --
    # a plain average, what core does with fusion off -- stays in the
    # handler only as the control for an A/B.
    def patch(self, model, window_seconds, overlap_seconds, fuse_method="pyramid"):
        length = fr.steps_for_seconds(window_seconds)
        overlap = fr.steps_for_seconds(overlap_seconds) if float(overlap_seconds) > 0 else 0
        if overlap >= length:
            raise ValueError("H3 Context Windowing: the overlap (%.2f s = %d "
                             "steps) must be shorter than the window (%.2f s "
                             "= %d steps)." % (float(overlap_seconds), overlap,
                                               float(window_seconds), length))
        import comfy.patcher_extension
        model = model.clone()
        handler = H3WindowHandler(length, overlap, fuse_method)
        model.model_options["context_handler"] = handler
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "obvpm_h3_context_windows", prepare_sampling_for_window)
        _LOG.info("obvpm.h3 context windows: window %.2f s -> %d steps (%d "
                  "frames, clip-shaped), overlap %.2f s -> %d steps (%d frames, "
                  "starts on the 5-step cycle), %s",
                  float(window_seconds), handler.context_length,
                  fr.pixel_frames(handler.context_length), float(overlap_seconds),
                  handler.context_overlap, fr.pixel_frames(handler.context_overlap),
                  fuse_method)
        return (model,)


def timeline_steps(conds):
    """Video steps of the joint timeline, read off the conditioning table."""
    lists = conds.values() if isinstance(conds, dict) else conds
    for cond_list in lists:
        for cond in cond_list or []:
            table = cond.get(COND_KEY) if isinstance(cond, dict) else None
            if table and table.get("spans"):
                return max(int(s) + int(n) for s, n in table["spans"])
    return None


def prepare_sampling_for_window(executor, model, noise_shape, conds, *args, **kwargs):
    """Budget VRAM for one window, not for the whole timeline.

    Core sizes the model load from the noise shape: activations for the
    full joint latent would not fit next to the weights, so it offloads
    every weight to RAM and the run crawls (0 MB loaded, 20 GB streamed
    per window). The handler only ever runs one window at a time, so the
    estimate is scaled to a window plus its overlap, and the packed AV
    shape is unpacked enough for core's area formula to stop counting
    channels. Core's own handler scales the window too, but skips packed
    AV latents; this one reads the timeline length from the conditioning
    table instead.
    """
    model_options = kwargs.get("model_options") or {}
    handler = model_options.get("context_handler")
    total = timeline_steps(conds) if isinstance(handler, H3WindowHandler) else None
    if total:
        span = int(handler.context_length + handler.context_overlap)
        frac = min(1.0, span / float(total))
        noise_shape = list(noise_shape)
        if len(noise_shape) == 3 and noise_shape[1] == 1:
            # packed AV latent [B, 1, N]. Core's estimate multiplies
            # everything after the batch and channel dims, so the packed
            # form counts the 24 video channels as area -- a 24x
            # overestimate (186 GB for this timeline, 51 GB for one
            # window). Hand it the window's video area with the channels
            # back in their own dim: [B, C, T*H*W]. The audio rows are
            # under one percent of N and ride along as margin.
            import comfy.latent_formats
            ch = int(comfy.latent_formats.MiniMaxH3Video.latent_channels)
            noise_shape = [noise_shape[0], ch, max(1, int(noise_shape[2] * frac / ch))]
        elif frac < 1.0 and len(noise_shape) >= 3:
            noise_shape[2] = max(1, int(noise_shape[2] * frac))
        est = None
        try:
            est = model.model.memory_required(noise_shape) / (1024 * 1024)
        except Exception:
            pass
        need = _window_activation_mb(noise_shape)
        slack = 0.0 if _dynamic_vram() else _CLASSIC_SLACK_MB
        need += slack
        if est and need > est:
            # Core's formula for this model is far too small (under 1 GB
            # for a 5 s window at 1920x1088 that measures 9.2 GB), so
            # it loads every weight and the activations spill into system
            # RAM, where the window runs ten times slower. The formula is
            # linear in the area after batch and channels, so stretching
            # the last dim by need / estimate makes it ask for the
            # measured amount instead -- the same request whichever
            # attention branch it picks. Weights that no longer fit
            # beside it stream from RAM, about a second per window.
            noise_shape[-1] = max(1, int(noise_shape[-1] * (need / est)))
        _LOG.info("obvpm.h3 context windows: VRAM budgeted for one window "
                  "(%d of %d steps%s, %.0f MB kept for activations%s)", span,
                  total, "" if est is None else ", %.0f MB core estimate" % est,
                  need - slack,
                  "" if not slack else " + %.0f MB allocator slack under the "
                  "classic loader" % slack)
    return executor(model, noise_shape, conds, *args, **kwargs)


def _dynamic_vram():
    """True when core's dynamic loader manages the weights."""
    try:
        import comfy.cli_args
        return bool(comfy.cli_args.enables_dynamic_vram())
    except Exception:
        return True


# Under the classic loader (--disable-dynamic-vram) the resident weights
# are fixed for the run and the stream-ordered allocator keeps what it
# freed, so the card holds a few GB the activations never asked for.
# Measured 2026-09-15 on the 32 GB card, 5 s windows at 1920x1088: peak
# reserved 21.3 GB = 7.9 GB resident + 10.7 GB activations + 2.7 GB of
# slack, beside core's own 1.2 GB buffer. Budgeting 4 GB for it keeps
# fewer weights resident and streams the rest from RAM each window
# (about a second per GB here, so every GB resident matters); a spill
# costs the whole run.
_CLASSIC_SLACK_MB = 4 * 1024.0


# Activation memory per video token, measured on the 32 GB card: a 37
# step window at 1920x1088 (75k tokens) peaks at 9.2 GB -- about 122 KB
# a token, and linear in the token count -- plus a fifth for the audio
# rows, keyframe rows and the sampler's own buffers.
_ACTIVATION_MB_PER_TOKEN = 0.122 * 1.2
_PATCH_AREA = 2 * 2                # H3 patchifies 1x2x2 latent cells per token


def _window_activation_mb(noise_shape):
    """VRAM one window's activations need, from its video element count.

    `noise_shape` is already the window's share in [B, C, T*H*W] form
    (or the model's own [B, C, T, H, W]); every C x 2 x 2 latent cells
    make one token.
    """
    import comfy.latent_formats
    ch = int(comfy.latent_formats.MiniMaxH3Video.latent_channels)
    area = 1
    for n in noise_shape[2:]:
        area *= int(n)
    tokens = int(noise_shape[0]) * int(noise_shape[1]) * area / float(ch * _PATCH_AREA)
    return tokens * _ACTIVATION_MB_PER_TOKEN
