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

The sampled timeline is written to disk once and then sliced back into
the clips' own raw spans inside the upscale loop, which decodes, trims
and saves each as a refined take and keeps the profile manifest, so H3
Assemble Upscale plays the cut unchanged.

Nodes, in wiring order: H3 Joint Latent (timeline -> one latent), H3
Joint Conditioning (every clip's conditioning, for the window handler),
H3 Joint Audio Mask (hold the finished soundtrack), H3 Context Windows
(the model patch), H3 Joint Store (the sampled timeline -> disk), and
inside the loop H3 Joint Slice (one clip's span back out, with trim-only
pins mirroring how the take was made).
"""

import contextlib
import logging
import os
import time

import torch

from . import avpack
from . import condload
from . import frames as fr
from . import mctx
from . import nodes_load
from . import upscale
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

JOINT_FILE = "joint.mctx.safetensors"
JOINT_BLOB = "joint"
# The key under which the sampler's CONDITIONING carries every clip's own
# conditioning for the window handler. Core copies every key of a
# conditioning entry's extras into the cond dict and hands only
# `model_conds` to the model, so the table rides the one wire the
# sampler already has and never reaches the model itself.
COND_KEY = "obvpm_h3_joint"


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
    for line, header in zip(lines, headers):
        suffix = line[len(line.split("@")[0].rstrip()):]
        m = upscale._MARKER.match(suffix)
        enter = int(m.group(2)) if m and m.group(2) else 0
        exit_ = int(m.group(4)) if m and m.group(3) and m.group(4) else None
        delivered = int(header.get("delivered_frames", 0) or 0)
        head = int(header.get("pinned_head_frames", 0) or 0)
        starts.append(t - enter - head)
        end = exit_ if exit_ is not None else delivered
        t += max(0, end - enter)
    return starts


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
                "H3 Joint Latent: a clip starts %d frames into the timeline, "
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
    # what each clip holds at its head, and whether that hold is exact
    # (a masked pin without a ramp), for the placement check in assemble
    heads, hard = [], []
    for header in headers:
        pins = mctx.parse_pins(header)
        heads.append(int(header.get("pinned_head_frames", 0) or 0))
        hard.append(len(pins) == 1 and pins[0].get("place") == "before"
                    and pins[0].get("mode") == "masked"
                    and not pins[0].get("mask_ramp_frames")
                    and not pins[0].get("mask_hold"))
    return {"steps": steps, "ticks": ticks,
            "total_steps": int(total_steps), "total_frames": int(total_frames),
            "total_ticks": int(total_ticks),
            "owner_steps": owners(steps, j_steps, int(total_steps)),
            "owner_ticks": owners(ticks, j_ticks, int(total_ticks)),
            "heads": heads, "hard_hold": hard}


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
            raise ValueError("H3 Joint Latent: clip %d's latent has %d steps "
                             "but its header says %d" % (i, v.shape[2], n))
        if tuple(v.shape[3:]) != tuple(v0.shape[3:]):
            raise ValueError("H3 Joint Latent: clip %d is %s, clip 0 is %s; "
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
        _LOG.info("obvpm.h3 joint: clip %d covers %d step(s) the cut shows from "
                  "another clip; kept theirs (rel diff %.4f)", i, len(theirs), rel)
    if not v_written.all():
        holes = torch.nonzero(~v_written).flatten().tolist()
        raise ValueError("H3 Joint Latent: the timeline has gaps at latent "
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
    RETURN_TYPES = ("LATENT", wt.JOINT, "STRING")
    RETURN_NAMES = ("latent", "joint", "info")
    DESCRIPTION = (
        "Lays every clip of the timeline onto ONE raw AV latent at its true "
        "position (held windows coincide), for the joint refine: upscale it "
        "as one, sample it as one under H3 Context Windows, store it with H3 "
        "Joint Store, and let the loop slice each clip's span back out. "
        "Texture is then decided across the joins, not per clip."
    )
    OUTPUT_TOOLTIPS = (
        "The joint raw AV latent (video + audio), for the upscaler.",
        "Where each clip sits on it. Wire to H3 Joint Conditioning and H3 "
        "Joint Store.",
        "One line per clip: its span.",
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
        clips = upscale.parse_sequence(sequence)
        lines = upscale.parse_sequence_lines(sequence)
        if not clips:
            raise ValueError("H3 Joint Latent: the sequence is empty.")
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
        info = "\n".join(
            "%s: steps %d..%d, ticks %d..%d"
            % (clip, s, s + n, t, t + m)
            for clip, (s, n), (t, m) in zip(clips, layout["steps"], layout["ticks"]))
        _LOG.info("obvpm.h3 joint: %d clip(s) -> %d steps (%d frames), %d "
                  "audio ticks\n%s", len(clips), layout["total_steps"],
                  layout["total_frames"], layout["total_ticks"], info)
        return (avpack.pack_av(video, audio, name="joint"), record, info)


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
        "with (its .cond, or a rebuild from its recorded references) and "
        "hands them to the sampler as one CONDITIONING: H3 Context Windows "
        "samples every window under the conditioning of the clip that owns "
        "most of it, so a timeline of several prompts and reference sets "
        "stays several. Wire to the guider/sampler."
    )
    OUTPUT_TOOLTIPS = ("For the sampler's conditioning input.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joint": (wt.JOINT, {"tooltip": "From H3 Joint Latent."}),
            },
            "optional": {
                "clip": ("CLIP", {
                    "lazy": True,
                    "tooltip": "Needed only for a clip with no .cond that "
                               "recorded its references. Must be the text "
                               "encoder the takes were generated with. Lazy: "
                               "not loaded when every clip has a .cond."}),
                "vae": ("VAE", {
                    "lazy": True,
                    "tooltip": "Video VAE, for the same rebuild. Lazy."}),
                "audio_vae": ("VAE", {
                    "lazy": True,
                    "tooltip": "Audio VAE, for a rebuild with audio "
                               "references. Lazy."}),
                "single_clip": ("INT", {
                    "default": 0, "min": 0, "max": 999,
                    "tooltip": "0: every window samples under the conditioning "
                               "of the clip that owns it (the per-window "
                               "table). N: every window samples under clip "
                               "N's conditioning alone (1 = first clip of the "
                               "timeline) -- the A/B for artefacts at a "
                               "window blend where the conditioning changes "
                               "hands."}),
            },
        }

    @staticmethod
    def _needs_models(joint):
        for clip in joint["clips"]:
            try:
                have = condload.describe_sources(nodes_load.resolve_clip_path(clip))
            except Exception:
                return False
            if not have["cond"] and have["refs"]:
                return True
        return False

    def check_lazy_status(self, joint, clip=None, vae=None, audio_vae=None, **_):
        if not self._needs_models(joint):
            return []
        return [name for name, value in (("clip", clip), ("vae", vae),
                                         ("audio_vae", audio_vae))
                if value is None]

    def build(self, joint, clip=None, vae=None, audio_vae=None, single_clip=0):
        clips = list(joint["clips"])
        if single_clip:
            if single_clip > len(clips):
                raise ValueError("H3 Joint Conditioning: single_clip %d, but the "
                                 "timeline has %d clip(s)" % (single_clip, len(clips)))
            name = clips[single_clip - 1]
            cond, _source = condload.load_for_clip(
                nodes_load.resolve_clip_path(name), clip=clip, vae=vae,
                audio_vae=audio_vae)
            if not cond:
                raise ValueError("H3 Joint Conditioning: %s has an empty "
                                 "conditioning" % name)
            _LOG.info("obvpm.h3 joint: every window samples under clip %d's "
                      "conditioning alone (%s); the per-window table is off",
                      single_clip, name)
            return ([[entry[0], dict(entry[1])] for entry in cond],)
        conds = []
        for name in clips:
            cond, _source = condload.load_for_clip(
                nodes_load.resolve_clip_path(name), clip=clip, vae=vae,
                audio_vae=audio_vae)
            if not cond:
                raise ValueError("H3 Joint Conditioning: %s has an empty "
                                 "conditioning" % name)
            conds.append(cond)
        table = {"clips": clips,
                 "spans": [[int(s), int(n)] for s, n in joint["steps"]],
                 "conds": conds,
                 # who the cut shows at each step: the window handler
                 # anchors its windows per clip on this
                 "owners": [int(o) for o in (joint.get("owner_steps") or [])]}
        # the first clip's entries carry the table; without a window
        # handler the sampler simply runs under the first clip
        out = []
        for entry in conds[0]:
            extras = dict(entry[1])
            extras[COND_KEY] = table
            out.append([entry[0], extras])
        _LOG.info("obvpm.h3 joint: conditioning for %d clip(s) packed", len(clips))
        return (out,)


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
    re-derives lip sync from it), and the save then takes the SOURCE
    audio back (H3 Joint Slice's source_audio). Video is fully open.
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
        "sync, in which case save the SOURCE audio (H3 Joint Slice's "
        "source_audio) rather than the resample. Wire the sampler's "
        "latent_image from here."
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


def cut_weights(spans, total):
    """One window per unit: weight 1 where this window's centre is the
    nearest of all windows (ties to the earlier window), else 0.

    The blend alternative to averaging. Averaging two windows' predictions
    over an overlap superposes two placements of every fine structure --
    hair strands twice a few pixels apart make a mesh, two soft light
    patches sharpen into points at the next step -- and every artefact
    located in the joint refine sat inside an overlap (measured 2026-09-08;
    the per-clip refine, one window per clip and no blending, never showed
    them). With a hard hand-over at the overlap's midpoint every row is
    predicted by exactly one window, still with context on both sides of
    the hand-over, and nothing is ever mixed.
    """
    centres = [(s + e) / 2.0 for s, e in spans]
    owner = [-1] * total
    for t in range(total):
        best = None
        for k, c in enumerate(centres):
            s, e = spans[k]
            if not (s <= t < e):
                continue
            d = abs(t - c)
            if best is None or d < best[0]:
                best = (d, k)
        owner[t] = best[1] if best else -1
    out = []
    for k, (s, e) in enumerate(spans):
        w = torch.tensor([1.0 if owner[t] == k else 0.0 for t in range(s, e)],
                         dtype=torch.float32)
        out.append(w)
    return out


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

    def __init__(self, length, overlap, fuse="pyramid", headroom_gb=10.0,
                 prior=None, anchor=0.0):
        self.context_length = clip_shaped(length)
        self.context_overlap = min(int(overlap), self.context_length - 1)
        self.fuse = fuse
        self.headroom_mb = max(0.0, float(headroom_gb)) * 1024.0
        # the upscaled prior's video rows [1, C, T, H, W] on the CPU and the
        # strength of the frame-0 keyframe each window gets from it (0 = none)
        self.prior_video = prior
        self.anchor = max(0.0, min(1.0, float(anchor)))
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
        if self.prior_video is not None and self.anchor > 0:
            if int(self.prior_video.shape[2]) != int(total):
                raise ValueError("H3 Context Windows: the prior has %d steps but the "
                                 "timeline being sampled has %d; wire the upscaled "
                                 "joint latent the sampler refines"
                                 % (int(self.prior_video.shape[2]), int(total)))
            _LOG.info("obvpm.h3 context windows: %d of %d windows anchored to the "
                      "prior's frame at their start (strength %.3f)",
                      sum(1 for s, _e, _j in windows if self.anchors(s)),
                      len(windows), self.anchor)
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
        cut = self.fuse == "cut"
        if cut:
            spans_a = []
            for s, e, _o in windows:
                ta, tb = audio_span(s, e)
                tb = A if e == T else min(tb, A)
                spans_a.append((ta, tb))
            cut_v = cut_weights([(s, e) for s, e, _o in windows], T)
            cut_a = cut_weights(spans_a, A)
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
                    w_v = cut_v[k] if cut else self._weights(e - s)
                    w_a = cut_a[k] if cut else self._weights(tb - ta)
                    if self.anchors(s) and float(cnt_v[s]) > 0:
                        # the anchored row comes back as the keyframe, not a
                        # refinement; an earlier window already owns it
                        w_v = w_v.clone()
                        w_v[0] = 0.0
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
            v = (acc_v[i] / cnt_v).to(device=x_in.device, dtype=video.dtype)
            a = (acc_a[i] / cnt_a).to(device=x_in.device, dtype=audio.dtype)
            results.append(comfy.utils.pack_latents([v, a])[0])
        return results

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
            elif self.anchors(s) and cond.get("cross_attn") is not None:
                # no per-clip table (one conditioning for the run): the
                # anchor still needs the model's extra_conds rebuilt at
                # this window from the conditioning's own extras
                extras = {k: v for k, v in cond.items()
                          if k not in ("cross_attn", "model_conds")}
                mc.update(self._build_conds(model, ("base", s, e), cond["cross_attn"],
                                            extras, 0, s, e, sub_shapes, device, mc))
            new["model_conds"] = mc
            out.append(new)
        return out

    def anchors(self, s):
        """True when window [s, ...) gets a frame-0 keyframe from the prior."""
        return self.prior_video is not None and self.anchor > 0 and s > 0

    def _anchor_keyframe(self, s):
        """The prior's row at the window start as a frame-0 keyframe.

        Measured 2026-09-08: the same model, prior, noise field and chunk
        length hallucinated (a wireframe rectangle on a forehead, hard
        points on dappled hair, shimmering eyes) under our per-step
        windows and not under MMH3 Ultimate Upscale's sequential chunks;
        the one thing every clean chunk had that our windows lacked is a
        clean keyframe at its frame 0 (H3 packs it as a cond row at the
        target's time origin, `minimax_visual_cond_noise_aug` its
        strength). A window that starts mid-clip gets the upscaled
        prior's own row there: the content the window is refining anyway,
        presented the way the model expects a sequence to begin.
        """
        return {"resolved_frame_index": 0,
                "latent": self.prior_video[:, :, s:s + 1].contiguous()}

    def _build_conds(self, model, key, cross_attn, extras, clip_start, s, e,
                     sub_shapes, device, base_mc):
        """model_conds for window [s, e) from one clip's stored conditioning.

        Runs the model's own `extra_conds` at the window's shapes -- the
        same call the sampler made for the base conditioning -- so the
        text embedding, the reference blocks and the packed layout are
        exactly what that clip generated under. Content keyframes recorded
        against the clip's own frame 0 are moved to where the clip sits in
        the window; the frame-0 anchor from the prior goes in front.
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
        if self.anchors(s):
            kept = [kf for kf in (keyframes or [])
                    if not (isinstance(kf, dict) and kf.get("resolved_frame_index") == 0
                            and kf.get("latent") is not None)]
            keyframes = [self._anchor_keyframe(s)] + kept
            params["minimax_visual_cond_noise_aug"] = self.anchor
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
                "fuse_method": (["pyramid", "flat", "cut"], {
                    "default": "pyramid",
                    "tooltip": "What happens where windows overlap: pyramid "
                               "averages the two predictions with triangular "
                               "weights, flat averages them plainly, cut hands "
                               "every row to the window whose centre is "
                               "nearest and mixes nothing. Averaging "
                               "superposes two placements of fine detail "
                               "(hair strands become a mesh, soft light "
                               "patches hard points); cut avoids that and "
                               "leaves at most a subtle texture change at "
                               "the hand-over. Both windows still see across "
                               "it."}),
                "vram_headroom_gb": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 128.0, "step": 0.5,
                    "tooltip": "VRAM kept free of model weights for one "
                               "window's activations. Core's own estimate for "
                               "this model is a few GB and far too small: at "
                               "1920x1088 a 5 s window needs about 7 GB, a "
                               "13 s window more than 17 GB (one attention "
                               "layer's QKV alone is 7.8 GB), and a window "
                               "that does not fit either "
                               "fails with out-of-memory or spills into system "
                               "RAM and runs ten times slower. Weights that do "
                               "not fit beside the headroom stream from RAM, "
                               "which costs about a second per window. Each "
                               "step's summary log line reports the measured "
                               "peak; set this a little above it. 0 leaves "
                               "the budget to core (fine with dynamic VRAM)."}),
                "anchor_strength": ("FLOAT", {
                    "default": 0.999, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "With `prior` wired: every window that starts "
                               "mid-timeline gets the prior's frame at its "
                               "start as a frame-0 keyframe at this strength "
                               "(1 = exact, 0.999 = the model's default, 0 = "
                               "off). Sequential chunk refines (MMH3 Ultimate "
                               "Upscale) anchor every chunk this way and do "
                               "not hallucinate where our unanchored windows "
                               "did; the anchored row itself is taken from "
                               "the neighbouring window."}),
            },
            "optional": {
                "prior": ("LATENT", {
                    "tooltip": "The upscaled joint AV latent the sampler "
                               "refines (H3 Joint Audio Mask's output). With "
                               "it wired, each window starting mid-timeline "
                               "gets the prior's frame at its start as a "
                               "frame-0 keyframe (anchor_strength)."}),
            },
        }

    def patch(self, model, window_seconds, overlap_seconds, fuse_method,
              vram_headroom_gb=10.0, anchor_strength=0.999, prior=None):
        length = fr.steps_for_seconds(window_seconds)
        overlap = fr.steps_for_seconds(overlap_seconds) if float(overlap_seconds) > 0 else 0
        if overlap >= length:
            raise ValueError("H3 Context Windows: the overlap (%.2f s = %d "
                             "steps) must be shorter than the window (%.2f s "
                             "= %d steps)." % (float(overlap_seconds), overlap,
                                               float(window_seconds), length))
        import comfy.patcher_extension
        model = model.clone()
        prior_video = None
        if prior is not None:
            prior_video = avpack.unpack_av(prior, name="prior")[0].detach().to("cpu")
        handler = H3WindowHandler(length, overlap, fuse_method, vram_headroom_gb,
                                  prior=prior_video, anchor=anchor_strength)
        model.model_options["context_handler"] = handler
        if prior_video is not None and handler.anchor > 0:
            _LOG.info("obvpm.h3 context windows: windows starting mid-timeline get "
                      "the prior's frame as a frame-0 keyframe at strength %.3f",
                      handler.anchor)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "obvpm_h3_context_windows", prepare_sampling_for_window)
        _LOG.info("obvpm.h3 context windows: window %.2f s -> %d steps (%d "
                  "frames, clip-shaped), overlap %.2f s -> %d steps (%d frames, "
                  "starts on the 5-step cycle), %s, %.1f GB headroom",
                  float(window_seconds), handler.context_length,
                  fr.pixel_frames(handler.context_length), float(overlap_seconds),
                  handler.context_overlap, fr.pixel_frames(handler.context_overlap),
                  fuse_method, float(vram_headroom_gb))
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


def prepare_sampling_unpacked(executor, model, noise_shape, conds, *args, **kwargs):
    """Core's VRAM estimate for a packed AV latent, with the channels back
    in their own dim (see prepare_sampling_for_window; here the latent
    already is one window, so no timeline scaling)."""
    noise_shape = list(noise_shape)
    if len(noise_shape) == 3 and noise_shape[1] == 1:
        import comfy.latent_formats
        ch = int(comfy.latent_formats.MiniMaxH3Video.latent_channels)
        noise_shape = [noise_shape[0], ch, max(1, int(noise_shape[2] // ch))]
    return executor(model, noise_shape, conds, *args, **kwargs)


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
        if est and handler.headroom_mb > 0:
            # core's formula is linear in the area after batch and
            # channels, so stretching the last dim by (estimate +
            # headroom) / estimate makes it ask for the headroom too --
            # the same request whichever attention branch it picks
            grow = (est + handler.headroom_mb) / est
            noise_shape[-1] = max(1, int(noise_shape[-1] * grow))
        _LOG.info("obvpm.h3 context windows: VRAM budgeted for one window "
                  "(%d of %d steps%s)", span, total,
                  "" if est is None else ", %.0f MB core estimate + %.0f MB headroom"
                  % (est, handler.headroom_mb))
    return executor(model, noise_shape, conds, *args, **kwargs)


# ---------------------------------------------------------------------------
# sequential windows: each a complete sampling, joined by frozen context
# ---------------------------------------------------------------------------

def sequence_plan(windows, total):
    """Per window (start, end, owner, frozen): frozen[i] is True where row
    start+i was finished by an earlier window and is held as context."""
    done = [False] * total
    plan = []
    for s, e, j in windows:
        frozen = [done[t] for t in range(s, e)]
        plan.append((s, e, j, frozen))
        for t in range(s, e):
            done[t] = True
    return plan


def window_conditioning(conditioning, table, owner, s):
    """The CONDITIONING one window samples under: the owner clip's stored
    entries (or the given conditioning without the table), content
    keyframes moved from the clip's frame 0 to the window's origin."""
    if table and owner is not None:
        entries, clip_start = table["conds"][owner], int(table["spans"][owner][0])
    else:
        entries, clip_start = conditioning, 0
    shift = fr.frame_at_latent(clip_start) - fr.frame_at_latent(s)
    out = []
    for cross_attn, extras in entries:
        ex = {k: v for k, v in extras.items() if k != COND_KEY}
        kfs = ex.get("minimax_keyframes")
        if kfs:
            ex["minimax_keyframes"] = [
                dict(kf, resolved_frame_index=kf.get("resolved_frame_index", 0) + shift)
                if isinstance(kf, dict) else kf for kf in kfs]
        out.append([cross_attn, ex])
    return out


class H3JointSequentialRefine:
    """The joint refine as a sequence of complete samplings, joined by
    frozen context rather than by blending predictions.

    Measured 2026-09-08 on one timeline, one prior, one noise field and one
    window length: our per-step windows (the model's prediction blended or
    cut across windows at every step) hallucinated -- a wireframe rectangle
    on a forehead, hard points on dappled hair, shimmering eyes -- at any
    noise level, with or without the turbo LoRA, with a frame-0 keyframe
    anchor or without; the per-clip refine and MMH3 Ultimate Upscale's
    sequential chunks, each one complete trajectory, did not. Here every
    window is sampled to the end before the next begins, and the rows it
    shares with the finished window before it are held frozen through the
    noise mask: the model sees the finished texture at every step and
    continues it (MMH3 joins its spatial tiles the same way), which is a
    stronger join than a one-frame anchor plus a cross-fade. The noise is
    one field over the timeline, sliced per window.
    """

    CATEGORY = "obvpm/h3"
    FUNCTION = "refine"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Refines the upscaled joint latent window by window, each window a "
        "complete sampling run, the rows shared with the finished window "
        "before it frozen as context (noise mask). With H3 Joint "
        "Conditioning on `conditioning`, windows are anchored per clip and "
        "each samples under its clip's conditioning. Replaces H3 Context "
        "Windows + the sampler: wire model (with the sigma shift), sampler "
        "and sigmas as for SamplerCustomAdvanced, the latent from H3 Joint "
        "Audio Mask, and the output to H3 Joint Store."
    )
    OUTPUT_TOOLTIPS = ("The refined joint AV latent. Wire to H3 Joint Store.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 model with its sigma shift applied."}),
                "conditioning": ("CONDITIONING", {"tooltip": "From H3 Joint Conditioning (per-clip table) or any single conditioning."}),
                "latent": ("LATENT", {"tooltip": "The upscaled joint AV latent from H3 Joint Audio Mask (its audio hold is kept)."}),
                "noise": ("NOISE", {"tooltip": "One noise field for the whole timeline, sliced per window."}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS", {"tooltip": "The refine schedule (BasicScheduler with denoise = first sigma)."}),
                "window_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "max": 600.0, "step": 0.25,
                    "tooltip": "Window length in seconds of picture, rounded to a clip-shaped number of latent steps."}),
                "overlap_seconds": ("FLOAT", {
                    "default": 1.25, "min": 0.0, "max": 600.0, "step": 0.25,
                    "tooltip": "Seconds a window shares with the finished one before it; those rows are frozen context, not resampled."}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "With it, a CFG guider at `cfg`; without, positive only."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    def refine(self, model, conditioning, latent, noise, sampler, sigmas,
               window_seconds, overlap_seconds, negative=None, cfg=1.0):
        import comfy.samplers
        import comfy.utils
        import comfy.model_management
        import comfy.nested_tensor
        import latent_preview
        from tqdm import tqdm
        video, audio = avpack.unpack_av(latent, name="latent")
        T, A = int(video.shape[2]), int(audio.shape[-1])
        length = clip_shaped(fr.steps_for_seconds(window_seconds))
        overlap = fr.steps_for_seconds(overlap_seconds) if float(overlap_seconds) > 0 else 0
        overlap = min(int(overlap), length - 1)
        table = None
        for entry in conditioning or []:
            if isinstance(entry[1], dict) and entry[1].get(COND_KEY):
                table = entry[1][COND_KEY]
                break
        windows = layout_windows(T, length, overlap, table)
        plan = sequence_plan(windows, T)
        # the given mask (Joint Audio Mask): video all open, audio held
        mv_all = ma_all = None
        if latent.get("noise_mask") is not None:
            mv_all, ma_all = latent["noise_mask"].unbind()
        full_noise = noise.generate_noise({"samples": latent["samples"]})
        nv, na = full_noise.unbind()
        acc = video.clone()
        # core sizes the model load from the packed [B, 1, N] window shape,
        # counting the 24 video channels as area (a 24x overestimate that
        # made it offload every weight and crawl); hand it the window's
        # area with the channels in their own dim
        import comfy.patcher_extension
        model = model.clone()
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                   "obvpm_h3_sequential_refine", prepare_sampling_unpacked)
        with _log_under_bars():
            _LOG.info("obvpm.h3 sequential refine: %d steps in %d window(s) of %d "
                      "(overlap %d frozen as context)%s: %s", T, len(windows), length,
                      overlap, " anchored per clip" if table and table.get("owners") else "",
                      [(s, e) if j is None else (s, e, j) for s, e, j in windows])
        show = comfy.utils.PROGRESS_BAR_ENABLED
        bar = tqdm(total=len(plan), desc="obvpm.h3 sequential refine", leave=False,
                   dynamic_ncols=True, disable=not show,
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}, {rate_inv_fmt}]")
        t_all = time.time()
        try:
            with _log_under_bars():
                for k, (s, e, j, frozen) in enumerate(plan):
                    t0 = time.time()
                    ta, tb = audio_span(s, e)
                    tb = A if e == T else min(tb, A)
                    frozen_t = torch.tensor(frozen, dtype=torch.bool)
                    mv = torch.ones((1, 1, e - s) + tuple(video.shape[3:]), dtype=torch.float32)
                    mv[:, :, frozen_t] = 0.0
                    if mv_all is not None:
                        mv = mv * mv_all[:, :, s:e].to(mv.dtype)
                    if ma_all is not None:
                        ma = ma_all[..., ta:tb].clone().float()
                    else:
                        ma = torch.zeros((1, 1) + tuple(audio.shape[2:-1]) + (tb - ta,), dtype=torch.float32)
                    latent_image = comfy.nested_tensor.NestedTensor(
                        (acc[:, :, s:e].clone(), audio[..., ta:tb].clone()))
                    win_noise = comfy.nested_tensor.NestedTensor(
                        (nv[:, :, s:e].clone(), na[..., ta:tb].clone()))
                    dm = comfy.nested_tensor.NestedTensor((mv, ma))
                    cond = window_conditioning(conditioning, table, j, s)
                    guider = comfy.samplers.CFGGuider(model)
                    if negative is not None:
                        guider.set_conds(cond, negative)
                        guider.set_cfg(cfg)
                    else:
                        guider.inner_set_conds({"positive": cond})
                    callback = latent_preview.prepare_callback(model, int(sigmas.shape[-1]) - 1)
                    _LOG.debug("obvpm.h3 sequential refine: window %d/%d steps %d..%d, "
                               "%d frozen row(s)%s", k + 1, len(plan), s, e,
                               int(frozen_t.sum()), "" if j is None else ", clip %d" % j)
                    out = guider.sample(win_noise, latent_image, sampler, sigmas,
                                        denoise_mask=dm, callback=callback,
                                        disable_pbar=not show, seed=noise.seed)
                    out = out.to(comfy.model_management.intermediate_device())
                    ov = out.unbind()[0] if getattr(out, "is_nested", False) else out
                    free = ~frozen_t
                    idx = torch.nonzero(free).flatten()
                    acc[:, :, s + idx] = ov[:, :, idx].to(device=acc.device, dtype=acc.dtype)
                    bar.set_description_str("obvpm.h3 sequential refine (steps %d..%d%s, %.0fs)"
                                            % (s, e, "" if j is None else ", clip %d" % j,
                                               time.time() - t0), refresh=False)
                    bar.update(1)
                    del out, latent_image, win_noise, guider
                    comfy.model_management.soft_empty_cache()
        finally:
            bar.close()
        with _log_under_bars():
            _LOG.info("obvpm.h3 sequential refine: %d window(s) in %s",
                      len(plan), _hms(time.time() - t_all))
        return (avpack.pack_av(acc, audio, name="joint"),)


# ---------------------------------------------------------------------------
# store and slice
# ---------------------------------------------------------------------------

class H3JointStore:
    """Writes the sampled joint latent next to the profile it belongs to."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "store"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("joint_path", "profile_folder")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Saves the jointly refined timeline latent as <base_folder>/_upscale/"
        "<profile>/joint.mctx.safetensors, with each clip's span recorded in "
        "it. Wire joint_path to H3 Upscale Loop Start: the loop slices the "
        "clips out of it into the same folder. With reuse_existing on, a "
        "joint file already there for this timeline is kept and the sampler "
        "is NOT run again -- the Timeline re-runs everything downstream on "
        "every press, and 20 minutes of sampling is not something to repeat "
        "because a save step failed. Off = sample again (a new joint file, "
        "and the loop delivers every clip again)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "lazy": True,
                    "tooltip": "The sampler's output. Only asked for when "
                               "there is no reusable joint file."}),
                "joint": (wt.JOINT, {"tooltip": "From H3 Joint Latent."}),
                "base_folder": ("STRING", {"default": "project1"}),
                "profile": ("STRING", {
                    "default": "joint01",
                    "tooltip": "Profile folder name under "
                               "<base_folder>/_upscale/."}),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep a joint file already stored for this "
                               "timeline instead of sampling again. Off = "
                               "always resample and overwrite."}),
            },
        }

    @staticmethod
    def _target(base_folder, profile):
        import folder_paths
        name = str(profile or "").strip()
        if not name:
            raise ValueError("H3 Joint Store: profile must be a name.")
        rel = upscale.profile_folder(base_folder, name)
        folder = os.path.join(folder_paths.get_output_directory(), *rel.split("/"))
        return rel, folder, os.path.join(folder, JOINT_FILE)

    @staticmethod
    def _reusable(path, joint):
        """True when the stored joint was built for exactly this timeline."""
        if not os.path.isfile(path):
            return False
        try:
            record = mctx.read_blob(path, JOINT_BLOB) or {}
        except Exception:
            return False
        return (record.get("clips") == list(joint["clips"])
                and record.get("total_steps") == joint["total_steps"]
                and [list(x) for x in record.get("steps", [])]
                == [list(x) for x in joint["steps"]])

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # whether the file stands is a fact about the disk, not the inputs:
        # served from cache, a deleted profile folder would hand the loop a
        # path to nothing. Re-running is cheap -- `samples` is lazy, so the
        # sampler is only asked for when there is no reusable file.
        return float("nan")

    def check_lazy_status(self, joint, base_folder, profile, reuse_existing=True,
                          samples=None, **_):
        if samples is not None:
            return []
        _rel, _folder, path = self._target(base_folder, profile)
        if reuse_existing and self._reusable(path, joint):
            return []          # nothing to sample: the file stands
        return ["samples"]

    def store(self, joint, base_folder, profile, reuse_existing=True, samples=None):
        rel, folder, path = self._target(base_folder, profile)
        if samples is None:
            if not (reuse_existing and self._reusable(path, joint)):
                raise RuntimeError("H3 Joint Store: no samples and no reusable "
                                   "joint file -- this should not happen.")
            _LOG.info("obvpm.h3 joint: reusing %s (same timeline); the "
                      "sampler was not run", path)
            return ("%s/%s" % (rel, JOINT_FILE), rel)
        video, audio = avpack.unpack_av(samples, name="samples")
        if video.shape[2] != joint["total_steps"]:
            raise ValueError("H3 Joint Store: the sampled latent has %d steps "
                             "but the joint layout expects %d"
                             % (video.shape[2], joint["total_steps"]))
        os.makedirs(folder, exist_ok=True)
        meta = {
            "format": mctx.FORMAT, "self_id": "", "parent_id": "",
            "relation": "joint", "parent_join_frame": "",
            "width": str(int(video.shape[4]) * 16),
            "height": str(int(video.shape[3]) * 16),
            "fps": str(fr.FPS), "raw_frames": str(joint["total_frames"]),
            "pinned_head_frames": "0", "pinned_tail_frames": "0",
            "delivered_frames": str(joint["total_frames"]), "pins": "[]",
            "parent_grade": "latent", "user_meta": "{}",
        }
        record = {k: (list(map(list, v)) if k in ("steps", "ticks") else v)
                  for k, v in joint.items()}
        # which sampling this is: a clip sliced from an earlier joint file
        # in the same folder must not pass for one sliced from this one
        record["stamp"] = "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"),
                                     os.urandom(4).hex())
        mctx.write_sidecar(path, video, audio, meta, blobs={JOINT_BLOB: record})
        _LOG.info("obvpm.h3 joint: stored %d steps for %d clip(s) -> %s",
                  video.shape[2], len(joint["clips"]), path)
        return ("%s/%s" % (rel, JOINT_FILE), rel)


_CACHE = {}


def load_joint(path):
    """(video, audio, record) for a joint file, cached by path + mtime."""
    key = (path, os.path.getmtime(path))
    hit = _CACHE.get(key)
    if hit is None:
        _CACHE.clear()
        video, audio, _ = mctx.load_sidecar(path)
        record = mctx.read_blob(path, JOINT_BLOB)
        if not record:
            raise ValueError("%s carries no joint layout" % path)
        hit = _CACHE[key] = (video, audio, record)
    return hit


def read_record(path):
    """The joint layout alone (no latents read)."""
    record = mctx.read_blob(path, JOINT_BLOB)
    if not record or not record.get("clips"):
        raise ValueError("%s is not a joint latent file (no layout in it)"
                         % path)
    return record


def mirror_pins(header, id_map=None):
    """Trim-only PINS from a take's recorded recipe.

    The refined clip covers the source's raw span exactly, so the rows to
    trim off, the lineage to record and the mask shape the assembly reads
    are the source's own -- no slicing, no keyframes, nothing sampled.
    `id_map` renames pin sources from the ORIGINAL neighbour's id to its
    refined rendering's, so the refined clips carry the same lineage among
    themselves that the sources had, and H3 Assemble derives the same cut.
    """
    pins = []
    for spec in mctx.parse_pins(header):
        spec = dict(spec)
        sid = spec.get("source_id")
        if id_map and sid in id_map:
            spec["source_id"] = id_map[sid]
        pins.append({"place": spec.get("place"),
                     "covered": int(spec.get("source_frames", 0) or 0),
                     "spec": spec})
    return pins


class H3JointSlice:
    """One clip's span of the jointly refined latent, for decode and save."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "slice"
    RETURN_TYPES = ("LATENT", wt.PINS, "LATENT")
    RETURN_NAMES = ("latent", "pins", "source_audio")
    DESCRIPTION = (
        "Inside the loop: takes this iteration's clip out of the stored "
        "joint latent -- the clip's own raw span -- and mirrors the take's "
        "recorded pins as trim-only pins, so the save node trims and records "
        "it exactly as the source was, pointing at the refined neighbours. "
        "source_audio is the clip's original audio latent, for a save that "
        "keeps the finished soundtrack."
    )
    OUTPUT_TOOLTIPS = (
        "The refined raw AV latent of this clip. Wire to decode and to the "
        "save node's samples.",
        "Trim-only pins. Wire to the save node's pins.",
        "The source take's audio latent. Decode it for the save node's audio "
        "when the joint pass re-sampled sound.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (wt.LOOP, {"tooltip": "From H3 Upscale Loop Start."}),
            },
        }

    def slice(self, flow):
        if not isinstance(flow, dict) or "index" not in flow or not flow.get("joint_path"):
            raise ValueError("H3 Joint Slice: the flow must come from Loop Start.")
        video, audio, record = load_joint(flow["joint_path"])
        index = int(flow["index"])
        clips = record.get("clips") or []
        if index >= len(clips) or clips[index] != flow.get("source"):
            raise ValueError(
                "H3 Joint Slice: the loop is on %r (position %d) but the joint "
                "latent was built for %s. Rebuild the joint latent for this "
                "timeline." % (flow.get("source"), index, clips))
        s, n = record["steps"][index]
        t, m = record["ticks"][index]
        v = video[:, :, s:s + n].clone()
        a = audio[..., t:t + m].clone()

        source = mctx.sidecar_path(nodes_load.resolve_clip_path(clips[index]))
        _sv, source_audio, header = mctx.load_sidecar(source)
        id_map = {}
        for neighbour, output in (flow.get("pinned_to") or {}).items():
            try:
                sid = mctx.read_header(mctx.sidecar_path(
                    nodes_load.resolve_clip_path(neighbour))).get("self_id")
                rid = mctx.read_header(mctx.sidecar_path(
                    nodes_load.resolve_clip_path(
                        "%s/%s" % (flow["rel_folder"], output)))).get("self_id")
            except Exception:
                _LOG.exception("obvpm.h3 joint: could not read %s / %s for "
                               "the refined lineage", neighbour, output)
                continue
            if sid and rid:
                id_map[sid] = rid
        pins = mirror_pins(header, id_map)
        _LOG.info("obvpm.h3 joint: slice %d -> steps %d..%d, ticks %d..%d; "
                  "%d pin(s) mirrored%s", index, s, s + n, t, t + m, len(pins),
                  (", lineage -> refined %s" % ", ".join(
                      (flow.get("pinned_to") or {}).values())) if id_map else "")
        return (avpack.pack_av(v, a, name="joint"), pins,
                {"samples": source_audio})
