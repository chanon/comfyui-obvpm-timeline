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


def raw_starts(lines, headers):
    """Each clip's RAW-latent start on the timeline, in frames.

    Delivered frame f of clip i sits at T_i + (f - enter_i), where T_i is
    the delivered duration of everything before it; raw frame r is
    delivered frame r - head_i. So the raw latent starts at
    T_i - enter_i - head_i -- negative for a first clip whose pinned head
    precedes the timeline, which is fine: it is a coordinate. For a
    masked extend this puts the child's held window on exactly the
    parent's frames it was cut from.
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
    return {"steps": steps, "ticks": ticks,
            "total_steps": int(total_steps), "total_frames": int(total_frames),
            "total_ticks": int(total_ticks)}


def assemble(videos, audios, layout):
    """Lay the clips' raw latents onto one joint (video, audio) pair.

    First come, first written: an overlap (a held window) keeps the
    earlier clip's rows, and the later clip's copy is compared and
    logged -- identical for a masked chain, a measure of the join's
    fidelity otherwise.
    """
    v0, a0 = videos[0], audios[0]
    T, A = layout["total_steps"], layout["total_ticks"]
    video = torch.zeros(v0.shape[:2] + (T,) + v0.shape[3:], dtype=v0.dtype)
    audio = torch.zeros(a0.shape[:-1] + (A,), dtype=a0.dtype)
    v_written = torch.zeros(T, dtype=torch.bool)
    a_written = torch.zeros(A, dtype=torch.bool)
    for i, (v, a) in enumerate(zip(videos, audios)):
        s, n = layout["steps"][i]
        if v.shape[2] != n:
            raise ValueError("H3 Joint Latent: clip %d's latent has %d steps "
                             "but its header says %d" % (i, v.shape[2], n))
        if tuple(v.shape[3:]) != tuple(v0.shape[3:]):
            raise ValueError("H3 Joint Latent: clip %d is %s, clip 0 is %s; "
                             "one timeline needs one size"
                             % (i, tuple(v.shape[3:]), tuple(v0.shape[3:])))
        overlap = v_written[s:s + n]
        if overlap.any():
            k = int(overlap.sum())
            idx = torch.nonzero(overlap).flatten()
            have = video[:, :, s + idx].float()
            new = v[:, :, idx].float()
            rel = float(((have - new) ** 2).mean().sqrt() / (have ** 2).mean().sqrt().clamp(min=1e-8))
            _LOG.info("obvpm.h3 joint: clip %d overlaps %d step(s) already "
                      "placed; keeping the earlier rows (rel diff %.4f)", i, k, rel)
        free = ~overlap
        if free.any():
            fidx = torch.nonzero(free).flatten()
            video[:, :, s + fidx] = v[:, :, fidx].to(video.dtype)
            v_written[s + fidx] = True
        t, m = layout["ticks"][i]
        m = min(m, a.shape[-1], A - t)
        a_free = ~a_written[t:t + m]
        if a_free.any():
            aidx = torch.nonzero(a_free).flatten()
            audio[..., t + aidx] = a[..., aidx].to(audio.dtype)
            a_written[t + aidx] = True
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

    def build(self, joint, clip=None, vae=None, audio_vae=None):
        clips = list(joint["clips"])
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
                 "conds": conds}
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


def _log_under_bars():
    """Context: console log lines go through tqdm.write.

    A tqdm bar (the sampler's step bar, or ours) leaves the cursor at the
    end of its line; a log line written then starts there. tqdm.write
    clears every live bar, prints at column 0 and redraws the bars.
    """
    from tqdm.contrib.logging import logging_redirect_tqdm
    return logging_redirect_tqdm()


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

    def __init__(self, length, overlap, fuse="pyramid", headroom_gb=10.0):
        self.context_length = clip_shaped(length)
        self.context_overlap = min(int(overlap), self.context_length - 1)
        self.fuse = fuse
        self.headroom_mb = max(0.0, float(headroom_gb)) * 1024.0
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
            windows = (static_windows(total, self.context_length, self.context_overlap)
                       if use else [(0, total)])
            with _log_under_bars():
                self._announce(total, use, windows, table)
        return use

    def _announce(self, total, use, windows, table):
        """The run's windows and their owners, once per conditioning table."""
        _LOG.info("obvpm.h3 context windows: %d video steps, window %d, "
                  "overlap %d -> %s", total, self.context_length,
                  self.context_overlap,
                  "windows %s" % windows if use else "one window, plain sampling")
        if table is None:
            _LOG.info("obvpm.h3 context windows: no per-clip conditioning "
                      "on the wire; every window samples under the "
                      "conditioning given (wire H3 Joint Conditioning "
                      "for per-clip prompts)")
        else:
            for k, (s, e) in enumerate(windows):
                j = window_owner(table["spans"], s, e)
                _LOG.info("obvpm.h3 context windows: window %d/%d steps "
                          "%d..%d conditioned by clip %d (%s)", k + 1,
                          len(windows), s, e, j, table["clips"][j])

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        import comfy.utils
        shapes = self._shapes(conds)
        video, audio = comfy.utils.unpack_latents(x_in, shapes)[:2]
        T, A = int(video.shape[2]), int(audio.shape[-1])
        windows = static_windows(T, self.context_length, self.context_overlap)
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
                   bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} "
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
            line.set_description_str("    " + text + left, refresh=True)
        t_step, worst, spilled = time.time(), None, False
        # while the bar is up, log lines from anywhere go through tqdm.write
        # so they land at column 0 and the bar is redrawn under them
        try:
            with _log_under_bars():
                for k, (s, e) in enumerate(windows):
                    t0 = time.time()
                    ta, tb = audio_span(s, e)
                    tb = min(tb, A)
                    if e == T:
                        tb = A          # the last window owns the grid's last ticks
                    v_win, a_win = video[:, :, s:e], audio[..., ta:tb]
                    sub_x, sub_shapes = comfy.utils.pack_latents([v_win, a_win])
                    sub_conds = [self._window_conds(model, c, s, e, ta, tb, sub_shapes,
                                                    x_in.device)
                                 for c in conds]
                    owner = window_owner(table["spans"], s, e) if table else None
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
            v = (acc_v[i] / cnt_v).to(device=x_in.device, dtype=video.dtype)
            a = (acc_a[i] / cnt_a).to(device=x_in.device, dtype=audio.dtype)
            results.append(comfy.utils.pack_latents([v, a])[0])
        return results

    # -- pieces -----------------------------------------------------------------

    def _weights(self, n):
        if self.fuse == "flat":
            return torch.ones(n)
        return pyramid(n)

    def _window_conds(self, model, cond_list, s, e, ta, tb, sub_shapes, device):
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
                                           device, mc))
                new.pop(COND_KEY, None)
            new["model_conds"] = mc
            out.append(new)
        return out

    def _clip_conds(self, model, table, s, e, sub_shapes, device, base_mc):
        """The owning clip's model_conds for window [s, e), cached per run.

        Runs the model's own `extra_conds` on the clip's stored
        conditioning at the window's shapes -- the same call the sampler
        made for the base conditioning -- so the text embedding, the
        reference blocks and the packed layout are exactly what that clip
        generated under. Content keyframes recorded against the clip's
        own frame 0 are moved to where the clip sits in the window.
        """
        j = window_owner(table["spans"], s, e)
        key = (j, s, e)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        entry = table["conds"][j][0]
        cross_attn, extras = entry[0], entry[1]
        params = {k: v for k, v in extras.items()
                  if k not in (COND_KEY, "model_conds")}
        params["cross_attn"] = cross_attn
        params["device"] = device
        params["latent_shapes"] = list(sub_shapes)
        payload = base_mc.get("minimax_payload")
        params["seed"] = (payload.cond or {}).get("seed", 0) if payload is not None else 0
        keyframes = params.get("minimax_keyframes")
        if keyframes:
            clip_start = int(table["spans"][j][0])
            shift = fr.frame_at_latent(clip_start) - fr.frame_at_latent(s)
            params["minimax_keyframes"] = [
                dict(kf, resolved_frame_index=kf.get("resolved_frame_index", 0) + shift)
                if isinstance(kf, dict) else kf for kf in keyframes]
        built = model.extra_conds(**params)
        keep = {k: v for k, v in built.items() if k not in _PER_CLIP_SKIP}
        self._cache[key] = keep
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
                "fuse_method": (["pyramid", "flat"], {
                    "default": "pyramid",
                    "tooltip": "How overlapping predictions are weighted: "
                               "pyramid (triangular over the window), flat "
                               "(plain average)."}),
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
            },
        }

    def patch(self, model, window_seconds, overlap_seconds, fuse_method,
              vram_headroom_gb=10.0):
        length = fr.steps_for_seconds(window_seconds)
        overlap = fr.steps_for_seconds(overlap_seconds) if float(overlap_seconds) > 0 else 0
        if overlap >= length:
            raise ValueError("H3 Context Windows: the overlap (%.2f s = %d "
                             "steps) must be shorter than the window (%.2f s "
                             "= %d steps)." % (float(overlap_seconds), overlap,
                                               float(window_seconds), length))
        import comfy.patcher_extension
        model = model.clone()
        handler = H3WindowHandler(length, overlap, fuse_method, vram_headroom_gb)
        model.model_options["context_handler"] = handler
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
