"""Joint refine: sample the whole timeline in one loop, in sliding windows.

Every per-clip fix so far (renoise, chain noise, keyframe reference,
upscaler padding, a 27-step hold) left the refined seam where it was,
because a per-clip refine re-invents fine texture from its own prior and
noise, and no amount of held neighbour makes it copy the neighbour's
invention (measured 2026-09-04: the refine moves 74% of the high-pass
texture away from the upscaled prior, per clip, per seed).

A single long clip refines seamlessly because every row is sampled in
one pass with every other row in view. This module gives the chain the
same treatment: the clips' raw latents are laid onto ONE timeline latent
at their true positions, upscaled as one, and sampled as one -- in
context windows the size of a clip, overlapping by a third, with the
model's predictions blended across the overlap at every step (core's
context-windows machinery, MultiDiffusion along time). Texture decisions
are then shared across every join. Per step the model runs once per
window, so a two-clip chain costs what two clips cost; memory is one
window's worth.

The result is split back into the clips' own raw spans and handed to the
existing loop, which decodes, trims and saves each as a refined take and
keeps the manifest, so H3 Assemble Upscale plays it unchanged.

Nodes: H3 Joint Latent (timeline -> one latent), H3 Context Windows (the
model patch: a window handler written for H3's two streams), H3 Joint Store (the sampled timeline -> disk), H3 Joint Slice
(one clip's span back out, inside the loop).
"""

import logging
import os

import torch

from . import avpack
from . import frames as fr
from . import mctx
from . import nodes_load
from . import nodes_noise
from . import upscale
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

JOINT_FILE = "joint.mctx.safetensors"
JOINT_BLOB = "joint"


# ---------------------------------------------------------------------------
# the timeline as one latent
# ---------------------------------------------------------------------------

def timeline_layout(lines, headers, audio_ticks):
    """Where each clip's raw latent sits on one joint latent.

    `lines` are the sequence lines (markers kept), `headers` the clips'
    sidecar headers, `audio_ticks` each clip's audio length in ticks.
    Returns {"steps": [(start, len)], "ticks": [(start, len)],
    "frame_offset", "total_steps", "total_frames", "total_ticks"}.
    Starts come from the same raw-start arithmetic chain noise uses, so
    a clip's rows land on exactly the steps its noise did.
    """
    from .nodes_loop import H3UpscaleLoopStart
    starts_frames = H3UpscaleLoopStart._raw_starts(lines, headers)
    first = min(starts_frames)
    steps, ticks = [], []
    for header, start, a_len in zip(headers, starts_frames, audio_ticks):
        rel = start - first
        s = nodes_noise.frame_offset_to_steps(rel)
        if fr.frame_at_latent(s) != rel:
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
    return {"steps": steps, "ticks": ticks, "frame_offset": int(first),
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
    RETURN_TYPES = ("LATENT", wt.JOINT, "INT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "joint", "frame_offset", "first_clip", "info")
    DESCRIPTION = (
        "Lays every clip of the timeline onto ONE raw AV latent at its true "
        "position (held windows coincide), for a joint refine: upscale it as "
        "one, sample it as one under H3 Context Windows, store it with H3 "
        "Joint Store, and let the loop slice each clip's span back out. "
        "Texture is then decided across the joins, not per clip."
    )
    OUTPUT_TOOLTIPS = (
        "The joint raw AV latent (video + audio), for the upscaler.",
        "Where each clip sits on it. Wire to H3 Joint Store.",
        "Timeline frame the joint latent starts at, for H3 Chain Noise.",
        "The first clip's path, for H3 MCtx Load by Path -> Load Conditioning "
        "(the joint pass samples under one clip's prompt and refs).",
        "One line per clip: its span.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The timeline, one output-relative clip per "
                               "line -- the same text Loop Start gets."}),
            },
        }

    def build(self, sequence):
        from .nodes_loop import H3UpscaleLoopStart
        clips = upscale.parse_sequence(sequence)
        lines = upscale.parse_sequence_lines(sequence)
        if not clips:
            raise ValueError("H3 Joint Latent: the sequence is empty.")
        headers = H3UpscaleLoopStart._headers(clips)
        videos, audios = [], []
        for clip in clips:
            v, a, _ = mctx.load_sidecar(mctx.sidecar_path(
                nodes_load.resolve_clip_path(clip)))
            videos.append(v)
            audios.append(a)
        layout = timeline_layout(lines, headers, [a.shape[-1] for a in audios])
        video, audio = assemble(videos, audios, layout)
        record = dict(layout, clips=list(clips))
        info = "\n".join(
            "%s: steps %d..%d, ticks %d..%d"
            % (clip, s, s + n, t, t + m)
            for clip, (s, n), (t, m) in zip(clips, layout["steps"], layout["ticks"]))
        _LOG.info("obvpm.h3 joint: %d clip(s) -> %d steps (%d frames), %d "
                  "audio ticks, timeline frame %d\n%s", len(clips),
                  layout["total_steps"], layout["total_frames"],
                  layout["total_ticks"], layout["frame_offset"], info)
        return (avpack.pack_av(video, audio, name="joint"), record,
                int(layout["frame_offset"]), clips[0], info)


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
# proportion), conds sliced on the dim each one keeps time on, and the
# two streams' predictions blended with pyramid weights along their own
# time axes.
# ---------------------------------------------------------------------------

def static_windows(total, length, overlap):
    """Fixed windows of `length`, as few as keep at least `overlap` shared
    steps, spread evenly so the last is not a near-copy of the one before.

    Core's static schedule steps by length-overlap and pulls the final
    window back to the end, which for 162/87/21 makes three windows with
    the last two nearly coincident. Here the count is the minimum and the
    starts are spread from 0 to total-length; for a two-clip chain, a
    window of 92 with overlap 22 is exactly two windows [0:92],[70:162].
    """
    import math
    if total <= length:
        return [(0, total)]
    n = math.ceil((total - length) / max(1, length - overlap)) + 1
    starts = [round(i * (total - length) / (n - 1)) for i in range(n)]
    return [(s, s + length) for s in starts]


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


class H3WindowHandler:
    """Windowed calc_cond_batch for H3's packed video+audio latent."""

    def __init__(self, length, overlap, fuse="pyramid"):
        self.context_length = int(length)
        self.context_overlap = int(overlap)
        self.fuse = fuse
        self._announced = False

    # -- what core calls ----------------------------------------------------

    @staticmethod
    def _shapes(conds):
        for cond_list in conds:
            for cond in cond_list or []:
                shapes = (cond.get("model_conds") or {}).get("latent_shapes")
                if shapes is not None and len(shapes.cond) > 1:
                    return list(shapes.cond)
        return None

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        shapes = self._shapes(conds)
        if shapes is None:
            return False
        total = int(shapes[0][2])
        use = total > self.context_length
        if not self._announced:
            self._announced = True
            _LOG.info("obvpm.h3 context windows: %d video steps, window %d, "
                      "overlap %d -> %s", total, self.context_length,
                      self.context_overlap,
                      "windows %s" % static_windows(total, self.context_length,
                                                    self.context_overlap)
                      if use else "one window, plain sampling")
        return use

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        import comfy.utils
        shapes = self._shapes(conds)
        video, audio = comfy.utils.unpack_latents(x_in, shapes)[:2]
        T, A = int(video.shape[2]), int(audio.shape[-1])
        windows = static_windows(T, self.context_length, self.context_overlap)
        acc_v = [torch.zeros_like(video) for _ in conds]
        acc_a = [torch.zeros_like(audio) for _ in conds]
        cnt_v = torch.zeros(T, device=video.device, dtype=torch.float32)
        cnt_a = torch.zeros(A, device=audio.device, dtype=torch.float32)
        import time
        for k, (s, e) in enumerate(windows):
            t0 = time.time()
            ta, tb = audio_span(s, e)
            tb = min(tb, A)
            if e == T:
                tb = A          # the last window owns the grid's last ticks
            v_win, a_win = video[:, :, s:e], audio[..., ta:tb]
            sub_x, sub_shapes = comfy.utils.pack_latents([v_win, a_win])
            sub_conds = [self._slice_conds(c, s, e, ta, tb, sub_shapes)
                         for c in conds]
            _LOG.info("obvpm.h3 context windows: sigma %.3f window %d/%d "
                      "steps %d..%d ticks %d..%d", float(timestep.flatten()[0]),
                      k + 1, len(windows), s, e, ta, tb)
            outs = calc_cond_batch(model, sub_conds, sub_x, timestep, model_options)
            _LOG.info("obvpm.h3 context windows: window %d/%d done in %.0fs",
                      k + 1, len(windows), time.time() - t0)
            w_v = self._weights(e - s).to(video.device)
            w_a = self._weights(tb - ta).to(audio.device)
            for i, out in enumerate(outs):
                if out is None:
                    continue
                ov, oa = comfy.utils.unpack_latents(out, sub_shapes)[:2]
                acc_v[i][:, :, s:e] += ov * w_v.view(1, 1, -1, 1, 1).to(ov.dtype)
                acc_a[i][..., ta:tb] += oa * w_a.view(1, 1, 1, -1).to(oa.dtype)
            cnt_v[s:e] += w_v
            cnt_a[ta:tb] += w_a
            del outs, sub_x, sub_conds
            # the window's activations are gone; hand the cached blocks
            # back so the next window does not push the allocator past
            # the card (Windows spills silently to system RAM, and then
            # every step crawls)
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        cnt_v = cnt_v.clamp(min=1e-6).view(1, 1, -1, 1, 1)
        cnt_a = cnt_a.clamp(min=1e-6).view(1, 1, 1, -1)
        results = []
        for i in range(len(conds)):
            if conds[i] is None:
                # core hands back zeros for an absent cond (cfg 1 still
                # computes uncond + (cond - uncond) with it), never None
                results.append(torch.zeros_like(x_in))
                continue
            v = acc_v[i] / cnt_v.to(acc_v[i].dtype)
            a = acc_a[i] / cnt_a.to(acc_a[i].dtype)
            results.append(comfy.utils.pack_latents([v, a])[0])
        return results

    # -- pieces -----------------------------------------------------------------

    def _weights(self, n):
        if self.fuse == "flat":
            return torch.ones(n)
        return pyramid(n)

    @staticmethod
    def _slice_conds(cond_list, s, e, ta, tb, sub_shapes):
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
            new["model_conds"] = mc
            out.append(new)
        return out


class H3ContextWindows:
    """Sliding-window sampling for an H3 model, in latent steps."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Samples a long AV latent in windows of context_length steps that "
        "overlap by context_overlap, blending the model's predictions across "
        "the overlap every step (MultiDiffusion along time, for H3's "
        "video+audio latent). One window's worth of memory, one model call "
        "per window per step. For the joint refine: a clip-sized window over "
        "the whole timeline."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": ("INT", {
                    "default": 96, "min": 2, "max": 4096,
                    "tooltip": "Window length in VIDEO latent steps (87 = a "
                               "294-frame clip). Tokens per window are what "
                               "attention and memory pay for: at 1920x1088, "
                               "96 steps is about a clip's cost."}),
                "context_overlap": ("INT", {
                    "default": 30, "min": 0, "max": 4096,
                    "tooltip": "Steps shared by neighbouring windows; the "
                               "blend happens here. About a third of the "
                               "window."}),
                "fuse_method": (["pyramid", "flat"], {
                    "default": "pyramid",
                    "tooltip": "How overlapping predictions are weighted: "
                               "pyramid (triangular over the window), flat "
                               "(plain average)."}),
            },
        }

    def patch(self, model, context_length, context_overlap, fuse_method):
        if int(context_overlap) >= int(context_length):
            raise ValueError("H3 Context Windows: the overlap must be smaller "
                             "than the window.")
        model = model.clone()
        model.model_options["context_handler"] = H3WindowHandler(
            context_length, context_overlap, fuse_method)
        _LOG.info("obvpm.h3 context windows: %d steps, overlap %d, %s",
                  int(context_length), int(context_overlap), fuse_method)
        return (model,)


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
        "it. H3 Joint Slice reads clips back out of it inside the loop. Give "
        "Loop Start the SAME profile name so the sliced clips land in the "
        "same folder. With reuse_existing on, a joint file already there for "
        "this timeline is kept and the sampler is NOT run again -- the Timeline "
        "re-runs everything downstream on every press, and 20 minutes of "
        "sampling is not something to repeat because a save step failed."
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
                    "tooltip": "Profile folder name; type the same into Loop "
                               "Start's profile."}),
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


class H3JointSlice:
    """One clip's span of the jointly refined latent, for decode and save."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "slice"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Inside the loop: takes this iteration's clip out of the stored "
        "joint latent -- the clip's own raw span, so the save node trims "
        "and records it exactly as a per-clip refine would."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joint_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "From H3 Joint Store."}),
                "flow": (wt.LOOP, {"tooltip": "From H3 Upscale Loop Start."}),
            },
        }

    def slice(self, joint_path, flow):
        if not isinstance(flow, dict) or "index" not in flow:
            raise ValueError("H3 Joint Slice: the flow must come from Loop Start.")
        path = nodes_load.resolve_clip_path(str(joint_path))
        video, audio, record = load_joint(path)
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
        _LOG.info("obvpm.h3 joint: slice %d -> steps %d..%d, ticks %d..%d",
                  index, s, s + n, t, t + m)
        return (avpack.pack_av(v, a, name="joint"),)
