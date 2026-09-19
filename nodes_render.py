"""H3 Joint VAE Decode and Save: the sampled timeline latent -> one finished MP4.

The joint refine samples the whole timeline as one latent (nodes_joint.py),
and that latent IS the cut: every row belongs to the clip the cut shows
there. So the finished video is one decode of it, and this node is the
end of the upscale branch -- there is no loop, no profile folder and no
per-clip take to reassemble. (Until 2026-09-14 the sampled latent was
sliced back into refined takes and reassembled; that produced clips
nobody edits again, at the price of a loop in the graph.)

DECODED IN WINDOWS, STREAMED TO THE ENCODER. A timeline is long: 1400
frames at 1920x1088 are 35 GB of float pixels, so the frames never exist
all at once. The video latent is decoded a few seconds at a time and each
block is piped straight into ffmpeg (the save node's own encoder path).
The window is exact, not approximate: core's H3 decoder already works in
chunks of 5 latent steps with a 2-step lookahead and blends 5 frames into
the next chunk, so a window that starts and ends on 5-step boundaries,
decoded with one extra chunk of run-in before it and the lookahead after
it, reproduces the whole-latent decode bit for bit (the run-in's 17
frames are dropped). Measured 2026-09-14 on a 42-step clip latent
decoded whole and in three windows: max abs difference 0.0 on every
frame.

WHAT IS SAVED BESIDE THE MP4, and why: the rendered cut is a TAKE, with
a `.mctx.safetensors` sidecar holding the refined AV latent (a root
clip: no pins, no lineage) and, when the joint conditioning is wired, a
`.cond` holding what the sampler was conditioned with -- the table of
every source clip's conditioning and span. Put on a timeline by itself,
the rendered cut can therefore be refined AGAIN (4x from the sources):
H3 Joint Conditioning splices a nested table back in with its spans
offset, so the second pass still samples each stretch under the prompt
and references it was made with.
"""

import json
import logging
import os
import time

import folder_paths

from . import avpack
from . import frames as fr
from . import mctx
from . import nodes_save
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

# the decoder's own chunk (5 latent steps) and lookahead (2), in steps --
# see comfy/ldm/minimax/vae.py: tokens_chunk_size and token_overlap
CHUNK = 5
LOOKAHEAD = 2


def window_plan(total_steps, window_steps):
    """[(s, e, s0, e1)] windows over a latent of `total_steps` steps.

    [s, e) is the span whose frames a window delivers; [s0, e1) is what
    it decodes: one chunk of run-in before (except at 0) and the
    decoder's lookahead after (except at the end). Every span starts on
    a chunk boundary and every decoded slice is clip-shaped (5k+2), so
    the decoder pads nothing and its chunks fall exactly where the
    whole-latent decode's would. A final span shorter than a chunk is
    folded into the window before it.
    """
    total = int(total_steps)
    step = max(CHUNK, (int(window_steps) // CHUNK) * CHUNK)
    out, s = [], 0
    while s < total:
        e = min(s + step, total)
        if total - e < CHUNK:
            e = total
        out.append((s, e, max(s - CHUNK, 0), min(e + LOOKAHEAD, total)))
        s = e
    return out


def decode_windows(vae, video, window_steps):
    """Yield the whole video latent's frames as [n, H, W, 3] blocks."""
    import comfy.utils
    from tqdm.auto import tqdm
    from .nodes_joint import _hms, _log_under_bars
    total_steps = int(video.shape[2])
    total = fr.pixel_frames(total_steps)
    plan = window_plan(total_steps, window_steps)
    node_bar = comfy.utils.ProgressBar(len(plan))       # the node's own arc
    show = comfy.utils.PROGRESS_BAR_ENABLED
    # a console gauge like the sampler's windows: one tick per window, the
    # time left from the pace so far
    bar = tqdm(total=len(plan), desc="obvpm.h3 render", leave=False,
               dynamic_ncols=True, disable=not show,
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}, {rate_inv_fmt}]")
    produced, durations, t_all = 0, [], time.time()
    try:
        with _log_under_bars():
            for k, (s, e, s0, e1) in enumerate(plan):
                t0 = time.time()
                part = vae.decode(video[:, :, s0:e1])
                # [B, F, H, W, 3] from a video VAE; core's own decode node
                # flattens the batch the same way
                part = part.reshape(-1, *part.shape[-3:])
                skip = fr.frame_at_latent(s) - fr.frame_at_latent(s0)
                want = ((fr.frame_at_latent(e) - fr.frame_at_latent(s))
                        if e < total_steps else total - fr.frame_at_latent(s))
                block = part[skip:skip + want]
                if int(block.shape[0]) != want:
                    raise RuntimeError(
                        "H3 Joint VAE Decode and Save: decoding steps %d..%d gave %d frames, "
                        "expected at least %d -- that is not an H3 video VAE, "
                        "or the latent grid moved."
                        % (s0, e1, int(part.shape[0]), skip + want))
                took = time.time() - t0
                durations.append(took)
                left = (len(plan) - k - 1) * sum(durations) / len(durations)
                produced += want
                node_bar.update(1)
                bar.update(1)
                _LOG.info("obvpm.h3 render: window %d/%d, steps %d..%d -> frames "
                          "%d..%d in %.0fs%s", k + 1, len(plan), s, e,
                          produced - want, produced, took,
                          (", ~%s left" % _hms(left)) if left > 0 else "")
                yield block
                del part, block
    finally:
        bar.close()
    if produced != total:
        raise RuntimeError("H3 Joint VAE Decode and Save: decoded %d frames of %d"
                           % (produced, total))
    _LOG.info("obvpm.h3 render: %d frames decoded in %s", produced,
              _hms(time.time() - t_all))


def _audio_of(latent):
    """The audio latent [B, 32, 2, T] inside a LATENT: either an H3 AV
    pair (H3 Join Latents' output) or the audio-only latent Separate AV
    Latent hands out. None when there is no audio stream."""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False) or isinstance(samples, (tuple, list)):
        return avpack.unpack_av(latent, name="source_audio")[1]
    if getattr(samples, "ndim", 0) == 3:
        samples = samples.unsqueeze(0)
    if getattr(samples, "ndim", 0) == 4 and int(samples.shape[1]) == 32:
        return samples
    return None


def decode_audio(audio_vae, audio):
    """The soundtrack as core's AUDIO dict, through core's own decode."""
    from comfy_extras.nodes_audio import vae_decode_audio
    return vae_decode_audio(audio_vae, {"samples": audio})


class H3JointRender:
    """The sampled timeline latent, decoded in windows into one MP4."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "render"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    DESCRIPTION = (
        "Writes the jointly refined timeline as one finished MP4, with a "
        ".mctx.safetensors sidecar (the refined latent) and, when the joint "
        "conditioning is wired, a .cond -- so the rendered sequence is a take "
        "that can itself be put on a timeline and refined again. The "
        "picture is decoded a few seconds at a time and streamed to the "
        "encoder, so a long timeline costs one window of memory. Wire the "
        "sampler's output to samples."
    )
    OUTPUT_TOOLTIPS = ("Path of the written MP4; the sidecar sits next to it.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "The sampler's output: the refined joint AV "
                               "latent."}),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE."}),
                "base_folder": ("STRING", {
                    "default": "project1",
                    "tooltip": "Output-relative folder to save into. Same "
                               "meaning as the Timeline's base_folder. "
                               "Empty = the output root."}),
                "filename_prefix": ("STRING", {
                    "default": "upscale",
                    "tooltip": "Filename prefix within base_folder; "
                               "numbering is appended automatically."}),
                "crf": ("INT", {
                    "default": 19, "min": 0, "max": 51,
                    "tooltip": "H.264 quality (lower = better, bigger)."}),
                "window_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 600.0, "step": 0.5,
                    "tooltip": "Seconds of picture decoded at a time, rounded "
                               "to whole decoder chunks. Only memory changes "
                               "with it: the frames of one window sit in RAM "
                               "while they are encoded (about 5 GB for 5 s "
                               "at 1920x1088). The output is identical at "
                               "any value."}),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "The CONDITIONING the sampler ran under (H3 "
                               "Joint Conditioning's output). Stored beside "
                               "the render as .cond so the rendered take can be "
                               "refined again. Unwired = the render saves "
                               "normally and cannot be refined further."}),
                "source_audio": ("LATENT", {
                    "tooltip": "Optional: a latent whose AUDIO is rendered "
                               "instead of the sampled one (H3 Join Latents' "
                               "latent, or Separate AV Latent's audio_latent). "
                               "Wire it when H3 Joint Audio Mask "
                               "re-sampled the sound (audio_denoise above 0, "
                               "for lip sync), so the finished render keeps the "
                               "source soundtrack. The sidecar stores the "
                               "audio that was rendered."}),
                "layout": (wt.JOINT, {
                    "tooltip": "From H3 Join Latents, for provenance: the "
                               "source clips and sequence lines are recorded "
                               "in the sidecar, and a latent that is not the "
                               "timeline the joint describes is refused."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def render(self, samples, vae, audio_vae, base_folder, filename_prefix, crf,
               window_seconds=5.0, conditioning=None, layout=None,
               source_audio=None, prompt=None, extra_pnginfo=None):
        video, audio = avpack.unpack_av(samples, name="samples")
        if source_audio is not None:
            src = _audio_of(source_audio)
            if src is None or tuple(src.shape) != tuple(audio.shape):
                raise ValueError(
                    "H3 Joint VAE Decode and Save: source_audio's audio latent is %s but "
                    "the sampled one is %s; wire H3 Join Latents' latent "
                    "for THIS timeline." % (None if src is None else tuple(src.shape),
                                            tuple(audio.shape)))
            audio = src
            _LOG.info("obvpm.h3 render: rendering the source soundtrack, not "
                      "the sampled one")
        total_steps = int(video.shape[2])
        frames = fr.pixel_frames(total_steps)
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        if layout and int(layout.get("total_steps", total_steps)) != total_steps:
            raise ValueError(
                "H3 Joint VAE Decode and Save: the latent has %d steps but the joint "
                "layout describes %d. Wire the sampler that refined THIS "
                "timeline." % (total_steps, int(layout["total_steps"])))

        full_folder, filename, counter, _subfolder, _ = \
            folder_paths.get_save_image_path(
                nodes_save._save_prefix(base_folder, filename_prefix),
                folder_paths.get_output_directory(), width, height)
        video_path = os.path.join(full_folder, "%s_%05d.mp4" % (filename, counter))

        window_steps = fr.steps_for_seconds(window_seconds) or CHUNK
        plan = window_plan(total_steps, window_steps)
        _LOG.info("obvpm.h3 render: %d steps (%d frames, %dx%d) in %d window(s) "
                  "of up to %d steps -> %s", total_steps, frames, width, height,
                  len(plan), max(e - s for s, e, _, _ in plan), video_path)

        sound = decode_audio(audio_vae, audio)
        blobs = nodes_save._workflow_blobs(prompt, extra_pnginfo)
        # a looping cut: the joint holds the loop take's tail rows (the
        # opening again, as context for the refine); the FILE must not,
        # so the render is cropped to the wrap the layout recorded
        keep = (layout or {}).get("keep_frames")
        lo, hi = (int(keep[0]), int(keep[1])) if keep else (0, frames)
        blocks = decode_windows(vae, video, window_steps)
        if keep:
            from .nodes_joint import crop_blocks
            blocks = crop_blocks(blocks, lo, hi)
            sr = int(sound["sample_rate"])
            wf = sound["waveform"]
            sound = {"waveform": wf[..., round(lo / fr.FPS * sr):
                                     round(hi / fr.FPS * sr)],
                     "sample_rate": sr}
            _LOG.info("obvpm.h3 render: looping sequence, keeping frames %d..%d "
                      "of %d", lo, hi, frames)
        nodes_save.encode_mp4_stream(
            video_path, blocks, hi - lo,
            height, width, sound, crf, metadata=nodes_save._workflow_tags(blobs))

        self_id = mctx.hash_file(video_path)
        relation, parent_id, join = mctx.summarize_pins([])
        user_meta = {"render": "joint"}
        if layout:
            user_meta["source_clips"] = list(layout.get("clips") or [])
            user_meta["source_lines"] = list(layout.get("lines") or [])
        if keep:
            user_meta["loop"] = True
            user_meta["keep_frames"] = [lo, hi]
        meta = {
            "format": mctx.FORMAT,
            "self_id": self_id,
            "parent_id": parent_id,
            "relation": relation,
            "parent_join_frame": str(join),
            "width": str(width),
            "height": str(height),
            "fps": str(fr.FPS),
            # the sidecar keeps the WHOLE raw latent; a looping cut's
            # crop is recorded as the head and tail it took off, so
            # raw and delivered coordinates keep their usual relation
            "raw_frames": str(frames),
            "pinned_head_frames": str(lo),
            "pinned_tail_frames": str(frames - hi),
            "delivered_frames": str(hi - lo),
            "parent_grade": mctx.lineage_grade([]),
            "pins": mctx.serialize_pins([]),
            "user_meta": json.dumps(user_meta, ensure_ascii=False),
            "overlap_frames": "0",
            "overlap_tail_frames": "0",
        }
        sidecar = mctx.write_sidecar(mctx.sidecar_path(video_path), video, audio,
                                     meta, blobs=blobs)
        nodes_save._save_conditioning(video_path, conditioning, self_id, True)
        _LOG.info("obvpm.h3 render: wrote %s (+ sidecar %s): %d frames, %.1f MB%s",
                  video_path, os.path.basename(sidecar), frames,
                  os.path.getsize(video_path) / 1e6,
                  "" if conditioning else " -- no conditioning wired, so this "
                  "cut cannot be refined again")
        return (video_path,)
