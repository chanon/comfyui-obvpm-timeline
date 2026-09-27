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

SEGMENTS. A cut with hard cuts or empty space reaches this node as a
LIST of segment latents (H3 Join Latents splits it; everything between
runs once per segment), and this node takes the whole list in one call.
Each segment is written as a take of its own and the finished cut is
joined from them: the picture copied packet for packet, black for empty
space, the sound encoded once with the cuts de-clicked (join_cut). A cut
in one segment is written exactly as before.

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
    # the upscale branch runs once per SEGMENT of the cut (H3 Join Latents
    # hands out a list); this node takes the whole list in one call, so it
    # can write every segment and then join them
    INPUT_IS_LIST = True
    DESCRIPTION = (
        "Writes the jointly refined timeline as one finished MP4, with a "
        ".mctx.safetensors sidecar (the refined latent) and, when the joint "
        "conditioning is wired, a .cond -- so the rendered sequence is a take "
        "that can itself be put on a timeline and refined again. A cut with "
        "hard cuts or empty space arrives as several segments: each is "
        "written as a take of its own (name_s01, _s02, ...) and the finished "
        "cut is joined from them. The picture is decoded a few seconds at a "
        "time and streamed to the encoder, so a long timeline costs one "
        "window of memory. Wire the sampler's output to samples."
    )
    OUTPUT_TOOLTIPS = ("Path of the finished MP4 (for a cut in several "
                       "segments, the joined file; the segments' takes sit "
                       "beside it).",)

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
               window_seconds=None, conditioning=None, layout=None,
               source_audio=None, prompt=None, extra_pnginfo=None):
        # INPUT_IS_LIST: every input is a list, one entry per segment where
        # the upscale branch ran per segment, one entry where it is shared
        vae, audio_vae = _one(vae), _one(audio_vae)
        base_folder = _one(base_folder, "project1")
        filename_prefix = _one(filename_prefix, "upscale")
        crf = int(_one(crf, 19))
        window_seconds = float(_one(window_seconds, 5.0))
        prompt, extra_pnginfo = _one(prompt), _one(extra_pnginfo)
        samples = samples if isinstance(samples, list) else [samples]
        n = len(samples)
        layouts = _each(layout, n)
        conds = _each(conditioning, n)
        sources = _each(source_audio, n)

        first_video, _ = avpack.unpack_av(samples[0], name="samples")
        width = int(first_video.shape[4]) * 16
        height = int(first_video.shape[3]) * 16
        full_folder, filename, counter, _subfolder, _ = \
            folder_paths.get_save_image_path(
                nodes_save._save_prefix(base_folder, filename_prefix),
                folder_paths.get_output_directory(), width, height)
        cut_path = os.path.join(full_folder, "%s_%05d.mp4" % (filename, counter))
        gap_before = int((layouts[0] or {}).get("gap_before") or 0)
        gap_after = int((layouts[-1] or {}).get("gap_after") or 0)
        single = n == 1 and not gap_before and not gap_after

        # every segment's soundtrack, cut to what it shows, before any
        # picture: the joined cut needs the whole cut's sound, and so does
        # anything else fed the frames as they are decoded
        prepared = [self._prepare(samples[k], audio_vae, layouts[k], sources[k])
                    for k in range(n)]
        # the finished cut, part by part: empty space and segments in order
        parts = []
        for k in range(n):
            lay = layouts[k] or {}
            gap = int(lay.get("gap_before") or 0)
            if gap:
                parts.append({"gap": True, "path": None, "frames": gap,
                              "sound": None, "declick": True})
            parts.append({"path": None, "segment": k,
                          "frames": prepared[k]["hi"] - prepared[k]["lo"],
                          "sound": prepared[k]["sound"],
                          "declick": _declick_into(lay)})
        if gap_after:
            parts.append({"gap": True, "path": None, "frames": gap_after,
                          "sound": None, "declick": True})
        sound = prepared[0]["sound"] if single else join_audio(parts)
        blobs = nodes_save._workflow_blobs(prompt, extra_pnginfo)
        sink = self._open_sink({
            "base_folder": base_folder, "counter": counter,
            "cut_path": cut_path, "width": width, "height": height,
            "fps": fr.FPS,
            "frames": sum(int(p["frames"]) for p in parts), "sound": sound,
            "metadata": nodes_save._workflow_tags(blobs)})
        try:
            if single:
                self._write(samples[0], vae, audio_vae, window_seconds, crf,
                            cut_path, conds[0], layouts[0], sources[0],
                            prompt, extra_pnginfo, prepared=prepared[0],
                            sink=sink)
            else:
                # several segments (or empty space around one): each is
                # written as a take of its own, then the finished cut is
                # joined from them
                for p in parts:
                    if _is_gap(p):
                        if sink is not None:
                            for block in _black_blocks(p["frames"], height, width):
                                sink.write(block)
                        continue
                    k = p["segment"]
                    p["path"] = os.path.join(full_folder, "%s_%05d_s%02d.mp4"
                                             % (filename, counter, k + 1))
                    _LOG.info("obvpm.h3 render: segment %d of %d -> %s", k + 1,
                              n, os.path.basename(p["path"]))
                    self._write(samples[k], vae, audio_vae, window_seconds, crf,
                                p["path"], conds[k], layouts[k], sources[k],
                                prompt, extra_pnginfo,
                                extra_meta={"segment": [k + 1, n],
                                            "cut": os.path.basename(cut_path)},
                                prepared=prepared[k], sink=sink)
                join_cut(parts, cut_path, crf, width, height, audio=sound)
        except BaseException:
            if sink is not None:
                sink.abort()
            raise
        if sink is not None:
            try:
                sink.close()
            except RuntimeError as exc:
                raise RuntimeError("H3 Joint VAE Decode and Save: %s. The upscale "
                                   "itself was saved: %s" % (exc, cut_path)) from None
        return (cut_path,)

    def _open_sink(self, ctx):
        """Anything else to feed the finished cut's frames to, or None.

        ctx: base_folder, counter, cut_path, width, height, fps, frames
        (the whole cut's, empty space included), sound (the whole cut's
        AUDIO), metadata. What it returns gets write(block) for every block of the
        cut in order (float [n, H, W, 3], black for empty space), then
        close() -- which raises RuntimeError when it failed -- or abort().
        """
        return None

    @staticmethod
    def _prepare(samples, audio_vae, layout=None, source_audio=None):
        """One segment's latent pair, the frames it keeps and its soundtrack
        cut to them -- everything but the picture decode."""
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
        if layout and int(layout.get("total_steps", total_steps)) != total_steps:
            raise ValueError(
                "H3 Joint VAE Decode and Save: the latent has %d steps but the joint "
                "layout describes %d. Wire the sampler that refined THIS "
                "timeline." % (total_steps, int(layout["total_steps"])))
        sound = decode_audio(audio_vae, audio)
        # the frames the cut SHOWS: a segment is laid out with a little
        # context around them (and a looping cut with the loop take's tail
        # rows, the opening again, as context for the refine); the FILE
        # keeps only the shown frames
        keep = (layout or {}).get("keep_frames")
        lo, hi = (int(keep[0]), int(keep[1])) if keep else (0, frames)
        if keep:
            sr = int(sound["sample_rate"])
            wf = sound["waveform"]
            sound = {"waveform": wf[..., round(lo / fr.FPS * sr):
                                     round(hi / fr.FPS * sr)],
                     "sample_rate": sr}
        return {"video": video, "audio": audio, "frames": frames,
                "keep": keep, "lo": lo, "hi": hi, "sound": sound}

    def _write(self, samples, vae, audio_vae, window_seconds, crf, video_path,
               conditioning=None, layout=None, source_audio=None, prompt=None,
               extra_pnginfo=None, extra_meta=None, prepared=None, sink=None):
        """One segment's latent -> its MP4 + sidecar (+ .cond): a take.

        `sink` (see _open_sink) is fed the same kept frames as they are
        decoded. Returns {"frames": frames written, "sound": the AUDIO dict
        written}.
        """
        p = prepared or self._prepare(samples, audio_vae, layout, source_audio)
        video, audio, frames = p["video"], p["audio"], p["frames"]
        keep, lo, hi, sound = p["keep"], p["lo"], p["hi"], p["sound"]
        total_steps = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16

        window_steps = fr.steps_for_seconds(window_seconds) or CHUNK
        plan = window_plan(total_steps, window_steps)
        _LOG.info("obvpm.h3 render: %d steps (%d frames, %dx%d) in %d window(s) "
                  "of up to %d steps -> %s", total_steps, frames, width, height,
                  len(plan), max(e - s for s, e, _, _ in plan), video_path)

        blobs = nodes_save._workflow_blobs(prompt, extra_pnginfo)
        blocks = decode_windows(vae, video, window_steps)
        if keep:
            from .nodes_joint import crop_blocks
            blocks = crop_blocks(blocks, lo, hi)
            _LOG.info("obvpm.h3 render: keeping frames %d..%d of %d%s", lo, hi,
                      frames, " (the loop's wrap)" if (layout or {}).get("loop") else "")
        if sink is not None:
            blocks = _tee(blocks, sink)
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
            user_meta["keep_frames"] = [lo, hi]
        if (layout or {}).get("loop"):
            user_meta["loop"] = True
        if extra_meta:
            user_meta.update(extra_meta)
        meta = {
            "format": mctx.FORMAT,
            "self_id": self_id,
            "parent_id": parent_id,
            "relation": relation,
            "parent_join_frame": str(join),
            "width": str(width),
            "height": str(height),
            "fps": str(fr.FPS),
            # the sidecar keeps the WHOLE raw latent; the crop is recorded
            # as the head and tail it took off, so raw and delivered
            # coordinates keep their usual relation
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
                  video_path, os.path.basename(sidecar), hi - lo,
                  os.path.getsize(video_path) / 1e6,
                  "" if conditioning else " -- no conditioning wired, so this "
                  "cut cannot be refined again")
        return {"frames": hi - lo, "sound": sound}


# ---------------------------------------------------------------------------
# joining the segments into the finished cut
# ---------------------------------------------------------------------------

def _one(value, default=None):
    """A shared input under INPUT_IS_LIST: its (first) value."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _each(value, n):
    """A per-segment input under INPUT_IS_LIST, as n entries. ComfyUI's own
    rule for lists of unequal length: a short list repeats its last entry;
    an unwired input is None for every segment."""
    if value is None:
        return [None] * n
    if not isinstance(value, list):
        return [value] * n
    if not value:
        return [None] * n
    return [value[i] if i < len(value) else value[-1] for i in range(n)]


def _tee(blocks, sink):
    """Pass decoded blocks through, feeding each to `sink` on the way."""
    for block in blocks:
        sink.write(block)
        yield block


def _is_gap(part):
    """Is this part of the joined cut empty space? Marked `gap` by the
    render; a part with neither a file nor a segment is one too."""
    return bool(part.get("gap")) or (part.get("path") is None
                                     and "segment" not in part)


def _declick_into(layout):
    """The declick setting of the join in front of a segment: the seam
    settings of its first line (the clip after a join owns it, as in the
    timeline's export), else on."""
    from . import nodes_assemble as na
    lines = (layout or {}).get("lines") or []
    if not lines:
        return True
    return bool(na.split_seam_opts(str(lines[0]).strip())[1].get(
        "audio_declick", True))


def join_audio(parts):
    """One soundtrack for the joined cut: each part's sound cut or padded
    to exactly its frames, silence for empty space, and each hard cut
    tapered on both sides (the timeline export's declick) unless the join
    says otherwise. None when a part that should have sound has none."""
    import torch
    from . import crossfade as xf
    rate, channels = None, None
    for p in parts:
        if _is_gap(p):
            continue
        if not p.get("sound"):
            return None
        wave = p["sound"]["waveform"]
        wave = wave[0] if wave.dim() == 3 else wave
        if rate is None:
            rate, channels = int(p["sound"]["sample_rate"]), int(wave.shape[0])
        elif int(p["sound"]["sample_rate"]) != rate or int(wave.shape[0]) != channels:
            _LOG.warning("obvpm.h3 render: the segments' soundtracks differ in "
                         "rate or channels; the joined cut carries no audio")
            return None
    if rate is None:
        return None
    pos = lambda f: int(round(f / float(fr.FPS) * rate))
    pieces, at = [], 0
    for p in parts:
        n = pos(at + int(p["frames"])) - pos(at)
        at += int(p["frames"])
        piece = torch.zeros(channels, n, dtype=torch.float32)
        if p.get("sound"):
            wave = p["sound"]["waveform"]
            wave = (wave[0] if wave.dim() == 3 else wave).float().cpu()
            m = min(n, int(wave.shape[-1]))
            piece[:, :m] = wave[:, :m]
        pieces.append(piece)
    for i in range(len(pieces) - 1):
        if parts[i + 1].get("declick", True):
            pieces[i] = xf.taper_audio(pieces[i], rate, tail=True)
            pieces[i + 1] = xf.taper_audio(pieces[i + 1], rate, head=True)
    return {"waveform": torch.cat(pieces, -1).unsqueeze(0), "sample_rate": rate}


def _black_blocks(frames, height, width, chunk=16):
    import torch
    left = int(frames)
    while left > 0:
        n = min(chunk, left)
        yield torch.zeros(n, height, width, 3)
        left -= n


def _file_blocks(path, frames, chunk=16):
    """A written MP4's first `frames` frames as float blocks (the fallback
    join decodes and re-encodes)."""
    import av
    import torch
    got, buf = 0, []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            buf.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            got += 1
            if len(buf) == chunk or got == frames:
                yield torch.stack(buf).float() / 255.0
                buf = []
            if got == frames:
                break
    if buf:
        yield torch.stack(buf).float() / 255.0
    if got != frames:
        raise RuntimeError("H3 Joint VAE Decode and Save: %s holds %d frames, "
                           "expected %d" % (path, got, frames))


def join_cut(parts, out_path, crf, width, height, audio=None):
    """Join written segments (and empty space) into one MP4.

    The picture is COPIED packet for packet -- every segment came out of
    the same encoder with the same settings, so they splice without a
    re-encode (nodes_assemble._mux_pieces, the timeline export's own
    splicer) -- and the sound is encoded once for the whole cut, so no
    per-file AAC padding lands at a join. Empty space is black, encoded
    through the same encoder so it splices too. If the pieces will not
    splice, the cut is decoded and encoded again instead (slower, one
    more generation of H.264, same result otherwise). `audio` is the
    joined soundtrack when the caller already has it (join_audio).
    """
    import shutil
    import tempfile
    from . import nodes_assemble as na
    if audio is None:
        audio = join_audio(parts)
    total = sum(int(p["frames"]) for p in parts)
    tmp = tempfile.mkdtemp(prefix="obvpm_h3_join_")
    try:
        pieces = []
        for i, p in enumerate(parts):
            if _is_gap(p):
                bp = os.path.join(tmp, "gap_%d.mp4" % i)
                nodes_save.encode_mp4_stream(
                    bp, _black_blocks(p["frames"], height, width),
                    int(p["frames"]), height, width, None, crf)
                p = dict(p, path=bp)
            pieces.append({"path": p["path"], "from": 0, "to": int(p["frames"])})
        try:
            na._mux_pieces(pieces, out_path, audio)
            how = "copied"
        except Exception as exc:
            _LOG.warning("obvpm.h3 render: the segments do not splice (%s); "
                         "encoding the joined cut again", exc)

            def blocks():
                for piece in pieces:
                    yield from _file_blocks(piece["path"], piece["to"])
            nodes_save.encode_mp4_stream(out_path, blocks(), total, height,
                                         width, audio, crf)
            how = "re-encoded"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _LOG.info("obvpm.h3 render: joined %d segment(s)%s into %s, %d frames "
              "(picture %s)", sum(1 for p in parts if not _is_gap(p)),
              (" and %d stretch(es) of empty space"
               % sum(1 for p in parts if _is_gap(p)))
              if any(_is_gap(p) for p in parts) else "",
              out_path, total, how)
