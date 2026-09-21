"""H3MCtxFromFrames: the pack's only pixels -> latents crossing.

A pin is normally a direct latent slice out of a verified sidecar: exact,
hash-checked, no VAE anywhere. Footage with no sidecar -- an import, an
old take, a clip whose sidecar was deleted -- has no latents at all, so
the only way to pin it is to encode its pixels. That is what this node
does, and it is deliberately a separate, explicit graph edge rather than
a fallback hidden inside Apply: crossing into pixel space costs fidelity
and loses identity, and both facts should be visible in the graph.

What the bundle it emits is NOT:

  * not verified.  There is no file to hash -- frames arrive on a wire and
    may have been graded, upscaled or cropped on the way. `self_id` stays
    empty, so a take saved from a pin on this bundle records itself as a
    ROOT rather than claiming a lineage it cannot prove.
  * not exact.  A VAE round trip is lossy, so continuity across the join
    is pixel-grade, not latent-grade. Apply says so in the log whenever a
    pin's origin is "encoded".

Everything downstream is unchanged: the bundle carries the same meta
fields the sidecar header does, so PinSpec, Apply, Trim and Save need no
knowledge that this path exists.
"""

import logging
import math
import os

import torch

import comfy.utils

from . import frames as fr
from . import wiretypes as wt
from . import mctx
from .avpack import unpack_av

_LOG = logging.getLogger("obvpm.h3")

# What the audio VAE runs at when it does not say. The same default core's
# _encode_ref_audio uses.
_DEFAULT_AUDIO_SR = 32000

# Seconds of audio decoded before a window starts, then discarded: enough
# for a codec's own priming to settle (see _decode_audio_range).
_AUDIO_PREROLL = 0.5


def snap_clip_frames_down(n):
    """Largest legal H3 clip length (17k+5) that is <= n, or None below 5.

    frames.snap_frames_up() is the sampler-side direction (a requested
    length is grown onto the grid). Encoding goes the other way: the
    footage is what it is and the surplus has to be dropped, because
    rounding UP would invent frames that do not exist.
    """
    n = int(n)
    if n < fr.FRAME_BASE:
        return None
    return (((n - fr.FRAME_BASE) // fr.FRAMES_PER_GROUP)
            * fr.FRAMES_PER_GROUP + fr.FRAME_BASE)


def cfr_source_index(i, src_fps, dst_fps=fr.FPS):
    """The source frame an output frame samples. Floor, never round.

    Rounding lets an output frame reach FORWARD past its own timestamp,
    which at a join means the pinned window ends on a frame the source
    has not reached yet. Upstream resamples the same way, so footage
    resampled here and footage resampled there agree.

    Indices are ABSOLUTE, so asking for output frame 100 gives the same
    source frame whether or not frames 0..99 were ever decoded -- that is
    what lets the file decoder below fetch an interior window directly
    instead of resampling from the start.
    """
    return int(math.floor(i * float(src_fps) / float(dst_fps)))


def cfr_out_count(n_src, src_fps, dst_fps=fr.FPS):
    """How many output frames a source of n_src frames yields."""
    n_src = int(n_src)
    if n_src < 1:
        return 0
    if abs(float(src_fps) - float(dst_fps)) < 1e-9:
        return n_src
    return int(math.floor((n_src - 1) * float(dst_fps) / float(src_fps))) + 1


def cfr_indices(n_src, src_fps, dst_fps=fr.FPS):
    """Source frame index per output frame, floor-indexed constant rate."""
    n_src = int(n_src)
    n_out = cfr_out_count(n_src, src_fps, dst_fps)
    return [min(n_src - 1, cfr_source_index(i, src_fps, dst_fps))
            for i in range(n_out)]


def _resize(images, width, height, crop):
    """[N,H,W,C] -> [N,height,width,3]. The same path core's H3 nodes use."""
    samples = images[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos",
                                         crop)
    return samples.movedim(1, -1)


def _audio_window(audio, sr_out, start_frame, frames):
    """The waveform under a frame range, at sr_out, exactly the right length.

    Returns ([1, C, L], padded_samples) with L pinned to the audio grid's
    own idea of how long `frames` frames are, so the encode below lands on
    exactly fr.audio_total(frames) steps instead of wherever the VAE's own
    rounding falls. Silence when there is no soundtrack: real ENCODED
    silence, never a zero latent -- zero is not silence in latent space,
    it is noise, and it would arrive at the join.
    """
    want = int(round(frames / float(fr.FPS) * sr_out))
    if audio is None:
        return torch.zeros((1, 1, want)), 0
    wave = audio["waveform"]
    if wave.dim() == 2:
        wave = wave.unsqueeze(0)
    wave = wave[:1]
    sr_in = int(audio["sample_rate"])
    if sr_in != sr_out:
        import torchaudio
        wave = torchaudio.functional.resample(wave, sr_in, sr_out)
    lo = int(round(start_frame / float(fr.FPS) * sr_out))
    cut = wave[..., lo:lo + want]
    short = want - int(cut.shape[-1])
    if short > 0:
        cut = torch.cat(
            [cut, torch.zeros(cut.shape[:-1] + (short,), dtype=cut.dtype)],
            dim=-1)
    return cut, short


# ---------------------------------------------------------------------------
# reading a window straight out of a file
#
# The node above takes footage off a wire; a timeline pin names a FILE. A
# file is the better source of the two: it carries its own frame rate, so
# nothing has to be declared, and only the pinned window need ever be
# decoded. A 22-frame pin from a ten-minute import should not cost ten
# minutes of RAM.
# ---------------------------------------------------------------------------

# Probe results, keyed on (path, size, mtime). Disposable memoization,
# never authority: a container with no frame count in its header costs a
# full decode to measure, and the timeline asks for the same clip once
# per join it takes part in.
_PROBES = {}
_PROBES_CAP = 512


def probe_clip(path):
    """Header facts about a clip, without decoding it if we can help it.

    `frames` is in H3's 24 fps timebase -- what a pin window is measured
    in -- while `src_frames` is what the file actually holds.
    """
    import av
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _PROBES:
        return dict(_PROBES[key])
    with av.open(path) as c:
        if not c.streams.video:
            raise ValueError(
                "obvpm.h3 pixel route: %s has no video stream." % path)
        v = c.streams.video[0]
        rate = v.average_rate or v.base_rate
        fps = float(rate) if rate else float(fr.FPS)
        n = int(v.frames or 0)
        info = {
            "fps": fps,
            "width": int(v.codec_context.width),
            "height": int(v.codec_context.height),
            "has_audio": bool(c.streams.audio),
        }
        if n <= 0:   # some containers do not carry a frame count
            n = sum(1 for _ in c.decode(video=0))
    info["src_frames"] = n
    info["frames"] = cfr_out_count(n, fps)
    if key is not None:
        if len(_PROBES) >= _PROBES_CAP:
            _PROBES.clear()
        _PROBES[key] = dict(info)
    return info


def _decode_video_frames(path, wanted, fps):
    """{source frame index: RGB ndarray} for the indices in `wanted`.

    Seeks to the first wanted frame rather than decoding from the start,
    and indexes by PTS rather than by counting, because a seek lands on
    the keyframe BEFORE the target and the frames between are decoded but
    not wanted. If that pass comes back short -- a container with no
    usable timestamps -- it falls back to a sequential pass that still
    keeps only the wanted frames, so the slow path costs time but never
    memory.
    """
    import av

    want = set(wanted)
    lo, hi = min(want), max(want)
    out = {}

    def harvest(container, stream, use_pts):
        idx = 0
        for frame in container.decode(stream):
            if use_pts:
                if frame.pts is None:
                    return False
                k = int(round(float(frame.pts * stream.time_base) * fps))
            else:
                k = idx
                idx += 1
            if k in want and k not in out:
                out[k] = frame.reformat(format="rgb24").to_ndarray()
                if len(out) == len(want):
                    return True
            if k > hi:   # decoding past the last wanted frame is waste
                break
        return len(out) == len(want)

    with av.open(path) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        try:
            c.seek(int(lo / fps / v.time_base), stream=v, backward=True)
        except Exception:
            _LOG.debug("obvpm.h3: seek failed on %s; decoding forward", path)
        if harvest(c, v, True):
            return out

    out.clear()
    _LOG.info("obvpm.h3: %s has no usable frame timestamps; decoding it "
              "sequentially", path)
    with av.open(path) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        harvest(c, v, False)
    return out


def _decode_audio_range(path, t0, t1):
    """The soundtrack between two times, as an AUDIO dict. None if silent.

    Absolute sample positions come from the stream's own timestamps, so a
    seek that overshoots backwards trims correctly instead of shifting the
    window by however far it overshot.
    """
    import av
    import numpy as np

    with av.open(path) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[0]
        sr = int(a.rate)
        try:
            layout = a.layout.name
        except Exception:
            layout = "stereo"
        # fltp normalises to_ndarray() to (channels, samples) float32
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout=layout, rate=sr)
        lo = int(round(t0 * sr))
        hi = int(round(t1 * sr))
        # Seek EARLY and throw the lead-in away: cheap insurance for
        # codecs that genuinely need warm-up (MP3's bit reservoir), since
        # the slice below is by absolute sample position and extra
        # lead-in simply falls outside the window.
        #
        # It does NOT make a seeked decode bit-identical to a sequential
        # one, and no preroll length does -- measured on the test import,
        # a seeked AAC decode differs from a decode-from-zero by ~0.0016
        # mean / 0.03 peak on a -1..1 waveform (about -56 dB), decaying
        # but never vanishing, and the same with 0.5 s or 2 s of lead-in.
        # What IS exact is the ALIGNMENT: best-offset search lands on the
        # expected sample, correlation 0.99994, RMS equal to five places.
        # That is the property the join depends on; the residual is far
        # under what the audio VAE discards a moment later, on a route
        # whose whole premise is pixel-grade rather than exact.
        try:
            c.seek(int(max(0.0, t0 - _AUDIO_PREROLL) / a.time_base),
                   stream=a, backward=True)
        except Exception:
            pass
        chunks = []          # (absolute start sample, ndarray)
        for frame in c.decode(a):
            base = (int(round(float(frame.pts * a.time_base) * sr))
                    if frame.pts is not None else None)
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]
                if base is None:
                    base = (chunks[-1][0] + chunks[-1][1].shape[-1]
                            if chunks else lo)
                chunks.append((base, arr))
                base += arr.shape[-1]
            if chunks and chunks[-1][0] >= hi:
                break

    if not chunks:
        return None
    ch = max(x.shape[0] for _, x in chunks)
    buf = np.zeros((ch, max(0, hi - lo)), dtype=np.float32)
    for base, arr in chunks:
        s0, s1 = max(lo, base), min(hi, base + arr.shape[-1])
        if s1 <= s0:
            continue
        buf[:arr.shape[0], s0 - lo:s1 - lo] = arr[:, s0 - base:s1 - base]
    return {"waveform": torch.from_numpy(buf).unsqueeze(0),
            "sample_rate": sr}


def decode_window(path, start, count, info=None):
    """`count` frames of `path` from frame `start`, in 24 fps terms.

    Returns (images [N,H,W,3] float32 0..1, audio dict or None) already on
    H3's timebase, so the caller hands them to encode_bundle with fps=24
    and nothing left to declare. ABSOLUTE cfr indexing is what makes that
    sound: output frame `start + i` maps to the same source frame whether
    or not the frames before it were ever read.
    """
    import numpy as np

    info = info or probe_clip(path)
    fps, n_src = info["fps"], info["src_frames"]
    if count < 1:
        raise ValueError("obvpm.h3 pixel route: empty window requested.")
    src = [min(n_src - 1, cfr_source_index(start + i, fps))
           for i in range(count)]
    frames = _decode_video_frames(path, sorted(set(src)), fps)
    missing = [k for k in src if k not in frames]
    if missing:
        raise ValueError(
            "obvpm.h3 pixel route: %s did not yield frames %s of the "
            "requested window (frames %d..%d at %g fps). The file may be "
            "truncated, or its timestamps unusable."
            % (path, missing[:4], start, start + count - 1, fps))
    stack = np.stack([frames[k] for k in src]).astype(np.float32) / 255.0
    audio = _decode_audio_range(path, start / float(fr.FPS),
                                (start + count) / float(fr.FPS))
    return torch.from_numpy(stack), audio


def encode_bundle(images, vae, audio_vae, width, height, fps=fr.FPS,
                  keep="tail", max_frames=0, fit="cover", audio=None):
    """Footage -> an encoded MCTX bundle. THE pixels->latents crossing.

    A module function, not just the node's body, because two callers need
    it: the node (footage on a wire) and H3MCtxApplyPins (a clip named by
    a timeline pin, decoded from disk). One implementation, so a pin built
    either way is the same bundle -- the alternative is two encoders that
    agree until they quietly do not.

    `width`/`height` come from the TARGET latent in both cases; frames are
    resized to them, so a pin can never mismatch the clip it is pinned
    into.
    """
    n_src = int(images.shape[0])
    if n_src < 1:
        raise ValueError(
            "obvpm.h3 pixel route: no frames to encode.")
    idx = cfr_indices(n_src, fps)
    if len(idx) != n_src:
        _LOG.info("obvpm.h3: resampled %d frames at %.4f fps to %d "
                  "frames at %d fps (floor-indexed CFR)",
                  n_src, float(fps), len(idx), fr.FPS)

    avail = len(idx)
    want = min(avail, int(max_frames)) if max_frames else avail
    n = snap_clip_frames_down(want)
    if n is None:
        raise ValueError(
            "obvpm.h3 pixel route: %d frames at %d fps is not enough to "
            "encode -- H3 clip lengths are 5, 22, 39, ... (17k+5), so 5 "
            "is the minimum. Decode more of the source."
            % (want, fr.FPS))
    if n != avail:
        _LOG.info("obvpm.h3: keeping the %s %d of %d frames (%d dropped "
                  "to reach the 17k+5 grid%s)", keep, n, avail, avail - n,
                  ", max_frames %d" % max_frames if max_frames else "")

    start = avail - n if keep == "tail" else 0
    take = idx[start:start + n]
    frames = images[torch.as_tensor(take, dtype=torch.long)]
    frames = _resize(frames, width, height,
                     "center" if fit == "cover" else "disabled")

    video_lat = vae.encode(frames)
    if video_lat.ndim == 4:
        video_lat = video_lat.unsqueeze(0)
    got_t = int(video_lat.shape[2])
    want_t = fr.frames_to_latents(n)
    if got_t != want_t:
        raise RuntimeError(
            "obvpm.h3 pixel route: the video VAE returned %d latent steps "
            "for %d frames, expected %d. That is not an H3 video VAE, "
            "or the grid moved upstream; refusing rather than pinning a "
            "misaligned window." % (got_t, n, want_t))

    sr = int(getattr(audio_vae, "audio_sample_rate", _DEFAULT_AUDIO_SR))
    wave, padded = _audio_window(audio, sr, start, n)
    if padded:
        _LOG.warning("obvpm.h3: the soundtrack is %.3fs short of the "
                     "encoded window; padded with silence",
                     padded / float(sr))
    audio_lat = audio_vae.encode(wave.movedim(1, -1))
    if audio_lat.ndim == 3:
        audio_lat = audio_lat.unsqueeze(0)
    want_at = fr.audio_total(n)
    got_at = int(audio_lat.shape[-1])
    if got_at > want_at:
        audio_lat = audio_lat[..., :want_at].clone()
    elif got_at < want_at:
        # edge-clamp rather than zero-fill, for the reason in
        # _audio_window: a zero audio latent is noise, not silence.
        _LOG.warning("obvpm.h3: the audio VAE returned %d steps for %d "
                     "frames, expected %d; holding the last step",
                     got_at, n, want_at)
        pad = audio_lat[..., -1:].expand(
            audio_lat.shape[:-1] + (want_at - got_at,))
        audio_lat = torch.cat([audio_lat, pad], dim=-1).clone()

    meta = {
        "format": mctx.FORMAT,
        # empty on purpose: nothing here can be verified, and
        # summarize_pins reads an empty source_id as "no lineage", so a
        # take pinned on this bundle is saved as a root.
        "self_id": "",
        "parent_id": "",
        "relation": "",
        "parent_join_frame": "0",
        "width": str(width),
        "height": str(height),
        "fps": str(fr.FPS),
        "raw_frames": str(n),
        "pinned_head_frames": "0",
        "pinned_tail_frames": "0",
        "delivered_frames": str(n),
        "pins": "[]",
        "user_meta": "",
    }
    _LOG.info("obvpm.h3: encoded %d frames at %dx%d -> %d video steps + "
              "%d audio steps (pixel-grade, unverified)",
              n, width, height, want_t, want_at)
    return mctx.make_mctx("", video_lat, audio_lat, meta,
                          origin="encoded")


class H3MCtxFromFrames:
    CATEGORY = "obvpm/h3"
    FUNCTION = "build"
    RETURN_TYPES = (wt.MCTX,)
    RETURN_NAMES = ("mctx",)
    DESCRIPTION = (
        "Builds a motion-context bundle by VAE-ENCODING footage -- the "
        "route for clips with no .mctx sidecar (imports, old takes). The "
        "only pixels-to-latents crossing in the pack, and deliberately an "
        "explicit one: the result is pixel-grade rather than exact and "
        "carries no verified identity, so a take pinned on it saves as a "
        "new root, not a continuation. Clips that DO have a sidecar must "
        "keep using it -- load those with H3 MCtx Load instead."
    )
    OUTPUT_TOOLTIPS = (
        "An encoded bundle for the pins pipeline. Same shape as a loaded "
        "one; origin is 'encoded' and Apply logs that continuity is soft.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Decoded footage. Only the kept window is "
                               "encoded, so decode as little as you need."}),
                "video_vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "audio_vae": ("VAE", {
                    "tooltip": "The H3 audio VAE. Needed even for silent "
                               "footage -- a pin's audio has to be encoded "
                               "silence, not an empty latent."}),
                "latent": ("LATENT", {
                    "tooltip": "The TARGET clip's empty AV latent, exactly "
                               "as wired to Apply and the sampler. Only its "
                               "resolution is read: frames are resized to "
                               "it, so the pin cannot mismatch."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "The footage's OWN frame rate. Nothing here "
                               "can measure it -- an IMAGE wire has no "
                               "timebase -- so it is declared, and anything "
                               "other than 24 is resampled by floor-indexed "
                               "CFR. A wrong value pins at the wrong "
                               "speed."}),
                "keep": (["tail", "head"], {
                    "default": "tail",
                    "tooltip": "Which end survives when the footage does "
                               "not fit the 17k+5 clip grid (or exceeds "
                               "max_frames). tail for extending -- the pin "
                               "comes from the end; head for prepending."}),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 3600, "step": 17,
                    "tooltip": "Encode at most this many frames from the "
                               "keep end; 0 = all of them. A 22-frame pin "
                               "needs 22 frames, and the VAE cost is per "
                               "frame."}),
                "fit": (["cover", "stretch"], {
                    "default": "cover",
                    "tooltip": "How footage of a different aspect meets the "
                               "target canvas. cover crops, stretch "
                               "distorts; cover is right for a continuation "
                               "because the pinned content has to line up "
                               "geometrically with what follows it."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "The footage's soundtrack, aligned with "
                               "frame 0 of images. Absent = the pin carries "
                               "silence."}),
            },
        }

    def build(self, images, video_vae, audio_vae, latent, fps, keep, max_frames,
              fit, audio=None):
        target_video, _ = unpack_av(latent, name="latent")
        return (encode_bundle(
            images, video_vae, audio_vae,
            width=int(target_video.shape[4]) * 16,
            height=int(target_video.shape[3]) * 16,
            fps=fps, keep=keep, max_frames=max_frames, fit=fit,
            audio=audio),)
