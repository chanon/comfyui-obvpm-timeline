"""Packing and unpacking MiniMax H3 AV latents (NestedTensor pairs).

H3 AV latents carry two streams with DIFFERENT temporal axes:

    video  [B, 24, T,  h, w]   temporal dim 2
    audio  [B, 32, 2,  T40]    temporal dim 3 (dim 2 is the stereo axis)

NestedTensor.__getitem__ broadcasts indices into every contained tensor
rather than selecting one, so samples[0] strips the batch dim off both
streams; unbind() returns the pair.

Module-level this file imports nothing: the comfy import lives inside
pack_av so the unpack half stays usable from a plain `python` run, the
way frames.py is.
"""


def unpack_av(latent, name="latent", allow_video_only=False):
    """(video, audio) from an H3 AV latent dict; audio may be None.

    With allow_video_only a plain 5D video latent is accepted (what
    VAEEncode produces from real footage) and audio comes back None.
    """
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    elif allow_video_only and getattr(samples, "ndim", 0) == 5:
        parts = [samples, None]
    else:
        raise ValueError(
            "'%s' is not a MiniMax H3 AV latent (NestedTensor video+audio "
            "pair). Wire an H3 sampler output or Empty MiniMax H3 AV "
            "Latent." % name)
    if not parts:
        raise ValueError("'%s': AV latent contains no streams" % name)

    video = parts[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("'%s': expected video latent [B,C,T,H,W], got %s"
                         % (name, tuple(video.shape)))

    audio = parts[1] if len(parts) > 1 else None
    if audio is not None:
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        if audio.ndim != 4:
            raise ValueError("'%s': expected audio latent [B,C,2,T], got %s"
                             % (name, tuple(audio.shape)))
    elif not allow_video_only:
        raise ValueError(
            "'%s' has no audio stream. Wire the sampler output of an H3 AV "
            "graph, not a video-only latent." % name)
    return video, audio


def pack_av(video, audio, name="latent"):
    """(video, audio) -> an H3 AV LATENT dict. Inverse of unpack_av.

    The batch dim is added when a stream was stored without one -- the
    sidecar keeps what the sampler produced, which has it, but an mctx
    built by another route may not. Both streams are required: an H3
    sampler reads the pair, and a video-only latent silently loses the
    soundtrack rather than failing.
    """
    # local, so the unpack half of this module stays comfy-free
    import comfy.nested_tensor

    if video is None:
        raise ValueError("'%s': no video stream to pack" % name)
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("'%s': expected video latent [B,C,T,H,W], got %s"
                         % (name, tuple(video.shape)))

    if audio is None:
        raise ValueError(
            "'%s': no audio stream to pack. An H3 AV latent carries both; "
            "packing video alone would drop the soundtrack." % name)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("'%s': expected audio latent [B,C,2,T], got %s"
                         % (name, tuple(audio.shape)))

    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
