"""Chain noise: one noise field for the whole timeline.

A refine invents its fine texture largely from its noise. With a fixed
seed every clip of the same size gets the IDENTICAL noise tensor,
positionally -- so the child's first free frame after a junction carries
the noise its parent had at frame 39, not a continuation of the noise
the parent's last frame was refined under. Two samples with unrelated
noise at the join invent unrelated detail there.

This node draws noise per ABSOLUTE timeline position instead: video
latent step k of a clip whose raw latent starts at timeline frame F gets
the noise of absolute step steps(F) + k, and every audio tick likewise.
Two clips that share timeline frames (a pinned window, a bridge's
context) then share the noise on them exactly, and the free frames on
either side of a join sit in one continuous field. Generation per step
is independent (a fresh generator per absolute index), so a clip's
noise does not depend on where the clip starts being generated -- only
on where it sits.

Wire `frame_offset` from H3 Upscale Loop Start's `noise_offset`, which
is each clip's raw-latent start on the timeline.
"""

import logging
import math

import torch

from . import frames as fr

_LOG = logging.getLogger("obvpm.h3")

# stream tags mixed into the per-index seed so video and audio draws
# never coincide, and a different base seed changes everything
_VIDEO = 0x5649
_AUDIO = 0x4155
_MASK64 = (1 << 63) - 1


def index_seed(seed, stream, index):
    """A deterministic 63-bit seed for one stream at one absolute index."""
    h = (int(seed) & _MASK64) * 0x9E3779B97F4A7C15
    h ^= (int(stream) & 0xFFFF) << 40
    h ^= (int(index) + (1 << 40)) * 0xBF58476D1CE4E5B9
    return h & _MASK64


def frame_offset_to_steps(frame_offset):
    """Latent steps from timeline frame 0 to `frame_offset` (phase 0 grid).

    Off-grid offsets (a plain cut somewhere upstream) snap to the step
    that covers the frame; continuity across THAT join was never on
    offer, and the field is still one field.
    """
    f = int(frame_offset)
    sign = -1 if f < 0 else 1
    f = abs(f)
    steps = fr.steps_for_frames(f)
    if steps is None:
        steps = 0
        while fr.frame_at_latent(steps + 1) <= f:
            steps += 1
        _LOG.info("obvpm.h3 chain noise: frame offset %d is off the latent "
                  "grid; using step %d (frame %d)", sign * f, sign * steps,
                  sign * fr.frame_at_latent(steps))
    return sign * steps


def _draw(seed, stream, first_index, count, shape, dtype):
    """`count` independent draws of `shape`, stacked on a new axis 0."""
    out = torch.empty((count,) + tuple(shape), dtype=torch.float32)
    for i in range(count):
        g = torch.Generator(device="cpu")
        g.manual_seed(index_seed(seed, stream, first_index + i))
        out[i] = torch.randn(shape, generator=g, device="cpu")
    return out.to(dtype)


def video_noise(seed, step_offset, shape, dtype=torch.float32):
    """[B,C,T,H,W] noise; step t is absolute step step_offset + t."""
    b, c, t, h, w = (int(v) for v in shape)
    draws = _draw(seed, _VIDEO, step_offset, t, (b, c, h, w), dtype)  # [T,B,C,H,W]
    return draws.permute(1, 2, 0, 3, 4).contiguous()


def audio_noise(seed, tick_offset, shape, dtype=torch.float32):
    """[B,C,2,N] noise; tick n is absolute tick tick_offset + n."""
    b, c, two, n = (int(v) for v in shape)
    draws = _draw(seed, _AUDIO, tick_offset, n, (b, c, two), dtype)  # [N,B,C,2]
    return draws.permute(1, 2, 3, 0).contiguous()


class ChainNoise:
    """The NOISE object SamplerCustomAdvanced calls `generate_noise` on."""

    def __init__(self, seed, frame_offset):
        self.seed = int(seed)
        self.frame_offset = int(frame_offset)

    def generate_noise(self, input_latent):
        samples = input_latent["samples"]
        step_offset = frame_offset_to_steps(self.frame_offset)
        tick_offset = int(round(self.frame_offset * fr.FRAME_RESCALE))
        if getattr(samples, "is_nested", False):
            import comfy.nested_tensor
            tensors = list(samples.unbind())
            out = [video_noise(self.seed, step_offset, tensors[0].shape, tensors[0].dtype)]
            if len(tensors) > 1:
                out.append(audio_noise(self.seed, tick_offset, tensors[1].shape,
                                       tensors[1].dtype))
            for extra in tensors[2:]:
                g = torch.Generator(device="cpu")
                g.manual_seed(index_seed(self.seed, 0x5858, 0))
                out.append(torch.randn(extra.shape, generator=g, device="cpu").to(extra.dtype))
            _LOG.info("obvpm.h3 chain noise: seed %d, clip at timeline frame %d "
                      "= video step %d, audio tick %d", self.seed,
                      self.frame_offset, step_offset, tick_offset)
            return comfy.nested_tensor.NestedTensor(out)
        if samples.ndim == 5:
            return video_noise(self.seed, step_offset, samples.shape, samples.dtype)
        g = torch.Generator(device="cpu")
        g.manual_seed(index_seed(self.seed, 0x5858, self.frame_offset))
        return torch.randn(samples.shape, generator=g, device="cpu").to(samples.dtype)


class H3ChainNoise:
    CATEGORY = "obvpm/h3"
    FUNCTION = "make"
    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    DESCRIPTION = (
        "Noise drawn per absolute timeline position, so every clip of a "
        "chain samples inside ONE continuous noise field: a pinned window "
        "gets the same noise in the parent and the child, and the free "
        "frames on both sides of a join continue the same field. A fixed "
        "seed alone gives every same-sized clip the identical noise, "
        "positionally, which puts unrelated noise on the two sides of every "
        "join. Wire frame_offset from Loop Start's noise_offset."
    )
    OUTPUT_TOOLTIPS = ("Wire to the sampler's noise input.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "The field's seed. Same seed = same field for "
                               "every clip of the chain."}),
                "frame_offset": ("INT", {
                    "default": 0, "min": -(1 << 31), "max": (1 << 31) - 1,
                    "tooltip": "Where this clip's RAW latent starts on the "
                               "timeline, in frames (may be negative for a "
                               "first clip whose pinned head precedes the "
                               "timeline). Wire from Loop Start's "
                               "noise_offset."}),
            },
        }

    def make(self, noise_seed, frame_offset=0):
        return (ChainNoise(noise_seed, frame_offset),)
