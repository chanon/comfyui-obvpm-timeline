"""Hold extension: give the refine MORE of the parent to hold.

A masked extend holds 39 frames (12 latent steps) of its parent -- the
window the take was generated with. That is plenty for GENERATION, where
the free frames start as pure noise and the held window is the only
appearance in the clip, so the model copies it. A REFINE is different:
its free frames start as the upscaled source at 25% plus noise, which
already carries a complete appearance of its own, over 75 steps. The
12 held steps are a minority and the free rows resolve toward the prior
they contain and toward each other. Measured 2026-09-04 (refine08..14):
holding exactly, matching the noise level, sharing the noise field,
feeding the window as a reference, and making the upscaled latents
continuous all left the seam where it was.

This node lengthens the argument the parent gets to make. The target
latent is EXTENDED on the held side by `extend_frames` (video steps and
audio ticks on the shared AV grid), Loop Start widens the junction pin
by the same amount, and Apply Pins then writes 39 + extend frames of
the parent's REFINED latent into the head (or tail) and holds them. The
sampler sees a longer clip whose held part is a much larger share; the
save's trim removes the whole held window as usual, so the delivered
clip is unchanged in span. Nothing here touches the free frames -- it
changes who they are outnumbered by.

The rows this node fills are placeholders (edge-replicated): Apply Pins
overwrites every one of them with the parent's rows, video and audio
both (the pin's audio window follows its video window). The extended
AUDIO latent is also handed out on its own, for the decode that feeds
the save node: the refine keeps the source audio, decoded from the
source latent, and the trim takes the pinned head off it -- so that
latent has to be as long as the sampled one or the trim lands on the
wrong frames.

Grid: the held window must sit on the shared AV grid (39/90/141/192/...,
see nodes_masked.masked_window_ok), so the extension is a multiple of
51 frames = 15 video steps = 85 audio ticks.
"""

import logging

import torch

from . import avpack
from . import frames as fr
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

# 51 frames: the smallest step that keeps a 39-frame window on the
# shared AV grid (39 + 51k) and is itself whole in video steps (15)
# and audio ticks (85)
GRID_FRAMES = 51


def check_extend(frames):
    """The extension in frames, or raise: a whole number of grid units."""
    n = int(frames or 0)
    if n < 0 or n % GRID_FRAMES:
        raise ValueError(
            "hold_extend must be a multiple of %d frames (0, 51, 102, "
            "153, ...): the held window has to stay on the shared AV "
            "grid 39/90/141/192. Got %d." % (GRID_FRAMES, n))
    return n


def sides_for(pin_specs, frames):
    """{"before": frames or 0, "after": frames or 0} from the junction pins.

    Extension goes on the side the pin holds: a before-pin (extend) at
    the head, an after-pin (prepend, or the departing half of a bridge)
    at the tail. Pixel-grade pins are never junction pins here.
    """
    out = {"before": 0, "after": 0}
    n = check_extend(frames)
    if n <= 0:
        return out
    for spec in (pin_specs or []):
        if spec.get("source_kind") == "clip" and spec.get("place") in out:
            out[spec["place"]] = n
    return out


def extend_rows(video, audio, before_frames, after_frames):
    """Edge-replicated rows added to head/tail of both streams."""
    def rep(t, n, dim, head):
        if n <= 0:
            return t
        idx = 0 if head else t.shape[dim] - 1
        edge = t.narrow(dim, idx, 1)
        pad = edge.expand(*[n if d == dim else -1 for d in range(t.ndim)])
        return torch.cat([pad, t] if head else [t, pad], dim=dim)

    v_before = fr.steps_for_frames(before_frames) if before_frames else 0
    v_after = fr.steps_for_frames(after_frames) if after_frames else 0
    if v_before is None or v_after is None:
        raise ValueError("hold extension is not a whole number of latent steps")
    a_before = fr.audio_total(before_frames) if before_frames else 0
    a_after = fr.audio_total(after_frames) if after_frames else 0
    video = rep(rep(video, v_before, 2, True), v_after, 2, False)
    if audio is not None:
        audio = rep(rep(audio, a_before, audio.ndim - 1, True),
                    a_after, audio.ndim - 1, False)
    return video, audio, (v_before, v_after, a_before, a_after)


class H3RefineHoldExtend:
    """Lengthens the target latent so a wider junction pin can be held."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "extend"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "audio_latent")
    DESCRIPTION = (
        "Extends the refine's target AV latent on the held side(s) by "
        "extend_frames, so the junction pin Loop Start widened by the "
        "same amount can write 39 + extend frames of the parent's REFINED "
        "latent into it and hold them -- the parent then outnumbers the "
        "free frames' own prior instead of being a 12-step minority. The "
        "save's trim removes the whole held window, so the delivered clip "
        "is unchanged. Put it between the AV concat and H3 MCtx Apply Pins."
    )
    OUTPUT_TOOLTIPS = (
        "The extended AV latent, for H3 MCtx Apply Pins' latent.",
        "The extended AUDIO latent alone -- wire to the VAEDecodeAudio "
        "that feeds the save node, in place of the source audio latent, "
        "so the trim takes the pinned head off a track of the same "
        "length as the picture.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The upscaled AV target latent (after "
                               "LTXVConcatAVLatent), before Apply Pins."}),
                "pin_specs": (wt.PINSPECS, {
                    "tooltip": "From Loop Start: says which side(s) hold a "
                               "neighbour, so the extension lands there."}),
                "extend_frames": ("INT", {
                    "default": 0, "min": 0, "max": 306, "step": GRID_FRAMES,
                    "tooltip": "Frames of extra held parent, a multiple of "
                               "51 (102 makes the 39-frame window 141 = 42 "
                               "steps). Wire from Loop Start's hold_extend "
                               "so the pin, the latent and the profile hash "
                               "agree. 0 = pass through."}),
            },
        }

    def extend(self, latent, pin_specs, extend_frames):
        video, audio = avpack.unpack_av(latent, name="latent")
        sides = sides_for(pin_specs, extend_frames)
        if not any(sides.values()):
            if int(extend_frames or 0) > 0:
                _LOG.info("obvpm.h3 hold extend: no junction pin on this "
                          "clip; nothing to extend")
            return (latent, {"samples": audio})
        if audio is None:
            raise ValueError(
                "H3 Refine Hold Extend: the target latent has no audio "
                "stream; a refine target is the AV pair from "
                "LTXVConcatAVLatent.")
        v2, a2, (vb, va, ab, aa) = extend_rows(
            video, audio, sides["before"], sides["after"])
        _LOG.info("obvpm.h3 hold extend: +%d/+%d frames head/tail -> video "
                  "%d -> %d steps, audio %d -> %d ticks (the pin overwrites "
                  "them with the parent's refined rows)",
                  sides["before"], sides["after"], video.shape[2], v2.shape[2],
                  audio.shape[-1], a2.shape[-1])
        out = dict(latent)
        out.pop("noise_mask", None)
        out["samples"] = avpack.pack_av(v2, a2, name="latent")["samples"]
        return (out, {"samples": a2})
