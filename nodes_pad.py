"""Upscaler padding: give the latent upscaler its neighbours' context.

The refine upscales each clip's raw latent on its own, and the upscaler
(a stack of temporal convolutions, zero-padded) sees a clip EDGE where
the timeline has none: measured 2026-09-03, the same source rows come
out 11-15% different upscaled at the end of the parent than at the start
of the child, against ~4% in the interior. The two clips' refines then
start from different upscaled content on the very rows the junction is
supposed to hold, and each re-derives its own detail there.

The fix is to hide the edge from the upscaler. Before upscaling, the
clip's raw latent is extended on each held side with the neighbour's
own raw (SOURCE) latent rows from beyond the shared window -- the frames
that really do come before and after this clip on the timeline -- so the
convolutions' receptive field is filled with the true continuation
instead of zeros. After upscaling, exactly those rows are cropped off
again. Nothing sampled changes length; only what the upscaler saw.

A plain cut is not padded: nothing continues across it, and the
neighbour's frames would be a lie there. The sides that are padded are
exactly the held joins Loop Start pins (`_joins`), so the same
definition of "continuous" is used in both places.

Geometry, in latent steps. A pin on the HOLDER H holds the window
N[s : s+w] of neighbour N -- at H's head (place "before") or its tail
(place "after"). So H step t is N step s + t - off, with off = 0 for a
head window and off = T_H - w for a tail window. The padding this clip
needs on a side is the neighbour's rows on the steps just outside its
own [0, T), mapped through that relation and clipped to what the
neighbour has.
"""

import logging

import torch

from . import frames as fr
from . import mctx
from . import nodes_load
from . import upscale
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

DEFAULT_STEPS = 24  # ~ the receptive field the upscaler's stack reaches


def to_steps(frame_count):
    """Frames -> latent steps on the phase-0 grid, snapping off-grid up."""
    n = int(frame_count)
    steps = fr.steps_for_frames(n)
    if steps is None:
        steps = 0
        while fr.frame_at_latent(steps) < n:
            steps += 1
    return steps


def holder_relation(pin, holder_steps):
    """(s, off): holder step t == neighbour step s + t - off."""
    s = to_steps(pin.get("source_start") or 0)
    w = to_steps(pin.get("source_frames") or 0)
    if pin.get("place") == "after":
        return s, int(holder_steps) - w
    return s, 0


def side_ranges(index, deps, steps_of, pad_steps):
    """Which neighbour rows pad each side of clip `index`.

    Returns {"before": (j, lo, hi) | None, "after": (j, lo, hi) | None}
    with [lo, hi) in NEIGHBOUR j's step coordinates, unclipped (a caller
    clips to the neighbour's real length). `steps_of(i)` is clip i's
    latent length; `deps` is Loop Start's join map {holder: [(nb, pin)]}.
    """
    K = int(pad_steps)
    T = int(steps_of(index))
    out = {"before": None, "after": None}
    if K <= 0:
        return out

    def held(holder, neighbour):
        for j, pin in deps.get(holder, []):
            if j == neighbour:
                return pin
        return None

    # the join with the clip that plays BEFORE this one
    prev = index - 1
    if prev >= 0:
        pin = held(index, prev)
        if pin is not None and pin.get("place") == "before":
            s, off = holder_relation(pin, T)          # we hold its tail
            out["before"] = (prev, s - K - off, s - off)
        else:
            pin = held(prev, index)
            if pin is not None and pin.get("place") == "after":
                s, off = holder_relation(pin, steps_of(prev))  # it holds our head
                # our step u == its step u - s + off
                out["before"] = (prev, off - s - K, off - s)
    # the join with the clip that plays AFTER this one
    nxt = index + 1
    pin = held(index, nxt)
    if pin is not None and pin.get("place") == "after":
        s, off = holder_relation(pin, T)              # we hold its head
        out["after"] = (nxt, s + T - off, s + T + K - off)
    else:
        pin = held(nxt, index)
        if pin is not None and pin.get("place") == "before":
            s, off = holder_relation(pin, steps_of(nxt))     # it holds our tail
            out["after"] = (nxt, T - s + off, T - s + K + off)
    return out


def clip_range(lo, hi, length):
    """[lo, hi) intersected with [0, length)."""
    lo, hi = max(0, int(lo)), min(int(length), int(hi))
    return (lo, hi) if hi > lo else (0, 0)


def _video_samples(latent, who):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not isinstance(samples, torch.Tensor) \
            or samples.dim() != 5:
        raise ValueError(
            "%s: expected a VIDEO latent [B,C,T,H,W]. Put this node after "
            "LTXVSeparateAVLatent, on the video_latent branch that feeds "
            "the upscaler -- the audio is not upscaled and is not padded."
            % who)
    return samples


class H3UpscalePad:
    """Extends the clip's raw latent with its neighbours' rows for the upscaler."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "pad"
    RETURN_TYPES = ("LATENT", wt.PAD)
    RETURN_NAMES = ("latent", "pad")
    DESCRIPTION = (
        "Pads the clip's raw video latent on each HELD side with the "
        "neighbour clip's own raw latent rows from beyond the shared "
        "window -- the frames that really precede and follow this clip on "
        "the timeline -- so the latent upscaler sees the true continuation "
        "instead of a zero-padded edge. H3 Upscale Crop takes the rows off "
        "again after the upscaler. Plain cuts are not padded."
    )
    OUTPUT_TOOLTIPS = (
        "The padded video latent, for the upscaler's latent input.",
        "How many rows were added on each side. Wire to H3 Upscale Crop.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (wt.LOOP, {
                    "tooltip": "From H3 Upscale Loop Start: which clip this "
                               "is and who its neighbours are."}),
                "latent": ("LATENT", {
                    "tooltip": "The clip's raw VIDEO latent (after "
                               "LTXVSeparateAVLatent), as it would go into "
                               "the upscaler."}),
                "pad_steps": ("INT", {
                    "default": 0, "min": 0, "max": 128,
                    "tooltip": "Latent steps of neighbour context to add on "
                               "each held side (24 covers the upscaler's "
                               "receptive field; 0 = pass through). Wire "
                               "from Loop Start's pad_steps so the profile "
                               "hash agrees."}),
            },
        }

    def pad(self, flow, latent, pad_steps):
        samples = _video_samples(latent, "H3 Upscale Pad")
        K = int(pad_steps)
        empty = {"before": 0, "after": 0}
        if K <= 0:
            return (latent, empty)
        if not isinstance(flow, dict) or "index" not in flow:
            raise ValueError(
                "H3 Upscale Pad: the flow input must come from an H3 "
                "Upscale Loop Start.")
        clips = list(flow.get("clips") or upscale.parse_sequence(
            "\n".join(flow.get("lines") or [])))
        index = int(flow["index"])
        if not 0 <= index < len(clips):
            raise ValueError("H3 Upscale Pad: flow index %d is outside a %d "
                             "clip timeline." % (index, len(clips)))
        from .nodes_loop import H3UpscaleLoopStart
        headers = H3UpscaleLoopStart._headers(clips)
        deps = H3UpscaleLoopStart._joins(headers)
        T = int(samples.shape[2])

        def steps_of(i):
            if i == index:
                return T
            return fr.frames_to_latents(
                int(headers[i].get("raw_frames", 0) or 0))

        ranges = side_ranges(index, deps, steps_of, K)
        pieces, counts = [], {}
        for side in ("before", "after"):
            spec = ranges[side]
            rows = None
            if spec is not None:
                j, lo, hi = spec
                video, _audio, _meta = mctx.load_sidecar(mctx.sidecar_path(
                    nodes_load.resolve_clip_path(clips[j])))
                if tuple(video.shape[3:]) != tuple(samples.shape[3:]) \
                        or video.shape[1] != samples.shape[1]:
                    raise ValueError(
                        "H3 Upscale Pad: neighbour %s is %s but this clip "
                        "is %s; clips of one join must share a size."
                        % (clips[j], tuple(video.shape), tuple(samples.shape)))
                lo, hi = clip_range(lo, hi, video.shape[2])
                if hi > lo:
                    rows = video[:, :, lo:hi].to(
                        device=samples.device, dtype=samples.dtype)
            counts[side] = 0 if rows is None else int(rows.shape[2])
            pieces.append(rows)
        before, after = pieces
        parts = [p for p in (before, samples, after) if p is not None]
        padded = torch.cat(parts, dim=2) if len(parts) > 1 else samples
        _LOG.info("obvpm.h3 upscale pad: clip %d -- %d step(s) before, %d "
                  "after (%d -> %d)%s", index, counts["before"], counts["after"],
                  T, padded.shape[2],
                  "" if any(counts.values()) else " (no held join: unpadded)")
        out = dict(latent)
        out["samples"] = padded
        return (out, counts)


class H3UpscaleCrop:
    """Removes the rows H3 Upscale Pad added, after the upscaler."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "crop"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    DESCRIPTION = (
        "Takes off the neighbour rows H3 Upscale Pad added, after the "
        "upscaler, so the clip is back to its own length -- upscaled with "
        "its real continuation in view."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The upscaler's output."}),
                "pad": (wt.PAD, {
                    "tooltip": "From H3 Upscale Pad."}),
            },
        }

    def crop(self, latent, pad):
        samples = _video_samples(latent, "H3 Upscale Crop")
        before = int((pad or {}).get("before", 0) or 0)
        after = int((pad or {}).get("after", 0) or 0)
        T = int(samples.shape[2])
        if before + after >= T:
            raise ValueError(
                "H3 Upscale Crop: %d + %d padding rows leave nothing of a "
                "%d step latent; the upscaler must keep the temporal "
                "length (it did not)." % (before, after, T))
        if before == 0 and after == 0:
            return (latent,)
        out = dict(latent)
        out["samples"] = samples[:, :, before:T - after].contiguous()
        return (out,)
