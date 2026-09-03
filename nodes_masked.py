"""Masked continuation: preserved latent prefixes/suffixes on the target.

Core PR #15375 (merged 2026-08-18) gave MiniMax H3 per-token, per-stream
denoise masks: mask 0 preserves a latent token verbatim, mask 1 generates
it. The pins pipeline uses this as a second MODE. A guide pin hands the
model the window as native keyframe cond rows and the window is
re-rendered (then trimmed); a masked pin writes the very same slice INTO
the target latent and protects it with a mask, so exactly one rendering
of the window exists. The level-step and post-overlap flicker mechanisms
of the guide route cannot occur, and pinned audio is KEPT rather than
asked to be reproduced (measured: video pins reproduce, audio pins do
not).

This module holds the masked half of Apply: window legality, the latent
writes, and the mask assembly. Slicing itself stays in nodes_pins --
masked pins go through the SAME _prepare_pins slicer, so coordinate
rules (delivered->raw mapping, the 17-frame phase refusal, cumulative
audio boundaries) are shared by construction.

The masked window is doubly constrained -- stricter than the pin ladder:
a valid video run (17k + 5) that ALSO covers an integer number of 40 Hz
audio ticks (frames divisible by 3). The shared grid is 39, 90, 141,
192, ... (51k + 39). A 22-frame window is a legal GUIDE pin but not a
legal masked one: its audio would end between ticks.
"""

import logging

import torch

from . import frames as fr

_LOG = logging.getLogger("obvpm.h3")

# Valid masked window lengths: valid video run AND integer audio ticks.
SHARED_AV_GRID_BASE = 39   # 51k + 39
SHARED_AV_GRID_STEP = 51


def core_masks_available():
    """Whether this ComfyUI carries PR #15375's per-stream H3 masks."""
    try:
        import comfy.ldm.minimax.model as mm
        import comfy.model_base as mb
    except Exception:
        return False
    return (hasattr(mm, "mask_row_values")
            and "scale_latent_inpaint" in vars(mb.MiniMaxH3))


def masked_window_ok(n):
    return n >= SHARED_AV_GRID_BASE and (
        (n - SHARED_AV_GRID_BASE) % SHARED_AV_GRID_STEP == 0)


def shared_av_snap_down(n):
    """Largest legal masked window <= n, or None below 39."""
    if n < SHARED_AV_GRID_BASE:
        return None
    return (SHARED_AV_GRID_BASE
            + (n - SHARED_AV_GRID_BASE) // SHARED_AV_GRID_STEP
            * SHARED_AV_GRID_STEP)


def shared_av_snap_nearest(n):
    """Closest length on the shared AV grid (never below 39).

    For choosing a RUN length rather than a window: the rungs are 2.125 s
    apart, so snapping down alone would silently lose up to that much of
    a requested duration. Ties go up -- asking for the midpoint should
    not shorten the clip.
    """
    lo = shared_av_snap_down(n)
    if lo is None:
        return SHARED_AV_GRID_BASE
    hi = lo + SHARED_AV_GRID_STEP
    return lo if (n - lo) * 2 < SHARED_AV_GRID_STEP else hi


def _feather_ramp(feather, device, dtype):
    """Half-cosine 0 -> 1 release over `feather` ticks (ends at 1.0)."""
    i = torch.arange(1, feather + 1, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(torch.pi * i / float(feather))


# The mask is a STRENGTH, not a flag.
#
# Core reads a fractional value as a per-row TIMESTEP: mask m puts that
# token row at sigma = m * sigma_stream and labels it 1 - m * sigma
# (comfy/ldm/minimax/model.py -- mask_row_values, then the seg_t table
# and rows_to_mod_index). So a partly-held row is not a lie told to the
# model; it is told exactly how much signal that row carries. m = 0 pins
# the row at the cond timestep -- the same footing a keyframe cond row
# stands on -- and m = 1 is ordinary generation. The space between
# masked and guided is therefore continuous, and this profile is how we
# sample it.
#
# Three numbers, all measured from the JOIN: the window edge that faces
# the delivered content. For an after-pin (a prepend arriving) that is
# the window's FIRST frame -- the take plays up to it and the pinned
# clip continues from it. For a before-pin (an extend departing) it is
# the LAST. The window is trimmed away either way, so these frames are
# not output: they are the approach.
#
#   ramp_frames  how far the ramp reaches back from the join (0 = none)
#   edge         the mask value AT the join
#   deep         the mask value from `ramp_frames` inward
#
# (0, 0.0, 0.0) is a hard hold -- today's behaviour, bit for bit. The
# shapes worth testing:
#
#   runway   ramp 17, edge 0.6, deep 0.0   a landing strip the model may
#            bend, converging onto exact destination content
#   partial  ramp 0,  deep 0.3             the whole window held softly,
#            participating in the flow rather than standing as a wall
#   anchor   ramp 17, edge 0.0, deep 0.5   exact at the join, negotiable
#            behind it
#
# CAUTION -- the reason this is a widget and not a default: for an
# after-pin the join edge is precisely where the take has to be exact,
# and the audio version of this idea (an 8-tick feather at a_lo) was
# measured on 2026-08-25 making a prepend WORSE, which is why after-pin
# audio is hard-bounded below. The counter-argument is that a hard
# interior wall is out of distribution for a model trained to arrive via
# keyframe rows, and that a row it can partly draw is a row it can
# reconcile with. Both are hypotheses; the widget is how they settle.
#
# Video only. The audio mask keeps its hard join for the measured
# reason -- softening it moved a boundary the delivered sound owns.
def mask_profile(covered, steps, place, ramp_frames=0, edge=0.0, deep=0.0,
                 device=None, dtype=torch.float32):
    """Per-latent-step mask values for one pinned window, join-relative.

    Returns a `steps` long tensor. Every value is clamped to [0, 1];
    the all-zero result (the default) is exactly the hard hold this
    replaced.
    """
    steps = int(steps)
    covered = int(covered)
    edge = min(1.0, max(0.0, float(edge)))
    deep = min(1.0, max(0.0, float(deep)))
    ramp_frames = max(0, int(ramp_frames))
    if steps <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    if ramp_frames <= 0 or edge == deep:
        return torch.full((steps,), deep, device=device, dtype=dtype)

    # Where each step sits inside the window, in frames. The window
    # starts at cycle phase 0 (_prepare_pins refused anything else), so
    # the offsets are the same table the slicer used.
    starts = fr.step_offsets(steps)
    ends = list(starts[1:]) + [covered]
    if place == "before":
        # join at the window's END: distance measured from each step's
        # last frame back to it
        dist = [covered - e for e in ends]
    else:
        # join at the window's START: distance from it to each step's
        # first frame
        dist = [s for s in starts]
    span = float(ramp_frames)
    vals = [edge + (deep - edge) * min(1.0, max(0.0, d / span))
            for d in dist]
    return torch.tensor(vals, device=device, dtype=dtype)


def handover_frames(covered, place, ramp_frames=0, edge=0.0, deep=0.0):
    """Where responsibility for the pinned window changes hands.

    A ramped window is only PART scaffolding. The rows the ramp
    loosened were re-drawn by this take, and they were re-drawn to meet
    what it generated -- so if we trim them away and let the source
    supply its own originals instead, we have thrown away exactly the
    frames that were doing the matching, and put back the ones the
    model was told it need not match. The runway would lead to the
    wrong runway.

    So the take DELIVERS its ramped frames and the source enters later,
    past them. The cut then lands where the mask is 0 on BOTH sides:
    the take's frames there decode from the source's own latents, which
    is the same property that makes a masked extend seamless.

    Returns the offset INSIDE the window, in frames, at which the
    handover happens:

      after-pin (a prepend arriving)  the take delivers [0, n) of the
                                     window; the target enters at n
      before-pin (an extend departing) the parent plays to n; the take
                                     delivers [n, covered)

    A hard hold gives 0 and `covered` respectively -- exactly the old
    behaviour, which is why nothing changes for an unshaped pin. The
    boundary can only fall on a LATENT STEP, since that is the finest
    thing a mask can address: within a 39 frame window those are frames
    0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34, 35.

    THE HANDOVER DOES NOT TOUCH AUDIO ALIGNMENT -- it CANCELS. This has
    now been got wrong twice, in both directions, so the algebra is
    written out. Continuity at the join needs the take's sound at the
    cut to be the sound the destination resumes with:

        delivered/24 - a_lo/40  ==  handover/24
        delivered = frame_count - covered + handover
      =>  (frame_count - covered)/24 == a_lo/40
      =>  audio_total(frame_count) == frame_count * 5/3

    `handover` drops out of both sides: moving the cut shifts the take's
    end and the destination's entry by the SAME amount. What is left is
    a condition on the RUN LENGTH alone -- the pinned audio has to sit
    where the pinned video sits, and it does not when the run's tick
    count had to round up. Off the AV grid the take's own sound is ~8 ms
    late against its own picture, before any cut is made, and no choice
    of handover can move it back (verified numerically 2026-08-27 over
    runs 100/124/141/192/243 x handovers 0/13/14/15: the offset is a
    function of the run alone). That is what apply_masked_pins warns
    about, and the fix is the run length.
    """
    covered = int(covered)
    # A FULLY OPEN window (deep 1, no ramp) is not a hold at all: the
    # window was written as this take's starting point and steering
    # rows, and the take re-drew every frame of it in its own hand. Those
    # frames ARE the transition from the source's rendering to this
    # take's, so the take delivers all of them and the source hands over
    # at the window's START (before) / picks up past its END (after).
    # Trimming them instead would put the source's originals back and
    # cut straight to the take's free frames -- the switch the open
    # window exists to avoid.
    if ramp_frames <= 0 and float(deep) >= 1.0:
        return 0 if place == "before" else covered
    steps = fr.steps_for_frames(covered)
    if steps is None:
        return 0 if place == "after" else covered
    prof = mask_profile(covered, steps, place, ramp_frames, edge,
                        deep).tolist()
    held = [k for k, v in enumerate(prof) if v == 0.0]
    if not held:
        # Nowhere is fully held -- a flat partial hold, say. There is no
        # frame the two sides agree on exactly, so hand over at the join
        # as before and let the seam repairs earn their keep.
        return 0 if place == "after" else covered
    starts = fr.step_offsets(steps)
    if place == "after":
        return int(starts[held[0]])
    ends = list(starts[1:]) + [covered]
    return int(ends[held[-1]])


def holds_exactly(covered, place, ramp_frames=0, edge=0.0, deep=0.0):
    """Whether any of the window is preserved VERBATIM (mask 0).

    The question a join asks is not "was there a ramp" but "is there a
    frame both sides agree on exactly", because that is where the cut
    goes. A hard hold and a ramped hold both have one; a flat partial
    hold does not, and its join is a genuine re-render meeting an
    original.
    """
    steps = fr.steps_for_frames(int(covered))
    if not steps:
        # No window length to reason about (an older recipe, a pin kind
        # that carries none): the hard hold is the only thing it can
        # have meant, and that is exact.
        return True
    prof = mask_profile(covered, steps, place, ramp_frames, edge,
                        deep).tolist()
    return any(v == 0.0 for v in prof)


def _pin_mask_shape(pin):
    """(ramp_frames, edge, deep) for a prepared pin, from its spec."""
    spec = pin.get("spec") or {}
    return (int(spec.get("mask_ramp_frames", 0) or 0),
            float(spec.get("mask_ramp_edge", 0.0) or 0.0),
            float(spec.get("mask_hold", 0.0) or 0.0))


def apply_masked_pins(latent, masked_pins, audio_feather_ticks=8,
                      freeze_audio=False, audio_denoise=0.0):
    """Write masked pins into the target latent; return the new LATENT.

    `masked_pins` are PREPARED pins from _prepare_pins (mode "masked",
    at most one per side -- Apply enforced both). before-pins occupy the
    head of both streams, after-pins the tail; the noise mask protects
    exactly what was written, with a half-cosine release on the audio
    edge that faces the generated region.

    `freeze_audio` holds the WHOLE audio mask at 0, so the run re-renders
    picture only and the soundtrack comes out exactly as it went in. That
    is what an upscale/refine pass wants: the sampler denoises the nested
    AV pair together, so a refine re-renders sound that was already
    finished and approved. It works with NO pins at all -- the first clip
    of a timeline has no junction to pin but still must not have its
    audio regenerated.

    `audio_denoise` (with freeze_audio) holds the mask at that value
    instead of 0. At 0.5 the audio is half re-sampled alongside the
    video, which is what lets the model re-derive LIP SYNC during a
    refine -- a fully frozen track gives a near-noise video nothing to
    move the mouth for, because generation learned AV jointly, not
    audio-driven. The sound that comes out is then a resample and must
    be DISCARDED at save time in favour of the source audio (decode the
    source audio latent for the save node's `audio`). Junction-pinned
    audio windows stay at 0 regardless: the handover is a measured,
    approved join.
    """
    import comfy.nested_tensor

    if latent.get("noise_mask") is not None:
        raise ValueError(
            "H3MCtxApplyPins: the target latent already carries a noise "
            "mask; composing masks is not supported. Feed the fresh "
            "empty AV latent.")
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
        raise ValueError(
            "H3MCtxApplyPins: masked pins need a MiniMax H3 AV target "
            "latent (nested video+audio).")
    video, audio = samples.tensors[0], samples.tensors[1]
    frame_count = fr.pixel_frames(int(video.shape[2]))
    total_ticks = int(audio.shape[-1])

    out_video = video.clone()
    out_audio = audio.clone()
    video_mask = torch.ones((1, 1) + tuple(video.shape[2:]),
                            device=video.device, dtype=torch.float32)
    audio_mask = torch.ones((1, 1) + tuple(audio.shape[2:]),
                            device=audio.device, dtype=torch.float32)

    if freeze_audio:
        fill = min(1.0, max(0.0, float(audio_denoise)))
        if fill > 0.0:
            _LOG.info("obvpm.h3: audio held at denoise %.2f for the whole "
                      "run (%d tick(s)) -- it re-samples alongside the "
                      "picture for lip sync; save the SOURCE audio, not "
                      "this resample", fill, total_ticks)
        else:
            _LOG.info("obvpm.h3: audio frozen for the whole run (%d "
                      "tick(s)) -- picture is re-rendered, sound is kept "
                      "as it is", total_ticks)
    pinned_audio_spans = []

    for pin in masked_pins:
        place = pin.get("place")
        covered = int(pin["covered"])
        steps = int(pin["steps"])
        if not masked_window_ok(covered):
            raise ValueError(
                "H3MCtxApplyPins: a masked pin's window must sit on the "
                "shared AV grid 39/90/141/192/... (a video run that also "
                "ends on a 40 Hz audio tick); %d frames does not. Use 39, "
                "or switch the pin's mode to guide, which takes any "
                "ladder window." % covered)
        span = covered * fr.FRAME_RESCALE
        if span != int(span):
            raise RuntimeError(
                "H3MCtxApplyPins: %d masked frames -> non-integer audio "
                "span; grid drift upstream." % covered)
        span = int(span)

        ramp_frames, edge, deep = _pin_mask_shape(pin)
        prof = mask_profile(covered, steps, place, ramp_frames, edge, deep,
                            device=video_mask.device,
                            dtype=video_mask.dtype).view(1, 1, steps, 1, 1)
        if ramp_frames or edge or deep:
            _LOG.info(
                "obvpm.h3: masked %s-pin holds SOFTLY -- mask %.2f at the "
                "join, %.2f from %d frames in (0 = verbatim, 1 = free); "
                "per-step %s", place, prof.flatten()[0 if place == "after"
                                                     else -1].item(),
                deep, ramp_frames,
                ", ".join("%.2f" % v for v in prof.flatten().tolist()))

        vp = pin["video"].to(out_video.device, out_video.dtype)
        if place == "before":
            out_video[:, :, :steps] = vp
            video_mask[:, :, :steps] = prof
            a_lo, a_hi = 0, span
        else:  # "after" -- Apply refused place "at_frame" long before here
            out_video[:, :, -steps:] = vp
            video_mask[:, :, -steps:] = prof
            a_hi = fr.audio_total(frame_count)
            a_lo = a_hi - span

        rt = int(pin.get("audio_steps", 0))
        pin_audio = pin.get("audio") if rt > 0 else None
        if pin_audio is None:
            # Preserved picture, generated sound: legal (a silent parent,
            # a sidecar without audio). The audio mask stays 1 there.
            _LOG.info("obvpm.h3: masked %s-pin has no pinned audio; the "
                      "model generates sound under the preserved picture",
                      place)
        else:
            if rt != span:
                raise ValueError(
                    "H3MCtxApplyPins: a masked pin preserves its whole "
                    "window on both streams -- %d frames need %d audio "
                    "steps, the pin carries %d (is audio_window set? "
                    "masked pins take the full window)." % (covered, span, rt))
            if a_hi > total_ticks:
                raise ValueError(
                    "H3MCtxApplyPins: masked audio %d..%d exceeds the "
                    "target's %d ticks." % (a_lo, a_hi, total_ticks))
            out_audio[..., a_lo:a_hi] = pin_audio.to(
                out_audio.device, out_audio.dtype)
            audio_mask[..., a_lo:a_hi] = 0.0
            pinned_audio_spans.append((a_lo, a_hi))
            # The feather belongs on the boundary the clip's own audio
            # RELEASES from, never on the one it has to ARRIVE at.
            #
            # A before-pin owns the run's opening: its only internal
            # boundary is at a_hi, where pinned hands over to generated,
            # and the delivered audio starts there. Ramping it lets the
            # model leave the pinned audio smoothly, which is exactly
            # what the join needs.
            #
            # An after-pin is NOT the index mirror of that, though it
            # looks like one. Its internal boundary is at a_lo -- and
            # that is the join itself, the point the take's delivered
            # audio runs up to and the target's own audio continues
            # from. Softening the mask there tells the model it need not
            # match the target until `feather` ticks LATER, all of which
            # are trimmed away: the audio we keep is the audio that was
            # never held to anything. Measured on a masked prepend as
            # 200 ms of unconstrained approach at the one place it has
            # to be exact. So an after-pin gets a hard boundary.
            feather = max(0, min(int(audio_feather_ticks), span))
            if feather > 0 and place == "before":
                ramp = _feather_ramp(feather, audio_mask.device,
                                     audio_mask.dtype)
                audio_mask[..., a_hi - feather:a_hi] = ramp.expand(
                    audio_mask.shape[:-1] + (feather,)).clone()
            elif feather > 0:
                _LOG.debug("obvpm.h3: after-pin audio is hard-bounded at "
                           "the join; the %d-tick feather is not applied "
                           "there", feather)

        if place == "after" and pin_audio is not None:
            # The join is where this run's DELIVERED audio stops and the
            # pinned clip's own audio takes over. Video is cut at
            # (frame_count - covered)/24 s; the pinned audio begins at
            # its tick boundary, (ticks)/40 s. Those agree only when the
            # delivered length is a whole number of 40 Hz ticks -- which,
            # among the legal 17k+5 run lengths, is exactly the shared AV
            # grid. Off it, the join skips or repeats up to 8.3 ms of
            # content: a click, however good the picture is.
            #
            # The condition is on the RUN, not on the cut: the pinned
            # audio must sit where the pinned video sits, and it cannot
            # when the run's tick count had to round up. The handover
            # cancels out of this entirely (see handover_frames), so it
            # is not part of the test -- but it IS part of the delivered
            # length the message quotes, which is what the user trims to.
            delivered = frame_count - covered + handover_frames(
                covered, "after", ramp_frames, edge, deep)
            if fr.audio_total(frame_count) != frame_count * fr.FRAME_RESCALE:
                _LOG.warning(
                    "obvpm.h3: masked prepend into a %d frame run: that "
                    "length needs %d audio ticks but covers %.2f, so the "
                    "pinned sound sits %.1f ms later than the pinned "
                    "picture and the join skips that much -- an audible "
                    "click over a clean picture, whatever the %d "
                    "delivered frames are cut at. Use a run length on the "
                    "shared AV grid (%s); the window alone is not enough.",
                    frame_count, fr.audio_total(frame_count),
                    frame_count * fr.FRAME_RESCALE,
                    (fr.audio_total(frame_count)
                     - frame_count * fr.FRAME_RESCALE)
                    / float(fr.AUDIO_LATENT_FPS) * 1000.0,
                    delivered,
                    "/".join(str(SHARED_AV_GRID_BASE + SHARED_AV_GRID_STEP * k)
                             for k in range(5)))

        if float(pin.get("overhang", 0.0)) and place == "before":
            _LOG.warning(
                "obvpm.h3: masked prefix audio carries the source's "
                "clip-end overhang (~%.1f ms); for a bit-exact prefix use "
                "a parent whose delivered length is on the 39/90/141 grid",
                float(pin["overhang"]) * 1000.0 / fr.AUDIO_LATENT_FPS)
        _LOG.info(
            "obvpm.h3: masked %s-pin preserved %d frames = %d video steps "
            "/ %d audio ticks of a %d frame target (%s-grade)",
            place, covered, steps, span if pin_audio is not None else 0,
            frame_count,
            "pixel" if pin.get("origin") == "encoded" else "latent")

    if freeze_audio:
        # LAST, so nothing above can re-open it: a before-pin's feather
        # ramps its own edge back toward 1, and "frozen" has to mean the
        # whole stream, not the whole stream except the handover. With
        # audio_denoise the whole stream is held at that value instead --
        # except the junction-pinned windows, which stay exactly 0: the
        # handover was measured and approved, and re-sampling it would
        # re-open the join.
        audio_mask[...] = min(1.0, max(0.0, float(audio_denoise)))
        for a_lo, a_hi in pinned_audio_spans:
            audio_mask[..., a_lo:a_hi] = 0.0

    out = dict(latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((out_video, out_audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, audio_mask))
    return out
