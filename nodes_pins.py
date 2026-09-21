"""The pins pipeline: decide -> slice+execute.

  H3MCtxPinSpec     spec builder: pure data, no tensor work
  H3MCtxApplyPins   slices (sole frame<->latent math site, via
                    _prepare_pins) and attaches pinned runs as native
                    `minimax_keyframes` entries; outputs the resolved
                    pins wire
  H3TrimPinned      removes pinned scaffolding from the decoded output

Phase 1 scope: a single `before` pin sourced from a clip's tail -- the
extend case. The wire formats (PINSPECS, PINS) already carry the general
shape (placement, multiple pins, image sources) so later phases add
capability without changing these interfaces.
"""

import logging

import node_helpers

from . import frames as fr
from . import mctx
from . import nodes_masked
from . import wiretypes as wt
from .avpack import unpack_av

_LOG = logging.getLogger("obvpm.h3")

# Marks a `minimax_keyframes` entry this node appended for a LINEAGE pin,
# as opposed to a CONTENT keyframe some other node anchored (an image at
# frame 60, a stock first/last frame, an AddGuide chain).
#
# The distinction is not cosmetic. Conditioning is per-SHOT and travels
# with the clip; a lineage pin is per-POSITION and is rebuilt by whatever
# walks the timeline, at whatever resolution that pass runs. Persisting a
# lineage keyframe would hand a later pass a second, stale anchor for a
# window it is about to pin itself -- at the wrong resolution, since these
# latents are the generation's size.
#
# Core reads only `latent` / `audio_latent` / `resolved_frame_index` from
# each entry (model_base.py), so an extra key rides along untouched.
# Video and audio are separate ENTRIES, so one tag per entry covers both
# streams with no special case.
PIN_ORIGIN = "pin"


def _keyframe_origin(pin):
    """PIN_ORIGIN for a lineage pin, None for a content one.

    DERIVED from the spec, never asserted by this node. Everything that
    reaches here today continues from another clip, but the same wire is
    the obvious road for a CONTENT anchor -- an image the timeline holds
    at frame 60, a sound cue -- and such a pin's keyframe has to survive
    into the `.cond`, because it is part of what the shot IS.

    The test is the one mctx.lineage_grade already uses for the header:
    a lineage edge is a known source kind plus a source id. A content
    pin will arrive with neither and go untagged, which the save side
    reads as content without having to know it exists.
    """
    # A RESOLVED pin (what apply() holds) names its source as `kind` +
    # `source_id` at the top level and keeps the recipe's `source_kind`
    # inside `spec`; a bare spec has only the latter. Reading just one
    # spelling tagged nothing for months: every guided and both-mode
    # keyframe reached the .cond as "content", which a same-resolution
    # refine tolerates and an upscaled one cannot lay out (a 34x60
    # keyframe on a 68x120 target grid).
    spec = pin.get("spec") or {}
    kind = pin.get("source_kind") or spec.get("source_kind") or pin.get("kind")
    source_id = pin.get("source_id") or spec.get("source_id")
    if kind in mctx.LINEAGE_KINDS and source_id:
        return PIN_ORIGIN
    return None


def _ensure_native_anchors():
    """Refuse unless live core natively places pins where we need them.

    Core merged arbitrary-position, multi-frame keyframe anchoring
    (PR #15439): PackedLayout places every keyframe at
    `cursor + FRAME_RESCALE * resolved_frame_index`, with the cursor past
    the reference spans, and extra_conds concatenates keyframe and
    reference latents. That is the whole mechanism this pack used to
    carry as runtime patches; pins now ride plain `minimax_keyframes`.

    Checked here, on first apply, because the failure modes are silent:
    an OLDER core hard-rejects interior anchors only at sampling time
    (or, worse, a foreign pack's patch places them by different rules).
    The new signature dropped `frame_count`, so its absence plus the
    `keyframes` parameter identifies a native core; a wrapped or
    replaced constructor identifies a foreign patcher (H3Studio,
    MMH3Tools, the Motion-Context lineage) whose coordinate rules we
    must not build on.
    """
    import inspect
    import comfy.ldm.minimax.model as _mm
    cls = getattr(_mm, "PackedLayout", None)
    init = getattr(cls, "__init__", None)
    cls_mod = getattr(cls, "__module__", "")
    if cls_mod and not cls_mod.startswith("comfy."):
        raise RuntimeError(
            "obvpm.h3: another pack has REPLACED H3's PackedLayout class "
            "(it now comes from %r -- ComfyUI-H3Studio does this). The two "
            "continuation mechanisms cannot coexist in one session; disable "
            "one pack and restart." % cls_mod)
    if hasattr(init, "__wrapped__") or getattr(init, "__module__", cls_mod) != cls_mod:
        raise RuntimeError(
            "obvpm.h3: another pack has patched H3's PackedLayout "
            "constructor (it now comes from %r). This pack pins through "
            "core's own keyframe anchoring and cannot run under a foreign "
            "layout patch; disable that pack and restart."
            % getattr(init, "__module__", "?"))
    try:
        params = tuple(inspect.signature(init).parameters)
    except (TypeError, ValueError):
        params = ()
    if "keyframes" not in params or "frame_count" in params:
        raise RuntimeError(
            "obvpm.h3: this ComfyUI predates native arbitrary-position "
            "keyframe anchors for MiniMax H3 (PR #15439, merged 2026-08-13). "
            "Update ComfyUI to use the mctx pin nodes.")


def pins_trim_totals(pins):
    """(head, tail) frames of pinned scaffolding, from PREPARED pins.

    Authoritative by construction: `covered` is what Prepare actually
    sliced (snap_down may have shrunk a requested window), so trim and
    save must read these, never the original spec numbers.
    """
    head = tail = 0
    for p in (pins or []):
        covered = int(p.get("covered", 0))
        spec = p.get("spec") or {}
        shape = (int(spec.get("mask_ramp_frames", 0) or 0),
                 float(spec.get("mask_ramp_edge", 0.0) or 0.0),
                 float(spec.get("mask_hold", 0.0) or 0.0))
        # A softly-held window is trimmed only as far as it is HELD --
        # the ramped frames are this take's own work, matched to what it
        # generated, so they ship and the source enters past them
        # (nodes_masked.handover_frames).
        if p.get("place") == "before":
            head += nodes_masked.handover_frames(covered, "before", *shape)
        elif p.get("place") == "after":
            tail += covered - nodes_masked.handover_frames(
                covered, "after", *shape)
    return head, tail


class H3MCtxPinSpec:
    CATEGORY = "obvpm/h3"
    FUNCTION = "build"
    RETURN_TYPES = (wt.PINSPECS,)
    RETURN_NAMES = ("pin_specs",)
    DESCRIPTION = (
        "Describes a pin as pure data: which part of a source clip to "
        "carry into the next generation, and where it sits in the target. "
        "No tensor work happens here -- Apply slices later. Chain "
        "several to pin from several sources."
    )
    OUTPUT_TOOLTIPS = (
        "The upstream stack plus this pin's spec. Feed H3MCtxApplyPins; "
        "Trim/Save ride the resolved pins wire Apply produces.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mctx": (wt.MCTX, {
                    "tooltip": "The source clip's latents+lineage bundle "
                               "from H3LoadVideoWithMCtx."}),
                "window": (list(fr.WINDOW_CHOICES), {
                    "default": "39",
                    "tooltip": "Frames of the source to pin. Only these "
                               "lengths are whole latent steps. 5 is barely "
                               "fluid, 22 nearly seamless; longer windows pin "
                               "more motion but cost more of the new clip. "
                               "1 is reserved for the future keyframe path."}),
                "take_from": (["head", "tail", "at_frame"], {
                    "default": "tail",
                    "tooltip": "Which part of the source supplies the "
                               "window: tail = the end (extend), head = the "
                               "start (for future prepends), at_frame = the "
                               "window ENDING at take_from_frame (a "
                               "timeline cut). Latent-grade cuts must land "
                               "on the 17-frame grid; misaligned cuts are "
                               "refused with the nearest valid frames."}),
                "take_from_frame": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Only used when take_from is at_frame: the "
                               "delivered frame the window ends at (your "
                               "cut point)."}),
                "place": (["before", "after", "at_frame"], {
                    "default": "before",
                    "tooltip": "Where the pinned run sits in the target: "
                               "before = context leading in (extend; "
                               "re-rendered at the head, trimmed). after = "
                               "context leading out (prepend; generation "
                               "runs toward it, re-rendered at the tail, "
                               "trimmed -- typically with take_from=head). "
                               "at_frame (kept interior content) is NOT "
                               "YET IMPLEMENTED."}),
                "place_at_frame": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Only used when place is at_frame (not yet "
                               "implemented)."}),
                "audio_window": ("INT", {
                    "default": 0, "min": 0, "max": 240,
                    "tooltip": "Frames of tail audio to pin, end-aligned with "
                               "the video window. 0 follows the video "
                               "window."}),
                "mode": (["masked", "guide", "both"], {
                    "default": "masked",
                    "tooltip": "guide: the window rides as native keyframe "
                               "cond rows and the model RE-RENDERS it "
                               "(trimmed after; any ladder window). "
                               "masked: the window's latents are written "
                               "INTO the target and preserved with a "
                               "denoise mask -- one rendering, no level "
                               "step, audio kept verbatim. Needs ComfyUI "
                               ">= 2026-08-18 and a window on the shared "
                               "AV grid (39/90/141/...). both: the same "
                               "window as BOTH -- the keyframe rows give "
                               "the model something to steer toward, the "
                               "mask makes the arrival exact. They agree "
                               "by construction (one slice, one source), "
                               "so they reinforce rather than fight; the "
                               "cost is the extra cond tokens."}),
                "mask_ramp_frames": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "Masked/both only. Softens the hold over "
                               "this many frames measured back from the "
                               "JOIN -- the window edge facing the "
                               "delivered content (an after-pin's first "
                               "frame, a before-pin's last). 0 = the "
                               "whole window is held at mask_hold. The "
                               "mask is a strength, not a flag: core "
                               "runs a row at sigma = mask * sigma, so "
                               "0 preserves verbatim, 1 generates "
                               "freely, and between them the model is "
                               "told how much of the destination is "
                               "already there. The softened frames are "
                               "DELIVERED rather than trimmed (the "
                               "source enters past them), so this is "
                               "the run's remaining runway to arrive "
                               "in. Prefer a value one frame PAST a "
                               "latent step offset -- 6/10/14/19/23 -- "
                               "since a ramp landing exactly on an "
                               "offset (5/9/13/17/22) zeroes that step "
                               "and loses it from the runway."}),
                "mask_ramp_edge": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "The mask value AT the join, falling to "
                               "mask_hold across mask_ramp_frames. Raise "
                               "it to give the model room to bend into "
                               "the destination (a landing strip); leave "
                               "it 0 to keep the join exact. Ignored "
                               "when mask_ramp_frames is 0."}),
                "mask_hold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "The mask value through the rest of the "
                               "window. 0 = verbatim (the default, and "
                               "what makes a masked extend seamless). "
                               "Raising it with mask_ramp_frames at 0 "
                               "holds the WHOLE window softly, so the "
                               "pinned rows join the flow instead of "
                               "standing as a fixed wall."}),
            },
            "optional": {
                "pin_specs": (wt.PINSPECS, {
                    "tooltip": "Upstream pin stack to append to."}),
            },
        }

    def build(self, mctx, window, take_from, take_from_frame, place,
              place_at_frame, audio_window, mode="masked",
              mask_ramp_frames=0, mask_ramp_edge=0.0, mask_hold=0.0,
              pin_specs=None):
        if mctx is None:
            raise ValueError(
                "H3MCtxPinSpec: mctx is None -- the loaded clip has no "
                "verified sidecar, so there are no latents to pin. Use the "
                "pixel route (H3 MCtx From Frames) or re-save the "
                "take with H3SaveVideoWithMCtx.")
        meta = mctx["meta"]
        delivered = int(meta.get("delivered_frames", 0))
        w = int(window)
        if take_from == "tail":
            start = delivered - w
        elif take_from == "head":
            start = 0
        else:
            start = int(take_from_frame) - w
        spec = {
            "source": mctx,
            "source_id": mctx.get("self_id", ""),
            "source_kind": "clip",
            "take_from": take_from,
            "take_from_frame": int(take_from_frame),
            "requested_window": w,
            # resolved best-effort range; Apply re-derives under its snap
            # policy and is authoritative
            "source_start": max(0, start),
            "source_frames": min(w, delivered) if delivered else w,
            "place": place,
            "place_at_frame": int(place_at_frame),
            "audio_window": int(audio_window),
            "mode": mode,
            # Only meaningful for a mode that masks; carried regardless
            # so a spec stays a faithful record of what was asked for.
            "mask_ramp_frames": int(mask_ramp_frames),
            "mask_ramp_edge": float(mask_ramp_edge),
            "mask_hold": float(mask_hold),
        }
        return (list(pin_specs or []) + [spec],)


def _prepare_pins(pin_specs, snap_window_down_to_available, ctx=None):
    """Materialize specs into sliced pins. THE sole slicing site.

    Lives as a module function (not a node) since 2026-08-12: with Save
    and Trim consuming the resolved pins wire, a separate Prepare node
    added a mandatory hop with no unique contribution -- slicing is cheap
    and Apply outputs the pins itself. The single-implementation-site
    principle holds here regardless of node shape.

    `ctx` carries what a PIXEL source needs and a latent one does not:
    the VAEs and the target resolution. Absent = latent sources only.
    """
    if not pin_specs:
        raise ValueError("H3MCtxApplyPins: the pin_specs stack is empty.")
    pins = []
    for i, spec in enumerate(pin_specs):
        kind = spec.get("source_kind")
        if kind == "clip":
            pins.append(_prepare_clip_pin(
                i, spec, snap_window_down_to_available))
        elif kind == "clip_pixels":
            pins.append(_prepare_pixel_pin(
                i, spec, snap_window_down_to_available, ctx))
        else:
            raise ValueError(
                "H3MCtxApplyPins: spec %d has source_kind %r; known kinds "
                "are 'clip' (latents from a sidecar) and 'clip_pixels' "
                "(a file, VAE-encoded)." % (i, kind))
    return pins


def _prepare_pixel_pin(i, spec, snap_down, ctx):
    """A pin from a clip with no usable sidecar: decode, encode, slice.

    The window is chosen in PIXEL space and encoded as a clip of exactly
    that length, which is why this route has no 17-frame cut-grid rule.
    The grid exists because a latent slice must start at cycle phase 0;
    here the encode DEFINES phase 0 at the window's first frame, so any
    cut frame is legal. That is the one thing the pixel route does better
    than the latent one, and it is what makes an arbitrary timeline cut
    continuable at all.

    Everything after the encode goes through _prepare_clip_pin, so the
    slicing rules, the audio boundary arithmetic and the resolved-spec
    bookkeeping have exactly one implementation.
    """
    from . import nodes_encode as ne
    from . import nodes_load

    clip = spec.get("source_path") or ""
    if not clip:
        raise ValueError(
            "H3MCtxApplyPins: spec %d is a pixel source with no "
            "source_path." % i)
    if not ctx or ctx.get("vae") is None or ctx.get("audio_vae") is None:
        raise ValueError(
            "H3MCtxApplyPins: %s has no usable mctx sidecar, so pinning it "
            "means VAE-encoding its pixels -- connect the video_vae "
            "and audio_vae inputs. (Continuity from an encoded source is "
            "pixel-grade, not exact.)" % clip)

    path = nodes_load.resolve_clip_path(clip)
    info = ne.probe_clip(path)
    available = int(info["frames"])
    take_from = spec.get("take_from", "tail")
    cut = int(spec.get("take_from_frame", 0) or 0)
    if take_from == "at_frame":
        if cut < 1 or cut > available:
            raise ValueError(
                "H3MCtxApplyPins: spec %d's cut frame %d is outside %s "
                "(1..%d frames at 24 fps)." % (i, cut, clip, available))
        available = cut
    elif take_from not in ("tail", "head"):
        raise ValueError(
            "H3MCtxApplyPins: spec %d has unknown take_from %r."
            % (i, take_from))

    n = int(spec.get("requested_window", 22))
    if n > available:
        snapped = fr.snap_window(available)
        if not snap_down or snapped is None or snapped == 1:
            raise ValueError(
                "H3MCtxApplyPins: spec %d wants a %d frame window but %s "
                "only offers %d frames (%s). Enable "
                "snap_window_down_to_available to use %s instead."
                % (i, n, clip, available, take_from,
                   snapped if snapped and snapped > 1 else "nothing"))
        _LOG.warning("obvpm.h3: spec %d window %d -> %d (only %d frames "
                     "available in %s)", i, n, snapped, available, clip)
        n = snapped
    if n == 1:
        raise ValueError(
            "H3MCtxApplyPins: the 1-frame window is not sliceable from a "
            "latent. Use 5 or more; 1-frame pins arrive with the keyframe "
            "path.")

    if take_from == "head":
        start = 0
    else:                       # tail, and at_frame (which ENDS at `cut`)
        start = available - n
    if start < 0:
        raise ValueError(
            "H3MCtxApplyPins: spec %d's %d frame window does not fit "
            "before frame %d of %s." % (i, n, available, clip))

    images, audio = ne.decode_window(path, start, n, info=info)
    a_window = int(spec.get("audio_window", 0) or 0)
    if a_window > n:
        _LOG.warning("obvpm.h3: audio_window %d exceeds the %d frame "
                     "window encoded from %s; the pixel route pins audio "
                     "only over the window it decodes", a_window, n, clip)
    bundle = ne.encode_bundle(
        images, ctx["vae"], ctx["audio_vae"],
        width=ctx["width"], height=ctx["height"],
        fps=fr.FPS,                 # decode_window already resampled
        keep="head", max_frames=0, fit=ctx.get("fit", "cover"),
        audio=audio)
    _LOG.info("obvpm.h3: pixel pin from %s: frames %d..%d of %d (%s), "
              "encoded at %dx%d -- pixel-grade, unverified",
              clip, start, start + n - 1, int(info["frames"]), take_from,
              ctx["width"], ctx["height"])

    # the encoded bundle IS the window, so the slice is its whole self
    inner = dict(spec, source=bundle, source_kind="clip",
                 take_from="tail", take_from_frame=0,
                 requested_window=n)
    pin = _prepare_clip_pin(i, inner, snap_down)
    # ...but the RECIPE must name the real source and the real range in
    # the original clip's frames, not 0..n of a bundle that exists only
    # inside this call.
    #
    # The hash is the SOURCE FILE's, and it is honest here in a way it
    # would not be on H3MCtxFromFrames: that node takes frames off a
    # WIRE, which may have graded or cropped them since the load, so a
    # hash there would describe pixels that no longer exist. This route
    # opened the file itself, so "encoded from frames %d..%d of the file
    # with this hash" is simply true -- and without it the take is a
    # ROOT, which leaves the Result Preview with no pair to show, the
    # seam with no link to repair, and the timeline butt-joining at the
    # parent's full length instead of at the cut.
    source_id = nodes_load._cached_hash(path)
    pin["spec"] = dict(
        {k: v for k, v in spec.items() if k != "source"},
        source_kind="clip_pixels", source_id=source_id,
        source_start=start, source_frames=n)
    pin["source_id"] = source_id
    return pin


def _prepare_clip_pin(i, spec, snap_down):
        bundle = spec.get("source")
        if bundle is None:
            raise ValueError(
                "H3MCtxApplyPins: spec %d carries no source bundle. Specs must "
                "come from H3MCtxPinSpec in this same run." % i)
        meta = bundle["meta"]
        video, audio = bundle["video_latent"], bundle["audio_latent"]

        raw_steps = int(video.shape[2])
        raw_frames = fr.pixel_frames(raw_steps)
        meta_raw = int(meta.get("raw_frames", raw_frames))
        if meta_raw != raw_frames:
            raise ValueError(
                "H3MCtxApplyPins: sidecar header says %d raw frames but the "
                "stored latent covers %d. The sidecar is inconsistent; "
                "refusing." % (meta_raw, raw_frames))
        delivered = int(meta.get("delivered_frames", raw_frames))
        pinned_head = int(meta.get("pinned_head_frames", 0))
        take_from = spec.get("take_from", "tail")
        cut_frame = int(spec.get("take_from_frame", 0))

        # window availability depends on where it's taken from
        if take_from == "at_frame":
            if cut_frame < 1 or cut_frame > delivered:
                raise ValueError(
                    "H3MCtxApplyPins: spec %d's cut frame %d is outside the "
                    "source's delivered range (1..%d)."
                    % (i, cut_frame, delivered))
            available = cut_frame
        elif take_from in ("tail", "head"):
            available = delivered
        else:
            raise ValueError(
                "H3MCtxApplyPins: spec %d has unknown take_from %r."
                % (i, take_from))

        n = int(spec.get("requested_window", 22))
        if n > available:
            snapped = fr.snap_window(available)
            if not snap_down or snapped is None:
                raise ValueError(
                    "H3MCtxApplyPins: spec %d wants a %d frame window but "
                    "only %d frames are available (%s). Enable "
                    "snap_window_down_to_available to use %s instead."
                    % (i, n, available, take_from,
                       snapped if snapped else "nothing"))
            _LOG.warning("obvpm.h3: spec %d window %d -> %d (only %d frames "
                         "available)", i, n, snapped, available)
            n = snapped
        if n == 1:
            raise ValueError(
                "H3MCtxApplyPins: the 1-frame window is not sliceable from a "
                "latent (the last step of a clip spans 4 frames). Use 5 or "
                "more; 1-frame pins arrive with the keyframe path.")

        steps = fr.steps_for_frames(n)
        if steps is None:
            raise RuntimeError(
                "H3MCtxApplyPins: %d frames is not a whole number of latent "
                "steps; the window ladder no longer matches the VAE grid."
                % n)

        # Delivered-frame start of the slice, then mapped onto the RAW
        # timeline (the stored latent includes any pinned scaffolding).
        if take_from == "tail":
            d_start = delivered - n
        elif take_from == "head":
            d_start = 0
        else:
            d_start = cut_frame - n
        raw_start = pinned_head + d_start

        # Latent-grade cut rule: phase-0 slice starts occur only at
        # 17-frame group boundaries. This one check covers everything the
        # old special cases did -- pinned tails, misaligned continuation
        # heads (DESIGN 4a), arbitrary interior cuts.
        if raw_start % fr.FRAMES_PER_GROUP != 0:
            lo = (raw_start // fr.FRAMES_PER_GROUP) * fr.FRAMES_PER_GROUP
            hi = lo + fr.FRAMES_PER_GROUP
            suggest = sorted(set(
                b - pinned_head + n for b in (lo, hi)
                if 0 <= b - pinned_head and b - pinned_head + n <= delivered))
            raise ValueError(
                "H3MCtxApplyPins: spec %d's window would start at raw frame "
                "%d, which is not on the 17-frame latent grid -- the slice "
                "would be unsound. Nearest latent-grade window END frames: "
                "%s. (Or cross to pixels via H3 MCtx From Frames, which "
                "encodes any cut at the cost of exactness.)"
                % (i, raw_start, suggest))
        k = raw_start // fr.FRAMES_PER_GROUP * fr.LATENTS_PER_GROUP
        if fr.frame_at_latent(k) != raw_start:
            raise RuntimeError(
                "H3MCtxApplyPins: step mapping disagreement (frame %d -> "
                "step %d -> frame %d). Upstream grid change; refusing."
                % (raw_start, k, fr.frame_at_latent(k)))
        raw_end = raw_start + n
        if raw_end > raw_frames or k + steps > raw_steps:
            raise ValueError(
                "H3MCtxApplyPins: spec %d's window (raw frames %d..%d) "
                "exceeds the stored latent (%d frames)."
                % (i, raw_start, raw_end, raw_frames))

        video_slice = video[:1, :, k:k + steps].clone()
        covered = fr.pixel_frames(steps)
        if covered != n:
            raise RuntimeError(
                "H3MCtxApplyPins: %d steps cover %d frames, expected %d. "
                "Upstream VAE grid change; refusing." % (steps, covered, n))

        # Audio: end-aligned with the video window's END, boundaries on
        # the cumulative grid, never converted deltas.
        a_frames = int(spec.get("audio_window", 0)) or n
        total_t = int(audio.shape[-1])
        a_lo = max(0, raw_end - a_frames)
        start_idx = fr.audio_total(a_lo)
        end_idx = min(total_t, fr.audio_total(raw_end))
        rt = max(0, end_idx - start_idx)
        if rt < fr.audio_span(a_lo, raw_end):
            _LOG.warning("obvpm.h3: audio window wants %d steps, the latent "
                         "supplies %d; pinning what there is",
                         fr.audio_span(a_lo, raw_end), rt)
        # The grid overhang only exists at the CLIP's end; an interior
        # window ends exactly on its boundary coordinate.
        if raw_end == raw_frames:
            overhang = fr.audio_overhang(total_t, raw_frames)
            if overhang is None:
                _LOG.warning("obvpm.h3: unexpected audio grid (%d steps for "
                             "%d frames); assuming no overhang",
                             total_t, raw_frames)
                overhang = 0.0
        else:
            overhang = 0.0
        audio_slice = (audio[:1, ..., start_idx:end_idx].clone()
                       if rt > 0 else None)

        return {
            "kind": "clip",
            "place": spec.get("place", "before"),
            "video": video_slice,
            "steps": steps,
            "covered": covered,
            "audio": audio_slice,
            "audio_steps": rt,
            "overhang": overhang,
            "width": int(video.shape[4]) * 16,
            "height": int(video.shape[3]) * 16,
            "origin": bundle.get("origin", "sampled"),
            "source_id": bundle.get("self_id", ""),
            # the AUTHORITATIVE resolved spec: source range recomputed from
            # the actually-sliced window (snap_down may have shrunk it), so
            # downstream trim/save read what really happened. source_start
            # is in the source's RAW coordinates -- that is what the
            # sidecar recipe means (summarize_pins derives parent_join_
            # frame from it, seam_report slices the raw latent with it);
            # the delivered-based d_start only equals it for root sources.
            "spec": dict(
                {k: v for k, v in spec.items() if k != "source"},
                source_start=raw_start,
                source_frames=n,
            ),
        }


class H3MCtxApplyPins:
    CATEGORY = "obvpm/h3"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "LATENT", wt.PINS)
    RETURN_NAMES = ("conditioning", "latent", "pins")
    DESCRIPTION = (
        "Slices the pinned windows out of the sources (the sole "
        "frame/latent math site: window snapping, phase checks, "
        "cumulative-total audio cut) and attaches each pinned run to the "
        "conditioning as native H3 keyframe anchors. The pins "
        "output carries the resolved slices -- feed it to Trim/Save so "
        "they read what was actually pinned."
    )
    OUTPUT_TOOLTIPS = (
        "Conditioning with the pinned runs attached. Feed the guider/sampler.",
        "The target latent. ALWAYS wire the sampler from here, not from "
        "the empty-latent node: masked pins write their preserved windows "
        "and denoise mask into it (guide-only runs pass it through "
        "untouched).",
        "The resolved pins. Feed H3TrimPinned / the Save nodes.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT", {
                    "tooltip": "The TARGET clip's empty AV latent -- "
                               "authoritative for frame count and "
                               "resolution. Wire the same latent to the "
                               "sampler."}),
                "snap_window_down_to_available": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When a source cannot supply the requested "
                               "window, snap down the ladder (56/39/22/5) "
                               "instead of refusing. Off = refuse loudly; "
                               "snapping silently weakens continuity."}),
                "freeze_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-render PICTURE ONLY: hold the whole "
                               "audio mask at 0 so the soundtrack comes "
                               "out exactly as it went in. For an upscale/"
                               "refine pass, where the sound is already "
                               "finished -- the sampler denoises the "
                               "nested AV pair together, so without this "
                               "a refine re-renders audio nobody asked it "
                               "to touch. Works with no pins at all, "
                               "which is what the first clip of a "
                               "timeline needs."}),
            },
            "optional": {
                "video_vae": ("VAE", {
                    "tooltip": "Video VAE. Needed ONLY for pins from clips "
                               "with no usable mctx sidecar, which have to "
                               "be encoded from their pixels (the Timeline "
                               "emits those for imported footage). Pins "
                               "from a sidecar never touch a VAE."}),
                "audio_vae": ("VAE", {
                    "tooltip": "Audio VAE, for the same pixel-encoded pins "
                               "as video_vae -- required alongside it even "
                               "when the footage is silent."}),
                "pin_specs": (wt.PINSPECS, {
                    "tooltip": "The spec stack from H3MCtxPinSpec or a "
                               "loader's create_pins output. Unconnected or "
                               "empty = pass-through: the conditioning is "
                               "returned untouched and pins is empty (a "
                               "plain root generation), so the node can "
                               "stay in the graph when not pinning."}),
                "masked_audio_feather": ("INT", {
                    "default": 8, "min": 0, "max": 256,
                    "tooltip": "Masked pins only: half-cosine release of "
                               "the preserved audio across its final ticks "
                               "toward the generated region (8 = 0.2 s). "
                               "0 = hard mask edge."}),
                "audio_denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "With freeze_audio: hold the audio mask at "
                               "this value instead of 0, so the sound "
                               "re-samples alongside the picture. A "
                               "refine at high sigma needs ~0.5 here for "
                               "the model to re-derive LIP SYNC (frozen "
                               "audio gives a near-noise video nothing "
                               "to move the mouth for). The resampled "
                               "audio must then be DISCARDED at save "
                               "time: wire the save node's audio from a "
                               "decode of the SOURCE audio latent. "
                               "Junction-pinned windows stay frozen "
                               "regardless. 0 = today's exact freeze."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    @staticmethod
    def _all_input_values(dynprompt):
        for node_id in dynprompt.all_node_ids():
            try:
                node = dynprompt.get_node(node_id)
            except Exception:
                continue
            yield from (node.get("inputs") or {}).values()

    def apply(self, conditioning, latent, snap_window_down_to_available,
              freeze_audio=False, pin_specs=None, video_vae=None,
              audio_vae=None, masked_audio_feather=8, audio_denoise=0.0,
              dynprompt=None, unique_id=None):
        if not pin_specs:
            if freeze_audio:
                # the first clip of a refine has nothing to pin and still
                # must not have its sound regenerated, so this is a real
                # case rather than a degenerate one
                _LOG.info("obvpm.h3: no pin specs, but audio is frozen; "
                          "the conditioning passes through and the latent "
                          "carries an audio-only mask")
                return (conditioning,
                        nodes_masked.apply_masked_pins(
                            latent, [], masked_audio_feather,
                            freeze_audio=True,
                            audio_denoise=audio_denoise),
                        [])
            _LOG.info("obvpm.h3: no pin specs; conditioning passes through "
                      "untouched (plain root generation)")
            return (conditioning, latent, [])
        for i, s in enumerate(pin_specs or []):
            if s.get("place") == "at_frame":
                raise ValueError(
                    "H3MCtxApplyPins: spec %d places 'at_frame' -- not yet "
                    "implemented; arrives with inside pins/repaint (a later "
                    "phase)." % i)

        # The target latent is read BEFORE slicing now: a pixel source is
        # encoded to this resolution rather than merely checked against
        # it, so the answer has to be in hand first.
        target_video, _ = unpack_av(latent, name="latent")
        latent_t = int(target_video.shape[2])
        frame_count = fr.pixel_frames(latent_t)
        width = int(target_video.shape[4]) * 16
        height = int(target_video.shape[3]) * 16

        pins = _prepare_pins(
            pin_specs, snap_window_down_to_available,
            ctx={"vae": video_vae, "audio_vae": audio_vae,
                 "width": width, "height": height})
        n_before = sum(1 for p in pins if p.get("place") == "before")
        n_after = sum(1 for p in pins if p.get("place") == "after")
        if n_before > 1 or n_after > 1:
            raise ValueError(
                "H3MCtxApplyPins: at most one pin per side; got %d before "
                "/ %d after. (Several pins on the same side have no "
                "coordinate meaning.)" % (n_before, n_after))

        for pin in pins:
            if (pin["width"], pin["height"]) != (width, height):
                raise ValueError(
                    "H3MCtxApplyPins: the pinned clip is %dx%d but this "
                    "clip is %dx%d. A latent cannot be resized; regenerate "
                    "the source at this resolution, or cross to pixels "
                    "explicitly (H3 MCtx From Frames, which resizes to this "
                    "clip's latent)."
                    % (pin["width"], pin["height"], width, height))
            if pin.get("origin") == "encoded":
                _LOG.info("obvpm.h3: pin from an encoded (pixel-grade) "
                          "source; continuity is soft, not exact")
        total_covered = sum(int(p["covered"]) for p in pins)
        if total_covered >= frame_count:
            raise ValueError(
                "H3MCtxApplyPins: pinning %d frames into a %d frame clip; "
                "the generated part must have room -- raise the clip "
                "length or shrink the pin window(s)."
                % (total_covered, frame_count))

        # "both" is in each list: the SAME pin, so it is still one pin
        # per side, trimmed once (pins_trim_totals reads `pins`, not
        # these), and the keyframe rows and the mask describe the same
        # window from the same slice.
        guide_pins, masked_pins = [], []
        for pin in pins:
            mode = (pin.get("spec") or {}).get("mode", "guide")
            if mode in ("masked", "both"):
                masked_pins.append(pin)
            if mode in ("guide", "both"):
                guide_pins.append(pin)
            if mode == "guide" and any(
                    (pin.get("spec") or {}).get(k)
                    for k in ("mask_ramp_frames", "mask_ramp_edge",
                              "mask_hold")):
                _LOG.warning(
                    "obvpm.h3: pin %s carries a mask shape but rides in "
                    "guide mode, which has no mask -- the shape is "
                    "ignored. Switch the mode to masked or both.",
                    pin.get("place"))
        if masked_pins and not nodes_masked.core_masks_available():
            raise ValueError(
                "H3MCtxApplyPins: masked pins need ComfyUI's native "
                "per-stream H3 denoise masks (PR #15375, merged "
                "2026-08-18). Update ComfyUI, or switch the pin's mode "
                "to guide.")
        if guide_pins:
            _ensure_native_anchors()

        # Pinned run coordinates on the target timeline. before: indices
        # 0..covered-1 (extend; trimmed off the head). after: the LAST
        # covered indices (prepend; generation runs TOWARD the pinned
        # content, which generalizes stock's trained-in last-frame anchor;
        # trimmed off the tail). BOTH AT ONCE = bridging: the run departs
        # the before-context and must arrive at the after-context; the
        # free frames in between are the generated connective content.
        # One keyframe entry carries each whole window: core assigns its
        # steps cond_t + cumsum(FRAME_RESCALE * FRAME_PER_TOKEN[k % 5])
        # from k=0, which matches the target's own step pattern at `base`
        # because both the slice start and `base` sit on the 17-frame
        # group grid (phase 0; _prepare_pins refused anything else).
        keyframes = []
        for pin in guide_pins:
            origin = _keyframe_origin(pin)
            place = pin.get("place")
            covered = int(pin["covered"])
            base = 0 if place == "before" else frame_count - covered
            keyframes.append({
                "resolved_frame_index": base,
                "latent": pin["video"],
            })
            if origin:
                keyframes[-1]["origin"] = origin

            rt = int(pin.get("audio_steps", 0))
            pin_audio = pin.get("audio") if rt > 0 else None
            end_frame = 0.0
            if pin_audio is not None:
                # End-align the pinned audio with the pinned video's END
                # on the target timeline: frame `covered` for a
                # before-pin, `frame_count` for an after-pin. The sliced
                # audio reaches `overhang` of a step past the source
                # slice's end (nonzero only when the slice ends at the
                # source clip's end), so the end coordinate moves by that
                # much, then snaps onto the target's own audio grid (a
                # third of a step is 8.3 ms; the snap stops the offset
                # cycling across chains). Core anchors an audio window's
                # START at FRAME_RESCALE * resolved_frame_index and
                # advances 1.0 per audio latent step, so ending at
                # integer audio coordinate `end_coord` means a fractional
                # frame index -- fine, PackedLayout only ever multiplies
                # it (the int restriction lives in the AddGuide node, not
                # the model).
                end_base = covered if place == "before" else frame_count
                end_frame = (float(end_base) +
                             float(pin["overhang"]) / fr.FRAME_RESCALE)
                end_coord = round(fr.FRAME_RESCALE * end_frame)
                end_frame = end_coord / fr.FRAME_RESCALE
                keyframes.append({
                    "resolved_frame_index":
                        (end_coord - rt) / fr.FRAME_RESCALE,
                    "audio_latent": pin_audio,
                })
                if origin:
                    keyframes[-1]["origin"] = origin

            _LOG.info(
                "obvpm.h3: pinned %d frames (%d steps) at the %s of a %d "
                "frame clip at %dx%d (indices %d..%d); audio %s; trim %d "
                "off the %s",
                covered, int(pin["steps"]),
                "head" if place == "before" else "tail",
                frame_count, width, height, base, base + covered - 1,
                ("%d steps ending at frame %.3f" % (rt, end_frame))
                if pin_audio is not None else "off", covered,
                "head" if place == "before" else "tail")

        if n_before and n_after:
            _LOG.info(
                "obvpm.h3: BRIDGING -- %d free frames generated between "
                "the pinned head and tail; the run must depart one "
                "context and arrive at the other (seam convergence on "
                "both ends is seed-dependent, like prepend)",
                frame_count - total_covered)

        # Append to any keyframes already on the conditioning (a stock
        # first/last frame, an AddGuide chain): coordinates now come from
        # one native rulebook, so e.g. extend-toward-image = this pin +
        # a stock last-frame keyframe is a legal combination.
        out = conditioning
        if keyframes:
            prior = list(conditioning[0][1].get("minimax_keyframes", []))
            if prior:
                _LOG.info("obvpm.h3: appending pin to %d existing keyframe "
                          "anchor(s) on the conditioning", len(prior))
            out = node_helpers.conditioning_set_values(conditioning, {
                "minimax_keyframes": prior + keyframes,
            })
        latent_out = latent
        if masked_pins:
            # A masked pin acts through the latent OUTPUT. If nothing
            # consumes it, the sampler runs on the untouched empty
            # latent and the "extend" is silently a root generation --
            # a wrong-but-plausible default, so refuse instead. Scan the
            # DYNPROMPT, not the PROMPT: under the upscale loop's graph
            # expansion this node runs as a prefixed clone, and only the
            # dynamic prompt holds the clones (and their links) under
            # the ids this unique_id can match.
            if dynprompt is not None and unique_id is not None and not any(
                    isinstance(v, list) and len(v) == 2
                    and str(v[0]) == str(unique_id) and int(v[1]) == 1
                    for v in self._all_input_values(dynprompt)):
                raise ValueError(
                    "H3MCtxApplyPins: this run has masked pins, but the "
                    "node's latent output is not wired to anything. "
                    "Masked pins take effect through that latent -- wire "
                    "it to the sampler (in place of the empty latent), "
                    "or the preserved window never reaches sampling.")
            latent_out = nodes_masked.apply_masked_pins(
                latent, masked_pins, masked_audio_feather,
                freeze_audio=freeze_audio, audio_denoise=audio_denoise)
        elif freeze_audio:
            # guide-only pins carry no mask of their own, so the freeze
            # would otherwise be accepted and silently do nothing
            latent_out = nodes_masked.apply_masked_pins(
                latent, [], masked_audio_feather, freeze_audio=True,
                audio_denoise=audio_denoise)
        return (out, latent_out, pins)


class H3TrimPinned:
    CATEGORY = "obvpm/h3"
    FUNCTION = "trim"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    DESCRIPTION = (
        "Removes pinned scaffolding from a decoded clip, picture and "
        "sound together, and truncates the audio tail to exactly "
        "frames/fps (H3 rounds its audio grid up ~8 ms per clip; the "
        "error compounds at every join in a chain). Trim amounts come "
        "from the pins wire -- the same one that fed Apply."
    )
    OUTPUT_TOOLTIPS = (
        "The delivered frames. Feed H3SaveVideoWithMCtx.",
        "The delivered audio, duration-locked to the frames.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pins": (wt.PINS, {
                    "tooltip": "The resolved pins from H3MCtxApplyPins. The "
                               "trim amounts are derived from what was "
                               "actually pinned: before-pins come off the "
                               "head, after-pins off the tail."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip; trimmed by "
                               "the same durations so sound stays locked to "
                               "picture, and tail-matched to exactly "
                               "frames/fps (H3 ships ~8 ms extra sound per "
                               "clip; the error compounds at joins). Leave "
                               "unwired for silent clips."}),
            },
        }

    def trim(self, images, pins, audio=None):
        head, tail = pins_trim_totals(pins)
        total = int(images.shape[0])
        if head + tail >= total:
            raise ValueError(
                "H3TrimPinned: trimming %d+%d frames from a %d frame clip "
                "leaves nothing." % (head, tail, total))
        out_images = images[head:total - tail] if (head or tail) else images
        remaining = total - head - tail

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            cut_head = int(round(head / float(fr.FPS) * sr))
            cut_tail = int(round(tail / float(fr.FPS) * sr))
            length = int(waveform.shape[-1])
            if cut_head + cut_tail >= length:
                raise ValueError(
                    "H3TrimPinned: trimming %.3fs+%.3fs from %.3fs of audio "
                    "leaves nothing. Wire the audio decoded from this same "
                    "clip." % (cut_head / sr, cut_tail / sr, length / sr))
            waveform = waveform[..., cut_head:length - cut_tail]
            # tail-match unconditionally: exactly one right answer (the
            # ~8 ms/clip audio surplus compounds at every join otherwise)
            want = int(round(remaining / float(fr.FPS) * sr))
            have = int(waveform.shape[-1])
            if have > want:
                _LOG.info("obvpm.h3: tail-matched audio, cut %d samples "
                          "(%.2f ms)", have - want,
                          (have - want) / sr * 1000.0)
                waveform = waveform[..., :want]
            out_audio = {"waveform": waveform, "sample_rate": sr}
        elif head or tail:
            _LOG.info("obvpm.h3: trimmed %d head / %d tail frames with no "
                      "audio wired; if this clip has sound, trim it through "
                      "this node too or it will drift %.3fs.",
                      head, tail, (head + tail) / float(fr.FPS))
        return (out_images, out_audio)
