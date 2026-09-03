"""H3LoadVideoWithMCtx: clip picker with sidecar verification (DESIGN.md 4).

Mirrors the core Load Video idea but browses the OUTPUT folder (where
takes land) and adds mctx awareness: when a sidecar pairs with the file
(hash-verified), the clip's full latents + header ride out on the MCTX
wire. No sidecar, or a failed pairing check, emits None -- route the
IMAGE/AUDIO outputs through H3MCtxFromFrames instead.
"""

import logging
import os

import folder_paths

from . import avpack
from . import condload
from . import frames as fr
from . import wiretypes as wt
from . import mctx

_LOG = logging.getLogger("obvpm.h3")

# create_pins convenience modes on the loaders: the two everyday pins,
# collapsing Load -> PinSpec -> Apply to Load -> Apply. Anything fancier
# (at_frame cuts, audio windows, multi-pin stacks) is H3MCtxPinSpec's job.
_PIN_MODES = {
    "none": None,
    "extend (pin tail)": ("tail", "before"),
    "prepend (pin head)": ("head", "after"),
}


def resolve_clip_path(clip):
    """Output-relative clip -> absolute path, refusing anything outside.

    The Timeline's source arrives as serialized widget text, so traversal
    is guarded HERE, once, rather than at each caller.
    """
    root = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(root, clip))
    if os.path.commonprefix([path, root + os.sep]) != root + os.sep:
        raise ValueError(
            "obvpm.h3: %s escapes the output folder; clips are "
            "output-relative paths only." % clip)
    if not os.path.isfile(path):
        raise ValueError(
            "obvpm.h3: %s does not exist under the output folder (moved "
            "or deleted since it was picked?)." % clip)
    return path


def synthetic_header(path):
    """A header for a clip with no trustworthy sidecar: identity, no more.

    A plain video has no latents and no lineage -- but it does have an
    IDENTITY (its content hash) and a length, and those two facts are all
    the seam machinery needs from the LEFT side of a join. Supplying them
    is what lets a take pinned from imported footage name it as a parent,
    the seam between them be derived instead of guessed, and level lock /
    crossfade (both pixel-domain) act on that join at all.

    `no_sidecar` marks it, so nothing mistakes the absence of latents for
    the presence of them: there is nothing here to slice, only something
    to point at.
    """
    from . import nodes_encode as ne
    info = ne.probe_clip(path)
    n = str(int(info["frames"]))
    return {
        "format": mctx.FORMAT,
        "self_id": _cached_hash(path),
        "parent_id": "", "relation": "", "parent_join_frame": "0",
        "width": str(info["width"]), "height": str(info["height"]),
        "fps": str(fr.FPS),
        "raw_frames": n, "delivered_frames": n,
        "pinned_head_frames": "0", "pinned_tail_frames": "0",
        "pins": "[]", "user_meta": "",
        "no_sidecar": "1",
    }


def verified_bundle_or_none(clip):
    """The clip's bundle when its sidecar can be trusted, else None.

    None means exactly one thing: "there are no latents here you may
    rely on" -- no sidecar at all, or one that no longer pairs with the
    video. Both are cases the PIXEL route can still serve, so this
    reports them rather than refusing, and callers that require latents
    turn None into a refusal themselves (load_verified_bundle).

    A missing file or a traversal attempt still raises. Those are not
    "no latents", they are wrong.
    """
    path = resolve_clip_path(clip)
    side = mctx.sidecar_path(path)
    if not os.path.isfile(side):
        return None
    header = mctx.read_header(side)
    actual = _cached_hash(path)
    if header.get("self_id") != actual:
        _LOG.warning(
            "obvpm.h3: %s does not pair with its sidecar (re-encoded, "
            "edited or swapped since the take was saved); its latents "
            "cannot be trusted", clip)
        return None
    video_lat, audio_lat, header = mctx.load_sidecar(side)
    return mctx.make_mctx(actual, video_lat, audio_lat, header,
                          origin="sampled", clip=clip)


def load_verified_bundle(clip):
    """Output-relative clip -> verified MCTX bundle, video never decoded.

    The shared trust path for every LATENT continuation that starts from
    a filename instead of a wire (H3LoadMCtx). Refuses rather than
    degrading: a caller that can accept pixel grade should ask
    verified_bundle_or_none and route accordingly.
    """
    bundle = verified_bundle_or_none(clip)
    if bundle is not None:
        return bundle
    path = resolve_clip_path(clip)
    if not os.path.isfile(mctx.sidecar_path(path)):
        raise ValueError(
            "obvpm.h3: %s has no .mctx.safetensors sidecar. Only clips "
            "saved by the H3 MCtx save nodes carry one; for plain videos "
            "use H3LoadVideoWithMCtx + the pixel route." % clip)
    raise ValueError(
        "obvpm.h3: %s does not pair with its sidecar (the video was "
        "re-encoded, edited or swapped since the take was saved). "
        "Latent continuation refused." % clip)


def _create_pins(bundle, clip, create_pins, pin_window, at_frame=None,
                 pin_mode="masked", mask_shape=None):
    """`at_frame` = the timeline CUT to continue from, in delivered frames.

    None means the clip's own edge (its tail for an extend, its head for
    a prepend). With a cut, an extend pins the window ENDING at the cut
    and a prepend pins the window STARTING there -- so "extend from the
    current cut" needs no separate code path. The grid shift below then
    applies to that frame exactly as it does to a clip edge, which is
    what makes a freely-dragged (off-grid) cut still pin soundly.
    """
    mode = _PIN_MODES[create_pins]
    if mode is None:
        return []
    if bundle is None:
        raise ValueError(
            "create_pins is '%s' but %s has no verified sidecar -- there "
            "are no latents to pin. Re-save the take with an H3 MCtx save "
            "node, set create_pins to none, or take the pixel route: "
            "images/audio -> H3 MCtx From Frames -> H3 MCtx Pin Spec "
            "(encoded, so pixel-grade rather than exact)."
            % (create_pins, clip))
    take_from, place = mode

    # Continuation clips have off-grid edges: a pinned head pushes the
    # delivered content 5 past a group boundary (prepend case), a pinned
    # tail pulls the delivered end 12 short of one (extending a
    # prepend-made clip). The manual PinSpec refuses those cuts and makes
    # the user pick a frame; the convenience mode shifts to the nearest
    # latent-grade cut itself and says where the seam will land. The new
    # clip's pins recipe records the shifted join, so assembly needs no
    # extra bookkeeping.
    meta = bundle["meta"]
    ph = int(meta.get("pinned_head_frames", 0) or 0)
    delivered = int(meta.get("delivered_frames", 0) or 0)
    w = int(pin_window)
    if at_frame is None:
        end = w if take_from == "head" else delivered
    else:
        cut = int(at_frame)
        end = cut + w if take_from == "head" else cut
    off = (ph + end - w) % fr.FRAMES_PER_GROUP
    take_from_frame = 0
    if off:
        if take_from == "head":
            end += fr.FRAMES_PER_GROUP - off
            if end > delivered:
                raise ValueError(
                    "create_pins: %s is too short to prepend to -- the "
                    "first latent-grade window ends at frame %d but only "
                    "%d frames were delivered." % (clip, end, delivered))
            skip = end - w
            _LOG.info(
                "obvpm.h3: %s: head sits off the latent grid (continuation "
                "clip); prepend window shifted to end at frame %d. The new "
                "clip leads into this one at frame %d -- assemble as "
                "new + this[%d:].", clip, end, skip, skip)
        else:
            end -= off
            if end < w:
                raise ValueError(
                    "create_pins: %s is too short to extend -- no "
                    "latent-grade %d-frame window fits before frame %d."
                    % (clip, w, delivered))
            _LOG.info(
                "obvpm.h3: %s: tail sits off the latent grid (pinned-tail "
                "clip); extend window shifted to end at frame %d, skipping "
                "the last %d delivered frames. The new clip continues from "
                "frame %d -- assemble as this[:%d] + new.",
                clip, end, off, end, end)
        take_from, take_from_frame = "at_frame", end

    # An explicit cut must be honoured whether or not it needed shifting.
    # This used to sit inside the `if off:` branch, so a cut that was
    # ALREADY latent-grade -- which is what the widget's snapping
    # produces -- fell through and pinned the clip's own edge instead.
    if at_frame is not None:
        take_from, take_from_frame = "at_frame", end

    from .nodes_pins import H3MCtxPinSpec
    ramp, edge, hold = mask_shape or (0, 0.0, 0.0)
    (specs,) = H3MCtxPinSpec().build(
        mctx=bundle, window=pin_window, take_from=take_from,
        take_from_frame=take_from_frame, place=place, place_at_frame=0,
        audio_window=0, mode=pin_mode, mask_ramp_frames=ramp,
        mask_ramp_edge=edge, mask_hold=hold)
    return specs


def _create_pixel_pins(clip, create_pins, pin_window, at_frame=None,
                       pin_mode="masked", mask_shape=None):
    """The same convenience pin, for a clip with no usable sidecar.

    Deliberately does NOT mirror _create_pins' grid-shifting: that exists
    because a latent slice must begin at cycle phase 0, and an encode
    defines phase 0 wherever the window starts. So a pixel pin honours
    the cut it was given, exactly, and never announces a shifted seam.

    Nothing is resolved here -- not the frame count, not the fps, not
    even whether the file opens. Apply owns that, because Apply is where
    the VAE and the target resolution are, and a spec is pure data
    (DESIGN.md 4).
    """
    mode = _PIN_MODES[create_pins]
    if mode is None:
        return []
    take_from, place = mode
    w = int(pin_window)
    take_from_frame = 0
    if at_frame is not None:
        cut = int(at_frame)
        # "at_frame" always names the frame the window ENDS at: an extend
        # leaves the clip at the cut, a prepend leads into it from there.
        take_from, take_from_frame = "at_frame", (
            cut + w if take_from == "head" else cut)
    return [{
        "source": None,
        "source_id": "",            # nothing to verify: the take is a root
        "source_kind": "clip_pixels",
        "source_path": clip,
        "take_from": take_from,
        "take_from_frame": take_from_frame,
        "requested_window": w,
        # placeholders; Apply resolves the real range against the file
        "source_start": 0,
        "source_frames": w,
        "place": place,
        "place_at_frame": 0,
        "audio_window": 0,
        "mode": pin_mode,
        "mask_ramp_frames": int((mask_shape or (0,))[0]),
        "mask_ramp_edge": float((mask_shape or (0, 0.0))[1]),
        "mask_hold": float((mask_shape or (0, 0.0, 0.0))[2]),
    }]


_PIN_INPUTS = {
    "create_pins": (list(_PIN_MODES), {
        "default": "none",
        "tooltip": "Also emit a ready-made pin spec for this clip: "
                   "'extend' pins its tail so the next generation "
                   "continues after it; 'prepend' pins its head so the "
                   "next generation leads into it. Off-grid edges of "
                   "continuation clips are auto-shifted to the nearest "
                   "latent-grade cut (the log states the seam frame). "
                   "Wire pin_specs straight to H3MCtxApplyPins. Use "
                   "H3MCtxPinSpec instead for explicit at_frame cuts, "
                   "audio windows or multi-pin stacks."}),
    "pin_window": (list(fr.WINDOW_CHOICES), {
        "default": "39",
        "tooltip": "Context window for create_pins. 39 is the everyday "
                   "choice (and the shortest masked-legal window); 90 "
                   "pins more motion at the cost of more "
                   "of the new clip."}),
}

_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
_SCAN_DEPTH = 3

# hash cache: {abs_path: (size, mtime_ns, sha256)}. Disposable memoization,
# never authority (DESIGN.md section 3).
_HASH_CACHE = {}


def _list_clips():
    """Video files under the output folder, relative paths, depth-limited."""
    root = folder_paths.get_output_directory()
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= _SCAN_DEPTH:
            dirnames[:] = []
        for f in filenames:
            if f.lower().endswith(_VIDEO_EXTS):
                p = f if rel == "." else os.path.join(rel, f)
                out.append(p.replace(os.sep, "/"))
    out.sort()
    return out


def _cached_hash(path):
    st = os.stat(path)
    key = os.path.abspath(path)
    hit = _HASH_CACHE.get(key)
    if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
        return hit[2]
    digest = mctx.hash_file(path)
    _HASH_CACHE[key] = (st.st_size, st.st_mtime_ns, digest)
    return digest


class H3LoadMCtx:
    """MCTX-only loader: pick a clip, get its verified bundle -- no decode.

    The lean continuation path: when a graph only needs the latents (the
    standard extend), decoding the MP4 is pure waste. This loader
    hash-verifies the pairing and reads the sidecar; the video itself is
    never opened. Use H3LoadVideoWithMCtx when you also want frames/audio.
    """

    CATEGORY = "obvpm/h3"
    FUNCTION = "load"
    RETURN_TYPES = (wt.MCTX, wt.PINSPECS, "LATENT")
    RETURN_NAMES = ("mctx", "pin_specs", "latent")
    DESCRIPTION = (
        "Loads ONLY a clip's motion-context bundle (latents + lineage) "
        "from its verified sidecar -- the video is never decoded, so this "
        "is the fast path for extend graphs. Refuses when no sidecar "
        "pairs with the file (unlike the full loader, there is no pixel "
        "route to fall back to here). The `latent` output hands the "
        "stored AV latent straight back to a sampler or an upscaler, "
        "with no VAE round trip to lose grade to."
    )
    OUTPUT_TOOLTIPS = (
        "The clip's latents + header for the pins pipeline.",
        "Ready-made pin spec per create_pins; empty when none.",
        "The clip's stored AV latent (video + audio), exactly as the "
        "sampler produced it. Split it with Separate AV Latent to reach "
        "the video stream on its own.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        clips = _list_clips()
        return {
            "required": {
                "clip": (clips if clips else [""], {
                    "tooltip": "A video in the output folder with an mctx "
                               "sidecar (saved by the H3 MCtx save "
                               "nodes)."}),
                **_PIN_INPUTS,
            },
        }

    @classmethod
    def IS_CHANGED(cls, clip, **_):
        return H3LoadVideoWithMCtx.IS_CHANGED(clip)

    @classmethod
    def VALIDATE_INPUTS(cls, clip, **_):
        return H3LoadVideoWithMCtx.VALIDATE_INPUTS(clip)

    def load(self, clip, create_pins="none", pin_window="22"):
        clip = (clip or "").strip()
        if not clip:
            raise ValueError(
                "obvpm.h3: H3LoadMCtx has no clip -- the picker is empty. "
                "To drive the clip from a wire, use H3 MCtx Load by Path.")
        bundle = load_verified_bundle(clip)
        meta = bundle["meta"]
        _LOG.info("obvpm.h3: %s mctx loaded (%s, %s frames delivered), "
                  "video not decoded", clip,
                  meta.get("relation") or "root",
                  meta.get("delivered_frames"))
        return (bundle,
                _create_pins(bundle, clip, create_pins, pin_window),
                avpack.pack_av(bundle["video_latent"], bundle["audio_latent"],
                               clip))


class H3LoadMCtxPath:
    """The same loader, addressed by a path instead of a picker.

    Separate class rather than an optional override on H3LoadMCtx,
    because the two differ in more than one socket and the difference is
    not cosmetic:

    - **No combo.** `H3LoadMCtx`'s picker calls `_list_clips()`, which
      walks the output tree every time the node's schema is built. In a
      loop that folder is being WRITTEN as the run proceeds, so the list
      is stale by construction and the value in it is ignored anyway.
    - **No preview.** The video preview and the drag-drop handler attach
      by class name (`web/h3_mctx_ui.js`, `web/h3_mctx_drop.js`); this
      class is deliberately absent from both, so nothing decodes a clip
      the graph only wanted latents from.
    - **Honest caching.** A picker whose value is overridden must not
      decide `IS_CHANGED`, and the old override could not avoid that --
      see the note on IS_CHANGED below.

    `clip_path` is an ordinary widget, not `forceInput`, so it can be
    typed for a one-off re-run of a single clip and driven by a wire the
    rest of the time.
    """

    CATEGORY = "obvpm/h3"
    FUNCTION = "load"
    RETURN_TYPES = (wt.MCTX, wt.PINSPECS, "LATENT")
    RETURN_NAMES = ("mctx", "pin_specs", "latent")
    DESCRIPTION = (
        "Loads a clip's motion-context bundle from an output-relative "
        "PATH rather than a picker -- for graphs where another node "
        "chooses the clip, such as the upscale loop walking a timeline. "
        "Identical to H3 MCtx Load otherwise: the video is never "
        "decoded, and there is no preview."
    )
    OUTPUT_TOOLTIPS = H3LoadMCtx.OUTPUT_TOOLTIPS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_path": ("STRING", {
                    "default": "",
                    "tooltip": "Output-relative path of a clip with an "
                               "mctx sidecar, e.g. "
                               "'selfie_walk2/clip_00086.mp4'. Usually "
                               "wired from a node that chooses clips; "
                               "type one to re-run a single clip."}),
                **_PIN_INPUTS,
            },
        }

    @classmethod
    def IS_CHANGED(cls, clip_path="", **_):
        path = (clip_path or "").strip()
        if not path:
            # Link-driven: IS_CHANGED runs before the graph does and sees
            # no upstream outputs, so the path is genuinely unknown here.
            # Returning a constant would let run N reuse run N-1's clip;
            # NaN never compares equal, so the sidecar is re-read.
            return float("nan")
        return H3LoadVideoWithMCtx.IS_CHANGED(path)

    @classmethod
    def VALIDATE_INPUTS(cls, clip_path="", **_):
        # Same blindness as above: an empty value at validation time may
        # still resolve to a real clip, so let load() refuse it by name.
        if not (clip_path or "").strip():
            return True
        return H3LoadVideoWithMCtx.VALIDATE_INPUTS(clip_path)

    def load(self, clip_path="", create_pins="none", pin_window="22"):
        clip = (clip_path or "").strip()
        if not clip:
            raise ValueError(
                "obvpm.h3: H3LoadMCtxPath has no clip_path -- nothing is "
                "wired to it and the field is empty.")
        bundle = load_verified_bundle(clip)
        meta = bundle["meta"]
        _LOG.info("obvpm.h3: %s mctx loaded by path (%s, %s frames "
                  "delivered), video not decoded", clip,
                  meta.get("relation") or "root",
                  meta.get("delivered_frames"))
        return (bundle,
                _create_pins(bundle, clip, create_pins, pin_window),
                avpack.pack_av(bundle["video_latent"], bundle["audio_latent"],
                               clip))


class H3LoadVideoWithMCtx:
    CATEGORY = "obvpm/h3"
    FUNCTION = "load"
    RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", wt.MCTX, wt.PINSPECS)
    RETURN_NAMES = ("video", "images", "audio", "mctx", "pin_specs")
    DESCRIPTION = (
        "Loads a clip from the output folder with motion-context "
        "awareness: when a hash-verified .mctx.safetensors sidecar pairs "
        "with the file, its latents + lineage ride out on the mctx output "
        "for exact latent-grade continuation. Without one (or when the "
        "video was re-encoded/swapped), mctx is None and only the pixel "
        "route is available."
    )
    OUTPUT_TOOLTIPS = (
        "The clip as a VIDEO object (core-compatible).",
        "Decoded frames.",
        "Decoded audio track.",
        "The clip's latents + header for the pins pipeline; None when no "
        "verified sidecar pairs with this file.",
        "Ready-made pin spec per create_pins; empty when none.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        clips = _list_clips()
        return {
            "required": {
                "clip": (clips if clips else [""], {
                    "tooltip": "A video in the output folder. Clips saved by "
                               "H3SaveVideoWithMCtx carry a sidecar and load "
                               "with exact continuation latents."}),
                **_PIN_INPUTS,
            },
        }

    @classmethod
    def IS_CHANGED(cls, clip, **_):
        path = os.path.join(folder_paths.get_output_directory(), clip)
        parts = []
        for p in (path, mctx.sidecar_path(path)):
            try:
                st = os.stat(p)
                parts.append("%d:%d" % (st.st_size, st.st_mtime_ns))
            except OSError:
                parts.append("absent")
        return "|".join(parts)

    @classmethod
    def VALIDATE_INPUTS(cls, clip, **_):
        if not clip:
            return "no clip selected (the output folder has no videos)"
        path = os.path.join(folder_paths.get_output_directory(), clip)
        if not os.path.isfile(path):
            return "clip not found: %s" % path
        return True

    def load(self, clip, create_pins="none", pin_window="22"):
        path = os.path.join(folder_paths.get_output_directory(), clip)
        from comfy_api.input_impl import VideoFromFile
        video = VideoFromFile(path)
        components = video.get_components()
        images, audio = components.images, components.audio

        bundle = None
        side = mctx.sidecar_path(path)
        if os.path.isfile(side):
            try:
                header = mctx.read_header(side)
                actual = _cached_hash(path)
                if header.get("self_id") != actual:
                    _LOG.warning(
                        "obvpm.h3: %s does not pair with its sidecar (the "
                        "video was re-encoded, edited or swapped since the "
                        "take was saved). Latent continuation refused; the "
                        "pixel route still works.", clip)
                else:
                    video_lat, audio_lat, header = mctx.load_sidecar(side)
                    bundle = mctx.make_mctx(actual, video_lat, audio_lat,
                                            header, origin="sampled",
                                            clip=clip)
                    _LOG.info(
                        "obvpm.h3: %s + verified sidecar (%s, %s frames "
                        "delivered)", clip, header.get("relation") or "root",
                        header.get("delivered_frames"))
            except Exception:
                _LOG.exception("obvpm.h3: failed to read sidecar %s; loading "
                               "the clip without it", side)
        return (video, images, audio, bundle,
                _create_pins(bundle, clip, create_pins, pin_window))


class H3LoadConditioning:
    """A saved take's CONDITIONING, for re-sampling it at a larger size.

    Takes the MCTX rather than a path of its own, deliberately: the
    bundle carries the take's location (`clip`), so one node names the
    clip and everything else follows the wire. A widget here would be a
    second place that could name a different take -- and, when the
    upscale loop starts choosing clips for itself, a widget cannot be
    driven by a link at all.
    """

    CATEGORY = "obvpm/h3"
    FUNCTION = "load"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "source")
    DESCRIPTION = (
        "The conditioning a saved take was sampled with, so an upscale/"
        "refine pass can sample it again larger. Reads the clip's .cond "
        "when it is there, otherwise rebuilds it from the references "
        "recorded beside it (wire the same CLIP and VAE that generated "
        "the take), otherwise refuses and says which is missing."
    )
    OUTPUT_TOOLTIPS = (
        "The take's conditioning.",
        "Where it came from: 'cond' (loaded) or 'refs' (rebuilt).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mctx": (wt.MCTX, {
                    "tooltip": "The take to fetch conditioning for, from "
                               "H3 MCtx Load or the timeline."}),
                "prefer": (["cache", "rebuild"], {
                    "default": "cache",
                    "tooltip": "Which source wins when the take has both. "
                               "'cache' reads the .cond and needs no model "
                               "at all; 'rebuild' re-encodes the recorded "
                               "references, which is only useful for "
                               "comparing one against the other."}),
                "canvas": (["source", "target"], {
                    "default": "source",
                    "tooltip": "'source' replays the take's own geometry "
                               "-- right for a same-size refine. 'target' "
                               "rebuilds from the recorded references at "
                               "the canvas of the latent wired into "
                               "target_latent, so an UPSCALE pass's "
                               "'match' picture refs scale to the pass-2 "
                               "area instead of describing the small "
                               "canvas (needs clip + vae; ignores "
                               "prefer, since a .cond is source geometry "
                               "by definition)."}),
            },
            "optional": {
                "target_latent": ("LATENT", {
                    "tooltip": "canvas='target' only: the pass-2 latent "
                               "whose resolution the conditioning is "
                               "rebuilt for -- wire the UPSCALED video "
                               "latent (or the AV concat) here."}),
                "clip": ("CLIP", {
                    "lazy": True,
                    "tooltip": "Needed only to rebuild from recorded "
                               "references. Must be the same text encoder "
                               "the take was generated with. LAZY: wiring "
                               "it costs nothing on takes that have a "
                               ".cond -- the encoder is not even loaded."}),
                "vae": ("VAE", {
                    "lazy": True,
                    "tooltip": "Video VAE, for re-encoding recorded "
                               "reference images. Rebuild only, and lazy "
                               "like clip."}),
                "audio_vae": ("VAE", {
                    "lazy": True,
                    "tooltip": "Audio VAE. Rebuild only, and only when the "
                               "take had audio references."}),
            },
        }

    def check_lazy_status(self, mctx, prefer="cache", canvas="source",
                          clip=None, vae=None, audio_vae=None, **_):
        """Ask for the models only when this take has to be rebuilt.

        The wire is how a graph says "rebuilding is possible here", and
        it should not also mean "load a 32B encoder every run". Which
        source will win is knowable without any of them: it is a
        question about two filenames beside the clip. So resolve that
        first and request nothing when the cache answers it.

        Silence on every error. This runs before the node does, and its
        only job is deciding what to evaluate -- a bad mctx or a missing
        clip must produce `load`'s own message, not a stack trace from
        the scheduler.
        """
        try:
            clip_name = (mctx or {}).get("clip")
            if not clip_name:
                return []
            have = condload.describe_sources(resolve_clip_path(clip_name))
        except Exception:
            return []
        if have["cond"] and prefer == "cache" and canvas != "target":
            return []
        if not have["refs"]:
            # nothing to rebuild FROM: either the .cond is the only
            # source, or load() is about to refuse by name
            return []
        return [name for name, value in (("clip", clip), ("vae", vae),
                                         ("audio_vae", audio_vae))
                if value is None]

    # the socket is named `mctx`, so the parameter must be too -- which
    # shadows this module's `mctx` import for the body below. Nothing here
    # needs that module; keep it that way.
    def load(self, mctx, prefer="cache", canvas="source", clip=None,
             vae=None, audio_vae=None, target_latent=None):
        clip_name = (mctx or {}).get("clip")
        if not clip_name:
            raise ValueError(
                "obvpm.h3: this MCTX has no clip behind it (it was built "
                "from frames rather than loaded from a saved take), so "
                "there is no conditioning to fetch.")
        path = resolve_clip_path(clip_name)
        target_size = None
        if canvas == "target":
            if target_latent is None:
                raise ValueError(
                    "obvpm.h3: canvas='target' needs target_latent wired "
                    "-- the pass-2 latent whose resolution the "
                    "conditioning is rebuilt for.")
            video, _audio = avpack.unpack_av(
                target_latent, name="target_latent", allow_video_only=True)
            target_size = (int(video.shape[4]) * 16,
                           int(video.shape[3]) * 16)
        conditioning, source = condload.load_for_clip(
            path, clip=clip, vae=vae, audio_vae=audio_vae, prefer=prefer,
            target_size=target_size)
        return (conditioning, source)
