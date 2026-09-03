"""Getting a take's CONDITIONING back, for a refine pass.

Three sources, in order of how much they can be trusted to be what the
take was actually sampled with:

1. `clip_00086.cond.safetensors` -- the encoded tensors themselves. Exact
   and needs no model, but it is 64.7 MB a clip, so it is a CACHE the
   user may delete.
2. `clip_00086.refs.json` -- the recorded reference pixels plus the
   prompt text and sizing, re-encoded through the same core node that
   made the original. Measured at 2.6e-5 relative to the saved tensors
   (E7b in private/UPSCALE-REFINE.md), which is far below anything that
   could move a frame; conditioning is not a pin.
3. Nothing. Refuse BY NAME. A refine pass that quietly conditions on the
   wrong thing is the failure this whole area exists to prevent, and the
   take is still perfectly playable -- it simply cannot be refined.

WHY NOT REBUILD FROM THE PROMPT GRAPH. The sidecar carries the executed
graph, so re-running it looks tempting and works today. It decays: a node
class the tracer does not know, one of our own loaders changing its
resize, or `pasted/image.png` being overwritten by the next paste. Each
fails safely, and each fails MONTHS LATER at delivery. Recorded pixels
are bytes and do not depend on any of that.
"""

import logging
import os

from . import condstore
from . import refstore

_LOG = logging.getLogger("obvpm.h3")

REF2VA = "MiniMaxH3ReferenceToVideo"


def describe_sources(video_path):
    """Which sources exist for this take, for tooltips and messages."""
    return {"cond": condstore.has_conditioning(video_path),
            "refs": refstore.has_refs(video_path)}


def _core_ref2va():
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    return MiniMaxH3ReferenceToVideo


def _reencode(video_path, clip, vae, audio_vae, target_size=None):
    """Re-run the take's own encode from its recorded references.

    Calls the CORE node rather than reimplementing tokenize/encode, so
    the presentation order, the sizing and the `<Picture i>` labelling
    can never drift out of step with what generation does.

    `target_size` (width, height) re-sizes the encode for a PASS-2
    canvas: under ref_image_size 'match' the pictures scale to the
    generation's pixel area, so a 2x refine that replays the source
    geometry hands the model references half the relative size it
    generated with. Contex-loop's pass-2 rebuilds at the target canvas
    for the same reason. None = replay the recorded geometry exactly.
    """
    bundle, document = refstore.load_bundle(video_path)
    prompt = document.get("prompt")
    if prompt is None:
        raise ValueError(
            "%s recorded its references but not its prompt text, so the "
            "conditioning cannot be rebuilt. Keep this take's .cond."
            % os.path.basename(video_path))

    images, audios = {}, {}
    for index, item in enumerate(bundle["items"]):
        if item["kind"] == "image":
            images["ref_image_%d" % len(images)] = item["image"]
        else:
            audios["ref_audio_%d" % len(audios)] = item["audio"]

    # By default REPLAY the recorded geometry, never the refine
    # resolution: under ref_image_size 'max' the sizing ignores
    # width/height entirely, but under 'match' it is sqrt(W*H / w*h) --
    # resizing the references must be a caller's explicit decision
    # (target_size), not a side effect of the pass being larger.
    width = int(document.get("width") or 0)
    height = int(document.get("height") or 0)
    if target_size is not None:
        width, height = int(target_size[0]), int(target_size[1])
    out = _core_ref2va().execute(
        clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
        width=width,
        height=height,
        length=int(document.get("length") or 5),
        ref_image_size=document.get("ref_image_size") or "max",
        ref_images=images or None, ref_audios=audios or None)
    conditioning = out[0] if isinstance(out, (tuple, list)) else out.result[0]
    return conditioning


def load_for_clip(video_path, clip=None, vae=None, audio_vae=None,
                  prefer="cache", target_size=None):
    """(conditioning, source) for a saved take.

    `prefer` picks which source wins when both exist: "cache" reads the
    `.cond` and needs no model at all, "rebuild" re-encodes. Rebuilding
    when a cache is present is only useful for checking one against the
    other, which is what the probe is for -- so "cache" is the default
    and skipping a 32B encoder load is the reason.
    """
    have = describe_sources(video_path)
    name = os.path.basename(video_path)

    if target_size is not None:
        # A .cond is the SOURCE geometry by definition, so a target-canvas
        # request can only be answered by re-encoding the recorded refs.
        if not have["refs"]:
            raise ValueError(
                "%s has no recorded references, so its conditioning "
                "cannot be rebuilt at the pass-2 canvas. Refine it at "
                "the source canvas (canvas='source'), or regenerate the "
                "take with H3 Record References wired." % name)
        if clip is None or vae is None:
            raise ValueError(
                "%s: rebuilding conditioning at the pass-2 canvas needs "
                "the CLIP and VAE wired (the same ones that generated "
                "it)." % name)
        conditioning = _reencode(video_path, clip, vae, audio_vae,
                                 target_size=target_size)
        _LOG.info("obvpm.h3: %s -- conditioning rebuilt at %dx%d from "
                  "recorded reference(s)", name,
                  int(target_size[0]), int(target_size[1]))
        return conditioning, "refs@target"

    if have["cond"] and prefer == "cache":
        conditioning, _meta = condstore.load_conditioning(
            condstore.cond_path(video_path))
        _LOG.info("obvpm.h3: %s -- conditioning from its .cond", name)
        return conditioning, "cond"

    if have["refs"]:
        if clip is None or vae is None:
            raise ValueError(
                "%s has recorded references but rebuilding their "
                "conditioning needs the CLIP and VAE wired (the same ones "
                "that generated it)." % name)
        conditioning = _reencode(video_path, clip, vae, audio_vae)
        _LOG.info("obvpm.h3: %s -- conditioning rebuilt from %d recorded "
                  "reference(s)", name, len(refstore.load_bundle(video_path)[0]["items"]))
        return conditioning, "refs"

    if have["cond"]:
        conditioning, _meta = condstore.load_conditioning(
            condstore.cond_path(video_path))
        _LOG.info("obvpm.h3: %s -- conditioning from its .cond", name)
        return conditioning, "cond"

    raise ValueError(
        "%s has neither a .cond nor recorded references, so there is "
        "nothing to condition a refine on. It was saved before either "
        "existed, or with save_conditioning off and no recorder wired. "
        "The take is fine; it just cannot be refined." % name)


# ---------------------------------------------------------------------------
# what the recorder cannot see: the prompt text and the sizing
# ---------------------------------------------------------------------------

_TEXT_NODES = ("PrimitiveStringMultiline", "PrimitiveString", "String")


def recipe_from_prompt(prompt, meta=None):
    """Prompt text + sizing for the ref2va node in an executed graph.

    Read at SAVE time, not at refine time, and stored in the
    `.refs.json`. The difference matters: a failure here shows up in the
    log while the graph is still in front of you, instead of months later
    when the pass runs.

    Only the ref2va node's own literals are read -- nothing upstream is
    re-executed and no other node's behaviour is assumed. `width` and
    `height` usually come from a computed wire (a resolution selector),
    so they are taken from the take's own header instead, which is what
    the latent actually is.
    """
    if not prompt:
        return {}
    found = [node for node in prompt.values()
             if node.get("class_type") == REF2VA]
    if len(found) != 1:
        return {}
    inputs = found[0].get("inputs", {})

    text = inputs.get("prompt")
    if isinstance(text, list) and len(text) == 2:
        source = prompt.get(str(text[0]))
        text = (source.get("inputs", {}).get("value")
                if source and source.get("class_type") in _TEXT_NODES else None)
    if not isinstance(text, str):
        return {}

    recipe = {"prompt": text}
    size = inputs.get("ref_image_size")
    if isinstance(size, str):
        recipe["ref_image_size"] = size
    for key in ("width", "height"):
        value = (meta or {}).get(key)
        if value is not None:
            recipe[key] = int(value)
    length = (meta or {}).get("raw_frames")
    if length is not None:
        recipe["length"] = int(length)
    return recipe
