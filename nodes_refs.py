"""Recording the reference pixels a take is conditioned on, in-line.

This node is ADDED to a graph, never swapped in for anything. The
references pass through it and continue to `MiniMaxH3ReferenceToVideo`
unchanged; the core node keeps its job, its sockets and its behaviour.
What comes out the side is an OBVPM_H3_REFS bundle for the save node.

    LoadImageCrop A --+                          +--> ref_image_0
                      +--> H3 Record References -+
    LoadImageCrop B --+                          +--> ref_image_1
                                                 +--> refs --> save

IN-LINE RATHER THAN BRANCHED, deliberately. A wire that must pass through
cannot be forgotten; branching means dragging a second link off an output
that is already wired, and its failure mode is a take that looks correct
and is missing a reference.

WHY NINE STATIC OUTPUTS. ComfyUI has no dynamic outputs -- `DynamicOutput`
in `comfy_api/latest/_io.py` is abstract with no subclass, and that is not
an oversight: a link's type is resolved from the upstream CLASS's static
RETURN_TYPES indexed by slot number (`execution.py`, `r[val[1]]`). Inputs
can autogrow because a prompt addresses them BY NAME; outputs are
addressed BY POSITION, so inserting one would silently re-point every
downstream link. The unused slots are hidden by `web/h3_mctx_ui.js`, and
that hiding is COSMETIC ONLY -- slot 3 is slot 3 whether or not it is
drawn, so a graph opened without the JS shows nine sockets and still runs
correctly. Nothing here may depend on it.

Video references are not recorded. Core presents them to Qwen at 2 fps but
VAE-encodes every frame for the DiT payload, so reproducing one means
keeping all of it -- hundreds of megabytes per reference. A take with
video references keeps its `.cond` instead, and the save node says so
rather than leaving it to be discovered at refine time.
"""

import logging

from . import refstore
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

MAX_IMAGES = 9      # mirrors MiniMaxH3ReferenceToVideo's ref_images
MAX_AUDIOS = 3      # ... and its ref_audios

IMAGE_TOOLTIP = (
    "A reference image on its way to MiniMax H3 Reference to Video. Wire "
    "the loader here and this node's matching output onward, so the "
    "reference cannot reach the sampler without being recorded."
)

REFS_TOOLTIP = (
    "The reference pixels this take is conditioned on. Wire into a save "
    "node so an upscale/refine pass can rebuild the same conditioning at "
    "a larger size without depending on the source files still being "
    "there, or on these loader nodes still behaving the same way."
)


class H3RecordReferences:
    """Pass reference images and audio through, and record them."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "record"
    DESCRIPTION = (
        "Records the reference images and audio a take is conditioned on, "
        "in-line on their way to MiniMax H3 Reference to Video. Add it "
        "between the loaders and the reference node; it changes nothing "
        "about what the sampler sees."
    )

    RETURN_TYPES = ("IMAGE",) * MAX_IMAGES + ("AUDIO",) * MAX_AUDIOS + (wt.REFS,)
    RETURN_NAMES = (
        tuple("image_%d" % i for i in range(MAX_IMAGES))
        + tuple("audio_%d" % i for i in range(MAX_AUDIOS))
        + ("refs",)
    )
    OUTPUT_TOOLTIPS = (
        ("the same image, unchanged",) * MAX_IMAGES
        + ("the same audio, unchanged",) * MAX_AUDIOS
        + (REFS_TOOLTIP,)
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(MAX_IMAGES):
            optional["image_%d" % i] = ("IMAGE", {"tooltip": IMAGE_TOOLTIP})
        for i in range(MAX_AUDIOS):
            optional["audio_%d" % i] = ("AUDIO", {
                "tooltip": "A reference audio clip on its way to the "
                           "reference node, recorded on the way past."})
        return {"required": {}, "optional": optional}

    def record(self, **kwargs):
        """Collect what is wired, in socket order, and pass it all through.

        Socket ORDER is what matters, not which numbers are used: core
        presents images first, then videos, then standalone audio, and
        numbers references within each type in the order they arrive. A
        graph that wires image_0 and image_3 gets <Picture 1> and
        <Picture 2>, so this collects in the same order and the save
        node's check against `minimax_refs` lines up.
        """
        items = []
        for i in range(MAX_IMAGES):
            image = kwargs.get("image_%d" % i)
            if image is not None:
                items.append({"kind": "image", "image": image,
                              "socket": "image_%d" % i})
        for i in range(MAX_AUDIOS):
            audio = kwargs.get("audio_%d" % i)
            if audio is not None:
                items.append({"kind": "audio", "audio": audio,
                              "socket": "audio_%d" % i})

        bundle = {"format": refstore.FORMAT, "items": items}
        outputs = tuple(kwargs.get("image_%d" % i) for i in range(MAX_IMAGES))
        outputs += tuple(kwargs.get("audio_%d" % i) for i in range(MAX_AUDIOS))
        return outputs + (bundle,)


def describe(bundle):
    """One line for a log: what a bundle is carrying."""
    items = (bundle or {}).get("items") or []
    images = sum(1 for i in items if i["kind"] == "image")
    audios = sum(1 for i in items if i["kind"] == "audio")
    parts = []
    if images:
        parts.append("%d image%s" % (images, "" if images == 1 else "s"))
    if audios:
        parts.append("%d audio" % audios)
    return " + ".join(parts) if parts else "nothing"


def verify(bundle, conditioning):
    """Check a bundle against the conditioning's own reference blocks.

    Returns a list of complaints, empty when they agree. The comparison is
    possible at all because `minimax_refs` records each reference's latent
    grid, and a reference image is `latent_h * 16` by `latent_w * 16` --
    so a missing, extra, reordered or differently-sized reference shows up
    without needing the VAE.

    What it CANNOT catch is a same-sized substitution. That needs the
    latents themselves and therefore a VAE, which is a bigger ask of the
    graph than this check is worth; the recorder being in-line already
    makes the realistic mistakes impossible.
    """
    if not bundle or not conditioning:
        return []
    try:
        blocks = conditioning[0][1].get("minimax_refs") or []
    except (IndexError, KeyError, TypeError, AttributeError):
        return ["the conditioning has no reference blocks to check against"]

    items = bundle.get("items") or []
    video_blocks = [b for b in blocks if b.get("kind") in ("video", "video_audio")]
    if video_blocks:
        return ["%d video reference(s), which are not recorded -- keep this "
                "take's .cond to refine it" % len(video_blocks)]

    problems = []
    recorded = [i["kind"] for i in items]
    expected = [b.get("kind") for b in blocks]
    if recorded != expected:
        problems.append(
            "the recorder has %s but the conditioning has %s"
            % (", ".join(recorded) or "nothing", ", ".join(expected) or "nothing"))
        return problems

    for index, (item, block) in enumerate(zip(items, blocks)):
        if item["kind"] != "image":
            continue
        shape = tuple(item["image"].shape)
        height, width = int(shape[-3]), int(shape[-2])
        want_h, want_w = int(block["latent_h"]) * 16, int(block["latent_w"]) * 16
        # the recorded pixels are PRE-sizing, so they match only when core
        # did not resize; a mismatch is reported as a note, not a failure,
        # unless nothing about it can be reconciled
        if (height, width) != (want_h, want_w):
            scale_h = height / want_h if want_h else 0
            scale_w = width / want_w if want_w else 0
            if abs(scale_h - scale_w) > 0.02:
                problems.append(
                    "reference %d is %dx%d but the conditioning's is %dx%d, "
                    "and the aspect ratios do not match -- the recorder and "
                    "the reference node are wired to different images"
                    % (index, width, height, want_w, want_h))
    return problems
