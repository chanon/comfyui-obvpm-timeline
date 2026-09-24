"""comfyui-obvpm-timeline: non-linear clip composition, MiniMax H3 today.

Takes are clip pairs (MP4 + .mctx.safetensors latent sidecar) with
content-addressed lineage; continuation runs through the pins pipeline
(spec -> prepare -> apply) on core's native keyframe anchoring, and a
timeline widget composes the takes by smart-cut.

Companion to comfyui-obvpm (image loaders, gates, switches, bundles,
Value Presets). The two packs share nothing at import time, so either
installs alone; node class names are unchanged from when both lived in
one pack, so saved workflows keep loading.
"""

from .nodes_assemble import H3Assemble, H3Timeline
from .nodes_budget import H3VramHeadroom
from .nodes_encode import H3MCtxFromFrames
from .nodes_joint import (H3ContextWindows, H3JointAudioMask,
                          H3JointConditioning, H3JointLatent)
from .nodes_load import H3LoadMCtx, H3LoadVideoWithMCtx
from .nodes_render import H3JointRender
from .nodes_result import H3ResultPreview
from .nodes_pins import (
    H3MCtxApplyPins,
    H3MCtxPinSpec,
    H3TrimPinned,
)
from .nodes_save import (
    H3SaveMCtxForVideo,
    H3SaveVideoWithMCtx,
    H3TrimAndSaveVideoWithMCtx,
)

try:
    from . import preview_route
    preview_route.register()
except Exception:  # headless/test runs have no PromptServer; nodes still work
    import logging
    logging.getLogger("obvpm.h3").info(
        "obvpm.h3: preview route not registered (no server)", exc_info=True)

NODE_CLASS_MAPPINGS = {
    "H3SaveVideoWithMCtx": H3SaveVideoWithMCtx,
    "H3TrimAndSaveVideoWithMCtx": H3TrimAndSaveVideoWithMCtx,
    "H3SaveMCtxForVideo": H3SaveMCtxForVideo,
    "H3LoadVideoWithMCtx": H3LoadVideoWithMCtx,
    "H3LoadMCtx": H3LoadMCtx,
    "H3MCtxFromFrames": H3MCtxFromFrames,
    "H3MCtxPinSpec": H3MCtxPinSpec,
    "H3MCtxApplyPins": H3MCtxApplyPins,
    "H3TrimPinned": H3TrimPinned,
    "H3JointLatent": H3JointLatent,
    "H3JointConditioning": H3JointConditioning,
    "H3JointAudioMask": H3JointAudioMask,
    "H3ContextWindows": H3ContextWindows,
    "H3VramHeadroom": H3VramHeadroom,
    "H3JointRender": H3JointRender,
    "H3Assemble": H3Assemble,
    "H3Timeline": H3Timeline,
    "H3ResultPreview": H3ResultPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SaveVideoWithMCtx": "H3 MCtx Save Video",
    "H3TrimAndSaveVideoWithMCtx": "H3 MCtx Trim and Save Video",
    "H3SaveMCtxForVideo": "H3 MCtx Save",
    "H3LoadVideoWithMCtx": "H3 MCtx Load Video",
    "H3LoadMCtx": "H3 MCtx Load",
    "H3MCtxFromFrames": "H3 MCtx From Frames",
    "H3MCtxPinSpec": "H3 MCtx Pin Spec",
    "H3MCtxApplyPins": "H3 MCtx Apply Pins",
    "H3TrimPinned": "H3 MCtx Trim Pinned",
    "H3JointLatent": "H3 Join Latents",
    "H3JointConditioning": "H3 Joint Conditioning",
    "H3JointAudioMask": "H3 Joint Audio Mask",
    "H3ContextWindows": "H3 Context Windowing",
    "H3VramHeadroom": "H3 VRAM Headroom",
    "H3JointRender": "H3 Joint VAE Decode and Save",
    "H3Assemble": "H3 MCtx Assemble",
    "H3Timeline": "H3 MCtx Timeline",
    "H3ResultPreview": "H3 MCtx Result Preview",
}

WEB_DIRECTORY = "./web"

# Every node carries the pack's name, so a search for "obvpm" finds them
# all and a node in a workflow says where it came from. The suffix is the
# same one comfyui-obvpm applies, deliberately: display names are what
# saved workflows show, and they did not change with the split.
_SUFFIX = "(obvpm)"
NODE_DISPLAY_NAME_MAPPINGS = {
    key: name if name.endswith(_SUFFIX) else "%s %s" % (name, _SUFFIX)
    for key, name in NODE_DISPLAY_NAME_MAPPINGS.items()
}
for _key in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS.setdefault(_key, "%s %s" % (_key, _SUFFIX))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
