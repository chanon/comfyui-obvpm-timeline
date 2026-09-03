"""comfyui-obvpm-h3: non-linear MiniMax H3 clip composition.

Takes are clip pairs (MP4 + .mctx.safetensors latent sidecar) with
content-addressed lineage; continuation runs through the pins pipeline
(spec -> prepare -> apply) on core's native keyframe anchoring, and a
timeline widget composes the takes by smart-cut.

Companion to comfyui-obvpm (image loaders, gates, switches, bundles,
Value Presets). The two packs share nothing at import time, so either
installs alone; node class names are unchanged from when both lived in
one pack, so saved workflows keep loading.
"""

from .nodes_assemble import H3Assemble, H3AssembleUpscale, H3Timeline
from .nodes_encode import H3MCtxFromFrames
from .nodes_load import (H3LoadConditioning, H3LoadMCtx, H3LoadMCtxPath,
                         H3LoadVideoWithMCtx)
from .nodes_loop import H3UpscaleLoopEnd, H3UpscaleLoopStart
from .nodes_mode import H3RunModeGate
from .nodes_refs import H3RecordReferences
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
    "H3LoadMCtxPath": H3LoadMCtxPath,
    "H3LoadConditioning": H3LoadConditioning,
    "H3MCtxFromFrames": H3MCtxFromFrames,
    "H3MCtxPinSpec": H3MCtxPinSpec,
    "H3MCtxApplyPins": H3MCtxApplyPins,
    "H3TrimPinned": H3TrimPinned,
    "H3Assemble": H3Assemble,
    "H3AssembleUpscale": H3AssembleUpscale,
    "H3Timeline": H3Timeline,
    "H3ResultPreview": H3ResultPreview,
    "H3RecordReferences": H3RecordReferences,
    "H3UpscaleLoopStart": H3UpscaleLoopStart,
    "H3UpscaleLoopEnd": H3UpscaleLoopEnd,
    "H3RunModeGate": H3RunModeGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SaveVideoWithMCtx": "H3 MCtx Save Video",
    "H3TrimAndSaveVideoWithMCtx": "H3 MCtx Trim and Save Video",
    "H3SaveMCtxForVideo": "H3 MCtx Save",
    "H3LoadVideoWithMCtx": "H3 MCtx Load Video",
    "H3LoadMCtx": "H3 MCtx Load",
    "H3LoadMCtxPath": "H3 MCtx Load by Path",
    "H3LoadConditioning": "H3 MCtx Load Conditioning",
    "H3MCtxFromFrames": "H3 MCtx From Frames",
    "H3MCtxPinSpec": "H3 MCtx Pin Spec",
    "H3MCtxApplyPins": "H3 MCtx Apply Pins",
    "H3TrimPinned": "H3 MCtx Trim Pinned",
    "H3Assemble": "H3 MCtx Assemble",
    "H3AssembleUpscale": "H3 Assemble Upscale",
    "H3Timeline": "H3 MCtx Timeline",
    "H3ResultPreview": "H3 MCtx Result Preview",
    "H3RecordReferences": "H3 Record References",
    "H3UpscaleLoopStart": "H3 Upscale Loop Start",
    "H3UpscaleLoopEnd": "H3 Upscale Loop End",
    "H3RunModeGate": "H3 Run Mode Gate",
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
