"""What this pack's workflows need from the install, checked in one place.

A workflow arrives with a list of node packs it uses, and ComfyUI Manager
can install the missing ones. What nothing checks is whether the packs
that ARE installed are the right ones: a pack too old to read the
workflow's settings, or a fork that registers a node under the same
name with different widgets, loads without a word and fails somewhere
downstream with a message about the symptom.

So every requirement is a `Check` here, run on demand by the timeline
node (server side, before it does anything) and by its widget (which
shows the problems instead of its controls). Each problem says what was
found, what to do, and where to go. A check reads the install -- the
node registry, a pack's pyproject, a node's declared inputs -- and never
imports anything a workflow names.
"""

import logging
import os
import re
import sys

_LOG = logging.getLogger("obvpm.h3")

# ---------------------------------------------------------------- versions

# ComfyUI: the pins pipeline needs core's native keyframe anchoring,
# which shipped in v0.33.0 (2026-08-13).
MIN_COMFYUI = (0, 33, 0)
# comfyui-obvpm: the workflow's settings presets use `when` conditions
# and `# hints`, which the Value Presets node reads from 0.2.3 on. An
# older pack reads the hint as part of the default and refuses the run.
MIN_OBVPM = (0, 2, 3)

GITHUB = "https://github.com/"
MANAGER_HELP = ("In ComfyUI Manager: Custom Nodes Manager, find the pack, "
                "press Update (or Install), then restart ComfyUI.")


class Problem:
    """One thing the install must change before the workflow can run."""

    def __init__(self, key, title, detail, fix, links=()):
        self.key = key
        self.title = title
        self.detail = detail
        self.fix = fix
        self.links = [{"label": label, "url": url} for label, url in links]

    def as_dict(self):
        return {"key": self.key, "title": self.title, "detail": self.detail,
                "fix": self.fix, "links": list(self.links)}

    def as_text(self):
        lines = ["%s: %s" % (self.title, self.detail), "  Fix: %s" % self.fix]
        for link in self.links:
            lines.append("  %s: %s" % (link["label"], link["url"]))
        return "\n".join(lines)


# ----------------------------------------------------------------- helpers

def _registry():
    """The node classes ComfyUI has loaded: a dict lookup, never an import
    of anything named by a workflow."""
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except Exception:
        return {}
    return NODE_CLASS_MAPPINGS


def parse_version(text):
    """'0.2.3' -> (0, 2, 3); anything unreadable -> None."""
    found = re.match(r"\s*v?(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not found:
        return None
    return tuple(int(part or 0) for part in found.groups())


def version_text(version):
    return ".".join(str(part) for part in version)


def pack_version(node_class):
    """The version in the pyproject.toml of the pack a node class came
    from, or None when there is none to read (a copy without the file,
    a class whose module has no file)."""
    module = sys.modules.get(getattr(node_class, "__module__", ""))
    path = getattr(module, "__file__", None)
    if not path:
        return None
    folder = os.path.dirname(os.path.abspath(path))
    # The class may live a folder or two below the pack root, so walk
    # up -- but no further than the folder directly under custom_nodes:
    # above that is ComfyUI itself, whose own pyproject would otherwise
    # answer for any pack that ships without one.
    for _ in range(4):
        candidate = os.path.join(folder, "pyproject.toml")
        if os.path.isfile(candidate):
            try:
                with open(candidate, encoding="utf-8") as fh:
                    text = fh.read(64 * 1024)
            except OSError:
                return None
            found = re.search(r'^\s*version\s*=\s*"([^"]*)"', text, re.M)
            return parse_version(found.group(1)) if found else None
        parent = os.path.dirname(folder)
        if parent == folder or os.path.basename(parent) == "custom_nodes":
            break
        folder = parent
    return None


def declared_inputs(node_class):
    """The input names a node class declares, or None if it will not say.

    Both node APIs end up with an INPUT_TYPES classmethod (core builds
    one for V3 nodes), so this is one call either way. Referenced
    installed classes are trusted code, as they are for /object_info;
    a class that raises while describing itself is reported as unknown
    rather than as broken.
    """
    try:
        spec = node_class.INPUT_TYPES()
    except Exception:
        return None
    names = set()
    for section in ("required", "optional"):
        part = spec.get(section) if isinstance(spec, dict) else None
        if isinstance(part, dict):
            names.update(str(k) for k in part)
    return names


# ------------------------------------------------------------------ checks

def _pack_missing(key, pack, node_id, used_for, repo):
    return Problem(
        key, "%s is not installed" % pack,
        "The workflow uses its %s node (%s), and ComfyUI has no node "
        "of that name." % (node_id, used_for),
        MANAGER_HELP + " ComfyUI Manager's 'Install Missing Custom Nodes' "
        "finds it as well.",
        [("%s on GitHub" % pack, GITHUB + repo)])


def check_comfyui():
    try:
        from comfyui_version import __version__
    except Exception:
        return None                         # too old to say, or not core
    have = parse_version(__version__)
    if have is None or have >= MIN_COMFYUI:
        return None
    return Problem(
        "comfyui", "ComfyUI is too old",
        "This is ComfyUI %s; the workflow needs %s or newer (the timeline "
        "runs on core's native keyframe anchoring, which arrived there)."
        % (__version__, version_text(MIN_COMFYUI)),
        "Update ComfyUI (the Manager's 'Update ComfyUI', update.bat on the "
        "portable build, or git pull), then restart it.",
        [("ComfyUI releases", GITHUB + "comfyanonymous/ComfyUI/releases")])


def check_obvpm():
    registry = _registry()
    presets = registry.get("ValuePresets (obvpm)")
    if presets is None:
        return _pack_missing(
            "obvpm", "comfyui-obvpm", "ValuePresets (obvpm)",
            "the settings presets", "chanon/comfyui-obvpm")
    have = pack_version(presets)
    # The version says what is installed; what the workflow actually
    # needs is a parser that reads `when` and `# hint`. Asked directly
    # when the version cannot be read, so a copy without a pyproject
    # still gets a true answer.
    capable = None
    if have is None:
        try:
            module = __import__(presets.__module__, fromlist=["describe"])
            capable = "hint" in module.describe("x: bool # h")[0]
        except Exception:
            capable = None
    if (have is not None and have >= MIN_OBVPM) or capable:
        return None
    found = ("comfyui-obvpm %s is installed" % version_text(have)
             if have is not None else
             "The installed comfyui-obvpm cannot read this workflow's "
             "settings presets")
    return Problem(
        "obvpm", "comfyui-obvpm needs updating",
        "%s; the workflow needs %s or newer. Its settings presets hide the "
        "turbo LoRA fields while the loader is off and carry hover hints, "
        "which an older pack reads as part of the values and refuses."
        % (found, version_text(MIN_OBVPM)),
        MANAGER_HELP + " Or, in custom_nodes/comfyui-obvpm: git pull.",
        [("comfyui-obvpm releases", GITHUB + "chanon/comfyui-obvpm/releases"),
         ("comfyui-obvpm on the Comfy Registry",
          "https://registry.comfy.org/publishers/chanon/nodes/comfyui-obvpm")])


UPSCALER_REPO = "LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler"
UPSCALER_FORK = "xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus"


def check_upscaler():
    registry = _registry()
    upscaler = registry.get("MinimaxH3LatentUpscaler3D")
    if upscaler is None:
        return _pack_missing(
            "upscaler", "Comfyui_Minimax_h3_latent_Upscaler",
            "MinimaxH3LatentUpscaler3D", "the upscale pass", UPSCALER_REPO)
    inputs = declared_inputs(upscaler)
    if inputs is None or "enable_temporal_chunking" in inputs:
        return None
    fork = "keep_proportion" in inputs or "offload_after_upscale" in inputs
    return Problem(
        "upscaler",
        "The wrong latent upscaler is installed"
        if fork else "The latent upscaler has no temporal chunking",
        ("The installed H3 Latent Upscaler 3D is the 'Plus' fork (%s), "
         "which registers the same node name with different settings. "
         "The workflow then loads it wrong -- device shows 'true' and "
         "precision shows 'cuda' -- and the fork has no temporal chunking, "
         "which the upscale pass relies on: without it, long sequences "
         "come back with drifting, invented detail." % UPSCALER_FORK)
        if fork else
        ("The installed H3 Latent Upscaler 3D does not offer "
         "enable_temporal_chunking, which the upscale pass relies on."),
        "Uninstall the installed upscaler pack, install the original "
        "(%s), restart ComfyUI and reload the workflow." % UPSCALER_REPO,
        [("Comfyui_Minimax_h3_latent_Upscaler on GitHub", GITHUB + UPSCALER_REPO)])


def check_kjnodes():
    if "ModelPreviewOverrideKJ" in _registry():
        return None
    return _pack_missing("kjnodes", "ComfyUI-KJNodes", "ModelPreviewOverrideKJ",
                         "the model preview override, the Set/Get nodes and "
                         "the Sage attention patch", "kijai/ComfyUI-KJNodes")


def check_turbo():
    if "MiniMaxH3TurboLoRA" in _registry():
        return None
    return _pack_missing("turbo", "ComfyUI-MiniMax-H3-Turbo", "MiniMaxH3TurboLoRA",
                         "the larryvrh turbo LoRA loader",
                         "Larryvrh/ComfyUI-MiniMax-H3-Turbo")


def check_spectrum():
    if "SpectrumApplyMiniMaxH3" in _registry():
        return None
    return _pack_missing("spectrum", "ComfyUI-Spectrum-MiniMax-H3",
                         "SpectrumApplyMiniMaxH3", "Spectrum acceleration",
                         "xmarre/ComfyUI-Spectrum-MiniMax-H3")


def check_rgthree():
    if "Power Lora Loader (rgthree)" in _registry():
        return None
    return _pack_missing("rgthree", "rgthree-comfy", "Power Lora Loader (rgthree)",
                         "the LoRA loader", "rgthree/rgthree-comfy")


# Order = order shown. Core first, then the packs that must be RIGHT
# (installed but wrong is the case nothing else reports), then the
# packs that must be there.
CHECKS = (check_comfyui, check_obvpm, check_upscaler, check_kjnodes,
          check_turbo, check_spectrum, check_rgthree)


def problems(checks=CHECKS):
    """Every problem found, in order. A check that itself fails is
    logged and skipped: a broken check must not block a working install."""
    found = []
    for check in checks:
        try:
            problem = check()
        except Exception:
            _LOG.exception("obvpm.h3: compatibility check %s failed",
                           getattr(check, "__name__", check))
            continue
        if problem is not None:
            found.append(problem)
    return found


def require():
    """Raise, naming every problem and its fix, unless the install can
    run the workflow. Called by the timeline node before it does
    anything, so a queued run stops with the reason instead of with a
    symptom somewhere downstream."""
    found = problems()
    if not found:
        return
    raise RuntimeError(
        "obvpm.h3: this workflow cannot run on this install yet:\n\n"
        + "\n\n".join(p.as_text() for p in found))


def register():
    """GET /obvpm/h3/compat -> {"problems": [...]}. Guarded by the caller."""
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/obvpm/h3/compat")
    async def _compat(request):
        import asyncio
        try:
            # off the event loop: a node describing its inputs may scan
            # a models folder
            found = await asyncio.to_thread(problems)
            return web.json_response({"problems": [p.as_dict() for p in found]})
        except Exception as exc:
            _LOG.exception("obvpm.h3: compat route failed")
            return web.json_response({"error": str(exc)}, status=500)
