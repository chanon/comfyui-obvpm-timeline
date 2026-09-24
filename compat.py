"""The floor this pack needs from the install, checked before a run.

The full list of what a WORKFLOW needs -- the packs it uses, the fork
that must not be installed under the same node name -- lives in the
workflow itself, in comfyui-obvpm's Compatibility Check node, which
shows every result on its face and refuses a run with the fixes. This
module is the safety net under that: the two things the timeline pack
itself cannot do without, checked server side by the Timeline node so
that a user whose comfyui-obvpm is too old to HAVE the checker node
still gets told what to update, rather than a failure somewhere
downstream. No UI, no route: the node is the face.
"""

import logging
import os
import re
import sys

_LOG = logging.getLogger("obvpm.h3")

# ---------------------------------------------------------------- versions

# ComfyUI: the pins pipeline needs core's native keyframe anchoring
# (0.33); the shipped workflow's Model Optimization also uses core's
# Model Sparse Attention node, which arrived in 0.35.0 (2026-09-09).
MIN_COMFYUI = (0, 35, 0)
# comfyui-obvpm: the shipped workflow names the pack's nodes by their
# suffixed ids -- "Bundle (obvpm)" and the rest -- which 0.2.0
# introduced; an older pack registers the bare names and every one of
# those nodes loads as missing. 0.2.5 is the floor because it carries
# fixes the timeline workflow runs into and that were reported against
# this pack: presets not switching on frontend 1.53 (obvpm #12), Bundle
# using translated pin names on a non-English frontend, and the circular
# JSON error loading a saved video's workflow with Nodes 2.0 on
# ComfyUI 0.36 (obvpm #13 / timeline #6).
MIN_OBVPM = (0, 2, 5)

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


# ------------------------------------------------------------------ checks

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
        "This is ComfyUI %s; the timeline needs %s or newer."
        % (__version__, version_text(MIN_COMFYUI)),
        "Update ComfyUI (the Manager's 'Update ComfyUI', update.bat on the "
        "portable build, or git pull), then restart it.",
        [("ComfyUI releases", GITHUB + "comfyanonymous/ComfyUI/releases")])


def check_obvpm():
    registry = _registry()
    presets = registry.get("ValuePresets (obvpm)")
    if presets is None:
        return Problem(
            "obvpm", "comfyui-obvpm is not installed",
            "The workflow uses its nodes (settings presets, bundles, gates), "
            "and ComfyUI has none of them.",
            MANAGER_HELP + " ComfyUI Manager's 'Install Missing Custom Nodes' "
            "finds it as well.",
            [("comfyui-obvpm on GitHub", GITHUB + "chanon/comfyui-obvpm")])
    # The pack is found by a suffixed id, so its being found at all
    # already proves 0.2.0 or newer; the version is read to enforce the
    # floor above. Unreadable (a copy without a pyproject -- every
    # release and checkout has one): let it through rather than refuse
    # on a guess.
    have = pack_version(presets)
    if have is None or have >= MIN_OBVPM:
        return None
    return Problem(
        "obvpm", "comfyui-obvpm needs updating",
        "comfyui-obvpm %s is installed; the workflow needs %s or newer."
        % (version_text(have), version_text(MIN_OBVPM)),
        MANAGER_HELP + " Or, in custom_nodes/comfyui-obvpm: git pull.",
        [("comfyui-obvpm releases", GITHUB + "chanon/comfyui-obvpm/releases"),
         ("comfyui-obvpm on the Comfy Registry",
          "https://registry.comfy.org/publishers/chanon/nodes/comfyui-obvpm")])


CHECKS = (check_comfyui, check_obvpm)


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
    run the timeline. Called by the Timeline node before it does
    anything, so a queued run stops with the reason instead of with a
    symptom somewhere downstream."""
    found = problems()
    if not found:
        return
    raise RuntimeError(
        "obvpm.h3: this workflow cannot run on this install yet:\n\n"
        + "\n\n".join(p.as_text() for p in found))
