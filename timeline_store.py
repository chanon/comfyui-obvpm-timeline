"""On-disk timeline sequence for a project folder under ComfyUI output.

The Timeline widget's `sequence` text (cuts, gaps, loop line, seam
overrides) is written beside the takes as plain UTF-8 so opening another
project folder can restore the last cut without rereading the workflow.
"""

import os

import folder_paths

from .nodes_load import _SCAN_DEPTH, _VIDEO_EXTS
from .nodes_save import folder_text

STATE_NAME = "obvpm_h3_timeline.sequence"


def _output_root():
    return os.path.abspath(folder_paths.get_output_directory())


def state_path(base_folder):
    """Absolute path to the state file; refuses traversal outside output."""
    sub = folder_text(base_folder)
    root = _output_root()
    rel = "%s/%s" % (sub, STATE_NAME) if sub else STATE_NAME
    path = os.path.abspath(os.path.join(root, rel.replace("/", os.sep)))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(
            "timeline state path escapes the output folder: %r" % rel)
    return path


def load_state(base_folder):
    """The saved sequence text, or None when no file exists yet."""
    path = state_path(base_folder)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_state(base_folder, sequence):
    """Write the sequence text atomically beside the folder's takes."""
    path = state_path(base_folder)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    text = "" if sequence is None else str(sequence)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


_SEARCH_LIMIT = 250


def list_output_folders():
    """Output-relative folders that hold videos or a saved timeline."""
    return search_output_folders("")


def search_output_folders(query):
    """Folders under output, filtered by a case-insensitive path substring.

    With an empty query, returns the same depth-limited project list as
    before (folders that contain a video or a saved timeline). With a
    query, walks the whole output tree and returns every directory whose
    path contains the query, up to _SEARCH_LIMIT matches.
    """
    root = folder_paths.get_output_directory()
    q = folder_text(query).lower()
    found = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if not q:
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= _SCAN_DEPTH:
                dirnames[:] = []
        folder = "" if rel == "." else rel.replace(os.sep, "/")
        if folder in seen:
            continue
        if q:
            if q not in folder.lower():
                continue
        else:
            if STATE_NAME not in filenames and not any(
                    f.lower().endswith(_VIDEO_EXTS) for f in filenames):
                continue
        seen.add(folder)
        found.append(folder)
        if len(found) >= _SEARCH_LIMIT:
            break
    return sorted(found, key=lambda x: (x.count("/"), x.lower()))


def browse_folder(parent):
    """Immediate subdirectories of an output-relative folder.

    `parent` is "" for the output root. Refuses traversal. `parent` in
    the result is the folder one step up, or None at the root.
    """
    sub = folder_text(parent)
    root = _output_root()
    path = root if not sub else os.path.abspath(
        os.path.join(root, sub.replace("/", os.sep)))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(
            "folder browse escapes the output folder: %r" % sub)
    if not os.path.isdir(path):
        raise ValueError("folder not found: %r" % (sub or "(output root)"))
    children = []
    try:
        names = os.listdir(path)
    except OSError:
        names = []
    for name in names:
        if name.startswith("."):
            continue
        child = os.path.join(path, name)
        if not os.path.isdir(child):
            continue
        rel = name if not sub else "%s/%s" % (sub, name)
        children.append({"path": rel.replace("\\", "/"), "name": name})
    children.sort(key=lambda c: c["name"].lower())
    up = None
    if sub:
        up = sub.rsplit("/", 1)[0] if "/" in sub else ""
    return {"folder": sub, "parent": up, "children": children}


def ensure_folder(base_folder):
    """Create an output-relative folder (and parents). Returns the path."""
    sub = folder_text(base_folder)
    root = _output_root()
    path = root if not sub else os.path.abspath(
        os.path.join(root, sub.replace("/", os.sep)))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(
            "new folder escapes the output folder: %r" % sub)
    os.makedirs(path, exist_ok=True)
    return sub
