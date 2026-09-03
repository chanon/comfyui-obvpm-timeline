"""Persisting a MiniMax H3 CONDITIONING to safetensors, exactly.

A CONDITIONING is `list[[Tensor, dict]]`: the text embedding and a dict
of extras. For H3 that dict carries `minimax_refs`, the reference blocks
core builds in MiniMaxH3ReferenceToVideo -- each a small dict holding
its own VAE-encoded latent. So the object is a tree of JSON-able values
with tensors at the leaves, and that is exactly what this module writes:
every tensor into the safetensors tensor area under a path-shaped key,
the tree itself into a `structure` blob with `{"__tensor__": key}` in
each tensor's place.

WHY STORE IT AT ALL, when the prompt graph is already in the sidecar:
the reference IMAGES are the output of upstream nodes (loaders, crops,
composes), so rebuilding conditioning means re-executing a subgraph.
That can silently differ -- an input file overwritten, a crop node whose
behaviour moved, a swapped CLIP checkpoint -- and silently-different
conditioning is not detectable until a seam looks wrong. An artifact
fails loudly; a recipe fails quietly.

Refusals are loud on purpose. Anything that is neither a tensor nor
JSON-able (a HookGroup from the scheduled-CLIP path, say) is refused by
name rather than dropped, because a conditioning that loads without the
piece that shaped it is the failure this file exists to prevent.
"""

import json
import os

FORMAT = "obvpm_cond_v1"

# safetensors metadata is str->str; the tree lives in a tensor blob
# instead, the way mctx.py carries its workflow/prompt JSON.
STRUCTURE_KEY = "structure"

_SCALARS = (str, int, float, bool)


def _is_tensor(obj):
    # duck-typed so this module needs no torch import to be read
    return hasattr(obj, "dtype") and hasattr(obj, "shape") and \
        hasattr(obj, "detach")


def _encode(obj, path, tensors, seen):
    """Tree -> JSON-able tree, tensors collected into `tensors`.

    `seen` maps id(tensor) -> key so a tensor referenced twice is stored
    once. safetensors refuses tensors that share storage, and two
    entries pointing at one embedding is the ordinary case for a
    negative built by conditioning_set_values.
    """
    if obj is None or isinstance(obj, _SCALARS):
        return obj
    if _is_tensor(obj):
        key = seen.get(id(obj))
        if key is None:
            key = path or "tensor"
            tensors[key] = obj.detach().cpu().contiguous()
            seen[id(obj)] = key
        return {"__tensor__": key}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(
                    "conditioning: %s has a non-string key %r; only string "
                    "keys can be stored" % (path or "<root>", k))
            out[k] = _encode(v, "%s/%s" % (path, k) if path else k,
                             tensors, seen)
        return {"__dict__": out}
    if isinstance(obj, (list, tuple)):
        kind = "__tuple__" if isinstance(obj, tuple) else "__list__"
        return {kind: [_encode(v, "%s/%d" % (path, i) if path else str(i),
                               tensors, seen)
                       for i, v in enumerate(obj)]}
    raise ValueError(
        "conditioning: %s is a %s, which cannot be stored. Only tensors, "
        "dicts, lists and JSON scalars round-trip; a conditioning carrying "
        "anything else (hooks, model patchers) must not be persisted, "
        "because loading it back would silently drop the piece."
        % (path or "<root>", type(obj).__name__))


def _decode(node, tensors, path=""):
    if node is None or isinstance(node, _SCALARS):
        return node
    if not isinstance(node, dict):
        raise ValueError("conditioning: malformed structure at %s" % path)
    if "__tensor__" in node:
        key = node["__tensor__"]
        if key not in tensors:
            raise ValueError(
                "conditioning: structure names tensor %r, which the file "
                "does not carry" % key)
        return tensors[key]
    if "__dict__" in node:
        return {k: _decode(v, tensors, "%s/%s" % (path, k))
                for k, v in node["__dict__"].items()}
    if "__list__" in node:
        return [_decode(v, tensors, "%s/%d" % (path, i))
                for i, v in enumerate(node["__list__"])]
    if "__tuple__" in node:
        return tuple(_decode(v, tensors, "%s/%d" % (path, i))
                     for i, v in enumerate(node["__tuple__"]))
    raise ValueError("conditioning: unknown structure node at %s" % path)


def flatten(cond):
    """CONDITIONING -> (tensors dict, structure dict). No file touched."""
    if not isinstance(cond, (list, tuple)) or not cond:
        raise ValueError(
            "conditioning: expected a non-empty CONDITIONING list, got %r"
            % type(cond).__name__)
    tensors = {}
    structure = _encode(list(cond), "cond", tensors, {})
    return tensors, structure


def unflatten(tensors, structure):
    """(tensors dict, structure dict) -> CONDITIONING."""
    return _decode(structure, tensors)


def save_conditioning(path, cond, meta=None):
    """Write a CONDITIONING beside a clip. Atomic: tmp + os.replace.

    Its own file (`clip_00001.cond.safetensors`), never a blob in the
    mctx sidecar: it is bigger than the latents it accompanies, it is
    scaffolding that can be deleted once a timeline is refined, and
    folding it in would widen the crash window of the commit that makes
    a take real.
    """
    import torch
    from safetensors.torch import save_file

    tensors, structure = flatten(cond)
    blob = json.dumps(structure, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    if STRUCTURE_KEY in tensors:
        raise ValueError("conditioning: a tensor is named %r, which the "
                         "structure blob needs" % STRUCTURE_KEY)
    tensors[STRUCTURE_KEY] = torch.frombuffer(bytearray(blob),
                                              dtype=torch.uint8)
    clean = {"format": FORMAT}
    for k, v in (meta or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("cond metadata must be str->str, got %r=%r"
                             % (k, v))
        clean[k] = v
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=clean)
    os.replace(tmp, path)
    return path


def load_conditioning(path):
    """(CONDITIONING, metadata dict) from a .cond.safetensors."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
    if meta.get("format") != FORMAT:
        raise ValueError("%s is not an %s file (format=%r)"
                         % (path, FORMAT, meta.get("format")))
    tensors = load_file(path, device="cpu")
    blob = tensors.pop(STRUCTURE_KEY, None)
    if blob is None:
        raise ValueError("%s carries no %r blob" % (path, STRUCTURE_KEY))
    structure = json.loads(bytes(blob.numpy().tobytes()).decode("utf-8"))
    return unflatten(tensors, structure), meta


def cond_path(video_path):
    """clip_00001.mp4 -> clip_00001.cond.safetensors, same folder.

    Mirrors mctx.sidecar_path. The suffix deliberately does NOT end in
    `.mctx.safetensors`: mctx.scan_for_parent selects lineage candidates
    with `name.endswith(SIDECAR_SUFFIX)`, and a file that matched would
    be scanned as a possible parent.
    """
    base, _ext = os.path.splitext(video_path)
    return base + ".cond.safetensors"


def has_conditioning(video_path):
    """Whether this clip can be refined -- a file test, not a format one."""
    return os.path.isfile(cond_path(video_path))
