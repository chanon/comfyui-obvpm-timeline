"""Persisting the reference PIXELS a take was conditioned on.

The conditioning a take was sampled with is a pure function of the prompt
text, the reference pixels and the sizing parameters -- there is no seed and
no sampled state in `MiniMaxH3ReferenceToVideo`. The prompt text and the
sizing are already in the sidecar's `prompt` blob. The pixels are the only
part nothing on disk holds, and they are the part a refine pass cannot do
without: reconstructing them by re-running the graph that made them works
today and decays, because it depends on the source files still being there
and on our own loader nodes still behaving the same way (see
private/UPSCALE-REFINE.md 5c).

So they are RECORDED, as bytes, at generation time. Bytes do not depend on
which nodes existed or how they behaved. Same reasoning that put full
latents in the sidecar instead of the seed that would regenerate them.

Layout, beside the clips in the same profile folder:

    _refs/<sha256>.<ext>       one reference, shared by every take using it
    clip_00086.refs.json       which references this take used, in order

Content-addressed, because a character sheet used across twenty clips is
one file. The per-take JSON is a few hundred bytes and is the ASSET; the
`.cond.safetensors` beside it is a deletable CACHE of the same information
already encoded.
"""

import hashlib
import json
import os

FORMAT = "obvpm_refs_v1"
STORE_DIR = "_refs"
SUFFIX = ".refs.json"

# what the recorder can capture. Video references are deliberately absent:
# core presents them to Qwen at 2 fps but VAE-encodes every frame for the
# DiT payload, so reproducing one means keeping all of it -- hundreds of
# megabytes per reference. A take with video references keeps its .cond
# instead, and the save node says so.
KINDS = ("image", "audio")


def _tensor_bytes(tensor):
    """Raw bytes of a tensor, contiguous and on the CPU.

    Hashing these rather than the encoded file keeps the identity of a
    reference independent of how we happen to store it: change the
    container later and the same pixels still hash the same.
    """
    array = tensor.detach().to("cpu").contiguous()
    return array.numpy().tobytes()


def digest(item):
    """The content hash of one recorded reference."""
    h = hashlib.sha256()
    kind = item["kind"]
    h.update(kind.encode("utf-8"))
    if kind == "image":
        tensor = item["image"]
        h.update(("%s%s" % (tuple(tensor.shape), tensor.dtype)).encode("utf-8"))
        h.update(_tensor_bytes(tensor))
    elif kind == "audio":
        audio = item["audio"]
        h.update(str(int(audio["sample_rate"])).encode("utf-8"))
        h.update(_tensor_bytes(audio["waveform"]))
    else:
        raise ValueError("refstore cannot hash a %r reference" % (kind,))
    return h.hexdigest()


def store_dir(folder):
    return os.path.join(folder, STORE_DIR)


def refs_path(video_path):
    """`clip_00086.mp4` -> `clip_00086.refs.json`, beside it."""
    base = video_path
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base + SUFFIX


def has_refs(video_path):
    return os.path.exists(refs_path(video_path))


def _write_atomic(path, write):
    """Write via a temp file in the same directory, then replace.

    A half-written reference is worse than a missing one: missing refuses
    loudly at refine time, half-written verifies as a mismatch against a
    conditioning that was actually fine.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def on_8bit_grid(tensor):
    """True when every value is exactly k/255, so PNG is LOSSLESS for it.

    Usually true, and not by luck: core's reference resize is
    `comfy.utils.lanczos`, which round-trips through PIL in uint8
    (`comfy/utils.py:1099`), so anything that passed through `_resize`
    comes out on the grid whatever floats went in. MEASURED on clip 87 --
    quantizing its references to 8 bits changed the conditioning by
    exactly nothing, in every digit.

    It is false for pixels that reached the socket without a resize and
    without a file behind them: a blend or a compose node's output. Those
    get stored exactly instead, which is why this is decided per
    reference rather than once for the pack.
    """
    scaled = tensor.detach().to("cpu").float() * 255.0
    return bool(((scaled - scaled.round()).abs() < 1e-6).all())


def _save_image(path, tensor, exact):
    import numpy

    if exact:
        from safetensors.numpy import save_file
        array = tensor.detach().to("cpu").contiguous().numpy()
        _write_atomic(path, lambda tmp: save_file({"image": array}, tmp))
        return
    from PIL import Image
    array = (tensor.detach().to("cpu").clamp(0.0, 1.0) * 255.0).round()
    array = array.to("cpu").numpy().astype(numpy.uint8)
    if array.ndim == 4:
        array = array[0]
    _write_atomic(path, lambda tmp: Image.fromarray(array).save(tmp, format="PNG"))


def _load_image(path):
    import numpy
    import torch

    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        return torch.from_numpy(load_file(path)["image"])
    from PIL import Image
    array = numpy.array(Image.open(path).convert("RGB")).astype(numpy.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _save_audio(path, audio):
    from safetensors.numpy import save_file
    waveform = audio["waveform"].detach().to("cpu").contiguous().numpy()
    _write_atomic(path, lambda tmp: save_file(
        {"waveform": waveform}, tmp,
        metadata={"sample_rate": str(int(audio["sample_rate"]))}))


def _load_audio(path):
    import json as _json
    import torch
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        waveform = handle.get_tensor("waveform")
        meta = handle.metadata() or {}
    del _json
    return {"waveform": waveform, "sample_rate": int(meta.get("sample_rate", 0))}


def save_bundle(video_path, bundle, meta=None):
    """Write the take's `.refs.json` and any references not already stored.

    Each image is stored in whichever container is LOSSLESS for it: PNG
    when its values are already on the 1/255 grid, which is the usual case
    (see `on_8bit_grid`), and raw float32 safetensors when they are not.
    Never a lossy container, and never a global choice -- a JPEG or a
    blanket 8-bit cast would change the tokens for exactly the references
    that did not come from a file, which are the ones hardest to rebuild
    any other way.
    """
    items = bundle.get("items") or []
    if not items:
        return None
    folder = os.path.dirname(os.path.abspath(video_path))
    store = store_dir(folder)
    os.makedirs(store, exist_ok=True)

    entries = []
    for index, item in enumerate(items):
        kind = item["kind"]
        if kind not in KINDS:
            raise ValueError("refstore cannot store a %r reference" % (kind,))
        sha = digest(item)
        exact = kind != "image" or not on_8bit_grid(item["image"])
        name = sha + (".safetensors" if exact else ".png")
        path = os.path.join(store, name)
        if not os.path.exists(path):
            if kind == "image":
                _save_image(path, item["image"], exact)
            else:
                _save_audio(path, item["audio"])
        entry = {"kind": kind, "sha256": sha, "file": "%s/%s" % (STORE_DIR, name),
                 "socket": item.get("socket")}
        if kind == "image":
            shape = tuple(item["image"].shape)
            entry["height"], entry["width"] = int(shape[-3]), int(shape[-2])
        entries.append(entry)

    document = {"format": FORMAT, "refs": entries}
    if meta:
        document.update(meta)
    path = refs_path(video_path)
    _write_atomic(path, lambda tmp: open(tmp, "w", encoding="utf-8").write(
        json.dumps(document, indent=1, sort_keys=True)))
    return path


def load_bundle(video_path):
    """Read a take's recorded references back, pixels included.

    Refuses by name on a missing reference file rather than returning a
    short list: a refine pass that silently conditions on two references
    where the original had three is the failure this module exists to
    prevent.
    """
    path = refs_path(video_path)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format") != FORMAT:
        raise ValueError("%s: unknown reference format %r"
                         % (os.path.basename(path), document.get("format")))
    folder = os.path.dirname(os.path.abspath(video_path))
    items = []
    for index, entry in enumerate(document.get("refs", [])):
        file_path = os.path.join(folder, entry["file"].replace("/", os.sep))
        if not os.path.exists(file_path):
            raise ValueError(
                "%s: reference %d (%s) is missing from the store: %s"
                % (os.path.basename(path), index, entry["kind"], entry["file"]))
        if entry["kind"] == "image":
            items.append({"kind": "image", "image": _load_image(file_path),
                          "socket": entry.get("socket")})
        else:
            items.append({"kind": "audio", "audio": _load_audio(file_path),
                          "socket": entry.get("socket")})
    return {"items": items}, document


def store_size(folder):
    """(files, bytes) currently in a folder's reference store."""
    store = store_dir(folder)
    if not os.path.isdir(store):
        return 0, 0
    total = 0
    names = os.listdir(store)
    for name in names:
        try:
            total += os.path.getsize(os.path.join(store, name))
        except OSError:
            pass
    return len(names), total
