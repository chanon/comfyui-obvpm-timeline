"""mctx_v1 sidecar files: read, write, hash, lineage (DESIGN.md sections 2-3).

A take is a clip pair with the same basename:

    clip_00001.mp4                   the video (lossy delivery copy)
    clip_00001.mctx.safetensors      full raw AV latents + header (the RAW)

The sidecar is a standard safetensors file. Tensors: the clip's FULL
unsliced sampled latents, `video` [1,24,T,h/16,w/16] and `audio`
[1,32,2,T40], plus optional JSON BLOBS (see below). The header
(safetensors __metadata__, all strings) carries identity, lineage and the
generation recipe, and is readable without loading tensors: 8-byte length
prefix + JSON. That property is what makes registry-free scanning cheap,
server- or browser-side.

Blobs are how bulky JSON rides along WITHOUT costing that property: each
is a uint8 tensor of UTF-8 JSON (`workflow`, `prompt` -- the same two
things core embeds in a PNG). They live in the tensor area, so the header
stays ~1KB and every scan stays cheap, while safetensors' own index still
allows seeking straight to one blob without touching the latents. Putting
them in __metadata__ instead would make every header read -- including the
browser's ranged read, once per clip in a folder -- drag the whole
workflow across. Stored as plain readable JSON, not compressed: the
latents dominate the file anyway, and mctx is meant to stay open.

The format string is deliberately un-namespaced: mctx sidecars are an open
convention any H3 pack can read or write.
"""

import hashlib
import json
import os
import struct

FORMAT = "mctx_v1"
SIDECAR_SUFFIX = ".mctx.safetensors"

# Header keys written by this implementation. Readers must ignore unknown
# keys; additions are non-breaking, semantic changes bump the format.
HEADER_KEYS = (
    "format", "self_id", "parent_id", "relation", "parent_join_frame",
    "width", "height", "fps", "raw_frames", "pinned_head_frames",
    "pinned_tail_frames", "delivered_frames", "pins",
    "parent_grade",  # "latent" (sliced from the parent's sidecar) or
                     # "pixel" (VAE-encoded from the parent's FRAMES).
                     # The lineage edge is equally real either way; what
                     # differs is the fidelity of the pinned content.
    "user_meta",  # opaque JSON; well-known keys: prompt, seed, steps, refs_note
)

# Spec kinds that carry a lineage edge. "clip_pixels" names a FILE and a
# window instead of carrying latents; the join it describes is just as
# real, which is why it counts here.
LINEAGE_KINDS = ("clip", "clip_pixels")


def sidecar_path(video_path):
    """clip_00001.mp4 -> clip_00001.mctx.safetensors, same folder."""
    base, _ext = os.path.splitext(video_path)
    return base + SIDECAR_SUFFIX


def hash_file(path, chunk_size=1024 * 1024):
    """Streamed SHA-256 of a file, lowercase hex. Identity == content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_header_json(path):
    """The whole safetensors header dict: tensor index + __metadata__."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("%s: too short to be a safetensors file" % path)
        (length,) = struct.unpack("<Q", prefix)
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ValueError("%s: implausible safetensors header length %d"
                             % (path, length))
        raw = f.read(length)
        if len(raw) != length:
            raise ValueError("%s: truncated safetensors header" % path)
    return json.loads(raw.decode("utf-8"))


def read_header(path):
    """The sidecar's string->string metadata, without loading tensors.

    safetensors layout: 8 bytes little-endian header length, then that
    many bytes of JSON whose "__metadata__" key holds the user metadata.
    Raises ValueError on anything that is not a plausible sidecar.
    """
    meta = _read_header_json(path).get("__metadata__") or {}
    if meta.get("format") != FORMAT:
        raise ValueError("%s: not a %s sidecar (format=%r)"
                         % (path, FORMAT, meta.get("format")))
    return meta


def is_sidecar(path):
    """Cheap predicate: readable mctx_v1 header, no exception surface."""
    try:
        read_header(path)
        return True
    except Exception:
        return False


def load_sidecar(path):
    """Full load: (video latent, audio latent, meta dict), tensors on CPU."""
    from safetensors.torch import load_file
    tensors = load_file(path, device="cpu")
    if "video" not in tensors or "audio" not in tensors:
        raise ValueError("%s: sidecar is missing the video/audio tensors" % path)
    return tensors["video"], tensors["audio"], read_header(path)


# ---------------------------------------------------------------------------
# blobs: bulky JSON carried in the tensor area, out of the header
# ---------------------------------------------------------------------------

LATENT_TENSORS = ("video", "audio")

# Written by this implementation; the same two things core embeds in a PNG.
# Readers must tolerate blobs they don't know, and their absence.
BLOB_KEYS = ("workflow", "prompt")


def encode_blob(obj):
    """JSON-able -> uint8 tensor of UTF-8 JSON. None when there is nothing.

    `default=str` because a workflow can carry values json refuses (numpy
    scalars turn up in widget values); losing their exact type is far
    better than failing the save of a finished render.
    """
    import torch
    if obj is None:
        return None
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False,
                     default=str).encode("utf-8")
    if not raw:
        return None
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)


def encode_bytes(raw):
    """Raw bytes -> uint8 tensor, for blobs that are not JSON.

    The overlap clip (see nodes_save) is an encoded MP4: it has to stay
    bytes, and round-tripping it through JSON would inflate it by a third
    for nothing.
    """
    import torch
    if not raw:
        return None
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)


def read_bytes_blob(path, name):
    """One blob back as raw bytes, or None. Never parses."""
    from safetensors import safe_open
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            if name not in f.keys():
                return None
            return bytes(f.get_tensor(name).cpu().numpy().tobytes())
    except (OSError, ValueError):
        return None


def blob_names(path):
    """Blob tensors present, from the header alone -- no tensor reads."""
    header = _read_header_json(path)
    return tuple(k for k in header
                 if k != "__metadata__" and k not in LATENT_TENSORS)


def read_blob(path, name):
    """One blob back as parsed JSON, or None if this sidecar has no such blob.

    Uses safe_open so only that tensor's bytes are read: the latents are
    the bulk of the file and are never touched.
    """
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        if name not in f.keys():
            return None
        raw = bytes(f.get_tensor(name).cpu().numpy().tobytes())
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ValueError("%s: blob %r is not valid JSON (%s)"
                         % (path, name, exc))


def write_sidecar(path, video, audio, meta, blobs=None):
    """Atomic sidecar write: tmp file + os.replace.

    Callers order the larger transaction as MP4 first -> hash -> this;
    sidecar existence is the commit point, so a crash in between leaves a
    plain playable video, never a corrupt take.

    `blobs` is an optional {name: JSON-able} written as blob tensors.
    A `bytes` value is stored verbatim instead of being JSON-encoded.
    """
    from safetensors.torch import save_file
    clean = {}
    for k, v in meta.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("sidecar metadata must be str->str, got %r=%r" % (k, v))
        clean[k] = v
    if clean.get("format") != FORMAT:
        raise ValueError("sidecar metadata must carry format=%s" % FORMAT)
    tensors = {"video": video.contiguous().cpu(),
               "audio": audio.contiguous().cpu()}
    for name, obj in (blobs or {}).items():
        if name in LATENT_TENSORS:
            raise ValueError("blob %r would overwrite a latent tensor" % name)
        encoded = (encode_bytes(obj) if isinstance(obj, (bytes, bytearray))
                   else encode_blob(obj))
        if encoded is not None:
            tensors[name] = encoded
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=clean)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# pins: the generation recipe (serialized PINSPECS)
# ---------------------------------------------------------------------------

def serialize_pins(pin_specs):
    """PINSPECS -> the header's `pins` JSON: specs minus in-memory handles."""
    entries = []
    for spec in pin_specs or []:
        e = {
            "source_id": spec.get("source_id", ""),
            "source_kind": spec.get("source_kind", "clip"),
            "source_start": int(spec.get("source_start", 0)),
            "source_frames": int(spec.get("source_frames", 0)),
            "place": spec.get("place", ""),
            "audio_window": int(spec.get("audio_window", 0)),
        }
        # A pixel source is found by PATH as well as by hash: its parent
        # has no sidecar, so the usual "scan the folder for the sidecar
        # whose self_id matches" cannot reach it. The hash still decides
        # whether what is found is the same file.
        if spec.get("source_path"):
            e["source_path"] = spec["source_path"]
        # Written only for the modes that did not exist before, so
        # pre-mode sidecars stay byte-stable; absent means guide, the
        # only mode there was.
        if spec.get("mode") in ("masked", "both"):
            e["mode"] = spec["mode"]
        # Same rule for the mask shape: written only when it is not the
        # hard hold, so an ordinary masked take's recipe is unchanged.
        # Recorded because a soft hold changes what the join IS --
        # nodes_assemble.masked_continuation reads these to decide
        # whether seam repairs apply.
        for key, zero in (("mask_ramp_frames", 0), ("mask_ramp_edge", 0.0),
                          ("mask_hold", 0.0)):
            if spec.get(key) and spec[key] != zero:
                e[key] = spec[key]
        entries.append(e)
    return json.dumps(entries, separators=(",", ":"))


def parse_pins(meta):
    try:
        return json.loads(meta.get("pins") or "[]")
    except Exception:
        return []


def summarize_pins(pin_specs):
    """The fast-path lineage summary: (relation, parent_id, parent_join_frame).

    Single before-pin from a clip = "extends" (this clip sits AFTER the
    junction at source_start + source_frames); single after-pin from a
    clip = "prepends" (this clip sits BEFORE the junction at
    source_start). Anything the summary cannot express -- multiple pins,
    inside pins, image sources -- yields ("", "", 0) and consumers read
    the full `pins` recipe instead (DESIGN.md section 3).
    """
    specs = [s for s in (pin_specs or [])]
    if len(specs) != 1:
        return "", "", 0
    s = specs[0]
    if s.get("source_kind") not in LINEAGE_KINDS or not s.get("source_id"):
        return "", "", 0
    place = s.get("place")
    # The junction is where the two clips actually change hands, which
    # for a softly-held window is NOT the window edge: the take ships
    # its ramped frames and the source picks up past them. A hard hold
    # gives the window edge back, unchanged.
    from .nodes_masked import handover_frames
    covered = int(s.get("source_frames", 0))
    shape = (int(s.get("mask_ramp_frames", 0) or 0),
             float(s.get("mask_ramp_edge", 0.0) or 0.0),
             float(s.get("mask_hold", 0.0) or 0.0))
    if place == "before":
        return ("extends", s["source_id"],
                int(s.get("source_start", 0))
                + handover_frames(covered, "before", *shape))
    if place == "after":
        return ("prepends", s["source_id"],
                int(s.get("source_start", 0))
                + handover_frames(covered, "after", *shape))
    return "", "", 0


def lineage_grade(pin_specs):
    """"latent" | "pixel" | "" -- how the lineage edge's content was got.

    Companion to summarize_pins, kept separate rather than widening its
    tuple: every existing caller wants the edge, only the saver wants the
    grade, and a 4-tuple would have rippled through all of them.
    """
    specs = list(pin_specs or [])
    if len(specs) != 1:
        return ""
    s = specs[0]
    if s.get("source_kind") not in LINEAGE_KINDS or not s.get("source_id"):
        return ""
    if s.get("place") not in ("before", "after"):
        return ""
    return "pixel" if s.get("source_kind") == "clip_pixels" else "latent"


def pinned_totals(pin_specs):
    """(before_total, after_total) delivered-frame counts across the specs."""
    head = sum(int(s.get("source_frames", 0)) for s in (pin_specs or [])
               if s.get("place") == "before")
    tail = sum(int(s.get("source_frames", 0)) for s in (pin_specs or [])
               if s.get("place") == "after")
    return head, tail


# ---------------------------------------------------------------------------
# MCTX bundles (the wire object; meta IS the parsed header)
# ---------------------------------------------------------------------------

def make_mctx(self_id, video, audio, meta, origin="sampled", clip=None):
    """`clip` is the take's LOCATION; `self_id` is its IDENTITY.

    The two are deliberately different things. `self_id` answers "is this
    the same take?", is stable, and is written to the sidecar. `clip` is
    output-relative and answers "where is it right now?", which is true
    only of this machine at this moment -- so it lives in the bundle and
    NEVER in `meta`, which IS the sidecar header. `serialize_pins` copies
    named fields rather than walking the bundle, so it cannot ride along
    into a file by accident.

    It is here because everything a take owns is derived from its clip
    path -- the sidecar, the `.cond`, the recorded references -- so a
    node holding an MCTX can reach all of them without a second wire that
    might name a different clip.
    """
    return {
        "self_id": self_id,
        "video_latent": video,
        "audio_latent": audio,
        "meta": dict(meta),
        "origin": origin,
        "clip": clip,
    }


def scan_for_parent(folder, parent_id):
    """Find the sidecar whose self_id matches, header-scan only.

    Same-folder scan; returns the sidecar path or None. Missing parent is
    "lineage unknown", never an error (DESIGN.md section 3). The slow
    hash-the-MP4s path is deliberately not implemented here -- callers
    decide when that cost is worth paying.
    """
    if not parent_id:
        return None
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for name in names:
        if not name.endswith(SIDECAR_SUFFIX):
            continue
        p = os.path.join(folder, name)
        try:
            if read_header(p).get("self_id") == parent_id:
                return p
        except Exception:
            continue
    return None
