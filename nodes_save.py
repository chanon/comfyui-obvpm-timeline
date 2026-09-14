"""H3SaveVideoWithMCtx: MP4 + mctx sidecar, one transaction (DESIGN.md 2-4).

Owns its encode so the MP4 and the sidecar are written by one node with
one trustworthy pairing hash -- deliberately NOT downstream of VHS or
another saver. Transaction order: MP4 first -> hash it -> sidecar tmp +
os.replace. Sidecar existence is the commit point; a crash in between
leaves a plain playable video.

The encode itself pipes raw frames to an ffmpeg subprocess when a binary
is available (VideoHelperSuite's mechanism, ~2.2x faster), and otherwise
falls back to core's in-process PyAV path. Owning the transaction and
borrowing the faster mechanism are independent choices -- see
_encode_mp4.
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile

import folder_paths

from . import condstore
from . import frames as fr
from . import wiretypes as wt
from . import mctx
from .avpack import unpack_av

_LOG = logging.getLogger("obvpm.h3")

_VIDEO_EXTS = (".mp4",)

# Frames converted to bytes at a time. An IMAGE tensor is already large
# (float32); materialising the whole uint8 copy alongside it would add a
# gigabyte on a long take for no gain.
_ENCODE_CHUNK = 32

# memoised ffmpeg discovery: empty = not looked yet, [None] = none found
_FFMPEG = []


def _ffmpeg_exe():
    """An ffmpeg binary to encode with, or None to stay in-process.

    Discovery order mirrors VideoHelperSuite's, so a machine already set
    up for that pack needs no extra configuration here.
    """
    if _FFMPEG:
        return _FFMPEG[0]
    found = None
    for var in ("OBVPM_FFMPEG_PATH", "VHS_FORCE_FFMPEG_PATH"):
        candidate = os.environ.get(var)
        if candidate and os.path.isfile(candidate):
            found = candidate
            break
    if found is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            candidate = get_ffmpeg_exe()
            if candidate and os.path.isfile(candidate):
                found = candidate
        except Exception:
            pass
    if found is None:
        found = shutil.which("ffmpeg")
    _FFMPEG.append(found)
    _LOG.info("obvpm.h3: video encoder = %s",
              found or "PyAV, in process (no ffmpeg binary found)")
    return found


def _write_audio_raw(audio, frames):
    """Delivered audio as interleaved f32le, for ffmpeg's second input.

    Raw floats rather than a WAV: no header to get wrong, and no integer
    quantisation before the AAC encoder ever sees the samples. Returns
    (path, sample_rate, channels).
    """
    import numpy as np
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    wf = waveform[0] if waveform.dim() == 3 else waveform
    # same truncation the in-process encoder applies
    want = int(math.ceil(sample_rate / float(fr.FPS) * frames))
    wf = wf[..., :want]
    interleaved = wf.transpose(0, 1).contiguous().cpu().numpy()
    fd, path = tempfile.mkstemp(prefix="obvpm_h3_audio_", suffix=".f32")
    with os.fdopen(fd, "wb") as handle:
        handle.write(interleaved.astype(np.float32).tobytes())
    return path, sample_rate, int(wf.shape[0])


def encode_mp4_stream(path, blocks, frames, height, width, audio, crf,
                      metadata=None):
    """Encode `frames` frames that arrive as an iterable of [n, H, W, 3]
    float blocks, without ever holding them all.

    The streaming entry for anything that decodes more picture than fits
    in memory (H3 Joint VAE Decode and Save decodes a whole timeline a few seconds at
    a time). ffmpeg reads the blocks off a pipe as they come; without an
    ffmpeg binary the in-process encoder needs the tensor whole, so the
    blocks are gathered first -- correct, and a warning says what it
    costs.
    """
    exe = _ffmpeg_exe()
    if exe:
        _encode_with_ffmpeg(exe, path, blocks, audio, crf, metadata=metadata,
                            frames=frames, height=height, width=width)
        return
    import torch
    _LOG.warning("obvpm.h3: no ffmpeg binary, so %d frames are gathered in "
                 "memory for the in-process encoder (%.1f GB); install ffmpeg "
                 "or set OBVPM_FFMPEG_PATH to stream instead", frames,
                 frames * height * width * 3 * 4 / 1e9)
    _encode_with_pyav(path, torch.cat([b for b in blocks], 0), audio, crf,
                      metadata=metadata)


def _encode_with_ffmpeg(exe, path, images, audio, crf, metadata=None,
                        frames=None, height=None, width=None):
    """Pipe raw frames to ffmpeg (the mechanism VHS Video Combine uses).

    `images` is one [F, H, W, 3] tensor, or -- with `frames`, `height`
    and `width` given -- any iterable of such blocks, written in order.

    Encoder settings are kept identical to the in-process path -- libx264,
    default preset, same crf, yuv420p -- so the two produce the same
    stream parameters. That matters: H3Assemble stream-copies packets
    between clips and refuses to splice pieces whose codec extradata
    differs, so clips saved before and after this change must remain
    interchangeable (verified byte-identical extradata).
    """
    if frames is None:
        frames = int(images.shape[0])
        height, width = int(images.shape[1]), int(images.shape[2])
        images = [images]
    head = [exe, "-v", "error", "-nostdin", "-y"]
    meta_path = _ffmetadata_file(metadata) if metadata else None
    if meta_path:
        # FIRST input, so ffmpeg's default -map_metadata 0 reads the tags
        # from here; the raw video/audio follow as inputs 1 and 2. It
        # carries no streams, so stream selection is unaffected.
        head += ["-i", meta_path]
    args = head + ["-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "%dx%d" % (width, height), "-r", str(fr.FPS),
                   "-i", "-"]
    audio_path = None
    try:
        if audio is not None:
            audio_path, sample_rate, channels = _write_audio_raw(
                audio, frames)
            args += ["-f", "f32le", "-ar", str(sample_rate),
                     "-ac", str(channels), "-i", audio_path, "-c:a", "aac"]
        args += ["-c:v", "libx264", "-crf", str(int(crf)),
                 "-pix_fmt", "yuv420p"]
        if meta_path:
            # custom tag NAMES only survive in isobmff with this flag,
            # and ComfyUI finds a clip's workflow by tag name
            args += ["-map_metadata", "0",
                     "-movflags", "use_metadata_tags+faststart"]
        args += [path]

        proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        written = 0
        try:
            for part in images:
                for lo in range(0, int(part.shape[0]), _ENCODE_CHUNK):
                    block = part[lo:lo + _ENCODE_CHUNK]
                    # ROUND, not truncate. .byte() truncates toward zero,
                    # so every pixel lost an average half level and a
                    # decode->encode->decode round trip came out a full
                    # luma darker than its source. Copied packets do not
                    # go through this, so a re-encoded bridge sat visibly
                    # below the stream-copied frames on either side of it
                    # -- the same class of artefact the level lock exists
                    # to remove.
                    block = (block * 255).round().clamp(0, 255).byte()
                    block = block.cpu().numpy()
                    proc.stdin.write(block.tobytes())
                    written += int(block.shape[0])
        except (BrokenPipeError, OSError):
            pass  # ffmpeg died early; its stderr says why
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        proc.stderr.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg exited %d: %s"
                               % (proc.returncode, err[:400]))
        if err:
            _LOG.warning("obvpm.h3: ffmpeg: %s", err[:400])
        if written != frames:
            raise RuntimeError("obvpm.h3: %d frames were written to %s but "
                               "%d were promised" % (written, path, frames))
    finally:
        for tmp in (audio_path, meta_path):
            if not tmp:
                continue
            try:
                os.remove(tmp)
            except OSError:
                pass


def _encode_with_pyav(path, images, audio, crf, metadata=None):
    """Core's own encoder, frame by frame in this process."""
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
    from fractions import Fraction
    components = VideoComponents(
        images=images, audio=audio, frame_rate=Fraction(fr.FPS))
    video = VideoFromComponents(components)
    if metadata:
        try:
            video.save_to(path, crf=float(crf), metadata=dict(metadata))
            return
        except TypeError:
            # older comfy_api without the metadata kwarg: the clip still
            # gets written, it just carries no workflow
            _LOG.warning("obvpm.h3: this ComfyUI's video writer takes no "
                         "metadata; the MP4 will carry no workflow (the "
                         "sidecar still does)")
    video.save_to(path, crf=float(crf))


def _ffmetadata_file(tags):
    """Write an FFMETADATA1 file for `tags` (str -> str); returns its path.

    The values are whole workflow documents -- tens to hundreds of KB --
    so they cannot ride on the command line: Windows caps one near 32KB
    and ffmpeg would simply fail on any real graph. Reading them from a
    file is how VideoHelperSuite does it too.
    """
    import tempfile
    lines = [";FFMETADATA1"]
    for key, value in tags.items():
        v = str(value)
        # backslash FIRST, or the escapes below get escaped again
        for a, b in (("\\", "\\\\"), (";", "\\;"), ("#", "\\#"),
                     ("=", "\\="), ("\n", "\\\n")):
            v = v.replace(a, b)
        lines.append("%s=%s" % (key, v))
    fd, path = tempfile.mkstemp(suffix=".ffmeta")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def read_video_workflow(path):
    """The workflow embedded in a video file, or None.

    Two schemes exist in the wild and both are read here, because a clip
    without an mctx sidecar may well have come from either:

      * core's SaveVideo -- and ours -- write separate `workflow` and
        `prompt` container tags, which is also the only shape ComfyUI's
        own MP4 reader accepts.
      * VideoHelperSuite writes ONE JSON blob into `comment`, holding
        `prompt` plus every extra_pnginfo key, so `workflow` sits inside
        it. Its `prompt` is itself a JSON *string*, hence the re-parse.

    Returns {"workflow": obj|None, "prompt": obj|None, "source": str} or
    None when the file carries nothing.
    """
    import json

    def loads(value):
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except ValueError:
            return None

    import av
    if not os.path.isfile(path):
        return None          # absent is an answer, not an error
    try:
        with av.open(path) as container:
            tags = dict(container.metadata or {})
    except Exception:
        _LOG.exception("obvpm.h3: could not read metadata from %s", path)
        return None

    workflow = loads(tags.get("workflow"))
    prompt = loads(tags.get("prompt"))
    if workflow is not None or prompt is not None:
        return {"workflow": workflow, "prompt": prompt, "source": "tags"}

    blob = loads(tags.get("comment"))
    if isinstance(blob, dict):
        wf = blob.get("workflow")
        return {"workflow": loads(wf) if isinstance(wf, str) else wf,
                "prompt": loads(blob.get("prompt")),
                "source": "comment"}
    return None


def _workflow_tags(blobs):
    """The graph as MP4 container tags: name -> JSON string.

    The tag NAMES match core's SaveVideo (`prompt`, `workflow`) on
    purpose: ComfyUI's own MP4 reader (isobmff.ts) accepts a tag only if
    its name is one of those two, so anything else embeds fine and then
    never loads.
    """
    import json
    return {k: json.dumps(v, separators=(",", ":"), ensure_ascii=False,
                          default=str)
            for k, v in (blobs or {}).items()}


def _strip_lineage(conditioning):
    """Conditioning minus the keyframes a lineage pin put there.

    Makes the wiring not matter: the same file comes out whether the
    CONDITIONING was taken before H3MCtxApplyPins or after it. That is
    worth enforcing here rather than asking of every workflow, because
    the two are not even consistently different -- a MASKED pin acts
    through the latent and leaves the conditioning untouched, while a
    guided pin appends keyframes to it, so "wire it from the other node"
    is a rule whose consequences vary by pin mode.

    What stays is everything per-SHOT: the prompt, the references, and
    any CONTENT keyframe another node anchored. What goes is per-POSITION
    and belongs to the pass that walks the timeline. Untagged keyframes
    are kept: nothing but this pack's pins carries the tag, so anything
    without it came from elsewhere and is content by definition.

    Returns (conditioning, dropped) without touching the input -- the
    sampler downstream is still using it.
    """
    dropped = 0
    out = []
    for entry in conditioning:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2
                and isinstance(entry[1], dict)):
            out.append(entry)
            continue
        extras = entry[1]
        kfs = extras.get("minimax_keyframes")
        if not kfs:
            out.append(entry)
            continue
        kept = [kf for kf in kfs
                if not (isinstance(kf, dict)
                        and kf.get("origin") == nodes_pins_PIN_ORIGIN())]
        dropped += len(kfs) - len(kept)
        if len(kept) == len(kfs):
            out.append(entry)
            continue
        trimmed = dict(extras)
        if kept:
            trimmed["minimax_keyframes"] = kept
        else:
            trimmed.pop("minimax_keyframes", None)
        out.append([entry[0], trimmed])
    return out, dropped


def nodes_pins_PIN_ORIGIN():
    # imported lazily: nodes_pins imports this module's siblings, and a
    # top-level import here would close the cycle
    from .nodes_pins import PIN_ORIGIN
    return PIN_ORIGIN


def _save_conditioning(video_path, conditioning, self_id, enabled=True):
    """Write the take's `.cond.safetensors`, or nothing when unwired.

    AFTER the sidecar, deliberately. The sidecar is the commit point of
    the save transaction, so a crash here leaves a complete, playable,
    extendable take that merely cannot be refined -- which is exactly
    the state every clip made before this input existed is in.

    Its own file rather than a sidecar blob: it is comparable to the
    latents in size, it is scaffolding that can be deleted once a
    timeline has been refined, and folding it in would widen the crash
    window of the write that makes a take real.

    It is also the ONLY record of what the take was conditioned on. An
    earlier design kept the reference pixels beside it as a recipe to
    re-encode from, and was dropped when reference adapters ("refmods",
    pooled or trained latents with no pixel origin) arrived: once part
    of the conditioning has no recipe, the artefact is the only faithful
    copy, and a partial recipe rebuilds something that looks right and
    is missing a block.
    """
    if not conditioning:
        return None
    if not enabled:
        # Worth a line: the wire says "refinable" and the toggle says
        # otherwise, and the consequence only shows up much later, when
        # an upscale pass refuses a clip the user thought was covered.
        _LOG.info("obvpm.h3: %s -- conditioning is wired but "
                  "save_conditioning is off, so this take can NOT be "
                  "refined later; the .cond is the only record of its "
                  "conditioning", os.path.basename(video_path))
        return None
    clean, dropped = _strip_lineage(conditioning)
    path = condstore.cond_path(video_path)
    condstore.save_conditioning(path, clean, {"self_id": self_id})
    _LOG.info("obvpm.h3: saved conditioning %s (%.1f MB)%s -- this take can "
              "be refined", os.path.basename(path),
              os.path.getsize(path) / 1e6,
              (" without %d lineage keyframe(s), which a refine pass "
               "rebuilds for itself" % dropped) if dropped else "")
    return path


COND_TOOLTIP = (
    "Optional: the CONDITIONING this take was sampled with, stored "
    "beside the clip as .cond.safetensors. An upscale/refine pass has "
    "to sample the same clip again at a larger size, and it needs the "
    "same conditioning to do it -- text, reference images and audio, "
    "and any refmods applied on top. Rebuilding that from the prompt "
    "graph would mean re-running everything that made it, which can "
    "silently differ. Unwired = the take saves normally and simply "
    "cannot be refined later."
)

COND_SAVE_TOOLTIP = (
    "Write the wired conditioning beside the clip. It is a real amount "
    "of disk -- comparable to the latents, and dominated by the "
    "reference images' share of the prompt -- so turn this off for "
    "throwaway takes and leave it on for anything you may want to "
    "upscale later. Off costs nothing now and cannot be recovered "
    "afterwards: the conditioning only exists while the graph that "
    "built it is still wired up. Does nothing when the conditioning "
    "input is unconnected."
)


def _workflow_blobs(prompt, extra_pnginfo):
    """The graph that produced this take, for the sidecar's blob tensors.

    The same two things core embeds in a PNG or MP4: the API-format
    `prompt` (what actually ran) and the UI `workflow` (what you can
    reopen and edit). ComfyUI passes both as hidden inputs; they are None
    when the node is called outside a running prompt (tests, direct calls).

    `--disable-metadata` is honoured, so a user who has turned provenance
    off everywhere does not silently get a full workflow written here --
    it names paths, model files and often the OS username.
    """
    try:
        from comfy.cli_args import args
        if args.disable_metadata:
            return {}
    except Exception:
        pass  # not running under ComfyUI; nothing to honour
    blobs = {}
    if prompt is not None:
        blobs["prompt"] = prompt
    if isinstance(extra_pnginfo, dict) and extra_pnginfo.get("workflow"):
        blobs["workflow"] = extra_pnginfo["workflow"]
    return blobs


OVERLAP_BLOB = "overlap"
OVERLAP_TAIL_BLOB = "overlap_tail"


def guided_sides(pins):
    """{"before", "after"} -- the sides whose window was RE-RENDERED.

    Only a guided pin re-renders its window: it rides as native keyframe
    rows and the model draws that stretch again, which is what disagrees
    with the context and what a crossfade blends away. A masked pin
    preserves the source latents verbatim, so its "re-render" is the
    source's own rendering -- measured at 0.005 luma from it, against
    the ~2.87 the fade exists to hide. Keeping that costs ~1 MB and an
    extra encode per take and can never be used, so it is not kept.
    """
    out = set()
    for p in pins or []:
        spec = p.get("spec") or {}
        # exactly "guide": a masked window is not re-rendered, and
        # "both" masks it too, so neither has a second rendering to keep
        if spec.get("mode", "guide") == "guide":
            place = p.get("place") or spec.get("place")
            if place in ("before", "after"):
                out.add(place)
    return out


def _overlap_bytes(untrimmed_images, untrimmed_audio, n, crf, tail=False):
    """The take's own re-render of a pinned window, as a small MP4.

    This is the one thing a crossfade needs and the delivered clip does
    not contain. An extend is handed its parent's latent tail, re-renders
    exactly that window, then continues; we deliver from AFTER the window
    and keep the parent's pixels for it. Those two renderings of the same
    moment disagree on level, and the join sits precisely where they
    meet.

    Keeping the re-render lets assembly fade between them instead of
    cutting: same moment, same content, so the fade is invisible as
    content and spreads the level difference over the whole window.

    Returns None whenever it cannot be captured -- no untrimmed frames
    wired, or a root clip with no pinned head. Takes saved without it
    stay valid; they simply cannot be crossfaded.
    """
    if untrimmed_images is None or n <= 0:
        return None
    total = int(untrimmed_images.shape[0])
    if total < n:
        _LOG.warning("obvpm.h3: untrimmed images hold %d frames but the "
                     "pinned window is %d; not storing an overlap",
                     total, n)
        return None
    images = untrimmed_images[total - n:] if tail else untrimmed_images[:n]
    audio = None
    if untrimmed_audio is not None:
        try:
            wave = untrimmed_audio["waveform"]
            rate = int(untrimmed_audio["sample_rate"])
            cut = int(round(n / float(fr.FPS) * rate))
            if tail:
                # The head of the audio is a real boundary; its END is
                # not -- H3 ships ~8 ms of surplus per clip, which the
                # trim node removes from the DELIVERED audio but not
                # from this untrimmed copy. Count back from where the
                # picture actually ends, or the overlap carries the
                # surplus and drifts against its own frames.
                end = min(int(wave.shape[-1]),
                          int(round(total / float(fr.FPS) * rate)))
                audio = {"waveform": wave[..., max(0, end - cut):end],
                         "sample_rate": rate}
            else:
                audio = {"waveform": wave[..., :min(cut, wave.shape[-1])],
                         "sample_rate": rate}
        except (KeyError, TypeError, ValueError):
            audio = None
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "overlap.mp4")
        try:
            # THE encode site, so the overlap splices with everything else
            H3SaveVideoWithMCtx._encode_mp4(p, images, audio, int(crf))
            with open(p, "rb") as f:
                return f.read()
        except Exception:
            _LOG.exception("obvpm.h3: could not encode the overlap clip; "
                           "saving without it")
            return None


def read_overlap(clip_path, tail=False):
    """The overlap MP4 bytes stored beside a take, or None.

    `tail` picks the take's re-render of its pinned TAIL (a prepend)
    rather than its head (an extend).
    """
    return mctx.read_bytes_blob(
        mctx.sidecar_path(clip_path),
        OVERLAP_TAIL_BLOB if tail else OVERLAP_BLOB)


def _save_prefix(base_folder, filename_prefix):
    """Join the folder and the name prefix into one save path.

    Kept separate on the node so the folder can be driven from the same
    value as the Timeline's base_folder while the prefix stays per-node.
    Either side may be empty: no folder saves at the output root, and no
    prefix saves into the folder under its own name.

    The traversal guard matches the Timeline's (preview_route): the pair
    must land inside the output root, or the save is refused rather than
    writing somewhere unexpected.
    """
    base = str(base_folder or "").strip().strip("/\\")
    prefix = str(filename_prefix or "").strip().lstrip("/\\")
    rel = "%s/%s" % (base, prefix) if base and prefix else (base or prefix)
    if not rel:
        raise ValueError(
            "H3 save: base_folder and filename_prefix are both empty; "
            "give at least one so the clip has somewhere to go.")
    root = os.path.abspath(folder_paths.get_output_directory())
    landed = os.path.abspath(os.path.join(root, rel))
    if os.path.commonpath([root, landed]) != root:
        raise ValueError(
            "H3 save: %r escapes the output folder." % rel)
    return rel


class H3SaveVideoWithMCtx:
    CATEGORY = "obvpm/h3"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    DESCRIPTION = (
        "Saves a finished H3 take as a clip pair: the MP4 plus a "
        ".mctx.safetensors sidecar holding the clip's full raw latents "
        "and lineage. Any clip saved here can later seed extensions via "
        "its sidecar. Wire the TRIMMED images/audio (the delivered clip) "
        "and the sampler's raw latent; wire the same pins wire that fed "
        "Apply so lineage is recorded. For the one-node version, use "
        "H3 MCtx Trim and Save Video."
    )
    OUTPUT_TOOLTIPS = ("Path of the written MP4; the sidecar sits next to it.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "The sampler's RAW output latent for this take "
                               "(the same one you decode). Stored whole in "
                               "the sidecar; the pinned head/tail stay in it "
                               "and are mapped out via the header."}),
                "images": ("IMAGE", {
                    "tooltip": "Delivered frames (AFTER the trim node removed "
                               "pinned scaffolding)."}),
                "base_folder": ("STRING", {
                    "default": "project1",
                    "tooltip": "Output-relative folder to save into. Same "
                               "meaning as the Timeline's base_folder, so "
                               "one value can drive both. Empty = the "
                               "output root."}),
                "filename_prefix": ("STRING", {
                    "default": "clip",
                    "tooltip": "Filename prefix within base_folder, like "
                               "core save nodes. Numbering is appended "
                               "automatically."}),
                "crf": ("INT", {
                    "default": 19, "min": 0, "max": 51,
                    "tooltip": "H.264 quality (lower = better, bigger). THE "
                               "one that matters: assembly stream-copies "
                               "these frames and matches this crf for the "
                               "few it re-encodes, so the take's own "
                               "setting is what the delivered cut looks "
                               "like. The MP4 is the delivery copy; the "
                               "sidecar keeps the lossless latents "
                               "regardless."}),
                "save_conditioning": ("BOOLEAN", {
                    "default": True, "tooltip": COND_SAVE_TOOLTIP}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Delivered audio (after the same trim). Leave "
                               "unwired for a silent MP4; the sidecar still "
                               "stores the audio latent."}),
                "pins": (wt.PINS, {
                    "tooltip": "The resolved pins from H3MCtxApplyPins. "
                               "Carries the authoritative generation recipe "
                               "for the sidecar and derives the lineage "
                               "edge. Unconnected = root clip."}),
                "metadata": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional provenance as JSON, stored verbatim "
                               "in the sidecar header (user_meta). Well-known "
                               "keys browser UI looks for: prompt, seed, "
                               "steps, refs_note (nothing can fingerprint "
                               "reference images automatically -- a refs_note "
                               "is the mitigation). Any other keys are yours. "
                               "Empty is fine; invalid JSON is refused."}),
                "untrimmed_images": ("IMAGE", {
                    "tooltip": "Optional: the decode output BEFORE the trim "
                               "node. Only its pinned head is kept, stored "
                               "in the sidecar so assembly can crossfade "
                               "this take's own re-render of the join "
                               "against its parent's. Unwired = the take "
                               "still saves, but cannot be crossfaded."}),
                "untrimmed_audio": ("AUDIO", {
                    "tooltip": "Optional: the matching untrimmed audio, so "
                               "the crossfade covers sound as well as "
                               "picture."}),
                "conditioning": ("CONDITIONING", {"tooltip": COND_TOOLTIP}),
            },
            # Stored as sidecar blobs, not in the header -- see mctx.py.
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save(self, images, samples, base_folder, filename_prefix, crf,
             save_conditioning=True,
             audio=None, pins=None, metadata="", prompt=None,
             extra_pnginfo=None, untrimmed_images=None,
             untrimmed_audio=None, conditioning=None):
        video_lat, audio_lat = unpack_av(samples, name="samples")
        raw_steps = int(video_lat.shape[2])
        raw_frames = fr.pixel_frames(raw_steps)
        width = int(video_lat.shape[4]) * 16
        height = int(video_lat.shape[3]) * 16

        delivered = int(images.shape[0])
        if (int(images.shape[2]), int(images.shape[1])) != (width, height):
            raise ValueError(
                "H3SaveVideoWithMCtx: images are %dx%d but the latent decodes "
                "to %dx%d. Wire the images decoded from this same latent."
                % (int(images.shape[2]), int(images.shape[1]), width, height))

        from .nodes_pins import pins_trim_totals
        head, tail = pins_trim_totals(pins)
        if delivered != raw_frames - head - tail:
            raise ValueError(
                "H3SaveVideoWithMCtx: %d delivered frames but the raw latent "
                "covers %d and the pin specs account for %d pinned head + %d "
                "pinned tail frames (expected %d delivered). Wire the images "
                "through the trim node with the trim counts from Apply, and "
                "wire the SAME pin_specs here."
                % (delivered, raw_frames, head, tail, raw_frames - head - tail))

        user_meta = (metadata or "").strip()
        if user_meta:
            import json
            try:
                json.loads(user_meta)
            except ValueError as exc:
                raise ValueError(
                    "H3SaveVideoWithMCtx: metadata is not valid JSON (%s). "
                    "Leave it empty or pass a JSON object, e.g. "
                    "{\"prompt\": \"...\", \"seed\": 7}." % exc)

        specs_public = [dict(p.get("spec") or {}) for p in (pins or [])]
        relation, parent_id, join = mctx.summarize_pins(specs_public)
        grade = mctx.lineage_grade(specs_public)

        full_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                _save_prefix(base_folder, filename_prefix),
                folder_paths.get_output_directory(), width, height)
        file = "%s_%05d.mp4" % (filename, counter)
        video_path = os.path.join(full_folder, file)

        blobs = _workflow_blobs(prompt, extra_pnginfo)
        # container tags carry the WORKFLOW only: they are JSON strings,
        # and the overlap is binary. It rides in the sidecar instead.
        self._encode_mp4(video_path, images, audio, crf,
                         metadata=_workflow_tags(blobs))
        self_id = mctx.hash_file(video_path)
        # Kept for the GUIDED side only -- that is the one that was
        # re-rendered and therefore the only one a crossfade can blend.
        guided = guided_sides(pins)
        overlap = (_overlap_bytes(untrimmed_images, untrimmed_audio,
                                  head, crf)
                   if "before" in guided else None)
        overlap_tail = (_overlap_bytes(untrimmed_images, untrimmed_audio,
                                       tail, crf, tail=True)
                        if "after" in guided else None)
        sidecar_blobs = dict(blobs or {})
        if overlap:
            sidecar_blobs[OVERLAP_BLOB] = overlap
        if overlap_tail:
            sidecar_blobs[OVERLAP_TAIL_BLOB] = overlap_tail

        meta = {
            "format": mctx.FORMAT,
            "self_id": self_id,
            "parent_id": parent_id,
            "relation": relation,
            "parent_join_frame": str(join),
            "width": str(width),
            "height": str(height),
            "fps": str(fr.FPS),
            "raw_frames": str(raw_frames),
            "pinned_head_frames": str(head),
            "pinned_tail_frames": str(tail),
            "delivered_frames": str(delivered),
            "parent_grade": grade,
            "pins": mctx.serialize_pins(specs_public),
            "user_meta": user_meta,
            "overlap_frames": str(head if overlap else 0),
            "overlap_tail_frames": str(tail if overlap_tail else 0),
        }
        sidecar = mctx.write_sidecar(
            mctx.sidecar_path(video_path), video_lat, audio_lat, meta,
            blobs=sidecar_blobs)
        _save_conditioning(video_path, conditioning, self_id,
                           save_conditioning)
        _LOG.info("obvpm.h3: saved take %s (+ sidecar %s): %d delivered of "
                  "%d raw frames, %s%s%s", video_path,
                  os.path.basename(sidecar),
                  delivered, raw_frames,
                  relation or "root",
                  (" <- %s..." % parent_id[:12]) if parent_id else "",
                  (" (+%d frame overlap)" % head) if overlap else "")
        return (video_path,)

    @staticmethod
    def _encode_mp4(path, images, audio, crf, metadata=None):
        """Encode the delivered clip. THE encode site for the whole pack.

        Assembly bridges and preview builds come through here too, so the
        choice made here applies to everything that writes video.

        `metadata` (container tags) defaults to None precisely because of
        that: only a SAVED TAKE carries its workflow. Bridges, full
        previews and exports are derived files, and stamping a workflow
        onto each rebuild would add its weight to every preview for
        something that was never generated by that graph.

        An ffmpeg subprocess fed raw frames runs ~2.2x faster than
        encoding frame by frame in process (measured on 136 frames at
        1376x768: 3.8s -> 1.7s, same file size, byte-identical stream
        parameters). It stays an optional speed-up rather than a
        dependency: ComfyUI guarantees PyAV, not an ffmpeg binary, so a
        missing or failing binary quietly falls back.
        """
        exe = _ffmpeg_exe()
        if exe:
            try:
                _encode_with_ffmpeg(exe, path, images, audio, crf,
                                    metadata=metadata)
                return
            except Exception:
                _LOG.warning(
                    "obvpm.h3: ffmpeg encode failed (%s); falling back to "
                    "the in-process encoder", exe, exc_info=True)
                # never leave a half-written file for the hash to pair with
                try:
                    os.remove(path)
                except OSError:
                    pass
        _encode_with_pyav(path, images, audio, crf, metadata=metadata)


class H3SaveMCtxForVideo:
    """Sidecar-only writer: pair an mctx with a video saved by ANY node.

    The escape hatch for users who prefer their own video saver (VHS
    Video Combine for speed/format options, hardware encoders, etc.):
    wire that saver's output path here and this node hashes the written
    file and commits the sidecar next to it. The pairing hash is exactly
    as trustworthy as the all-in-one nodes; what this path cannot fully
    guarantee is that the video CONTENT came from `samples`, so it
    validates what the container cheaply reveals (resolution and frame
    count against the latent + pins) and refuses on mismatch.
    """

    CATEGORY = "obvpm/h3"
    FUNCTION = "save_sidecar"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sidecar_path",)
    DESCRIPTION = (
        "Writes ONLY the .mctx.safetensors sidecar, paired to a video "
        "some other node already saved (e.g. VHS Video Combine -- wire "
        "its filenames output here). Use this when you want a specific "
        "video saver; the all-in-one Save/TrimAndSave nodes remain the "
        "simplest guaranteed-consistent route. The video must be the "
        "DELIVERED (trimmed) clip."
    )
    OUTPUT_TOOLTIPS = ("Path of the written sidecar.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path of the saved video: absolute, or "
                               "relative to the output folder. Accepts a "
                               "VHS filenames output wired via any "
                               "to-string conversion; the LAST path is "
                               "used."}),
                "samples": ("LATENT", {
                    "tooltip": "The sampler's RAW output latent for this "
                               "take (same as the all-in-one Save)."}),
                "save_conditioning": ("BOOLEAN", {
                    "default": True, "tooltip": COND_SAVE_TOOLTIP}),
            },
            "optional": {
                "pins": (wt.PINS, {
                    "tooltip": "The resolved pins from H3MCtxApplyPins. "
                               "Unconnected = root clip."}),
                "metadata": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional provenance JSON, stored verbatim "
                               "as user_meta (see the all-in-one Save)."}),
                "conditioning": ("CONDITIONING", {"tooltip": COND_TOOLTIP}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_sidecar(self, video_path, samples, save_conditioning=True,
                     pins=None, metadata="",
                     prompt=None, extra_pnginfo=None, conditioning=None):
        path = self._resolve_path(video_path)
        video_lat, audio_lat = unpack_av(samples, name="samples")
        raw_steps = int(video_lat.shape[2])
        raw_frames = fr.pixel_frames(raw_steps)
        width = int(video_lat.shape[4]) * 16
        height = int(video_lat.shape[3]) * 16

        from .nodes_pins import pins_trim_totals
        head, tail = pins_trim_totals(pins)
        expect_delivered = raw_frames - head - tail

        vid_w, vid_h, vid_frames = self._probe(path)
        if (vid_w, vid_h) != (width, height):
            raise ValueError(
                "H3SaveMCtxForVideo: %s is %dx%d but the latent decodes to "
                "%dx%d. This video was not rendered from these samples."
                % (path, vid_w, vid_h, width, height))
        if vid_frames and vid_frames != expect_delivered:
            raise ValueError(
                "H3SaveMCtxForVideo: %s has %d frames but %d delivered "
                "frames were expected (%d raw - %d pinned head - %d pinned "
                "tail). Save the TRIMMED clip, and wire the same pins."
                % (path, vid_frames, expect_delivered, raw_frames, head, tail))
        if not vid_frames:
            _LOG.warning("obvpm.h3: could not determine %s's frame count; "
                         "skipping the delivered-frames check", path)

        user_meta = (metadata or "").strip()
        if user_meta:
            import json
            try:
                json.loads(user_meta)
            except ValueError as exc:
                raise ValueError(
                    "H3SaveMCtxForVideo: metadata is not valid JSON (%s)."
                    % exc)

        specs_public = [dict(p.get("spec") or {}) for p in (pins or [])]
        relation, parent_id, join = mctx.summarize_pins(specs_public)
        grade = mctx.lineage_grade(specs_public)
        self_id = mctx.hash_file(path)
        meta = {
            "format": mctx.FORMAT,
            "self_id": self_id,
            "parent_id": parent_id,
            "relation": relation,
            "parent_join_frame": str(join),
            "width": str(width),
            "height": str(height),
            "fps": str(fr.FPS),
            "raw_frames": str(raw_frames),
            "pinned_head_frames": str(head),
            "pinned_tail_frames": str(tail),
            "delivered_frames": str(expect_delivered),
            "parent_grade": grade,
            "pins": mctx.serialize_pins(specs_public),
            "user_meta": user_meta,
        }
        sidecar = mctx.write_sidecar(
            mctx.sidecar_path(path), video_lat, audio_lat, meta,
            blobs=_workflow_blobs(prompt, extra_pnginfo))
        _save_conditioning(path, conditioning, self_id,
                           save_conditioning)
        _LOG.info("obvpm.h3: paired sidecar %s to externally saved %s (%s)",
                  os.path.basename(sidecar), path, relation or "root")
        return (sidecar,)

    @staticmethod
    def _resolve_path(video_path):
        raw = video_path
        # tolerate VHS-style filenames structures stringified or wired
        if isinstance(raw, (tuple, list)):
            if len(raw) == 2 and isinstance(raw[1], (tuple, list)):
                raw = raw[1]
            raw = raw[-1] if raw else ""
        p = str(raw).strip().strip('"').strip("'")
        if not p:
            raise ValueError("H3SaveMCtxForVideo: video_path is empty.")
        candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise ValueError(
            "H3SaveMCtxForVideo: no file at %r (tried absolute and "
            "output-relative). Wire the path your video saver actually "
            "wrote." % p)

    @staticmethod
    def _probe(path):
        """(width, height, frame_count) from the container, cheaply."""
        import av
        with av.open(path) as container:
            stream = container.streams.video[0]
            w = int(stream.codec_context.width)
            h = int(stream.codec_context.height)
            frames = int(stream.frames or 0)
            if not frames and stream.duration and stream.average_rate:
                frames = int(round(
                    float(stream.duration * stream.time_base)
                    * float(stream.average_rate)))
        return w, h, frames


class H3TrimAndSaveVideoWithMCtx:
    """H3TrimPinned + H3SaveVideoWithMCtx in one node: the common case.

    Takes the UNTRIMMED decode output plus the pins wire, removes the
    pinned scaffolding itself, then runs the exact save transaction. The
    separate Trim/Save nodes remain for graphs that want the delivered
    frames mid-graph (preview, post-processing before save).
    """

    CATEGORY = "obvpm/h3"
    FUNCTION = "trim_and_save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("path", "images", "audio")
    DESCRIPTION = (
        "Trims the pinned scaffolding off a decoded H3 take and saves the "
        "clip pair (MP4 + mctx sidecar) in one step. Wire the RAW decode "
        "output (untrimmed images/audio), the sampler's raw latent, and "
        "the pins wire that fed Apply. For a root clip (no pins), it "
        "saves as-is."
    )
    OUTPUT_TOOLTIPS = (
        "Path of the written MP4; the sidecar sits next to it.",
        "The delivered (trimmed) frames, for preview.",
        "The delivered (trimmed) audio, for preview.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        base = H3SaveVideoWithMCtx.INPUT_TYPES()
        req = dict(base["required"])
        req["images"] = ("IMAGE", {
            "tooltip": "The decode output, UNTRIMMED -- this node removes "
                       "the pinned scaffolding itself using the pins wire."})
        opt = dict(base["optional"])
        opt["audio"] = ("AUDIO", {
            "tooltip": "The decoded audio, untrimmed. Trimmed here in lock "
                       "step with the frames and tail-matched to exactly "
                       "frames/fps."})
        # this node is HANDED the untrimmed decode, so the crossfade
        # overlap costs the user no extra wiring -- the separate
        # Trim/Save pair is where it has to be connected by hand
        opt.pop("untrimmed_images", None)
        opt.pop("untrimmed_audio", None)
        # Rebuilt from the base, so the hidden inputs have to be carried
        # over too -- without them ComfyUI passes neither and this node
        # would quietly save takes with no workflow in the sidecar.
        return {"required": req, "optional": opt, "hidden": dict(base["hidden"])}

    def trim_and_save(self, images, samples, base_folder, filename_prefix,
                      crf, save_conditioning=True,
                      audio=None, pins=None, metadata="", prompt=None,
                      extra_pnginfo=None, conditioning=None):
        from .nodes_pins import H3TrimPinned
        raw_images, raw_audio = images, audio
        if pins:
            images, audio = H3TrimPinned().trim(images, pins, audio=audio)
        (path,) = H3SaveVideoWithMCtx().save(
            images, samples, base_folder, filename_prefix, crf,
            save_conditioning=save_conditioning,
            audio=audio, pins=pins, metadata=metadata, prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            untrimmed_images=raw_images, untrimmed_audio=raw_audio,
            conditioning=conditioning)
        return (path, images, audio)
