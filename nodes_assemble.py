"""H3Assemble: sequence clips into one MP4, seams derived from sidecars.

The assembler's job is the last mile of the lineage system: the sidecar
headers already record who joins whom and at which raw frame, so given an
ORDERED list of clips this node derives every seam cut itself:

  R extends L   ->  L exits at (R.join_raw - L.pinned_head); R enters at 0.
                    A plain tail extend puts that exit exactly at L's
                    delivered end (no cut); a trim-point extension or the
                    auto-shifted extend of a prepend-made clip lands the
                    exit earlier, cutting the frames that belong to the
                    other seam.
  L prepends R  ->  L plays whole; R enters at (L.join_raw - R.pinned_head).
                    Prepend to a root enters at 0; prepend to a
                    continuation enters at the conserved 12-frame offset.
  no relation   ->  butt join (L whole, R from 0), with a warning.

Writing is smart-cut: every clip's played range [enter, exit) is split
into COPY runs -- packet spans that start at an IDR keyframe and end at
one (or at the clip end), stream-copied bit for bit -- and BRIDGE runs,
the sub-GOP remainders around a mid-clip seam, re-encoded from the
already-decoded frames with the same PyAV H.264 path the save nodes use.
A whole-clip sequence is therefore 100% copied (zero video generation
loss); a mid-clip seam re-encodes only the frames between the cut and the
nearest keyframe on its side. Since our saves use x264's default GOP
(~250 frames), short takes carry a single IDR at frame 0 and a mid-clip
entry re-encodes that one clip; longer takes with interior keyframes get
true sub-clip copying automatically.

Audio is decoded, concatenated sample-accurately and encoded ONCE as a
continuous AAC track regardless (copying AAC packets across a splice
always clicks: each frame overlap-adds with its neighbor, so the first
~21ms after a splice decodes with a fade-in artifact).

The whole path is defensive, because splicing packets from different
encodes is exactly where players glitch: every piece (bridges included)
must match the first clip's codec/extradata/dimensions/time base byte for
byte, frame counts are verified per piece, and DTS monotonicity is
enforced while muxing. Any surprise raises _StreamMismatch and the node
falls back to one full decode -> re-encode rather than writing a broken
file. The source takes remain the masters either way.

No sidecar is written: an assembly is a delivery, not a take -- it has no
latents of its own to store.
"""

import json
import logging
import os
import re
import shutil
import time

import folder_paths

from . import frames as fr
from . import wiretypes as wt
from . import levellock
from . import mctx
from . import nodes_load
from . import nodes_masked

_LOG = logging.getLogger("obvpm.h3")

# "clip.mp4", "clip.mp4 @ 12" (enter), "clip.mp4 @ 12..90" (enter..exit).
# The exit half is what a right-hand cut writes; either half may be blank
# ("@ ..90" cuts only the tail) so a cut on one side never has to invent
# a value for the other.
_LINE_RE = re.compile(
    r"^(?P<path>.+?)"
    r"(?:\s*@\s*(?P<enter>\d+)?(?:\s*\.\.\s*(?P<exit>\d+))?)?$")

# A GAP: "~ 40" reserves 40 delivered frames of empty space. Deliberately
# not a clip path, so an old reader fails loudly on the unknown line
# rather than silently dropping the hole.
_GAP_RE = re.compile(r"^~\s*(?P<frames>\d+)$")

# PER-SEAM SETTINGS ride at the end of a clip's line, in brackets:
#
#   clip.mp4 @ 12..250 [crossfade=off level_lock_frames=18]
#
# They belong to the clip on the RIGHT of a join -- the child -- which is
# how the assembly already keys its corrections and where the seam badge
# sits. An absent key inherits the timeline's own setting.
#
# WHY IN THE LINE. The alternative is a side table keyed by join index,
# and indices shift on every add, delete, reorder and gap change -- so a
# setting written for the join between A and B silently ends up on the
# one between C and D, with nothing to notice it by except the output
# being wrong. In the line, the override moves with the clip for free:
# reorder moves it, delete removes it, hand-editing shows it.
#
# The bracket section is stripped BEFORE the cut section is read, so the
# line regex above -- the one that has to stay in lockstep with the JS
# mirror or a cut silently becomes a different cut -- is untouched.
_OPTS_RE = re.compile(r"^(?P<head>.*?)\s*\[(?P<opts>[^\[\]]*)\]$")

SEAM_KEYS = {
    "level_lock": bool,
    "level_lock_frames": int,
    "level_lock_flicker": bool,
    "level_lock_local": bool,
    "crossfade": bool,
    "crossfade_frames": int,
    "audio_declick": bool,
}
# Written in THIS order, always, so formatting a parsed line reproduces
# it -- a round trip that depends on dict order is not a round trip.
SEAM_ORDER = ("level_lock", "level_lock_frames", "level_lock_flicker",
              "level_lock_local", "crossfade", "crossfade_frames",
              "audio_declick")
_TRUE = ("1", "on", "true", "yes")
_FALSE = ("0", "off", "false", "no")


def split_seam_opts(line):
    """(line without its [...] tail, {key: value}).

    Unknown keys and unreadable values are dropped with a warning rather
    than refused. A wrong SETTING degrades to the timeline's default; an
    unknown ENTRY would drop content, which is why "~" gaps fail loudly
    and this does not.
    """
    m = _OPTS_RE.match(line)
    if not m:
        return line, {}
    out = {}
    for token in m.group("opts").split():
        key, sep, raw = token.partition("=")
        kind = SEAM_KEYS.get(key)
        if not sep or kind is None:
            _LOG.warning("obvpm.h3: ignoring unknown seam setting %r in "
                         "%r", token, line)
            continue
        if kind is bool:
            low = raw.strip().lower()
            if low in _TRUE:
                out[key] = True
            elif low in _FALSE:
                out[key] = False
            else:
                _LOG.warning("obvpm.h3: %r is not on/off in %r", token,
                             line)
            continue
        try:
            out[key] = int(raw)
        except ValueError:
            _LOG.warning("obvpm.h3: %r is not a number in %r", token, line)
    return m.group("head").strip(), out


def format_seam_opts(opts):
    """The bracket section for `opts`, or "" when there is nothing to say."""
    if not opts:
        return ""
    parts = []
    for key in SEAM_ORDER:
        val = opts.get(key)
        if val is None:
            continue
        parts.append("%s=%s" % (
            key, ("on" if val else "off") if SEAM_KEYS[key] is bool
            else int(val)))
    return " [%s]" % " ".join(parts) if parts else ""


# The one directive a sequence may carry. `loop` makes the cut a RING:
# its last entry is joined to its first exactly as neighbours are, so
# the export ends where its own opening begins and a player set to
# repeat wraps onto continuous motion. It is a property of the cut and
# so lives in the cut's text, beside the clips and gaps it applies to
# -- a widget or property for it would be a second store for one fact.
LOOP_RE = re.compile(r"^loop$", re.IGNORECASE)
LOOP_LINE = "loop"


def sequence_loops(sequence):
    """Does the sequence text carry the `loop` directive?"""
    return any(LOOP_RE.match(raw.strip())
               for raw in (sequence or "").splitlines())


def _parse_sequence(sequence):
    """Lines of "clip.mp4", "clip.mp4 @ enter[..exit]", or "~ frames".

    The `loop` directive is not an entry and is skipped here; read it
    with `sequence_loops`. Any other line is a clip path, so a directive
    unknown to this parser would be looked up as a file -- which is the
    right failure: loud, and naming the line.
    """
    out = []
    for raw in (sequence or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or LOOP_RE.match(line):
            continue
        gap = _GAP_RE.match(line)
        if gap:
            out.append({"clip": None, "gap": int(gap.group("frames")),
                        "enter_override": None, "exit_override": None,
                        "seam_opts": {}})
            continue
        # settings first, so the cut regex sees the line it always saw
        line, seam = split_seam_opts(line)
        m = _LINE_RE.match(line)
        if m.group("enter") is None and m.group("exit") is None \
                and "@" in line:
            raise ValueError(
                "H3Assemble: %r has an empty @ cut. Write '@ 12', "
                "'@ 12..90' or '@ ..90'." % line)
        enter, exit_ = m.group("enter"), m.group("exit")
        if enter is not None and exit_ is not None and int(exit_) <= int(enter):
            raise ValueError(
                "H3Assemble: %r cuts to nothing (exit %s is not after "
                "enter %s)." % (line, exit_, enter))
        out.append({"clip": m.group("path").strip(), "gap": None,
                    "enter_override": None if enter is None else int(enter),
                    "exit_override": None if exit_ is None else int(exit_),
                    "seam_opts": seam})
    return out


def format_sequence_line(clip, enter=None, exit_=None, gap=None,
                         seam=None):
    """The inverse of _parse_sequence, so the UI and the parser agree.

    Kept here rather than in the widget: a round-trip that disagrees with
    the parser is how a cut silently becomes a different cut.
    """
    if gap is not None:
        return "~ %d" % int(gap)
    tail = format_seam_opts(seam)
    if enter is None and exit_ is None:
        return "%s%s" % (clip, tail)
    if exit_ is None:
        return "%s @ %d%s" % (clip, int(enter), tail)
    if enter is None:
        return "%s @ ..%d%s" % (clip, int(exit_), tail)
    return "%s @ %d..%d%s" % (clip, int(enter), int(exit_), tail)


def _read_verified_header(path):
    """The clip's header: its sidecar's when trustworthy, else synthetic.

    Never None any more. A plain video used to fall out of the seam logic
    entirely -- "no sidecar data: butt join" -- which meant a take pinned
    FROM imported footage was played after the whole import rather than
    from the frame it actually continues. The synthetic header carries
    identity and length and nothing else, which is exactly what the LEFT
    side of a join needs; the child still supplies the lineage.
    """
    from .nodes_load import _cached_hash, synthetic_header
    side = mctx.sidecar_path(path)
    if os.path.isfile(side):
        try:
            header = mctx.read_header(side)
            if header.get("self_id") == _cached_hash(path):
                return header
            _LOG.warning("obvpm.h3: %s does not pair with its sidecar; "
                         "assembling it by identity only (no latents)",
                         os.path.basename(path))
        except Exception:
            _LOG.exception("obvpm.h3: unreadable sidecar for %s; assembling "
                           "it by identity only", os.path.basename(path))
    try:
        return synthetic_header(path)
    except Exception:
        _LOG.exception("obvpm.h3: %s could not be probed; it joins butt",
                       os.path.basename(path))
        return None


def _handover(spec, place):
    """Frames of a recorded pin's window the TAKE delivers itself.

    The junction between two clips is not the window edge when the
    window was held softly -- see nodes_masked.handover_frames. Reads
    the recipe alone, so it answers the same for a take saved before
    these fields existed (they are absent, the shape is hard, and the
    answer is the window edge as it always was).
    """
    from .nodes_masked import handover_frames
    return handover_frames(
        int(spec.get("source_frames", 0) or 0), place,
        int(spec.get("mask_ramp_frames", 0) or 0),
        float(spec.get("mask_ramp_edge", 0.0) or 0.0),
        float(spec.get("mask_hold", 0.0) or 0.0))


def _extend_parent(header):
    """(parent_id, join_raw) of the clip this one EXTENDS, or None.

    relation/parent_id are the single-pin fast-path summary; a
    multi-pin take (a bridge, relation "") carries its parents only in
    the pins recipe, so fall through to it.
    """
    if header.get("relation") == "extends":
        return (header.get("parent_id"),
                int(header.get("parent_join_frame", 0) or 0))
    if header.get("relation"):
        return None
    for s in mctx.parse_pins(header):
        if (s.get("place") == "before" and
                s.get("source_kind") in mctx.LINEAGE_KINDS
                and s.get("source_id")):
            return (s["source_id"], int(s.get("source_start", 0) or 0)
                    + _handover(s, "before"))
    return None


def _prepend_child(header):
    """(child_id, join_raw) of the clip this one PREPENDS INTO, or None."""
    if header.get("relation") == "prepends":
        return (header.get("parent_id"),
                int(header.get("parent_join_frame", 0) or 0))
    if header.get("relation"):
        return None
    for s in mctx.parse_pins(header):
        if (s.get("place") == "after" and
                s.get("source_kind") in mctx.LINEAGE_KINDS
                and s.get("source_id")):
            return (s["source_id"], int(s.get("source_start", 0) or 0)
                    + _handover(s, "after"))
    return None


def _derive_seam(left, right):
    """(left_exit_delivered_frame_or_None, right_enter_frame, note).

    None exit = play the left clip to its delivered end.
    """
    lh, rh = left.get("header"), right.get("header")
    if not lh or not rh:
        return None, 0, "no sidecar data: butt join"
    l_id, r_id = lh.get("self_id"), rh.get("self_id")

    ex = _extend_parent(rh)
    if ex and ex[0] == l_id:
        join = ex[1]
        exit_f = join - int(lh.get("pinned_head_frames", 0) or 0)
        delivered = int(lh.get("delivered_frames", 0) or 0)
        if not 0 < exit_f <= delivered:
            raise ValueError(
                "H3Assemble: %s extends %s at raw frame %d, which maps "
                "outside the parent's %d delivered frames. The sidecars "
                "disagree with themselves; re-check the takes."
                % (right["clip"], left["clip"], join, delivered))
        note = ("extends: seamless" if exit_f == delivered else
                "extends at a cut: %s exits at frame %d (%d frames held "
                "back)" % (left["clip"], exit_f, delivered - exit_f))
        return (None if exit_f == delivered else exit_f), 0, note

    pr = _prepend_child(lh)
    if pr and pr[0] == r_id:
        join = pr[1]
        enter = join - int(rh.get("pinned_head_frames", 0) or 0)
        delivered = int(rh.get("delivered_frames", 0) or 0)
        if not 0 <= enter < delivered:
            raise ValueError(
                "H3Assemble: %s prepends %s at raw frame %d, which maps "
                "outside the target's %d delivered frames. The sidecars "
                "disagree with themselves; re-check the takes."
                % (left["clip"], right["clip"], join, delivered))
        note = ("prepends: seamless" if enter == 0 else
                "prepends at the conserved offset: %s enters at frame %d"
                % (right["clip"], enter))
        return None, enter, note

    return None, 0, "no recorded relation: butt join"


# Which mode a pin rides in follows the ROLE, and is not a taste.
#
# An EXTEND continues FORWARD from a fixed window. Masked preserves that
# window verbatim in one rendering, so the join is latent-identical --
# seamless, and nothing to repair.
#
# A PREPEND has to ARRIVE at a clip that already exists. A masked suffix
# fixes the destination latents but gives the model no reason to steer
# toward them: it wanders and the pinned tail then appears as a jump
# (measured 2026-08-25). Arriving somewhere is what the model's own
# keyframe conditioning is trained for, and that is what guided rides
# on -- native `minimax_keyframes` at a resolved frame index.
#
# A BRIDGE is one of each, so it takes them per side.
#
# Measured 2026-08-27 (clip_00080-84, the clip_00024 -> clip_00063
# bridge): "both" -- keyframe cond rows AND the mask, from the same
# slice -- is the only arriving mode that has produced a genuinely
# seamless join (83, 84; frame-by-frame sweep, not just the boundary
# step). The rows give the model something to steer toward, the mask
# makes the arrival exact. So arriving sides default to "both".
PIN_MODE_FOR_ROLE = {"extend": "masked", "prepend": "both"}
# The one role where the mode is still a choice: "both" is the default
# and the one that has converged, but guide/masked stay selectable so
# the pairing can be re-tested rather than taken on faith.
PIN_MODE_CHOOSABLE = frozenset({"prepend"})
BRIDGE_MODES = {"extend": "masked", "prepend": "both"}


def _shape_note(shape):
    """Human phrase for a soft mask hold, for the log lines."""
    ramp, edge, hold = shape
    if ramp:
        return (" held softly: mask %.2f at the join -> %.2f over %d frames"
                % (float(edge), float(hold), int(ramp)))
    return " held softly: mask %.2f across the whole window" % float(hold)


def pin_mode_for(role):
    """The mode a pin of this role must ride in."""
    return PIN_MODE_FOR_ROLE.get(role, "masked")


def masked_continuation(lh, rh):
    """True when the join between these two headers rides a masked recipe.

    A masked child's opening decodes from the parent's OWN latents, so
    the two sides of the join are latent-identical: there is no level
    step to correct and no second rendering to fade between. Seam
    repairs on such a join operate on noise -- measured live 2026-08-20
    as INTRODUCING flicker -- so the plan layer uses this to default
    them off (an explicit per-seam override still wins).
    """
    def pins(h):
        try:
            return json.loads(h.get("pins") or "[]")
        except ValueError:
            return []
    if not lh or not rh:
        return False
    l_id, r_id = lh.get("self_id"), rh.get("self_id")
    # "both" masks the window as well, so its join is latent-identical
    # for the same reason -- the keyframe rows change what the model
    # steers toward, not what the pinned frames end up being
    held = ("masked", "both")

    def hard(p):
        # A ramped window still meets its neighbour at a FULLY held
        # boundary: the take ships the ramped frames and the source
        # enters past them, so the frames either side of the cut decode
        # from the same latents. That is the property this test is
        # really asking about, so a ramp does not disqualify a join --
        # only a window with no fully held region at all does, because
        # then there is no frame the two sides agree on exactly.
        if p.get("mode") not in held:
            return False
        from .nodes_masked import holds_exactly
        return holds_exactly(
            int(p.get("source_frames", 0) or 0), p.get("place"),
            int(p.get("mask_ramp_frames", 0) or 0),
            float(p.get("mask_ramp_edge", 0.0) or 0.0),
            float(p.get("mask_hold", 0.0) or 0.0))

    if any(hard(p) and p.get("place") == "before"
           and p.get("source_id") == l_id for p in pins(rh)):
        return True
    return any(hard(p) and p.get("place") == "after"
               and p.get("source_id") == r_id for p in pins(lh))


class EmptySequence(ValueError):
    """The sequence names no clips at all.

    Its own type because the two audiences differ. To the Assemble node
    an empty sequence IS a mistake -- it was asked to build something
    out of nothing. To the timeline widget it is just the state of a
    strip with nothing on it yet, and the widget asks about the
    sequence it has on every repaint rather than checking first. The
    routes catch this separately so an empty strip stays quiet instead
    of filling the console with tracebacks for a non-event.
    """


def resolve_sequence(sequence):
    """sequence text -> entries with path/header/enter/exit, seams derived.

    Shared by the Assemble node and the preview route; the traversal
    guard matters for the latter (the route accepts arbitrary text).
    """
    root = os.path.abspath(folder_paths.get_output_directory())
    entries = _parse_sequence(sequence)
    if not entries:
        raise EmptySequence(
            "H3Assemble: the sequence is empty. List one clip per "
            "line, in playback order.")
    for e in entries:
        if e.get("gap"):
            e["path"] = None
            e["header"] = None
            continue
        e["path"] = os.path.abspath(os.path.join(root, e["clip"]))
        if not e["path"].startswith(root + os.sep):
            raise ValueError(
                "H3Assemble: %s escapes the output folder" % e["clip"])
        if not os.path.isfile(e["path"]):
            raise ValueError("H3Assemble: clip not found: %s" % e["clip"])
        e["header"] = _read_verified_header(e["path"])

    # seams: enter[i] from the join with the left neighbor, exit[i]
    # from the join with the right neighbor (None = delivered end)
    for e in entries:
        e["enter"], e["exit"] = 0, None
    # A gap has no lineage, so it breaks the chain: the clips either side
    # of it keep their own ends rather than being joined to each other.
    clips = [e for e in entries if not e.get("gap")]
    for left, right in zip(entries, entries[1:]):
        if left.get("gap") or right.get("gap"):
            continue
        exit_f, enter_f, note = _derive_seam(left, right)
        if left["exit"] is None:
            left["exit"] = exit_f
        right["enter"] = enter_f
        _LOG.debug("obvpm.h3: seam %s | %s -- %s",
                  left["clip"], right["clip"], note)
    # A looping cut has one more join: its last entry into its first,
    # derived by the same rule. A loop take is a bridge whose arriving
    # side is entry 0, so its recipe carries an after-pin on that clip
    # and _prepend_child finds it: the take exits where clip 0's kept
    # frames begin, and clip 0 enters past the frames the take delivers
    # itself. A single clip with no take yet simply butt-joins itself.
    if sequence_loops(sequence):
        first, last = entries[0], entries[-1]
        if first.get("gap") or last.get("gap"):
            raise ValueError(
                "H3Assemble: a looping sequence cannot start or end on "
                "empty space -- a gap has nothing to wrap onto. Remove "
                "the `loop` line or the gap at that end.")
        exit_f, enter_f, note = _derive_seam(last, first)
        if last["exit"] is None:
            last["exit"] = exit_f
        first["enter"] = enter_f
        _LOG.debug("obvpm.h3: wrap seam %s | %s -- %s",
                  last["clip"], first["clip"], note)
    # Manual cuts win over derived seams -- they ARE the user's decision.
    # Logged at DEBUG: resolve_sequence runs on every preview probe, so
    # at INFO a single drag buries the console.
    for e in clips:
        if e["enter_override"] is not None:
            e["enter"] = e["enter_override"]
            _LOG.debug("obvpm.h3: %s enters at frame %d (manual cut)",
                       e["clip"], e["enter"])
        if e["exit_override"] is not None:
            e["exit"] = e["exit_override"]
            _LOG.debug("obvpm.h3: %s exits at frame %d (manual cut)",
                       e["clip"], e["exit"])
    return entries


# x264 stamps its full option string into an SEI on the first keyframe.
# The crf is in there, which is the only place a finished clip records
# the quality it was written at -- our sidecars never stored it.
_X264_SEI = re.compile(rb"x264 - core \d+.{0,4000}?\bcrf=([0-9.]+)", re.S)
_CRF_CACHE = {}


def source_crf(path, default=None):
    """The crf a clip was encoded at, or `default` if it cannot be read.

    A bridge is spliced between stream-copied packets of the same clip,
    and H.264 puts pic_init_qp -- derived from the target quality -- in
    the PPS. So a bridge written at a different crf has different codec
    extradata, _mux_pieces refuses the splice, and the whole assembly
    falls back to re-encoding every frame in order to repair twelve.

    Measured: takes were written at one crf while previews defaulted to
    another, so the smart cut was being lost on every repaired sequence
    -- silently, because the fallback still produces a correct file. The
    SPS matched; only the PPS differed. This is why the defaults were
    later moved together (to 19, 2026-08-28) rather than one at a time:
    a mismatch here costs quality on every frame of the assembly, not
    just the repaired ones. Older takes carry their own crf and are
    matched individually, so a folder may hold several and still splice.

    Matching the SOURCE rather than the caller's request is also the
    right output: every other frame of that clip is copied at its
    original quality, so a bridge at a different crf would be a quality
    step inside one clip.

    A non-integral crf returns `default`: the pack's encoders take an
    int, so we could not reproduce it anyway and guessing would put the
    mismatch back.
    """
    import av
    try:
        key = (os.path.abspath(path), os.stat(path).st_mtime_ns)
    except OSError:
        return default
    if key in _CRF_CACHE:
        got = _CRF_CACHE[key]
        return default if got is None else got
    blob = b""
    try:
        with av.open(path) as c:
            v = c.streams.video[0]
            for pkt in c.demux(v):
                if pkt.dts is None:
                    continue
                blob += bytes(pkt)
                # the stamp rides on the first keyframe; a couple of
                # packets is plenty and we do not want the whole file
                if len(blob) > 1 << 20:
                    break
                if len(blob) and pkt.is_keyframe and len(blob) > 4096:
                    break
    except Exception:
        _LOG.debug("obvpm.h3: could not probe the crf of %s", path,
                   exc_info=True)
        blob = b""
    got = None
    m = _X264_SEI.search(blob)
    if m:
        try:
            val = float(m.group(1))
            if abs(val - round(val)) < 1e-6:
                got = int(round(val))
        except ValueError:
            got = None
    if len(_CRF_CACHE) > 256:
        _CRF_CACHE.clear()
    _CRF_CACHE[key] = got
    return default if got is None else got


class _StreamMismatch(Exception):
    """Pieces cannot be spliced losslessly; fall back to full re-encode."""


def _video_stream_info(container, path):
    """(stream, pts_step, base_pts) with CFR sanity checks."""
    v = container.streams.video[0]
    rate = v.average_rate or fr.FPS
    step = int(round((1 / rate) / v.time_base))
    if step <= 0:
        raise _StreamMismatch("cannot derive a frame step from %s"
                              % os.path.basename(path))
    return v, step


def _scan_keyframes(path):
    """(keyframe frame indices, total frames) of the clip's video stream.

    Frame indices are display positions derived from pts on the CFR grid;
    anything that does not divide evenly is not a clip our encoder wrote.
    """
    import av
    with av.open(path) as c:
        v, step = _video_stream_info(c, path)
        pts = []
        kf_pts = []
        for packet in c.demux(v):
            if packet.dts is None:
                continue
            pts.append(packet.pts)
            if packet.is_keyframe:
                kf_pts.append(packet.pts)
    if not pts:
        raise _StreamMismatch("no video packets in %s"
                              % os.path.basename(path))
    base = min(pts)
    kfs = []
    for p in sorted(kf_pts):
        if (p - base) % step:
            raise _StreamMismatch("%s is not on a uniform frame grid"
                                  % os.path.basename(path))
        kfs.append((p - base) // step)
    return kfs, len(pts)


def _plan_pieces(enter, exit_f, keyframes, n_total):
    """Split [enter, exit) into ("copy"|"bridge", from, to) runs.

    Copy runs must START at an IDR (decoding joins the stream there) and
    STOP at one (dropping a closed GOP's tail packets is only safe at the
    next IDR boundary) -- or at the clip end. Everything else bridges.
    """
    end = n_total if exit_f is None else exit_f
    kin = next((k for k in keyframes if k >= enter), None)
    if end == n_total:
        kout = n_total
    else:
        kout = max((k for k in keyframes if k <= end), default=None)
    if kin is None or kout is None or kin >= kout:
        return [("bridge", enter, end)]
    pieces = []
    if kin > enter:
        pieces.append(("bridge", enter, kin))
    pieces.append(("copy", kin, kout))
    if kout < end:
        pieces.append(("bridge", kout, end))
    return pieces


# ---------------------------------------------------------------------------
# The preview and the export both land on a STABLE filename, and on Windows
# that filename is regularly held open by someone else at the exact moment a
# build finishes: the browser is still streaming the PREVIOUS preview through
# /api/view, or ComfyUI's own asset indexer is walking the output folder (its
# "Indexing changed assets" line lands in the log right beside these
# failures). Such a handle does not block WRITING the file -- only renaming
# or deleting it -- so os.replace raises
#
#     PermissionError: [WinError 5] Access is denied
#
# while an in-place overwrite of the same bytes goes straight through.
#
# So: try the atomic rename, wait out a holder that is merely passing by,
# and only then fall back to copying the bytes over the target. The fallback
# is NOT atomic -- a reader mid-stream sees a mix of both files -- which is
# why it is the last resort rather than the first move. The client reloads
# with a fresh cache key immediately afterwards, so the mix is never what
# anyone ends up watching.
REPLACE_WAITS = (0.15, 0.3, 0.6, 1.2)      # 2.25s of patience in total


def replace_when_free(tmp, dst, waits=REPLACE_WAITS):
    """os.replace(tmp, dst), tolerating another process's handle on `dst`.

    Returns "renamed" or "overwritten". `tmp` is gone either way.
    """
    held = None
    for i, wait in enumerate((0.0,) + tuple(waits)):
        if wait:
            time.sleep(wait)
        try:
            os.replace(tmp, dst)
            if i:
                _LOG.info("obvpm.h3: %s was held open; the rename went "
                          "through after %.2fs",
                          os.path.basename(dst), sum(waits[:i]))
            return "renamed"
        except PermissionError as exc:
            held = exc
        except OSError:
            raise
    # Still held after all that. Overwriting in place is what the holder's
    # share mode does permit, and it is preferable to losing the build.
    try:
        with open(tmp, "rb") as src, open(dst, "r+b") as out:
            shutil.copyfileobj(src, out)
            out.truncate()
    except OSError as exc:
        raise PermissionError(
            "%s is open in another program and could not be replaced (%s). "
            "Stop whatever is playing or scanning it -- often the preview "
            "still running in another tab -- and build again."
            % (os.path.basename(dst), exc)) from held
    try:
        os.remove(tmp)
    except OSError:
        pass                                # a scanner has it; it is spare
    _LOG.warning("obvpm.h3: %s stayed open through %.2fs of retries; wrote "
                 "it in place instead of renaming over it",
                 os.path.basename(dst), sum(waits))
    return "overwritten"


def _mux_pieces(pieces, out_path, audio):
    """Splice piece runs into one MP4: copy packets, one audio encode.

    pieces: [{"path", "from", "to"}] -- "from"/"to" are frame indices
    into that file; bridges arrive as whole little files (from 0 to
    their length). Every piece must share the first piece's stream
    signature; per-piece frame counts and DTS monotonicity are verified,
    and any violation raises _StreamMismatch (caller falls back).
    """
    import av

    def sig(c):
        v = c.streams.video[0]
        cc = v.codec_context
        return (cc.name, bytes(cc.extradata or b""), cc.width,
                cc.height, v.time_base)

    with av.open(out_path, mode="w") as out:
        vout = aout = layout = None
        ref = None
        offset = 0
        last_dts = None
        ref_lead = None
        for piece in pieces:
            with av.open(piece["path"]) as c:
                if ref is None:
                    ref = sig(c)
                    vout = out.add_stream_from_template(
                        template=c.streams.video[0])
                    if audio is not None:
                        wave = audio["waveform"]
                        wave = wave[0] if wave.dim() == 3 else wave
                        layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(
                            int(wave.shape[0]), "stereo")
                        aout = out.add_stream(
                            "aac", rate=int(audio["sample_rate"]),
                            layout=layout)
                elif sig(c) != ref:
                    raise _StreamMismatch(
                        "%s was encoded with different stream parameters"
                        % os.path.basename(piece["path"]))

                v, step = _video_stream_info(c, piece["path"])
                base = None
                shift = None
                dts_fix = 0
                copying = piece["from"] == 0
                want = piece["to"] - piece["from"]
                n = 0
                for packet in c.demux(v):
                    if packet.dts is None:
                        continue
                    if base is None:
                        base = packet.pts  # first packet is the first IDR
                    frame = (packet.pts - base) // step
                    if packet.is_keyframe:
                        if not copying and frame == piece["from"]:
                            copying = True
                        elif copying and frame >= piece["to"]:
                            break
                    if not copying:
                        continue
                    if shift is None:
                        # rebase the run so it looks like a fresh file:
                        # first display frame at pts 0, dts keeping its
                        # native B-frame lead
                        shift = base + piece["from"] * step
                        # ...but that LEAD has to match across pieces.
                        # A copied take carries two B-frames of reorder
                        # delay, so its first DTS sits two frames BEFORE
                        # its first PTS. A re-encoded bridge of two
                        # frames has no room for a two-frame reorder and
                        # starts at DTS 0 instead, which leaves it
                        # occupying DTS the NEXT copied run still wants:
                        # that run steps back behind the bridge, the
                        # monotonic check below refuses, and a cut that
                        # could have been copied gets fully re-encoded.
                        #
                        # Normalising each run to the FIRST run's lead
                        # fixes the seam between conventions. It only
                        # ever moves DTS earlier (max(0, ...)), so
                        # dts <= pts still holds for every packet, and a
                        # run with a longer lead than the reference is
                        # left alone -- the check still guards it.
                        lead = shift - packet.dts
                        if ref_lead is None:
                            ref_lead = lead
                        dts_fix = max(0, ref_lead - lead)
                    packet.stream = vout
                    packet.pts += offset - shift
                    packet.dts += offset - shift - dts_fix
                    if last_dts is not None and packet.dts <= last_dts:
                        raise _StreamMismatch(
                            "non-monotonic DTS at the splice into %s"
                            % os.path.basename(piece["path"]))
                    last_dts = packet.dts
                    out.mux(packet)
                    n += 1
                if n != want:
                    raise _StreamMismatch(
                        "expected %d frames from %s [%d..%d), muxed %d"
                        % (want, os.path.basename(piece["path"]),
                           piece["from"], piece["to"], n))
                offset += want * step

        if aout is not None:
            frame = av.AudioFrame.from_ndarray(
                wave.float().cpu().contiguous().numpy(),
                format="fltp", layout=layout)
            frame.sample_rate = int(audio["sample_rate"])
            frame.pts = 0
            out.mux(aout.encode(frame))
            out.mux(aout.encode())


class H3Timeline:
    """The timeline hub: composes, previews, exports -- and sources pins.

    All the composing work happens through the timeline widget
    (web/h3_mctx_ui.js) and the stateless routes (preview_route.py): the
    full preview is a server-built smart cut, and the export button
    promotes that exact file into output/. The node is not an output
    node, so queueing a generation while composing costs nothing.

    The one thing that CAN execute is the `pin_specs` output: pin a clip
    in the widget (extend its tail / prepend its head) and wire the
    output to H3MCtxApplyPins, and the timeline becomes the source of the
    next generation's continuation -- no loader node, no MCTX wire. The
    pin lives in the hidden `pin_state` JSON widget (owned by the UI, one
    opaque channel so richer pin shapes never change the node's
    signature); the clip is re-verified against its sidecar hash at run
    time, exactly like H3LoadMCtx. With no pin the node emits an empty
    spec list, which Apply treats as plain root generation.

    For an in-graph assembled result (feed frames onward in one run),
    use H3 MCtx Assemble instead.
    """

    # UI pin roles -> the loaders' create_pins convenience modes; the
    # timeline reuses that exact pipeline (off-grid auto-shift included).
    _PIN_ROLES = {
        "extend": "extend (pin tail)",
        "prepend": "prepend (pin head)",
    }

    CATEGORY = "obvpm/h3"
    FUNCTION = "emit"
    RETURN_TYPES = (wt.PINSPECS, "INT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("pin_specs", "length", "sequence", "upscaling")
    OUTPUT_TOOLTIPS = (
        "Pin spec for the clip pinned in the timeline widget (extend/"
        "prepend); empty when nothing is pinned. Wire to H3MCtxApplyPins "
        "to generate the next clip straight from the timeline.",
        "duration_seconds as a frame count, snapped to the grid THIS "
        "pin needs -- the shared AV grid when the arrival masks (so the "
        "audio join stays exact), the ordinary 17k+5 ladder otherwise. "
        "Wire to the empty AV latent's length.",
        "The clip list exactly as the widget holds it, so a pass that "
        "walks the timeline reads the SAME list the cut is built from "
        "rather than a copy that has to be kept in step. Wire to H3 "
        "Join Latents' sequence. Verbatim, comments and cut "
        "markers included -- readers strip what they do not need.",
        "The toolbar's upscale toggle, on a wire: true when a Run should "
        "refine and render the timeline, false when it should generate "
        "the next take. Drive a Mute If gate on each branch from here "
        "(the generation branch muted when this is true, the refine "
        "branch when it is false, through a Not node) so one switch "
        "decides and the branches cannot disagree.",
    )
    DESCRIPTION = (
        "Timeline editor for composed clips: seam-aware blocks, "
        "drag-reorder, seamless server-built preview, and an export "
        "button that writes the current cut to the output folder "
        "(byte-identical to the full preview). Previewing and exporting "
        "are button-driven and never run in a workflow; the pin_specs "
        "output makes the timeline the SOURCE of the next generation -- "
        "pin a clip's tail or head in the widget and wire it to "
        "H3MCtxApplyPins."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "One clip per line, in playback order, as "
                               "output-relative paths (the same values the "
                               "H3 loaders list), e.g. h3/clip_00001.mp4. "
                               "Append ' @ N' to force a clip to enter at "
                               "delivered frame N instead of the derived "
                               "seam. Lines starting with # are ignored."}),
                "base_folder": ("STRING", {
                    "default": "project1",
                    "tooltip": "Output-relative folder this node works "
                               "in: scopes the clip picker and seam "
                               "suggestions, and holds the preview file "
                               "and exports. Empty = the output "
                               "root."}),
                "preview_filename": ("STRING", {
                    "default": "obvpm_h3_preview",
                    "tooltip": "Name (no extension) of the single "
                               "preview file, written into base_folder "
                               "and overwritten on every full build. "
                               "Give each Timeline node its own name if "
                               "you use several."}),
                "export_filename_prefix": ("STRING", {
                    "default": "full",
                    "tooltip": "Filename prefix for the export button; "
                               "written into base_folder with the usual "
                               "counter, like core save nodes."}),
                "crf": ("INT", {
                    "default": 19, "min": 0, "max": 51,
                    "tooltip": "H.264 quality for re-encoded seam bridges "
                               "and the export (lower = better, bigger). "
                               "Mostly a FALLBACK: a bridge is written at "
                               "the crf its own clip was encoded at, so "
                               "it splices without a quality step (see "
                               "source_crf). This applies where that "
                               "cannot be read."}),
                # DEPRECATED and inert. The timeline no longer watches for
                # new clips -- the Result Preview offers them via its
                # "+ add to timeline" button. Kept declared, and hidden by
                # the widget, ONLY because widgets_values is positional:
                # deleting it here would shift pin_state and
                # restore_groups on every saved Timeline node.
                "auto_add": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Deprecated and unused. New takes are "
                               "offered by the Result Preview's '+ add to "
                               "timeline' button instead."}),
            },
            "optional": {
                # Owned by the timeline widget (hidden there): the pin
                # selection as one JSON blob, e.g. {"source":
                # "h3/clip_00012.mp4", "role": "extend", "window": 22}.
                # One opaque channel instead of per-field widgets so
                # future pin shapes (keyframes, multi-pin) never touch
                # the node signature.
                "pin_state": ("STRING", {"default": ""}),
                # Declared LAST on purpose: widgets_values is positional,
                # so a new widget may only be appended or every saved
                # Timeline shifts by one.
                "restore_groups": ("STRING", {
                    "default": "[restore]",
                    "tooltip": "Keyword for the 'load settings' button on "
                               "the selected clip: settings are applied "
                               "only to nodes inside GROUPS whose name "
                               "contains this text (case-insensitive; "
                               "comma-separate for several). Tag a group "
                               "by putting [restore] in its title. Empty "
                               "restores nothing, so the button cannot "
                               "fire by accident. Widget values and "
                               "bypass/mute state only -- nothing is "
                               "rewired or created, top-level nodes "
                               "only, and result previews are always "
                               "skipped."}),
                "skip_restore_nodes": ("STRING", {
                    "default": "[skip]",
                    "tooltip": "Nodes whose TITLE contains this text are "
                               "left alone by 'load settings', even "
                               "inside a matching group (case-"
                               "insensitive; comma-separate for "
                               "several). Rename a node to include "
                               "[skip] to pin its current values. Empty "
                               "skips nothing."}),
                # Declared LAST (see the note above): widgets_values is
                # positional, so new widgets may only be appended.
                "snap_cuts_to_grid": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "How a dragged seam behaves. ON: the cut "
                               "snaps to H3's 17-frame latent grid "
                               "(~0.7s steps), so extending or "
                               "prepending from it is always exact. OFF: "
                               "frame-accurate cuts for playback and "
                               "export; a pin taken from an off-grid cut "
                               "is shifted to the nearest latent-grade "
                               "frame and the log says where the seam "
                               "landed."}),
                "level_lock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Match each clip's opening to the clip it "
                               "continues. A take opens at a different "
                               "level than its parent -- measured at ~6x "
                               "the frame-to-frame noise -- and settles "
                               "back over about half a second; the step "
                               "is what you see. This corrects the "
                               "opening and decays to nothing by "
                               "level_lock_frames, so nothing "
                               "accumulates down a chain. Affects the "
                               "full preview and the export."}),
                "level_lock_frames": ("INT", {
                    "default": levellock.FRAMES, "min": 2, "max": 120,
                    "tooltip": "Frames the correction spans (24 = 1s). The "
                               "default 12 is where measured transients "
                               "have decayed. Longer flattens real "
                               "lighting changes; shorter reads as a "
                               "ramp."}),
                "level_lock_flicker": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also straighten a wobble in those opening "
                               "frames. Some joins do not step at all but "
                               "oscillate for ~6 frames, which reads as "
                               "flicker; this puts them on the trend line "
                               "they should have followed."}),
                "crossfade": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fade the join across the overlap both "
                               "takes rendered, instead of cutting at it. "
                               "A take re-renders its parent's last "
                               "frames before continuing; when that "
                               "re-render was saved (see the Save node's "
                               "untrimmed_images) the join becomes a "
                               "handover between two renderings of the "
                               "SAME moment rather than a cut between "
                               "two different ones. Takes saved without "
                               "it are simply not crossfaded."}),
                "crossfade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 240,
                    "tooltip": "0 = fade across the whole stored overlap, "
                               "which is what the measurements support: "
                               "the two renderings disagree almost as "
                               "much at the overlap's start as its end, "
                               "so a short fade ramps part of the step "
                               "and cuts the rest. Lower only to taste."}),
                "audio_declick": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "At every boundary the audio crossfade does "
                               "NOT cover, taper 5ms out of the outgoing "
                               "clip and 5ms into the incoming one. On a "
                               "MASKED or BOTH route that is every "
                               "boundary in the sequence: only a guided "
                               "window is re-rendered, so only a guided "
                               "join has a second recording of the same "
                               "moment to fade across (see "
                               "nodes_save.guided_sides). Even on a "
                               "guided route it is still most of them -- "
                               "a cut moves the join away from where the "
                               "overlap was rendered, and butt joins and "
                               "the edges of empty space never had one. "
                               "It removes the CLICK -- the one-sample "
                               "jump at the join -- but cannot hide a "
                               "change in room tone, which needs real "
                               "overlapping material. The cut's own "
                               "start and end are left alone."}),
                # APPENDED LAST: widgets_values is positional, so a new
                # widget may only go on the end or every saved timeline
                # shifts.
                "level_lock_local": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Let the level lock correct different parts "
                               "of the picture by different amounts, "
                               "instead of the whole frame by one. Needs "
                               "the crossfade, because the per-region "
                               "measurement only exists where both takes "
                               "rendered the same moment: the fade "
                               "measures it there and the lock carries it "
                               "forward, fading to whole-frame across "
                               "level_lock_frames. Measured on the one "
                               "join whose takes both carry the data, it "
                               "removed 39% of the per-region difference "
                               "at the first delivered frame and 31% at "
                               "the second, without moving the overall "
                               "level (0.085 luma) or softening detail. "
                               "Joins with nothing stored to measure "
                               "from are simply corrected whole-frame, "
                               "as before."}),
                # Declared LAST (see the note above): widgets_values is
                # positional, so new widgets may only be appended.
                "duration_seconds": ("FLOAT", {
                    "default": 8.0, "min": 0.2, "max": 150.0, "step": 0.1,
                    "tooltip": "How long the NEXT generation should be. "
                               "Wire the node's `length` output to the "
                               "empty AV latent (or the r2v node's "
                               "length): seconds is the unit you think "
                               "in, frames is what the model takes, and "
                               "the node converts and snaps between "
                               "them. The snap depends on the pin: a "
                               "masked or both arrival needs the shared "
                               "AV grid (39/90/141/192/243 frames = "
                               "1.63/3.75/5.88/8.00/10.13 s), the only "
                               "lengths whose audio ticks come out whole "
                               "-- off it a prepend clicks. Anything "
                               "else takes the ordinary 17k+5 ladder "
                               "(~0.7 s steps)."}),
                # Declared LAST (see the note above): widgets_values is
                # positional, so new widgets may only be appended.
                "upscaling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Which half of the workflow this Run is for. "
                               "Off: generate the next take from the pin. "
                               "On: refine the whole timeline as one piece "
                               "and render the finished cut. Flipped from "
                               "the strip's upscale toggle; comes out on "
                               "the upscaling output for the Mute If gates.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, pin_state="", **_):
        # Re-run when the pin selection OR the pinned file(s) change
        # (same clip name, new content after a re-save).
        state = cls._parse_pin_state(pin_state)
        if state is None:
            return ""
        sig = pin_state
        for key in ("source", "source2"):
            if not state.get(key):
                continue
            path = os.path.join(folder_paths.get_output_directory(),
                                state[key])
            try:
                st = os.stat(path)
                sig += ":%d:%d" % (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        return sig

    @classmethod
    def _parse_pin_state(cls, pin_state):
        """None when nothing is pinned; {source, role, window} otherwise.

        Forgiving on the empty shapes the widget legitimately produces
        (missing input on old workflows, "", "{}", role "none"), loud on
        everything else -- a malformed blob means the UI and backend
        disagree, which must never pass silently as "no pin".
        """
        text = (pin_state or "").strip()
        if not text:
            return None
        try:
            state = json.loads(text)
        except ValueError:
            raise ValueError(
                "H3Timeline: pin_state is not valid JSON (%r). Clear the "
                "pin in the timeline widget or fix the workflow." % text)
        if not isinstance(state, dict):
            raise ValueError(
                "H3Timeline: pin_state must be a JSON object, got %r."
                % text)
        role = state.get("role") or "none"
        if role == "none":
            return None
        if role != "bridge" and role not in cls._PIN_ROLES:
            raise ValueError(
                "H3Timeline: unknown pin role %r (this build understands "
                "%s -- workflow saved by a newer version?)."
                % (role, "/".join(sorted(cls._PIN_ROLES)) + "/bridge"))
        source = str(state.get("source") or "")
        if not source:
            raise ValueError(
                "H3Timeline: pin role %r has no source clip." % role)
        window = str(state.get("window") or "22")
        if window not in fr.WINDOW_CHOICES:
            raise ValueError(
                "H3Timeline: pin window %r is not one of %s."
                % (window, "/".join(fr.WINDOW_CHOICES)))
        # The mode comes from the ROLE (see PIN_MODE_FOR_ROLE), not from
        # the state: a stored one is from before the rule existed, or
        # from a hand-edited workflow, and honouring it would reinstate
        # exactly the combination that does not work. A bridge's two
        # sides differ, so its state carries the before side's mode and
        # emit() picks per side.
        pin_mode = pin_mode_for("extend" if role == "bridge" else role)
        stored = state.get("mode")
        if (role in PIN_MODE_CHOOSABLE
                and stored in ("guide", "masked", "both")):
            pin_mode = stored          # a real choice, honoured
        elif stored and stored != pin_mode and role != "bridge":
            _LOG.info("obvpm.h3: pin_state says mode %r for a %s; using %r "
                      "-- the mode follows the role now", stored, role,
                      pin_mode)
        if pin_mode in ("masked", "both"):
            from . import nodes_masked
            if not nodes_masked.masked_window_ok(int(window)):
                _LOG.warning(
                    "obvpm.h3: pin window %s is not on the shared AV grid; "
                    "masked continuation uses 39 instead", window)
                window = "39"
        # A bridge has two sides and therefore two modes. The departing
        # one is settled (masked extends are seamless); the ARRIVING one
        # is the open question, so that is the one the widget may set --
        # `mode2`, defaulting to the role's own answer.
        arrive = state.get("mode2")
        if arrive not in ("guide", "masked", "both"):
            arrive = BRIDGE_MODES["prepend"]
        out = {"source": source, "role": role, "window": window,
               "mode": pin_mode, "mode2": arrive}
        # Mask shape for the ARRIVING side (prepend, or a bridge's second
        # half): how softly the pinned window is held near the join. An
        # ABSENT key means the user never touched the control, which
        # defaults to the measured-best soft hold (ramp 10, edge 0.40 --
        # clips 83/84); an explicit 0 they cycled to stays a hard hold.
        # emit() drops the shape for roles/modes it cannot apply to.
        out["mask_ramp_frames"] = max(0, min(
            256, int(state.get("mask_ramp_frames", 10) or 0)))
        out["mask_ramp_edge"] = max(0.0, min(1.0, float(
            state.get("mask_ramp_edge", 0.4) or 0.0)))
        out["mask_hold"] = max(0.0, min(1.0, float(
            state.get("mask_hold", 0.0) or 0.0)))
        # The timeline cut to continue from, in delivered frames. Absent
        # or null = the clip's own edge, which is the old behaviour.
        for key in ("cut", "cut2"):
            cut = state.get(key)
            if cut is None:
                continue
            try:
                out[key] = int(cut)
            except (TypeError, ValueError):
                raise ValueError(
                    "H3Timeline: pin_state %s %r is not a frame number."
                    % (key, cut))
            if out[key] < 0:
                raise ValueError(
                    "H3Timeline: pin_state %s %d is negative."
                    % (key, out[key]))
        if role == "bridge":
            # bridge = extend `source` AND prepend into `source2` in one
            # run (regenerating the clip between two links)
            source2 = str(state.get("source2") or "")
            if not source2:
                raise ValueError(
                    "H3Timeline: a bridge pin needs both source (the "
                    "clip to extend) and source2 (the clip to prepend "
                    "into).")
            out["source2"] = source2
        return out

    def _specs_for(self, clip, role, window, at_frame, pin_mode="guide",
                   mask_shape=None):
        """One side's pin specs, latent-grade if the clip allows it.

        The choice is made HERE rather than refused: the timeline lists
        whatever is in the folder, and imported footage is a legitimate
        thing to continue from. A clip with a trustworthy sidecar keeps
        the exact route; anything else gets a pixel spec that Apply
        encodes. Nothing degrades silently -- the log says which route a
        pin took, and the resulting take is saved as a root rather than
        claiming a lineage the encode cannot support.
        """
        bundle = nodes_load.verified_bundle_or_none(clip)
        if bundle is not None:
            return nodes_load._create_pins(
                bundle, clip, self._PIN_ROLES[role], window,
                at_frame=at_frame, pin_mode=pin_mode,
                mask_shape=mask_shape)
        _LOG.info("obvpm.h3: %s has no usable mctx sidecar; pinning it "
                  "through the PIXEL route (encoded at run time, "
                  "pixel-grade continuity, and the new take will be a "
                  "root)", clip)
        return nodes_load._create_pixel_pins(
            clip, self._PIN_ROLES[role], window, at_frame=at_frame,
            pin_mode=pin_mode, mask_shape=mask_shape)

    @staticmethod
    def _run_length(seconds, masks):
        """duration_seconds -> a frame count on the grid this pin needs.

        Two grids, and the pin decides which. A masked or both ARRIVAL
        preserves the destination's audio, and that audio only lines up
        with its own picture when the run's tick count comes out whole --
        the shared AV grid, 39/90/141/192/243 frames. Off it the join
        clicks by ~8 ms no matter where the cut is put (the handover
        cancels; see nodes_masked.handover_frames). Everything else is
        free to use the ordinary 17k+5 ladder, whose rungs are four
        times closer together.

        Snapping to the NEAREST rung rather than up: the AV grid steps
        2.125 s at a time, and silently adding two seconds to a
        requested duration is worse than silently removing one.
        """
        frames = max(1.0, float(seconds)) * fr.FPS
        if masks:
            return int(nodes_masked.shared_av_snap_nearest(frames))
        return int(fr.snap_run_nearest(frames))

    def _emit_length(self, state, seconds):
        """(length, log phrase) for this pin state."""
        masks = False
        if state is not None:
            arriving = (state["mode2"] if state["role"] == "bridge"
                        else state["mode"])
            masks = (arriving in ("masked", "both")
                     or state["mode"] in ("masked", "both"))
        n = self._run_length(seconds, masks)
        return n, ("%.2f s -> %d frames (%.3f s, %s)"
                   % (float(seconds), n, n / float(fr.FPS),
                      "shared AV grid, audio-exact" if masks
                      else "17k+5 ladder"))

    def emit(self, sequence="", pin_state="", duration_seconds=8.0,
             upscaling=False, **_):
        # A graph saved while this was the run_mode combo hands over its
        # string; "upscale" is the only value that meant on
        if isinstance(upscaling, str):
            upscaling = upscaling.strip().lower() in ("upscale", "true", "1")
        upscaling = bool(upscaling)
        state = self._parse_pin_state(pin_state)
        length, phrase = self._emit_length(state, duration_seconds)
        _LOG.info("obvpm.h3: timeline run: %s; length %s",
                  "upscale and render" if upscaling else "generate", phrase)
        if state is None:
            return ([], length, sequence, upscaling)
        # The soft hold applies to the ARRIVING side only: the
        # departing side of a bridge is a masked extend, which is
        # already seamless and has nothing to gain from being loosened.
        # Guide mode has no mask to shape, so the numbers are dropped
        # rather than passed and warned about downstream.
        # ...and an EXTEND never wears one at all: a masked extend is
        # already seamless. The state keeps its numbers across a role
        # switch, so without this an extend would inherit the ramp the
        # user set while the clip was a prepend.
        arriving = (state["mode2"] if state["role"] == "bridge"
                    else state["mode"])
        shape = ((state.get("mask_ramp_frames", 0),
                  state.get("mask_ramp_edge", 0.0),
                  state.get("mask_hold", 0.0))
                 if (arriving in ("masked", "both")
                     and state["role"] in ("prepend", "bridge"))
                 else (0, 0.0, 0.0))
        shaped = any(shape)
        if state["role"] == "bridge":
            # masked departing the left clip; the arriving side is
            # whatever mode2 says -- see PIN_MODE_FOR_ROLE
            arrive = state["mode2"]
            specs = (self._specs_for(state["source"], "extend",
                                     state["window"], state.get("cut"),
                                     BRIDGE_MODES["extend"]) +
                     self._specs_for(state["source2"], "prepend",
                                     state["window"], state.get("cut2"),
                                     arrive, mask_shape=shape))
            _LOG.info("obvpm.h3: timeline pins a BRIDGE: extends %s (%s) "
                      "+ prepends into %s (%s%s), window %s",
                      state["source"], BRIDGE_MODES["extend"],
                      state["source2"], arrive,
                      _shape_note(shape) if shaped else "",
                      state["window"])
            return (specs, length, sequence, upscaling)
        specs = self._specs_for(state["source"], state["role"],
                                state["window"], state.get("cut"),
                                state["mode"], mask_shape=shape)
        _LOG.info("obvpm.h3: timeline pins %s (%s, %s mode%s, window %s%s)",
                  state["source"], state["role"], state["mode"],
                  _shape_note(shape) if shaped else "",
                  state["window"],
                  "" if state.get("cut") is None
                  else ", from the cut at frame %d" % state["cut"])
        return (specs, length, sequence, upscaling)


class H3Assemble:
    CATEGORY = "obvpm/h3"
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("path", "images", "audio")
    DESCRIPTION = (
        "Plays an ordered list of clips as one MP4, deriving every seam "
        "cut from the mctx sidecars: extend seams butt seamlessly, "
        "trim-point and prepend seams enter/exit at the recorded join "
        "(including the 12-frame offset of continuation prepends). "
        "Writes by SMART-CUT: video packets are stream-copied bit for "
        "bit wherever possible, and only the sub-GOP frames around a "
        "mid-clip seam are re-encoded (with crf); audio is encoded once "
        "as a continuous track. The source takes remain the masters."
    )
    OUTPUT_TOOLTIPS = (
        "Path of the written MP4.",
        "The assembled frames.",
        "The assembled audio (present when every clip carries audio).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "One clip per line, in playback order, as "
                               "output-relative paths (the same values the "
                               "H3 loaders list), e.g. h3/clip_00001.mp4. "
                               "Append ' @ N' to force a clip to enter at "
                               "delivered frame N instead of the derived "
                               "seam. Lines starting with # are ignored."}),
                "filename_prefix": ("STRING", {
                    "default": "project1/cut",
                    "tooltip": "Output path prefix, like core save nodes. "
                               "The leading folder is the project, matching "
                               "base_folder elsewhere; an assembly is "
                               "delivery, so it sits beside the takes it "
                               "was cut from rather than inside them."}),
                "crf": ("INT", {
                    "default": 19, "min": 0, "max": 51,
                    "tooltip": "H.264 quality (lower = better, bigger)."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, sequence, **_):
        root = folder_paths.get_output_directory()
        parts = []
        for entry in _parse_sequence(sequence):
            if entry.get("gap"):
                continue           # empty space has no file to watch
            p = os.path.join(root, entry["clip"])
            for f in (p, mctx.sidecar_path(p)):
                try:
                    st = os.stat(f)
                    parts.append("%d:%d" % (st.st_size, st.st_mtime_ns))
                except OSError:
                    parts.append("absent")
        return "|".join(parts)

    def assemble(self, sequence, filename_prefix, crf):
        import torch

        root = folder_paths.get_output_directory()
        entries = resolve_sequence(sequence)

        from comfy_api.input_impl import VideoFromFile
        frames_parts, audio_parts, sample_rate = [], [], None
        have_audio = True
        size = None
        for e in entries:
            comps = VideoFromFile(e["path"]).get_components()
            images = comps.images
            n = int(images.shape[0])
            lo = min(e["enter"], n)
            hi = n if e["exit"] is None else min(e["exit"], n)
            if hi <= lo:
                raise ValueError(
                    "H3Assemble: the seam cuts leave nothing of %s "
                    "(frames %d..%d of %d)." % (e["clip"], lo, hi, n))
            this = (int(images.shape[2]), int(images.shape[1]))
            if size is None:
                size = this
            elif this != size:
                raise ValueError(
                    "H3Assemble: %s is %dx%d but the sequence started at "
                    "%dx%d. Clips must share one resolution."
                    % (e["clip"], this[0], this[1], size[0], size[1]))
            frames_parts.append(images[lo:hi])

            wav = (comps.audio or {}).get("waveform") \
                if isinstance(comps.audio, dict) else None
            if wav is None:
                have_audio = False
            elif have_audio:
                sr = int(comps.audio["sample_rate"])
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    _LOG.warning("obvpm.h3: %s has sample rate %d, sequence "
                                 "started at %d; dropping audio from the "
                                 "assembly", e["clip"], sr, sample_rate)
                    have_audio = False
                if have_audio:
                    a_lo = round(lo / fr.FPS * sample_rate)
                    a_hi = round(hi / fr.FPS * sample_rate)
                    audio_parts.append(wav[..., a_lo:min(a_hi, wav.shape[-1])])

            _LOG.info("obvpm.h3: + %s frames %d..%d (%d of %d)",
                      e["clip"], lo, hi, hi - lo, n)

        images = torch.cat(frames_parts, dim=0)
        audio = None
        if have_audio and audio_parts:
            audio = {"waveform": torch.cat(audio_parts, dim=-1),
                     "sample_rate": sample_rate}

        full_folder, filename, counter, _subfolder, _ = \
            folder_paths.get_save_image_path(
                filename_prefix, root, size[0], size[1])
        out_path = os.path.join(full_folder, "%s_%05d.mp4"
                                % (filename, counter))

        # smart-cut: plan copy/bridge runs per clip, splice packets. Any
        # _StreamMismatch (foreign encode, odd timestamps, bridge params
        # not byte-matching) falls back to one full re-encode.
        import tempfile
        from .nodes_save import H3SaveVideoWithMCtx
        wrote = None
        tmp = out_path + ".tmp.mp4"
        try:
            with tempfile.TemporaryDirectory() as btd:
                pieces, n_copy, n_bridge = [], 0, 0
                for i, (e, part) in enumerate(zip(entries, frames_parts)):
                    enter, exit_f = e["enter"], e["exit"]
                    if enter == 0 and exit_f is None:
                        pieces.append({"path": e["path"], "from": 0,
                                       "to": int(part.shape[0])})
                        n_copy += int(part.shape[0])
                        continue
                    kfs, n_total = _scan_keyframes(e["path"])
                    for kind, a, b in _plan_pieces(enter, exit_f, kfs,
                                                   n_total):
                        if kind == "copy":
                            pieces.append({"path": e["path"],
                                           "from": a, "to": b})
                            n_copy += b - a
                        else:
                            bp = os.path.join(btd,
                                              "bridge_%d_%d.mp4" % (i, a))
                            # the clip's OWN crf, or the splice fails
                            H3SaveVideoWithMCtx._encode_mp4(
                                bp, part[a - enter:b - enter], None,
                                source_crf(e["path"], crf))
                            pieces.append({"path": bp, "from": 0,
                                           "to": b - a})
                            n_bridge += b - a
                            _LOG.info("obvpm.h3: bridge for %s frames "
                                      "%d..%d (no keyframe at the cut)",
                                      e["clip"], a, b)
                _mux_pieces(pieces, tmp, audio)
                replace_when_free(tmp, out_path)
            wrote = ("video lossless" if n_bridge == 0 else
                     "smart-cut: %d frames copied, %d re-encoded"
                     % (n_copy, n_bridge))
        except _StreamMismatch as why:
            _LOG.warning("obvpm.h3: smart-cut not possible (%s); "
                         "re-encoding everything", why)
            try:
                os.remove(tmp)
            except OSError:
                pass
        if wrote is None:
            H3SaveVideoWithMCtx._encode_mp4(out_path, images, audio, crf)
            wrote = "re-encoded"

        _LOG.info("obvpm.h3: assembled %d clips -> %s (%d frames, %.1fs%s, "
                  "%s)", len(entries), out_path, int(images.shape[0]),
                  images.shape[0] / fr.FPS,
                  ", audio" if audio is not None else ", silent", wrote)
        return (out_path, images, audio)
