"""H3ResultPreview: a run's take shown beside its lineage neighbor.

Wire the Save node's `path` output in; after the run the node's widget
shows a MINI timeline -- the fresh take together with the parent it
extends (parent first) or the clip it prepends (take first) -- and plays
the pair seamlessly through the same server-built smart-cut route the
Timeline node uses. The answer it exists for: "did this take actually
connect?", one glance, right where the generation happened.

Lineage is resolved server-side from the take's sidecar (parent_id ->
scan the take's folder for the matching sidecar); everything visual
happens in web/h3_mctx_ui.js from the ui payload returned here.
"""

import json
import logging
import os

import folder_paths

from . import frames as fr
from . import mctx

_LOG = logging.getLogger("obvpm.h3")


# Identifies the seam measurement in IS_CHANGED. "2" = phase-normalised
# latent scan + pixel-domain cut detection (2026-08-17). "3" = the cut
# detection judges an extend's opening ACROSS the join, against the
# parent's own motion, rather than against the take's later content
# (2026-08-24). "4" = the latent seam is measured at the pinned WINDOW
# edge rather than the trim point (a soft hold makes them differ, and
# the old reading swept the window opening and called it a cut), and a
# prepend's frame scan now gets its target's frames as the baseline the
# way an extend always got its parent's (2026-08-27). Bumped because the
# numbers a take was shown with are no longer the numbers it would get:
# without this, a take measured under an older rule keeps its stale
# verdict when the node is shown again.
_MEASURE_VERSION = "4"

_SEVERITY = {"seamless": 0, "soft bump": 1, "hard cut": 2}


def _severity(m):
    """How bad a measurement reads, for comparing the two metrics.

    Their ratios are on different scales (a latent ratio tops out near
    3, a pixel one reaches 17), so they are compared by verdict.
    """
    return _SEVERITY.get((m or {}).get("verdict"), -1)


def _pinned_window(meta, place):
    """Frames PINNED on one side, from the recipe (0 when none).

    Distinct from pinned_head/tail_frames, which are what was TRIMMED:
    with a soft hold the take delivers part of its window, so the two
    differ and only this one marks where generated content ends.
    """
    return sum(int(s.get("source_frames", 0) or 0)
               for s in mctx.parse_pins(meta)
               if s.get("place") == place)


def _measure_seam(side, which=None):
    """Measured seam quality where generated content meets pinned.

    Delegates the metric to seam_report.measure_seam (one implementation,
    so the node and the CLI report can never disagree): the worst
    adjacent-latent-step difference from the boundary through the next
    ~1s of GENERATED content, as a ratio of the clip's median step
    difference. Scanning past the boundary is what catches the common
    failure -- a run that holds the pinned window faithfully and only
    then breaks away looks perfect measured at the join alone.
    Metadata can only say the join coordinates line up; this says
    whether motion actually flows through them. ~1x = seamless, >>1x =
    the model planted the anchor and cut to it. `which` picks the
    boundary ("head"/"tail") for multi-pin takes; None derives it from
    the relation summary. Returns {"ratio", "verdict", "at", "boundary"}
    (`at` = seconds from the join to the worst point) or None when not
    measurable.
    """
    from .seam_report import measure_seam, verdict_for
    video, _audio, meta = mctx.load_sidecar(side)
    video = video.float()
    rel = meta.get("relation")
    # The boundary this metric wants is where GENERATED content meets
    # PINNED content -- the window edge. pinned_head/tail_frames are the
    # TRIM amounts, which a softly-held window makes smaller than the
    # window (the take delivers its ramped frames). Reading them as the
    # edge put the seam frames late, inside verbatim content, and the
    # backward scan then swept over the window opening and called it a
    # cut. Prefer the window from the recipe; fall back to the trim for
    # recipes that predate the pins list.
    head = _pinned_window(meta, "before") or int(
        meta.get("pinned_head_frames", 0) or 0)
    tail = _pinned_window(meta, "after") or int(
        meta.get("pinned_tail_frames", 0) or 0)
    lt = video.shape[2]
    if which is None:
        which = ("head" if rel == "extends" else
                 "tail" if rel == "prepends" else None)
    if which == "head" and head:
        # pinned run first: the generated content follows the boundary
        seam, into = fr.frames_to_latents(head) - 1, 1
    elif which == "tail" and tail:
        # generated first, pinned run last: look back from the boundary
        seam, into = lt - fr.frames_to_latents(tail) - 1, -1
    else:
        return None
    if not 0 <= seam < lt - 1:
        return None
    m = measure_seam(video, seam, into)
    if m is None:
        return None
    ratio = m["ratio"]
    return {
        "ratio": round(ratio, 2),
        "verdict": verdict_for(ratio),
        "at": round(m["at_frames"] / float(fr.FPS), 2),
        "boundary": round(m["boundary"], 2),
    }


def _join_context(parent_path, header, place):
    """(neighbour clip, the delivered frame the join sits at).

    What measure_cuts needs to judge a join against the motion on the
    OTHER side of it, rather than against the take's own content. An
    extend without it reads a take that joins mid-action and then
    settles as a lurch -- which is what extending from a trim point
    produces, since people cut where something is happening.

    A prepend needs it just as much, in mirror image: it ends by
    arriving at a clip that may be moving faster than it was, and
    measured against its own calmer opening that acceleration reads as
    a burst (clip_00085 scored 6.5x on a join the eye calls seamless).
    `place` says which end: "before" for an extend's opening, "after"
    for a prepend's close.
    """
    try:
        from .nodes_assemble import _extend_parent, _prepend_child
        # the same two readers assembly cuts with, so the frame measured
        # is the frame the viewer will actually see joined; both are
        # handover-aware and both fall through to the pins recipe when
        # the single-pin summary cannot express the take (a bridge)
        found = (_prepend_child(header) if place == "after"
                 else _extend_parent(header))
        if not found:
            return None
        head = int(mctx.read_header(
            mctx.sidecar_path(parent_path)).get(
                "pinned_head_frames", 0) or 0)
        join = int(found[1]) - head
        # 0 is a real answer for a prepend (a hard hold enters at the
        # target's own frame 0); measure_cuts falls back on its own if
        # too few frames come back.
        return (parent_path, join) if join >= 0 else None
    except Exception:
        return None


class H3ResultPreview:
    CATEGORY = "obvpm/h3"
    FUNCTION = "show"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    DESCRIPTION = (
        "Mini timeline of a just-saved take and the clip it continues: "
        "wire a Save node's path output in, and after each run this "
        "shows the take beside its parent (extend) or its target "
        "(prepend), seam derived from the sidecars, playable as one "
        "seamless preview."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The saved clip's path -- wire the path "
                               "output of an H3 MCtx Save node."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, path):
        # The measurement version rides along so that changing how seams
        # are judged re-runs every preview once. Without it a clip whose
        # file has not moved keeps showing a verdict from the old metric
        # for good: the node is cached, show() is never called again, and
        # the widget restores the payload saved in the workflow. Bump
        # _MEASURE_VERSION whenever the numbers or thresholds change.
        try:
            st = os.stat(cls._abs(path))
            return "%s:%d:%d" % (_MEASURE_VERSION, st.st_size, st.st_mtime_ns)
        except (OSError, ValueError):
            return "absent:" + _MEASURE_VERSION

    @staticmethod
    def _abs(path):
        p = str(path or "").strip()
        if not os.path.isabs(p):
            p = os.path.join(folder_paths.get_output_directory(), p)
        return os.path.abspath(p)

    def show(self, path):
        root = os.path.abspath(folder_paths.get_output_directory())
        ap = self._abs(path)
        if os.path.commonpath([root, ap]) != root:
            raise ValueError(
                "H3ResultPreview: %s is outside the output folder" % path)
        if not os.path.isfile(ap):
            raise ValueError("H3ResultPreview: clip not found: %s" % path)
        rel = os.path.relpath(ap, root).replace(os.sep, "/")

        relation, parent_rel, parent2_rel = "", None, None
        seam, seam2 = None, None
        render = None   # "joint" for a rendered cut (H3 Joint VAE Decode and Save)

        def _clip_for(pid, recorded_path=None):
            """The parent clip, output-relative. None when not findable.

            Two routes, because a parent does not have to own a sidecar:
            the usual one scans the folder for the sidecar whose self_id
            matches, and the fallback follows the path the pins recipe
            recorded -- which is how a take reaches the plain video it
            was encoded from. The hash still decides: a file at that path
            that is not the file we pinned is not the parent.
            """
            pside = mctx.scan_for_parent(os.path.dirname(ap), pid)
            if pside:
                pclip = pside[:-len(mctx.SIDECAR_SUFFIX)] + ".mp4"
                if os.path.isfile(pclip):
                    return os.path.relpath(pclip, root).replace(os.sep, "/")
            if not recorded_path:
                return None
            cand = os.path.abspath(os.path.join(root, recorded_path))
            if os.path.commonpath([root, cand]) != root:
                return None
            if not os.path.isfile(cand):
                return None
            from .nodes_load import _cached_hash
            if pid and _cached_hash(cand) != pid:
                _LOG.warning("obvpm.h3: %s records a parent at %s, but that "
                             "file has changed since; previewing the take "
                             "alone", rel, recorded_path)
                return None
            return os.path.relpath(cand, root).replace(os.sep, "/")

        def _recorded_path(pins, place):
            for s in pins:
                if s.get("place") == place and s.get("source_path"):
                    return s["source_path"]
            return None

        side = mctx.sidecar_path(ap)
        if os.path.isfile(side):
            try:
                header = mctx.read_header(side)
                relation = header.get("relation") or ""
                try:
                    render = (json.loads(header.get("user_meta") or "{}")
                              or {}).get("render") or None
                except (TypeError, ValueError):
                    render = None
                pid = header.get("parent_id") or ""
                if not relation:
                    # multi-pin take: derive lineage from the pins
                    # recipe (a BRIDGE has a before-parent it extends
                    # and an after-parent it prepends into)
                    pins = mctx.parse_pins(header)
                    bpin = next((p for p in pins
                                 if p.get("place") == "before" and
                                 p.get("source_kind") in mctx.LINEAGE_KINDS
                                 and p.get("source_id")), None)
                    apin = next((p for p in pins
                                 if p.get("place") == "after" and
                                 p.get("source_kind") in mctx.LINEAGE_KINDS
                                 and p.get("source_id")), None)
                    if bpin and apin:
                        relation = "bridges"
                        parent_rel = _clip_for(bpin["source_id"],
                                               bpin.get("source_path"))
                        parent2_rel = _clip_for(apin["source_id"],
                                                apin.get("source_path"))
                        try:
                            seam = _measure_seam(side, "head")
                            seam2 = _measure_seam(side, "tail")
                        except Exception:
                            _LOG.exception("obvpm.h3: seam measurement "
                                           "failed for %s", rel)
                if relation in ("extends", "prepends"):
                    try:
                        seam = _measure_seam(side)
                    except Exception:
                        _LOG.exception(
                            "obvpm.h3: seam measurement failed for %s", rel)
                    if pid:
                        parent_rel = _clip_for(
                            pid, _recorded_path(
                                mctx.parse_pins(header),
                                "before" if relation == "extends"
                                else "after"))
            except Exception:
                _LOG.exception(
                    "H3ResultPreview: unreadable sidecar for %s", rel)

        # The latent metric cannot see a jump cut -- one latent step spans
        # 3.4 frames, so a single-frame jump averages away inside it. Look
        # at the delivered frames too, which is what actually gets watched,
        # and let the worse of the two speak. Only the take's opening is
        # covered, which is where an extend joins; a prepend joins at its
        # end and still relies on the latent number alone.
        # Which end carries the join: an extend begins at its parent, a
        # prepend runs INTO its target so the failure shows at its close,
        # and a bridge has one of each.
        if relation in ("extends", "prepends", "bridges"):
            try:
                from .seam_report import measure_cuts
                if relation == "bridges":
                    scans = (("start", "seam"), ("end", "seam2"))
                elif relation == "prepends":
                    scans = (("end", "seam"),)
                else:
                    scans = (("start", "seam"),)
                for where, slot in scans:
                    # The neighbour's own frames are the baseline a join
                    # is judged against, in BOTH directions: an extend
                    # opens at its parent, a prepend closes into its
                    # target. Which neighbour depends on the end -- a
                    # bridge has a different one at each.
                    nb = (parent2_rel if where == "end" and parent2_rel
                          else parent_rel)
                    ctx = (_join_context(
                        self._abs(nb), header,
                        "after" if where == "end" else "before")
                        if nb else None)
                    cuts = measure_cuts(ap, where=where, context=ctx)
                    current = seam if slot == "seam" else seam2
                    # Worse-of-the-two, EXCEPT when the frame scan had
                    # the parent to compare against: then it is the only
                    # measurement that actually spans the join, while the
                    # latent one still judges the opening against the
                    # take's own later content -- which reads any join
                    # into a moving moment as a break. The latent ratio
                    # rides along either way.
                    if cuts and (ctx or
                                 _severity(cuts) > _severity(current)):
                        cuts["latent"] = current and current.get("ratio")
                        if slot == "seam":
                            seam = cuts
                        else:
                            seam2 = cuts
            except Exception:
                _LOG.exception("obvpm.h3: cut scan failed for %s", rel)

        if relation == "bridges":
            parts = ([parent_rel] if parent_rel else []) + [rel] + \
                ([parent2_rel] if parent2_rel else [])
            sequence = "\n".join(parts)
        elif parent_rel and relation == "extends":
            sequence = parent_rel + "\n" + rel
        elif parent_rel and relation == "prepends":
            sequence = rel + "\n" + parent_rel
        else:
            sequence = rel
        if parent_rel is None and relation in ("extends", "prepends"):
            _LOG.warning("obvpm.h3: %s %s a parent whose clip is not in "
                         "its folder; previewing the take alone",
                         rel, relation)
        return {"ui": {"h3_result": [{
            "clip": rel, "parent": parent_rel, "parent2": parent2_rel,
            "relation": relation, "sequence": sequence,
            "seam": seam, "seam2": seam2, "render": render,
        }]}}
