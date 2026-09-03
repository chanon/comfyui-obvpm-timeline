"""The upscale/refine loop: one visible body, run once per timeline clip.

ComfyUI graphs are acyclic and always will be -- a link's type is
resolved from the upstream class's static RETURN_TYPES indexed by slot,
so a cycle has no well-defined type or cache key. Iteration is spelled
differently: a node returns `{"expand": subgraph}` and the executor runs
that subgraph in its place. A node inside the expansion can expand again,
so it is genuine recursion, and no cycle exists at any instant.

The loop BODY stays visible and editable in the graph. `Loop End` finds
it by walking the dependency graph backwards to `Loop Start` -- so the
body is defined by reachability, and dragging a node into it is enough
to put it in the loop, with no list to maintain anywhere.

    H3 Timeline (outside; supplies the ordered clip list)
        |
        v
    Loop Start --clip_path--> H3 MCtx Load --> Load Conditioning
        |      --pin_specs--> Apply Pins (freeze_audio on)
        |      --out_folder-> the save node's base_folder
        |      --first_sigma-> the refine schedule (BasicScheduler denoise)
        |  flow                                        |
        +---------------------> Loop End <--after------+

The traversal is adapted from Ethanfel's SxCP loop nodes in
ComfyUI-Prompt-Builder, by way of ComfyUI-MiniMaxH3-Contex-Loop's
`chain_nodes.py` (GPL-3.0, as this pack is).

WHY THE TIMELINE IS NOT IN THE LOOP: its whole contribution is the
ordered list, which does not change between iterations. The junction pin
is loop state -- it references the previous iteration's REFINED output,
which does not exist until the loop runs -- and the per-clip load is
per-iteration. So the Timeline is evaluated once, outside.
"""

import logging
import os

from . import mctx
from . import nodes_load
from . import upscale
from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

# A join is only worth reproducing when it was latent-identical to begin
# with. "both" masks its window as well as guiding, so it qualifies for
# the same reason "masked" does.
HELD_MODES = ("masked", "both")


def _held_pin(header, place, neighbour_id):
    """The pin by which this clip holds `neighbour_id`, or None.

    `place` says which edge: a "before" pin holds the neighbour that
    plays BEFORE this clip (an extend), an "after" pin holds the one
    that plays AFTER it (a prepend, or the departing half of a bridge).
    """
    for pin in mctx.parse_pins(header):
        if pin.get("place") != place:
            continue
        if pin.get("mode") not in HELD_MODES:
            continue
        if pin.get("source_id") and neighbour_id \
                and pin["source_id"] != neighbour_id:
            continue
        return pin
    return None

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except ImportError:  # pragma: no cover - only in a build without expansion
    GraphBuilder = None
    is_link = None


class H3UpscaleLoopStart:
    """Opens the loop: resolves which clip this iteration refines."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "start"
    RETURN_TYPES = (wt.LOOP, "STRING", wt.PINSPECS, "STRING", "FLOAT",
                    "INT", "INT")
    RETURN_NAMES = ("flow", "clip_path", "pin_specs", "out_folder",
                    "first_sigma", "index", "total")
    DESCRIPTION = (
        "Opens an upscale/refine loop over a timeline. Everything between "
        "this node and H3 Upscale Loop End is the loop body and runs once "
        "per clip, in delivery order, each pinned to the previous clip's "
        "REFINED tail. Resumable: the profile folder records what is done, "
        "so pressing Run again continues rather than restarting."
    )
    OUTPUT_TOOLTIPS = (
        "Wire to Loop End. Carries which node opened the loop.",
        "This iteration's source clip, output-relative. Wire to H3 MCtx "
        "Load's clip_path.",
        "The junction pin: the previous REFINED clip's tail, mirroring "
        "the geometry of the pin the original take was made with. Empty "
        "for the first clip, and for any join that was already a cut.",
        "Where this profile's refined clips go (<base_folder>/_upscale/"
        "<profile>). Wire to the save node's base_folder.",
        "How much of the pass re-samples -- wire to the refine "
        "BasicScheduler's denoise, so the value that defines the pass "
        "lives with the pass (and is part of its config hash).",
        "0-based position in the timeline.",
        "How many clips the timeline holds.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The timeline, one output-relative clip per "
                               "line in delivery order -- the same text the "
                               "H3 Timeline node holds. Refined in this "
                               "order, so each clip can pin to the one "
                               "before it."}),
                "base_folder": ("STRING", {
                    "default": "project1",
                    "tooltip": "The project folder the clips live in. Same "
                               "meaning as the Timeline's and the save "
                               "nodes' base_folder, so one value can drive "
                               "all three."}),
                "profile": ("STRING", {
                    "default": "",
                    "tooltip": "Leave EMPTY and the folders manage "
                               "themselves: a pass resumes the profile "
                               "whose settings match these, and otherwise "
                               "creates the next free refineNN -- so "
                               "changing a setting lands in a new folder "
                               "instead of being refused. Type a name to "
                               "pin one folder. Refined clips are written "
                               "to <base_folder>/_upscale/<profile>/; the "
                               "sources are never touched, and the folder "
                               "comes out of Loop End as profile_folder "
                               "for H3 Assemble Upscale."}),
                "first_sigma": ("FLOAT", {
                    "default": 0.24, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "How much of the pass re-samples -- wire "
                               "it out to the schedule so the profile "
                               "hash covers the real value. Into a "
                               "BasicScheduler's denoise it is a FRACTION "
                               "of the schedule (0.24 enters near sigma "
                               "0.79 at shift 12: a refine); into a "
                               "manual ladder it is the entry sigma "
                               "itself, and CONST.noise_scaling is "
                               "sigma*noise + (1-sigma)*latent, so 0.9 "
                               "keeps about a tenth of the source -- a "
                               "restyle that flickers and drops lip "
                               "sync (measured 2026-09-01)."}),
                "upscale_note": ("STRING", {
                    "default": "",
                    "tooltip": "A short label for the upscaler settings "
                               "this pass uses (e.g. '2x chunking-off'). "
                               "Not read -- it is hashed, so a profile "
                               "cannot be half-finished at one setting and "
                               "completed at another."}),
                "start_index": ("INT", {
                    "default": -1, "min": -1, "max": 4096,
                    "tooltip": "-1 = continue where the profile left off, "
                               "which is what you want. A number forces the "
                               "starting clip, for re-doing one. The loop "
                               "drives this on every iteration after the "
                               "first."}),
                "junction_ramp": ("INT", {
                    "default": 0, "min": 0, "max": 56,
                    "tooltip": "Soften the junction: over this many frames "
                               "before the join the held window's mask "
                               "ramps from exact (0) up to junction_edge, "
                               "so the two clips' re-derived detail blends "
                               "across the ramp instead of switching on "
                               "one frame -- a refine invents fine texture "
                               "(distant people, foliage) and each clip "
                               "invents its own. The ramped frames are "
                               "re-drawn by the LATER clip and delivered "
                               "by it, so the earlier clip exits that much "
                               "sooner; the total length is unchanged. 0 = "
                               "mirror the recorded hold exactly (a hard "
                               "hold for most takes). Part of the profile "
                               "hash."}),
                "junction_edge": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "With junction_ramp: the mask value AT the "
                               "join, as a fraction of the pass's own "
                               "sigma (1 = fully this clip's refine, 0 = "
                               "still exact). 0.4 is the generation-side "
                               "default for arriving joins."}),
                "junction_mode": (["mirror", "both", "guided"], {
                    "default": "mirror",
                    "tooltip": "How the junction window is pinned. mirror "
                               "= the mode the take was made with (masked "
                               "for an extend: the window is held exactly, "
                               "nothing more). both = held exactly AND fed "
                               "to the model as keyframe conditioning rows. "
                               "guided = NOT held: the window is the clip's "
                               "starting point and its steering rows, the "
                               "clip re-draws it in its own hand and "
                               "DELIVERS it, and the neighbour hands over "
                               "at the window's start -- so the neighbour's "
                               "texture becomes this clip's across 39 "
                               "frames of one generation instead of "
                               "switching on one frame (the join a refine "
                               "invents differently on each side of). The "
                               "seam is then a guided one, pixel-grade. "
                               "Part of the profile hash when not mirror."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # the clip this resolves to depends on what is on disk, not on any
        # widget, so a byte-identical prompt must still re-run
        return float("nan")

    def start(self, sequence, base_folder, profile, first_sigma,
              upscale_note, start_index=-1, junction_ramp=0,
              junction_edge=0.4, junction_mode="mirror", unique_id=None,
              **_):
        import folder_paths

        clips = upscale.parse_sequence(sequence)
        lines = upscale.parse_sequence_lines(sequence)
        if not clips:
            raise ValueError(
                "H3 Upscale Loop: the sequence is empty, so there is "
                "nothing to refine. Paste the timeline's clip list in.")
        config = {"first_sigma": round(float(first_sigma), 6),
                  "upscale": str(upscale_note or "")}
        # (ramp, edge, deep) the junction pins ride in place of the
        # recorded hard hold; None = mirror the recipe, which keeps the
        # hash of every profile made before this existed
        junction = None
        if int(junction_ramp) > 0:
            junction = [int(junction_ramp), round(float(junction_edge), 4), 0.0]
            config["junction"] = junction
        mode = str(junction_mode or "mirror")
        if mode != "mirror":
            config["junction_mode"] = mode
        out_dir = folder_paths.get_output_directory()
        name = str(profile or "").strip()
        if not name:
            # No name = the folders manage themselves. The loop carries
            # the choice into every later iteration (see _recurse), so it
            # is made exactly once per pass.
            base_parts = [p for p in str(base_folder or "")
                          .strip().strip("/\\").replace("\\", "/").split("/") if p]
            name, existing = upscale.choose_profile(
                os.path.join(out_dir, *base_parts), config, clips)
            _LOG.info("obvpm.h3: upscale profile %s -> %s",
                      "resumes" if existing else "starts new", name)
        rel_folder = upscale.profile_folder(base_folder, name)
        profile = name
        folder = os.path.join(out_dir, *rel_folder.split("/"))

        document = upscale.read_manifest(folder)
        hash_value = upscale.validate(document, config)

        headers = self._headers(clips)
        deps = self._joins(headers)
        order = self._refine_order(len(clips), deps)
        # what the CURRENT timeline can reuse: a refined clip counts only
        # while the junctions it was made with are the ones this timeline
        # needs, so an edit redoes exactly the clips it changed
        valid = upscale.refined_for(
            document, folder, clips, {i: [j for j, _ in d] for i, d in deps.items()})
        remaining = [i for i in order if i not in valid]

        if int(start_index) < 0:
            if not remaining:
                raise ValueError(
                    "H3 Upscale Loop: profile %r has already refined all "
                    "%d clip(s) of this timeline. Delete the folder to redo "
                    "it, or set start_index to redo one."
                    % (profile, len(clips)))
            index = remaining[0]
            following = remaining[1] if len(remaining) > 1 else None
        else:
            index = int(start_index)
            if index >= len(clips):
                raise ValueError(
                    "H3 Upscale Loop: start_index %d is past the end of a "
                    "%d clip timeline." % (index, len(clips)))
            missing = [j for j, _ in deps[index] if j not in valid]
            if missing:
                raise ValueError(
                    "H3 Upscale Loop: clip %d holds clip(s) %s, which have "
                    "not been refined yet -- it cannot be redone on its "
                    "own until they are." % (index, missing))
            following = next((i for i in remaining if i != index), None)

        clip = clips[index]
        _LOG.info("obvpm.h3: upscale %s [position %d, %d of %d] -> %s%s",
                  clip, index, len(valid) + 1, len(clips), rel_folder,
                  (" pinned to refined %s"
                   % ", ".join(str(j) for j, _ in deps[index]))
                  if deps[index] else " (no junction pin)")
        specs = self._junction(index, deps, document, rel_folder, clips,
                               junction, mode)
        # what this rendering is pinned to, recorded with it so a later
        # timeline can tell whether the join it needs is the join it got
        pinned_to = {clips[j]: valid[j]["output"] for j, _ in deps[index]}
        # the id is how Loop End finds which node opened the loop, and
        # therefore where the body starts
        flow = {"node_id": str(unique_id) if unique_id is not None else None,
                "index": index, "next_index": following, "total": len(clips),
                "done": len(valid) + 1, "pinned_to": pinned_to,
                # the source's own trimmed head: a ramped junction ships
                # part of the window, so the refined head is shorter and
                # the source line's cut markers move by the difference
                "source_head": int(headers[index].get("pinned_head_frames", 0) or 0),
                "lines": lines, "profile": profile,
                "folder": folder, "rel_folder": rel_folder, "source": clip,
                "config": config, "hash": hash_value}
        return (flow, clip, specs, rel_folder, float(first_sigma),
                index, len(clips))

    @staticmethod
    def _headers(clips):
        """Every clip's sidecar header, in delivery order."""
        out = []
        for clip in clips:
            path = nodes_load.resolve_clip_path(clip)
            out.append(mctx.read_header(mctx.sidecar_path(path)))
        return out

    @staticmethod
    def _joins(headers):
        """One decision per adjacent pair: who is refined after whom.

        Returns {i: [(j, pin)]} -- clip i is refined AFTER clip j, and
        pins to j's REFINED output using `pin`'s recorded geometry.

        Which side carries a join depends on how it was generated, and
        the pass has to follow rather than assume. An extend puts the
        overlap at the arriving clip's HEAD, so the later clip depends
        on the earlier one. A prepend or the departing half of a bridge
        puts it at the earlier clip's TAIL, so the dependency RUNS
        BACKWARDS and delivery order is the wrong order to work in.

        Exactly one edge per join, so the graph cannot contain a cycle:
        when both sides hold each other, the extend wins, because
        walking forwards is what the rest of the pass is shaped around.
        """
        ids = [h.get("self_id") for h in headers]
        deps = {i: [] for i in range(len(headers))}
        for a in range(len(headers) - 1):
            b = a + 1
            arriving = _held_pin(headers[b], "before", ids[a])
            if arriving is not None:
                deps[b].append((a, arriving))
                continue
            departing = _held_pin(headers[a], "after", ids[b])
            if departing is not None:
                deps[a].append((b, departing))
        return deps

    @staticmethod
    def _refine_order(count, deps):
        """Delivery positions, ordered so dependencies come first.

        Ties break on delivery position, so the order is stable and a
        timeline of plain cuts is refined exactly front to back.
        """
        done, order = set(), []
        while len(order) < count:
            ready = [i for i in range(count)
                     if i not in done
                     and all(j in done for j, _ in deps[i])]
            if not ready:
                raise ValueError(
                    "H3 Upscale Loop: the joins in this timeline have no "
                    "workable order (clips %s depend on each other). This "
                    "should be impossible -- please report it."
                    % sorted(set(range(count)) - done))
            order.append(ready[0])
            done.add(ready[0])
        return order

    @staticmethod
    def _junction(index, deps, document, rel_folder, clips, junction=None,
                  mode="mirror"):
        """The pins that hold this clip to its already-refined neighbours.

        MIRRORS THE RECORDED PIN rather than inventing one. The take
        wrote down how it was made -- window length, source point, mask
        mode and ramp shape -- and the refined neighbour has the same
        frame count, so the same geometry lands in the same place. Only
        the CONTENT changes, from the neighbour's original latents to
        its refined ones.

        THE MASK SHAPE IS PART OF THE GEOMETRY, not a matter of taste: a
        ramped window ships its ramped frames and trims only the held
        ones, so `pins_trim_totals` reads the shape to decide where the
        clip ENDS. Replace a ramped hold with a hard one and the refined
        clip comes out short by exactly the ramp.

        Empty for a clip with no held join -- a cut in the delivery is
        nothing to preserve, and pinning across it would invent
        continuity that was never there.
        """
        specs = []
        for neighbour, pin in deps.get(index, []):
            entry = upscale.entry_for(document, clips[neighbour])
            output = entry.get("output") if entry else None
            if not output:
                raise ValueError(
                    "H3 Upscale Loop: clip %d pins to clip %d, which this "
                    "profile has not refined yet. The refine order is "
                    "supposed to prevent this -- please report it."
                    % (index, neighbour))
            parent_clip = "%s/%s" % (rel_folder, output)
            parent = nodes_load.load_verified_bundle(parent_clip)
            head = int(parent["meta"].get("pinned_head_frames", 0) or 0)

            window = int(pin.get("source_frames") or 0)
            raw_start = int(pin.get("source_start") or 0)
            # The refine's own ramp, when set, replaces the recorded hold:
            # a refine invents detail per clip, and a hard switch between
            # two inventions is visible even where the motion is exact.
            # The window and its position are still the recipe's -- only
            # how firmly it is held changes, and the trim reads the same
            # shape, so the ramped frames are delivered by this take.
            if mode == "guided":
                # The window is this take's STARTING POINT and its
                # steering rows, never a hold: written into the latent,
                # fed as keyframe conditioning ("both"), mask fully open.
                # The take re-draws it in its own hand and DELIVERS it
                # (handover_frames: an open window hands over at its
                # start), so the source's texture becomes this take's
                # across 39 frames of one generation, not on one frame.
                shape = (0, 0.0, 1.0)
            elif junction:
                shape = (int(junction[0]), float(junction[1]),
                         float(junction[2]) if len(junction) > 2 else 0.0)
            else:
                shape = (int(pin.get("mask_ramp_frames", 0) or 0),
                         float(pin.get("mask_ramp_edge", 0.0) or 0.0),
                         float(pin.get("mask_hold", 0.0) or 0.0))

            # The recorded window is in the neighbour's RAW frames;
            # `_create_pins` takes a DELIVERED cut, and the two differ by
            # the neighbour's own pinned head. An extend names the frame
            # the window ENDS at, a prepend the frame it STARTS at.
            if pin.get("place") == "before":
                how, cut = "extend (pin tail)", raw_start + window - head
            else:
                how, cut = "prepend (pin head)", raw_start - head

            # "both" keeps the exact hold and ADDS the window as keyframe
            # conditioning rows, so the clip's free frames are steered to
            # stay consistent with what it was handed; "mirror" pins the
            # way the take was made
            pin_mode = {"mirror": pin.get("mode") or "masked",
                        "guided": "both"}.get(mode, mode)
            specs.extend(nodes_load._create_pins(
                parent, parent_clip, how, str(window), at_frame=cut,
                pin_mode=pin_mode, mask_shape=shape))
        return specs

class H3UpscaleLoopEnd:
    """Closes the loop: commits this iteration, then expands the next."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "end"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "profile_folder")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Closes an upscale/refine loop. Records the clip this iteration "
        "wrote, then re-runs the whole body for the next one until the "
        "timeline is done. Everything between Loop Start and this node is "
        "the body -- it is found by following the wires, so there is no "
        "list to keep up to date."
    )
    OUTPUT_TOOLTIPS = (
        "What the pass did, one line. Only becomes a real value at the "
        "last iteration, so anything wired to it waits for the whole pass.",
        "The finished profile's folder, output-relative. Wire to H3 "
        "Assemble Upscale's profile_folder and it plays the refined cut "
        "when the pass ends, with nothing typed twice.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (wt.LOOP, {
                    "tooltip": "From H3 Upscale Loop Start."}),
                "after": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The save node's path output. Wiring it here "
                               "is what makes the save part of the loop "
                               "body and forces it to finish before the "
                               "next iteration starts."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    def end(self, flow, after, dynprompt=None, unique_id=None):
        if not isinstance(flow, dict) or not flow.get("node_id"):
            raise ValueError(
                "H3 Upscale Loop End: the flow input must come from an H3 "
                "Upscale Loop Start.")
        index, total = int(flow["index"]), int(flow["total"])
        following = flow.get("next_index")
        folder = flow["folder"]

        # COMMIT FIRST, RECURSE SECOND. The manifest is written after the
        # clip it describes, so a crash between them leaves a refined clip
        # the manifest does not know about -- which the next run simply
        # redoes. The other order would leave the manifest naming a file
        # that is not there, and the next junction pin would try to load it.
        # How much earlier the refined clip's delivered frames start
        # than the source's: a ramped junction ships part of the held
        # window. The source line's `@` cut markers index the SOURCE's
        # delivered frames, so the assembly moves them by this.
        head_shift = 0
        if flow.get("pinned_to"):
            try:
                saved = str(after)
                if not os.path.isabs(saved):
                    saved = nodes_load.resolve_clip_path(saved)
                refined_head = int(mctx.read_header(
                    mctx.sidecar_path(saved)).get("pinned_head_frames", 0) or 0)
                head_shift = int(flow.get("source_head") or 0) - refined_head
            except Exception:
                _LOG.exception("obvpm.h3: could not read the refined "
                               "clip's head; cut markers keep the source "
                               "frame numbers")
        document = upscale.read_manifest(folder)
        upscale.record(document, flow["source"], os.path.basename(str(after)),
                       flow["config"], flow["hash"],
                       pinned_to=flow.get("pinned_to"),
                       head_shift=head_shift)
        # the timeline's own lines, so the assembly can put the cut
        # markers back on clips whose frame counts are unchanged
        document["lines"] = flow.get("lines") or []
        upscale.write_manifest(folder, document)

        done = int(flow.get("done") or index + 1)
        report = "refined %d of %d: %s -> %s" % (
            done, total, flow["source"], os.path.basename(str(after)))
        _LOG.info("obvpm.h3: %s", report)
        # The next clip is whatever the DEPENDENCY order says, not the
        # next delivery position: a join carried by a tail is refined
        # after the clip that plays behind it.
        if following is None:
            _LOG.info("obvpm.h3: upscale profile complete -- %d clip(s) in %s",
                      total, flow["rel_folder"])
            return {"ui": {"text": [report + " -- done"]},
                    "result": (report, str(flow["rel_folder"]))}
        return self._recurse(flow, int(following), dynprompt, unique_id, report)

    # ---------------------------------------------------------------- body

    def _contained(self, dynprompt, unique_id, open_node):
        """The loop body: what DEPENDS ON Loop Start and reaches this node.

        Both halves matter, and getting only one of them is the obvious
        mistake. "Everything upstream of Loop End" would sweep in the
        model loaders and the Timeline -- they are upstream of End, but
        they do not depend on Start, so cloning them would re-run a
        checkpoint load every iteration. "Everything downstream of Start"
        would sweep in anything else the user hung off its outputs.

        So: walk BACKWARDS from End to learn the cone that feeds it,
        recording who consumes whom, then walk FORWARDS from Start
        through that same cone. What survives both is the body.
        """
        consumers, seen = {}, set()
        queue = [str(unique_id)]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            try:
                node = dynprompt.get_node(current)
            except Exception:
                continue
            for value in (node.get("inputs") or {}).values():
                if is_link is not None and is_link(value):
                    consumers.setdefault(str(value[0]), []).append(current)
                    queue.append(str(value[0]))

        open_id, end_id = str(open_node), str(unique_id)
        body, stack = {open_id, end_id}, [open_id]
        while stack:
            for child in consumers.get(stack.pop(), []):
                if child not in body:
                    body.add(child)
                    stack.append(child)
        # Start first so its clone exists before anything sets an input on
        # it; the rest sorted, so an expansion is reproducible
        return [open_id] + sorted(n for n in body if n != open_id)

    def _recurse(self, flow, next_index, dynprompt, unique_id, report):
        if GraphBuilder is None:
            raise RuntimeError(
                "H3 Upscale Loop needs ComfyUI's graph expansion "
                "(comfy_execution.graph_utils); this build has none.")
        if dynprompt is None or unique_id is None:
            raise RuntimeError(
                "H3 Upscale Loop End did not receive DYNPROMPT/UNIQUE_ID, "
                "so it cannot clone the loop body.")
        open_node = str(flow["node_id"])
        if dynprompt.get_node(open_node).get("class_type") != "H3UpscaleLoopStart":
            raise ValueError(
                "H3 Upscale Loop End: flow did not come from a Loop Start.")

        contained = self._contained(dynprompt, unique_id, open_node)
        graph = GraphBuilder()
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            # only THIS node is renamed; everything else keeps its id so
            # the frontend can still point at the node the user authored
            clone_id = "Recurse" if node_id == str(unique_id) else node_id
            node = graph.node(original["class_type"], clone_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == str(unique_id) else node_id
            node = graph.lookup_node(clone_id)
            for key, value in (original.get("inputs") or {}).items():
                if is_link(value) and str(value[0]) in contained:
                    node.set_input(key, graph.lookup_node(
                        "Recurse" if str(value[0]) == str(unique_id)
                        else str(value[0])).out(value[1]))
                else:
                    # a link from OUTSIDE the body is kept as-is: the node
                    # it points at is not cloned, so it runs once and its
                    # output is shared by every iteration
                    node.set_input(key, value)
        opener = graph.lookup_node(open_node)
        opener.set_input("start_index", next_index)
        # an automatically chosen profile is chosen ONCE: every later
        # iteration is told the name, never asked to pick again
        opener.set_input("profile", str(flow.get("profile") or ""))

        recurse = graph.lookup_node("Recurse")
        return {
            "ui": {"text": [report]},
            "result": tuple(recurse.out(i)
                            for i in range(len(self.RETURN_TYPES))),
            "expand": graph.finalize(),
        }
