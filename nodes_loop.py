"""The upscale loop: one visible body, run once per timeline clip.

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

    H3 Joint Store (outside; the sampled timeline, on disk)
        |  joint_path
        v
    Loop Start --flow-------> H3 Joint Slice --latent/pins--> decode, save
        |      --out_folder-> the save node's base_folder          |
        |  flow                                                    |
        +---------------------> Loop End <--after-------------------+

The traversal is adapted from Ethanfel's SxCP loop nodes in
ComfyUI-Prompt-Builder, by way of ComfyUI-MiniMaxH3-Contex-Loop's
`chain_nodes.py` (GPL-3.0, as this pack is).

WHY THE SAMPLING IS NOT IN THE LOOP: the timeline is sampled as one
latent (nodes_joint.py), which is what makes the joins seamless, and the
sampled file is the loop's input. Each iteration only slices one clip's
span out of it, decodes it, trims it as the source was trimmed, saves it
and records it in the profile manifest -- so the loop is resumable per
clip, and the sampling is never repeated because a save step failed.
"""

import logging
import os

from . import mctx
from . import nodes_joint
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
    """Opens the loop: resolves which clip this iteration slices and saves."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "start"
    RETURN_TYPES = (wt.LOOP, "STRING", "INT", "INT")
    RETURN_NAMES = ("flow", "out_folder", "index", "total")
    DESCRIPTION = (
        "Opens the loop that turns a jointly refined timeline into refined "
        "takes. Everything between this node and H3 Upscale Loop End is the "
        "loop body and runs once per clip: H3 Joint Slice cuts the clip's "
        "span out of the joint latent, the body decodes and saves it. "
        "Resumable: the profile folder records what is done, so pressing "
        "Run again continues rather than restarting."
    )
    OUTPUT_TOOLTIPS = (
        "Wire to H3 Joint Slice and to Loop End. Carries which node opened "
        "the loop and which clip this iteration is.",
        "The profile folder the joint latent sits in (<base_folder>/_upscale/"
        "<profile>). Wire to the save node's base_folder.",
        "0-based position in the timeline.",
        "How many clips the timeline holds.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joint_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "From H3 Joint Store: the sampled timeline "
                               "latent. Its folder is the profile folder "
                               "the refined clips land in."}),
                "start_index": ("INT", {
                    "default": -1, "min": -1, "max": 4096,
                    "tooltip": "-1 = continue where the profile left off, "
                               "which is what you want. A number forces the "
                               "starting clip, for re-doing one. The loop "
                               "drives this on every iteration after the "
                               "first."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # the clip this resolves to depends on what is on disk, not on any
        # widget, so a byte-identical prompt must still re-run
        return float("nan")

    def start(self, joint_path, start_index=-1, unique_id=None, **_):
        rel_path = str(joint_path or "").strip().replace("\\", "/").strip("/")
        if not rel_path:
            raise ValueError("H3 Upscale Loop: wire joint_path from H3 Joint "
                             "Store -- the loop slices the sampled timeline.")
        try:
            path = nodes_load.resolve_clip_path(rel_path)
        except ValueError:
            raise ValueError(
                "H3 Upscale Loop: the joint file %s is not there. H3 Joint "
                "Store writes it; if the profile folder was deleted, run "
                "again and the store re-creates it." % rel_path)
        record = nodes_joint.read_record(path)
        clips = list(record["clips"])
        lines = list(record.get("lines") or clips)
        stamp = record.get("stamp")
        rel_folder = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
        folder = os.path.dirname(path)

        document = upscale.read_manifest(folder)
        headers = nodes_joint.clip_headers(clips)
        deps = self._joins(headers)
        order = self._refine_order(len(clips), deps)
        # what this joint sampling has already delivered: a sliced clip
        # counts only while it was cut from THIS sampling and the refined
        # neighbours its lineage points at are the ones present now
        valid = upscale.refined_for(
            document, folder, clips, {i: [j for j, _ in d] for i, d in deps.items()},
            stamp)
        remaining = [i for i in order if i not in valid]

        if int(start_index) < 0:
            if not remaining:
                raise ValueError(
                    "H3 Upscale Loop: %s has already delivered all %d clip(s) "
                    "of this timeline from the joint file that is there. To "
                    "SAMPLE AGAIN, turn H3 Joint Store's reuse_existing off "
                    "(or give it a new profile name); the loop then delivers "
                    "every clip again. To redo one clip from the same "
                    "sampling, set start_index." % (rel_folder, len(clips)))
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
                    "not been delivered yet -- it cannot be redone on its "
                    "own until they are." % (index, missing))
            following = next((i for i in remaining if i != index), None)

        clip = clips[index]
        # what this rendering's lineage points at, recorded with it so a
        # later run can tell whether the join it needs is the join it got
        pinned_to = {clips[j]: valid[j]["output"] for j, _ in deps[index]}
        _LOG.info("obvpm.h3: slice %s [position %d, %d of %d] -> %s%s",
                  clip, index, len(valid) + 1, len(clips), rel_folder,
                  (" lineage -> refined %s"
                   % ", ".join(str(j) for j, _ in deps[index]))
                  if deps[index] else " (no held join)")
        # the id is how Loop End finds which node opened the loop, and
        # therefore where the body starts
        flow = {"node_id": str(unique_id) if unique_id is not None else None,
                "index": index, "next_index": following, "total": len(clips),
                "done": len(valid) + 1, "pinned_to": pinned_to,
                "lines": lines, "clips": clips, "source": clip,
                "folder": folder, "rel_folder": rel_folder,
                "joint_path": path, "stamp": stamp}
        return (flow, rel_folder, index, len(clips))

    @staticmethod
    def _joins(headers):
        """One decision per adjacent pair: whose lineage points at whom.

        Returns {i: [(j, pin)]} -- clip i is delivered AFTER clip j, and
        its refined pins point at j's REFINED output.

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
        timeline of plain cuts is delivered exactly front to back.
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


class H3UpscaleLoopEnd:
    """Closes the loop: commits this iteration, then expands the next."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "end"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "profile_folder")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Closes the upscale loop. Records the clip this iteration wrote, "
        "then re-runs the whole body for the next one until the timeline "
        "is done. Everything between Loop Start and this node is the body "
        "-- it is found by following the wires, so there is no list to "
        "keep up to date."
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
        # that is not there, and the next slice's lineage would point at it.
        document = upscale.read_manifest(folder)
        upscale.record(document, flow["source"], os.path.basename(str(after)),
                       flow.get("stamp"), pinned_to=flow.get("pinned_to"))
        # the timeline's own lines, so the assembly plays the same cut
        document["lines"] = flow.get("lines") or []
        upscale.write_manifest(folder, document)

        done = int(flow.get("done") or index + 1)
        report = "delivered %d of %d: %s -> %s" % (
            done, total, flow["source"], os.path.basename(str(after)))
        _LOG.info("obvpm.h3: %s", report)
        # The next clip is whatever the DEPENDENCY order says, not the
        # next delivery position: a join carried by a tail is delivered
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
        model loaders and the joint sampling -- they are upstream of End,
        but they do not depend on Start, so cloning them would re-run the
        sampling every iteration. "Everything downstream of Start" would
        sweep in anything else the user hung off its outputs.

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

        recurse = graph.lookup_node("Recurse")
        return {
            "ui": {"text": [report]},
            "result": tuple(recurse.out(i)
                            for i in range(len(self.RETURN_TYPES))),
            "expand": graph.finalize(),
        }
