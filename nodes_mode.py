"""One switch that decides which half of a graph runs.

A workflow that can both GENERATE a take and REFINE a finished timeline
holds two branches that must never run together: they want the same GPU,
they write different folders, and the refine branch raises when there is
nothing left to refine. Muting the output nodes by hand works and is what
this graph did first, but it is state kept in three places that has to be
flipped in step, and getting it half-right silently runs both.

So the branch selection becomes a value, and the value comes from the
Timeline -- the node that already knows which clips exist and which one
is being continued.

TWO MECHANISMS, AND BOTH ARE NEEDED. They cut in opposite directions:

  * a LAZY input prunes what is UPSTREAM. `check_lazy_status` is asked
    what this node needs before its inputs are evaluated, and anything
    it does not ask for is never executed. This is why a gate beats
    MuteGate, whose own documentation says "the nodes upstream of
    'input' still run" -- an eager input has already been computed by
    the time the node can object to it.
  * an `ExecutionBlocker` prunes what is DOWNSTREAM, output nodes
    included. Every OUTPUT_NODE is an execution root, so a save node is
    not reached THROUGH anything and cannot be pruned by laziness; the
    only way to stop one is to hand it an input it must refuse. Core's
    own docstring names this exact case: "You want to conditionally
    prevent an output node from executing."

A gate placed on one wire therefore switches off everything that wire
feeds, however many roots hang off it, plus everything that feeds the
gate and nothing else.

WHERE TO PUT ONE. As far upstream in the branch as a single wire can
reach, because the block travels forwards: put it on the last shared
value before the branch's own work begins. In h3_obvpm_r2v that is the
Timeline's `length` into the reference-to-video node (so the 32B encode,
the sampler, the decode and the save all sit behind it) and the
Timeline's `sequence` into Loop Start (so the whole refine loop,
including its own resume check, sits behind it). Nodes that merely feed
the gate -- image loaders, a seed -- still run; they are cheap, and
buying their silence would cost a gate each.
"""

import logging

try:
    from comfy_execution.graph_utils import ExecutionBlocker, is_link
except ImportError:  # pragma: no cover - only in a build without expansion
    ExecutionBlocker = None
    is_link = None

from .common import ANY

_LOG = logging.getLogger("obvpm.h3")

GENERATION = "generation"
UPSCALE = "upscale"
RUN_MODES = (GENERATION, UPSCALE)

RUN_MODE_TOOLTIP = (
    "Which half of the workflow this Run is for. GENERATION renders the "
    "next take from the pin; UPSCALE walks the timeline and refines every "
    "clip it holds. Wire this to H3 Run Mode Gate nodes -- one per branch "
    "-- and the unselected branch is pruned before it costs anything."
)


def check_run_mode(value, who="H3 Run Mode Gate"):
    """The run mode as a known constant, or a loud refusal.

    Loud on purpose. A run mode that is not recognised would leave every
    gate closed, which looks exactly like a graph that ran and did
    nothing -- the most expensive kind of quiet failure, because you
    only find out after waiting for it.
    """
    text = str(value or "").strip().lower()
    if text not in RUN_MODES:
        raise ValueError(
            "%s: %r is not a run mode (this build knows %s). The run mode "
            "comes from the H3 Timeline's run_mode widget and travels on "
            "its run_mode output." % (who, value, " / ".join(RUN_MODES)))
    return text


class H3RunModeGate:
    """Pass a value through in one run mode; block the branch in the other."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "gate"
    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    DESCRIPTION = (
        "A one-wire on/off switch for a whole branch. Put it on the last "
        "wire before the work you want to make conditional: when the "
        "Timeline's run_mode matches pass_when the value goes through "
        "unchanged, and when it does not, everything downstream is "
        "skipped -- save nodes and previews included -- while everything "
        "that only feeds this gate is never evaluated at all."
    )
    OUTPUT_TOOLTIPS = (
        "The value, untouched, when this gate is open. When it is shut "
        "the wire carries a block instead, and every node downstream is "
        "quietly skipped.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_mode": ("STRING", {
                    "forceInput": True,
                    "tooltip": "From the H3 Timeline's run_mode output. "
                               "One source for both gates, so the two "
                               "branches cannot disagree about which one "
                               "is running."}),
                "pass_when": (list(RUN_MODES), {
                    "default": GENERATION,
                    "tooltip": "The run mode this gate is open in. Give "
                               "the generation branch a gate set to "
                               "'generation' and the refine branch one "
                               "set to 'upscale'."}),
            },
            "optional": {
                "value": (ANY, {
                    "lazy": True,
                    "forceInput": True,
                    "tooltip": "Anything at all -- the gate does not look "
                               "at it. LAZY: while the gate is shut this "
                               "is never asked for, so whatever produces "
                               "it does not run either."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    def check_lazy_status(self, run_mode, pass_when, value=None,
                          dynprompt=None, unique_id=None):
        # asked BEFORE the inputs are evaluated: naming nothing here is
        # what keeps the upstream from running at all
        if check_run_mode(run_mode) != pass_when:
            return []
        if value is not None:
            return []
        if not self._is_wired(dynprompt, unique_id):
            # core would refuse this too, but as "there is no input to
            # that node at all", which describes the wrong problem
            raise ValueError(
                "H3 Run Mode Gate: nothing is wired into `value`, so an "
                "open gate has nothing to pass on. A gate goes ON a wire "
                "-- the value in, the same value out -- not beside one.")
        return ["value"]

    @staticmethod
    def _is_wired(dynprompt, unique_id):
        """Is `value` a link? Assume yes when we cannot tell."""
        if dynprompt is None or unique_id is None or is_link is None:
            return True
        try:
            inputs = dynprompt.get_node(str(unique_id)).get("inputs") or {}
        except Exception:
            return True
        return "value" in inputs and is_link(inputs["value"])

    def gate(self, run_mode, pass_when, value=None, dynprompt=None,
             unique_id=None):
        mode = check_run_mode(run_mode)
        if mode != pass_when:
            if ExecutionBlocker is None:
                raise RuntimeError(
                    "H3 Run Mode Gate needs ComfyUI's ExecutionBlocker "
                    "(comfy_execution.graph_utils); this build has none.")
            _LOG.info("obvpm.h3: run mode is %s, so the %s branch is "
                      "switched off", mode, pass_when)
            # silent: a shut gate is the normal case, not a failure, and
            # a message here would report every skipped node as an error
            return (ExecutionBlocker(None),)
        return (value,)
