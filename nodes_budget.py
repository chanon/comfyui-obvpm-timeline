"""H3 VRAM Headroom: keep VRAM free while an H3 model samples.

WHY THIS EXISTS
---------------
Measured live 2026-09-18 on the 32 GB card, a 5.9 s prepend at 960x544:
the first sampling step took FOUR MINUTES at 100% GPU utilisation and
117 W (a computing step draws 570 W), then the run carried on at its
normal 6.6 s a step. The adapter had 35.7 GB committed against 32.6 GB
present -- ComfyUI held 30 GB and a browser, OBS, three video players and
the rest of a desktop held about 5 -- so Windows was paging VRAM over
PCIe. Nothing errors and nothing logs; it reads as a hang at "Model
Initializing", which is only core's label for "step one has not returned".

Core's dynamic loader keeps about a quarter of a gigabyte free and fills
the rest with weights. That is the right call on a card it has to itself
and the wrong one beside a desktop, and its activation estimate also
leaves out every token that is not the target: H3's keyframe rows and
references ride in a constant payload the estimate never sizes. A
both-mode window of 39 frames at 960x544 is 6,120 tokens on top of a
target of 18,870 -- a third more -- which is why prepends and bridges tip
over where a plain extend does not.

WHAT IT DOES
------------
For the duration of one sampling run it raises the loader's headroom to

    headroom_gb  +  (keyframe and reference tokens) x the measured cost

and puts it back afterwards. Raising the headroom evicts weight pages,
which then stream from RAM: a little slower per step, never a spill.
`enabled` off is a true no-op -- the model passes through untouched.

Under the classic loader (--disable-dynamic-vram) the same amount goes
into core's reserve for other applications instead, which is what
--reserve-vram sets.
"""
import logging

_LOG = logging.getLogger("obvpm.h3")

WRAPPER_KEY = "obvpm_h3_vram_headroom"
OPTION_KEY = "obvpm_h3_vram_headroom"
GB = 1024 ** 3
_PATCH_AREA = 2 * 2           # H3 patchifies 1x2x2 latent cells per token
_COND_KEYS = ("minimax_keyframes", "minimax_refs")


def _latent_tokens(latent):
    """Tokens one conditioning latent [B, C, T, H, W] adds to attention."""
    shape = getattr(latent, "shape", None)
    if shape is None or len(shape) != 5:
        return 0
    t, h, w = int(shape[2]), int(shape[3]), int(shape[4])
    # odd sizes pad up to a whole patch, exactly as the model does
    return t * ((h + 1) // 2) * ((w + 1) // 2)


def extra_tokens(conds):
    """(keyframe tokens, reference tokens) the run attends to beyond its
    target -- the larger of the conditioning sets, since positive and
    negative run as separate rows of one batch that core already doubles.
    """
    best = (0, 0)
    lists = conds.values() if isinstance(conds, dict) else (conds or [])
    for cond_list in lists:
        for cond in cond_list or []:
            if not isinstance(cond, dict):
                continue
            counts = []
            for key in _COND_KEYS:
                n = 0
                for entry in cond.get(key) or []:
                    if isinstance(entry, dict):
                        n += _latent_tokens(entry.get("latent"))
                counts.append(n)
            if sum(counts) > sum(best):
                best = tuple(counts)
    return best


def headroom_bytes(base_gb, conds, count_guides=True):
    """(bytes to keep free, keyframe tokens, reference tokens)."""
    from .nodes_joint import _ACTIVATION_MB_PER_TOKEN
    kf, refs = extra_tokens(conds) if count_guides else (0, 0)
    extra = (kf + refs) * _ACTIVATION_MB_PER_TOKEN * 1024 * 1024
    return int(max(0.0, float(base_gb)) * GB + extra), kf, refs


class _Headroom:
    """Set, and later restore, whichever reserve the running loader reads."""

    def __init__(self):
        self.previous = None      # (kind, value) while raised

    @staticmethod
    def _aimdo():
        try:
            import comfy.memory_management as mm
            if not getattr(mm, "aimdo_enabled", False):
                return None
            import comfy_aimdo.control as control
            if not hasattr(control, "set_simple_vram_headroom"):
                return None
            return control
        except Exception:
            return None

    def raise_to(self, want):
        """Returns the loader it spoke to, or None if nothing changed."""
        if self.previous is not None:
            self.restore()
        control = self._aimdo()
        if control is not None:
            try:
                now = int(control.get_simple_vram_headroom())
                if want > now:
                    control.set_simple_vram_headroom(int(want))
                    self.previous = ("dynamic", now)
                    return "dynamic"
                return None
            except Exception:
                _LOG.exception("obvpm.h3 vram headroom: the dynamic loader "
                               "refused the headroom")
                return None
        try:
            import comfy.model_management as mm
            now = int(mm.EXTRA_RESERVED_VRAM)
            if want > now:
                mm.EXTRA_RESERVED_VRAM = int(want)
                self.previous = ("classic", now)
                return "classic"
        except Exception:
            _LOG.exception("obvpm.h3 vram headroom: could not set core's "
                           "reserve")
        return None

    def restore(self):
        prev, self.previous = self.previous, None
        if prev is None:
            return
        kind, value = prev
        try:
            if kind == "dynamic":
                control = self._aimdo()
                if control is not None:
                    control.set_simple_vram_headroom(int(value))
            else:
                import comfy.model_management as mm
                mm.EXTRA_RESERVED_VRAM = int(value)
        except Exception:
            _LOG.exception("obvpm.h3 vram headroom: could not restore the "
                           "previous headroom")


_STATE = _Headroom()


def prepare_sampling_with_headroom(executor, model, noise_shape, conds,
                                   *args, **kwargs):
    """PREPARE_SAMPLING: the conditioning is resolved here and the weights
    are loaded by the call we wrap, so this is the last moment the
    headroom can shape that load and the first it can be sized."""
    options = (kwargs.get("model_options") or {}).get(OPTION_KEY)
    if options is None and len(args) >= 1 and isinstance(args[0], dict):
        options = args[0].get(OPTION_KEY)
    if options:
        want, kf, refs = headroom_bytes(options.get("headroom_gb", 0.0),
                                        conds,
                                        options.get("count_guides", True))
        kind = _STATE.raise_to(want)
        if kind:
            _LOG.info("obvpm.h3 vram headroom: keeping %.1f GB free while "
                      "sampling (%.1f GB asked%s; %s loader)",
                      want / GB, float(options.get("headroom_gb", 0.0)),
                      "" if not (kf or refs) else
                      " + %.1f GB for %d keyframe and %d reference tokens "
                      "core does not count" % (
                          (want / GB) - float(options.get("headroom_gb", 0.0)),
                          kf, refs),
                      kind)
    return executor(model, noise_shape, conds, *args, **kwargs)


def outer_sample_restoring_headroom(executor, *args, **kwargs):
    """OUTER_SAMPLE wraps prepare_sampling and the whole run, so its
    `finally` is the one place that still runs after an interrupt."""
    try:
        return executor(*args, **kwargs)
    finally:
        _STATE.restore()


class H3VramHeadroom:
    """Keep VRAM free while this model samples."""

    CATEGORY = "obvpm/h3"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Keeps some VRAM free while this H3 model samples, so a run beside "
        "a browser, a recorder or a video player streams a few more weights "
        "from RAM instead of spilling into system memory -- which shows as "
        "minutes stuck at \"Model Initializing\" with the GPU busy and "
        "drawing little power. Adds room for the keyframe rows and "
        "references of the run, which core's own estimate leaves out "
        "(prepends and bridges carry the most). Put it anywhere on the "
        "model path before the sampler. Off is a true pass-through."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off passes the model through untouched."}),
                "headroom_gb": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 24.0, "step": 0.5,
                    "tooltip": "VRAM to keep free while sampling, for the "
                               "desktop and whatever else is open. About "
                               "what the other programs hold: 1 on a bare "
                               "desktop, 3 with a browser open, 5 or 6 "
                               "while recording. Too high only costs speed "
                               "(more weights stream from RAM); too low is "
                               "the stall this node exists for."}),
                "count_guides": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add room for the run's keyframe rows and "
                               "references on top, from their actual sizes. "
                               "Core sizes the target latent only."}),
            },
        }

    def patch(self, model, enabled, headroom_gb, count_guides):
        if not enabled:
            return (model,)
        import comfy.patcher_extension
        model = model.clone()
        model.model_options[OPTION_KEY] = {
            "headroom_gb": float(headroom_gb),
            "count_guides": bool(count_guides),
        }
        W = comfy.patcher_extension.WrappersMP
        model.add_wrapper_with_key(W.PREPARE_SAMPLING, WRAPPER_KEY,
                                   prepare_sampling_with_headroom)
        model.add_wrapper_with_key(W.OUTER_SAMPLE, WRAPPER_KEY,
                                   outer_sample_restoring_headroom)
        return (model,)
