"""Drift control: schedule-matched junction context for the refine pass.

A masked pin holds its window at mask 0, and core labels a mask-0 row as
CLEAN conditioning for every step of the run. In a generation that is
fine: the free rows start from pure noise and have nothing but the held
rows to continue, so they continue them. In a refine the free rows
already carry content (the clip's own upscaled latents, noised to the
pass's small sigma), and a clean wall next to noisy content is measured
to pull the prediction beside it -- the delivered picture right after a
pinned head comes out ~2 luma off the parent's tail and recovers over
two to three seconds, at every join, whatever the junction mode.

Two ways to take the wall down, both on a cloned MODEL and both video
only (audio keeps its policy mask):

  blend    Contex-Loop's Drift-Control AV rule. Core's inpaint step mixes
           x = m*x + (1-m)*clean; the held rows get m = sigma_next /
           sigma_current every step, so the context sits one step ahead
           of the content and is labelled to match. Validated by them at
           20 steps. MEASURED HERE TO FAIL under a 5-step turbo schedule
           (0.83 .79 .72 .62 .43 0): the ratios are .94 .92 .86 .70 then
           0, so the context is nearly free for every step that matters,
           drifts with the sampler's state, and snaps clean for the last
           step, which is 43% of the range. Kept for schedules with a
           small final step.

  renoise  The held rows are REBUILT every step from the carried clean
           latent, noised to the chosen level with the run's own fixed
           noise tensor -- the way ordinary inpainting samplers treat a
           masked region. The row's noise then IS what its label says,
           by construction, whatever the sampler did last step, so it
           works at any step count and nothing can drift. Core's post-
           step x0 blend still returns the rows to clean, so the join and
           the trim geometry are unchanged. The level is free to choose:
           `matched` = the content's own sigma (the context is
           indistinguishable from content at every step; no wall at any
           step, the big last one included), `ahead` = sigma_next /
           sigma_current, `constant` = a fixed fraction.

Three hooks that must agree: core's `denoise_mask_function` (once per
step, before the mix, with the packed [B,1,N] mask) decides the per-row
level for this sigma; a SAMPLER_SAMPLE wrapper swaps the model's
`scale_latent_inpaint` for the run so the injected value for held rows
is the re-noised clean latent (renoise only); an APPLY_MODEL wrapper
hands H3 the levels as its per-row timestep labels, because the label
mask was pooled once from the STATIC mask before sampling began and
would otherwise still say "clean".

Credit: the blend rule and the taper are from ComfyUI-MiniMaxH3-Contex-
Loop (NikoDemon80, GPL-3.0); see THIRD_PARTY_NOTICES.md. Written against
our own pins layout.
"""

import logging
import math

import torch

from . import wiretypes as wt

_LOG = logging.getLogger("obvpm.h3")

WRAPPER_KEY = "obvpm_h3_drift_mask"
SAMPLER_WRAPPER_KEY = "obvpm_h3_drift_mask_inject"
MIXES = ("renoise", "blend")
LEVELS = ("matched", "ahead", "constant")
DEFAULT_TAPER_STEPS = 0
# core pools the label mask to the token grid and ceil-quantizes it to
# 1/256 (model_base.MiniMaxH3._token_grid_masks); the same quantum keeps
# the wrapper's mask bf16-exact
MASK_QUANTUM = 256.0


def schedule_values(sigmas):
    """Finite, non-negative schedule values, descending, deduplicated."""
    if torch.is_tensor(sigmas):
        values = sigmas.detach().float().reshape(-1).tolist()
    else:
        values = list(sigmas or ())
    out = set()
    for v in values:
        v = float(v)
        if math.isfinite(v) and v >= 0.0:
            out.add(v)
    return sorted(out, reverse=True)


def next_sigma(current, sigmas):
    """The next strictly lower sigma in the schedule, else 0."""
    current = float(current)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    tol = max(1e-7, abs(current) * 1e-6)
    for v in schedule_values(sigmas):
        if v < current - tol:
            return v
    return 0.0


def matched_ratio(current, sigmas):
    """sigma_next / sigma_current, clamped to a mask value."""
    current = float(current)
    if not math.isfinite(current) or current <= 0.0:
        return 0.0
    return max(0.0, min(1.0, next_sigma(current, sigmas) / current))


def window_weights(steps, taper, level, place):
    """Per-latent-step level for one held window at this evaluation.

    Rows away from the join carry `level`; the `taper` rows nearest the
    join fall (taper-1)/taper ... 1/taper, 0 times `level`, so the row
    AT the join is exact. A before-pin's join is the window's END, an
    after-pin's its START.
    """
    steps = int(steps)
    taper = max(0, min(int(taper), steps))
    level = max(0.0, min(1.0, float(level)))
    weights = [level] * (steps - taper)
    weights += [level * (taper - 1 - i) / float(taper) for i in range(taper)]
    if place == "after":
        weights.reverse()
    return weights


def _sigma_value(sigma):
    return float(torch.as_tensor(sigma).detach().float().reshape(-1)[0])


class DriftMaskState:
    """Per-run state shared by the three hooks."""

    def __init__(self, windows, taper_steps=DEFAULT_TAPER_STEPS, mix="renoise",
                 level="matched", level_value=0.5):
        # [(place, steps)] -- the held windows, in latent steps
        self.windows = [(str(p), int(s)) for p, s in windows]
        self.taper_steps = int(taper_steps)
        if mix not in MIXES:
            raise ValueError("drift mask: unknown mix %r" % (mix,))
        if level not in LEVELS:
            raise ValueError("drift mask: unknown level rule %r" % (level,))
        self.mix = mix
        self.level = level
        self.level_value = max(0.0, min(1.0, float(level_value)))
        # per-row levels for the CURRENT step, as a [1,1,T,1,1] video
        # tensor (None until the first step); what the labels and the
        # re-noising read
        self.row_levels = None
        self.label_mask = None
        self.current_sigma = None
        self.evaluations = 0

    # -- the level rule --------------------------------------------------

    def level_for(self, current, sigmas):
        if self.level == "matched":
            return 1.0
        if self.level == "ahead":
            return matched_ratio(current, sigmas)
        return self.level_value

    def _levels(self, latent_t, device, dtype, level):
        """[1,1,T,1,1] of per-row levels: 0 where nothing is held."""
        rows = torch.zeros((1, 1, latent_t, 1, 1), device=device, dtype=dtype)
        for place, steps in self.windows:
            if steps <= 0 or steps > latent_t:
                raise ValueError(
                    "obvpm.h3 drift mask: a %d-step window does not fit a "
                    "%d-step latent." % (steps, latent_t))
            w = torch.tensor(window_weights(steps, self.taper_steps, level, place),
                             device=device, dtype=dtype).view(1, 1, steps, 1, 1)
            if place == "before":
                rows[:, :, :steps] = torch.maximum(rows[:, :, :steps], w)
            else:
                rows[:, :, -steps:] = torch.maximum(rows[:, :, -steps:], w)
        return rows

    def _held(self, latent_t, device, dtype):
        """[1,1,T,1,1] boolean-ish: 1 on rows inside a held window."""
        rows = torch.zeros((1, 1, latent_t, 1, 1), device=device, dtype=dtype)
        for place, steps in self.windows:
            if place == "before":
                rows[:, :, :steps] = 1.0
            else:
                rows[:, :, -steps:] = 1.0
        return rows

    # -- step 1: the sampler's mask ---------------------------------------

    def rewrite(self, sigma, packed, shapes, sigmas):
        """Decide this step's levels; return (packed mask, level).

        blend:   held rows of the returned mask carry the level (max with
                 the static mask, so a row it already frees stays free).
        renoise: the returned mask is the static one -- core takes the
                 injected value whole for held rows, and the injection
                 (scale_latent_inpaint) is where the level acts.
        Either way the label mask is rebuilt from the levels.
        """
        current = _sigma_value(sigma)
        level = self.level_for(current, sigmas)
        video_shape = tuple(int(v) for v in shapes[0])
        video_n = math.prod(video_shape[1:])
        if int(packed.shape[-1]) < video_n:
            raise ValueError(
                "obvpm.h3 drift mask: the packed sampler mask (%d) is "
                "shorter than the video stream (%d)."
                % (int(packed.shape[-1]), video_n))
        batch = int(packed.shape[0])
        video = packed[:, :, :video_n].reshape((batch,) + video_shape[1:]).clone()
        latent_t = video_shape[2]
        levels = self._levels(latent_t, video.device, video.dtype, level)
        self.row_levels = levels
        self.current_sigma = current
        # the labels H3 reads: the level on held rows, the static mask
        # elsewhere; same quantum as core's own pooling (our rows are
        # uniform across the frame, so per-patch pooling is the identity)
        held = self._held(latent_t, video.device, video.dtype)
        label = video[:1, :1] * (1.0 - held) + torch.maximum(video[:1, :1], levels) * held
        self.label_mask = torch.ceil(label.float() * MASK_QUANTUM) / MASK_QUANTUM
        if self.mix == "blend":
            video = torch.maximum(video, levels.expand_as(video))
            out = packed.clone()
            out[:, :, :video_n] = video.reshape(batch, 1, -1)
            return out, level
        return packed, level

    # -- step 2: what core injects for held rows (renoise) ------------------

    def inject(self, original, base, sigma, noise, latent_image, x=None,
               denoise_mask=None, **kwargs):
        """Replacement for the model's scale_latent_inpaint during a run.

        Calls the original for everything (audio scaling, the cond-
        timestep aug), then overwrites the held VIDEO rows with the clean
        carried latent noised to level * sigma using the run's own noise
        -- the flow forward process, sigma*(s*noise) + (1-sigma)*clean,
        exactly as the sampler noised the run at its first step.
        """
        injected = original(sigma=sigma, noise=noise, latent_image=latent_image,
                            x=x, denoise_mask=denoise_mask, **kwargs)
        if self.mix != "renoise" or self.row_levels is None:
            return injected
        shapes = getattr(base, "latent_shapes", None)
        if not shapes or len(shapes) < 2 or noise is None or latent_image is None:
            return injected
        video_shape = tuple(int(v) for v in shapes[0])
        video_n = math.prod(video_shape[1:])
        batch = int(injected.shape[0])
        inj_v = injected[:, :, :video_n].reshape((batch,) + video_shape[1:]).clone()
        noise_v = noise[:, :, :video_n].reshape((batch,) + video_shape[1:])
        clean_v = latent_image[:, :, :video_n].reshape((batch,) + video_shape[1:])
        current = _sigma_value(sigma)
        scale = float(getattr(base.model_sampling, "noise_scale", 1.0))
        held = self._held(video_shape[2], inj_v.device, inj_v.dtype)
        sig = (self.row_levels.to(inj_v.device, inj_v.dtype) * current)
        noised = (sig * (scale * noise_v.to(inj_v.dtype))
                  + (1.0 - sig) * clean_v.to(inj_v.dtype))
        inj_v = inj_v * (1.0 - held) + noised * held
        out = injected.clone()
        out[:, :, :video_n] = inj_v.reshape(batch, 1, -1)
        return out

    # -- hooks -----------------------------------------------------------

    def denoise_mask_function(self, sigma, denoise_mask, extra_options=None):
        extra = extra_options or {}
        guider = extra.get("model")
        base = getattr(guider, "inner_model", None)
        shapes = getattr(base, "latent_shapes", None)
        if not shapes or len(shapes) < 2:
            if self.evaluations == 0:
                _LOG.warning("obvpm.h3 drift mask: the sampler is not running "
                             "a packed AV latent; the mask is left as it is")
            self.evaluations += 1
            return denoise_mask
        out, level = self.rewrite(sigma, denoise_mask, shapes,
                                  extra.get("sigmas", ()))
        current = _sigma_value(sigma)
        _LOG.info("obvpm.h3 drift mask (%s/%s): step %d sigma %.4f -> %.4f, "
                  "held rows at level %.3f = sigma %.4f (taper %d)", self.mix,
                  self.level, self.evaluations, current,
                  next_sigma(current, extra.get("sigmas", ())), level,
                  level * current, self.taper_steps)
        self.evaluations += 1
        return out

    def apply_model_wrapper(self, executor, *args, **kwargs):
        if self.label_mask is not None:
            x = args[0] if args else kwargs.get("x")
            mask = self.label_mask
            if x is not None:
                mask = mask.to(device=x.device)
            kwargs["denoise_mask"] = mask
        return executor(*args, **kwargs)

    def sampler_sample_wrapper(self, executor, guider, *args, **kwargs):
        """Swap the model's scale_latent_inpaint for the run (renoise)."""
        base = getattr(guider, "inner_model", None)
        if self.mix != "renoise" or base is None or not hasattr(base, "scale_latent_inpaint"):
            return executor(guider, *args, **kwargs)
        original = base.scale_latent_inpaint
        state = self

        def patched(sigma, noise, latent_image, x=None, denoise_mask=None, **kw):
            return state.inject(original, base, sigma, noise, latent_image,
                                x=x, denoise_mask=denoise_mask, **kw)

        base.scale_latent_inpaint = patched
        try:
            return executor(guider, *args, **kwargs)
        finally:
            # the BaseModel outlives the run; leave it as we found it
            try:
                del base.scale_latent_inpaint
            except AttributeError:
                base.scale_latent_inpaint = original


def held_windows(pins):
    """(place, steps) for every pin that holds rows through the mask."""
    out = []
    for pin in pins or []:
        mode = (pin.get("spec") or {}).get("mode", "guide")
        if mode not in ("masked", "both"):
            continue
        if pin.get("place") not in ("before", "after"):
            continue
        out.append((pin["place"], int(pin["steps"])))
    return out


def install(model, windows, taper_steps=DEFAULT_TAPER_STEPS, mix="renoise",
            level="matched", level_value=0.5):
    """Clone the MODEL and install the coupled hooks."""
    from comfy.patcher_extension import WrappersMP

    inner = getattr(model, "model", None)
    if inner is None or inner.__class__.__name__ != "MiniMaxH3":
        raise ValueError(
            "H3 MCtx Drift Mask: the connected MODEL is not MiniMax H3 "
            "(got %s)." % (inner.__class__.__name__ if inner else "nothing"))
    patched = model.clone()
    options = patched.model_options
    if callable(options.get("denoise_mask_function")):
        raise ValueError(
            "H3 MCtx Drift Mask: the MODEL already carries a dynamic "
            "denoise-mask patch (Differential Diffusion, or a second Drift "
            "Mask); remove it from this model path.")
    state = DriftMaskState(windows, taper_steps, mix, level, level_value)
    patched.set_model_denoise_mask_function(state.denoise_mask_function)
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, WRAPPER_KEY,
                                 state.apply_model_wrapper)
    patched.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, SAMPLER_WRAPPER_KEY,
                                 state.sampler_sample_wrapper)
    patched.model_options[WRAPPER_KEY] = state
    _LOG.info("obvpm.h3 drift mask: installed %s/%s%s for %s (taper %d step(s))",
              mix, level,
              " %.2f" % state.level_value if level == "constant" else "",
              ", ".join("%s-pin %d steps" % w for w in state.windows),
              state.taper_steps)
    return patched


class H3MCtxDriftMask:
    CATEGORY = "obvpm/h3"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Drift control for a refine's junction pins: takes down the clean "
        "wall a held window puts next to the content being refined. "
        "renoise rebuilds the held rows every step from the carried latent, "
        "noised to the chosen level with the run's own noise (matched = the "
        "content's own sigma, so the context is indistinguishable from "
        "content at every step); blend is Contex-Loop's sigma_next/sigma "
        "mix, which needs a schedule with a small last step (fails under "
        "4-5 step turbo). Labels follow the levels, so what the model is "
        "told matches what it is given. Put it on the refine sampler's "
        "MODEL path; with no held pins, or `enabled` off, the model passes "
        "through untouched. Video only."
    )
    OUTPUT_TOOLTIPS = (
        "The model with the hooks installed (a clone; the input model is "
        "not changed).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The MiniMax H3 model the refine samples with."}),
                "pins": (wt.PINS, {
                    "tooltip": "The resolved pins from H3 MCtx Apply Pins -- "
                               "which rows are held, and on which side."}),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = pass the model through. Wire it from "
                               "Loop Start's `drift` output so the profile "
                               "hash and the model agree."}),
                "mix": (list(MIXES), {
                    "default": "renoise",
                    "tooltip": "renoise: held rows are rebuilt each step from "
                               "the carried latent, noised to the level with "
                               "the run's own noise -- works at any step "
                               "count. blend: Contex-Loop's mix of the "
                               "sampler's state with the clean latent -- "
                               "needs a schedule whose last step is small."}),
                "level": (list(LEVELS), {
                    "default": "matched",
                    "tooltip": "How noisy the held rows are at each step. "
                               "matched: the content's own sigma (no wall at "
                               "any step). ahead: sigma_next / sigma_current "
                               "(one step ahead of the content; snaps clean "
                               "on the last step). constant: level_value x "
                               "sigma."}),
                "level_value": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "The fraction for level = constant; ignored "
                               "otherwise."}),
                "taper_steps": ("INT", {
                    "default": DEFAULT_TAPER_STEPS, "min": 0, "max": 12,
                    "tooltip": "Latent steps nearest the join that fall from "
                               "the level to exact (4 = .75/.50/.25/0, "
                               "Contex-Loop's recipe). 0 = the whole window "
                               "at the level, join row included -- the "
                               "output is returned to clean after every "
                               "step regardless, so the trim is unaffected."}),
            },
        }

    def patch(self, model, pins, enabled=True, mix="renoise", level="matched",
              level_value=0.5, taper_steps=DEFAULT_TAPER_STEPS):
        windows = held_windows(pins)
        if not enabled or not windows:
            _LOG.info("obvpm.h3 drift mask: %s; model passes through",
                      "disabled" if not enabled else "no held pins")
            return (model,)
        return (install(model, windows, taper_steps, mix, level, level_value),)
