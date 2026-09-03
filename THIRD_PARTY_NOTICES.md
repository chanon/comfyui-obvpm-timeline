# Third-party notices

This pack's own code is licensed under **GPL-3.0** (see `LICENSE`).
No third-party code is vendored; the acknowledgments below credit
projects whose published findings and designs informed this pack.

## ComfyUI-H3-Motion-Context

Copyright (C) 2026 NikoDemon80
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

The first pack to generalize H3 keyframe anchoring to arbitrary
positions (a capability ComfyUI core has since merged natively, PR
#15439, which this pack now uses directly). Mechanism findings from
that project informed the pins design: tail slicing soundness rules,
audio window end-alignment on the shared timeline, and trim behavior.

## ComfyUI-MiniMaxH3-Contex-Loop (GPL-3.0)

Copyright (C) 2026 NikoDemon80
https://github.com/NikoDemon80/ComfyUI-MiniMaxH3-Contex-Loop

`H3 MCtx Drift Mask` (`nodes_drift.py`) implements that pack's
Drift-Control AV rule: at every model evaluation the carried context
rows are held at `sigma_next / sigma_current` rather than at zero, with
the four rows nearest the generated content tapering .75/.50/.25/0 so
the boundary stays exact, and an apply-model wrapper hands H3 the same
mask as its per-row timestep labels. The two-hook structure (sampler
denoise-mask function plus apply-model wrapper) and the taper are
theirs; the code is written against this pack's own pin layout, with
nothing vendored.

## ComfyUI-MMH3Tools (MIT)

Copyright (c) ckinpdx
https://github.com/ckinpdx/ComfyUI-MMH3Tools

No code vendored verbatim; the following designs and findings are
adopted here (primarily `frames.py`), with thanks:

- The off-grid-safe `frame_at_latent` general inverse over the
  (1,4,4,4,4) frame-per-token cycle.
- The cumulative-total ("boundary difference") audio arithmetic rule and
  its rationale (round(frames/24*40) does not distribute over addition).
- The trim-dilemma rationale (latent-domain concatenation is unsound),
  which is why trimming here happens on decoded frames.
