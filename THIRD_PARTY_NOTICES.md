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

The upscale loop (`nodes_loop.py`) finds its body by reachability
between an opening and a closing node and re-expands it per iteration,
the traversal that pack's `chain_nodes.py` uses (itself after Ethanfel's
SxCP loop nodes in ComfyUI-Prompt-Builder). That pack's Drift-Control AV
rule was also implemented here for per-clip refines and retired when
the joint refine replaced them. No code is vendored.

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

## Comfyui-MMH3-UltimateUpscale (MIT)

Copyright (c) 2026 bbaudio-2025
https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale

A sequential chunked upscale-and-refine for H3. No code is vendored. It
served as the control in the 2026-09-08 investigation recorded in
`docs/joint-refine-investigation.md`: refining the same pair of clips
with the same conditioning and seed, it hallucinated on our
whole-timeline upscaled prior and was clean with its own per-chunk
upscale, which is what located the fault in the upscaling stage.
