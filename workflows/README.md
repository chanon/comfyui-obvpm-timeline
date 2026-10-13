# OBVPM Timeline Workflows

This folder contains ComfyUI workflow files to be used with the [comfyui-obvpm-timeline](https://github.com/chanon/comfyui-obvpm-timeline) custom node pack.

The version format is 

v`[comfyui-obvpm-timeline nodepack version to use it with]`-`[workflow version no]`

Use the latest workflow where possible. Older ones are kept for reference or in case a newer one has bugs or compatibility issues with a ComfyUI install.

## v0.1.4-007 (Latest HEAD)

- New workflow [`h3_obvpm_timeline_r2v_v0.1.4-007.json`](h3_obvpm_timeline_r2v_v0.1.4-007.json). It runs on this pack's 0.1.4.
- **Needs [comfyui-obvpm](https://github.com/chanon/comfyui-obvpm) 0.2.14 or newer and comfyui-obvpm-timeline 0.1.4 or newer.** The Compatibility Check node says so when the workflow loads.
- The generation and upscale engines are switched on and off as whole groups: a **Mute Group If** switch in each engine group (Run when Generating, Run when Upscaling) mutes the engine that is not used for the run, decided from the timeline's `upscaling` before the run starts. Before, a gate inside each engine stopped it during the run.
- **H3 Model** (an Override Bundle Values node): turn on its `model` override to run a different model than the one in the engine settings preset, without editing the preset.
- Generation and upscale load their VAEs separately: generation defaults to the faster `minimax_h3_video_vae_int8_convrot.safetensors`, while the upscale keeps `minimax_h3_video_vae_fp16.safetensors` for the final quality.
- A **Megapixels Resolution Reference** note beside the timeline lists the width x height each megapixels setting gives at 16:9 and 4:3.
- The live preview while sampling comes from the new **H3 Live Preview** node (with `taeh3.safetensors` in `models/vae_approx`) instead of KJNodes' Model Preview Override: the whole clip, pinned frames included, plays under the Result Preview's bar while it samples.

## v0.1.1-006 (2026-09-28)

- New workflow [`h3_obvpm_timeline_r2v_v0.1.1-006.json`](h3_obvpm_timeline_r2v_v0.1.1-006.json). It runs on the pack's 0.1.1, with the same requirements as 005.
- Fixed: The latent upscaler now references the new renamed filename `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` as reported by #16. Download it from [Hugging Face](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors) into `ComfyUI/models/latent_upscale_models/`.
- Fixed: Moved the `ModelSamplingMiniMaxH3` node from the sampling subgraphs to the `H3 Model Optimization` subgraph so that the `shift_video` is applied and shapes the schedule and the vsa sparse attention window correctly. The included presets all use 12, so their results are unchanged but if you changed the `shift_video` previously, this will change the result to be the correct one.

## v0.1.1-005 (2026-09-27)

- New workflow `h3_obvpm_timeline_r2v_v0.1.1-005.json`. It runs on the pack's 0.1.1.
- **Needs [comfyui-obvpm](https://github.com/chanon/comfyui-obvpm) 0.2.9 or newer.** Update it before loading the workflow.
- Feature: **Compatibility Check node**: when the workflow loads, it checks your ComfyUI version and every custom node pack the workflow uses, and lists anything missing or too old with a link to update it. Its **Copy Report** button copies the details for a bug report.
- Feature: Settings Presets: every setting has a tooltip, and the turbo LoRA and strength are only shown when a turbo loader is selected.
- Fixed: the `lightx2v` generation preset had its turbo loader set to off, so its LoRA was never loaded.
- Layout: A little bit less Get/Set nodes as Bundle and Unbundle can now Get/Set by themselves

## v0.1.1-004 (2026-09-22)

- Set the default generation preset to sample 20 steps instead of 8.

## v0.1.1-003 (2026-09-22)

- Fixes for errors when loading the workflow. (removed from repo)

## v0.1.0-002 (2026-09-22)

- Fixes for errors when loading the workflow. (removed from repo)

## v0.1.0-001 (2026-09-21)

- First workflow, released with the pack's 0.1.0. (removed from repo)
