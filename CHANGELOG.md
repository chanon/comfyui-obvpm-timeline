# Changelog: comfyui-obvpm-timeline node pack

Changes to the nodes themselves. The workflows have their own changelog: [workflows/README.md](workflows/README.md).

Changes not yet in a release are listed under **Latest HEAD**; they become a numbered version when the pack is released.

## Latest HEAD

- Fixed ([#15](https://github.com/chanon/comfyui-obvpm-timeline/issues/15)): clips could show **no mctx** (and extends fell back to the pixel route, with no loop option) although their `.mctx.safetensors` was right beside them, when the browser could not read the sidecar itself. The server now answers for it, and a clip without a usable sidecar says why in its tooltip: no sidecar, a sidecar that no longer matches the video, or one that could not be read.
- Takes extended or prepended from a clip in another folder now record where that clip was, so Result Preview can show them together instead of the take alone.
- A `base_folder` typed with backslashes (`projects\my_project`) now works like `projects/my_project`. The Timeline's clip menu used to find nothing in it.
- Upscaling: a timeline with hard cuts (unrelated clips, gray seams) or empty space can now be upscaled. It is split into segments that are upscaled, refined and rendered separately (`upscale_00001_s01.mp4`, `_s02`, ...) and joined into one finished `upscale_00001.mp4`, black for empty space, the sound de-clicked at the cuts. Before, such a timeline failed with "not on the latent grid", or its second shot started mid-cycle.
- Upscaling: only what the cut shows is upscaled and refined. A take trimmed to a few seconds is no longer processed whole, and the rendered file now ends where the last clip's cut says (it used to run to the end of the last take).
- Upscaling: a cut made by hand on a seamless (green) seam is now honoured as a cut. Before, the upscale silently played the natural join instead.
- The "not on the latent grid" error now names the clip and the nearest frames that would work.

## 0.1.1 (2026-09-23)

- **Update [comfyui-obvpm](https://github.com/chanon/comfyui-obvpm) to the latest version (0.2.5 or newer) as well.** It fixes compatibility bugs the timeline workflow runs into: presets not switching on ComfyUI frontend 1.53 (ComfyUI 0.37), Bundle pin names on a non-English frontend, and the "Converting circular structure to JSON" error when loading the workflow from a saved video with Nodes 2.0 on ComfyUI 0.36.
- Fixed: upscaling a timeline short enough to fit in one sampling window (about 7.8 s at the default window of 39) failed at the refine sampler with `TypeError: list indices must be integers or slices, not NoneType`. Longer timelines were not affected.
- The Timeline node refuses to run on a ComfyUI or comfyui-obvpm too old for the workflow (ComfyUI 0.35.0, comfyui-obvpm 0.2.5), saying what to update and where, instead of failing downstream.
- The package published to the Comfy Registry no longer contains the tests and CI helpers, only the pack itself (`.comfyignore`).

## 0.1.0 (2026-09-21)

- First release.
