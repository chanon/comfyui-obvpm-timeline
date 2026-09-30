# Changelog: comfyui-obvpm-timeline node pack

Changes to the nodes themselves. The workflows have their own changelog: [workflows/README.md](workflows/README.md).

Changes not yet in a release are listed under **Latest HEAD**; they become a numbered version when the pack is released.

## Latest HEAD

- Timeline: clips joined by a green seam can now be separated. Ctrl+drag the right-hand clip to open empty space between them; both keep playing the frames they played while joined. Dragging it back flush rejoins them.
- Timeline: new **split** button in the middle of the cut row. It splits the clip under the playhead into two clips at that frame (on the snap grid while **snap cuts** is on), which can then be moved, cut or deleted separately. Both halves play the same file; nothing is re-encoded. The button is dimmed while the playhead sits on the clip's edge. The two halves join back-to-back, so an upscale treats the split as a cut.
- Timeline: extend, prepend, bridge and loop now act on the one clip you picked when the same file is in the timeline more than once (the parts of a split clip). Before, all parts showed the pin and it was unclear which one the next run would continue from. The pin row names the part (for example `clip_00033.mp4 (part 2/3)`), the pin carries that part's cut, and the new take is placed next to the part it continues from.
- Timeline: removing a clip with its ✕ now always leaves its time as empty space of the same length (merged with any beside it), so nothing after it moves, and the clips either side keep the frames they played -- before, the rest of the cut slid left to close the hole, and a clip that the removed one had continued could grow back the frames it had been cut at. A clip with nothing after it is simply removed, and so is the first clip of a looping cut. The gap's own ✕ still closes a gap.
- Result Preview: keeps nothing in the workflow any more. A finished run, adding or dismissing a take, its seam settings and the live-preview toggle no longer create undo steps or change the workflow file, an undo elsewhere never brings an older result back, and a shared workflow no longer carries your last result. When a workflow opens, each preview shows this session's result, or else the newest take its Save node wrote (judged as a run would, cached per file). Dismissed takes, seam settings and the live-preview toggle are remembered in this browser; settings saved in older workflows are carried over once.
- Result Preview: **dismiss** now clears the result from the preview instead of only hiding the buttons (the take stays on disk), and a dismissed take is not shown again when the workflow is opened -- the preview stays empty until the next run rather than showing a take you no longer remember running. A take added to the timeline still shows, and dismiss is now offered for it too (and for a take already in a timeline), so any result on screen can be cleared.
- Result Preview: the take preview and the upscale preview each build their own preview file (`obvpm_h3_result_clip`, `obvpm_h3_result_upscale`, named after the Save prefix). They used to share `obvpm_h3_result.mp4` in the project folder, rebuilt each other's preview on every look, and two builds at once failed with "The system cannot find the file specified" (`...obvpm_h3_result.mp4.tmp.mp4`). Builds of one preview file now run one at a time, each with its own temp file, and an export copies the preview while holding that turn. The old `obvpm_h3_result.mp4` is no longer used and can be deleted.
- Timeline: the parts of a split clip carry a part id in their sequence line (`[id=k3x9]`; a file that is in the cut only once never gets one), and a pin names its part by that id rather than by position. The pin now stays on its part when other parts are moved, split or deleted; a pin on a clip you split moves to the half that keeps the edge it continues from; and a pin whose own part is deleted is cleared, with the pin row saying so, instead of silently jumping to another part. The new take records its part and the cut it was queued from, so it goes back next to that part even after a reload (or next to the other half, when that part was split meanwhile and the half now owns the edge it continues from), and if that part's cut was changed while the take was generating you are asked whether to put it back (seamless) or keep yours (a cut) -- before, the cut was reset without asking and the message blamed the 17-frame grid.
- Timeline: the seam menu's list of existing takes checks where each take joins, not only which file it continues. Takes of the same file that join at another part's edge, or at a point this part's cut no longer reaches, are listed apart under *joins elsewhere*, dimmed, with their join frame. A green link whose hand cut does not meet the natural join frame now shows as a cut on the strip, matching how the upscale treats it.

## 0.1.2-preview.3 (2026-09-28)

- Load settings: a Value Presets node ([comfyui-obvpm](https://github.com/chanon/comfyui-obvpm), 0.2.10-preview.2 or newer) is listed setting by setting instead of as raw widget text. Each changed value is its own row. A changed schema, and each workflow preset that differs, is one row, unticked by default, with what changed inside it (per field: type, default, `when`, tooltip; per preset: its values) behind a collapsed toggle. Loading the preset label no longer puts that preset's stored values over the take's own.

## 0.1.2-preview.2 (2026-09-28)

- Fixed ([#15](https://github.com/chanon/comfyui-obvpm-timeline/issues/15)): clips could show **no mctx** (and extends fell back to the pixel route, with no loop option) although their `.mctx.safetensors` was right beside them, when the browser could not read the sidecar itself. The server now answers for it, and a clip without a usable sidecar says why in its tooltip: no sidecar, a sidecar that no longer matches the video, or one that could not be read.
- Takes extended or prepended from a clip in another folder now record where that clip was, so Result Preview can show them together instead of the take alone.
- A `base_folder` typed with backslashes (`projects\my_project`) now works like `projects/my_project`. The Timeline's clip menu used to find nothing in it.
- The Timeline's upscale toggle can now be read before a run. With [comfyui-obvpm](https://github.com/chanon/comfyui-obvpm) 0.2.10-preview.1 or newer, the workflow shows which engine settings the next run will use: Override Bundle Values shows their values, and the Value Presets the run will not use are faded.

## 0.1.2-preview.1 (2026-09-26)

- Upscaling: a timeline with hard cuts (unrelated clips, gray seams) or empty space can now be upscaled. It is split into segments that are upscaled, refined and rendered separately (`upscale_00001_s01.mp4`, `_s02`, ...) and joined into one finished `upscale_00001.mp4`, black for empty space, the sound de-clicked at the cuts. Before, such a timeline failed with "not on the latent grid", or its second shot started mid-cycle.
- Upscaling: only what the cut shows is upscaled and refined. A take trimmed to a few seconds is no longer processed whole, and the rendered file now ends where the last clip's cut says (it used to run to the end of the last take).
- Upscaling: a cut made by hand on a seamless (green) seam is now honoured as a cut. Before, the upscale silently played the natural join instead.
- The "not on the latent grid" error now names the clip and the nearest frames that would work.
- Fixed: colours of saved takes and upscales looked shifted in browsers (including the timeline preview), video players and editors -- greens duller, skin and warm tones redder. Videos are now encoded with the BT.709 colour matrix and tagged as such, like ComfyUI's own Save Video. Takes saved before this update keep playing as before, and a timeline mixing old and new takes still previews and exports correctly, only slower: such a preview or export is re-encoded as a whole instead of copied.

## 0.1.1 (2026-09-23)

- **Update [comfyui-obvpm](https://github.com/chanon/comfyui-obvpm) to the latest version (0.2.5 or newer) as well.** It fixes compatibility bugs the timeline workflow runs into: presets not switching on ComfyUI frontend 1.53 (ComfyUI 0.37), Bundle pin names on a non-English frontend, and the "Converting circular structure to JSON" error when loading the workflow from a saved video with Nodes 2.0 on ComfyUI 0.36.
- Fixed: upscaling a timeline short enough to fit in one sampling window (about 7.8 s at the default window of 39) failed at the refine sampler with `TypeError: list indices must be integers or slices, not NoneType`. Longer timelines were not affected.
- The Timeline node refuses to run on a ComfyUI or comfyui-obvpm too old for the workflow (ComfyUI 0.35.0, comfyui-obvpm 0.2.5), saying what to update and where, instead of failing downstream.
- The package published to the Comfy Registry no longer contains the tests and CI helpers, only the pack itself (`.comfyignore`).

## 0.1.0 (2026-09-21)

- First release.
