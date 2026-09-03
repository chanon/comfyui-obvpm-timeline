# H3 node reference (obvpm/h3)

Per-node details for the clip-composition suite. Concepts and workflow
recipes are in [h3.md](h3.md); the wire types (`OBVPM_H3_MCTX`, `OBVPM_H3_PINSPECS`,
`OBVPM_H3_PINS`) are explained there too.

Saving · [H3 MCtx Trim and Save Video](#h3-mctx-trim-and-save-video) ·
[H3 MCtx Save Video](#h3-mctx-save-video) ·
[H3 MCtx Save](#h3-mctx-save)
Loading · [H3 MCtx Load](#h3-mctx-load) ·
[H3 MCtx Load by Path](#h3-mctx-load-by-path) ·
[H3 MCtx Load Video](#h3-mctx-load-video) ·
[H3 MCtx From Frames](#h3-mctx-from-frames)
Pinning · [H3 MCtx Pin Spec](#h3-mctx-pin-spec) ·
[H3 MCtx Apply Pins](#h3-mctx-apply-pins) ·
[H3 MCtx Trim Pinned](#h3-mctx-trim-pinned)
Composing · [H3 MCtx Timeline](#h3-mctx-timeline) ·
[H3 MCtx Result Preview](#h3-mctx-result-preview) ·
[H3 MCtx Assemble](#h3-mctx-assemble) ·
[H3 Assemble Upscale](#h3-assemble-upscale)
Refining · [H3 Record References](#h3-record-references) ·
[H3 MCtx Load Conditioning](#h3-mctx-load-conditioning) ·
[H3 Upscale Loop Start](#h3-upscale-loop-start) ·
[H3 Upscale Loop End](#h3-upscale-loop-end) ·
[H3 Run Mode Gate](#h3-run-mode-gate)

---

## H3 MCtx Trim and Save Video

The everyday save: trims the pinned scaffolding off a decoded take and
saves the clip pair (MP4 + mctx sidecar) in one step.

| Input | Type | Notes |
|---|---|---|
| `images` | IMAGE | the decode output, **untrimmed** — this node removes the scaffolding itself |
| `samples` | LATENT | the sampler's raw output latent (the same one you decode); stored whole in the sidecar |
| `base_folder` | STRING | output-relative folder to save into (default `project1`); same meaning as the Timeline's `base_folder`, so one value can drive both. Empty = the output root |
| `filename_prefix` | STRING | filename prefix within `base_folder`, like core save nodes (default `clip`) |
| `crf` | INT | H.264 quality for the MP4; the sidecar keeps lossless latents regardless |
| `audio` | AUDIO (optional) | untrimmed decoded audio; trimmed in lock step and tail-matched to exactly `frames/fps` |
| `pins` | PINS (optional) | the resolved pins from Apply; unconnected = root clip, saved as-is |
| `refs` | REFS (optional) | the reference pixels this take was conditioned on, from H3 Record References; recorded beside the clip so a refine can rebuild its conditioning without the `.cond`. A mismatch warns and still saves |
| `metadata` | STRING (hidden) | provenance JSON stored as `user_meta`; hidden, since the prompt and workflow are now captured automatically |

Outputs: `path` (the written MP4; the sidecar sits next to it), plus
the trimmed `images` / `audio` for preview.

## H3 MCtx Save Video

The same save transaction without the trim: wire the already-trimmed
(delivered) images/audio yourself, e.g. when you post-process the
delivered frames mid-graph before saving. Same inputs as above —
including `conditioning` and `refs` — except `images`/`audio` must
already be delivered content; refuses when the frame count doesn't
reconcile with the latent and the pins.

The node owns its H.264/AAC encode so the MP4 and the sidecar are
written in one transaction with one trustworthy pairing hash. The MP4
is written first; the sidecar is the commit point — a crash in between
leaves a plain playable video.

## H3 MCtx Save

Sidecar-only writer: pairs an mctx sidecar with a video some **other**
node already saved (e.g. VHS Video Combine — wire its filenames output
to `video_path`). For users who prefer their own video saver for
format/speed reasons; the all-in-one nodes remain the simplest
guaranteed-consistent route.

| Input | Type | Notes |
|---|---|---|
| `video_path` | STRING | the saved video: absolute or output-relative; a VHS filenames output works, the last path is used |
| `samples` | LATENT | the sampler's raw output latent |
| `pins` | PINS (optional) | unconnected = root |
| `refs` | REFS (optional) | the reference pixels this take was conditioned on, from H3 Record References; recorded beside the clip so a refine can rebuild its conditioning without the `.cond`. A mismatch warns and still saves |
| `metadata` | STRING (hidden) | `user_meta` JSON; hidden, see above |

The video must be the **delivered (trimmed)** clip. The node validates
what the container cheaply reveals — resolution and frame count against
the latent + pins — and refuses on mismatch. Output: `sidecar_path`.

## H3 MCtx Load

The lean continuation loader: pick a clip, get its verified `OBVPM_H3_MCTX`
bundle — the video itself is **never decoded**, so this is the fast
path for extend graphs. Refuses when no sidecar pairs with the file.

| Input | Type | Notes |
|---|---|---|
| `clip` | combo | videos under the output folder |
| `create_pins` | combo | `none` / `extend (pin tail)` / `prepend (pin head)` — emit a ready-made pin spec, collapsing Load → Pin Spec → Apply to Load → Apply |
| `pin_window` | combo | context window for `create_pins` (default 39) |

Outputs: `mctx` (latents + header), `pin_specs` (ready-made spec, or
empty when `create_pins` is `none`), `latent` (the clip's stored AV
latent exactly as the sampler produced it — split it with Separate AV
Latent to reach the video stream alone, with no VAE round trip to lose
grade to).

The `create_pins` convenience auto-shifts off-grid edges of
continuation clips to the nearest latent-grade cut and logs where the
seam will land; the new clip's recipe records the shifted join, so
assembly needs no extra bookkeeping. For explicit `at_frame` cuts,
audio windows or multi-pin stacks, use H3 MCtx Pin Spec instead.

## H3 MCtx Load by Path

The same loader addressed by a **path** instead of a picker, for graphs
where another node chooses the clip — the upscale loop walking a
timeline. Identical outputs; the video is never decoded and there is no
preview.

| Input | Type | Notes |
|---|---|---|
| `clip_path` | STRING | output-relative path, e.g. `selfie_walk2/clip_00086.mp4`. Usually wired; type one to re-run a single clip |
| `create_pins` | combo | as H3 MCtx Load |
| `pin_window` | combo | as H3 MCtx Load |

Separate node rather than an override on H3 MCtx Load, for three
reasons. The picker's combo rescans the output tree every time the
node's schema is built, and in a loop that folder is being *written* as
the run proceeds — so the list is stale by construction. The video
preview and drag-drop handler attach by class name, and a graph that
only wants latents should not carry either. And a picker whose value is
overridden must not decide caching: when `clip_path` is wired,
`IS_CHANGED` cannot see it (it runs before the graph does), so this
node reports "unknown" rather than a constant that would let one run
reuse the previous run's clip.

## H3 MCtx Load Video

The full loader: like the core video loader but browsing the **output**
folder (where takes land) with mctx awareness.

Outputs: `video` (core-compatible VIDEO object), `images`, `audio`,
`mctx`, `pin_specs`. When a hash-verified sidecar pairs with the file,
`mctx` carries the clip's latents + lineage; without one — or when the
video was re-encoded/edited/swapped since the take was saved — `mctx`
is `None` and only the pixel route is available. Same `create_pins` /
`pin_window` convenience as H3 MCtx Load.

The loader UI shows a native video preview with a sidecar badge
(`mctx ✓` / `no mctx` / `mctx ?`), and accepts drag & drop of video
files onto the preview (files already in the output folder are
selected in place, keeping their sidecar; others are uploaded).

## H3 MCtx From Frames

Builds an `OBVPM_H3_MCTX` bundle by **VAE-encoding footage** — the route for
clips with no sidecar: imports, old takes, anything whose
`.mctx.safetensors` was deleted. This is the only pixels-to-latents
crossing in the pack, and it is deliberately an explicit node rather
than a silent fallback, because crossing costs two things:

- **Fidelity.** A VAE round trip is lossy, so continuity across the
  join is *pixel-grade*, not exact. Apply logs this whenever a pin's
  origin is `encoded`.
- **Identity.** Frames arrive on a wire and may have been graded or
  cropped on the way, so there is nothing to hash. The bundle's
  `self_id` stays empty and a take pinned on it is saved as a **root**,
  not a continuation.

A clip that *does* have a sidecar must keep using it — load it with
H3 MCtx Load instead. This node is a fallback, never a replacement.

| Input | Type | Notes |
|---|---|---|
| `images` | IMAGE | decoded footage; only the kept window is encoded, so decode as little as you need |
| `vae` | VAE | the H3 video VAE |
| `audio_vae` | VAE | the H3 audio VAE — needed **even for silent footage**, because a pin's audio has to be encoded silence, not an empty latent |
| `latent` | LATENT | the **target** clip's empty AV latent, the same one wired to Apply and the sampler; only its resolution is read, so the pin can never mismatch |
| `fps` | FLOAT | the footage's own frame rate. An IMAGE wire carries no timebase, so this is declared rather than measured; anything other than 24 is resampled by floor-indexed CFR. A wrong value pins at the wrong speed |
| `keep` | combo | which end survives when the footage doesn't fit the 17k+5 clip grid (or exceeds `max_frames`): `tail` for extending, `head` for prepending |
| `max_frames` | INT | encode at most this many frames from the `keep` end; 0 = all. A 22-frame pin needs 22 frames, and the VAE cost is per frame |
| `fit` | combo | `cover` crops, `stretch` distorts. `cover` by default: a pinned run has to line up geometrically with what follows it |
| `audio` | AUDIO (optional) | the footage's soundtrack, aligned with frame 0 of `images`. Absent = the pin carries silence |

Output: `mctx`. Feed it to H3 MCtx Pin Spec exactly like a loaded
bundle — everything downstream is unchanged.

Footage is cut down to a legal H3 clip length (5, 22, 39, … 17k+5)
before encoding; surplus frames are always **dropped**, never invented,
which is why `keep` matters. Every offered pin window is itself a legal
clip length, so setting `max_frames` to your window size encodes the
minimum.

## H3 MCtx Pin Spec

Describes one pin as pure data — which part of a source clip to carry
into the next generation, and where it sits in the target. No tensor
work happens here; Apply slices later. Chain several Pin Spec nodes on
the `pin_specs` wire to pin from several sources (e.g. a bridge).

| Input | Type | Notes |
|---|---|---|
| `mctx` | MCTX | the source clip's bundle from a loader |
| `window` | combo | frames to pin: 5 / 22 / 39 / 56 (1 is reserved for the future keyframe path) |
| `take_from` | combo | `tail` (extend), `head` (prepend), `at_frame` = the window **ending** at `take_from_frame` (a timeline cut) |
| `take_from_frame` | INT | only with `take_from: at_frame` — the delivered frame the window ends at |
| `place` | combo | `before` = context leading in (extend), `after` = context leading out (prepend, typically with `take_from: head`), `at_frame` = **not yet implemented** (interior pins / repaint) |
| `place_at_frame` | INT | reserved for `place: at_frame` |
| `audio_window` | INT | frames of audio to pin, end-aligned with the video window; 0 follows the video window |
| `pin_specs` | PINSPECS (optional) | upstream stack to append to |

Unlike the loaders' convenience modes, this node is strict: an
`at_frame` cut that doesn't land on the 17-frame latent grid is refused
with the nearest valid end frames listed.

## H3 MCtx Apply Pins

The executor: slices the pinned windows out of their sources (the sole
frame/latent math site — window snapping, grid checks, cumulative-total
audio cut) and attaches each pinned run to the conditioning as native
H3 keyframe anchors.

| Input | Type | Notes |
|---|---|---|
| `conditioning` | CONDITIONING | your prompt conditioning |
| `latent` | LATENT | the **target** clip's empty AV latent — authoritative for frame count and resolution; wire the same latent to the sampler |
| `snap_window_down_to_available` | BOOLEAN | when a source can't supply the requested window, snap down the ladder instead of refusing (off by default — snapping silently weakens continuity) |
| `freeze_audio` | BOOLEAN | hold the audio mask at 0 for the **whole** clip, so the sampler re-renders picture only and the sound comes out exactly as it went in. For refine passes; off for generation, where the audio latent is noise |
| `audio_denoise` | FLOAT (optional) | with `freeze_audio`: hold the audio mask at this value instead of 0, so the sound re-samples **alongside** the picture. A refine entering at high sigma needs ~0.5 here for the model to re-derive lip sync — generation learned AV jointly, and a fully frozen track gives a near-noise video nothing to move the mouth for. The resampled audio is then a throwaway: wire the save node's `audio` from a decode of the **source** audio latent. Junction-pinned windows stay at exactly 0 regardless. Default 0 = the exact freeze |
| `pin_specs` | PINSPECS (optional) | unconnected or empty = pass-through: conditioning returned untouched, `pins` empty (plain root generation) — the node can stay in the graph when not pinning |
| `vae` | VAE (optional) | video VAE — needed **only** for pins from clips with no usable mctx sidecar, which have to be encoded from their pixels (the Timeline emits those for imported footage). Pins from a sidecar never touch a VAE |
| `audio_vae` | VAE (optional) | audio VAE, for the same pixel-encoded pins; required alongside `vae` even when the footage is silent |

Outputs: `conditioning` (feed the guider/sampler), `pins` (the resolved
slices — feed Trim / Save so they record what actually happened).

Accepts at most **one pin per side**: one `before`, one `after`, or one
of each (bridging). Refuses when the pinned frames leave no room for
generated content, when a source's resolution differs from the
target's, and when another pack has patched H3's layout machinery (two
continuation mechanisms cannot coexist in one session).

Pins are **appended** to any keyframe anchors already on the
conditioning, so combinations like extend-toward-image (a before-pin
plus a stock last-frame keyframe) are legal.

**Pins from clips with no sidecar.** When a spec names a file instead of
carrying latents (`source_kind: clip_pixels` — what the Timeline emits
for imported footage), Apply decodes just the pinned window out of that
file, VAE-encodes it at the target resolution, and slices it through the
very same path as a latent pin. Two things follow:

- It needs `vae` + `audio_vae`. Without them it refuses and says so.
- **There is no 17-frame cut-grid rule on this route.** The grid exists
  because a latent slice has to begin at cycle phase 0; an encode *defines*
  phase 0 at the window's first frame, so any cut frame is legal. A cut
  that the latent route refuses is simply encoded here — at the cost of
  exactness.

The resolved pin records the real range in the source file's own frames,
plus the file's content **hash and path** — so the take is saved as a
proper continuation (`relation: extends`), with `parent_grade: pixel`
marking that the pinned content came through a VAE. The lineage is exact;
only the content is soft. That is what lets the Result Preview pair the
take with its source, the timeline cut at the join instead of at the
source's full length, and level lock / crossfade act on the seam.

## H3 MCtx Trim Pinned

Removes pinned scaffolding from a decoded clip, picture and sound
together. Trim amounts come from the `pins` wire — the same one that
fed Apply — so they always reflect what was actually pinned.

Also truncates the audio tail to exactly `frames/fps`: H3 rounds its
audio grid up ~8 ms per clip, and the surplus would compound at every
join in a chain. Use this node when you want the delivered frames
mid-graph; H3 MCtx Trim and Save Video does the same trim internally.

## H3 MCtx Drift Mask

A model patch for the refine pass's junction pins. Sits on the refine
sampler's MODEL path and reads the `pins` wire from Apply.

A masked pin holds its window at mask 0, which core labels as clean
conditioning for the whole run. In a refine the free rows next to it
already carry content at the pass's small sigma, and a clean wall beside
noisy content pulls the prediction next to it: measured as a ~2 luma dip
right after every pinned head that recovers over two to three seconds,
in every junction mode. This node takes the wall down. Two mixes:

- **`renoise`** (default) rebuilds the held rows at every step from the
  carried clean latent, noised to the chosen level with the run's own
  fixed noise — the way ordinary inpainting samplers treat a masked
  region. The row's noise is then what its label says by construction,
  whatever the sampler did last step, so it works at any step count and
  nothing can drift. Core's post-step x0 blend still returns the rows to
  clean, so the join and the trim geometry are unchanged.
- **`blend`** is Contex-Loop's Drift-Control rule: core's own mix of the
  sampler's state with the clean latent, held rows at
  `sigma_next / sigma_current`. Validated by them at 20 steps; measured
  here to **fail under a 4–5 step turbo schedule**, where the ratios sit
  near 1 for every step that matters and then snap to clean for a last
  step covering 43% of the range. Kept for schedules with a small final
  step.

The **level** says how noisy the held rows are each step: `matched` is
the content's own sigma (the context is indistinguishable from content
at every step — no wall anywhere, the big last step included); `ahead`
is `sigma_next / sigma_current`; `constant` is `level_value × sigma`.
The model is handed the levels as its per-row labels, so what it is told
matches what it is given. Audio is untouched.

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | the MiniMax H3 model the refine samples with |
| `pins` | PINS | from Apply — which rows are held, and on which side |
| `enabled` | BOOLEAN | off = pass through. Wire from Loop Start's `drift` so the model patch and the profile hash agree |
| `mix` | COMBO | `renoise` or `blend`, above |
| `level` | COMBO | `matched`, `ahead` or `constant`, above |
| `level_value` | FLOAT | the fraction for `constant`; ignored otherwise |
| `taper_steps` | INT | latent steps nearest the join that fall from the level to exact (`4` = `.75/.50/.25/0`, Contex-Loop's recipe). `0` keeps the whole window at the level, join row included — the output is returned to clean after every step regardless, so the trim is unaffected |

These settings are not part of the profile hash (the loop cannot see
them); put a word in Loop Start's `upscale_note` when you change them so
the run opens its own folder.

Output: `model` (a clone with the hooks installed). With no held pins —
a root clip, a guide-only pin — the input model passes through. Refuses
a model that already carries a dynamic denoise-mask patch (Differential
Diffusion), since the two would fight over the same hook.

## H3 Chain Noise

A NOISE source for the refine sampler that draws noise per **absolute
timeline position** rather than per clip.

A refine invents its fine texture largely from its noise. With a fixed
seed, every clip of the same size gets the *identical* noise tensor,
positionally — so the child's first free frame after a junction carries
the noise its parent had at frame 39, not a continuation of the noise the
parent's last frame was refined under, and the two sides of the join
invent unrelated detail. This node indexes one noise field by timeline
frame: a clip whose raw latent starts at frame F gets, for its latent
step k, the noise of absolute step `steps(F) + k`, and each audio tick
likewise. A pinned window then carries the same noise in the parent and
the child, and the free frames on both sides of a join sit in one
continuous field. Each step's draw is independent, so a clip's noise
depends only on where it sits, not on which clip is being refined.

| Input | Type | Notes |
|---|---|---|
| `noise_seed` | INT | the field's seed; same seed = same field for every clip of the chain |
| `frame_offset` | INT | where this clip's raw latent starts on the timeline, in frames — wire from Loop Start's `noise_offset` (negative for a first clip whose pinned head precedes the timeline) |

Output: `noise`, for the sampler's noise input. Off-grid offsets (a plain
cut upstream) snap to the covering latent step.

## H3 MCtx Timeline

The composition hub — see the [widget guide](h3.md#the-timeline-widget)
for the interactive parts (strip, seam menus, pinning, preview,
export). The node itself:

| Input | Type | Notes |
|---|---|---|
| `sequence` | STRING | one output-relative clip per line, in playback order; ` @ N` forces an entry frame; `# ` comments. Hidden behind the widget, hand-editable via ✎ |
| `base_folder` | STRING | output-relative folder this node works in: scopes the clip picker, seam suggestions and new-clip detection, and holds the preview file and exports |
| `preview_filename` | STRING | name of the single overwritten full-preview file; give each Timeline node its own if you use several |
| `export_filename_prefix` | STRING | prefix for the export button, with the usual counter |
| `crf` | INT | H.264 quality for re-encoded seam bridges and the export |
| `auto_add` | BOOLEAN (hidden) | deprecated and inert; new takes are offered by the Result Preview instead. Still declared so saved `widgets_values` do not shift |
| `pin_state` | STRING (hidden) | the pin selection as one JSON blob, owned by the widget |
| `restore_groups` | STRING | keyword(s) naming the groups **load settings** writes into (default `[restore]`). Empty disables the button |
| `skip_restore_nodes` | STRING | nodes whose **title** contains this are left alone, even inside a matching group (default `[skip]`) |
| `snap_cuts_to_grid` | BOOLEAN (toolbar) | whether dragging a cut handle snaps to the 17-frame latent grid (default on). Lives on the strip toolbar, not among the setup widgets — it is a mode you flip while working |
| `level_lock` | BOOLEAN | match a continuing clip's opening to where its parent was heading (default **on**) |
| `level_lock_frames` | INT | how many frames the correction ramps out over (default **12**) |
| `level_lock_flicker` | BOOLEAN | also straighten an oscillating opening, not just a constant offset (default **on**) |
| `level_lock_local` | BOOLEAN | match per region rather than per frame, so no area is left several luma out (default **on**) |
| `crossfade` | BOOLEAN | hand the join over across the overlap both takes rendered (default **on**) |
| `crossfade_frames` | INT | fade length; **0 means the whole overlap**, which is the right answer — see the guide (default **0**) |
| `audio_declick` | BOOLEAN | 5 ms taper each side of joins no crossfade covers (default **off**) |
| `duration_seconds` | FLOAT | how long the next generation should be; the `length` output converts and snaps it |
| `run_mode` | COMBO (toolbar) | `generation` or `upscale` — which half of the workflow a Run is for. Driven by the **upscale** toggle on the strip toolbar past **export**, not from the setup widgets. See [H3 Run Mode Gate](#h3-run-mode-gate) |

Outputs: `pin_specs`, `length`, `sequence`, `run_mode`.

The seam settings above are **defaults, not decisions** — any single join can
override them from its own seam dialog, and those overrides ride in the
sequence line rather than in a side table. What each repair does, and
when it is available at all, is in
[Seam repair](h3.md#seam-repair).

### Loading a take's settings

Every take saved since workflow embedding carries the graph that made it
in its sidecar. Select a clip in the strip and press **load settings** on
the selected-clip row to replay part of that graph into the workflow you
are editing.

It asks first, and shows its working: the dialog lists every value it
will replace in a table — node, setting, current value, new value — plus
any bypass/mute change and the pin. Each row has an **Apply** checkbox
(ticked by default, with a toggle-all in the header): untick a row and
that value is left exactly as it is, so you can take a take's seed
without its prompt, or its prompt without its LoRA stack. The count in
the header follows what is ticked. Cancelling changes nothing — what the dialog shows
is a dry run of the same code that does the work, so it cannot promise
one thing and do another. When there is nothing to load (no keyword, no
matching group, or every value already matches) it says so instead of
opening a dialog.

The workflow is read from the clip's sidecar, and failing that from the
**MP4's own container tags** — so clips with no sidecar still work,
including videos saved by core's Save Video (separate `workflow` /
`prompt` tags) or by VideoHelperSuite (one JSON blob in `comment`).

`restore_groups` decides how much: settings are applied only to nodes
inside **groups whose title contains the keyword** (case-insensitive;
comma-separate for several). It defaults to `[restore]`, so the way to
opt a group in is to put `[restore]` somewhere in its title. Empty means
nothing is restored, so the button cannot fire by accident.

`skip_restore_nodes` is the opt-out, matched against a **node's title**:
a node called `Seed [skip]` keeps its current values even though it sits
in a `[restore]` group. Defaults to `[skip]`, same matching rules.

Both the groups and their membership are read from the **workflow you
are editing now**, not from the saved one — the keyword matches the
titles you can see, and "the nodes in this group" means the ones in it
at this moment. The take's workflow is only the value *source*, matched
per node id. So a group you have since renamed, moved or resized behaves
the way it currently looks; a node dragged into the group is included
even though the take predates it (reported as having no stored values),
and one dragged out is left alone.

What it does, and nothing more:

- **widget values** and **bypass/mute state** of matching nodes;
- **this Timeline's own pin options** — extend/prepend/bridge and window,
  as they were when that take was generated, so the take can be
  reproduced. Applied whatever the keyword matches, since it is this
  node's own state.

What it deliberately never does:

- create, delete, move or **rewire** anything — values only;
- descend into subgraphs: **top-level nodes only**;
- touch **result previews** (they carry a run's own payload) or the
  Timeline's `sequence` (your current cut is not a property of a clip).

It refuses rather than guesses: a node in the group that the take has no
values for, or that changed type, had its widgets change shape, or whose
combo no longer offers the stored value, is reported and skipped. Wire-driven widgets are
left alone, because their value comes from the wire at run time. The
toast summarises counts; the console carries the per-node detail.

Output: `pin_specs`. **Not an output node** — running a workflow does
nothing except emit the pin spec for whatever is pinned in the widget
(empty when nothing is; Apply then passes through). The pinned clip is
re-verified against its sidecar hash at run time, exactly like H3 MCtx
Load.

## H3 MCtx Result Preview

Mini timeline of a just-saved take in its lineage context. Wire a Save
node's `path` output to the `path` input; after each run the widget
shows the take beside its parent (extend), its target (prepend), or
between both (bridge), playable as one seamless preview.

Each seam carries a **measured verdict** — the motion discontinuity at
the join computed from the take's own latents, rated `seamless` /
`soft bump` / `hard cut` — so you can decide keep-or-reroll before the
timeline is involved. Buttons: **+ add to timeline** (shown when a
Timeline node matching the clip's folder exists; inserts at the
lineage-derived position), **✕ delete** (removes the take's MP4 and
sidecar from disk, after confirmation), **dismiss** (hides the
buttons).

## H3 MCtx Assemble

The in-graph assembler: same sequence format as the Timeline, but an
output node that builds the cut when the workflow runs.

| Input | Type | Notes |
|---|---|---|
| `sequence` | STRING | one clip per line, ` @ N` entry override, `# ` comments |
| `filename_prefix` | STRING | output path prefix (default `project1/cut`) |
| `crf` | INT | quality for re-encoded seam bridges |

Outputs: `path`, `images`, `audio` (audio present when every clip
carries it).

Every seam cut is derived from the mctx sidecars: extend seams butt
seamlessly, trim-point and prepend seams enter/exit at the recorded
join. Writes by **smart-cut**: video packets are stream-copied bit for
bit wherever possible and only the sub-GOP frames around a mid-clip
seam are re-encoded; audio is decoded and encoded once as a continuous
track (packet-spliced AAC always clicks). No sidecar is written — an
assembly is delivery, not a take; the source clips remain the masters.

## H3 Assemble Upscale

Plays a finished upscale profile as one MP4 — H3 MCtx Assemble with the
sequence filled in from the profile itself.

| Input | Type | Notes |
|---|---|---|
| `base_folder` | STRING | the project folder, same value the loop and save nodes use |
| `profile` | STRING | which profile under `<base_folder>/_upscale/` to play; leave empty when `profile_folder` is wired |
| `filename_prefix` | STRING | empty writes `<profile folder>/cut`, so the refined cut sits with the clips it was made from |
| `crf` | INT | quality for re-encoded seam bridges; match the takes |
| `profile_folder` | STRING (optional) | wire Loop End's `profile_folder` here: the folder the pass actually wrote, so nothing is typed twice. Overrides `base_folder` + `profile`, and — like `after` — only becomes a value when the loop finishes |
| `after` | STRING (optional) | wire Loop End's `report` here and this node runs **when the loop finishes** — see below |

Outputs: `path`, `images`, `audio`, and `sequence` — the list it built,
so you can see what it played or paste it into H3 MCtx Assemble to
adjust by hand.

Why a node rather than retyping the list: refined clips are numbered in
the order the loop **worked**, which is not delivery order — a join
carried by a clip's tail is refined after the clip that plays behind it.
The profile records each clip's delivery position, so the mapping is
already written down; this reads it. It also puts the original ` @ N`
cut markers back — shifted where a soft junction gave a refined clip a
longer delivered head than its source, and dropped on the side of a
junction the refine itself made, where the seam is derived from the
refined sidecars and a copied marker would put the cut back where the
*source* joined. Markers on plain cuts (no junction) are kept as typed.

A half-finished profile assembles its leading run and says so, rather
than splicing across the gap — a cut with a clip silently missing looks
exactly like a finished one.

**Running it automatically.** Wire Loop End's `report` into `after` and
the assembly happens by itself when the pass ends — once, not once per
clip. That works because of how the loop iterates: a non-final Loop End
hands back a *link* to the next iteration's outputs rather than a value,
so `report` only becomes a real string at the last one. Waiting for it
is therefore waiting for the whole pass, and nothing downstream can run
early by accident. Leave `after` unwired to assemble whatever the folder
holds right now, which is how you watch a pass that is still running.

## H3 Record References

Records the reference images and audio a take is conditioned on,
**in line** on their way to MiniMax H3 Reference to Video. Add it
between the loaders and the reference node; it changes nothing about
what the sampler sees.

| Input | Type | Notes |
|---|---|---|
| `image_0` … `image_8` | IMAGE (optional) | mirrors the reference node's nine `ref_images` |
| `audio_0` … `audio_2` | AUDIO (optional) | mirrors its three `ref_audios` |

Outputs: the same images and audio unchanged, plus `refs` — wire that
to a save node beside `conditioning`.

In line rather than branched, deliberately: a wire that must pass
through cannot be forgotten, whereas branching fails as a take that
looks correct and is missing a reference. Socket *order* is what
matters, not which numbers are used — the reference node numbers
references within each type in the order they arrive, so a graph wiring
`image_0` and `image_3` gets `<Picture 1>` and `<Picture 2>`.

Storage is content-addressed beside the clip: a reference already on
the 8-bit grid is written as lossless PNG (core's own resize round
trips through PIL in `uint8`, so this is the common case), anything
else as exact `safetensors`. Two takes sharing a reference share one
file.

**Video references are not recorded.** Core presents them to the text
encoder at 2 fps but VAE-encodes every frame for the payload, so
reproducing one means keeping all of it. A take with video references
keeps its `.cond` instead, and the save node says so rather than
leaving it to be discovered at refine time.

## H3 MCtx Load Conditioning

Rebuilds a saved take's conditioning, so a refine pass sees exactly
what generated the clip.

| Input | Type | Notes |
|---|---|---|
| `mctx` | MCTX | the take's bundle, from either loader — it carries the clip's own path |
| `prefer` | combo | `cache` (use the `.cond` when there is one) / `rebuild` (re-encode from recorded references) |
| `canvas` | combo | `source` replays the take's own geometry — right for a same-size refine. `target` rebuilds from the recorded references at the resolution of `target_latent`, so an **upscale** pass's `match` picture refs scale to the pass-2 area instead of describing the small canvas. Forces the rebuild path (a `.cond` is source geometry by definition) and needs `clip` + `vae` |
| `target_latent` | LATENT (optional) | `canvas='target'` only: the pass-2 latent whose resolution the conditioning is rebuilt for — wire the upscaled video latent |
| `clip` | CLIP (optional, **lazy**) | text encoder — needed only to rebuild |
| `vae` / `audio_vae` | VAE (optional, **lazy**) | needed only to rebuild |

The three model inputs are **lazy**: which source will win is a question
about two filenames beside the clip, so the node answers it before
asking for anything. Wire them and a take that has a `.cond` still costs
nothing — the 32B encoder is not even loaded. Leave them wired.

Outputs: `conditioning`, and `source` (`cond`, `refs`, or
`refs@target`) so the graph can say which route it took.

Three ways down, in order: a `.cond` beside the clip loads exactly and
needs no model at all; recorded references re-encode through the core
reference node (not a reimplementation of it), reproducing the
conditioning to within float noise; neither, and it refuses by name
rather than silently conditioning on something else. A `.cond` is ~65 MB
per take and 98% of it is one text embedding, which is why recording
the reference pixels is the cheaper habit.

## H3 Upscale Loop Start

Opens the refine loop: resolves which clip this iteration refines, and
where the result goes.

| Input | Type | Notes |
|---|---|---|
| `sequence` | STRING | the timeline, one output-relative clip per line in delivery order — the same text the Timeline holds |
| `base_folder` | STRING | the project folder the clips live in |
| `profile` | STRING | **empty = automatic**: the pass resumes the profile whose settings hash matches, otherwise creates the next free `refineNN` — so changing a setting lands in a new folder instead of being refused. A typed name pins one folder. Refined clips land in `<base_folder>/_upscale/<profile>/` |
| `first_sigma` | FLOAT | how much of the pass re-samples. Wired to a BasicScheduler's `denoise` it is a **fraction of the schedule** (0.24 ≈ entry σ 0.79 at shift 12 — a refine); handed to a manual sigma list it is the entry sigma itself (0.9 keeps ~10% of the source — a restyle that flickers and drops lip sync). Either way it is part of the profile hash, so it lives here |
| `upscale_note` | STRING | **not read, only hashed**: your declaration of the settings the loop cannot see (upscaler, scale, sampler schedule). Wire it from whatever states them and it stops being an honour system; with an automatic profile it is what makes a changed upscaler setting open a new folder |
| `start_index` | INT | `-1` resumes where the profile left off; the loop drives this itself after the first iteration |
| `junction_ramp` | INT | **soft junctions.** `0` mirrors each take's recorded hold (a hard hold for most). `N` ramps the held window's mask over its last `N` frames, from exact up to `junction_edge` at the join, so the two clips' re-derived detail blends across the ramp instead of switching on one frame — a refine invents fine texture (distant people, foliage), each clip invents its own, and a hard handover shows it even where the motion is exact. The ramped frames are re-drawn and delivered by the later clip, so the earlier one exits that much sooner; total length is unchanged. Part of the profile hash |
| `junction_edge` | FLOAT | with a ramp: the mask value at the join, as a fraction of the pass's own sigma (`0` still exact, `1` fully this clip's refine). `0.4` is the generation side's arriving default |
| `junction_mode` | COMBO | `mirror` pins the way the take was made (held exactly). `both` holds exactly **and** feeds the window as keyframe rows, so the parent's refined texture is a clean *reference* the model can copy appearance from, not only content it may not change. `guided` does not hold at all: the window is the clip's starting point and steering rows, the clip re-draws it and **delivers** it, and the neighbour hands over at the window's start — the neighbour's texture becomes this clip's across 39 frames of one generation rather than switching on one frame, which is the defect a per-clip refine shows at a hard join (each clip invents its own fine detail). The seam becomes a guided, pixel-grade one. Hashed when not `mirror` |

| `drift` | BOOLEAN | drift control for the held window: instead of a clean wall, the held rows are rebuilt every step at the content's own noise level by [H3 MCtx Drift Mask](#h3-mctx-drift-mask) on the refine model path, wired from this node's `drift` output. Removes the level dip a refine shows right after a clean held window. Works with `mirror` and `both`. Hashed when on |

Outputs: `flow` (to Loop End), `clip_path` (to the path loader),
`pin_specs` (the junction pin, mirroring the original pin's geometry
against the previously *refined* clip), `out_folder`, `first_sigma`,
`index`, `total`, `drift` (the toggle, for H3 MCtx Drift Mask's
`enabled`), and `noise_offset` (this clip's
raw-latent start on the timeline, for [H3 Chain Noise](#h3-chain-noise)).

The profile folder is the whole state: the refined clips plus a
`profile.json` recording what settings produced them, which **source**
each came from, and which refined neighbours each was pinned to. Sources
are never touched, so a pass can be re-run or abandoned, and several
profiles of one project coexist.

A timeline may start on an extension clip. That clip has no neighbour
to pin to, so it is refined as a root and its rendering keeps the
source's pinned head untrimmed; the loop records the difference and the
assembled cut enters the rendering where the source's delivery began, so
those scaffolding frames are never played.

**Editing the timeline does not restart the pass.** A profile is its
settings, not its sequence; a refined clip is reused wherever the current
timeline puts it, as long as the junctions it needs are the ones it was
refined with. Add a clip and only that clip is refined (plus a neighbour,
if the new clip creates a held join that neighbour was not refined
against); reorder or re-cut and nothing is redone. Re-refining a clip
invalidates exactly the clips that were pinned to its old rendering. The
settings hash is what stops a half-finished profile being completed under
different settings — a timeline whose first six clips were refined one
way and last six another is a subtle, expensive kind of broken; with an
automatic profile, changed settings simply open a new folder.

## H3 Upscale Loop End

Closes the loop: commits this iteration, then expands the next.

| Input | Type | Notes |
|---|---|---|
| `flow` | LOOP | from Loop Start |
| `after` | STRING | the save node's `path` — what makes the commit happen *after* the clip is on disk |

Outputs: `report` and `profile_folder`. This is an output node, so it is
an execution root. Both outputs are the pass's **done** signal — they only
become values at the last iteration — so wire `profile_folder` into H3
Assemble Upscale's `profile_folder` and the refined cut is built the
moment the last clip lands, from the folder the pass actually used. An
automatically chosen profile is chosen once, at the first iteration, and
every later iteration is handed the name.

ComfyUI graphs are acyclic, so iteration is **node expansion**: on each
pass the body is cloned with the next clip's index and handed back to
the executor. The body is what depends on Loop Start *and* reaches Loop
End — model loaders, VAEs and anything else feeding it from outside are
shared, loaded once, and not cloned. The manifest is written after the
clip it describes, so a crash between the two leaves a refined clip the
manifest does not know about, which the next run simply refines again.

## H3 Run Mode Gate

A one-wire on/off switch for a whole branch of the graph. The Timeline's
`run_mode` widget picks `generation` or `upscale`; each gate passes its
value through in the mode it is set to and shuts in the other.

| Input | Type | Notes |
|---|---|---|
| `run_mode` | STRING (wired) | from the Timeline's `run_mode` output. One source for every gate, so two branches cannot disagree |
| `pass_when` | COMBO | the mode this gate is open in |
| `value` | any (lazy, wired) | anything at all — the gate does not look at it |

Output: `value`, unchanged when open; a block when shut.

**Why a gate and not muting the output nodes.** Muting works, but it is
state kept in as many places as the branch has roots, and getting it
half-right runs both branches. One switch cannot be half-flipped.

**Why it needs two mechanisms.** They cut in opposite directions, and
neither alone is enough:

- The `value` input is **lazy**, which prunes everything *upstream*.
  ComfyUI asks the node what it needs before evaluating its inputs, and
  a shut gate asks for nothing. This is what `MuteGate` cannot do — its
  own docs say "the nodes upstream of `input` still run".
- A shut gate returns an **`ExecutionBlocker`**, which prunes everything
  *downstream*, output nodes included. Every `OUTPUT_NODE` is an
  execution root: a save node is not reached *through* anything, so
  laziness can never prune one. Handing it an input it must refuse is
  the only way.

**Where to put one.** As far upstream in the branch as a single wire can
reach, because the block travels forwards from there. In
`h3_obvpm_r2v.json` that is two wires off the Timeline:

| Gate | On the wire | Switches off |
|---|---|---|
| `generation` | `length` → MiniMax H3 Reference to Video | the reference encode, the sampler, the decode, the take's save node and its Result Preview |
| `upscale` | `sequence` → H3 Upscale Loop Start | the whole refine loop, its resume check included, the refined save, and H3 Assemble Upscale behind it |

Nodes that only *feed* a gate — image loaders, a seed — still run. They
are cheap, and buying their silence would cost a gate each.

An unrecognised run mode is refused loudly at both the Timeline and the
gate rather than quietly shutting everything: a graph that runs and does
nothing is the most expensive way to find a typo.
