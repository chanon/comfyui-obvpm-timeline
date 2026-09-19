# H3 node reference (obvpm/h3)

Per-node details for the clip-composition suite. Concepts and workflow
recipes are in [h3.md](h3.md); the wire types (`OBVPM_H3_MCTX`, `OBVPM_H3_PINSPECS`,
`OBVPM_H3_PINS`) are explained there too.

Saving · [H3 MCtx Trim and Save Video](#h3-mctx-trim-and-save-video) ·
[H3 MCtx Save Video](#h3-mctx-save-video) ·
[H3 MCtx Save](#h3-mctx-save)
Loading · [H3 MCtx Load](#h3-mctx-load) ·
[H3 MCtx Load Video](#h3-mctx-load-video) ·
[H3 MCtx From Frames](#h3-mctx-from-frames)
Pinning · [H3 MCtx Pin Spec](#h3-mctx-pin-spec) ·
[H3 MCtx Apply Pins](#h3-mctx-apply-pins) ·
[H3 MCtx Trim Pinned](#h3-mctx-trim-pinned)
Composing · [H3 MCtx Timeline](#h3-mctx-timeline) ·
[H3 MCtx Result Preview](#h3-mctx-result-preview) ·
[H3 MCtx Assemble](#h3-mctx-assemble)
Refining · [The joint refine](#the-joint-refine) ·
[H3 Join Latents](#h3-join-latents) ·
[H3 Joint Conditioning](#h3-joint-conditioning) ·
[H3 Joint Audio Mask](#h3-joint-audio-mask) ·
[H3 Context Windowing](#h3-context-windowing) ·
[H3 Joint VAE Decode and Save](#h3-joint-vae-decode-and-save)

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
| `metadata` | STRING (hidden) | provenance JSON stored as `user_meta`; hidden, since the prompt and workflow are now captured automatically |

Outputs: `path` (the written MP4; the sidecar sits next to it), plus
the trimmed `images` / `audio` for preview.

## H3 MCtx Save Video

The same save transaction without the trim: wire the already-trimmed
(delivered) images/audio yourself, e.g. when you post-process the
delivered frames mid-graph before saving. Same inputs as above —
including `conditioning` — except `images`/`audio` must
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
| `upscaling` | BOOLEAN (toolbar) | which half of the workflow a Run is for: off generates the next take, on refines the whole timeline and renders the full sequence. Driven by the **upscale** toggle on the strip toolbar past **export**, not from the setup widgets |

Outputs: `pin_specs`, `length`, `sequence`, `upscaling`.

**Switching branches.** A workflow that can both generate a take and
refine the timeline holds two branches that must never run together.
Put a **Mute If** gate (comfyui-obvpm) on the last wire before each
branch's own work begins and drive both from `upscaling`: the
generation branch's gate on `length` → MiniMax H3 Reference to Video,
muted when `upscaling` is true; the refine branch's gate on `sequence`
→ H3 Join Latents, muted when it is false (core's **Not** node inverts
the wire). Everything downstream of a muted gate is skipped, save nodes
and previews included, so one switch on the strip decides and the
branches cannot disagree.

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
timeline is involved. A take that **arrives** somewhere (a prepend, a
bridge, a loop take) gets a second line, `arrival 0.98x`: the
worst-changing parts of the picture at the frame where the take's own
frames give way to its pinned window, against the same moment one
latent step (4 frames) earlier and later. About 1x is clean; from
1.3x it reads `local bump` and from 1.8x `local jump` — something
moves or sharpens in a single frame, usually because the take arrives
slower (and so sharper) than the clip it lands in. Such a take can
look fine at generation size and pop after an upscale, which keeps
motion as it is; another seed is the cure. The thresholds rest on a
small sample. Buttons: **+ add to timeline** (shown when a
Timeline node matching the clip's folder exists; inserts at the
lineage-derived position), **✕ delete** (removes the take's MP4 and
sidecar from disk, after confirmation), **dismiss** (hides the
buttons). A decision sticks to its take: once added or dismissed, that
take never asks again, through reloads and workflow switches. A cut
from H3 Joint VAE Decode and Save is the finished product, so the
preview offers delete and dismiss for it but not add to timeline.

**Live preview while sampling.** With KJNodes' Model Preview Override
on the sampler's model, the frames it pushes each step are shown in
this panel too, in the box the result takes over when the take lands,
with the step count on the status line. That node can then be
collapsed: it stays the place where the preview's frame count and
resolution are set, but nothing needs to be looked at there. Only the
run this preview owns is shown, and without the KJ node the panel
behaves exactly as before.

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

## The joint refine

A per-clip refine re-invents fine texture from its own prior and noise,
and every per-clip fix — exact holds, one noise field, keyframe
references, continuous upscaled latents, a 27-step hold — left the
refined seam where it was (refine08–17, 2026-09-04; a refine moves about
74% of the fine texture away from its upscaled prior, per clip and per
seed). A single long clip refines seamlessly because every row is
sampled in one pass with every other row in view, so the pass below
gives a chain the same treatment: the timeline becomes one latent, is
upscaled (in short chunks along time, see below) and sampled as one, in clip-sized windows that overlap
by about a quarter, with the model's predictions blended across the
overlap at every step (MultiDiffusion along time) and each window
conditioned by the clip that owns most of it. Texture is then decided
across the joins. Per step the model runs once per window, so the pass
costs one pass over the timeline plus the overlaps; memory is one
window's worth.

The sampled timeline **is** the cut — every row is the clip the cut
shows there — so H3 Joint VAE Decode and Save decodes it straight into one finished
MP4, a few seconds at a time, and saves it as a take of its own.

```
H3 Join Latents ─ latent ─> LTXVSeparateAVLatent ─> upscaler (temporal chunking ON) ─> LTXVConcatAVLatent ─> H3 Joint Audio Mask ─> sampler.latent_image
                ─ layout ─> H3 Joint Conditioning ─────────────────────────────────────────────────────> sampler.conditioning
                         └> H3 Joint VAE Decode and Save.layout                                        └> H3 Joint VAE Decode and Save.conditioning
refine model ─> H3 Context Windowing ─> sampler.model        any NOISE ─> sampler.noise
sampler ─> H3 Joint VAE Decode and Save.samples ─ path ─> H3 MCtx Result Preview
```

The refine's schedule is `denoise` on the sampler's BasicScheduler.
Mind the model's timestep shift (12 for H3): `denoise` is a fraction of
the schedule, and the noise level the model actually starts from is
12d/(1+11d) — 0.2 starts at 75 % noise, 0.075 at 49 %. 0.2 is the
working value.

**The upscaler must chunk along time.** The H3 3D latent upscaler
normalises over channels, time, height and width together, and it was
trained on short clips. Handed the whole joint in one pass it
normalises every frame against the statistics of the entire video and
its output drifts everywhere (about 6 % on interior rows against the
same clip upscaled alone); the refine then reads that drift as detail
and hallucinates on it — hard white points on dappled hair, marks on a
face, shimmering eyes — at every noise level, under every sampling
scheme, in positions that follow the noise seed. Leave the upscaler
node's `enable_temporal_chunking` on (32-step chunks, 5-step overlap
blended inside the network); with it on, the joint refines cleanly at
0.2 with no seam (measured 2026-09-08, seven clips; the full record is
in [joint-refine-investigation.md](joint-refine-investigation.md)).

## H3 Join Latents

Lays every clip of the timeline onto one raw AV latent at its true
position. A pinned clip sits where its recipe says: its held frames
are the parent's frames at the pin's `source_start`, whatever the cut
shows of either clip — so a parent cut before it was extended (shown
to frame 158, extended from there) still gets its child on the rows
the child was made from. Only a clip with no pin into the timeline is
placed by the cut's arithmetic. Where clips overlap, each row goes to
the clip the cut shows there: the parent up to the junction recorded
in the child's sidecar, the child from it on. A held head therefore
stays the parent's rows and a cut-away tail never enters the joint.
Every clip's copy of rows it does not own is compared and logged; a
masked hard hold must match to the bit, and a difference there is
warned about as a placement error. A timeline with a gap between
clips is refused.

Measured 2026-09-07 (joint_all2): placing clips by the cut alone put
clip 7 34 frames after its pin (held head rel diff 1.25) and let clip
10's cut-away tail win 102 frames of clip 14's span — a flicker at the
first seam and four seconds of the wrong clip under the right audio.

| Input | Type | Notes |
|---|---|---|
| `sequence` | STRING | the timeline text — the Timeline's `sequence` output through the upscale gate |

Outputs: `latent` (the joint raw AV latent, for the upscaler), `layout`
(where each clip sits, for Joint Conditioning and Joint VAE Decode and Save). The
layout is logged when the node runs.

## H3 Joint Conditioning

Every clip's own conditioning, on the sampler's one CONDITIONING wire.
Each clip's saved `.cond` is loaded; the first clip's conditioning
carries the whole table together with the cut's ownership of every
step, and H3 Context Windowing lays each clip its own run of windows and
samples them under that clip's conditioning — so a timeline of several
prompts and reference sets stays several. Without Context Windowing on
the model the sampler simply runs under the first clip's.

| Input | Type | Notes |
|---|---|---|
| `layout` | JOINT | from Join Latents |

No model is loaded: the `.cond` holds the encoded tensors themselves,
including the reference blocks and any refmods applied on top. A clip
without a `.cond` is refused by name — it was saved with
`save_conditioning` off or with no conditioning wired, and there is
nothing to rebuild it from.

A clip whose `.cond` carries a joint table of its own — a cut rendered
by H3 Joint VAE Decode and Save, put back on a timeline to be refined again — is
spliced in rather than treated as one clip: its inner rows are offset
to where it sits on this joint, and the steps it owns go to whichever
inner clip owned them there. A second pass therefore still samples each
stretch of the rendered cut under the prompt and references it was made
with.

## H3 Joint Audio Mask

Holds the joint latent's soundtrack while the picture is refined. The
sampler denoises video and audio together, so without this a refine
re-renders sound that is already finished.

| Input | Type | Notes |
|---|---|---|
| `latent` | LATENT | the upscaled joint AV latent |
| `audio_denoise` | FLOAT | `0` keeps the audio exactly. `~0.5` re-samples it alongside the picture (the model re-derives lip sync from it); render the **source** audio then — wire H3 Join Latents' `latent` to H3 Joint VAE Decode and Save's `source_audio` |

Wire the sampler's `latent_image` from here.

## H3 Context Windowing

A model patch that samples a long AV latent in windows of
`window_seconds` overlapping by `overlap_seconds`, blending predictions
across the overlap every step. Core's context windows assume
every stream keeps time on the same dim; H3's audio latent keeps it
last, so this is a small handler written for H3's two streams: video
windows in steps, audio windows in ticks on the shared AV grid (exact,
never proportional), masks sliced on the dim each keeps time on, and
each stream blended along its own axis.

With H3 Joint Conditioning on the sampler, each window is conditioned
by the clip that owns most of its rows: the clip's stored conditioning
goes through the model's own conditioning step at the window's shape,
so its text embedding, reference blocks and packed layout are what that
clip generated under (content keyframes are moved to where the clip
sits in the window). Built once per window per run, not per step.

| Input | Type | Notes |
|---|---|---|
| `window_seconds` | FLOAT | window length in seconds of picture, rounded to a clip-shaped number of latent steps (5j+2, about 3.4 frames a step; the log reports what it became). 5 by default |
| `overlap_seconds` | FLOAT | seconds shared by neighbouring windows; the blend happens here. About a quarter of the window; 1.25 by default |

Where windows overlap the two predictions are blended with triangular
weights (each window counts most in its middle and fades out toward
its edges), so the hand-over is a ramp rather than a step.

A latent no longer than the window samples plainly. The windows and
which clip conditions each are logged when sampling starts.

**Windows are anchored per clip.** With H3 Joint Conditioning on the
wire, each clip gets its own run of windows over the rows the cut shows
from it: from its raw start to the step where the next clip takes over.
Adjacent runs overlap exactly over the next clip's held head — the rows
both clips agree on — and that is the only place two conditionings
blend. Windows on one grid across the timeline blended two clips'
conditionings wherever neighbouring windows happened to belong to
different clips, mid-body, where the clips' prompts and references may
disagree; anchoring the runs keeps each clip's body under its own
conditioning. Inside a run the windows follow
`window_seconds`/`overlap_seconds`; a clip shorter than a window gets
one window its own length. Without the table the windows fall back to
one grid over the timeline.

**Windows sit on the latent grid.** H3's latent runs in cycles of five
steps covering (1, 4, 4, 4, 4) frames, and every clip latent is 5k+2
steps starting at cycle phase 0. A window that starts off the cycle
shows the model a latent whose frame cycle is shifted from anything it
was trained on, and it paints a periodic artefact: measured 2026-09-06
as a flicker every 17 frames (one cycle) exactly under a window that
started at step 356, and absent under windows starting at 0 and 70. So
every window starts on a multiple of five steps, its length is snapped
to 5j+2, the gaps between starts are whole cycles no longer than
window minus overlap (an overlap can only grow), and the last window is
stretched to the end of the timeline rather than started off the cycle.

**Window size.** The work per step is the same at any window size —
the overlaps scale with the window, so a 417-step timeline is 552
window-steps at 13 s and 576 at 5 s. What changes is memory:
activations scale with the tokens in one call, so a 5 s window at
1920×1088 needs about 7 GB and a 13 s window more than 17 GB (measured:
one attention layer's QKV projection alone is 7.8 GB). What a longer
window buys is direct context — a join in the middle of a 13 s window
is seen with 6 s on either side, in a 5 s window with 2.5 s, and
consistency beyond that travels through the chain of overlaps, blended
at every step. The mechanism that fixed the seams (every row sampled
together, one step at a time, nothing committed ahead of its
neighbours) does not depend on the window length; the per-clip refine
saw a whole clip per call and still failed. Fit the window to the card
first, then lengthen it if a fast move across a window edge shows.

**Memory.** Core decides how much of the model to keep in VRAM from an
estimate of the sampling's activations, and it makes that estimate for
the whole latent in its packed form — for a seven-clip timeline that is
182 GB, so it loads no weights at all and streams 20 GB from RAM on
every window. The patch replaces the estimate with one window's (plus its
overlap) real need, sized from its token count at about 120 KB a
token, so core keeps that much VRAM free of weights; weights that no
longer fit beside it stream from RAM, about a second per window.
A window whose activations do not fit beside the weights spills into
system RAM and runs ten times slower, and the per-step summary line
reports the measured peak, so a spill shows in the log; shorten the
window if it does. The blended
prediction accumulates in system RAM, so the card holds one window's
tensors and the result, not the timeline several times over. The log
line `VRAM budgeted for one window` shows the numbers; the load line
after it should read `loaded partially` with most of the model
resident. The windows of each step are drawn as a console gauge under
the sampler's step gauge (window range, owning clip, seconds, peak),
and one line per step then reports what it cost: `sigma S: N window(s)
in T s, peak +X MB over Y MB resident (reserved R of T MB)` — X is the
largest window's activations (measured: 9.2 GB for a 5 s window at
1920×1088), Y the latents and buffers already on the card (the dynamic
loader's weights are not counted there; `reserved` is the honest
total), and a `spill is likely` suffix means the card was full.
Per-window detail is at DEBUG. X decides the window: activations scale
with `window_seconds`, so if the suffix appears or windows crawl,
shorten the window until X fits beside a few GB of weights.

## H3 Joint VAE Decode and Save

The end of the upscale branch: the sampler's output — the refined joint
AV latent — decoded and written as **one finished MP4**, with a
`.mctx.safetensors` sidecar beside it holding the refined latent and,
when `conditioning` is wired, a `.cond` holding what the sampler ran
under. The rendered cut is therefore a take: put it on a timeline by
itself and the same upscale branch refines it again (4x from the
sources), with H3 Joint Conditioning splicing its stored table back in.

| Input | Type | Notes |
|---|---|---|
| `samples` | LATENT | the sampler's output |
| `vae` / `audio_vae` | VAE | the H3 video and audio VAEs |
| `base_folder` | STRING | output-relative folder, same meaning as the Timeline's; empty = the output root |
| `filename_prefix` | STRING | filename prefix within `base_folder` (default `upscale`); numbering is appended |
| `crf` | INT | H.264 quality |
| `window_seconds` | FLOAT | seconds of picture decoded at a time (default 5). Only memory changes with it — the frames of one window sit in RAM while they are encoded, about 5 GB for 5 s at 1920×1088. The output is identical at any value |
| `conditioning` | CONDITIONING (optional) | H3 Joint Conditioning's output, stored beside the render as `.cond`. Unwired = the render cannot be refined again |
| `layout` | JOINT (optional) | from Join Latents: the source clips and sequence lines are recorded in the sidecar, and a latent that is not that timeline is refused |
| `source_audio` | LATENT (optional) | an AV latent whose **audio** is rendered instead of the sampled one — wire Join Latents' `latent` here when Audio Mask re-sampled the sound for lip sync. The sidecar stores the audio that was rendered |

Output: `path`, the written MP4. Wire it to H3 MCtx Result Preview to
watch the cut when the pass ends.

**Decoded in windows, streamed to the encoder.** A timeline is long —
1400 frames at 1920×1088 are 35 GB of float pixels — so the frames
never exist all at once: each window's frames are piped into ffmpeg as
they are decoded. The window is exact, not a tiling approximation:
core's H3 decoder already works in chunks of 5 latent steps with a
2-step lookahead and blends 5 frames into the next chunk, so a window
that starts and ends on 5-step boundaries, decoded with one chunk of
run-in before it and the lookahead after it, reproduces the
whole-latent decode (the run-in's 17 frames are dropped). Without an
ffmpeg binary the frames are gathered for the in-process encoder
instead, and a warning says what that costs.
