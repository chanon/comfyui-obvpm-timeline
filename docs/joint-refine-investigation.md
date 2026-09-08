# Joint refine: the artefact investigation (2026-09-06 to 2026-09-08)

A lab record of how the artefacts in the joint refine were tracked down.
Kept because most of the hypotheses along the way were plausible, several
were "confirmed" by a run that later turned out to be luck, and the real
cause was in the one stage nobody suspected. Footage: `selfie_walk3`,
seven clips, refined at 1920x1088 from 960x544 sources.

## The symptoms

- **Hard white points on dark hair** under dappled sunlight, 23 to 27 s.
- **Marks on the face**, 8 s: a wireframe rounded rectangle over the
  forehead where the source has wind-blown hair strands.
- **Coloured glyph-like streaks** on hair (early runs).
- **Shimmering eyes** (late runs).
- Earlier, and separately: a 17-frame periodic flicker, a clip shown for
  four seconds under the wrong audio, and a flicker at the first seam.

## What was real and is fixed

1. **Off-cycle window starts** (2026-09-06). Windows that did not start
   on the latent's 5-step cycle painted a flicker every 17 frames.
   Measured by frame differencing against a clean run. Fix: windows start
   on the cycle, lengths clip-shaped (`static_windows`, `clip_shaped`).
2. **Placement by the cut instead of by the pin** (2026-09-07). A parent
   cut before extending (clip 5 shown to frame 158) put its child 34
   frames late; the child's held head sat on the wrong rows (rel diff 1.25
   where a masked hold must be 0.0000). And first-come overlaps let a cut
   clip's discarded tail win 102 frames of the next clip: four seconds of
   the wrong picture under the right audio. Fix: `raw_starts` places
   pinned clips by `source_start`, `owners` gives every row to the clip
   the cut shows, `assemble` warns when a hard hold does not match.
3. **Log lines printed twice and mid-bar.** tqdm's `logging_redirect_tqdm`
   only recognises a console handler whose stream *is* `sys.stderr`; the
   manager wraps stderr after core creates its handler, so tqdm added a
   bare second handler. Fix: our own handler swap (`_log_under_bars`).
4. **The upscaler normalises over the whole tensor** (2026-09-08, the
   root cause of the hallucinations, see below).

## What looked like the cause and was not

Each of these was tested with a run and eliminated. The number in
brackets is the profile.

| hypothesis | test | result |
|---|---|---|
| conditioning changes hands inside a window blend | one conditioning for every window (`single_clip`) [joint_single] | 8 s marks gone, hair points stayed; the "fix" was later shown to be noise luck |
| windows blend across clip bodies | windows anchored per clip, hand-over only over a held head [joint_anchored] | still points |
| refine strength too high | first sigma bracket 0.05 to 0.2 on the pair | points scale with noise, clean at real start <= 0.47, but foliage too soft at 0.39 |
| the turbo LoRA over-contrasts | no LoRA, 20 steps [joint_pair_vanilla]; 4 steps [joint_pair_4step] | still points |
| the latent upscaler puts them in the prior | decode the prior with no refine [joint_prior] | prior looks clean at contact-sheet scale (true, and still the culprit: see below) |
| averaging two windows' predictions superposes detail | `cut` fuse, every row from one window [joint_pair_cut, joint_all_cut] | still artefacts |
| a fixed seed hides structural causes | seed 1266 -> 1267 [joint_all_seed1267] | artefacts moved and changed: they are noise-realisation hallucinations |
| windows start mid-clip with no frame-0 keyframe | prior's frame as a frame-0 keyframe per window [joint_all_anchor] | 16 of 17 windows anchored, still artefacts |
| per-step trajectory mixing across windows | sequential complete samplings with frozen overlap [joint_all_seq] | still artefacts |
| per-clip conditioning | sequential, one conditioning [joint_pair_seq_single] | still artefacts |

Useful negative facts along the way: the joint prior is continuous at
every hand-over (row-to-row change within a few percent of its
neighbours), the audio latent is too, the audio is quiet where the hair
points appear, and the sampled latent has no outlier cells.

## The decisive comparison

MMH3 Ultimate Upscale (bbaudio-2025, MIT) samples the timeline in
sequential chunks with a frame-0 anchor and a cross-fade. On the full
timeline it was clean but showed detail changes at its seams
[joint_mmh3]. Fed **our** upscaled prior with its own upscale stage off,
on the clip 14 + 15 pair, it hallucinated like every route of ours
[joint_pair_mmh3_ourprior]. With its **own** per-chunk upscale on the same
pair, same conditioning, same seed, it was clean [joint_pair_mmh3]. The
only difference was the prior.

## The mechanism

Both upscaler implementations load the same checkpoint, both with the
attention blocks off, both with the same z-score normalisation; on the
same 40-step chunk their outputs differ only by chunk-edge effects. But
the network's normalisation layers are GroupNorm, whose statistics run
over channels, **time**, height and width together. Upscaling the whole
407-step joint in one pass normalises every frame against the
statistics of the entire video. Trained on short clips, the network
never saw that, and its response shifts everywhere: interior rows of
clip 14 differ by 6 % from the clip upscaled alone, the output variance
is 2 % higher. The refine then treats that prior as unfamiliar detail
and invents structure on it, in every sampling scheme.

Neither the per-clip refine (one 87-step clip per upscale) nor MMH3
(136-frame chunks) ever upscaled more than a few seconds at once. Only
the joint route did, with the upscaler node's own temporal chunking
switched off.

## The fix

Upscale in short chunks. `MinimaxH3LatentUpscaler3D` has
`enable_temporal_chunking` (32-step chunks, 5-step overlap blended
inside the network); it was off in every joint workflow. The joint
route otherwise stays as it was: per-step windows with pyramid blending,
anchored per clip.

Confirmed 2026-09-08 with `h3_obvpm_r2v_pre_vid_joint_all_chunkup`: all
seven clips at first sigma 0.2, no artefacts at 8 s or 22 to 27 s, and no
visible seams or detail changes at the hand-overs. The joint route works
at full refine strength once the prior is upscaled in chunks.

## Things learned about the tools

- `first_sigma` is BasicScheduler's `denoise`; with H3's shift of 12
  the real start noise is 12d/(1+11d): 0.2 -> 0.75, 0.075 -> 0.49.
- With a fixed seed, every run with the same latent shape samples under
  the same noise field. Two 7-clip runs agreeing on where an artefact
  sits proves nothing; a 5-clip run disagreeing proves it is noise.
- Judge a mechanism at the noise level where the failure is reliable,
  not by finding a level where it is clean.
- Core's VRAM estimate counts the 24 channels of a packed AV latent as
  area (24x too big) and offloads every weight; anything that samples a
  packed window must register a PREPARE_SAMPLING wrapper that unpacks
  the shape.
- A CleanVRAM node inside the decode subgraph runs once per loop
  iteration and unloads the VAEs before every clip.

## Code that stays, code that is optional

Stays: pin-based placement and cut ownership, grid-aligned windows,
anchored-per-clip windows, the estimate wrapper, the log handler swap.

Removed once the cause was confirmed (2026-09-08, after commit
744fc88): `fuse_method = cut`, the `prior` / `anchor_strength` frame-0
keyframe on Context Windows, `single_clip` on Joint Conditioning, and
the `H3 Joint Sequential Refine` node. None of them changed the
hallucination; they only existed to test theories this record closes.
The code is in the history if a theory is ever reopened.
