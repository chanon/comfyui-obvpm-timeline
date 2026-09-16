// The handover of a pinned window, mirrored from nodes_masked.py.
//
// The strip derives its seams in the browser (tlDeriveSeam) so a cut
// answers on every repaint without a round trip, and the server derives
// them again for the build (_derive_seam). The two MUST agree, or the
// quick preview plays a join the export does not have. For a take with a
// single pin they read the same precomputed parent_join_frame; a
// multi-pin take (a bridge, a loop) carries its parents only in the pins
// recipe, and the join there is source_start + the window's HANDOVER --
// which for a softly held window is not the window edge. This module is
// that arithmetic, kept free of ComfyUI imports so tests/test_handover_
// mirror.py can run it under node against the Python original.
//
// Every function here mirrors one in frames.py / nodes_masked.py by name.

export const FRAME_PER_TOKEN = [1, 4, 4, 4, 4];

// frames.frame_at_latent: first pixel frame of latent step k
export function frameAtLatent(k) {
    k = Math.trunc(Number(k) || 0);
    if (k <= 0) return 0;
    const n = FRAME_PER_TOKEN.length;
    const full = Math.floor(k / n), rem = k % n;
    let sum = 0;
    for (const f of FRAME_PER_TOKEN) sum += f;
    let part = 0;
    for (let i = 0; i < rem; i++) part += FRAME_PER_TOKEN[i];
    return full * sum + part;
}

// frames.steps_for_frames: steps covering EXACTLY n frames, else null
export function stepsForFrames(n) {
    n = Math.trunc(Number(n) || 0);
    let k = 0, covered = 0;
    while (covered < n) {
        covered += FRAME_PER_TOKEN[k % FRAME_PER_TOKEN.length];
        k += 1;
    }
    return covered === n ? k : null;
}

// frames.step_offsets
export function stepOffsets(steps) {
    const out = [];
    for (let k = 0; k < steps; k++) out.push(frameAtLatent(k));
    return out;
}

const clamp01 = (v) => Math.min(1, Math.max(0, Number(v) || 0));

// nodes_masked.mask_profile: per-step mask values, join-relative
export function maskProfile(covered, steps, place, rampFrames = 0, edge = 0, deep = 0) {
    steps = Math.trunc(steps);
    covered = Math.trunc(covered);
    edge = clamp01(edge);
    deep = clamp01(deep);
    rampFrames = Math.max(0, Math.trunc(Number(rampFrames) || 0));
    if (steps <= 0) return [];
    if (rampFrames <= 0 || edge === deep) return new Array(steps).fill(deep);
    const starts = stepOffsets(steps);
    const ends = starts.slice(1).concat([covered]);
    const dist = place === "before" ? ends.map((e) => covered - e) : starts.slice();
    const span = rampFrames;
    return dist.map((d) => edge + (deep - edge) * Math.min(1, Math.max(0, d / span)));
}

// nodes_masked.handover_frames: offset INSIDE the window where the two
// sides change hands. after-pin: the take delivers [0, n), the target
// enters at n. before-pin: the parent plays to n, the take delivers
// [n, covered). A hard hold gives 0 / covered.
export function handoverFrames(covered, place, rampFrames = 0, edge = 0, deep = 0) {
    covered = Math.trunc(Number(covered) || 0);
    rampFrames = Math.trunc(Number(rampFrames) || 0);
    deep = Number(deep) || 0;
    if (rampFrames <= 0 && deep >= 1.0) return place === "before" ? 0 : covered;
    const steps = stepsForFrames(covered);
    if (steps == null) return place === "after" ? 0 : covered;
    const prof = maskProfile(covered, steps, place, rampFrames, edge, deep);
    const held = [];
    prof.forEach((v, k) => { if (v === 0) held.push(k); });
    if (!held.length) return place === "after" ? 0 : covered;
    const starts = stepOffsets(steps);
    if (place === "after") return starts[held[0]];
    const ends = starts.slice(1).concat([covered]);
    return ends[held[held.length - 1]];
}

// The handover a pins-recipe entry implies, read the way
// nodes_assemble._handover reads it (absent shape keys = hard hold).
export function specHandover(spec, place) {
    return handoverFrames(
        Number(spec?.source_frames) || 0, place,
        Number(spec?.mask_ramp_frames) || 0,
        Number(spec?.mask_ramp_edge) || 0,
        Number(spec?.mask_hold) || 0);
}
