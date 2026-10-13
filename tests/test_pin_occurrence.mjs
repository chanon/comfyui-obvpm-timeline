// Pins on a clip that stands in the sequence more than once (a split
// clip), LIFTED from web/h3_mctx_ui.js (not copied): the pin names ONE
// entry by its part id, the badges and sides follow it, its cut is that
// entry's cut, it survives edits to the cut around it, and a returned take
// lands beside the part it continues from.
//   node tests/test_pin_occurrence.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { specHandover } from "../web/h3_handover.js";

const SRC = readFileSync(new URL("../web/h3_mctx_ui.js", import.meta.url), "utf8");
const lift = (name) => {
    let at = SRC.indexOf("function " + name + "(");
    assert.ok(at >= 0, name + " is gone from the source");
    if (SRC.slice(at - 6, at) === "async ") at -= 6;   // keep it async
    let depth = 0, i = SRC.indexOf("{", at);
    for (;; i += 1) {
        if (SRC[i] === "{") depth += 1;
        if (SRC[i] === "}" && --depth === 0) break;
    }
    return SRC.slice(at, i + 1);
};
// a top-level `const NAME = ...;` (regexes, tables), up to its own `;`
const liftConst = (name) => {
    const at = SRC.indexOf("\nconst " + name + " =");
    assert.ok(at >= 0, name + " is gone from the source");
    let depth = 0, i = at + 1;
    for (;; i += 1) {
        const c = SRC[i];
        if ("{[(".includes(c)) depth += 1;
        if ("}])".includes(c)) depth -= 1;
        if (c === ";" && depth === 0) break;
    }
    return SRC.slice(at + 1, i + 1);
};

// what the panel around them does: the in-page question (tlAskChoice)
// answers from `answers` and counts how often it was asked, setCut is
// recorded, and new part ids count up (p1, p2, ...) so the expected text
// is fixed -- tlNewPart itself is tested on its own below
const answers = [];
const asked = [];
const cuts = [];
const body = [
    ...["TL_LINE_RE", "TL_GAP_RE", "TL_LOOP_RE", "TL_OPTS_RE", "TL_SEAM_KEYS",
        "TL_SEAM_ORDER", "TL_TRUE", "TL_FALSE", "TL_PART_KEY", "TL_PART_RE",
        "TL_GROUP", "TL_WINDOW_PHASE"].map(liftConst),
    ...["tlIsEntryLine", "tlSplitSeamOpts", "tlFormatSeamOpts", "tlParseSequence",
        "tlFormatLine", "tlPins", "tlExtendParent", "tlPrependChild",
        "tlPartEntry", "tlOccOf", "tlTakePart", "tlJoinFits", "tlWindowEdge",
        "tlParentIdx", "reconcileCut",
        "tlOccupant", "linkedMeta", "pinState", "pinIdx", "pinAt", "givePart",
        "pinRoleAt", "pinHasRoleAt", "impliesLoop", "pinName", "setBridge",
        "togglePinAt", "syncPinCut", "currentLines", "entryLines",
        "writeSequence", "reconcileParentCut", "partLabel",
        "tlDedupeParts"].map(lift),
    "let nextId = 0;",
    "function tlNewPart(taken) { let id; do { id = 'p' + (++nextId); } "
    + "while (taken.has(id)); return id; }",
    "const psWidget = { value: '' };",
    "const seqWidget = { value: '' };",
    "function setPinState(st) { psWidget.value = st ? JSON.stringify(st) : ''; }",
    "function setCut(i, c) { cuts.push([i, c]); }",
    "function clipBase(c) { return String(c).split('/').pop(); }",
    "function tlNoteEdit() {}",
    "function loopOn() { return false; }",
    "function setLoop() {}",
    "let lastEntries = [];",
    // the sequence text follows the rendered entries, as refresh keeps it
    "function render(es) { lastEntries = es; nextId = 0;"
    + " seqWidget.value = es.map(tlFormatLine).join('\\n'); }",
    "return { psWidget, seqWidget, render, pinState, pinIdx, pinRoleAt,"
    + " pinHasRoleAt, impliesLoop, pinName, togglePinAt, syncPinCut, tlParentIdx,"
    + " tlJoinFits, tlTakePart, setBridge, reconcileParentCut, tlDedupeParts };",
].join("\n");
const ui = new Function("specHandover", "tlAskChoice", "cuts", body)(
    specHandover, async (q) => { asked.push(q); return answers.shift(); }, cuts);

const A = { self_id: "a", delivered_frames: 120, pinned_head_frames: 0 };
const X = { self_id: "x", delivered_frames: 60, pinned_head_frames: 0 };
const Y = { self_id: "y", relation: "extends", parent_id: "x",
            parent_join_frame: 60, delivered_frames: 60, pinned_head_frames: 0 };
const Z = { self_id: "z", relation: "extends", parent_id: "y",
            parent_join_frame: 60, delivered_frames: 60, pinned_head_frames: 0 };
// one rendered entry: file, its sidecar header, the frames it plays, its id
const ent = (clip, meta, enter = 0, exit = null, cuts = {}, part = null) => ({
    clip, meta, gap: null, enter, exit, frames: meta.delivered_frames,
    enterOverride: cuts.enter ?? null, exitOverride: cuts.exit ?? null,
    seamOpts: {}, part,
});
// a.mp4 split twice: plays 0..40 | 40..80 | 80..120, parts a1 a2 a3
const P1 = () => ent("a.mp4", A, 0, 40, { exit: 40 }, "a1");
const P2 = () => ent("a.mp4", A, 40, 80, { enter: 40, exit: 80 }, "a2");
const P3 = () => ent("a.mp4", A, 80, null, { enter: 80 }, "a3");
const split3 = () => [P1(), P2(), P3()];
const set = (st) => { ui.psWidget.value = st ? JSON.stringify(st) : ""; };
const roles = (es) => es.map((_, i) => ui.pinRoleAt(i, ui.pinState(), es));
const cutOf = (es) => { ui.render(es); ui.syncPinCut(es); return ui.pinState()?.cut; };

// ---- the reported case: prepend on ONE part of a clip split twice ------------
{
    const es = split3();
    ui.render(es);
    set(null);
    ui.togglePinAt(2, "prepend");
    assert.deepEqual(ui.pinState(),
        { source: "a.mp4", part: "a3", role: "prepend", window: "22" });
    assert.deepEqual(roles(es), [null, null, "prepend"], "only that part");
    assert.equal(ui.pinName(ui.pinState()), "a.mp4 (part 3/3)");
    // toggling it again clears it
    ui.togglePinAt(2, "prepend");
    assert.equal(ui.pinState(), null);
}

// ---- extend one part, prepend the next: a bridge across the split ------------
{
    const es = split3();
    ui.render(es);
    set(null);
    ui.togglePinAt(0, "extend");
    ui.togglePinAt(1, "prepend");
    const st = ui.pinState();
    assert.deepEqual([st.role, st.part, st.part2], ["bridge", "a1", "a2"]);
    assert.deepEqual(roles(es), ["extend", "prepend", null]);
    assert.equal(ui.pinHasRoleAt(1, st, "extend", es), false);
    // dropping the extend side keeps the prepend side ON THAT PART
    ui.togglePinAt(0, "extend");
    assert.deepEqual(ui.pinState(),
        { source: "a.mp4", part: "a2", role: "prepend", window: "22" });
}

// ---- the cut a pin carries is its own part's ---------------------------------
{
    const es = split3();
    set({ source: "a.mp4", part: "a1", role: "extend" });
    assert.equal(cutOf(es), 40);
    set({ source: "a.mp4", part: "a2", role: "extend" });
    assert.equal(cutOf(es), 80);
    set({ source: "a.mp4", part: "a3", role: "extend" });
    assert.equal(cutOf(es), undefined, "the last part ends where the file does");
    set({ source: "a.mp4", part: "a3", role: "prepend" });
    assert.equal(cutOf(es), 80);
    set({ role: "bridge", source: "a.mp4", part: "a1", source2: "a.mp4", part2: "a2" });
    ui.syncPinCut(es);
    assert.deepEqual([ui.pinState().cut, ui.pinState().cut2], [40, 40]);
}

// ---- the pin stays on ITS part when the cut around it changes ----------------
// (these three moved it onto another part while pins were positions)
{
    // extend part 2, then delete part 1: still part 2, still from 80
    set({ source: "a.mp4", part: "a2", role: "extend", window: "22" });
    assert.equal(cutOf([P2(), P3()]), 80);
    assert.equal(ui.pinIdx(ui.pinState(), false, [P2(), P3()]), 0);
    // extend the last part, then drag it to the front: follows it, no cut
    set({ source: "a.mp4", part: "a3", role: "extend", window: "22" });
    assert.equal(cutOf([P3(), P1(), P2()]), undefined);
    assert.equal(ui.pinIdx(ui.pinState(), false, [P3(), P1(), P2()]), 0);
    // its part deleted: the pin names nothing (refresh then clears it)
    // rather than another part
    set({ source: "a.mp4", part: "a2", role: "extend", window: "22" });
    assert.equal(ui.pinIdx(ui.pinState(), false, [P1(), P3()]), -1);
}

// ---- a file in the cut once needs no id; one in it twice gets one ------------
{
    ui.render([ent("x.mp4", X), ent("a.mp4", A, 0, null, { exit: 90 })]);
    set(null);
    ui.togglePinAt(1, "extend");
    assert.deepEqual(ui.pinState(),
        { source: "a.mp4", role: "extend", window: "22" });
    assert.equal(ui.seqWidget.value, "x.mp4\na.mp4 @ ..90", "plain text stays plain");
    // the same file twice (dragged in again, not split): the pinned line
    // gets an id, the other one does not need it
    ui.render([ent("a.mp4", A, 0, null, { exit: 90 }), ent("x.mp4", X),
               ent("a.mp4", A)]);
    set(null);
    ui.togglePinAt(2, "extend");
    assert.deepEqual(ui.pinState(),
        { source: "a.mp4", part: "p1", role: "extend", window: "22" });
    assert.equal(ui.seqWidget.value, "a.mp4 @ ..90\nx.mp4\na.mp4 [id=p1]",
                 "only the pinned line, and its cut and settings kept");
    // an id it already has is kept
    ui.render([ent("a.mp4", A, 0, null, {}, "k3x9")]);
    set(null);
    ui.togglePinAt(0, "prepend");
    assert.equal(ui.pinState().part, "k3x9");
    assert.equal(ui.seqWidget.value, "a.mp4 [id=k3x9]");
}

// ---- the loop take: last part into the first --------------------------------
{
    const es = [ent("a.mp4", A, 0, 40, { exit: 40 }, "a1"),
                ent("a.mp4", A, 40, null, { enter: 40 }, "a2")];
    ui.render(es);
    const loop = { role: "bridge", source: "a.mp4", part: "a2",
                   source2: "a.mp4", part2: "a1" };
    assert.equal(ui.impliesLoop(loop, es), true);
    assert.deepEqual((set(loop), roles(es)), ["prepend", "extend"]);
    // the same file the other way round is NOT the loop take
    assert.equal(ui.impliesLoop({ ...loop, part: "a1", part2: "a2" }, es), false);
}

// ---- pins written before part ids mean the file's first entry ----------------
{
    const es = split3();
    assert.equal(ui.pinIdx({ source: "a.mp4", role: "extend" }, false, es), 0);
    assert.equal(ui.pinIdx({ source: "b.mp4", role: "extend" }, false, es), -1);
    // a file in the cut once is named plainly
    ui.render([ent("x.mp4", X)]);
    assert.equal(ui.pinName({ source: "x.mp4", role: "extend" }), "x.mp4");
}

// ---- where a take's join fits ------------------------------------------------
{
    const take = (relation, join) => ({ self_id: "t", relation, parent_id: "a",
        parent_join_frame: join, delivered_frames: 50, pinned_head_frames: 0 });
    const [p1, p2, p3] = split3();
    // extends part 1 (cut at 40): on the cut, or up to a group below it
    assert.equal(ui.tlJoinFits(p1, take("extends", 40), "after"), true);
    assert.equal(ui.tlJoinFits(p1, take("extends", 34), "after"), true);
    assert.equal(ui.tlJoinFits(p1, take("extends", 23), "after"), false);
    assert.equal(ui.tlJoinFits(p1, take("extends", 80), "after"), false);
    // the last part has no exit cut: any join inside it, none before it
    assert.equal(ui.tlJoinFits(p3, take("extends", 120), "after"), true);
    assert.equal(ui.tlJoinFits(p3, take("extends", 100), "after"), true);
    assert.equal(ui.tlJoinFits(p3, take("extends", 40), "after"), false);
    // prepends into part 3 (starts at 80): on the cut, or up to a group after
    assert.equal(ui.tlJoinFits(p3, take("prepends", 80), "before"), true);
    assert.equal(ui.tlJoinFits(p3, take("prepends", 92), "before"), true);
    assert.equal(ui.tlJoinFits(p3, take("prepends", 40), "before"), false);
    assert.equal(ui.tlJoinFits(p2, take("prepends", 40), "before"), true);
    // an uncut clip takes any join inside it, as it always did
    const whole = ent("a.mp4", A);
    assert.equal(ui.tlJoinFits(whole, take("extends", 55), "after"), true);
    assert.equal(ui.tlJoinFits(whole, take("prepends", 55), "before"), true);
    // another file's take never fits
    assert.equal(ui.tlJoinFits(ent("x.mp4", X), take("extends", 40), "after"),
                 false);
}

// ---- a returned take lands beside the part it continues from -----------------
{
    const es = split3();
    const take = (relation, join, rec) => ({ self_id: "t", relation,
        parent_id: "a", parent_join_frame: join, delivered_frames: 50,
        pinned_head_frames: 0,
        pins: rec ? JSON.stringify([{ place: relation === "extends"
                                          ? "before" : "after",
                                      source_id: "a", ...rec }]) : "" });
    // without a recorded part: nearest edge
    assert.equal(ui.tlParentIdx(es, take("extends", 40), "after"), 0);
    // the 17-frame grid moved the take's start a few frames down
    assert.equal(ui.tlParentIdx(es, take("extends", 73), "after"), 1);
    assert.equal(ui.tlParentIdx(es, take("extends", 120), "after"), 2);
    assert.equal(ui.tlParentIdx(es, take("prepends", 80), "before"), 2);
    assert.equal(ui.tlParentIdx(es, take("prepends", 40), "before"), 1);
    // the same cut twice: the pinned one wins the tie
    const twice = [ent("a.mp4", A), ent("x.mp4", X), ent("a.mp4", A)];
    assert.equal(ui.tlParentIdx(twice, take("extends", 120), "after", 2), 2);
    assert.equal(ui.tlParentIdx(twice, take("extends", 120), "after"), 0);
    // WITH one: its part wins, wherever the part's cut has moved to
    const rec2 = { tl_part: "a2", tl_cut: 80 };
    assert.deepEqual(ui.tlTakePart(take("extends", 80, rec2), "after"),
                     { part: "a2", cut: 80 });
    assert.equal(ui.tlTakePart(take("extends", 80), "after"), null);
    const moved = [P1(), ent("a.mp4", A, 40, 45, { enter: 40, exit: 45 }, "a2"), P3()];
    assert.equal(ui.tlParentIdx(moved, take("extends", 80, rec2), "after"), 1,
                 "part 2 cut back to 45 is still where the take belongs");
    assert.equal(ui.tlParentIdx(twice, take("extends", 120,
        { tl_part: "zz", tl_cut: null }), "after"), 0,
        "a part that is gone falls back to the nearest edge");
}

// ---- the part's cut when the take comes back ----------------------------------
{
    const take = (join, rec) => ({ self_id: "t", relation: "extends",
        parent_id: "a", parent_join_frame: join, delivered_frames: 50,
        pinned_head_frames: 0,
        pins: rec ? JSON.stringify([{ place: "before", source_id: "a",
                                      ...rec }]) : "" });
    const at = (exit) => [P1(), ent("a.mp4", A, 40, exit, { enter: 40, exit }, "a2"),
                          P3()];
    set(null);
    const reset = () => { cuts.length = 0; asked.length = 0; };
    // unchanged since queued, and on the grid: nothing to do
    ui.render(at(80));
    reset();
    assert.equal(await ui.reconcileParentCut(take(80, { tl_part: "a2", tl_cut: 80 })),
                 null);
    assert.deepEqual(cuts, []);
    // unchanged since queued, but the grid moved the start: follow it quietly
    ui.render(at(84));
    assert.match(await ui.reconcileParentCut(take(73, { tl_part: "a2", tl_cut: 84 })),
                 /grid moved/);
    assert.deepEqual([cuts, asked.length], [[[1, { exit: 73 }]], 0]);
    // cut changed while the take generated: ASKED -- OK puts it back...
    ui.render(at(60));
    reset();
    answers.push(true);
    assert.match(await ui.reconcileParentCut(take(80, { tl_part: "a2", tl_cut: 80 })),
                 /again/);
    assert.deepEqual(cuts, [[1, { exit: 80 }]]);
    assert.equal(asked.length, 1);
    assert.match(asked[0].message, /cut at frame 80 when this take was queued/);
    // ...Cancel keeps the user's cut, and says what that means
    reset();
    answers.push(false);
    assert.match(await ui.reconcileParentCut(take(80, { tl_part: "a2", tl_cut: 80 })),
                 /kept your cut.*plays as a cut/);
    assert.deepEqual(cuts, []);
    // its part is gone: no cut is touched, and the note says so
    ui.render([P1(), P3()]);
    reset();
    assert.match(await ui.reconcileParentCut(take(80, { tl_part: "a2", tl_cut: 80 })),
                 /no longer in the cut/);
    assert.deepEqual([cuts, asked.length], [[], 0]);
    // an older take (no part recorded) keeps the old proof: the pin's cut
    ui.render(at(90));
    set({ source: "a.mp4", part: "a2", role: "extend", cut: 90 });
    assert.match(await ui.reconcileParentCut(take(80)), /where this take continues/);
    assert.deepEqual(cuts, [[1, { exit: 80 }]]);
    set(null);
}

// ---- the recorded part was SPLIT after the take was queued -------------------
// (the review's bug: the id stays on the left half, but the edge the take
// continues from went to the right one)
{
    const take = (join, rec) => ({ self_id: "t", relation: "extends",
        parent_id: "a", parent_join_frame: join, delivered_frames: 50,
        pinned_head_frames: 0,
        pins: JSON.stringify([{ place: "before", source_id: "a", ...rec }]) });
    const reset = () => { cuts.length = 0; asked.length = 0; };
    // queued from a1's END (the whole clip was a1), then split at 40
    const halves = [ent("a.mp4", A, 0, 40, { exit: 40 }, "a1"),
                    ent("a.mp4", A, 40, null, { enter: 40 }, "b7")];
    const fromEnd = take(120, { tl_part: "a1", tl_cut: null });
    assert.equal(ui.tlParentIdx(halves, fromEnd, "after"), 1,
                 "goes after the half that now owns the end");
    ui.render(halves);
    reset();
    assert.equal(await ui.reconcileParentCut(fromEnd), null);
    assert.deepEqual([cuts, asked.length], [[], 0], "nothing asked, no cut moved");
    // queued from a cut at 60, then the part split BELOW it at 30: the
    // new part (30..60) owns the cut now
    const below = [ent("a.mp4", A, 0, 30, { exit: 30 }, "a1"),
                   ent("a.mp4", A, 30, 60, { enter: 30, exit: 60 }, "c2"),
                   ent("a.mp4", A, 60, null, { enter: 60 }, "a2")];
    const fromCut = take(56, { tl_part: "a1", tl_cut: 60 });
    assert.equal(ui.tlParentIdx(below, fromCut, "after"), 1);
    ui.render(below);
    reset();
    assert.equal(await ui.reconcileParentCut(fromCut), null);
    assert.deepEqual([cuts, asked.length], [[], 0]);
    // ...while a part whose cut was only MOVED, with nothing else fitting,
    // is still its own -- and asked about
    const moved = [ent("a.mp4", A, 0, 30, { exit: 30 }, "a1"),
                   ent("a.mp4", A, 60, null, { enter: 60 }, "a2")];
    assert.equal(ui.tlParentIdx(moved, fromCut, "after"), 0);
}

// ---- a bridge into the right half of a split: its START follows the take ------
// (the reported case, clip_00009: queued with the right part cut at 132; the
// grid moved the window start to 136 and the soft hold hands over at 149, so
// the part kept replaying 132..148 after the take)
{
    const S = { self_id: "s", delivered_frames: 200, pinned_head_frames: 0 };
    const bridge = (cut2) => ({ self_id: "t", relation: "", delivered_frames: 76,
        pinned_head_frames: 39, pins: JSON.stringify([
            { source_id: "s", source_kind: "clip", source_start: 17, source_frames: 39,
              place: "before", mode: "masked", tl_part: "a1", tl_cut: 56 },
            { source_id: "s", source_kind: "clip", source_start: 136, source_frames: 39,
              place: "after", mode: "both", mask_ramp_frames: 10, mask_ramp_edge: 0.4,
              tl_part: "a2", tl_cut: cut2 }]) });
    const cut = (enter) => [ent("s.mp4", S, 0, 56, { exit: 56 }, "a1"),
                            ent("s.mp4", S, enter, 192, { enter, exit: 192 }, "a2")];
    const reset = () => { cuts.length = 0; asked.length = 0; };
    set(null);
    // the part still fits: the grid moved the WINDOW 4 frames, the soft
    // hold's 13 more are not the grid's doing
    assert.equal(ui.tlJoinFits(cut(132)[1], bridge(132), "before"), true);
    assert.equal(ui.tlParentIdx(cut(132), bridge(132), "before"), 1);
    // unchanged since queued: the start follows the take quietly, the
    // extend side (already at its join) is left alone
    ui.render(cut(132));
    reset();
    assert.match(await ui.reconcileParentCut(bridge(132)),
                 /starts at 149 instead of 132, where this take hands over/);
    assert.deepEqual([cuts, asked.length], [[[1, { enter: 149 }]], 0]);
    // changed while the take generated: asked, and OK puts it at the join
    ui.render(cut(120));
    reset();
    answers.push(true);
    assert.match(await ui.reconcileParentCut(bridge(132)), /again/);
    assert.deepEqual(cuts, [[1, { enter: 149 }]]);
    assert.match(asked[0].message, /hands over at frame 149/);
    // already at the join: nothing to do
    ui.render(cut(149));
    reset();
    assert.equal(await ui.reconcileParentCut(bridge(149)), null);
    assert.deepEqual([cuts, asked.length], [[], 0]);
    // an older prepend take (no part recorded): only the pin proves the cut
    const old = { self_id: "u", relation: "prepends", parent_id: "s",
                  parent_join_frame: 149, delivered_frames: 50, pinned_head_frames: 0 };
    ui.render(cut(132));
    reset();
    assert.equal(await ui.reconcileParentCut(old), null, "no pin: left alone");
    set({ source: "s.mp4", part: "a2", role: "prepend", cut: 132 });
    assert.match(await ui.reconcileParentCut(old), /starts at 149 instead of 132/);
    assert.deepEqual(cuts, [[1, { enter: 149 }]]);
    set(null);
}

// ---- repeated ids from a copied line: later copies get fresh ones ------------
{
    const text = "# note\na.mp4 @ ..40 [id=a1]\n~ 12\na.mp4 @ ..40 [id=a1 crossfade=on]\n"
        + "loop\na.mp4 @ 40 [id=a2]";
    const out = ui.tlDedupeParts(text).split("\n");
    assert.equal(out[0], "# note");
    assert.equal(out[1], "a.mp4 @ ..40 [id=a1]", "the first keeps its id");
    assert.equal(out[2], "~ 12");
    assert.match(out[3], /^a\.mp4 @ \.\.40 \[id=(\w+) crossfade=on\]$/);
    assert.notEqual(out[3].match(/id=(\w+)/)[1], "a1");
    assert.equal(out[4], "loop");
    assert.equal(out[5], "a.mp4 @ 40 [id=a2]");
    const plain = "a.mp4\nb.mp4 @ 5";
    assert.equal(ui.tlDedupeParts(plain), plain, "nothing to do: untouched");
}

// ---- regenerating the middle of a chain still upgrades to a bridge -----------
{
    // x -> y -> z, with x split: extending x's LAST part replaces y and
    // binds to z
    const es = [ent("x.mp4", X, 0, 30, { exit: 30 }, "x1"),
                ent("x.mp4", X, 30, 60, { enter: 30 }, "x2"),
                ent("y.mp4", Y), ent("z.mp4", Z)];
    ui.render(es);
    set(null);
    ui.togglePinAt(1, "extend");
    const st = ui.pinState();
    assert.deepEqual([st.role, st.source, st.part, st.source2, st.part2],
                     ["bridge", "x.mp4", "x2", "z.mp4", undefined],
                     "z is in the cut once: no id needed");
    // its FIRST part's neighbour is its own other part: nothing to
    // replace, so a plain extend from that part's cut
    ui.render(es);
    set(null);
    ui.togglePinAt(0, "extend");
    assert.deepEqual(ui.pinState(),
        { source: "x.mp4", part: "x1", role: "extend", window: "22" });
    // prepending into z replaces y and binds to x's last part
    ui.render(es);
    set(null);
    ui.togglePinAt(3, "prepend");
    const pr = ui.pinState();
    assert.deepEqual([pr.role, pr.source, pr.part, pr.source2],
                     ["bridge", "x.mp4", "x2", "z.mp4"]);
    // y dragged away from x: no longer the neighbour, no upgrade
    const moved = [ent("x.mp4", X), ent("z.mp4", Z), ent("y.mp4", Y)];
    ui.render(moved);
    set(null);
    ui.togglePinAt(0, "extend");
    assert.equal(ui.pinState().role, "extend");
}

// ---- new part ids: short, well-formed, never one already in use --------------
{
    const [nl] = ["tlNewPart"].map(lift);
    const [re] = ["TL_PART_RE"].map(liftConst);
    const { tlNewPart, TL_PART_RE } = new Function(
        re + "\n" + nl + "\nreturn { tlNewPart, TL_PART_RE };")();
    const taken = new Set();
    for (let k = 0; k < 2000; k++) {
        const id = tlNewPart(taken);
        assert.ok(TL_PART_RE.test(id) && id.length === 4, id);
        assert.ok(!taken.has(id), "reused " + id);
        taken.add(id);
    }
}

console.log("test_pin_occurrence: all passed");
