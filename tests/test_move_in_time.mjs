// Ctrl+drag "move in time", LIFTED from web/h3_mctx_ui.js (not copied):
// canMoveInTime and the empty-space writer setGapBefore, run on the real
// sequence grammar and seam derivation. No DOM needed.
//   node tests/test_move_in_time.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { specHandover } from "../web/h3_handover.js";

const SRC = readFileSync(new URL("../web/h3_mctx_ui.js", import.meta.url), "utf8");
const lift = (name) => {
    const at = SRC.indexOf("function " + name + "(");
    assert.ok(at >= 0, name + " is gone from the source");
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

const seqWidget = { value: "" };
const body = [
    ...["TL_LINE_RE", "TL_GAP_RE", "TL_LOOP_RE", "TL_OPTS_RE", "TL_SEAM_KEYS",
        "TL_SEAM_ORDER", "TL_TRUE", "TL_FALSE", "TL_PART_KEY",
        "TL_PART_RE"].map(liftConst),
    ...["tlIsEntryLine", "tlSplitSeamOpts", "tlFormatSeamOpts", "tlParseSequence",
        "tlFormatLine", "tlPins", "tlExtendParent", "tlPrependChild", "tlDeriveSeam",
        "linkedMeta", "currentLines", "entryLines", "gapBefore", "canMoveInTime",
        "setGapBefore"].map(lift),
    // the panel's writer, minus the widget plumbing and the re-render
    "function setSequence(text) { seqWidget.value = text; }",
    "return { tlParseSequence, tlDeriveSeam, gapBefore, canMoveInTime, setGapBefore };",
].join("\n");
const ui = new Function("seqWidget", "specHandover", body)(seqWidget, specHandover);

// Sidecar headers, single-pin fast path. A extends into B at frame 90 of
// A's 120 (a trim-point join: A exits early); C prepends into D, which
// then enters at 30; B2 extends A at its very end (a clean join).
const META = {
    "a.mp4": { self_id: "a", delivered_frames: 120, pinned_head_frames: 0 },
    "b.mp4": { self_id: "b", relation: "extends", parent_id: "a",
               parent_join_frame: 90, delivered_frames: 100, pinned_head_frames: 0 },
    "b2.mp4": { self_id: "b2", relation: "extends", parent_id: "a",
                parent_join_frame: 120, delivered_frames: 100, pinned_head_frames: 0 },
    "c.mp4": { self_id: "c", relation: "prepends", parent_id: "d",
               parent_join_frame: 30, delivered_frames: 80, pinned_head_frames: 0 },
    "d.mp4": { self_id: "d", delivered_frames: 110, pinned_head_frames: 0 },
    "x.mp4": { self_id: "x", delivered_frames: 60, pinned_head_frames: 0 },
};

// The rendered entry list, the way refresh() builds it (mirrors
// resolve_sequence): gaps break the chain, manual cuts win.
function resolve(text) {
    const es = ui.tlParseSequence(text);
    es.forEach((e) => {
        e.meta = e.gap != null ? null : META[e.clip];
        e.frames = e.gap != null ? e.gap : e.meta.delivered_frames;
        e.enter = 0;
        e.exit = null;
    });
    for (let i = 1; i < es.length; i++) {
        if (es[i - 1].gap != null || es[i].gap != null) continue;
        const s = ui.tlDeriveSeam(es[i - 1].meta, es[i].meta);
        if (es[i - 1].exit === null) es[i - 1].exit = s.exitL;
        es[i].enter = s.enterR;
    }
    for (const e of es) {
        if (e.enterOverride !== null) e.enter = e.enterOverride;
        if (e.exitOverride !== null) e.exit = e.exitOverride;
    }
    return es;
}
// what actually plays, in order: [clip|~, enter, exit(delivered if open)]
const plays = (text) => resolve(text).map((e) => e.gap != null
    ? ["~", e.gap] : [e.clip, e.enter, e.exit ?? e.frames]);
// one ctrl+drag of clip i to `frames` of space in front of it
function drag(text, i, frames) {
    seqWidget.value = text;
    const entries = resolve(text);
    assert.ok(ui.canMoveInTime(entries, i), `clip ${i} refused to move`);
    if (frames !== ui.gapBefore(entries, i)) ui.setGapBefore(i, frames, entries);
    return seqWidget.value;
}

// ---- a linked clip may now be moved at all -----------------------------------
assert.equal(ui.canMoveInTime(resolve("a.mp4\nb.mp4"), 1), true);
assert.equal(ui.canMoveInTime(resolve("a.mp4\n~ 5\nb.mp4"), 1), false, "a gap is not a clip");

// ---- trim-point extend: separating keeps what plays, rejoining restores ------
{
    const joined = "a.mp4\nb.mp4";
    assert.deepEqual(plays(joined), [["a.mp4", 0, 90], ["b.mp4", 0, 100]]);
    const apart = drag(joined, 1, 20);
    assert.equal(apart, "a.mp4 @ ..90\n~ 20\nb.mp4");
    // same frames either side of the hole: nothing jumps, nothing replays
    assert.deepEqual(plays(apart), [["a.mp4", 0, 90], ["~", 20], ["b.mp4", 0, 100]]);
    // resizing the hole leaves the clips' lines alone
    assert.equal(drag(apart, 2, 34), "a.mp4 @ ..90\n~ 34\nb.mp4");
    // flush again: the frozen exit equals the derived one, so it goes
    assert.equal(drag(apart, 2, 0), joined);
}

// ---- trim-point prepend: the child's late enter is kept the same way ---------
{
    const joined = "c.mp4\nd.mp4";
    assert.deepEqual(plays(joined), [["c.mp4", 0, 80], ["d.mp4", 30, 110]]);
    const apart = drag(joined, 1, 17);
    assert.equal(apart, "c.mp4\n~ 17\nd.mp4 @ 30");
    assert.deepEqual(plays(apart), [["c.mp4", 0, 80], ["~", 17], ["d.mp4", 30, 110]]);
    assert.equal(drag(apart, 2, 0), joined);
}

// ---- clean extend: no cut to keep, so only the hole is written ---------------
assert.equal(drag("a.mp4\nb2.mp4", 1, 10), "a.mp4\n~ 10\nb2.mp4");
assert.equal(drag("a.mp4\n~ 10\nb2.mp4", 2, 0), "a.mp4\nb2.mp4");

// ---- a manual cut the user made survives both ways ---------------------------
{
    const apart = drag("a.mp4 @ ..80\nb.mp4", 1, 5);
    assert.equal(apart, "a.mp4 @ ..80\n~ 5\nb.mp4");
    // 80 is not what the seam derives (90): it was a decision, it stays
    assert.equal(drag(apart, 2, 0), "a.mp4 @ ..80\nb.mp4");
}

// ---- seam settings and neighbours ride along untouched -----------------------
assert.equal(drag("x.mp4\na.mp4\nb.mp4 [crossfade=on]", 2, 12),
             "x.mp4\na.mp4 @ ..90\n~ 12\nb.mp4 [crossfade=on]");

// ---- unlinked clips behave as before -----------------------------------------
assert.equal(drag("a.mp4\nx.mp4", 1, 7), "a.mp4\n~ 7\nx.mp4");
assert.equal(drag("a.mp4\n~ 7\nx.mp4", 2, 0), "a.mp4\nx.mp4");

console.log("test_move_in_time: all passed");
