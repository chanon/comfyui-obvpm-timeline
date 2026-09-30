// The split button, LIFTED from web/h3_mctx_ui.js (not copied): where the
// playhead may split a clip, the snapped split point, and the two lines the
// split writes -- run on the real sequence grammar and seam derivation.
//   node tests/test_split.mjs
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

// the panel state the split reads: the text, the playhead and the snap
// toggle here; the rendered entries and each one's start frame are the
// panel's own `lastEntries` / `cumStarts`, set through `render`
const st = { seq: "", frame: null, snap: true };
const body = [
    ...["TL_LINE_RE", "TL_GAP_RE", "TL_LOOP_RE", "TL_OPTS_RE", "TL_SEAM_KEYS",
        "TL_SEAM_ORDER", "TL_TRUE", "TL_FALSE", "TL_PART_KEY", "TL_PART_RE",
        "TL_GROUP", "TL_WINDOW_PHASE", "tlCutPhase"].map(liftConst),
    ...["tlIsEntryLine", "tlSplitSeamOpts", "tlFormatSeamOpts", "tlParseSequence",
        "tlFormatLine", "tlPins", "tlExtendParent", "tlPrependChild", "tlDeriveSeam",
        "tlCutStops", "currentLines", "entryLines", "clipAt", "playheadSplit",
        "splitPoint", "applyPlayheadSplit", "splitEntry", "tlPartEntry",
        "pinState", "pinIdx"].map(lift),
    // new part ids count up (p1, p2, ...) so the expected text is fixed
    "let nextId = 0;",
    "function tlNewPart(taken) { let id; do { id = 'p' + (++nextId); } "
    + "while (taken.has(id)); return id; }",
    "const psWidget = { value: '' };",
    "const seqWidget = { get value() { return st.seq; } };",
    "function setSequence(text) { st.seq = text; }",
    "function displayFrame() { return st.frame; }",
    "function snapOn() { return st.snap; }",
    "let lastEntries = [], cumStarts = [];",
    "function render(es, starts) { lastEntries = es; cumStarts = starts;"
    + " nextId = 0; }",
    "return { tlParseSequence, tlDeriveSeam, playheadSplit, applyPlayheadSplit,"
    + " render, psWidget };",
].join("\n");
const ui = new Function("st", "specHandover", body)(st, specHandover);

const META = {
    // plain clip, 120 frames
    "a.mp4": { self_id: "a", delivered_frames: 120, pinned_head_frames: 0 },
    // extends A at 90: A exits at 90 when B follows it
    "b.mp4": { self_id: "b", relation: "extends", parent_id: "a",
               parent_join_frame: 90, delivered_frames: 100, pinned_head_frames: 0 },
    "x.mp4": { self_id: "x", delivered_frames: 60, pinned_head_frames: 0 },
    // a continuation: its delivered frame 0 is raw frame 22
    "c.mp4": { self_id: "c", delivered_frames: 80, pinned_head_frames: 22 },
    "tiny.mp4": { self_id: "t", delivered_frames: 10, pinned_head_frames: 0 },
};

// render the way refresh() does (mirrors resolve_sequence)
function load(text) {
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
    const starts = [];
    let t = 0;
    for (const e of es) {
        starts.push(t);
        t += e.gap != null ? e.gap : (e.exit ?? e.frames) - e.enter;
    }
    st.seq = text;
    ui.render(es, starts);
    return es;
}
const plays = (text) => load(text).map((e) => e.gap != null
    ? ["~", e.gap] : [e.clip, e.enter, e.exit ?? e.frames]);
function split(text, frame, snap = false) {
    load(text);
    st.frame = frame;
    st.snap = snap;
    ui.applyPlayheadSplit();
    return st.seq;
}

// ---- a plain clip splits in two, and plays exactly as before ------------------
{
    const text = split("a.mp4", 50);
    assert.equal(text, "a.mp4 @ ..50 [id=p1]\na.mp4 @ 50 [id=p2]");
    assert.deepEqual(plays(text), [["a.mp4", 0, 50], ["a.mp4", 50, 120]]);
}

// ---- only the clip under the playhead; neighbours and their seams stay --------
{
    // A exits at 90 because B extends it there; split A at 30
    const text = split("x.mp4\na.mp4\nb.mp4", 60 + 30);
    assert.equal(text, "x.mp4\na.mp4 @ ..30 [id=p1]\na.mp4 @ 30 [id=p2]\nb.mp4");
    // the right half still hands over to B at the derived 90
    assert.deepEqual(plays(text),
        [["x.mp4", 0, 60], ["a.mp4", 0, 30], ["a.mp4", 30, 90], ["b.mp4", 0, 100]]);
}

// ---- manual cuts and seam settings go to the half they belong to --------------
{
    const text = split("x.mp4\na.mp4 @ 10..100 [crossfade=on]", 60 + 40);
    // enter 10 + 40 frames in = frame 50; the join's settings stay in front
    assert.equal(text,
        "x.mp4\na.mp4 @ 10..50 [id=p1 crossfade=on]\na.mp4 @ 50..100 [id=p2]");
}

// ---- the playhead on an edge, or on empty space, does not split ---------------
load("a.mp4\n~ 20\nx.mp4");
st.frame = 0;                        // a.mp4's first frame
assert.equal(ui.playheadSplit().blocked, true);
st.frame = 130;                      // inside the gap
assert.equal(ui.playheadSplit(), null);
st.frame = 140;                      // x.mp4's first frame (right after the gap)
assert.equal(ui.playheadSplit().blocked, true);
st.frame = 141;
assert.deepEqual([ui.playheadSplit().blocked, ui.playheadSplit().at], [false, 1]);
assert.equal(split("a.mp4\n~ 20\nx.mp4", 0), "a.mp4\n~ 20\nx.mp4", "edge: no edit");

// ---- snap cuts: the split lands on the tail-cut grid, strictly inside ---------
// grid for a tail cut: raw frame = 5 mod 17 -> 5, 22, 39, 56, 73, 90, 107
assert.equal(split("a.mp4", 50, true), "a.mp4 @ ..56 [id=p1]\na.mp4 @ 56 [id=p2]");
assert.equal(split("a.mp4", 1, true), "a.mp4 @ ..5 [id=p1]\na.mp4 @ 5 [id=p2]");
// a continuation's grid is offset by its pinned head (22): raw 39 = delivered 17
assert.equal(split("c.mp4", 15, true), "c.mp4 @ ..17 [id=p1]\nc.mp4 @ 17 [id=p2]");
// a clip too short to hold a stop inside it is refused while snapping...
load("tiny.mp4");
st.frame = 4;
st.snap = true;
assert.equal(ui.playheadSplit().at, 5);
load("tiny.mp4 @ 6");                // plays 6..10: no stop at 5 inside it
st.frame = 2;
assert.equal(ui.playheadSplit().blocked, true);
// ...and splits anywhere with snapping off
st.snap = false;
assert.equal(ui.playheadSplit().at, 8);

// ---- ids: a line's own id stays on the left half ------------------------------
assert.equal(split("a.mp4 [id=k3x9]", 50),
             "a.mp4 @ ..50 [id=k3x9]\na.mp4 @ 50 [id=p1]");
// splitting one part of a split clip leaves the other parts' ids alone
assert.equal(split("a.mp4 @ ..50 [id=a1]\na.mp4 @ 50 [id=a2]", 50 + 30),
             "a.mp4 @ ..50 [id=a1]\na.mp4 @ 50..80 [id=a2]\na.mp4 @ 80 [id=p1]");

// ---- a pin on the split clip follows the edge it continues from ---------------
{
    const pinAfter = (text, frame, pin) => {
        ui.psWidget.value = JSON.stringify(pin);
        split(text, frame);
        return JSON.parse(ui.psWidget.value);
    };
    // extend leaves from the END, which is now the right half's
    assert.equal(pinAfter("a.mp4", 50, { source: "a.mp4", role: "extend" }).part,
                 "p2");
    // prepend arrives at the START, which stays the left half's
    assert.equal(pinAfter("a.mp4", 50, { source: "a.mp4", role: "prepend" }).part,
                 "p1");
    // a bridge: its extend side (source) goes right, its arriving side left
    const b = pinAfter("a.mp4", 50, { source: "a.mp4", source2: "a.mp4",
                                      role: "bridge" });
    assert.deepEqual([b.part, b.part2], ["p2", "p1"]);
    // a pin on ANOTHER part is left exactly as it was
    const other = { source: "a.mp4", part: "a1", role: "extend" };
    assert.deepEqual(pinAfter("a.mp4 @ ..50 [id=a1]\na.mp4 @ 50 [id=a2]", 80, other),
                     other);
    // no pin, nothing written
    ui.psWidget.value = "";
    split("a.mp4", 50);
    assert.equal(ui.psWidget.value, "");
}

console.log("test_split: all passed");
