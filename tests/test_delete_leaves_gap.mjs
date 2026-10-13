// A clip's ✕, LIFTED from web/h3_mctx_ui.js (not copied): the clip goes and
// its time stays as empty space, so nothing after it moves -- neighbours
// keep the frames they played, empty space beside it merges, and where a
// hole would be wrong (nothing after it, the end of a loop) it is removed.
//   node tests/test_delete_leaves_gap.mjs
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

const st = { seq: "", loop: false };
const body = [
    ...["TL_LINE_RE", "TL_GAP_RE", "TL_LOOP_RE", "TL_OPTS_RE", "TL_SEAM_KEYS",
        "TL_SEAM_ORDER", "TL_TRUE", "TL_FALSE", "TL_PART_KEY", "TL_PART_RE"]
        .map(liftConst),
    ...["tlIsEntryLine", "tlSplitSeamOpts", "tlFormatSeamOpts", "tlParseSequence",
        "tlFormatLine", "tlPins", "tlExtendParent", "tlPrependChild", "tlDeriveSeam",
        "currentLines", "entryLines", "removeEntry", "deleteClip"].map(lift),
    "const seqWidget = { get value() { return st.seq; } };",
    "function setSequence(text) { st.seq = text; }",
    "function loopOn() { return st.loop; }",
    "let lastEntries = [];",
    "function render(es) { lastEntries = es; }",
    "return { tlParseSequence, tlDeriveSeam, deleteClip, render };",
].join("\n");
const ui = new Function("st", "specHandover", body)(st, specHandover);

const META = {
    "a.mp4": { self_id: "a", delivered_frames: 120, pinned_head_frames: 0 },
    // extends A at 90
    "b.mp4": { self_id: "b", relation: "extends", parent_id: "a",
               parent_join_frame: 90, delivered_frames: 100, pinned_head_frames: 0 },
    "x.mp4": { self_id: "x", delivered_frames: 60, pinned_head_frames: 0 },
    // prepends into X, arriving at X's frame 12
    "p.mp4": { self_id: "p", relation: "prepends", parent_id: "x",
               parent_join_frame: 12, delivered_frames: 40, pinned_head_frames: 0 },
};
// resolve the way refresh() does (mirrors resolve_sequence)
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
    st.seq = text;
    ui.render(es);
    return es;
}
// [start frame, clip, enter, exit] of every clip as the cut plays it
const timeline = (text) => {
    let t = 0;
    const out = [];
    for (const e of load(text)) {
        if (e.gap != null) { t += e.gap; continue; }
        const hi = e.exit ?? e.frames;
        out.push([t, e.clip, e.enter, hi]);
        t += hi - e.enter;
    }
    return out;
};
const del = (text, i) => { load(text); ui.deleteClip(i); return st.seq; };

// ---- the clip becomes empty space of its played length ------------------------
{
    const text = "x.mp4\na.mp4 @ 10..70\nx.mp4";
    const out = del(text, 1);
    assert.equal(out, "x.mp4\n~ 60\nx.mp4");
    assert.deepEqual(timeline(out).at(-1), timeline(text).at(-1),
                     "the clip after it did not move");
}

// ---- neighbours keep what they played, even across a lineage join -------------
{
    // A exits at 90 because B extends it there: deleting B must not hand
    // A's frames 90..120 back
    const text = "a.mp4\nb.mp4\nx.mp4";
    const out = del(text, 1);
    assert.equal(out, "a.mp4 @ ..90\n~ 100\nx.mp4");
    assert.deepEqual(timeline(out)[0], ["0", "a.mp4", 0, 90].map((v, k) => k ? v : 0));
    assert.deepEqual(timeline(out).at(-1), timeline(text).at(-1));
    // P prepends into X at 12: deleting P must not hand X's frames 0..12 back
    const t2 = "a.mp4\np.mp4\nx.mp4";
    const o2 = del(t2, 1);
    assert.equal(o2, "a.mp4\n~ 40\nx.mp4 @ 12");
    assert.deepEqual(timeline(o2).at(-1), timeline(t2).at(-1));
}

// ---- empty space beside it merges into one --------------------------------------
{
    assert.equal(del("x.mp4\n~ 20\na.mp4 @ 0..30\n~ 5\nx.mp4", 2),
                 "x.mp4\n~ 55\nx.mp4");
    assert.equal(del("x.mp4\n~ 20\na.mp4 @ 0..30\nx.mp4", 2), "x.mp4\n~ 50\nx.mp4");
    // comments and seam settings on other lines are left alone
    assert.equal(del("# my cut\nx.mp4\na.mp4\nx.mp4 [crossfade=off]", 1),
                 "# my cut\nx.mp4\n~ 120\nx.mp4 [crossfade=off]");
}

// ---- the first clip leaves a hole too (the rest keeps its place) ----------------
assert.equal(del("a.mp4\nx.mp4", 0), "~ 120\nx.mp4");

// ---- where a hole would be wrong, the clip is simply removed --------------------
// nothing after it: nothing would move
assert.equal(del("x.mp4\na.mp4", 1), "x.mp4");
assert.equal(del("x.mp4\na.mp4\n~ 30", 1), "x.mp4\n~ 30", "only empty space after it");
// a looping cut cannot start (or end) on empty space
st.loop = true;
assert.equal(del("loop\na.mp4\nx.mp4\nx.mp4", 0), "loop\nx.mp4\nx.mp4");
assert.equal(del("loop\nx.mp4\na.mp4\nx.mp4", 1), "loop\nx.mp4\n~ 120\nx.mp4",
             "a loop's middle still leaves a hole");
st.loop = false;

console.log("test_delete_leaves_gap: all passed");
