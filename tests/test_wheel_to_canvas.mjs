// The wheel-to-canvas helper, LIFTED from web/h3_mctx_ui.js (not copied), run
// against a DOM stand-in. Needs linkedom on the module path.
//   node tests/test_wheel_to_canvas.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseHTML } from "linkedom";

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

const { document, HTMLElement, Event } = parseHTML(`<html><body>
  <div id="host"><div id="panel">
    <div id="video"></div>
    <div id="strip" style="overflow-x:auto"><div id="block"></div></div>
    <div id="ruler"></div>
    <textarea id="text" style="overflow-y:auto"></textarea>
  </div></div><canvas id="graph"></canvas></body></html>`);
const canvas = document.getElementById("graph");
const app = { canvas: { canvas } };
// layout the stand-in cannot compute: scroll geometry per element
const geo = new Map();
const scrollable = (id, g) => geo.set(document.getElementById(id), g);
const getComputedStyle = (el) => ({ overflowX: el.style.overflowX || "visible", overflowY: el.style.overflowY || "visible" });
class WheelEvent extends Event {
    constructor(type, init = {}) { super(type, { bubbles: true, cancelable: true, ...init }); Object.assign(this, { deltaX: 0, deltaY: 0, deltaMode: 0, ...init }); }
}
const helpers = new Function("app", "HTMLElement", "getComputedStyle", "WheelEvent",
    lift("tlScrollsThatWay") + "\n" + lift("tlWheelToCanvas") + "\nreturn { tlWheelToCanvas };");
const { tlWheelToCanvas } = helpers(app, HTMLElement, getComputedStyle, WheelEvent);
for (const el of [...document.querySelectorAll("#panel *")]) {
    for (const k of ["scrollLeft", "scrollTop", "scrollWidth", "clientWidth", "scrollHeight", "clientHeight"]) {
        Object.defineProperty(el, k, { get: () => (geo.get(el) || {})[k] ?? 0, configurable: true });
    }
}

const got = [];
canvas.addEventListener("wheel", (e) => got.push({ dy: e.deltaY, dx: e.deltaX, ctrl: !!e.ctrlKey, x: e.clientX }));
const panel = document.getElementById("panel");
// the panel's OWN consumers, registered like the real ones
document.getElementById("ruler").addEventListener("wheel", (e) => { e.preventDefault(); e.stopPropagation(); });
document.getElementById("strip").addEventListener("wheel", (e) => { if (!e.ctrlKey) return; e.preventDefault(); e.stopPropagation(); });
tlWheelToCanvas(panel);
const wheel = (id, init) => {
    const e = new WheelEvent("wheel", { clientX: 5, clientY: 6, ...init });
    document.getElementById(id).dispatchEvent(e);
    return e;
};

// ---- a bare wheel anywhere on the panel zooms the graph ---------------------
let e = wheel("video", { deltaY: 100 });
assert.deepEqual(got.pop(), { dy: 100, dx: 0, ctrl: false, x: 5 });
assert.equal(e.defaultPrevented, true, "the page must not scroll instead");
wheel("block", { deltaY: -100 });                       // over the strip's clips
assert.equal(got.pop().dy, -100);

// ---- the panel's own uses keep the wheel ------------------------------------
wheel("ruler", { deltaY: 100 });
wheel("block", { deltaY: 100, ctrlKey: true });        // ctrl+wheel = strip zoom
assert.equal(got.length, 0);

// ---- ctrl+wheel where the panel has no use for it = the graph's zoom gesture
wheel("video", { deltaY: 100, ctrlKey: true });
assert.equal(got.pop().ctrl, true);

// ---- a strip that can scroll SIDEWAYS keeps a sideways wheel, until it ends --
scrollable("strip", { scrollLeft: 0, scrollWidth: 900, clientWidth: 300 });
e = wheel("block", { deltaX: 80 });
assert.equal(got.length, 0);
assert.equal(e.defaultPrevented, false, "native scrolling must be allowed");
wheel("block", { deltaY: 100, shiftKey: true });        // shift+wheel is sideways too
assert.equal(got.length, 0);
wheel("block", { deltaX: -80 });                        // already at the left end
assert.equal(got.pop().dx, -80);
wheel("block", { deltaY: 100 });                        // vertical: nothing to scroll
assert.equal(got.pop().dy, 100);

// ---- a long text box scrolls itself; a short one does not hold the wheel ----
scrollable("text", { scrollTop: 10, scrollHeight: 400, clientHeight: 100 });
wheel("text", { deltaY: 100 });
assert.equal(got.length, 0);
scrollable("text", { scrollTop: 0, scrollHeight: 100, clientHeight: 100 });
wheel("text", { deltaY: 100 });
assert.equal(got.pop().dy, 100);
console.log("wheel to canvas: all checks pass");
