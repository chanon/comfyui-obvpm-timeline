// The install-check panel, LIFTED from web/h3_mctx_ui.js (not copied), run
// against a DOM stand-in: the controls give way to the problem list and
// come back, each to its own display, when the install is fixed. Needs
// linkedom on the module path.
//   node tests/test_compat_panel.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseHTML } from "linkedom";

const SRC = readFileSync(new URL("../web/h3_mctx_ui.js", import.meta.url), "utf8");
const lift = (name, keyword = "function ") => {
    const at = SRC.indexOf(keyword + name + "(");
    assert.ok(at >= 0, name + " is gone from the source");
    let depth = 0, i = SRC.indexOf("{", at);
    for (;; i += 1) {
        if (SRC[i] === "{") depth += 1;
        if (SRC[i] === "}" && --depth === 0) break;
    }
    return SRC.slice(at, i + 1);
};

const { document } = parseHTML("<html><body><div id=host></div></body></html>");
const themePalette = () => ({ text: "#eee", sub: "#aaa", edge: "#444", rest: "#333" });
const helpers = new Function("document", "themePalette",
    lift("tlCompatPanel") + "\n" + lift("tlApplyCompat", "async function ")
    + "\nreturn { tlCompatPanel, tlApplyCompat };");
const { tlCompatPanel, tlApplyCompat } = helpers(document, themePalette);

const PROBLEMS = [
    { key: "obvpm", title: "comfyui-obvpm needs updating",
      detail: "comfyui-obvpm 0.2.2 is installed; the workflow needs 0.2.3 or newer.",
      fix: "Update it in ComfyUI Manager, then restart ComfyUI.",
      links: [{ label: "comfyui-obvpm releases", url: "https://github.com/chanon/comfyui-obvpm/releases" },
              { label: "not a link", url: "javascript:alert(1)" }] },
    { key: "upscaler", title: "The wrong latent upscaler is installed",
      detail: "<b>data</b>", fix: "Install the original.", links: [] },
];

let passed = 0;
function test(name, fn) {
    return Promise.resolve().then(fn).then(() => { passed += 1; console.log("ok  " + name); });
}

const make = () => {
    const container = document.createElement("div");
    const video = document.createElement("div"); video.style.display = "flex";
    const progress = document.createElement("div"); progress.style.display = "none";
    const strip = document.createElement("div");
    container.append(video, progress, strip);
    document.getElementById("host").replaceChildren(container);
    return { container, children: [video, progress, strip], video, progress, strip };
};

await test("the panel is text, numbered, with fixes and only http(s) links", async () => {
    const panel = tlCompatPanel(PROBLEMS, null);
    const text = panel.textContent;
    assert.match(text, /cannot run on this install yet/);
    assert.match(text, /2 things to fix/);
    assert.match(text, /1\. comfyui-obvpm needs updating/);
    assert.match(text, /2\. The wrong latent upscaler/);
    assert.match(text, /Fix: Update it in ComfyUI Manager/);
    assert.ok(text.includes("<b>data</b>"), "detail is text, not markup");
    assert.equal(panel.querySelectorAll("b").length, 2, "only the two Fix labels are bold");
    const anchors = [...panel.querySelectorAll("a")];
    assert.equal(anchors.length, 1, "the javascript: link is dropped");
    assert.equal(anchors[0].getAttribute("href"), "https://github.com/chanon/comfyui-obvpm/releases");
    assert.equal(anchors[0].getAttribute("target"), "_blank");
    assert.equal(anchors[0].getAttribute("rel"), "noopener noreferrer");
    assert.equal(panel.querySelectorAll("button").length, 0, "no recheck without a handler");
});

await test("problems hide the controls and stand in their place", async () => {
    const t = make();
    const got = await tlApplyCompat(t.container, t.children, async () => PROBLEMS);
    assert.equal(got.length, 2);
    for (const c of t.children) assert.equal(c.style.display, "none");
    assert.equal(t.container.lastElementChild, t.container.__tlCompatPanel);
    assert.match(t.container.__tlCompatPanel.textContent, /Check again/);
});

await test("a clean install puts every control back to its own display", async () => {
    const t = make();
    await tlApplyCompat(t.container, t.children, async () => PROBLEMS);
    const got = await tlApplyCompat(t.container, t.children, async () => []);
    assert.deepEqual(got, []);
    assert.equal(t.video.style.display, "flex");
    assert.equal(t.progress.style.display, "none", "the progress bar stays hidden");
    assert.equal(t.strip.style.display, "");
    assert.equal(t.container.__tlCompatPanel, null);
    assert.equal(t.container.querySelectorAll("a").length, 0, "the panel is gone");
});

await test("check again asks the server again and replaces the panel, once", async () => {
    const t = make();
    let asks = 0;
    const fetchProblems = async () => { asks += 1; return asks < 2 ? PROBLEMS : [PROBLEMS[1]]; };
    await tlApplyCompat(t.container, t.children, fetchProblems);
    t.container.__tlCompatPanel.querySelector("button").click();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(asks, 2);
    const panels = [...t.container.children].filter((c) => c === t.container.__tlCompatPanel);
    assert.equal(panels.length, 1);
    assert.match(t.container.__tlCompatPanel.textContent, /One thing to fix/);
    assert.doesNotMatch(t.container.__tlCompatPanel.textContent, /needs updating/);
});

await test("a server that cannot be asked leaves the timeline alone", async () => {
    const t = make();
    const got = await tlApplyCompat(t.container, t.children, async () => { throw new Error("offline"); });
    assert.equal(got, null);
    assert.equal(t.video.style.display, "flex");
    assert.equal(t.container.querySelectorAll("a").length, 0);
    const odd = await tlApplyCompat(t.container, t.children, async () => ({ error: "x" }));
    assert.equal(odd, null);
});

console.log(passed + " passed");
