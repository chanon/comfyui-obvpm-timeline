import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { specHandover } from "./h3_handover.js";
// drop-resolution helpers shared with the loader drop handler -- same
// trust rules for drops into the timeline strip (in-place when the
// content already lives in output, upload to dropped/ otherwise)
import {
    ARTIUS_MIME, ARTIUS_ROUTE_BASE,
    SIDECAR_SUFFIX as DROP_SIDECAR_SUFFIX,
    SUBFOLDER as DROP_SUBFOLDER, VIDEO_RE as DROP_VIDEO_RE,
    artiusRelativePath, baseName as dropBaseName, isArtiusVideo,
    readArtiusAssets, serverFileSize, uploadToOutput,
} from "./h3_mctx_drop.js";

// Video preview for the H3 MCtx loader nodes, mirroring core's own
// useNodeVideo pattern (src/composables/node/useNodeImage.ts) exactly:
// a comfy-img-preview container div as the DOM widget, a growable
// computeLayoutSize override with min dimensions refreshed on load, and
// canvas-gesture forwarding so wheel/middle-drag over the video still
// pan/zoom the graph. The node auto-grows via the layout; no manual
// sizing anywhere.

const NODES = ["H3LoadVideoWithMCtx", "H3LoadMCtx"];
const DEFAULT_SIZE = 256;

/**
 * Remove the input sockets belonging to widgets we have hidden.
 *
 * Every widget also gets an input socket (litegraphService adds one unless
 * the widget is `socketless`, which an extension cannot set -- it is read
 * while the node is being built). Hiding the widget hides the row but
 * leaves that socket: invisible, first in the list, and still hit-tested,
 * so it sits over the top of the node's real first pin and takes wires
 * aimed at it. The widget's value travels in widgets_values, not through
 * the socket, so dropping it costs nothing.
 *
 * Skipped if something is genuinely connected, and the remaining links are
 * re-pointed because they address their target by slot INDEX.
 */
export function dropWidgetSockets(node, names) {
    for (const name of names) {
        const at = (node.inputs ?? []).findIndex(
            (slot) => slot.widget && (slot.widget.name === name
                                      || slot.name === name));
        if (at < 0 || node.inputs[at].link != null) continue;
        node.removeInput(at);
    }
    // graph.links is an object at the root and a Map inside a subgraph,
    // so it cannot simply be bracket-indexed.
    const links = node.graph?.links;
    (node.inputs ?? []).forEach((slot, index) => {
        if (!links || slot.link == null) return;
        const link = typeof links.get === "function"
            ? links.get(slot.link) : links[slot.link];
        if (link) link.target_slot = index;
    });
}
const MIN_HEIGHT = 64;

// ---- pre-release surface -------------------------------------------------
// Guided mode and the seam repairs that exist FOR it are built, tested and
// still wired end to end -- they are withheld from the FACE of the UI, not
// taken out. Masked continuation is the mode now: its joins are
// latent-identical, so a repair applied to one corrects noise (measured
// live to INTRODUCE flicker). Offering the controls invites exactly the
// wrong move, so the streamlined surface is one mode and no seam dialogs.
//
// Withheld, not removed: anything that ALREADY says guided -- an old
// workflow's pin, a join carrying overrides of its own -- keeps its
// controls, so nothing becomes unreachable and no setting is silently
// dropped. Flip this to true to bring the whole surface back.
const SHOW_GUIDED = false;

// ---- diagnostics (set DEBUG true when hunting sizing bugs) ---------------
const DEBUG = false;
function dbg(...args) {
    if (DEBUG) console.log("[obvpm-h3-preview]", ...args);
}
dbg("extension loaded; vueNodesMode =",
    typeof LiteGraph !== "undefined" && !!LiteGraph.vueNodesMode);

// Samples the full sizing chain twice a second and logs ONLY when values
// change, tagging which link moved. Reproduce the freeze, then paste the
// console lines: the last link that stops changing is the broken one.
function attachSampler(node, widget, el) {
    if (!DEBUG) return;
    let prev = "";
    const sample = () => {
        if (!node.graph) return; // node removed; stop
        const r = el.getBoundingClientRect();
        const p = el.parentElement;
        const pr = p ? p.getBoundingClientRect() : null;
        const snap = {
            nodeSize: [Math.round(node.size[0]), Math.round(node.size[1])],
            widgetY: Math.round(widget.y ?? -1),
            computedHeight: Math.round(widget.computedHeight ?? -1),
            widgetWidth: widget.width ?? null,
            layout: widget.computeLayoutSize
                ? widget.computeLayoutSize(node) : null,
            videoRect: [Math.round(r.width), Math.round(r.height)],
            containerRect: pr ? [Math.round(pr.width), Math.round(pr.height)] : null,
            containerStyle: p ? { w: p.style.width, h: p.style.height } : null,
        };
        const s = JSON.stringify(snap);
        if (s !== prev) {
            prev = s;
            dbg("node", node.id, s);
        }
        setTimeout(sample, 500);
    };
    setTimeout(sample, 500);
}
// --------------------------------------------------------------------------

function viewRoute(value) {
    if (!value) return null;
    let filename = String(value);
    let subfolder = "";
    const slash = filename.lastIndexOf("/");
    if (slash >= 0) {
        subfolder = filename.slice(0, slash);
        filename = filename.slice(slash + 1);
    }
    return `/view?filename=${encodeURIComponent(filename)}` +
        `&subfolder=${encodeURIComponent(subfolder)}&type=output`;
}

function viewURL(clipValue) {
    const route = viewRoute(clipValue);
    return route ? api.apiURL(route) : null;
}

// ---- mctx sidecar badge --------------------------------------------------

const SIDECAR_SUFFIX = ".mctx.safetensors";
const MCTX_FORMAT = "mctx_v1";

function sidecarValue(clipValue) {
    // mctx.sidecar_path convention: video extension REPLACED by the suffix
    return String(clipValue).replace(/\.[^.]+$/, "") + SIDECAR_SUFFIX;
}

// Range-read the sidecar's safetensors header, mirroring mctx.read_header:
// 8 bytes little-endian header length, then that much JSON whose
// "__metadata__" holds the string metadata. Never touches the tensors.
// Returns the metadata dict, null when the file is absent, and throws
// when a file is there but is not a plausible mctx_v1 sidecar.
async function readSidecarMeta(value) {
    const route = viewRoute(value);
    // no-store: a take can be DELETED and regenerated under the same
    // name, and the /view URL carries no version of its own. A cached
    // range response then describes a file that no longer exists, and
    // the clip silently loses its lineage.
    const r1 = await api.fetchApi(route,
        { cache: "no-store", headers: { Range: "bytes=0-7" } });
    if (r1.status === 404) return null;
    // FileResponse honors Range; anything else risks pulling whole tensors
    if (r1.status !== 206) throw new Error(`no ranged read (${r1.status})`);
    const prefix = await r1.arrayBuffer();
    if (prefix.byteLength < 8) throw new Error("too short for safetensors");
    const length = Number(new DataView(prefix).getBigUint64(0, true));
    if (length <= 0 || length > 10 * 1024 * 1024) {
        throw new Error(`implausible header length ${length}`);
    }
    const r2 = await api.fetchApi(route,
        { cache: "no-store",
          headers: { Range: `bytes=8-${8 + length - 1}` } });
    if (r2.status !== 206) throw new Error(`no ranged read (${r2.status})`);
    const meta = JSON.parse(await r2.text())?.__metadata__ ?? {};
    if (meta.format !== MCTX_FORMAT) {
        throw new Error(`format=${meta.format ?? "?"}`);
    }
    return meta;
}

// Read one BLOB tensor (workflow / prompt) out of a sidecar, without
// pulling the latents. safetensors puts a per-tensor {data_offsets:
// [start,end]} in the header, relative to the data area that begins at
// 8 + headerLength -- so the exact bytes can be asked for by range, the
// same trick readSidecarMeta uses for the header itself.
// Returns null when the file or the blob is absent.
async function readSidecarBlob(value, name) {
    const route = viewRoute(value);
    if (!route) return null;
    const r1 = await api.fetchApi(route, { headers: { Range: "bytes=0-7" } });
    if (r1.status === 404) return null;
    if (r1.status !== 206) throw new Error(`no ranged read (${r1.status})`);
    const prefix = await r1.arrayBuffer();
    if (prefix.byteLength < 8) throw new Error("too short for safetensors");
    const headerLen = Number(new DataView(prefix).getBigUint64(0, true));
    if (headerLen <= 0 || headerLen > 10 * 1024 * 1024) {
        throw new Error(`implausible header length ${headerLen}`);
    }
    const r2 = await api.fetchApi(route,
        { headers: { Range: `bytes=8-${8 + headerLen - 1}` } });
    if (r2.status !== 206) throw new Error(`no ranged read (${r2.status})`);
    const header = JSON.parse(await r2.text());
    const entry = header?.[name];
    const [start, end] = entry?.data_offsets ?? [];
    if (start == null || end == null || end <= start) return null;
    const base = 8 + headerLen;
    const r3 = await api.fetchApi(route,
        { headers: { Range: `bytes=${base + start}-${base + end - 1}` } });
    if (r3.status !== 206) throw new Error(`no ranged read (${r3.status})`);
    // the blob is UTF-8 JSON stored as uint8 (see h3/mctx.py)
    return JSON.parse(new TextDecoder().decode(await r3.arrayBuffer()));
}

const short = (id) => (id ? String(id).slice(0, 12) : "");

function mctxTooltip(meta) {
    const lines = [`mctx sidecar found (${meta.format})`,
        `id ${short(meta.self_id)}`];
    if (meta.parent_id) {
        lines.push(`${meta.relation || "related to"} ${short(meta.parent_id)}`
            + (meta.parent_join_frame !== undefined && meta.parent_join_frame !== ""
                ? ` @ frame ${meta.parent_join_frame}` : ""));
    } else {
        lines.push("root clip (no parent)");
    }
    if (meta.width) {
        lines.push(`${meta.width}x${meta.height} @ ${meta.fps}fps`);
    }
    if (meta.delivered_frames) {
        lines.push(`${meta.delivered_frames} frames delivered`
            + ` (${meta.raw_frames} raw, pinned ${meta.pinned_head_frames}`
            + `+${meta.pinned_tail_frames})`);
    }
    lines.push("pairing hash is verified at load time");
    return lines.join("\n");
}

app.registerExtension({
    name: "obvpm.h3_mctx_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const clipWidget = node.widgets?.find((w) => w.name === "clip");
            if (!clipWidget) return result;

            // Container div is the widget element; core's
            // .comfy-img-preview CSS sizes media children to fit.
            const container = document.createElement("div");
            container.classList.add("comfy-img-preview");

            const el = document.createElement("video");
            el.playsInline = true;
            el.controls = true;
            el.loop = true;
            el.muted = true; // autoplay policy; unmute via controls
            el.autoplay = true;

            // Sidecar badge: does the selected clip have its mctx? The
            // wrapper Vue component positions the widget element but does
            // not set its inline position, so anchoring the container is
            // safe -- the badge rides the video's top-left corner.
            const badge = document.createElement("div");
            Object.assign(badge.style, {
                position: "absolute", top: "4px", left: "4px",
                padding: "1px 7px", borderRadius: "9px",
                font: "10px sans-serif", lineHeight: "16px",
                background: "rgba(0,0,0,0.65)", color: "#fff",
                zIndex: "1", whiteSpace: "pre",
            });
            badge.textContent = "";
            if (!container.style.position) {
                container.style.position = "relative";
            }
            container.replaceChildren(el, badge);

            let badgeSeq = 0; // async probe guard (gotchas section 9)
            async function updateBadge(value) {
                const seq = ++badgeSeq;
                // bright white text throughout; state lives in the bg color
                const set = (text, bg, tip) => {
                    if (seq !== badgeSeq) return;
                    badge.textContent = text;
                    badge.style.background = bg;
                    badge.title = tip;
                };
                if (!value) return set("", "rgba(0,0,0,0.65)", "");
                try {
                    const meta = await readSidecarMeta(sidecarValue(value));
                    if (meta) {
                        set("mctx ✓", "rgba(30,110,50,0.85)",
                            mctxTooltip(meta));
                    } else {
                        set("no mctx", "rgba(170,40,40,0.85)",
                            "No .mctx.safetensors next to this clip.\n" +
                            "Loads without latents (MCTX = None); extends " +
                            "must go through the pixel route.");
                    }
                } catch (err) {
                    dbg("sidecar probe failed for", value, err);
                    set("mctx ?", "rgba(190,120,30,0.85)",
                        "A sidecar file exists but is not a readable " +
                        `mctx_v1 sidecar (${err?.message ?? err}).`);
                }
            }

            // Growable widget with aspect-derived minimums, exactly like
            // core's video preview: refreshed when a video loads, consumed
            // by the layout on every arrange pass.
            let minWidth = DEFAULT_SIZE;
            let minHeight = DEFAULT_SIZE;
            el.addEventListener("loadeddata", () => {
                if (el.videoWidth > 0) {
                    minWidth = node.size?.[0] || DEFAULT_SIZE;
                    minHeight = Math.max(
                        minWidth * (el.videoHeight / el.videoWidth),
                        MIN_HEIGHT);
                }
                dbg("loadeddata: video", el.videoWidth, "x", el.videoHeight,
                    "-> min", Math.round(minWidth), "x", Math.round(minHeight),
                    "nodeSize", [...node.size]);
                node.graph?.setDirtyCanvas(true);
            });
            el.addEventListener("error", () =>
                dbg("video error for", clipWidget.value, el.error));

            const widget = node.addDOMWidget("mctx_preview", "video",
                container, { hideOnZoom: false });
            widget.serialize = false;
            widget.options.serialize = false;
            tlPanWithMiddleButton(container);
            widget.computeLayoutSize = () => ({ minHeight, minWidth });
            // THE frozen-width culprit (diagnosed via the sampler): once
            // something stores widget.width (the Vue legacy-widget mirror
            // does: WidgetLegacy.vue `widgetInstance.width = width`), the
            // element-size formula in DomWidgets.vue --
            //   width = (widget.width ?? node.width) - margin*2
            // -- prefers the stored value FOREVER, freezing the element's
            // width while height keeps tracking. Shield it: reads yield
            // undefined so the live node.width fallback always wins.
            Object.defineProperty(widget, "width", {
                configurable: true,
                get: () => undefined,
                set: () => {},
            });

            // Forward canvas gestures so the video doesn't swallow graph
            // navigation (core does the same via useCanvasInteractions).
            container.addEventListener("wheel", (e) => {
                const canvasEl = app.canvas?.canvas;
                if (!canvasEl) return;
                e.preventDefault();
                e.stopPropagation();
                const { clientX, clientY, deltaX, deltaY,
                        ctrlKey, metaKey, shiftKey } = e;
                canvasEl.dispatchEvent(new WheelEvent("wheel", {
                    clientX, clientY, deltaX, deltaY,
                    ctrlKey, metaKey, shiftKey,
                }));
            });
            const forwardMiddle = (e) => {
                if (e.button === 1 || (e.buttons & 4)) {
                    const canvasEl = app.canvas?.canvas;
                    if (!canvasEl) return;
                    e.preventDefault();
                    e.stopPropagation();
                    canvasEl.dispatchEvent(new PointerEvent(e.type, e));
                }
            };
            container.addEventListener("pointerdown", forwardMiddle);
            container.addEventListener("pointermove", forwardMiddle);

            // The container is a fixed-position DOM overlay ABOVE the
            // canvas: file drags over it never reach the canvas element,
            // so app.dragOverNode is never set and the node's classic
            // onDragDrop path is dead wherever the preview covers the
            // node (gotchas 12). Accept the drop here and route it to
            // the same node API the canvas path uses.
            // Accept both OS file drags and Artius-browser card drags
            // (custom MIME, no File objects). Routing Artius drops here is
            // a bonus: the overlay sits above the canvas, so Artius's own
            // capture-phase canvas bridge (which spawns a LoadVideo node)
            // never sees them either.
            const ARTIUS_MIME = "application/x-timesaver-artius-asset";
            const dropTargetsUs = (dt) =>
                Array.from(dt?.types ?? []).includes(ARTIUS_MIME) ||
                Array.from(dt?.items ?? []).some((i) => i.kind === "file");
            container.addEventListener("dragover", (e) => {
                if (dropTargetsUs(e.dataTransfer)) {
                    e.preventDefault(); // permits dropping here
                    e.stopPropagation();
                }
            });
            container.addEventListener("drop", (e) => {
                if (!dropTargetsUs(e.dataTransfer)) return;
                e.preventDefault();  // document drop handler checks this
                e.stopPropagation();
                node.onDragDrop?.(e);
            });
            if (DEBUG) {
                el.addEventListener("click", () =>
                    dbg("video clicked; node", node.id, "size", [...node.size]));
                attachSampler(node, widget, el);
            }
            dbg("widget attached to node", node.id, nodeData.name)

            let current = null;
            function updateSrc() {
                const value = clipWidget.value;
                if (value === current) return;
                current = value;
                const url = viewURL(value);
                if (url) {
                    el.src = url;
                } else {
                    el.removeAttribute("src");
                    el.load();
                }
                void updateBadge(value);
            }

            // user changes: wrap-and-chain the combo callback...
            const prevCallback = clipWidget.callback;
            clipWidget.callback = function (...args) {
                const r = prevCallback?.apply(this, args);
                updateSrc();
                return r;
            };
            // ...workflow load: widgets_values arrive with NO callbacks.
            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                updateSrc();
                return r;
            };

            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                el.pause();
                el.removeAttribute("src");
                el.load();
                return onRemoved?.apply(this, arguments);
            };

            updateSrc();
            return result;
        };
    },
});

// ======================= H3 MCtx Assemble: timeline =======================
// Lives in this file (which demonstrably loads -- the badge proves it)
// after a standalone h3_mctx_timeline.js was served by /extensions yet
// never executed in the user's browser; cause unresolved. It also reuses
// viewRoute/sidecarValue/readSidecarMeta directly instead of duplicating
// them. The multiline `sequence` widget stays the source of truth; this
// widget renders it as a timeline (blocks sized by played frames, seam
// chips derived like the Python _derive_seam) and previews the cut by
// chaining clips in one <video> honoring the derived enter/exit points.

// The save nodes' `metadata` widget is hidden: the prompt, seed and the
// rest are captured automatically now (sidecar blobs + MP4 container
// tags), so hand-typed provenance is redundant. Kept DECLARED so saved
// nodes' positional widgets_values do not shift.
const H3_SAVE_NODES = ["H3SaveVideoWithMCtx", "H3TrimAndSaveVideoWithMCtx",
                       "H3SaveMCtxForVideo"];

app.registerExtension({
    name: "obvpm.h3_save_ui",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!H3_SAVE_NODES.includes(nodeData.name)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            try {
                const w = this.widgets?.find((x) => x.name === "metadata");
                if (w) {
                    // both flags: canvas mode reads widget.hidden, Vue
                    // reads widget.options.hidden
                    w.hidden = true;
                    (w.options ??= {}).hidden = true;
                    const wEl = w.element ?? w.inputEl;
                    if (wEl) wEl.style.display = "none";
                }
                dropWidgetSockets(this, ["metadata"]);
            } catch (err) {
                console.error("[obvpm-h3-save] hiding metadata failed:", err);
            }
            return r;
        };
    },
});

// ============ restoring settings from a clip's stored workflow =========
// A take's sidecar carries the workflow that made it (h3/mctx.py blobs).
// "Restore" replays part of it into the CURRENT graph: widget values and
// bypass/mute state, for top-level nodes inside groups whose title
// matches the keyword. Deliberately narrow -- nothing is created, moved,
// rewired, or deleted, so the worst case is undoable by hand.
const TL_NODE = "H3Timeline";
const RP_NODE_TYPE = "H3ResultPreview";
// LGraphNode modes, for saying what a bypass change actually is
const TL_MODE = { 0: "active", 2: "muted", 4: "bypassed" };
// Group membership is NOT stored in a workflow. litegraph recomputes it
// geometrically (LGraphGroup.recomputeInsideNodes): a node belongs to a
// group when its bounding rect's CENTRE is inside the group. The saved
// pos/size are enough to redo that here.
const TL_TITLE_H = 30;   // LiteGraph.NODE_TITLE_HEIGHT

function tlNodeCentre(n) {
    const [x, y] = n.pos ?? [0, 0];
    const [w, h] = n.size ?? [0, 0];
    if (n.flags?.collapsed) return [x + 40, y - TL_TITLE_H / 2];
    // boundingRect spans the title bar above pos, so the centre sits
    // half a title height higher than the body's centre
    return [x + w / 2, y + h / 2 - TL_TITLE_H / 2];
}

function tlInside(bounding, centre) {
    const [bx, by, bw, bh] = bounding ?? [];
    if (bx == null) return false;
    return centre[0] >= bx && centre[0] <= bx + bw
        && centre[1] >= by && centre[1] <= by + bh;
}

// "engine, sampler" -> matches a group titled "H3 Engine Options"
function tlGroupMatcher(keyword) {
    const terms = String(keyword ?? "").split(",")
        .map((s) => s.trim().toLowerCase()).filter(Boolean);
    if (!terms.length) return null;          // empty means restore nothing
    return (title) => {
        const t = String(title ?? "").toLowerCase();
        return terms.some((term) => t.includes(term));
    };
}

// Saved widget values by NAME. Newer frontends serialize
// widgets_values_named alongside the positional array; falling back to
// positions is only safe when the widget count still agrees, since a
// node that gained an input since the take was saved would smear every
// value one slot over (exactly what base_folder did to saved workflows).
function tlSavedValues(saved, liveNode) {
    const widgets = liveNode.widgets ?? [];
    // A node with NO widgets has nothing to restore, which is not the
    // same as a node whose layout drifted. Subgraph instances are
    // usually like this -- they serialize without `widgets_values` at
    // all -- and reporting them as "its widgets changed" was wrong.
    // Returning an empty map still lets the bypass/mute state through.
    if (!widgets.length) return { values: {}, by: "none" };
    const named = saved.widgets_values_named;
    if (named && typeof named === "object" && !Array.isArray(named)) {
        return { values: named, by: "name" };
    }
    const arr = saved.widgets_values;
    if (!Array.isArray(arr) || arr.length !== widgets.length) return null;
    const values = {};
    widgets.forEach((w, i) => { values[w.name] = arr[i]; });
    return { values, by: "position" };
}

// `apply` false makes this a DRY RUN: it reports exactly what would
// change without touching anything. The confirm dialog is built from a
// dry run and the real pass repeats the same computation, so the dialog
// can never describe something other than what happens.
function tlApplyValues(liveNode, saved, report, apply, only) {
    const got = tlSavedValues(saved, liveNode);
    if (!got) {
        report.skipped.push(`${tlNodeName(liveNode)} — its widgets `
            + `changed since the take was saved`);
        return false;
    }
    let touched = false;
    for (const w of liveNode.widgets ?? []) {
        if (!(w.name in got.values)) continue;
        const v = got.values[w.name];
        if (v === undefined) continue;
        // A wired widget's value is dead -- the wire supplies it at run
        // time, so writing here would only mislead.
        const wired = (liveNode.inputs ?? []).some(
            (s) => (s.widget?.name === w.name || s.name === w.name)
                   && s.link != null);
        if (wired) continue;
        // A combo whose option list has moved on must not be forced to a
        // value it no longer offers.
        const opts = w.options?.values;
        if (Array.isArray(opts) && opts.length && !opts.includes(v)) {
            report.skipped.push(`${tlNodeName(liveNode)} — `
                + `${tlWidgetLabel(liveNode, w)}`
                + `: "${v}" is no longer an option`);
            continue;
        }
        if (tlSameValue(w.value, v)) continue;
        // `only` is the dialog's selection. A row the user unticked is
        // not a change at all: not applied, not counted, not listed.
        const key = `${liveNode.id}::${w.name}`;
        if (only && !only.has(key)) continue;
        touched = true;
        // recorded on every pass, so the dialog lists exactly the
        // changes the apply will make
        report.changes.push({ id: liveNode.id, type: liveNode.type,
                              name: tlNodeName(liveNode), key: key,
                              what: tlWidgetLabel(liveNode, w),
                              from: w.value, to: v });
        if (!apply) continue;
        w.value = v;
        try {
            w.callback?.(v, app.canvas, liveNode);
        } catch (err) {
            tldbg("restore: widget callback threw", liveNode.type, w.name, err);
        }
    }
    // bypass / mute / always -- explicitly requested, and it is just a
    // number on the node (0 always, 2 never/mute, 4 bypass)
    const modeKey = `${liveNode.id}::__mode__`;
    if (typeof saved.mode === "number" && saved.mode !== liveNode.mode
            && (!only || only.has(modeKey))) {
        report.changes.push({ id: liveNode.id, type: liveNode.type,
                              name: tlNodeName(liveNode), key: modeKey,
                              what: "node state",
                              from: TL_MODE[liveNode.mode] ?? liveNode.mode,
                              to: TL_MODE[saved.mode] ?? saved.mode });
        if (apply) liveNode.mode = saved.mode;
        touched = true;
    }
    return touched;
}

// A group's rect, live object or serialized -- Rectangle is indexable.
function tlGroupBounds(g) {
    return g?.bounding ?? g?._bounding ?? null;
}

// A live node's centre. litegraph already maintains boundingRect (which
// includes the title bar); only fall back to the saved-shape geometry
// when it is absent.
function tlLiveCentre(node) {
    const r = node?.boundingRect;
    if (r && r.length >= 4) return [r[0] + r[2] / 2, r[1] + r[3] / 2];
    return tlNodeCentre(node);
}

// The whole restore, as data: which live nodes would be touched and why
// not, so the caller can report instead of silently doing nothing.
//
// Groups and membership come from the CURRENT graph, never the saved
// one: the keyword is typed against the group titles on screen, and
// "the nodes in this group" means the ones in it now. The saved
// workflow is only a value SOURCE, looked up per node id. A group
// renamed, moved, or grown since the take was saved therefore behaves
// the way it looks, and a node dragged into the group since is included.
function tlPlanRestore(wf, rootGraph, keyword, selfId, apply, skipWords,
                       only) {
    const report = { groups: [], applied: [], skipped: [],
                     missing: [], changes: [] };
    const match = tlGroupMatcher(keyword);
    if (!match) return { ...report, reason: "no keyword" };
    // opt-out by node title: a node tagged [skip] keeps what it has,
    // even sitting inside a matching group
    const skip = tlGroupMatcher(skipWords);

    const groups = (rootGraph?.groups ?? rootGraph?._groups ?? [])
        .filter((g) => match(g.title));
    report.groups = groups.map((g) => g.title);
    if (!groups.length) {
        return { ...report, reason: "no group title matched" };
    }
    const saved = new Map(
        (wf?.nodes ?? []).map((n) => [String(n.id), n]));

    // Top level only: rootGraph's own node list, so a subgraph's
    // interior is never reached even if its instance sits in the group.
    for (const node of rootGraph?.nodes ?? rootGraph?._nodes ?? []) {
        const centre = tlLiveCentre(node);
        if (!groups.some((g) => tlInside(tlGroupBounds(g), centre))) continue;
        // Result previews carry a run's payload and its scope; replaying
        // an old one would make a preview claim a clip it never made.
        if (node.type === RP_NODE_TYPE) continue;
        // The timeline restores only its own pin options, below.
        if (node.type === TL_NODE) continue;
        if (String(node.id) === String(selfId)) continue;
        if (skip && skip(tlNodeName(node))) {
            report.skipped.push(`${tlNodeName(node)} — skipped by name`);
            continue;
        }
        const from = saved.get(String(node.id));
        if (!from) {
            report.missing.push(`${tlNodeName(node)} — not in the take`);
            continue;
        }
        if (from.type !== node.type) {
            report.skipped.push(`${tlNodeName(node)} — is a `
                + `${node.type} now, was a ${from.type}`);
            continue;
        }
        if (tlApplyValues(node, from, report, apply, only)) {
            report.applied.push(tlNodeName(node));
        }
    }
    return report;
}

// ---- the load-settings dialog ----------------------------------------
// Own modal rather than confirm(): the decision is "do I want THESE
// values on THESE nodes", which a one-line string cannot carry. Resolves
// true only on the Load button.
// Full text, never truncated -- long values (a prompt) get a scrollable
// cell instead, so nothing is hidden from the decision.
function tlValueText(v) {
    if (v === undefined) return "(unset)";
    if (v === null) return "null";
    if (typeof v === "string") return v === "" ? '""' : v;
    try {
        return JSON.stringify(v);
    } catch (err) {
        return String(v);
    }
}

// What the user calls this node: exactly the title on the canvas.
function tlNodeName(node) {
    return node?.title || node?.type || "node";
}

// What the node CALLS this setting. Widget `name` is the wire-level key
// and is often meaningless to a reader: a promoted subgraph widget is
// `value`, `value_1`, `value_2`. The readable name lives either on the
// widget itself or -- for promoted subgraph widgets -- on the matching
// INPUT SLOT's label, which is where the rename is actually stored
// (`value` -> "prompt", `value_4` -> "turbo_strength").
function tlWidgetLabel(node, w) {
    if (w?.label) return w.label;
    if (w?.options?.label) return w.options.label;
    const slot = (node?.inputs ?? []).find(
        (s) => s.widget && (s.widget.name === w?.name || s.name === w?.name));
    // slot.name is the same wire-level key, so only a real label counts
    return slot?.label || slot?.localized_name || w?.name || "";
}

// Values are not always primitives: rgthree's Power Lora Loader keeps
// objects on its widgets, and `!==` on two structurally identical
// objects is always true -- which listed every such widget as "changed"
// and showed an identical before and after. Compare by content.
function tlSameValue(a, b) {
    if (a === b) return true;
    if (a === null || b === null) return false;
    if (typeof a !== "object" || typeof b !== "object") return false;
    try {
        return JSON.stringify(a) === JSON.stringify(b);
    } catch (err) {
        return false;   // cyclic or otherwise unserialisable: assume differing
    }
}

// A build progress bar, for the Timeline's "full" and the Result
// Preview's own build.
//
// Repairing a seam re-encodes the clip it touches from the correction to
// the clip's end -- our takes carry a single keyframe, so there is no
// later point to resume stream-copying from. That turned a build that
// used to be mostly packet copying into real work, and a spinner with no
// sense of how far along it is reads as a hang.
const TL_PROGRESS_EVENT = "obvpm.h3.build_progress";

// Middle-button drag pans the canvas -- except over a DOM widget, whose
// element takes the pointerdown and the canvas never hears of it. The
// canvas element listens for pointerdown itself and, in its handler,
// takes pointer capture (CanvasPointer.setPointerCapture), after which
// every move and the release reach it whatever is under the pointer.
// So a middle press that lands on one of our panels is handed to the
// canvas element as a copy of itself and the pan just works; the copy
// carries the same clientX/Y and pointerId, which is all the canvas
// reads. Left and right buttons are untouched: those are the panel's.
function tlPanWithMiddleButton(el) {
    el.addEventListener("pointerdown", (e) => {
        if (e.button !== 1) return;
        const canvas = app.canvas?.canvas;
        if (!canvas) return;
        e.preventDefault();
        e.stopPropagation();
        canvas.dispatchEvent(new PointerEvent("pointerdown", e));
    }, true);
    // the browser's own middle-button behaviours (autoscroll, X11 paste)
    // would otherwise still fire on the element
    for (const type of ["mousedown", "auxclick"]) {
        el.addEventListener(type, (e) => {
            if (e.button === 1) e.preventDefault();
        }, true);
    }
}

function tlProgressBar() {
    const wrap = document.createElement("div");
    Object.assign(wrap.style, {
        display: "none", flexDirection: "column", gap: "2px",
        font: "10px sans-serif",
    });
    const label = document.createElement("div");
    Object.assign(label.style, {
        whiteSpace: "nowrap", overflow: "hidden",
        textOverflow: "ellipsis",
    });
    const track = document.createElement("div");
    Object.assign(track.style, {
        height: "4px", borderRadius: "2px", overflow: "hidden",
        background: "rgba(127,127,127,0.25)",
    });
    const fill = document.createElement("div");
    Object.assign(fill.style, {
        height: "100%", width: "0%", borderRadius: "2px",
        background: TL_ROLE.extend.bg,
        transition: "width 120ms linear",
    });
    track.appendChild(fill);
    wrap.append(label, track);
    return {
        el: wrap,
        set(stage, frac) {
            const P = themePalette();
            label.style.color = P.sub;
            label.textContent = stage || "building…";
            fill.style.width =
                Math.round(Math.max(0, Math.min(1, frac)) * 100) + "%";
            wrap.style.display = "flex";
        },
        // a build that starts from cache never reports anything, so the
        // bar has to be armed with a zero rather than left blank
        arm(stage) { this.set(stage, 0); },
        hide() {
            wrap.style.display = "none";
            fill.style.width = "0%";
        },
    };
}

// Every widget hears every build. The payload names the preview file it
// belongs to, so two nodes building at once do not drive each other.
function tlOnProgress(targetFn, bar) {
    const handler = (e) => {
        const d = e?.detail;
        if (!d || d.target !== targetFn()) return;
        bar.set(d.stage, d.progress);
    };
    api.addEventListener(TL_PROGRESS_EVENT, handler);
    return () => api.removeEventListener(TL_PROGRESS_EVENT, handler);
}

// Seam Improvement Settings.
//
// These six live on the node as hidden widgets (widgets_values is
// positional, so they must stay declared) and are edited here instead.
// They are set-and-forget repair policy, not per-cut controls, and
// leaving them on the node's face pushed the widgets people DO reach for
// off the bottom of it.
//
// Writes are immediate -- there is no OK/Cancel. Each control is its own
// decision, and a Cancel would have to undo a rebuild that the preview
// cache has already keyed on.
// `scope` turns this into ONE JOIN's settings instead of the timeline's:
// same dialog, same controls, same words, with each row marked
// "inherited" or "its own · clear". Reusing it rather than building a
// second, smaller editor is deliberate -- the two would drift, and the
// per-join one is where the wording matters most.
function tlSeamDialog({ get, set, seams, onChange, onClose, scope }) {
    const PAL = themePalette();
    const BG = (typeof LiteGraph !== "undefined"
        && LiteGraph.NODE_DEFAULT_BGCOLOR) || PAL.rest;
    const FG = (typeof LiteGraph !== "undefined"
        && LiteGraph.NODE_TEXT_COLOR) || PAL.text;
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "10000",
        background: "rgba(0,0,0,0.5)", display: "flex",
        alignItems: "center", justifyContent: "center",
        font: "13px sans-serif",
    });
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        background: BG, color: FG,
        border: "1px solid " + PAL.edge, borderRadius: "8px",
        boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
        width: "min(620px, 95vw)", maxHeight: "92vh",
        display: "flex", flexDirection: "column", overflow: "hidden",
    });
    const head = document.createElement("div");
    Object.assign(head.style, {
        padding: "14px 18px", borderBottom: "1px solid " + PAL.edge,
        flexShrink: "0",
    });
    const h = document.createElement("div");
    h.textContent = scope ? scope.title
        : "Seam Improvement Settings";
    Object.assign(h.style, { fontSize: "17px", fontWeight: "600" });
    const sub = document.createElement("div");
    sub.textContent = scope ? scope.sub
        : "How the joins between clips are improved to help make them "
          + "more seamless when the full preview and the export are "
          + "built.";
    Object.assign(sub.style, { color: PAL.sub, marginTop: "5px",
                               fontSize: "13px", lineHeight: "1.5" });
    // These repairs exist for GUIDED-mode joins (the re-rendered window
    // brings a level step and post-overlap flicker). A masked join is
    // latent-identical, so they default to off there -- and switching
    // them on can INTRODUCE flicker. Said once, in every scope.
    const modeNote = document.createElement("div");
    modeNote.textContent = "For joins with two renderings of the same "
        + "moment: a guided join, or an extend from a clip that had no "
        + "saved latents. Other masked joins are latent-identical, so "
        + "these default to off there.";
    Object.assign(modeNote.style, { color: PAL.sub, marginTop: "4px",
                                    fontSize: "12px", fontStyle: "italic",
                                    lineHeight: "1.5" });
    head.append(h, sub, modeNote);

    const body = document.createElement("div");
    Object.assign(body.style, {
        padding: "6px 18px 14px", overflowY: "auto", flex: "1",
    });

    const rows = [];
    // "inherited" vs "its own": without this the dialog cannot say
    // whether a value came from the timeline or from this join, and
    // there would be no way back to inheriting once a control is
    // touched.
    function ownMark(entry) {
        if (!scope || !entry?.name) return;
        const m = document.createElement("button");
        Object.assign(m.style, {
            marginLeft: "auto", background: "none", border: "none",
            font: "11px sans-serif", padding: "0 0 0 10px",
            whiteSpace: "nowrap", flexShrink: "0",
        });
        const paint = () => {
            const own = scope.own(entry.name);
            m.textContent = own ? "its own · clear" : "inherited";
            m.style.color = own ? PAL.text : PAL.sub;
            m.style.textDecoration = own ? "underline" : "none";
            m.style.cursor = own ? "pointer" : "default";
            m.style.opacity = own ? "1" : "0.7";
        };
        m.addEventListener("click", (ev) => {
            // the toggle rows are <label>s, so a click inside them would
            // otherwise flip the checkbox as well
            ev.preventDefault();
            ev.stopPropagation();
            if (!scope.own(entry.name)) return;
            scope.clear(entry.name);
            repaint();
        });
        entry.el.appendChild(m);
        rows.push({ paint });
        paint();
    }
    function paintAll() {
        for (const r of rows) r.paint();
    }
    // paintAll is the initial render; repaint is an EDIT. Keeping them
    // apart matters: onChange discards the built preview, so folding the
    // two together meant merely opening the dialog threw away a full
    // preview the user had just waited for.
    function repaint() {
        paintAll();
        onChange?.();
    }
    function section(title, note) {
        const s = document.createElement("div");
        // The divider SEPARATES sections, so the first one does not get
        // it -- the header's own bottom border already sits there, and
        // two rules a few pixels apart read as a mistake.
        const first = body.children.length === 0;
        Object.assign(s.style, {
            marginTop: first ? "14px" : "16px",
            paddingTop: first ? "0" : "12px",
            borderTop: first ? "none" : "1px solid " + PAL.edge,
        });
        const t = document.createElement("div");
        t.textContent = title;
        Object.assign(t.style, {
            fontWeight: "600", fontSize: "13px", letterSpacing: "0.02em",
            textTransform: "uppercase", color: PAL.sub,
        });
        s.appendChild(t);
        if (note) {
            const n = document.createElement("div");
            n.textContent = note;
            Object.assign(n.style, { color: PAL.sub, fontSize: "12px",
                                     lineHeight: "1.5", marginTop: "4px" });
            s.appendChild(n);
        }
        body.appendChild(s);
        return s;
    }
    // Every option in this dialog defaults ON, so an absent value reads
    // as ticked. A default-OFF control would need its own reading --
    // `get(name) === true` -- rather than this one.
    function toggle(host, name, label, note, { indent = false } = {}) {
        const state = () => get(name) !== false;
        const row = document.createElement("label");
        Object.assign(row.style, {
            display: "flex", gap: "9px", alignItems: "flex-start",
            marginTop: "10px", cursor: "pointer",
            marginLeft: indent ? "24px" : "0",
        });
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = state();
        box.style.marginTop = "2px";
        box.style.flexShrink = "0";
        const text = document.createElement("div");
        const lab = document.createElement("div");
        lab.textContent = label;
        lab.style.fontWeight = "500";
        text.appendChild(lab);
        if (note) {   // an empty note must not leave a gap under the label
            const desc = document.createElement("div");
            desc.textContent = note;
            Object.assign(desc.style, { color: PAL.sub, fontSize: "12px",
                                        lineHeight: "1.5",
                                        marginTop: "2px" });
            text.appendChild(desc);
        }
        row.append(box, text);
        host.appendChild(row);
        box.addEventListener("change", () => {
            set(name, box.checked);
            repaint();
        });
        const entry = {
            name, el: row, input: box, host,
            paint: () => { box.checked = state(); },
        };
        rows.push(entry);
        return entry;
    }
    function number(host, name, label, note, { min, max, indent = true }) {
        const row = document.createElement("div");
        Object.assign(row.style, {
            display: "flex", gap: "9px", alignItems: "flex-start",
            marginTop: "10px", marginLeft: indent ? "24px" : "0",
        });
        const inp = document.createElement("input");
        inp.type = "number";
        inp.min = String(min);
        inp.max = String(max);
        inp.value = String(get(name) ?? min);
        Object.assign(inp.style, {
            width: "68px", flexShrink: "0",
            background: PAL.rest, color: PAL.text,
            border: "1px solid " + PAL.edge, borderRadius: "4px",
            padding: "3px 6px", font: "13px sans-serif",
        });
        const text = document.createElement("div");
        const lab = document.createElement("div");
        lab.textContent = label;
        lab.style.fontWeight = "500";
        const desc = document.createElement("div");
        desc.textContent = note;
        Object.assign(desc.style, { color: PAL.sub, fontSize: "12px",
                                    lineHeight: "1.5", marginTop: "2px" });
        text.append(lab, desc);
        row.append(inp, text);
        host.appendChild(row);
        const commit = () => {
            // clamp HERE rather than trusting the spinner: typing a
            // value straight into a number input bypasses min/max
            let v = Math.round(Number(inp.value));
            if (!Number.isFinite(v)) v = min;
            v = Math.max(min, Math.min(max, v));
            inp.value = String(v);
            set(name, v);
            repaint();
        };
        inp.addEventListener("change", commit);
        inp.addEventListener("blur", commit);
        const entry = {
            name, el: row, input: inp, host,
            paint: () => { inp.value = String(get(name) ?? min); },
        };
        rows.push(entry);
        return entry;
    }
    function dependOn(master, kids) {
        const paint = () => {
            const on = get(master) !== false;
            for (const k of kids) {
                k.el.style.opacity = on ? "1" : "0.45";
                k.input.disabled = !on;
                k.el.style.pointerEvents = on ? "" : "none";
            }
        };
        rows.push({ paint });
        return paint;
    }

    const s1 = section("Level lock",
        "Some takes open at a different level (brightness) than the clip "
        + "they continue, then settle back over about half a second. This "
        + "causes a visible flicker or brightness change.");
    toggle(s1, "level_lock", "Match each clip's opening to the clip before it",
           "Corrects the opening and fades the correction out.");
    const f1 = number(s1, "level_lock_frames", "Frames to correct over",
                      "24 frames = one second. Longer also flattens real "
                      + "lighting changes; shorter can look like a ramp.",
                      { min: 2, max: 120 });
    const f2 = toggle(s1, "level_lock_flicker", "Also smooth flicker",
                      "Some joins don't jump, but wobble for a few frames. "
                      + "This smooths them out.", { indent: true });
    const f2b = toggle(s1, "level_lock_local",
                       "Correct regions separately",
                       "Normally the whole frame is corrected by one "
                       + "amount. This lets different parts of the "
                       + "picture be corrected by different amounts, "
                       + "which helps when only part of the frame "
                       + "changed. Needs crossfade switched on, because "
                       + "that is what measures where the difference is.",
                       { indent: true });
    // two masters, so dependOn (single master) will not do
    rows.push({ paint: () => {
        const on = get("level_lock") !== false && get("crossfade") !== false;
        f2b.el.style.opacity = on ? "1" : "0.45";
        f2b.input.disabled = !on;
        f2b.el.style.pointerEvents = on ? "" : "none";
    } });
    dependOn("level_lock", [f1, f2]);

    // NOT "spreads the difference out": that described the first
    // version, which faded from the parent onto the child's own
    // brightness by simply not matching, and so inherited the incoming
    // rendering's own wobbly level path -- a swell with a flash in it
    // where there had been a step. The incoming frames are matched to
    // the parent's level first. With the level lock on the fade STAYS
    // on the parent and the lock takes the level from there; with it
    // off the match target ramps onto the child instead, so the fade
    // carries the level over itself along a curve we choose rather
    // than the model's. See crossfade.handover_ramp.
    const s2 = section("Crossfade",
        "When a clip continues another, it re-generates the last moment of "
        + "the clip before it. That gives two versions of the same moment, "
        + "so instead of switching from one to the other in a single "
        + "frame, they can be blended across it.");
    toggle(s2, "crossfade", "Blend across the overlap",
           "Works together with the level lock, not instead of it: this "
           + "smooths the picture across the join, the level lock fixes "
           + "the brightness after it.");
    const f3 = number(s2, "crossfade_frames", "Fade length",
                      "0 = the whole stored overlap.",
                      { min: 0, max: 240 });
    dependOn("crossfade", [f3]);
    // Measured on clip_00047 -> clip_00053: with the ramp, crossfade
    // alone takes the step from 4.15 to 0.03. But the child then
    // decays 126 -> 122 over the following ~10 frames on its own, so
    // the join is stepless and still swells. With the level lock the
    // child is pulled DOWN to where the parent was heading and nothing
    // moves: 0.11. Hence "better together", not "no fix".
    const warn = document.createElement("div");
    Object.assign(warn.style, {
        marginTop: "10px", marginLeft: "24px", padding: "6px 9px",
        borderRadius: "4px", fontSize: "12px", lineHeight: "1.5",
        background: "rgba(201,151,59,0.15)",
        border: "1px solid rgba(201,151,59,0.5)",
    });
    warn.textContent = "On its own, blending carries the brightness "
        + "across the join as well — but the clip after it still "
        + "settles for about half a second. Turn the level lock on too "
        + "and it won't.";
    s2.appendChild(warn);
    rows.push({ paint: () => {
        warn.style.display = (get("crossfade") !== false
                              && get("level_lock") === false)
            ? "block" : "none";
    } });

    const s3 = section("Audio",
        "A blended join blends its sound too. Only guided joins can be "
        + "blended, so on a masked or both-mode sequence this is the "
        + "only audio treatment there is.");
    toggle(s3, "audio_declick", "De-click cuts with no overlap",
           "Tapers 5ms out of the outgoing clip and 5ms into the "
           + "incoming one. Removes the click at the join; it cannot "
           + "hide a change in room tone.");

    const s4 = section("Joins in this sequence");
    const measured = document.createElement("div");
    Object.assign(measured.style, { marginTop: "8px", fontSize: "12px",
                                    lineHeight: "1.7", color: PAL.sub,
                                    whiteSpace: "pre-line" });
    const list = Object.values(seams || {});
    if (!list.length) {
        measured.textContent = "Nothing to check yet. Add two clips where "
            + "one continues the other.";
    } else {
        const n = (k) => list.filter((s) => s.reads_as === k).length;
        const fixable = list.filter((s) => s.fixable).length;
        const found = [
            [n("step"), "a brightness jump"],
            [n("flicker"), "flicker"],
            [n("step + flicker"), "both a brightness jump and flicker"],
        ].filter(([c]) => c > 0);
        const lines = [
            `Checked ${list.length} join${list.length === 1 ? "" : "s"} `
            + `between clips.`,
        ];
        if (!found.length) {
            lines.push("None of them show a brightness jump or flicker.");
        } else {
            for (const [c, what] of found) {
                lines.push(`• ${c} ${c === 1 ? "has" : "have"} `
                           + `${what}.`);
            }
        }
        if (fixable) {
            lines.push(`${fixable} can be improved by the settings above.`);
        }
        measured.textContent = lines.join("\n");
    }
    s4.appendChild(measured);

    if (scope) {
        // A control that cannot act on THIS join is hidden, not greyed:
        // the repairs each need something of the join itself, and there
        // is nothing the reader could do to make an inapplicable one
        // work. What is left is exactly what this seam can be given.
        for (const r of rows.slice()) {
            if (!r.name) continue;
            if (scope.applies && !scope.applies(r.name)) {
                r.el.style.display = "none";
                continue;
            }
            // every remaining control gets its inherited/own marker, in
            // one pass, so a control added later cannot forget one
            ownMark(r);
        }
        // a section with nothing left in it goes too, heading and all
        for (const sec of [s1, s2, s3]) {
            const mine = rows.filter((r) => r.name && r.host === sec);
            if (mine.length
                    && mine.every((r) => r.el.style.display === "none")) {
                sec.style.display = "none";
            }
        }
        // the sequence-wide readout belongs to the timeline's dialog,
        // not to one join
        s4.style.display = "none";
    }

    // What the current sequence actually looks like, so the settings are
    // read against this cut rather than in the abstract. Written as
    // sentences, not a table of category counts: the counts only mean
    // something to someone who already knows what the categories are.
    const foot = document.createElement("div");
    Object.assign(foot.style, {
        padding: "12px 18px", borderTop: "1px solid " + PAL.edge,
        display: "flex", justifyContent: "flex-end", gap: "8px",
        flexShrink: "0",
    });
    if (scope) {
        // the way back to inheriting everything, in one action -- the
        // per-row "clear" is for changing your mind about one thing
        const all = document.createElement("button");
        all.textContent = "Reset to timeline defaults";
        Object.assign(all.style, {
            padding: "6px 14px", borderRadius: "5px", cursor: "pointer",
            border: "1px solid " + PAL.edge, background: "none",
            color: PAL.text, font: "13px sans-serif",
            marginRight: "auto",
        });
        const paintAllBtn = () => {
            const n = scope.count();
            all.style.display = n ? "block" : "none";
        };
        all.addEventListener("click", () => {
            scope.clearAll();
            repaint();
            paintAllBtn();
        });
        rows.push({ paint: paintAllBtn });
        paintAllBtn();
        foot.appendChild(all);
    }
    const done = document.createElement("button");
    done.textContent = "Done";
    Object.assign(done.style, {
        padding: "6px 18px", borderRadius: "5px", cursor: "pointer",
        border: "1px solid transparent", background: TL_ROLE.extend.bg,
        color: "#fff", font: "600 13px sans-serif",
    });
    foot.appendChild(done);

    panel.append(head, body, foot);
    overlay.appendChild(panel);
    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        overlay.remove();
        document.removeEventListener("keydown", onKey, true);
        // Escape and the backdrop close it too, so the OWNER has to be
        // told -- otherwise its "already open?" flag stays set and the
        // button needs two clicks to reopen.
        onClose?.();
    };
    const onKey = (ev) => {
        if (ev.key === "Escape") { ev.stopPropagation(); close(); }
    };
    document.addEventListener("keydown", onKey, true);
    overlay.addEventListener("click", (ev) => {
        if (ev.target === overlay) close();
    });
    done.addEventListener("click", close);
    document.body.appendChild(overlay);
    paintAll();
    return close;
}

// The everyday half of tlSeamDialog: one control, the audio de-click.
//
// Why a second dialog and not a mode of the first. The picture repairs
// (level lock, crossfade) can only act on a GUIDED join, and guided mode
// is withheld from this UI (SHOW_GUIDED) -- so five of the six controls
// in the full dialog cannot do anything on the route people actually
// run. The de-click is the exact opposite: it applies to every join that
// is NOT blended, which on a masked or both-mode sequence is all of
// them. The one setting that always matters was buried among five that
// usually cannot act.
//
// Deliberately self-contained rather than a `scope`-style variant of
// tlSeamDialog: that dialog still serves a join that carries overrides
// of its own, so nothing about the full editor has to change shape
// because this one exists. The cost is a copy of the panel chrome,
// which is the cheap half; the wording -- where the actual thinking is
// -- is not duplicated, because the two say different things.
// It used to offer "Advanced..." into the full editor; that went when
// guided mode was withheld, since none of the six controls there can
// act on a masked or both-mode join.
function tlAudioSeamDialog({ get, set, seams, onChange, onClose }) {
    const PAL = themePalette();
    const BG = (typeof LiteGraph !== "undefined"
        && LiteGraph.NODE_DEFAULT_BGCOLOR) || PAL.rest;
    const FG = (typeof LiteGraph !== "undefined"
        && LiteGraph.NODE_TEXT_COLOR) || PAL.text;
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "10000",
        background: "rgba(0,0,0,0.5)", display: "flex",
        alignItems: "center", justifyContent: "center",
        font: "13px sans-serif",
    });
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        background: BG, color: FG,
        border: "1px solid " + PAL.edge, borderRadius: "8px",
        boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
        width: "min(520px, 95vw)", maxHeight: "92vh",
        display: "flex", flexDirection: "column", overflow: "hidden",
    });

    const head = document.createElement("div");
    Object.assign(head.style, {
        padding: "14px 18px", borderBottom: "1px solid " + PAL.edge,
        flexShrink: "0",
    });
    const h = document.createElement("div");
    h.textContent = "Seam Audio";
    Object.assign(h.style, { fontSize: "17px", fontWeight: "600" });
    const sub = document.createElement("div");
    sub.textContent = "How the sound is handled where one clip meets the "
        + "next, when the full preview and the export are built.";
    Object.assign(sub.style, { color: PAL.sub, marginTop: "5px",
                               fontSize: "13px", lineHeight: "1.5" });
    head.append(h, sub);

    const body = document.createElement("div");
    Object.assign(body.style, {
        padding: "16px 18px", overflowY: "auto", flex: "1",
    });

    // Writes are immediate, like the full dialog: there is no OK/Cancel,
    // because a Cancel would have to undo a rebuild the preview cache
    // has already keyed on.
    const row = document.createElement("label");
    Object.assign(row.style, {
        display: "flex", gap: "9px", alignItems: "flex-start",
        cursor: "pointer",
    });
    const box = document.createElement("input");
    box.type = "checkbox";
    // Absent reads as ON, matching the widget default.
    box.checked = get("audio_declick") !== false;
    box.style.marginTop = "2px";
    box.style.flexShrink = "0";
    const text = document.createElement("div");
    const lab = document.createElement("div");
    lab.textContent = "De-click the joins";
    lab.style.fontWeight = "500";
    const desc = document.createElement("div");
    desc.textContent = "Fades 5ms out of the outgoing clip and 5ms into "
        + "the incoming one, on both sides of every join. That removes "
        + "the click \u2014 the jump from one clip's last sample to the "
        + "next clip's first \u2014 and 10ms is short enough that no gap "
        + "is audible. It cannot hide a change in room tone; that needs "
        + "real overlapping sound.";
    Object.assign(desc.style, { color: PAL.sub, fontSize: "12px",
                                lineHeight: "1.55", marginTop: "3px" });
    text.append(lab, desc);
    row.append(box, text);
    body.appendChild(row);
    box.addEventListener("change", () => {
        set("audio_declick", box.checked);
        onChange?.();
    });

    const note = document.createElement("div");
    Object.assign(note.style, {
        marginTop: "16px", padding: "9px 11px", borderRadius: "5px",
        border: "1px solid " + PAL.edge, color: PAL.sub,
        fontSize: "12px", lineHeight: "1.6",
    });
    note.textContent = "A join whose picture is blended blends its sound "
        + "across the same span and needs none of this. Only a guided "
        + "join can be blended \u2014 it is the one that re-generates the "
        + "moment it joins at, so there are two recordings of it to fade "
        + "between. A masked or both-mode join keeps the source exactly, "
        + "which is why it looks seamless, but it leaves nothing to fade "
        + "across: on those sequences this applies to every join.";
    body.appendChild(note);

    const list = Object.values(seams || {});
    if (list.length) {
        const count = document.createElement("div");
        Object.assign(count.style, { marginTop: "14px", color: PAL.sub,
                                     fontSize: "12px", lineHeight: "1.6" });
        count.textContent = `This sequence has ${list.length} join`
            + `${list.length === 1 ? "" : "s"} between clips.`;
        body.appendChild(count);
    }

    const foot = document.createElement("div");
    Object.assign(foot.style, {
        padding: "12px 18px", borderTop: "1px solid " + PAL.edge,
        display: "flex", justifyContent: "flex-end", gap: "8px",
        flexShrink: "0",
    });
    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        overlay.remove();
        document.removeEventListener("keydown", onKey, true);
        // Escape and the backdrop close it too, so the OWNER has to be
        // told -- otherwise its "already open?" flag stays set and the
        // button needs two clicks to reopen.
        onClose?.();
    };
    const done = document.createElement("button");
    done.textContent = "Done";
    Object.assign(done.style, {
        padding: "6px 18px", borderRadius: "5px", cursor: "pointer",
        border: "1px solid transparent", background: TL_ROLE.extend.bg,
        color: "#fff", font: "600 13px sans-serif",
    });
    foot.appendChild(done);

    panel.append(head, body, foot);
    overlay.appendChild(panel);
    const onKey = (ev) => {
        if (ev.key === "Escape") { ev.stopPropagation(); close(); }
    };
    document.addEventListener("keydown", onKey, true);
    overlay.addEventListener("click", (ev) => {
        if (ev.target === overlay) close();
    });
    done.addEventListener("click", close);
    document.body.appendChild(overlay);
    return close;
}

function tlLoadDialog({ clip, keyword, plan, pinRole }) {
    return new Promise((resolve) => {
        // Theme colours only. themePalette() has no `bg`/`accent` keys --
        // reaching for them silently fell through to a hardcoded dark
        // panel, which on a light theme meant dark text on dark ground.
        const PAL = themePalette();
        // The same ground a normal node is drawn on, so the dialog reads
        // as part of the graph rather than as a foreign surface.
        const BG = (typeof LiteGraph !== "undefined"
            && LiteGraph.NODE_DEFAULT_BGCOLOR) || PAL.rest;
        const FG = (typeof LiteGraph !== "undefined"
            && LiteGraph.NODE_TEXT_COLOR) || PAL.text;
        const overlay = document.createElement("div");
        Object.assign(overlay.style, {
            position: "fixed", inset: "0", zIndex: "10000",
            background: "rgba(0,0,0,0.5)", display: "flex",
            alignItems: "center", justifyContent: "center",
            font: "13px sans-serif",
        });
        const panel = document.createElement("div");
        Object.assign(panel.style, {
            background: BG, color: FG,
            border: "1px solid " + PAL.edge, borderRadius: "8px",
            boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
            // vw/vh caps keep it on screen whatever the content: the
            // body scrolls, the panel itself never overflows.
            width: "min(1500px, 95vw)", maxHeight: "92vh",
            display: "flex", flexDirection: "column", overflow: "hidden",
        });

        const head = document.createElement("div");
        Object.assign(head.style, {
            padding: "14px 18px", borderBottom: "1px solid " + PAL.edge,
            flexShrink: "0",
        });
        const h = document.createElement("div");
        h.textContent = "Load settings from " + clip.split("/").pop();
        Object.assign(h.style, { fontSize: "17px", fontWeight: "600" });
        const sub = document.createElement("div");
        sub.textContent = "Replaces " + plan.changes.length + " value(s) on "
            + plan.applied.length + " node(s), in " + plan.groups.length
            + " group(s) matching \"" + keyword + "\": "
            + plan.groups.join(", ");
        Object.assign(sub.style, { color: PAL.sub, marginTop: "5px",
                                   fontSize: "13px", lineHeight: "1.5" });
        head.append(h, sub);

        const body = document.createElement("div");
        Object.assign(body.style, {
            padding: "12px 18px", overflowY: "auto", flex: "1",
        });

        const table = document.createElement("table");
        Object.assign(table.style, {
            width: "100%", borderCollapse: "collapse", tableLayout: "fixed",
            fontSize: "13px",
        });
        const thead = document.createElement("thead");
        const hrow = document.createElement("tr");
        // Node and Setting hold short identifiers, so they get a FIXED
        // width sized to a name rather than a share of the table -- a
        // percentage grows them on a wide screen for no reason. The two
        // value columns are left unsized: under table-layout:fixed the
        // remainder is split evenly between them, which is where a
        // prompt actually needs the room. Long names wrap.
        const master = document.createElement("input");
        master.type = "checkbox";
        master.checked = true;
        master.title = "Apply all / none";
        Object.assign(master.style, { cursor: "pointer", margin: "0" });
        [["Node", "140px"], ["Setting", "105px"], ["Current", ""],
         ["New", ""], ["Apply", "58px"]].forEach(([label, width], i, all) => {
            const th = document.createElement("th");
            if (i === all.length - 1) {
                const wrap = document.createElement("label");
                Object.assign(wrap.style, {
                    display: "flex", gap: "5px", alignItems: "center",
                    cursor: "pointer",
                });
                const cap = document.createElement("span");
                cap.textContent = label;
                wrap.append(master, cap);
                th.appendChild(wrap);
            } else {
                th.textContent = label;
            }
            Object.assign(th.style, {
                width: width, textAlign: "left", padding: "4px 10px 6px 0",
                borderBottom: "1px solid " + PAL.edge,
                color: PAL.sub, fontSize: "12px", fontWeight: "600",
                position: "sticky", top: "0", background: BG,
            });
            hrow.appendChild(th);
        });
        thead.appendChild(hrow);
        table.appendChild(thead);

        // key -> checkbox, so the resolved selection and the live count
        // both read from the DOM the user actually ticked
        const boxes = new Map();
        let pinBox = null;
        function selectedKeys() {
            const out = new Set();
            for (const [key, box] of boxes) if (box.checked) out.add(key);
            return out;
        }
        function refreshCount() {
            const n = selectedKeys().size;
            const pin = pinBox?.checked ? 1 : 0;
            sub.textContent = "Applies " + n + " of " + plan.changes.length
                + " value(s) in " + plan.groups.length + " group(s) matching "
                + "\"" + keyword + "\": " + plan.groups.join(", ");
            ok.disabled = !n && !pin;
            ok.style.opacity = (!n && !pin) ? "0.5" : "1";
            ok.style.cursor = (!n && !pin) ? "not-allowed" : "pointer";
            if (boxes.size) {
                master.checked = n === boxes.size;
                master.indeterminate = n > 0 && n < boxes.size;
            }
        }

        const tbody = document.createElement("tbody");
        // grouped by node: the name is printed once per node, so six
        // changed widgets read as one decision rather than six
        let lastNode = null;
        for (const c of plan.changes) {
            const fresh = c.name !== lastNode;
            const tr = document.createElement("tr");
            const mk = (value, style) => {
                const td = document.createElement("td");
                Object.assign(td.style, {
                    padding: (fresh && lastNode !== null ? "8px" : "3px")
                             + " 10px 3px 0",
                    verticalAlign: "top",
                    borderTop: fresh && lastNode !== null
                        ? "1px solid " + PAL.edge : "none",
                });
                const d = document.createElement("div");
                d.textContent = value;
                Object.assign(d.style, {
                    // a long value (a prompt) scrolls in place rather
                    // than being cut off: nothing is hidden from the
                    // decision the dialog is asking for. Capped in vh as
                    // well as px so one huge prompt cannot fill a small
                    // screen and bury every other row.
                    maxHeight: "min(360px, 45vh)", overflowY: "auto",
                    whiteSpace: "pre-wrap", overflowWrap: "anywhere",
                    lineHeight: "1.45",
                });
                Object.assign(d.style, style || {});
                td.appendChild(d);
                return td;
            };
            const box = document.createElement("input");
            box.type = "checkbox";
            box.checked = true;
            box.title = "Apply this change";
            Object.assign(box.style, { cursor: "pointer", margin: "0" });
            boxes.set(c.key, box);
            const tick = document.createElement("td");
            Object.assign(tick.style, {
                padding: (fresh && lastNode !== null ? "8px" : "3px")
                         + " 10px 3px 0",
                verticalAlign: "top",
                borderTop: fresh && lastNode !== null
                    ? "1px solid " + PAL.edge : "none",
            });
            tick.appendChild(box);
            box.addEventListener("change", refreshCount);
            tr.append(
                mk(fresh ? c.name : "",
                   { fontWeight: "600", maxHeight: "none" }),
                mk(c.what, { color: PAL.sub, maxHeight: "none" }),
                mk(tlValueText(c.from), { opacity: "0.7" }),
                mk(tlValueText(c.to), {}),
                tick);
            // dim a row the user has opted out of, so the table shows
            // what will happen rather than what was proposed
            box.addEventListener("change", () => {
                for (const td of tr.children) {
                    td.style.opacity = box.checked ? "1" : "0.4";
                }
            });
            lastNode = c.name;
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        body.appendChild(table);

        if (pinRole) {
            const note = document.createElement("label");
            Object.assign(note.style, {
                marginTop: "12px", fontSize: "13px", color: FG,
                display: "flex", gap: "8px", alignItems: "center",
                cursor: "pointer",
            });
            pinBox = document.createElement("input");
            pinBox.type = "checkbox";
            pinBox.checked = true;
            Object.assign(pinBox.style, { cursor: "pointer", margin: "0" });
            pinBox.addEventListener("change", refreshCount);
            const txt = document.createElement("span");
            txt.textContent = "Set this timeline's pin to " + pinRole + ".";
            note.append(pinBox, txt);
            body.appendChild(note);
        }
        const left = [...plan.skipped, ...plan.missing];
        if (left.length) {
            const det = document.createElement("details");
            Object.assign(det.style, { marginTop: "12px" });
            const sm = document.createElement("summary");
            sm.textContent = left.length + " node(s) left alone";
            Object.assign(sm.style, { cursor: "pointer", color: PAL.sub,
                                      fontSize: "13px" });
            det.appendChild(sm);
            for (const line of left) {
                const d = document.createElement("div");
                d.textContent = line;
                Object.assign(d.style, { color: PAL.sub, fontSize: "12px",
                                         padding: "2px 0 2px 16px",
                                         lineHeight: "1.45" });
                det.appendChild(d);
            }
            body.appendChild(det);
        }

        const foot = document.createElement("div");
        Object.assign(foot.style, {
            padding: "12px 18px", borderTop: "1px solid " + PAL.edge,
            display: "flex", gap: "10px", alignItems: "center",
            flexShrink: "0",
        });
        const scope = document.createElement("span");
        scope.textContent = "Values and bypass state only \u2014 nothing is "
            + "created, rewired or moved; subgraphs untouched.";
        Object.assign(scope.style, { color: PAL.sub, fontSize: "12px",
                                     flex: "1", lineHeight: "1.45" });
        const mkBtn = (label, primary) => {
            const b = document.createElement("button");
            b.textContent = label;
            Object.assign(b.style, {
                background: primary ? PAL.active : BG,
                color: primary ? PAL.activeText : FG,
                border: "1px solid " + (primary ? "transparent" : PAL.edge),
                borderRadius: "4px", padding: "6px 16px", cursor: "pointer",
                font: "13px/19px sans-serif", whiteSpace: "nowrap",
            });
            return b;
        };
        const cancel = mkBtn("Cancel", false);
        const ok = mkBtn("Load settings", true);
        master.addEventListener("change", () => {
            // Capture the target FIRST. Each row's change handler calls
            // refreshCount, which rewrites master.checked from the running
            // tally -- so reading master.checked inside the loop flipped
            // it to false after the first row and the rest never changed.
            const want = master.checked;
            for (const box of boxes.values()) {
                if (box.checked !== want) {
                    box.checked = want;
                    box.dispatchEvent(new Event("change"));
                }
            }
            refreshCount();
        });
        refreshCount();
        foot.append(scope, cancel, ok);
        panel.append(head, body, foot);
        overlay.appendChild(panel);

        let done = false;
        const close = (value) => {
            if (done) return;
            done = true;
            document.removeEventListener("keydown", onKey, true);
            overlay.remove();
            resolve(value);
        };
        const accept = () => close({ only: selectedKeys(),
                                     pin: !!pinBox?.checked });
        function onKey(ev) {
            if (ev.key === "Escape") { ev.stopPropagation(); close(null); }
            else if (ev.key === "Enter") {
                ev.stopPropagation();
                if (!ok.disabled) accept();
            }
        }
        // capture phase: the canvas has its own global key handlers, and
        // Escape must not reach them while a modal is up
        document.addEventListener("keydown", onKey, true);
        overlay.addEventListener("mousedown", (ev) => {
            if (ev.target === overlay) close(null);
        });
        cancel.addEventListener("click", () => close(null));
        ok.addEventListener("click", () => { if (!ok.disabled) accept(); });
        document.body.appendChild(overlay);
        ok.focus();
    });
}

// The take's own pin options: what the Timeline was set to when that
// clip was generated. Restored regardless of the group keyword -- it is
// this node's own state, and it is the setting that reproduces the take.
// `sequence` is deliberately NOT restored: it is the user's current cut,
// not a property of the clip.
function tlSavedPinState(wf, selfId) {
    const timelines = (wf?.nodes ?? []).filter((n) => n.type === TL_NODE);
    const saved = timelines.find((n) => String(n.id) === String(selfId))
        ?? (timelines.length === 1 ? timelines[0] : null);
    if (!saved) return undefined;
    const named = saved.widgets_values_named;
    if (named && typeof named === "object" && !Array.isArray(named)
            && "pin_state" in named) {
        return named.pin_state;
    }
    return undefined;   // positional is unsafe here: pin_state is optional
}
const TL_FPS = 24;
const TL_DRAG_MIME = "application/x-obvpm-timeline-index";
const TL_PICKER_CLASS = "H3LoadVideoWithMCtx"; // combo lists the output tree
const TL_COLORS = { seamless: "#4d9960", cut: "#c9973b", butt: "#a4aab4" };
// pin-role colors: dark purple for BOTH roles (user rule, after green
// and dark-orange tries) -- the badge text ("extending"/"prepending")
// carries the distinction
const TL_ROLE = {
    extend: { bg: "rgba(96,60,160,0.92)", hover: "rgba(118,78,190,0.95)",
              text: "#a678e8" },
    prepend: { bg: "rgba(96,60,160,0.92)",
               hover: "rgba(118,78,190,0.95)", text: "#a678e8" },
};
// The run-mode badge's fill, deliberately NOT the pin purple. That row
// normally says which pin the next Run will generate from; in upscale
// mode it says the next Run is not a generation at all, which is the
// opposite claim. Wearing the pin colour to say it would read as one
// more pin state.
const TL_UPSCALE_BG = "rgba(28,106,116,0.92)";
const TL_UPSCALE_HOVER = "rgba(38,132,144,0.95)";
// sibling-widget channel: Timeline nodes register {folder, offer,
// forget} here so the Result Preview can hand them a good take or
// report a deleted one. Frontend-only on purpose -- the graph stays
// unwired, the Timeline stays outside execution.
const TL_REGISTRY = new Map();

// Dormant, like DEBUG above: flip on when hunting timeline bugs. The
// call sites stay -- they are the map of where past bugs lived.
const TL_DEBUG = false;
function tldbg(...args) {
    if (TL_DEBUG) console.log("[obvpm-h3-timeline]", ...args);
}
tldbg("timeline code loaded, build v39-resume-play");

// Mirrors _parse_sequence / format_sequence_line in nodes_assemble.py.
// The two MUST agree: the widget writes what the server reads back, so a
// cut that round-trips differently here is a cut that silently moves.
const TL_LINE_RE = /^(.+?)(?:\s*@\s*(\d+)?(?:\s*\.\.\s*(\d+))?)?$/;
const TL_GAP_RE = /^~\s*(\d+)$/;
// The one directive: `loop` makes the cut a RING -- its last entry joins
// its first. Mirrors _LOOP_RE / sequence_loops. Not an entry: every
// index map over the raw lines must skip it the way it skips comments,
// or a cut lands one line off.
const TL_LOOP_RE = /^loop$/i;
const TL_LOOP_LINE = "loop";
function tlSequenceLoops(text) {
    return String(text || "").split("\n")
        .some((raw) => TL_LOOP_RE.test(raw.trim()));
}
function tlIsEntryLine(trimmed) {
    return !!trimmed && !trimmed.startsWith("#")
        && !TL_LOOP_RE.test(trimmed);
}

// PER-SEAM SETTINGS: mirrors nodes_assemble's split_seam_opts /
// format_seam_opts, and must stay in lockstep with them. The bracket
// section is stripped BEFORE the cut regex runs, so TL_LINE_RE -- the
// part that has to match the Python one exactly or a cut silently
// becomes a different cut -- is untouched by all of this.
const TL_OPTS_RE = /^(.*?)\s*\[([^\[\]]*)\]$/;
const TL_SEAM_KEYS = {
    level_lock: "bool",
    level_lock_frames: "int",
    level_lock_flicker: "bool",
    level_lock_local: "bool",
    crossfade: "bool",
    crossfade_frames: "int",
    audio_declick: "bool",
};
// written in THIS order, always, so formatting a parsed line reproduces
// it -- a round trip that depends on key order is not a round trip
const TL_SEAM_ORDER = ["level_lock", "level_lock_frames",
                       "level_lock_flicker", "level_lock_local",
                       "crossfade", "crossfade_frames", "audio_declick"];
const TL_TRUE = ["1", "on", "true", "yes"];
const TL_FALSE = ["0", "off", "false", "no"];

// An unknown key or unreadable value is DROPPED, not refused: a wrong
// setting degrades to the timeline's default, while an unknown ENTRY
// would drop content -- which is why "~" gaps fail loudly and this
// deliberately does not.
function tlSplitSeamOpts(line) {
    const m = line.match(TL_OPTS_RE);
    if (!m) return { head: line, seam: {} };
    const seam = {};
    for (const token of m[2].split(/\s+/)) {
        if (!token) continue;
        const at = token.indexOf("=");
        if (at < 0) continue;
        const key = token.slice(0, at);
        const raw = token.slice(at + 1).trim();
        const kind = TL_SEAM_KEYS[key];
        if (!kind) continue;
        if (kind === "bool") {
            const low = raw.toLowerCase();
            if (TL_TRUE.includes(low)) seam[key] = true;
            else if (TL_FALSE.includes(low)) seam[key] = false;
            continue;
        }
        if (/^-?\d+$/.test(raw)) seam[key] = Number(raw);
    }
    return { head: m[1].trim(), seam };
}

function tlFormatSeamOpts(seam) {
    if (!seam) return "";
    const parts = [];
    for (const key of TL_SEAM_ORDER) {
        const v = seam[key];
        if (v === undefined || v === null) continue;
        parts.push(key + "=" + (TL_SEAM_KEYS[key] === "bool"
            ? (v ? "on" : "off") : Math.round(Number(v))));
    }
    return parts.length ? " [" + parts.join(" ") + "]" : "";
}

// The seam-repair defaults, in ONE place. The Timeline reads its own
// widgets and falls back to these; the Result Preview has no such
// widgets and uses them directly, so a take is judged under the values
// the export would use even when no Timeline claims its folder.
const TL_FIX_DEFAULTS = {
    level_lock: true,
    level_lock_frames: 12,
    level_lock_flicker: true,
    level_lock_local: true,
    crossfade: true,
    crossfade_frames: 0,
    audio_declick: true,
};

// What a join can actually be GIVEN. The repairs are not universal, and
// offering a control that cannot act is a promise the build will not
// keep:
//
//   the LEVEL LOCK needs `extends` lineage -- levellock's gains()
//   returns nothing otherwise, so an unrelated pair or a prepend gets
//   no correction;
//
//   the CROSSFADE needs that AND the join still at its natural place.
//   The overlap was rendered for one exact moment, so a cut on either
//   side moves the join away from it and fading would blend two
//   different moments. It also needs the take to have STORED that
//   overlap;
//
//   CARRIED REGIONS borrow the fade's measurement, so no fade means
//   nothing to carry;
//
//   AUDIO DE-CLICK is the mirror image: it treats the boundaries the
//   audio crossfade does NOT cover, so it is offered exactly where the
//   fade cannot run. Between them no join is ever left with an empty
//   dialog.
//
// Takes the two ENTRIES either side of the join, not an index, so the
// Timeline (which has a sequence) and the Result Preview (which has a
// clip pair and their sidecars) can ask the same question. null = not a
// join at all: empty space on either side is not a continuation.
// Which grade of pin a timeline entry can supply, or null for none.
//
//   "latent"  a verified sidecar: an exact slice, no VAE anywhere
//   "pixel"   no usable sidecar: VAE-encoded from the file at run time,
//             so the join is soft and the new take is a root
//
// A gap supplies nothing -- there is no clip there to pin. This used to
// be an inline `!e.meta` that HID the controls, which was
// indistinguishable from the feature being broken; now it decides how to
// LABEL them.
function tlHasSidecar(meta) {
    // A clip with no sidecar still gets a meta now -- a SYNTHETIC one
    // carrying its hash and length so the seam machinery can name it as
    // a parent (see nodes_load.synthetic_header). It has no latents, so
    // every "can I slice this?" question has to ask for the marker, not
    // merely for the presence of meta.
    return !!meta && !meta.no_sidecar;
}

function tlPinGrade(entry) {
    if (!entry || entry.gap != null || !entry.clip) return null;
    return tlHasSidecar(entry.meta) ? "latent" : "pixel";
}

function tlSeamCapability(L, R) {
    if (!L || !R || L.gap != null || R.gap != null) return null;
    const ex = tlExtendParent(R.meta);
    const extending = !!(ex && L.meta && ex.id === L.meta.self_id);
    // A prepend is the same join read backwards: the take on the LEFT
    // was rendered to arrive at the clip on the RIGHT, so the level
    // lock corrects the take's CLOSING instead of the child's opening.
    // The crossfade still cannot run -- it needs a stored overlap at
    // the join, which only an extend has.
    const pr = tlPrependChild(L.meta);
    const prepending = !!(pr && R.meta && pr.id === R.meta.self_id);
    const delivered = Number(L.meta?.delivered_frames ?? 0) || 0;
    // A window encoded from PIXELS may sit anywhere in its source, so
    // there the natural place is the exit the lineage derives and not
    // only the clip's end (crossfade.usable, same exception).
    const pixelExit = tlPixelJoin(L, R) && ex
        ? ex.join - (Number(L.meta?.pinned_head_frames ?? 0) || 0) : null;
    const natural = extending && !R.enter
        && (L.exit == null
            || (delivered && Number(L.exit) === delivered)
            || (pixelExit != null && Number(L.exit) === pixelExit));
    const fade = !!(natural && Number(R.meta?.overlap_frames ?? 0) > 0);
    // The prepend mirror: the take on the LEFT kept the re-render, and
    // the target must still enter where its lineage says -- a manual cut
    // moves the join away from the moment the overlap was rendered for.
    const tnatural = prepending && L.exit == null
        && Number(L.meta?.delivered_frames ?? 0) > 0;
    const tfade = !!(tnatural
                     && Number(L.meta?.overlap_tail_frames ?? 0) > 0);
    const anyFade = fade || tfade;
    return { lock: extending || prepending, fade: anyFade,
             local: anyFade, declick: !anyFade };
}
// Is this a join onto a clip that had NO sidecar? Its window was
// encoded from the neighbour's pixels, so the two sides are two
// renderings of one moment -- the original file, and the decode of a
// VAE round trip -- and the picture repairs have a real step to work
// on. nodes_assemble.masked_continuation makes the same exception, per
// pin, which is why this reads the pins too: a bridge has two and its
// header carries no single grade.
function tlPixelJoin(L, R) {
    if (!L || !R || L.gap != null || R.gap != null) return false;
    const from = (meta, place, other) => tlPins(meta).some((s) =>
        s.place === place && s.source_kind === "clip_pixels"
        && s.source_id && s.source_id === other?.self_id);
    return from(R.meta, "before", L.meta) || from(L.meta, "after", R.meta);
}
const TL_SEAM_NEEDS = {
    level_lock: "lock",
    level_lock_frames: "lock",
    level_lock_flicker: "lock",
    level_lock_local: "local",
    crossfade: "fade",
    crossfade_frames: "fade",
    audio_declick: "declick",
};

function tlParseSequence(text) {
    const out = [];
    for (const raw of String(text || "").split("\n")) {
        const line = raw.trim();
        if (!tlIsEntryLine(line)) continue;
        const g = line.match(TL_GAP_RE);
        if (g) {
            out.push({ clip: null, gap: Number(g[1]),
                       enterOverride: null, exitOverride: null,
                       seamOpts: {} });
            continue;
        }
        const { head, seam } = tlSplitSeamOpts(line);
        const m = head.match(TL_LINE_RE);
        // seamOpts, NOT seam: `entry.seam` is already the DERIVED seam
        // descriptor ({kind, note}) that drawLinks paints the boundary
        // from. Putting the settings there made every gap boundary look
        // like a link of unknown kind -- TL_COLORS[undefined] is no
        // colour at all, so the pill went transparent and its tooltip
        // read "undefined".
        out.push({ clip: m[1].trim(), gap: null,
                   enterOverride: m[2] === undefined ? null : Number(m[2]),
                   exitOverride: m[3] === undefined ? null : Number(m[3]),
                   seamOpts: seam });
    }
    return out;
}

// Tell ComfyUI's undo tracker that WE just changed the graph.
//
// It captures state on mouseup / keydown / promptQueued / graphCleared
// only -- never on a graph mutation -- and the capture is debounced by
// 50ms. A programmatic edit therefore leaves `activeState` stale: the
// mouseup that precedes a click has already been serviced with the OLD
// graph, so pressing Ctrl+Z straight after (say) deleting empty space
// pops one step too far back. Nudging the tracker after each edit gives
// every action its own undo entry.
function tlNoteEdit() {
    try {
        const wf = app.extensionManager?.workflow?.activeWorkflow
            ?? app.workflowManager?.activeWorkflow;
        const t = wf?.changeTracker;
        // captureCanvasState is the current name; checkState is its
        // deprecated alias, kept for older frontends
        (t?.captureCanvasState ?? t?.checkState)?.call(t);
    } catch (err) {
        /* no tracker (headless, older build): undo is unaffected */
    }
}

// ---- cutting -------------------------------------------------------
// H3 slices latents on a 17-frame group grid, in RAW frames. A cut is
// expressed in DELIVERED frames, so the clip's pinned head offsets it.
// Masked windows live on the shared AV grid (39 + 51k) -- the only
// lengths that are both a whole video run and a whole number of 40 Hz
// audio ticks. 39 was treated as the ONLY one, which quietly capped
// every masked continuation at 1.6s of context; the server has always
// accepted 90 and 141 (nodes_masked.masked_window_ok).
// A bridge has two sides, so it has two modes. `mode` is the
// DEPARTING one (a masked extend; settled) and `mode2` the ARRIVING one
// -- the open question, and the only one worth a button there.
const PIN_ARRIVE_ROLES = new Set(["prepend", "bridge"]);
const arriveModeOf = (st) => (
    st.role === "bridge"
        ? (PIN_MODES.includes(st.mode2) ? st.mode2 : "both")
        : pinModeOf(st));
// How softly the arriving window is held, measured from the JOIN. The
// mask is a strength: core runs a token row at sigma = mask * sigma, so
// 0 is verbatim, 1 is ordinary generation, and between them the model is
// told how much of the destination is already there. All zeros is the
// hard hold these controls replaced.
// A ramp that ends EXACTLY on a latent step offset kills that step: it
// evaluates to 0, counts as held, and drops out of the delivered runway
// -- while the step before it, which now touches the cut, is left at a
// large fraction of `edge`. One frame further and the same step returns
// as barely-bendable (edge / ramp), which costs the cut almost nothing
// and buys the whole step. So every rung here is a step offset PLUS
// ONE, and each strictly dominates the offset it sits next to:
//
//   ramp  runway delivered   mask on the frame meeting the cut
//      6        9 frames     0.067 x edge
//     10       13            0.040
//     14       17            0.029
//     19       22            0.021
//     23       26            0.017
//
// The runway does NOT need to land on an audio tick: moving the cut
// shifts the take's end and the destination's entry by the same amount,
// so it cancels out of the audio equation (nodes_masked.handover_frames
// writes the algebra out). Audio alignment is a RUN LENGTH question.
//
// MASK_RUNWAY is that middle column. It is the same for every window on
// the ladder (39/90/141), because it depends only on the step offsets
// below the ramp, and those are shared. test_ramp_ladder.py regenerates
// it from nodes_masked.handover_frames and fails if the two drift.
const MASK_RAMPS = [0, 6, 10, 14, 19, 23];
const MASK_RUNWAY = { 0: 0, 6: 9, 10: 13, 14: 17, 19: 22, 23: 26 };
// Measured 2026-08-27 on the clip_00024 -> clip_00063 bridge, same
// window and ramp, and it is not monotonic:
//
//   edge 0.00 (hard)  the jump lands on the SEAM              20.3
//   edge 0.25         the jump moves inside the take       13 - 17
//   edge 1.00         ...and grows                            36.4
//
// A free boundary row lets the model draw its OWN content where the
// window begins, which then has to reconcile with the anchored rows
// behind it; the latents there drifted 7x further from the destination
// at edge 1.0 than at 0.25 (max 3.28 vs 0.139). So the useful range is
// the low end, and the ladder brackets it rather than spanning 0..1.
// The H3MCtxPinSpec widget still takes any value if a high one needs
// re-testing.
const MASK_EDGES = [0, 0.1, 0.15, 0.25, 0.4, 0.6];
const MASK_HOLDS = [0, 0.1, 0.2, 0.3, 0.5];

// Help text for the pin selectors. Out here rather than inline because
// renderNextRun rebuilds the row on every strip refresh, and a tooltip
// is a constant, not something to re-concatenate a few times a second.
const MODE_HELP =
    "How the pin specs are created for prepends and extends.\n\n"
    + "both (default): guided keyframes AND mask -- the "
    + "keyframes give the model something to aim at, the mask makes "
    + "the arrival exact. Tested as best probability for "
    + "getting a seamless bridge or prepend.\n\n"
    + "masked: keep the window latents untouched. Used as default for "
    + "extends but is weak for arriving -- with nothing to aim at the model "
    + "wanders, then snaps or hard cuts.\n\n"
    + "guided: hand the window over as keyframes and let it re-render, "
    + "then trim it away. Gives model something to aim at but the frames get "
    + "re-rendered, so the join can flicker.";
const WINDOW_HELP_MASKED =
    "How many frames of the source this run pins.\n\n"
    + "39 is the default. Longer anchors harder, "
    + "but those frames eat into the total clip duration. "
    + "Masking needs the shared audio/video grid, so only 39/90/141.";
const WINDOW_HELP_GUIDED =
    "How many frames of context this run pins.\n\n"
    + "Guided re-renders the window rather than preserving it, so any "
    + "size on the ladder works.";
const RAMP_HELP =
    "How many frames at the start of the window are loosened, so the "
    + "model has room to converge onto the target latents instead of hitting a wall.";
const EDGE_HELP =
    "How loose the first frame of the window is, tightening to 'deep' "
    + "across the ramp.\n\n"
    + "1.0 = allows model to recreate the image freely.\n"
    + "0.0 = keeps the latents exactly as they are.\n\n"
    + "0.40 is the default and did well in tests. "
    + "From testing using 1.0 the model draws its OWN content at the boundary and then "
    + "cannot reconcile it with the frames behind, leading to hard cuts.";
const DEEP_HELP =
    "How loose the mask is through the REST of the window, past the "
    + "ramp.\n\n"
    + "1.0 = allows model to recreate the image freely.\n"
    + "0.0 = keeps the latents exactly as they are.\n\n"
    + "Should be left to the default 0.0 for seamless joins as it lets this run and the "
    + "next clip share identical content at the cut.";
// An ABSENT key is a control never touched, which defaults to the
// measured-best soft hold (ramp 10, edge 0.40 -- clips 83/84, the only
// seamless arrivals; mirrors _parse_pin_state). An explicit 0 the user
// cycled to stays a hard hold.
const maskShapeOf = (st) => ({
    ramp: st.mask_ramp_frames == null ? 10
        : MASK_RAMPS.includes(st.mask_ramp_frames) ? st.mask_ramp_frames : 0,
    edge: st.mask_ramp_edge == null ? 0.4
        : MASK_EDGES.includes(st.mask_ramp_edge) ? st.mask_ramp_edge : 0,
    hold: MASK_HOLDS.includes(st.mask_hold) ? st.mask_hold : 0,
});
const PIN_WINDOWS_MASKED = ["39", "90", "141"];

// The mode follows the ROLE, mirroring nodes_assemble.PIN_MODE_FOR_ROLE.
// An extend continues FORWARD from a fixed window, which masked does
// seamlessly. A prepend has to ARRIVE at a clip that already exists,
// and the only arriving mode that has produced a genuinely seamless
// join is "both" (measured 2026-08-27, clips 83/84): the keyframe rows
// give the model something to steer toward, the mask makes the arrival
// exact. A bridge is one of each: its state carries the before side's
// mode and the server picks per side.
const pinModeForRole = (role) => (role === "prepend" ? "both" : "masked");
// A prepend is the one role where the choice is still open: "both" is
// the default and the one that has converged, but guide/masked stay
// selectable so the pairing can be re-tested rather than taken on
// faith. Extends and bridges are decided by their role and offer no
// mode toggle (a bridge's ARRIVING side has its own, mode2).
const PIN_MODE_CHOOSABLE = new Set(["prepend"]);
const PIN_MODES = ["masked", "guide", "both"];
const pinModeOf = (st) => (
    PIN_MODE_CHOOSABLE.has(st.role) && PIN_MODES.includes(st.mode)
        ? st.mode : pinModeForRole(st.role));
// "both" masks the window as well, so it obeys the shared AV grid
const pinModeMasks = (mode) => mode === "masked" || mode === "both";
const maskedWindow = (w) => (
    PIN_WINDOWS_MASKED.includes(String(w)) ? String(w) : "39");

const TL_GROUP = 17;

// Which residue the cut's RAW frame must hit depends on the side, and
// it is not always 0. A pin's slice must START on the grid:
//   tail cut -> seeds an EXTEND, whose window ends at the cut, so
//               raw_start = ph + cut - W  =>  (ph + cut) === W (mod 17)
//   head cut -> seeds a PREPEND, whose window starts at the cut, so
//               raw_start = ph + cut      =>  (ph + cut) === 0 (mod 17)
// Every H3 window (5/22/39/56) is 5 mod 17, so the tail phase is 5 for
// all of them. Snapping both sides to 0 -- as this first did -- left
// every tail cut 12 frames off, which the server then shifted silently.
const TL_WINDOW_PHASE = 5;
const tlCutPhase = (side) => (side === "exit" ? TL_WINDOW_PHASE : 0);

function tlSnapCut(frame, meta, snap, side) {
    const f = Math.max(0, Math.round(Number(frame) || 0));
    if (!snap) return f;
    const ph = Number(meta?.pinned_head_frames ?? 0) || 0;
    const phase = tlCutPhase(side);
    // snap (raw - phase) to the grid, then put the phase back
    let raw = Math.round((ph + f - phase) / TL_GROUP) * TL_GROUP + phase;
    // Clamping the DELIVERED frame at 0 would hand back an off-grid cut
    // for any continuation (its delivered 0 is raw `ph`), and Apply
    // refuses those. Clamp in RAW space, to the first real stop.
    while (raw < ph) raw += TL_GROUP;
    return raw - ph;
}

// The delivered frames a cut may legally land on, for the tick marks
// drawn while dragging. Empty when snapping is off (anywhere is legal).
function tlCutStops(meta, frames, snap, side) {
    if (!snap || !frames) return [];
    const ph = Number(meta?.pinned_head_frames ?? 0) || 0;
    const phase = tlCutPhase(side);
    const out = [];
    let raw = Math.ceil((ph - phase) / TL_GROUP) * TL_GROUP + phase;
    while (raw < ph) raw += TL_GROUP;
    for (; raw - ph <= frames; raw += TL_GROUP) out.push(raw - ph);
    return out;
}

// Which side of which clip a seam may cut. A clip in the MIDDLE of a
// chain is off limits on the linked side: its seam belongs to the
// lineage, and moving it would contradict what the sidecars record.
//   side "exit"  -> entries[i]'s tail
//   side "enter" -> entries[i]'s head
function tlCanCut(entries, i, side, linked) {
    const e = entries?.[i];
    if (!e || e.gap != null) return false;
    if (side === "exit") {
        const right = entries[i + 1];
        if (!right || right.gap != null) return true;      // chain end
        return !linked(e.meta, right.meta);
    }
    const left = entries[i - 1];
    if (!left || left.gap != null) return true;            // chain start
    return !linked(left.meta, e.meta);
}

// The clip that currently OCCUPIES the gap a pin regenerates, or null.
//
// Lineage alone does not answer this: a clip keeps its lineage wherever
// you drag it, so "C extends B" stays true after reordering to A-C-B
// even though C no longer sits after B and nothing would replace it.
// Occupancy is positional AND live: the immediate neighbour on that
// side, with the link actually joining them there.
//   side "after"  -> the clip following `sourceClip` (an extend)
//   side "before" -> the clip preceding it (a prepend)
function tlOccupant(entries, sourceClip, side, linked) {
    const list = entries ?? [];
    const k = list.findIndex((e) => e.clip === sourceClip);
    if (k < 0) return null;
    const j = side === "after" ? k + 1 : k - 1;
    const src = list[k], occ = list[j];
    if (!occ || occ.gap != null || !occ.meta || !src?.meta) return null;
    if (occ.clip === sourceClip) return null;
    const live = side === "after"
        ? linked(src.meta, occ.meta) : linked(occ.meta, src.meta);
    return live ? { idx: j, entry: occ } : null;
}

function tlFormatLine(e) {
    if (e.gap != null) return `~ ${Math.max(1, Math.round(e.gap))}`;
    const a = e.enterOverride, b = e.exitOverride;
    const t = tlFormatSeamOpts(e.seamOpts);
    if (a == null && b == null) return `${e.clip}${t}`;
    if (b == null) return `${e.clip} @ ${a}${t}`;
    if (a == null) return `${e.clip} @ ..${b}${t}`;
    return `${e.clip} @ ${a}..${b}${t}`;
}

// mirrors nodes_assemble._derive_seam; returns cuts for the RIGHT clip's
// enter and the LEFT clip's exit, plus a display verdict
// Lineage accessors that understand MULTI-PIN takes: relation/parent_id
// are the single-pin fast-path summary; a bridge take (relation "")
// carries its two parents only in the pins recipe, so the accessors
// fall through to it. Every lineage consumer (seams, links, aiming)
// goes through these.
function tlPins(meta) {
    try {
        const p = JSON.parse(meta?.pins || "[]");
        return Array.isArray(p) ? p : [];
    } catch (err) {
        return [];
    }
}
function tlExtendParent(meta) {
    // {id, join} of the clip `meta` EXTENDS (its before-context)
    if (!meta) return null;
    if (meta.relation === "extends") {
        return { id: meta.parent_id,
                 join: Number(meta.parent_join_frame) || 0 };
    }
    if (meta.relation) return null;
    // join = source_start + the window's HANDOVER (nodes_assemble.
    // _extend_parent). For a hard hold that is the window's end,
    // source_start + source_frames, which is all this read until
    // 2026-09-16 -- so a bridge arriving on a soft hold played its
    // seam at the wrong frame in quick preview while the build (which
    // reads the handover) had it right.
    const p = tlPins(meta).find((s) => s.place === "before" &&
        s.source_kind === "clip" && s.source_id);
    return p ? { id: p.source_id,
                 join: (Number(p.source_start) || 0) +
                       specHandover(p, "before") } : null;
}
function tlPrependChild(meta) {
    // {id, join} of the clip `meta` PREPENDS INTO (its after-context)
    if (!meta) return null;
    if (meta.relation === "prepends") {
        return { id: meta.parent_id,
                 join: Number(meta.parent_join_frame) || 0 };
    }
    if (meta.relation) return null;
    // join = source_start + handover (nodes_assemble._prepend_child);
    // a hard hold hands over at 0, a soft one past its ramped frames
    const p = tlPins(meta).find((s) => s.place === "after" &&
        s.source_kind === "clip" && s.source_id);
    return p ? { id: p.source_id,
                 join: (Number(p.source_start) || 0) +
                       specHandover(p, "after") } : null;
}
function tlDeriveSeam(lh, rh) {
    if (!lh || !rh) return { exitL: null, enterR: 0, kind: "butt",
                             note: "no sidecar data: plays back-to-back" };
    const num = (m, k) => Number(m[k] ?? 0) || 0;
    // Every real lineage link is GREEN: a trim-point/auto-shifted join
    // is still a correct, intentional join (the note keeps the exact
    // cut frame). Amber in the RESULT PREVIEW means measured motion
    // quality -- a different axis; coloring geometry cuts amber here
    // read as a quality warning and confused the two.
    const ex = tlExtendParent(rh);
    if (ex && ex.id === lh.self_id) {
        const exitF = ex.join - num(lh, "pinned_head_frames");
        const delivered = num(lh, "delivered_frames");
        const clean = exitF === delivered;
        return { exitL: clean ? null : exitF, enterR: 0,
                 kind: "seamless",
                 note: clean ? "extends: seamless"
                     : `extends at a cut: exits at frame ${exitF}` };
    }
    const pr = tlPrependChild(lh);
    if (pr && pr.id === rh.self_id) {
        const enter = pr.join - num(rh, "pinned_head_frames");
        return { exitL: null, enterR: enter,
                 kind: "seamless",
                 note: enter === 0 ? "prepends: seamless"
                     : `prepends: enters at frame ${enter}` };
    }
    return { exitL: null, enterR: 0, kind: "butt",
             note: "no recorded relation: plays back-to-back" };
}

// duration of clips WITHOUT a sidecar: ask the browser for just the
// container metadata (no frame data is fetched with preload="metadata")
const tlFramesCache = new Map();
function tlProbeClipFrames(clip) {
    if (!tlFramesCache.has(clip)) {
        tlFramesCache.set(clip, new Promise((resolve) => {
            const v = document.createElement("video");
            v.preload = "metadata";
            v.onloadedmetadata = () =>
                resolve(Math.round(v.duration * TL_FPS) || 0);
            v.onerror = () => resolve(0);
            v.src = api.apiURL(viewRoute(clip));
        }));
    }
    return tlFramesCache.get(clip);
}

// dimensions probes for the drop resolution gate: chained takes must
// share resolution (latents are unresizable and assemble refuses
// mismatches), so a drop that can't match is rejected before any
// upload. Sidecar header when available (exact), else metadata-only
// <video> loads.
function tlProbeDimsUrl(url) {
    return new Promise((resolve) => {
        const v = document.createElement("video");
        v.preload = "metadata";
        v.onloadedmetadata = () => resolve(
            v.videoWidth && v.videoHeight
                ? [v.videoWidth, v.videoHeight] : null);
        v.onerror = () => resolve(null);
        v.src = url;
    });
}
async function tlClipDims(clip) {
    const meta = await tlClipMeta(clip);
    const w = Number(meta?.width);
    const h = Number(meta?.height);
    if (w && h) return [w, h];
    return tlProbeDimsUrl(api.apiURL(viewRoute(clip)));
}
async function tlFileDims(file) {
    const url = URL.createObjectURL(file);
    try {
        return await tlProbeDimsUrl(url);
    } finally {
        URL.revokeObjectURL(url);
    }
}

const TL_BUILD = "v7-panel-probe";

// The color of an actually-VISIBLE UI panel beats the legacy CSS var:
// modern Comfy paints its menus with different tokens, and a user
// palette can leave --comfy-menu-bg near-black while the real panels
// are lighter. Sample rendered panels first, fall back to the var.
function tlPanelColor() {
    const sels = [".comfyui-menu", ".side-tool-bar-container",
                  ".comfyui-body-top", ".p-menubar", ".actionbar"];
    for (const sel of sels) {
        const el = document.querySelector(sel);
        if (!el) continue;
        const bg = getComputedStyle(el).backgroundColor;
        if (bg && bg !== "transparent" &&
            !/rgba\(\s*\d+,\s*\d+,\s*\d+,\s*0\)/.test(bg)) {
            return { color: bg, source: sel };
        }
    }
    return { color: tlCssVar("--comfy-menu-bg", "#353535"),
             source: "--comfy-menu-bg" };
}

function tlCssVar(name, fallback) {
    try {
        const v = getComputedStyle(document.documentElement)
            .getPropertyValue(name).trim();
        return v || fallback;
    } catch {
        return fallback;
    }
}

// darken a CSS color by scaling its channels; plain rgb()/rgba() out so
// every engine applies it (fancy color functions get silently rejected
// by older CSSOMs, which leaves the property unset entirely)
function tlDarken(color, f, off) {
    let r, g, b, a = null;
    let m = color.match(/^#([0-9a-f]{3})$/i);
    if (m) {
        [r, g, b] = [...m[1]].map((c) => parseInt(c + c, 16));
    } else if ((m = color.match(/^#([0-9a-f]{6})([0-9a-f]{2})?$/i))) {
        r = parseInt(m[1].slice(0, 2), 16);
        g = parseInt(m[1].slice(2, 4), 16);
        b = parseInt(m[1].slice(4, 6), 16);
        if (m[2]) a = parseInt(m[2], 16) / 255;
    } else if ((m = color.match(/^rgba?\(([^)]+)\)$/i))) {
        const parts = m[1].split(/[,/\s]+/).filter(Boolean).map(Number);
        [r, g, b] = parts;
        if (parts.length > 3) a = parts[3];
        if ([r, g, b].some((x) => !Number.isFinite(x))) return color;
    } else {
        return color; // unknown notation: leave untouched
    }
    const sc = (x) =>
        Math.round(Math.max(0, Math.min(255, x * f + (off || 0))));
    return a === null
        ? `rgb(${sc(r)}, ${sc(g)}, ${sc(b)})`
        : `rgba(${sc(r)}, ${sc(g)}, ${sc(b)}, ${a})`;
}

function tlAlpha(color, a) {
    const c = tlDarken(color, 1, 0); // normalizes to rgb()/rgba()
    const m = c.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    return m ? `rgba(${m[1]}, ${m[2]}, ${m[3]}, ${a})` : c;
}

export function themePalette() {
    let light = false;
    try {
        const bg = getComputedStyle(document.body)
            .backgroundColor.match(/\d+/g);
        if (bg) {
            const [r, g, b] = bg.map(Number);
            light = 0.2126 * r + 0.7152 * g + 0.0722 * b > 128;
        }
    } catch { /* default dark */ }
    // dark chrome with light text in BOTH themes, but
    // tuned per ground: lighter on a light page, darker
    // on a dark one; buttons/picker share it
    if (light) {
        // same theme-variable derivation as dark mode:
        // surfaces follow the palette's input color, with
        // dark ink and darker mixes for edges/panel
        const lb = "var(--comfy-input-bg, #dfe2e8)";
        return {
            light: true,
            rest: lb, text: "#23262b", sub: "#5c6270",
            // rest, but resolved to a real color: canvas fillStyle
            // cannot read a var() string (it silently keeps the
            // previous fill), so canvas-drawn chrome uses this one
            restRgb: tlCssVar("--comfy-input-bg", "#dfe2e8"),
            active: "#3557b0", activeText: "#f2f4f8",
            edge: tlDarken(tlCssVar(
                "--comfy-input-bg", "#dfe2e8"), 0.76),
            tick: "#6a707a", tickLine: "#b6bac3",
            // the theme's own panel color, not a
            // black-mix (mixing black into a light color
            // makes mud, not shade)
            stripBg: tlAlpha(tlCssVar(
                "--comfy-input-bg", "#dfe2e8"), 0.25),
            drop: "#454b55",
        };
    }
    // dark: surfaces follow the theme's input color; the
    // derived shades (edge/panel) are computed in JS at
    // palette time, not with CSS color functions
    const base = "var(--comfy-input-bg, #17191f)";
    return {
        light: false,
        rest: base, text: "#e8eaee", sub: "#9aa1ac",
        // see the light branch: the canvas-safe form of rest
        restRgb: tlCssVar("--comfy-input-bg", "#17191f"),
        active: "#1f4390", activeText: "#e8eaee",
        // Darker than the ground, not lighter: the outline
        // reads as a seam between surfaces rather than a drawn
        // line. 0.6x is the middle of that direction's range --
        // 0.3x read nearly black, 1x vanishes (user-tuned
        // 2026-08-20).
        edge: tlDarken(tlCssVar(
            "--comfy-input-bg", "#17191f"), 0.6),
        tick: "#9aa1ac", tickLine: "#3a3f48",
        // the clip bars' surface at half opacity; the
        // node body blends through
        stripBg: tlAlpha(
            tlCssVar("--comfy-input-bg", "#17191f"), 0.25),
        drop: "#d7dbe2",
    };
}

const tlMetaCache = new Map();
async function tlClipMeta(clip) {
    if (!tlMetaCache.has(clip)) {
        let m = null;
        // Told apart deliberately: readSidecarMeta returns null when
        // there IS no sidecar (a plain video -- a settled answer worth
        // remembering) and THROWS when one is there but could not be
        // read (a stale cached range, a half-written file, a transient
        // error). Remembering the second as "no lineage" strands every
        // join to this clip as a butt cut for the rest of the session,
        // which no amount of reloading the strip can undo.
        let unsure = false;
        try {
            m = await readSidecarMeta(sidecarValue(clip));
        } catch { unsure = true; }
        // No sidecar (or a broken one): ask the server for the clip's
        // identity instead. The browser cannot work this out itself --
        // hashing means pulling the whole file across -- and without it
        // a plain video has no self_id, so nothing can name it as a
        // parent and every join to it degrades to a butt cut.
        if (!m) {
            try {
                const resp = await api.fetchApi("/obvpm/h3/clip_meta", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ clip }),
                });
                if (resp.ok) {
                    const j = await resp.json();
                    if (j && j.self_id) m = j;
                }
            } catch { /* offline/older server: butt joins, as before */ }
        }
        if (m || !unsure) tlMetaCache.set(clip, m);
        return m;
    }
    return tlMetaCache.get(clip);
}

async function tlFetchClipList() {
    const resp = await api.fetchApi(
        `/object_info/${encodeURIComponent(TL_PICKER_CLASS)}`);
    if (!resp.ok) return [];
    const info = await resp.json();
    return info?.[TL_PICKER_CLASS]?.input?.required?.clip?.[0] ?? [];
}

app.registerExtension({
    name: "obvpm.h3_mctx_timeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TL_NODE) return;
        tldbg("registering for", nodeData.name);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                return buildTimeline(this, result);
            } catch (err) {
                console.error("[obvpm-h3-timeline] widget build FAILED:",
                              err);
                return result;
            }
        };

        function buildTimeline(node, result) {
            const seqWidget = node.widgets?.find(
                (w) => w.name === "sequence");
            if (!seqWidget) {
                tldbg("node", node.id, "has no 'sequence' widget; widgets:",
                      node.widgets?.map((w) => w.name));
                return result;
            }
            tldbg("building timeline widget on node", node.id);

            const container = document.createElement("div");
            Object.assign(container.style, {
                display: "flex", flexDirection: "column", gap: "6px",
                font: "12px sans-serif", overflow: "hidden",
                position: "relative",
            });

            // ---- header: clip picker + preview control ----------------
            const header = document.createElement("div");
            Object.assign(header.style,
                { display: "flex", gap: "6px", alignItems: "center" });
            const picker = document.createElement("select");
            Object.assign(picker.style, {
                flex: "1", minWidth: "0", background: "#2a2e36",
                color: "#e8eaee", border: "1px solid #3a3f48",
                borderRadius: "4px", padding: "2px 4px",
            });
            const mkBtn = (label, title) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.title = title;
                Object.assign(b.style, {
                    background: "#2a2e36", color: "#e8eaee",
                    border: "1px solid #3a3f48", borderRadius: "4px",
                    padding: "2px 8px", cursor: "pointer",
                    whiteSpace: "nowrap",
                    // explicit line-height: glyph labels (▶ ⇪ ✎) carry
                    // different intrinsic heights, so without it the
                    // header buttons come out unequal
                    font: "12px/18px sans-serif",
                });
                return b;
            };
            const addBtn = mkBtn("+ add",
                "Add a clip to the end of the timeline (opens a menu)");
            const editBtn = mkBtn("✎",
                "Edit the sequence as text (one clip per line; '# ' "
                + "comments; ' @ N' forces a clip to enter at frame N)");
            const quickBtn = mkBtn("▶ quick",
                "Instant per-clip preview (near-gapless hops)");
            const fullBtn = mkBtn("▶ full",
                "Build the real cut into a temp file and play it "
                + "seamlessly -- identical content to the export");
            quickBtn.style.minWidth = "90px";
            fullBtn.style.minWidth = "90px";
            // View controls. They change nothing about the cut, so
            // they sit apart from the editing buttons; zoom is what you
            // reach for now that blocks have no minimum width.
            const zoomOutBtn = mkBtn("−",
                "Zoom out (or wheel over the ruler)");
            const zoomInBtn = mkBtn("+",
                "Zoom in (or wheel over the ruler)");
            const fitBtn = mkBtn("fit",
                "Set the scale so the whole sequence fits, once. It "
                + "stays there afterwards -- editing does not re-fit.");
            for (const b of [zoomOutBtn, zoomInBtn]) {
                b.style.padding = "2px 7px";
                b.style.font = "13px/18px sans-serif";
            }
            const exportBtn = mkBtn("export",
                "Write the current cut to the output folder under "
                + "filename_prefix (byte-identical to the full preview; "
                + "rebuilds first if the preview is stale)");
            const stateChip = document.createElement("span");
            Object.assign(stateChip.style, {
                font: "12px sans-serif", color: "#9aa1ac",
                whiteSpace: "nowrap",
            });
            // one header row: playback on the left, editing (+ add =
            // clip menu, ✎ = raw text) and export grouped at the right.
            // The <select> picker survives UNATTACHED as the live
            // clip-list data store (add menu, seam popups, detection).
            // Cut snapping is a mode you flip while working the strip,
            // so it rides the toolbar rather than sitting among the
            // node's setup widgets (the widget itself stays declared and
            // hidden -- widgets_values is positional).
            const snapWidget = node.widgets?.find(
                (w) => w.name === "snap_cuts_to_grid");
            if (snapWidget) {
                snapWidget.hidden = true;
                (snapWidget.options ??= {}).hidden = true;
            }
            const snapBtn = mkBtn("snap cuts", "");
            function snapOn() { return snapWidget?.value !== false; }
            function paintSnap() {
                const on = snapOn();
                // Worn like the prepend/extend toggles: filled when on,
                // bare frame when off. themePalette() is read here rather
                // than through PAL, which is declared further down and
                // would be in its temporal dead zone at build time.
                const P = themePalette();
                snapBtn.style.background = on ? TL_ROLE.extend.bg : P.rest;
                snapBtn.style.borderColor = on ? "transparent" : P.edge;
                snapBtn.style.color = on ? "#fff" : P.text;
                snapBtn.title = on
                    ? "Cuts snap to H3's 17-frame latent grid (~0.7s "
                      + "steps), so extending or prepending from a cut is "
                      + "always exact. Click for frame-accurate cuts."
                    : "Cuts are frame-accurate. A pin taken from an "
                      + "off-grid cut is shifted to the nearest "
                      + "latent-grade frame. Click to snap instead.";
            }
            snapBtn.addEventListener("mouseenter", () => {
                snapBtn.style.background = snapOn()
                    ? TL_ROLE.extend.hover : "rgba(127,127,127,0.3)";
            });
            snapBtn.addEventListener("mouseleave", paintSnap);
            snapBtn.addEventListener("click", () => {
                if (!snapWidget) return;
                snapWidget.value = !snapOn();
                paintSnap();
                node.setDirtyCanvas?.(true, true);
                void refresh();     // the cut ticks follow the mode
            });
            paintSnap();

            // ---- next run's DURATION -----------------------------------
            // The widget takes seconds, because that is the unit a cut is
            // thought in; the `length` output is frames, because that is
            // what the model takes. The two are not freely convertible --
            // a run sits on the 17k+5 ladder, and one whose arrival masks
            // must sit on the shared AV grid or its audio join clicks --
            // so the readout shows the frames that will actually be
            // emitted and the steppers move a whole rung at a time.
            // Mirrors H3Timeline._run_length; the node is authoritative.
            const RUN_LADDER = 17, RUN_BASE = 5;
            const AV_STEP = 51, AV_BASE = 39;
            const durWidget = node.widgets?.find(
                (w) => w.name === "duration_seconds");
            // `st` defaults to the stored pin; the gap fit asks about a
            // pin that is being armed, before it is the stored one
            function durMasks(st = pinState()) {
                if (!st) return false;
                const arriving = st.role === "bridge"
                    ? arriveModeOf(st) : pinModeOf(st);
                return pinModeMasks(arriving) || pinModeMasks(pinModeOf(st));
            }
            function durFrames(secs = durWidget?.value ?? 8, st = pinState()) {
                const want = Math.max(1, Number(secs) || 0) * TL_FPS;
                if (durMasks(st)) {
                    const lo = AV_BASE + Math.max(0, Math.floor(
                        (want - AV_BASE) / AV_STEP)) * AV_STEP;
                    return want <= AV_BASE ? AV_BASE
                        : (want - lo) * 2 < AV_STEP ? lo : lo + AV_STEP;
                }
                const n = Math.max(RUN_BASE, Math.round(want));
                const lo = n - ((n - RUN_BASE) % RUN_LADDER);
                return (n - lo) * 2 < RUN_LADDER ? lo : lo + RUN_LADDER;
            }
            // duration_seconds is a REAL widget, so it can be typed on the
            // node's face or driven by a wire. This box is a second view
            // of the same value, put where the decision is made -- and it
            // goes read-only when something upstream owns the value,
            // because a box you can type into that is then overwritten at
            // run time is worse than no box at all.
            function durWired() {
                return (node.inputs ?? []).some(
                    (s) => (s.widget?.name === "duration_seconds"
                            || s.name === "duration_seconds")
                        && s.link != null);
            }
            function durSeconds() {
                if (durWired()) {
                    // traced up the wire when it can be; null when the
                    // chain is not readable from here
                    const v = rpWidgetValue(node, "duration_seconds");
                    if (v != null && v !== "" && isFinite(Number(v))) {
                        return Number(v);
                    }
                    return null;
                }
                return Number(durWidget?.value ?? 8);
            }
            const durIn = document.createElement("input");
            durIn.type = "text";
            durIn.inputMode = "decimal";
            Object.assign(durIn.style, {
                width: "48px", flexShrink: "0", textAlign: "right",
                borderRadius: "4px", borderStyle: "solid",
                borderWidth: "1px",
                padding: "0 4px", font: "11px/16px sans-serif",
                height: "18px", boxSizing: "border-box",
            });
            // Colours are applied in paintDur, not here: PAL is declared
            // further down and would be in its temporal dead zone at
            // build time (same reason snapBtn reads themePalette()).
            // Painting them also keeps the box theme-live.
            const durOut = document.createElement("span");
            Object.assign(durOut.style, {
                flexShrink: "0", whiteSpace: "nowrap",
                font: "11px/16px sans-serif",
            });
            // What a run of `f` frames is actually made of. The snapped
            // length is what gets GENERATED; the pinned windows at one or
            // both ends are scaffolding that gets trimmed, so the clip
            // this run adds to the timeline is shorter -- sometimes by
            // more than half. Both numbers matter and they are not the
            // same question, so the row shows both.
            //
            // Mirrors nodes_pins.pins_trim_totals: a departing (before)
            // pin is a hard masked extend and trims its whole window; an
            // arriving (after) pin trims all but its runway, because the
            // take delivers the ramped frames itself.
            function durPlan(f, st = pinState()) {
                if (!st || f == null) return null;
                const arriving = st.role === "bridge"
                    ? arriveModeOf(st) : pinModeOf(st);
                const win = Number(pinModeMasks(pinModeOf(st))
                    ? maskedWindow(st.window) : (st.window ?? 22));
                const sh = maskShapeOf(st);
                const runway = (PIN_ARRIVE_ROLES.has(st.role)
                                && pinModeMasks(arriving))
                    ? (MASK_RUNWAY[sh.ramp] ?? 0) : 0;
                const head = (st.role === "extend" || st.role === "bridge")
                    ? win : 0;
                const tailWin = (st.role === "prepend"
                                 || st.role === "bridge") ? win : 0;
                const tail = tailWin ? tailWin - runway : 0;
                return { win, head, tail, tailWin, runway, st,
                         kept: f - head - tail };
            }
            const secTxt = (fr) => (fr / TL_FPS).toFixed(2) + "s";
            // below this much KEPT footage the readout warns: a run that
            // pins both ends and keeps under three seconds is a
            // window-and-a-bit of scaffolding around very little
            const DUR_SHORT_FRAMES = 3 * TL_FPS;
            const DUR_WARN = "#e09a2b";      // amber, legible on both grounds
            // The hover: what the run is made of, laid out on ITS OWN
            // timeline. The question this answers is a prompting one --
            // the prompt describes the whole generated run, pinned
            // scaffolding included, but only the kept span reaches the
            // timeline, so you need both spans and where they sit.
            function durBreakdown(f, secs, wired, masks, plan) {
                const name = (p) => String(p || "").split("/").pop();
                const span = (a, b) => secTxt(a) + "-" + secTxt(b);
                const L = [];
                L.push((wired ? "Driven by a wire: " : "Asked ")
                       + Number(secs).toFixed(2) + "s -> generates "
                       + f + " frames (" + (f / TL_FPS).toFixed(3)
                       + "s at " + TL_FPS + " fps).");
                if (wired) {
                    L.push("Change it at the source, or disconnect the "
                           + "input to type it here.");
                }
                L.push("");
                if (!plan || (!plan.head && !plan.tail)) {
                    L.push("No pin, so the whole run is delivered.");
                } else {
                    const st = plan.st;
                    L.push("What the run is made of:");
                    if (plan.head) {
                        L.push("  " + span(0, plan.head) + "  " + plan.head
                               + "f pinned from " + name(st.source)
                               + " (context; trimmed off)");
                    }
                    const freeTo = f - plan.tailWin;
                    L.push("  " + span(plan.head, freeTo) + "  "
                           + (freeTo - plan.head) + "f GENERATED FREELY");
                    if (plan.tailWin) {
                        L.push("  " + span(freeTo, f) + "  " + plan.tailWin
                               + "f pinned into "
                               + name(st.role === "bridge"
                                      ? st.source2 : st.source)
                               + (plan.runway
                                  ? " (the first " + plan.runway
                                    + "f of it are delivered)"
                                  : " (trimmed off)"));
                    }
                    L.push("");
                    L.push("KEPT: " + span(plan.head, f - plan.tail)
                           + " = " + plan.kept + "f ("
                           + secTxt(plan.kept) + "). That is the clip this "
                           + "run adds to the timeline.");
                    if (plan.kept <= 0) {
                        L.push("!! The pins leave nothing -- lengthen the "
                               + "run or shrink the window.");
                    } else if (plan.kept < DUR_SHORT_FRAMES) {
                        L.push("!! Only " + secTxt(plan.kept) + " of new "
                               + "footage survives the pins (under "
                               + (DUR_SHORT_FRAMES / TL_FPS) + "s). A "
                               + "bridge or loop pins BOTH ends, so a short "
                               + "run is mostly scaffolding -- lengthen the "
                               + "run or shrink the window.");
                    }
                    L.push("");
                    L.push("PROMPT FOR THE WHOLE " + (f / TL_FPS).toFixed(2)
                           + "s, not just the kept part: the model sees "
                           + "the pinned frames as context and generates "
                           + "across all of it. Action you want in the "
                           + "kept span should be described as happening "
                           + "after the first " + secTxt(plan.head) + ".");
                }
                L.push("");
                L.push(masks
                       ? "Length is on the shared AV grid "
                         + "(39/90/141/192/243 frames). A masked or both "
                         + "arrival needs it: only those lengths have "
                         + "whole audio ticks, and off them the join "
                         + "clicks ~8ms however it is cut."
                       : "Length is on the ordinary 17k+5 ladder (~0.71s "
                         + "steps). Nothing here masks audio, so the "
                         + "coarser AV grid is not needed.");
                return L.join("\n");
            }
            function paintDur() {
                if (!durWidget) return;
                const secs = durSeconds();
                const wired = durWired();
                const f = secs == null ? null : durFrames(secs);
                const masks = durMasks();
                const P = themePalette();
                // WIRED: no box at all. An input you cannot type into is
                // a control that lies about being one, so the whole
                // thing becomes static text and the row gets shorter.
                durIn.style.display = wired ? "none" : "";
                if (!wired && document.activeElement !== durIn) {
                    durIn.value = secs == null ? "" : String(
                        Number(secs.toFixed(3)));
                }
                durIn.disabled = wired;
                // the same surface the dropdowns beside it wear
                durIn.style.background = P.rest;
                durIn.style.color = P.text;
                durIn.style.borderColor = P.edge;
                durIn.title =
                    "How long the NEXT run should be, in seconds. The "
                    + "frame count beside it is what the `length` output "
                    + "will actually emit: runs live on a grid, so this "
                    + "is rounded to the nearest legal length.";
                const plan = durPlan(f);
                // a bridge or loop pins BOTH ends: with a short run most
                // of it is scaffolding and the clip that reaches the
                // timeline is a fraction of what was asked. Say so where
                // the length is set, not after the run.
                const short = !!(plan && (plan.head || plan.tail)
                                 && plan.kept < DUR_SHORT_FRAMES);
                const snapped = f == null ? null
                    : (f / TL_FPS).toFixed(2) + "s · " + f + "f"
                      // the length that ends up on the timeline, when a
                      // pin makes it differ from the length generated
                      + (plan && (plan.head || plan.tail)
                         ? " · keeps " + secTxt(plan.kept)
                           + (short ? " ⚠" : "") : "");
                durOut.textContent = wired
                    ? (secs == null ? "(driven)"
                       : Number(secs.toFixed(3)) + "s → " + snapped)
                    : (snapped == null ? "s → (driven)" : "s → " + snapped);
                durOut.style.color = f == null ? P.sub
                    : short ? DUR_WARN
                    : (masks ? P.text : P.sub);
                durOut.style.fontWeight = short ? "600" : "";
                durOut.title = f == null
                    ? "The wired value cannot be read from here, so the "
                      + "snapped length is only known at run time."
                    : durBreakdown(f, secs, wired, masks, plan);
                // the landing band on the ruler is as long as the kept
                // span, so a new length moves it. Next frame, not now:
                // this also runs while the widget is still being built,
                // before the ruler and the strip exist
                requestAnimationFrame(() => {
                    try { drawRuler(); } catch (err) { /* pre-build */ }
                });
            }
            function commitDur() {
                if (!durWidget || durWired()) return;
                const v = parseFloat(durIn.value);
                if (isFinite(v) && v > 0) {
                    durWidget.value = Math.min(150, Math.max(0.2, v));
                }
                paintDur();
                node.setDirtyCanvas?.(true, true);
            }
            durIn.addEventListener("change", commitDur);
            durIn.addEventListener("blur", commitDur);
            durIn.addEventListener("keydown", (ev) => {
                ev.stopPropagation();      // the canvas eats keys otherwise
                if (ev.key === "Enter") durIn.blur();
            });
            durIn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
            // One element so renderNextRun can place both halves after
            // the role pill with a single append, and neither can be
            // separated from the other by a reflow.
            const durWrap = document.createElement("span");
            Object.assign(durWrap.style, {
                display: "inline-flex", alignItems: "center", gap: "3px",
                flexShrink: "0",
            });
            durWrap.append(durIn, durOut);
            if (durWidget) {
                const durCb = durWidget.callback;
                durWidget.callback = function (...args) {
                    const r = durCb?.apply(this, args);
                    paintDur();
                    return r;
                };
            }
            // A DRIVEN duration changes with nothing here to hear it: the
            // widget callback never fires (the value is not typed), and
            // refresh() only runs when the strip changes. Two ways in:
            //
            //   connecting or cutting the wire -- onConnectionsChange,
            //   which is the moment the box has to switch between
            //   editable and read-only;
            //   the upstream value moving while connected -- nothing
            //   announces that, so poll, cheaply and only while wired.
            //
            // The poll re-paints only when the traced value actually
            // changed, so a wired-but-static graph costs one short walk
            // (RP_HOPS deep) every 400ms and no DOM work at all.
            const onConn = node.onConnectionsChange;
            node.onConnectionsChange = function (...args) {
                const r = onConn?.apply(this, args);
                paintDur();
                return r;
            };
            let lastDriven = null;
            const durPoll = setInterval(() => {
                if (!durWidget || !durWired()) return;
                const v = String(durSeconds());
                if (v === lastDriven) return;
                lastDriven = v;
                paintDur();
            }, 400);
            // NOT painted here: durMasks() reads pinState(), whose
            // psWidget is declared further down and would still be in its
            // temporal dead zone. refresh() paints it instead -- which is
            // where it belongs anyway, since switching a pin's mode
            // switches which grid the duration snaps to.

            // ---- seam improvement -------------------------------------
            // Six repair settings, hidden from the node's face and edited
            // in a dialog: they are set-and-forget policy rather than
            // per-cut controls, and on the node they crowded out the
            // widgets people actually reach for. The widgets stay
            // DECLARED and hidden -- widgets_values is positional.
            const SEAM_WIDGETS = [
                "level_lock", "level_lock_frames", "level_lock_flicker",
                "crossfade", "crossfade_frames", "audio_declick",
                "level_lock_local",
            ];
            for (const name of SEAM_WIDGETS) {
                const w = node.widgets?.find((x) => x.name === name);
                if (!w) continue;
                w.hidden = true;
                (w.options ??= {}).hidden = true;
            }
            const seamBtn = mkBtn("⚙ audio",
                "Seam audio: how the sound is handled where one clip "
                + "meets the next. The de-click tapers 5ms each side of "
                + "every join that is not blended, which on a masked or "
                + "both-mode sequence is all of them.");
            let seamDialogClose = null;
            function seamSettingsChanged() {
                // the built preview was made under the OLD settings, so
                // it is no longer what these controls describe
                single = null;
                usingSingle = false;
                seamFixSig = null;          // re-measure: the span may differ
                updateModeButtons();
                tlNoteEdit();               // one undo step per change
                node.setDirtyCanvas?.(true, true);
                void refresh();
            }
            const seamWidgetValue = (name) => node.widgets
                ?.find((w) => w.name === name)?.value;
            const setSeamWidget = (name, value) => {
                const w = node.widgets?.find((x) => x.name === name);
                if (w) w.value = value;
            };
            // The full editor (tlSeamDialog: level lock, crossfade) is
            // no longer reachable from here. Its controls act on guided
            // joins only, and guided mode is withheld (SHOW_GUIDED); a
            // join that already carries overrides still opens it from
            // its seam popup.
            function openAudioSeamDialog() {
                seamDialogClose = tlAudioSeamDialog({
                    get: seamWidgetValue,
                    set: setSeamWidget,
                    seams: seamFix,
                    onChange: seamSettingsChanged,
                    onClose: () => { seamDialogClose = null; },
                });
            }
            seamBtn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                if (seamDialogClose) {      // already open: toggle it shut
                    seamDialogClose();
                    seamDialogClose = null;
                    return;
                }
                openAudioSeamDialog();
            });
            // centred: auto margins on both the toggle and the right
            // group push it into the middle of the row
            snapBtn.style.marginLeft = "auto";
            addBtn.style.marginLeft = "auto";
            exportBtn.style.minWidth = "90px";
            // The button is no longer withheld with the rest of the
            // guided surface. What it opens now is the AUDIO dialog, and
            // the de-click applies to every join that is not blended --
            // which on the masked/both route is all of them, so on the
            // mode this UI actually offers it is always relevant. The
            // guided repairs are not offered from the toolbar at all.
            // (seamsTouched is kept: it is what used to reveal the
            // button, and it still says whether this timeline carries
            // settings of its own -- worth reading before removing.)
            const seamsTouched = () => SEAM_WIDGETS.some((name) => {
                const w = node.widgets?.find((x) => x.name === name);
                return !!w && TL_FIX_DEFAULTS[name] !== undefined
                    && w.value !== TL_FIX_DEFAULTS[name];
            });
            void seamsTouched;

            // ---- run mode (toolbar, not a setup widget) --------------
            // Which half of the workflow a Run is for: the `upscaling`
            // BOOLEAN, which the graph's Mute If gates read.
            // It rides the toolbar for snap's reason: it is a mode you
            // flip WHILE working, and the node's own widget rows are on
            // the far side of a thousand pixels of strip. Rightmost, past
            // export, because it is the widest-scoped thing here -- the
            // others act on this timeline, this one says what pressing
            // Run does at all.
            const runModeWidget = node.widgets?.find(
                (w) => w.name === "upscaling");
            if (runModeWidget) {
                runModeWidget.hidden = true;
                (runModeWidget.options ??= {}).hidden = true;
            }
            function upscaleMode() {
                const v = runModeWidget?.value;
                // "upscale" is the value the widget held while it was a
                // run_mode combo (before 2026-09-14); a saved graph may
                // still hand it over until onConfigure coerces it
                return v === true || v === "true" || v === "upscale";
            }
            // A TOGGLE, not a two-value selector: the label is fixed and
            // the fill says whether it is on, exactly like snap cuts.
            // Generation is the resting state of this workflow, so it is
            // the one that needs no word -- naming both modes made the
            // button re-label itself on every click, which reads as the
            // control moving rather than the state changing.
            const runModeBtn = mkBtn("upscale", "");
            // the header's wide-button width, shared with ▶ quick,
            // ▶ full and export: they are the row's landmarks, and a
            // short one among them reads as an afterthought
            runModeBtn.style.minWidth = "90px";
            function paintRunMode() {
                const up = upscaleMode();
                // themePalette() rather than PAL, which is declared
                // further down and is in its temporal dead zone here
                // (same reason as snapBtn); it also keeps the button
                // theme-live.
                const P = themePalette();
                // The SAME teal the next-run badge wears, so the button
                // and the row it changes read as one fact. Not the pin
                // purple: this is not a pin.
                runModeBtn.style.background = up ? TL_UPSCALE_BG : P.rest;
                runModeBtn.style.borderColor = up ? "transparent" : P.edge;
                runModeBtn.style.color = up ? "#fff" : P.text;
                runModeBtn.title = up
                    ? "ON: Run refines this whole timeline as one piece "
                      + "and renders the full sequence. Any pin is kept and "
                      + "comes back when you switch it off. Click to "
                      + "generate again."
                    : "OFF: Run generates the next take from the pin "
                      + "below. Click to refine this whole timeline "
                      + "instead.";
            }
            runModeBtn.addEventListener("mouseenter", () => {
                runModeBtn.style.background = upscaleMode()
                    ? TL_UPSCALE_HOVER : "rgba(127,127,127,0.3)";
            });
            runModeBtn.addEventListener("mouseleave", paintRunMode);
            runModeBtn.addEventListener("click", () => {
                if (!runModeWidget) return;
                runModeWidget.value = !upscaleMode();
                // through the widget's own callback, so a value set from
                // here and one set from anywhere else take the same road
                runModeWidget.callback?.(runModeWidget.value);
            });
            // Every route into the value repaints: the button paints
            // itself, and the strip repaints because the badges, the
            // clip toggles and the next-run row all describe what a Run
            // does -- and a Run now does something else.
            if (runModeWidget) {
                const prevRm = runModeWidget.callback;
                runModeWidget.callback = function (...args) {
                    const r = prevRm?.apply(this, args);
                    paintRunMode();
                    node.setDirtyCanvas?.(true, true);
                    // guarded: the widget can fire before the strip
                    // exists (workflow load)
                    try { void refresh(); } catch (err) { /* not yet */ }
                    return r;
                };
            }
            paintRunMode();

            // Whether the sequence is a RING, said where it can be seen.
            // The fact still lives in ONE place, the `loop` line of the
            // sequence text: the button reads it to paint and writes it
            // through setLoop, the line's one writer -- it holds nothing
            // of its own. The line was invisible before this (the text
            // is hidden), so a sequence that silently stopped looping
            // rendered as a line and nothing on the strip said so.
            const loopBtn = mkBtn("loop", "");
            // set by every strip render: does anything close the ring?
            // (true until the first render, so nothing greys out early)
            let loopPossible = true;
            function tlLoopToast(detail) {
                try {
                    app.extensionManager?.toast?.add?.({
                        severity: "info", summary: "H3 Timeline",
                        detail, life: 5000,
                    });
                } catch (e2) { /* toast API absent */ }
            }
            function paintLoop() {
                const on = loopOn();
                // an ON button is never disabled: it must stay the way
                // to switch a loaded, unclosed loop off
                const dead = !on && !loopPossible;
                const P = themePalette();   // PAL is in its TDZ here
                loopBtn.style.background = on ? TL_ROLE.extend.bg : P.rest;
                loopBtn.style.borderColor = on ? "transparent" : P.edge;
                loopBtn.style.color = on ? "#fff" : P.text;
                loopBtn.style.opacity = dead ? "0.4" : "1";
                loopBtn.style.cursor = dead ? "default" : "pointer";
                loopBtn.title = dead
                    ? "This sequence cannot loop yet: nothing leads from "
                      + "its last clip back into its first. Click an end "
                      + "pill and choose \"+ loop new\" to generate the "
                      + "take that closes it, or add an existing one as "
                      + "the last clip -- looping then switches on by "
                      + "itself."
                    : on
                    ? "ON: the sequence loops -- its last clip joins its "
                      + "first, playback wraps, and export and upscale "
                      + "keep only the frames of one turn. Click to stop "
                      + "looping."
                    : "OFF: the sequence plays once, start to end. Click "
                      + "to make it loop (its last clip should be a take "
                      + "that leads back into the first). Adding such a "
                      + "take switches this on by itself.";
            }
            loopBtn.addEventListener("mouseenter", () => {
                if (!loopOn() && !loopPossible) return;
                loopBtn.style.background = loopOn()
                    ? TL_ROLE.extend.hover : "rgba(127,127,127,0.3)";
            });
            loopBtn.addEventListener("mouseleave", paintLoop);
            loopBtn.addEventListener("click", () => {
                if (!loopOn() && !loopPossible) return;
                setLoop(!loopOn());
            });
            paintLoop();

            header.append(quickBtn, fullBtn, stateChip, snapBtn, loopBtn,
                          seamBtn, zoomOutBtn, zoomInBtn,
                          fitBtn, addBtn, editBtn, exportBtn, runModeBtn);
            // "next run" config bar: reserved space right below the
            // strip, ALWAYS visible -- states what the pin_specs output
            // will do (root generation vs extend/prepend) and carries
            // the pin options (window cycle, clear). Framed as a
            // CONTROL panel; colors re-applied per render (theme-live).
            const nextRunRow = document.createElement("div");
            Object.assign(nextRunRow.style, {
                // HARD-FIXED height: content changes (buttons
                // appearing, longer text) must NEVER reflow the
                // widget -- single line, ellipsis, overflow hidden.
                // No frame, no fill: plain text row.
                display: "flex", flexWrap: "nowrap", gap: "6px",
                alignItems: "center", height: "24px",
                boxSizing: "border-box", overflow: "hidden",
                padding: "0 2px",
            });
            // selected-clip info bar: a clip-surfaced BOX (name +
            // frames) with the prepend/extend toggles OUTSIDE it, to
            // the right; fixed height like the next-run row
            const infoRow = document.createElement("div");
            Object.assign(infoRow.style, {
                display: "flex", flexWrap: "nowrap", gap: "6px",
                alignItems: "stretch", height: "24px",
                boxSizing: "border-box", overflow: "hidden",
            });
            function barParts(row, capText) {
                row.replaceChildren();
                const cap = document.createElement("span");
                cap.textContent = capText;
                Object.assign(cap.style, {
                    // BOLD and in the body colour: this row states what
                    // the next generation will do, so its label leads
                    // the row rather than sitting behind it as a caption.
                    // BOLD, but the row's own type: 11px sans-serif like
                    // the badge, the duration and the buttons beside it.
                    // The 10px + letterspacing it had was caption
                    // styling, which is the thing it stopped being.
                    flexShrink: "0", color: PAL.text,
                    font: "700 11px/16px sans-serif",
                    whiteSpace: "nowrap",
                });
                row.appendChild(cap);
                return row;
            }

            // ---- pin state (hidden pin_state widget = source of truth)
            // One JSON blob {source, role, window} so richer pin shapes
            // (keyframes, multi-pin) never change the node's signature.
            // Hidden both ways per the widget gotchas: canvas mode reads
            // widget.hidden, Vue reads widget.options.hidden.
            const psWidget = node.widgets?.find(
                (w) => w.name === "pin_state");
            if (psWidget) {
                psWidget.hidden = true;
                (psWidget.options ??= {}).hidden = true;
            }
            // The raw sequence textarea is hidden too (the strip IS the
            // sequence UI); the ✎ button pops it up for hand editing.
            seqWidget.hidden = true;
            (seqWidget.options ??= {}).hidden = true;
            const seqEl = seqWidget.element ?? seqWidget.inputEl;
            if (seqEl) {
                seqEl.style.display = "none";
            }
            // auto_add is deprecated and inert -- the Result Preview owns
            // offering new takes now. Hidden rather than deleted because
            // widgets_values is positional: removing it server-side would
            // shift pin_state and restore_groups on every saved node.
            const autoAddWidget = node.widgets?.find(
                (w) => w.name === "auto_add");
            if (autoAddWidget) {
                autoAddWidget.hidden = true;
                (autoAddWidget.options ??= {}).hidden = true;
            }
            // upscaling joins these: hiding a widget hides its ROW but
            // leaves its input socket, invisible and still hit-tested
            // over the node's real pins. Anything genuinely wired is
            // skipped, so driving the mode from a graph still works.
            dropWidgetSockets(node, ["pin_state", "sequence", "auto_add",
                                     "snap_cuts_to_grid", "upscaling"]);
            // Editing the keyword re-renders the clip row, so the restore
            // button's enabled look tracks it instead of going stale.
            const rgWidget = node.widgets?.find(
                (w) => w.name === "restore_groups");
            if (rgWidget) {
                const prevCb = rgWidget.callback;
                rgWidget.callback = function (...args) {
                    const r = prevCb?.apply(this, args);
                    // guarded: the widget can fire before the row's state
                    // exists (workflow load), and that must not throw
                    try { renderClipInfo(); } catch (err) { /* not yet */ }
                    return r;
                };
            }
            let seqPopup = null;
            function closeSeqPopup() {
                seqPopup?._flush?.();
                seqPopup?.remove();
                seqPopup = null;
            }
            function openSeqPopup() {
                if (seqPopup) {
                    closeSeqPopup();
                    return;
                }
                const pop = document.createElement("div");
                seqPopup = pop;
                Object.assign(pop.style, {
                    position: "absolute", zIndex: "11",
                    left: "8px", right: "8px",
                    top: (header.offsetTop + header.offsetHeight + 4)
                        + "px",
                    background: PAL.rest, color: PAL.text,
                    border: "1px solid " + PAL.edge,
                    borderRadius: "5px", padding: "6px",
                    boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
                    display: "flex", flexDirection: "column",
                    gap: "4px",
                });
                pop.addEventListener("click",
                    (ev) => ev.stopPropagation());
                const hint = document.createElement("div");
                hint.textContent = "one clip per line · '# ' comments · "
                    + "' @ N' forces the enter frame";
                Object.assign(hint.style, {
                    color: PAL.sub, fontSize: "9px",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                });
                const ta = document.createElement("textarea");
                Object.assign(ta.style, {
                    width: "100%", minHeight: "90px",
                    resize: "vertical", boxSizing: "border-box",
                    background: "none", color: PAL.text,
                    border: "1px solid " + PAL.edge,
                    borderRadius: "4px", padding: "4px",
                    font: "11px monospace",
                });
                ta.value = String(seqWidget.value || "");
                let deb = null;
                const apply = () => {
                    deb = null;
                    if (ta.value !== seqWidget.value) {
                        setSequence(ta.value);
                    }
                };
                ta.addEventListener("input", () => {
                    clearTimeout(deb);
                    deb = setTimeout(apply, 300);
                });
                // canvas hotkeys (Delete removes the NODE) must not see
                // keystrokes meant for the textarea
                ta.addEventListener("keydown",
                    (ev) => ev.stopPropagation());
                pop._flush = () => {
                    if (deb !== null) {
                        clearTimeout(deb);
                        apply();
                    }
                };
                pop.append(hint, ta);
                container.appendChild(pop);
                ta.focus();
                setTimeout(() => document.addEventListener(
                    "click", closeSeqPopup, { once: true }), 0);
            }
            editBtn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                openSeqPopup();
            });
            function pinState() {
                try {
                    const s = JSON.parse(psWidget?.value || "null");
                    if (!s || !s.source || !s.role ||
                            s.role === "none") {
                        return null;
                    }
                    if (s.role === "bridge" && !s.source2) return null;
                    return s;
                } catch (err) {
                    return null;
                }
            }
            // which pin role a given clip carries under the current
            // state (a bridge gives its extend side to `source` and its
            // prepend side to `source2`)
            function pinRoleFor(clip, st) {
                if (!st) return null;
                if (st.role === "bridge") {
                    return st.source === clip ? "extend"
                        : st.source2 === clip ? "prepend" : null;
                }
                return st.source === clip ? st.role : null;
            }
            // Does the clip carry THIS role? Asked per side, because a
            // clip bridged onto itself (the take that loops a lone clip)
            // carries both, and pinRoleFor can only answer one.
            function pinHasRole(clip, st, role) {
                if (!st) return false;
                if (st.role === "bridge") {
                    return role === "extend" ? st.source === clip
                                             : st.source2 === clip;
                }
                return st.source === clip && st.role === role;
            }
            // A bridge from the LAST entry into the FIRST is the take
            // that closes a loop -- the same clip on both sides when the
            // strip holds only one.
            function impliesLoop(st) {
                if (!st || st.role !== "bridge" || !lastEntries?.length) {
                    return false;
                }
                return st.source === lastEntries[lastEntries.length - 1].clip
                    && st.source2 === lastEntries[0].clip;
            }
            // Arming that bridge is deciding the cut loops: a take with
            // both feet on the strip's ends has nowhere else to land.
            function setBridge(st) {
                setPinState(st);
                if (impliesLoop(st) && !loopOn()) setLoop(true);
            }
            // The pin AS SHOWN. The run mode itself lives on the toolbar
            // (see runModeBtn above); it changes NOTHING about what is
            // stored -- the pin stays exactly as it was, so switching
            // back to generation restores the strip you left. All that
            // changes is what the strip CLAIMS the next Run will do, and
            // a strip that shows a pin the Run cannot act on is lying
            // confidently, which is the kind of wrong you only catch
            // after the pass has finished. Every DISPLAY site reads this; every site
            // that writes pin_state keeps reading pinState(), because the
            // stored pin has to survive a trip through upscale mode
            // untouched. The controls that write are hidden meanwhile --
            // showing them the shown-pin would make an armed toggle look
            // unarmed, and clicking it would then CLEAR the pin it was
            // preserving.
            function shownPin() {
                return upscaleMode() ? null : pinState();
            }
            // The pin MODE (guided re-renders the window as cond
            // rows; masked preserves it in the target latent) is
            // remembered across pins and sessions: the last explicit
            // choice becomes the default for every NEW pin. Absent
            // means masked -- THE mode since the guided retirement;
            // the server's parse reads absent the same way.
            const PIN_MODE_KEY = "obvpm.h3.pin_mode";
            function rememberedPinMode() {
                // no toggle on the face means no choice to remember: a
                // preference stored before the retirement must not keep
                // stamping guided onto new pins
                if (!SHOW_GUIDED) return "masked";
                try {
                    const v = localStorage.getItem(PIN_MODE_KEY);
                    return v === "guide" ? "guide" : "masked";
                } catch (err) { return "masked"; }
            }
            function rememberPinMode(mode) {
                try { localStorage.setItem(PIN_MODE_KEY, mode); }
                catch (err) { /* private mode etc. -- forget quietly */ }
            }
            function setPinState(st) {
                if (!psWidget) return;
                const prev = pinState();
                if (st && st.role) {
                    // the role decides, every time: a mode left over
                    // from a hand-edited workflow would otherwise
                    // reinstate the pairing that does not work
                    st = { ...st, mode: pinModeOf(st) };
                    // masked windows live on the shared AV grid
                    if (pinModeMasks(st.mode)) {
                        st.window = PIN_WINDOWS_MASKED.includes(
                            String(st.window)) ? String(st.window) : "39";
                    }
                }
                psWidget.value = st ? JSON.stringify(st) : "";
                // Arming a pin that lands in empty space sizes the run to
                // fill it. Only when WHAT is pinned changes (role or
                // clips): cycling the window or the mask of an armed pin
                // must not keep overwriting a length that was typed.
                if (st && st.role && (!prev || prev.role !== st.role
                        || prev.source !== st.source
                        || (prev.source2 ?? null) !== (st.source2 ?? null))) {
                    fitDurationToGap(st);
                }
                tlNoteEdit();
                node.setDirtyCanvas?.(true, true);
                void refresh();
            }

            // ---- ruler + timeline strip + playhead --------------------
            const timelineWrap = document.createElement("div");
            Object.assign(timelineWrap.style, {
                position: "relative", display: "flex",
                flexDirection: "column", gap: "2px",
            });
            const ruler = document.createElement("canvas");
            ruler.height = 16;
            Object.assign(ruler.style, {
                width: "100%", height: "16px", display: "block",
                cursor: "ew-resize",
            });
            ruler.title = "Drag to scrub";
            if (!document.getElementById("obvpm-tl-style")) {
                const st = document.createElement("style");
                st.id = "obvpm-tl-style";
                st.textContent =
                    ".obvpm-tl-strip { scrollbar-width: thin; " +
                    "scrollbar-color: #555b66 transparent; } " +
                    ".obvpm-tl-strip::-webkit-scrollbar { height: 6px; } " +
                    ".obvpm-tl-strip::-webkit-scrollbar-thumb " +
                    "{ background: #555b66; border-radius: 3px; } " +
                    ".obvpm-tl-strip::-webkit-scrollbar-track " +
                    "{ background: transparent; }";
                document.head.appendChild(st);
            }
            const strip = document.createElement("div");
            strip.classList.add("obvpm-tl-strip");
            Object.assign(strip.style, {
                // NO gap: a separator between blocks is width that is
                // not time, so it would make a span measured across a
                // boundary wider than the same span inside a clip. The
                // blocks' own 1px borders meet to draw the divider
                // instead, which costs nothing.
                display: "flex", gap: "0", minHeight: "76px",
                alignItems: "stretch", overflowX: "scroll",
                position: "relative", paddingTop: "6px",
                paddingBottom: "6px",
                // small side gutters so the START/END pills straddle
                // the outer clip edges exactly like interior pills
                // straddle their gaps (not fully on top of the clip)
                paddingLeft: "8px", paddingRight: "8px",
            });
            // insertion indicator for drag-reorder: a green I-beam in
            // the GAP the dragged clip would land in; hidden when the
            // drop would not move it. Green = action, distinct from the
            // amber time playhead.
            const DROP_GREEN = "#2fce68";
            const dropLine = document.createElement("div");
            Object.assign(dropLine.style, {
                position: "absolute", top: "0", bottom: "0",
                width: "9px", pointerEvents: "none",
                display: "none", zIndex: "3",
            });
            const beam = document.createElement("div");
            Object.assign(beam.style, {
                position: "absolute", top: "0", bottom: "0",
                left: "3px", width: "3px",
                background: DROP_GREEN, borderRadius: "1px",
            });
            const capT = document.createElement("div");
            const capB = document.createElement("div");
            for (const cap of [capT, capB]) {
                Object.assign(cap.style, {
                    position: "absolute", left: "0", width: "9px",
                    height: "3px", background: DROP_GREEN,
                    borderRadius: "1px",
                });
            }
            capT.style.top = "0";
            capB.style.bottom = "0";
            dropLine.append(beam, capT, capB);
            let dragFrom = null;      // index being dragged
            let dropInsertAt = null;  // insertion index in the gap
            // a ctrl+drag that MOVED must not also read as a click on the
            // clip (which would seek the playhead there)
            let dragMoved = false;
            const playhead = document.createElement("div");
            Object.assign(playhead.style, {
                position: "absolute", top: "0", bottom: "0",
                width: "2px", background: "#e8a33d",
                pointerEvents: "none", display: "none", zIndex: "2",
            });
            // live read-out during a ctrl+drag: the amber of the playhead,
            // because it reports a position in TIME
            // Same badge as the cut drag's readout, and for the same
            // reason: both are "what this gesture will do", so they
            // should not look like two different kinds of message. It
            // rides on the STRIP (added for the drag, removed after),
            // which is what lets it sit on the clip rather than at the
            // pointer -- see the ctrl+drag handler.
            const dragTag = document.createElement("div");
            Object.assign(dragTag.style, {
                position: "absolute", top: "2px", zIndex: "7",
                display: "none", padding: "0 4px", borderRadius: "3px",
                background: DROP_GREEN, color: "#04220f",
                font: "600 10px sans-serif", pointerEvents: "none",
                whiteSpace: "nowrap", transform: "translateX(-50%)",
            });
            timelineWrap.append(ruler, strip, playhead);

            // Theme-aware strip palette: one hardcoded set cannot serve
            // both Comfy themes (dark blocks read as black holes on a
            // light page). Luminance of the page background decides.
            let PAL = themePalette();
            function applyChrome() {
                // snapBtn is NOT here on purpose: it wears its own
                // on/off fill. Everything else in the row -- seamBtn
                // included -- takes the theme's plain chrome, or it
                // keeps mkBtn's hardcoded dark and reads as a foreign
                // control on a light theme.
                for (const el of [picker, addBtn, editBtn, quickBtn,
                                  fullBtn, exportBtn, seamBtn,
                                  zoomOutBtn, zoomInBtn, fitBtn]) {
                    el.style.background = PAL.rest;
                    el.style.borderColor = PAL.edge;
                    el.style.color = PAL.text;
                }
                timelineWrap.style.background = PAL.stripBg;
                timelineWrap.style.borderRadius = "4px";
                timelineWrap.style.padding = "3px";
                {   // one-line diagnosis: paste this if colors look off
                    const panel = tlPanelColor();
                    requestAnimationFrame(() => tldbg("chrome:",
                        "panel", panel.color, "from", panel.source,
                        "| menu-var",
                        tlCssVar("--comfy-menu-bg", "(unset)"),
                        "| input-var",
                        tlCssVar("--comfy-input-bg", "(unset)"),
                        "| content-bg",
                        tlCssVar("--content-bg", "(unset)"),
                        "| stripBg set to", PAL.stripBg,
                        "| computed",
                        getComputedStyle(timelineWrap).backgroundColor));
                }
                for (const el of [beam, capT, capB]) {
                    el.style.background = PAL.drop;
                }
            }
            applyChrome();

            let cumStarts = [];   // global start frame of each clip
            let playedArr = [];   // played frames of each clip
            let totalFrames = 0;  // of the whole cut
            let blockEls = [];    // the strip's clip blocks, in order
            let linkEls = [];     // lineage connectors between blocks

            // Every time<->pixel mapping goes through the actual block
            // geometry rather than arithmetic on the scale. Blocks are
            // exactly proportional to their length now, but the 3px
            // separators between them are chrome and not time, so the
            // offsets accumulate; reading the boxes absorbs that and a
            // clip's end lands exactly on its own edge.
            // A ctrl+drag previews itself with CSS transforms, and a
            // transform does not move offsetLeft -- so the ruler and the
            // playhead kept reading the pre-drag layout and only caught
            // up when the drop re-rendered. That catch-up is the "scale
            // shifting on release": the blocks had moved for a second
            // and the time scale had not. This tells the mapping about
            // the preview, so everything moves together and the drop
            // changes nothing that was already on screen.
            let ghostShift = null;      // {from: entry index, px}
            function ghostAt(k) {
                return ghostShift && k >= ghostShift.from
                    ? ghostShift.px : 0;
            }
            function frameToX(f) {
                if (!blockEls.length) return 0;
                let k = 0;
                cumStarts.forEach((s, i2) => { if (f >= s) k = i2; });
                const el = blockEls[k];
                const frac = playedArr[k]
                    ? Math.min(1, (f - cumStarts[k]) / playedArr[k]) : 0;
                return el.offsetLeft + frac * el.offsetWidth + ghostAt(k);
            }
            function xToFrame(x) {
                if (!blockEls.length) return 0;
                for (let k = 0; k < blockEls.length; k++) {
                    const el = blockEls[k];
                    const lo = el.offsetLeft + ghostAt(k);
                    const hi = lo + el.offsetWidth;
                    if (x < lo) return cumStarts[k]; // in the gap before
                    if (x <= hi || k === blockEls.length - 1) {
                        const frac = Math.max(0, Math.min(1,
                            (x - lo) / (el.offsetWidth || 1)));
                        return cumStarts[k] + frac * playedArr[k];
                    }
                }
                return totalFrames;
            }

            // ---- the scale ------------------------------------------
            // The strip used to FIT: each block took a share of the
            // widget width proportional to its length, floored at a
            // minimum so short clips stayed readable. Two problems.
            // The scale moved whenever the sequence did, so dragging a
            // clip "one second" meant a different distance after every
            // edit; and the floor made the mapping non-uniform, so a
            // short clip covered more width than its duration.
            // Now the scale is the user's and it holds still. A block
            // is exactly played * scale wide with no floor, and one too
            // narrow for its own label simply clips it -- zoom is what
            // you reach for instead of a minimum width.
            const TL_SCALE_MIN = 0.015;   // ~20 min across a 1000px strip
            const TL_SCALE_MAX = 12;      // ~1.7s across it
            const TL_ZOOM_STEP = 1.4;          // the +/- buttons
            // Wheel zoom is continuous, not stepped: the factor is
            // exponential in the wheel delta, so a slow trackpad drag
            // and a fast mouse notch feel like the same gesture at
            // different speeds, and zooming out then back in returns to
            // where you started instead of drifting (exp(-d) * exp(d)
            // is exactly 1, which a linear factor would not be).
            const TL_ZOOM_PER_PX = 0.0022;     // ~1.25x per 100px notch
            // A cut handle is a fixed 11px whatever the zoom, so it
            // stays grabbable -- but a block narrower than two of them
            // would have both its edges under the same pixels, and a
            // cut you cannot aim is worse than one you cannot reach.
            // Below this the handles are simply not offered; zooming in
            // is the answer and it exists now.
            const CUT_HANDLE_MIN = 26;
            const TL_SCALE_PROP = "obvpm_tl_scale";

            function clampScale(v) {
                return Math.max(TL_SCALE_MIN, Math.min(TL_SCALE_MAX, v));
            }
            function stripInnerWidth() {
                // clientWidth excludes the scrollbar but not the padding
                return Math.max(80, (strip.clientWidth || 0) - 16);
            }
            // The scale that would show everything at once. Nothing to
            // subtract: the blocks butt together, so the content is
            // exactly totalFrames * scale wide.
            function fitScale(total) {
                return clampScale(stripInnerWidth()
                                  / Math.max(1, total ?? totalFrames));
            }
            function storedScale() {
                const v = Number(node.properties?.[TL_SCALE_PROP]);
                return Number.isFinite(v) && v > 0 ? v : null;
            }
            function saveScale(v) {
                node.properties = node.properties || {};
                node.properties[TL_SCALE_PROP] = v;
            }
            // px per frame in force right now. Null until the first
            // render has a length to fit to; asking before then would
            // bake in a scale for an empty timeline.
            function scaleNow() {
                return storedScale() ?? (totalFrames ? fitScale() : null);
            }
            // Zoom keeps the frame under `anchorX` (strip coordinates,
            // scroll included) where it is. Without that the view jumps
            // somewhere else on every step and repeated zooming is
            // unusable.
            function setScale(next, anchorX) {
                const now = scaleNow();
                const v = clampScale(next);
                if (now && Math.abs(v - now) / now < 1e-6) return;
                const ax = anchorX == null
                    ? strip.scrollLeft + strip.clientWidth / 2
                    : anchorX;
                const at = blockEls.length ? xToFrame(ax) : 0;
                saveScale(v);
                applyScale();
                strip.scrollLeft = Math.max(
                    0, strip.scrollLeft + (frameToX(at) - ax));
                drawRuler();
                drawLinks(lastEntries);
                setPlayhead(displayFrame() ?? heldFrame);
            }
            // Re-widths what is already on screen. Cheaper than a full
            // refresh (no probes, no metadata) and it keeps the blocks'
            // own state -- which matters because refresh() closes the
            // seam popup and drops any drag in progress.
            function applyScale() {
                const px = scaleNow();
                if (!px) return;
                for (const el of strip.children) {
                    const f = Number(el.dataset?.obvpmFrames);
                    if (Number.isFinite(f)) {
                        // same floor as the render, or a re-width would
                        // disagree with the layout it is repeating
                        el.style.width = Math.max(1, f * px) + "px";
                    }
                }
            }

            // Wheel events arrive far faster than a re-render is worth
            // doing -- drawLinks rebuilds every seam element -- so they
            // compound into one target and land on the next frame. The
            // base is the PENDING scale, not the applied one, or the
            // events fight each other and the zoom crawls.
            let zoomWant = null;
            let zoomRaf = 0;
            function zoomBy(factor, anchorX) {
                const base = zoomWant?.scale ?? scaleNow() ?? 1;
                zoomWant = { scale: clampScale(base * factor),
                             anchor: anchorX };
                if (zoomRaf) return;
                zoomRaf = requestAnimationFrame(() => {
                    zoomRaf = 0;
                    const want = zoomWant;
                    zoomWant = null;
                    if (want) setScale(want.scale, want.anchor);
                });
            }
            function wheelFactor(ev) {
                // deltaMode 0 = pixels, 1 = lines, 2 = pages. Firefox
                // reports lines for a mouse wheel and pixels for a
                // trackpad, so without this the two differ by ~16x.
                const unit = ev.deltaMode === 1 ? 16
                    : ev.deltaMode === 2 ? 400 : 1;
                // one violent event should not slam into the limit
                const d = Math.max(-240, Math.min(240, ev.deltaY * unit));
                return Math.exp(-d * TL_ZOOM_PER_PX);
            }
            // rect is SCREEN px, offsetLeft is LAYOUT px: the canvas
            // zoom CSS-scales the widget, so divide it out (the same
            // correction scrubTo makes)
            function anchorFrom(ev, el) {
                const r = el.getBoundingClientRect();
                const z = r.width / (el.offsetWidth || 1) || 1;
                return (ev.clientX - r.left) / z + strip.scrollLeft;
            }
            // The ruler IS the scale, so a bare wheel over it zooms --
            // no modifier, and no argument with the canvas, which never
            // sees a wheel there anyway.
            ruler.addEventListener("wheel", (ev) => {
                if (!totalFrames) return;
                ev.preventDefault();
                ev.stopPropagation();
                zoomBy(wheelFactor(ev), anchorFrom(ev, ruler));
            }, { passive: false });

            zoomInBtn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                setScale((scaleNow() || 1) * TL_ZOOM_STEP);
            });
            zoomOutBtn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                setScale((scaleNow() || 1) / TL_ZOOM_STEP);
            });
            fitBtn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                if (!totalFrames) return;
                // fit means fit: anchor at the left edge and scroll home,
                // or the anchoring in setScale would fight it
                saveScale(fitScale());
                applyScale();
                strip.scrollLeft = 0;
                drawRuler();
                drawLinks(lastEntries);
                setPlayhead(displayFrame() ?? heldFrame);
            });
            // ctrl+wheel zooms about the cursor. Plain wheel is left to
            // the canvas: the graph's own zoom is what people expect
            // from a bare wheel over a node, and stealing it would make
            // the timeline a trap to scroll past.
            // Over the blocks a bare wheel belongs to the canvas, so
            // zooming there needs the modifier. Same smooth path.
            strip.addEventListener("wheel", (ev) => {
                if (!ev.ctrlKey && !ev.metaKey) return;
                if (!totalFrames) return;
                ev.preventDefault();
                ev.stopPropagation();
                zoomBy(wheelFactor(ev), anchorFrom(ev, strip));
            }, { passive: false });

            function drawRuler() {
                const w = timelineWrap.clientWidth || 300;
                if (ruler.width !== w) ruler.width = w;
                const ctx = ruler.getContext("2d");
                ctx.clearRect(0, 0, w, 16);
                if (!totalFrames || !blockEls.length) return;
                drawLanding(ctx, w);
                ctx.fillStyle = PAL.tick;
                ctx.strokeStyle = PAL.tickLine;
                ctx.font = "9px sans-serif";
                const secs = totalFrames / TL_FPS;
                // seconds are evenly spaced now, but zoomed out they
                // can still land on top of each other, so density is
                // enforced in PIXELS: skip ticks closer than 8px,
                // labels than 26px
                let lastX = -1e9;
                let lastLabelX = -1e9;
                for (let s = 0; s <= Math.floor(secs); s++) {
                    const x = Math.round(frameToX(s * TL_FPS) -
                        strip.scrollLeft) + 0.5;
                    if (x < 0 || x > w) continue;
                    if (x - lastX < 8) continue;
                    lastX = x;
                    const label = x - lastLabelX >= 26;
                    ctx.beginPath();
                    ctx.moveTo(x, label ? 6 : 11);
                    ctx.lineTo(x, 16);
                    ctx.stroke();
                    if (label) {
                        ctx.fillText(`${s}s`, x + 2, 8);
                        lastLabelX = x;
                    }
                }
            }
            // ---- where the next run lands ------------------------------
            // The span the armed run's KEPT footage would occupy once it
            // is generated and added -- the same position placeClip
            // gives it (see insertFilling), in timeline frames:
            //   extend / bridge  starts where the pinned clip stops
            //                    playing, filling the empty space after it
            //   prepend          ends where the pinned clip starts,
            //                    right-aligned in the space before it
            //   loop bridge      after the last clip, on the wrap
            // A linked neighbour already sitting there is REPLACED by
            // the new take, so the span starts where that one does.
            // `frames` is null when the length cannot be read (a wired
            // duration nobody can trace); the band then covers the gap.
            function pinLanding(st = shownPin()) {
                const es = lastEntries;
                if (!st || !es?.length || !cumStarts.length) return null;
                const k = es.findIndex((e) => e.clip === st.source);
                if (k < 0) return null;
                const secs = durSeconds();
                const plan = secs == null ? null
                    : durPlan(durFrames(secs, st), st);
                const kept = plan ? Math.max(0, plan.kept) : null;
                const isGap = (i) => es[i] != null && es[i].gap != null;
                const endOf = (i) => cumStarts[i] + playedArr[i];
                if (st.role === "prepend") {
                    const occ = tlOccupant(es, st.source, "before", linkedMeta);
                    if (occ) {
                        return { start: cumStarts[occ.idx], gapIdx: -1,
                                 frames: kept ?? playedArr[occ.idx] };
                    }
                    if (isGap(k - 1)) {
                        const gf = playedArr[k - 1];
                        const f = kept ?? gf;
                        return { start: Math.max(cumStarts[k - 1],
                                                 cumStarts[k] - f),
                                 frames: f, gapIdx: k - 1, gapFrames: gf };
                    }
                    return { start: cumStarts[k], frames: kept, gapIdx: -1 };
                }
                if (st.role === "bridge" && impliesLoop(st)) {
                    return { start: totalFrames, frames: kept, gapIdx: -1 };
                }
                const occ = tlOccupant(es, st.source, "after", linkedMeta);
                if (occ) {
                    return { start: endOf(k), gapIdx: -1,
                             frames: kept ?? playedArr[occ.idx] };
                }
                if (isGap(k + 1)) {
                    const gf = playedArr[k + 1];
                    return { start: endOf(k), frames: kept ?? gf,
                             gapIdx: k + 1, gapFrames: gf };
                }
                return { start: endOf(k), frames: kept, gapIdx: -1 };
            }

            // The run length whose KEPT footage comes closest to the
            // empty space the pin lands in. Only legal lengths are tried
            // (the shared AV grid when the pin masks, the 17k+5 ladder
            // otherwise), so "closest" can still be a rung away: 2.125 s
            // on the AV grid. A length that keeps nothing is never
            // chosen. A wired duration is left alone -- its value belongs
            // to whatever drives it.
            function fitDurationToGap(st) {
                if (!durWidget || durWired() || !st) return;
                const land = pinLanding(st);
                if (!land || land.gapIdx < 0 || !land.gapFrames) return;
                const masks = durMasks(st);
                const base = masks ? AV_BASE : RUN_BASE;
                const step = masks ? AV_STEP : RUN_LADDER;
                let best = null;
                for (let f = base; f <= 150 * TL_FPS; f += step) {
                    const kept = durPlan(f, st)?.kept ?? f;
                    if (kept <= 0) continue;
                    const d = Math.abs(kept - land.gapFrames);
                    if (!best || d < best.d) best = { f, d };
                }
                if (!best) return;
                durWidget.value = Number((best.f / TL_FPS).toFixed(3));
                paintDur();
            }

            // Purple: not the playhead's amber (time), the pin's orange
            // (the join) or the drop green (an edit) -- this is a clip
            // that does not exist yet.
            function drawLanding(ctx, w) {
                const land = pinLanding();
                if (!land) return;
                const px = scaleNow() || 1;
                const frames = land.frames
                    ?? (land.gapIdx >= 0 ? land.gapFrames : null);
                const x0 = frameToX(land.start) - strip.scrollLeft;
                // an unknown length still marks WHERE: a narrow band
                const x1 = frames ? x0 + frames * px : x0 + 4;
                if (x1 < 0 || x0 > w) return;
                ctx.save();
                ctx.globalAlpha = 0.35;
                ctx.fillStyle = "rgb(150, 104, 222)";
                ctx.fillRect(x0, 0, Math.max(2, x1 - x0), 16);
                ctx.globalAlpha = 1;
                ctx.fillRect(x0, 13, Math.max(2, x1 - x0), 3);
                ctx.restore();
            }

            let lastEntries = [];
            // the clips on the strip at the previous render; null until
            // the first one, so a loaded graph is not read as "added"
            let seenClips = null;
            let lastPx = null;          // the scale the last render used
            // Where the playhead was, kept ACROSS rebuilds. A cut edits
            // the sequence, which invalidates the loaded playlist -- but
            // losing the position too made the strip look like it had
            // reset itself every time you trimmed a frame.
            let heldFrame = null;
            const rulerRO = new ResizeObserver(
                () => requestAnimationFrame(() => {
                    drawRuler();
                    drawLinks(lastEntries); // zones sit at computed x
                }));
            rulerRO.observe(timelineWrap);
            let themeTimer = null;
            const themeMO = new MutationObserver(() => {
                clearTimeout(themeTimer);
                themeTimer = setTimeout(() => {
                    // compare COMPUTED values: the var() strings are
                    // constants, but tlAlpha/tlDarken outputs change
                    // whenever the underlying theme vars do
                    const next = themePalette();
                    if (JSON.stringify(next) === JSON.stringify(PAL)) {
                        return;
                    }
                    PAL = next;
                    applyChrome();
                    void refresh(); // reskin blocks/pills/popup chrome
                }, 150);
            });
            for (const t of [document.documentElement, document.body]) {
                themeMO.observe(t, { attributes: true,
                    attributeFilter: ["class", "style", "data-theme"] });
            }
            // palette changes that arrive as injected or replaced
            // stylesheets never touch html/body attributes -- watch
            // the head too (debounce + computed-palette compare make
            // unrelated style insertions no-ops)
            themeMO.observe(document.head,
                { childList: true, subtree: true, characterData: true });
            strip.addEventListener("scroll",
                () => requestAnimationFrame(drawRuler));

            // While a cut handle is being dragged the time cursor
            // belongs ON the cut. It cannot simply be positioned in the
            // drag handler: showing the cut's frame SEEKS the player,
            // and the player puts the cursor back wherever it landed --
            // so the two fought and the cursor never appeared to move.
            // Pinning it here is the only place that wins.
            let cutPin = null;          // strip x, or null
            function setPlayhead(f) {
                try { updateCutRow(); } catch (err) { /* pre-build */ }
                if (cutPin != null) {
                    playhead.style.display = "block";
                    playhead.style.left =
                        (cutPin - strip.scrollLeft) + "px";
                    return;
                }
                if (!totalFrames || f == null || !blockEls.length) {
                    playhead.style.display = "none";
                    return;
                }
                const c = Math.max(0, Math.min(f, totalFrames));
                playhead.style.display = "block";
                playhead.style.left =
                    (frameToX(c) - strip.scrollLeft) + "px";
            }

            // ---- preview player ---------------------------------------
            // Double-buffered for near-gapless hops: while one <video>
            // plays, the next clip is already loaded and seeked in the
            // hidden twin, so a seam is a visibility swap + play(), not a
            // src teardown. Boundary cuts run on a per-frame rAF watcher
            // (timeupdate only fires ~4x/s -- up to 250ms late).
            const videoWrap = document.createElement("div");
            videoWrap.classList.add("comfy-img-preview");
            Object.assign(videoWrap.style,
                { flex: "1", position: "relative",
                  background: "#000", borderRadius: "4px" });
            const vids = [0, 1].map(() => {
                const v = document.createElement("video");
                v.playsInline = true;
                Object.assign(v.style, {
                    position: "absolute", inset: "0",
                    width: "100%", height: "100%",
                    objectFit: "contain", opacity: "0",
                    pointerEvents: "none",
                });
                return v;
            });
            // cut controls: a row directly under the strip, with the two
            // cut buttons at the far edges so they read as "trim from
            // this end" and uncut between them
            const cutRow = document.createElement("div");
            Object.assign(cutRow.style, {
                display: "flex", gap: "6px", alignItems: "center",
                height: "22px", boxSizing: "border-box",
            });
            const cutBtn = (label, side, title) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.title = title;
                Object.assign(b.style, {
                    borderRadius: "4px", padding: "1px 9px",
                    font: "11px/16px sans-serif", whiteSpace: "nowrap",
                    flexShrink: "0",
                });
                b.addEventListener("click", () => applyPlayheadCut(side));
                b.addEventListener("mouseenter", () => {
                    if (b.dataset.on === "1") {
                        b.style.background = TL_ROLE.extend.hover;
                    }
                });
                b.addEventListener("mouseleave", () => {
                    if (b.dataset.on === "1") {
                        b.style.background = TL_ROLE.extend.bg;
                    }
                });
                return b;
            };
            const cutLeftBtn = cutBtn("⟨ cut left", "left",
                "Cut the clip under the playhead from its LEFT edge to "
                + "here. If nothing leads into that clip, the time it "
                + "gave up stays as empty space to bridge into.");
            const cutRightBtn = cutBtn("cut right ⟩", "right",
                "Cut the clip under the playhead from here to its RIGHT "
                + "edge. The next run extends from this cut.");
            const uncutBtn = document.createElement("button");
            uncutBtn.textContent = "uncut";
            uncutBtn.title = "Remove the manual cut from the selected clip "
                + "and let its seam be derived from the sidecars again";
            Object.assign(uncutBtn.style, {
                background: "#2a2e36", color: "#e8eaee",
                border: "1px solid #3a3f48", borderRadius: "4px",
                padding: "1px 9px", cursor: "pointer",
                font: "11px/16px sans-serif", whiteSpace: "nowrap",
            });
            uncutBtn.addEventListener("click", () => {
                if (lastHighlight < 0) return;
                setCut(lastHighlight, { enter: null, exit: null });
            });
            function updateCutRow() {
                const has = selectedHasCut();
                uncutBtn.style.display = has ? "block" : "none";
                // themePalette() rather than PAL: this can run before
                // PAL's declaration is reached (temporal dead zone)
                const P = themePalette();
                for (const [b, side] of [[cutLeftBtn, "left"],
                                         [cutRightBtn, "right"]]) {
                    const t = playheadCut(side);
                    const ok = !!t && !t.blocked;
                    b.dataset.on = ok ? "1" : "0";
                    b.style.background = ok ? TL_ROLE.extend.bg : "none";
                    b.style.border = "1px solid "
                        + (ok ? "transparent" : P.edge);
                    b.style.color = ok ? "#fff" : P.text;
                    b.style.opacity = ok ? "1" : "0.5";
                    b.style.cursor = ok ? "pointer" : "not-allowed";
                }
            }
            // Two spacers rather than auto margins on `uncut`: hiding
            // an auto-margined element removes its margins too, which
            // let both cut buttons collapse to the left whenever there
            // was no cut to undo. Spacers hold the edges regardless.
            const cutSpacer = () => {
                const d = document.createElement("div");
                d.style.flex = "1";
                return d;
            };
            cutRow.append(cutLeftBtn, cutSpacer(), uncutBtn, cutSpacer(),
                          cutRightBtn);
            const progress = tlProgressBar();
            const dropProgress = tlOnProgress(
                () => String(widgetValue("preview_filename", "")
                             || "obvpm_h3_preview"),
                progress);
            container.append(videoWrap, header, progress.el, timelineWrap,
                             cutRow, infoRow, nextRunRow);
            videoWrap.append(...vids);

            const widget = node.addDOMWidget("mctx_timeline", "div",
                container, { hideOnZoom: false });
            widget.serialize = false;
            widget.options.serialize = false;
            tlPanWithMiddleButton(container);
            const minHeight = 320;
            widget.computeLayoutSize = () => ({ minHeight, minWidth: 340 });
            Object.defineProperty(widget, "width", {
                configurable: true, get: () => undefined, set: () => {},
            });

            // forward canvas gestures (wheel over the video area only;
            // the strip keeps its own horizontal scrolling)
            videoWrap.addEventListener("wheel", (e) => {
                const canvasEl = app.canvas?.canvas;
                if (!canvasEl) return;
                e.preventDefault();
                e.stopPropagation();
                const { clientX, clientY, deltaX, deltaY,
                        ctrlKey, metaKey, shiftKey } = e;
                canvasEl.dispatchEvent(new WheelEvent("wheel", {
                    clientX, clientY, deltaX, deltaY,
                    ctrlKey, metaKey, shiftKey,
                }));
            });

            // ---- sequence editing (text widget = source of truth) -----
            function currentLines() {
                return String(seqWidget.value || "").split("\n");
            }
            function setSequence(text) {
                seqWidget.value = text;
                const seqEl2 = seqWidget.element ?? seqWidget.inputEl;
                if (seqEl2) seqEl2.value = text;
                seqWidget.callback?.(text);
                tlNoteEdit();      // one undo step per edit
                void refresh();
            }
            function entryLines() {
                // indices of entry lines (not comments, not the loop
                // directive) within the raw line list
                const map = [];
                currentLines().forEach((raw, i) => {
                    const t = raw.trim();
                    if (tlIsEntryLine(t)) map.push(i);
                });
                return map;
            }
            // The cut as a RING: its last entry joins its first. Stored
            // as the `loop` line in the sequence text -- a property of
            // the cut, so it lives with the cut -- and set from the end
            // pills' menu ("+ loop new" arms the take that closes the
            // ring and writes the line; "stop looping" removes it), or
            // implied by pinning a lone clip on both sides. There is no
            // separate switch: it would be a second control for the
            // same fact. Everything that acts on the wrap (end pills,
            // quick playback, the export) reads the text.
            function loopOn() { return tlSequenceLoops(seqWidget?.value); }
            // The directive's one writer. It rides as the FIRST line, so
            // a glance at the text says the cut is a ring before it says
            // what is in it; reading accepts it anywhere.
            function setLoop(on) {
                const kept = currentLines().filter(
                    (raw) => !TL_LOOP_RE.test(raw.trim()));
                if (on) kept.unshift(TL_LOOP_LINE);
                setSequence(kept.join("\n"));
            }
            // One writer for cuts, going through the same formatter the
            // server's parser mirrors. `null` clears that side.
            function setCut(i, patch) {
                const es = tlParseSequence(seqWidget.value);
                const e = es[i];
                if (!e || e.gap != null) return;
                if ("enter" in patch) e.enterOverride = patch.enter;
                if ("exit" in patch) e.exitOverride = patch.exit;
                const lines = currentLines();
                lines[entryLines()[i]] = tlFormatLine(e);
                setSequence(lines.join("\n"));
            }
            // ONE join's own settings. They live on the clip to the
            // RIGHT of the join, so entry i carries the settings for the
            // join in front of it -- the same index the server keys its
            // corrections by. `value` undefined clears the override and
            // the join goes back to inheriting the timeline's value.
            function setSeamOpt(i, key, value) {
                const es = tlParseSequence(seqWidget.value);
                const e = es[i];
                if (!e || e.gap != null) return;
                e.seamOpts = { ...(e.seamOpts || {}) };
                if (value === undefined || value === null) {
                    delete e.seamOpts[key];
                } else {
                    e.seamOpts[key] = value;
                }
                const lines = currentLines();
                lines[entryLines()[i]] = tlFormatLine(e);
                setSequence(lines.join("\n"));
            }
            function seamOptsAt(i) {
                const e = tlParseSequence(seqWidget.value)[i];
                return (e && e.gap == null && e.seamOpts) || {};
            }
            function clearSeamOpts(i) {
                const es = tlParseSequence(seqWidget.value);
                const e = es[i];
                if (!e || e.gap != null || !e.seamOpts
                        || !Object.keys(e.seamOpts).length) {
                    return;
                }
                e.seamOpts = {};
                const lines = currentLines();
                lines[entryLines()[i]] = tlFormatLine(e);
                setSequence(lines.join("\n"));
            }
            // Overrides arriving from somewhere else -- today the Result
            // Preview, where a take's seams are tuned before it is added
            // here. The map is keyed by CLIP PATH, not by join index,
            // for the same reason the settings live in the sequence line
            // at all: indices shift on every add and reorder, and this
            // map was built against a different sequence entirely.
            //
            // Merged, not replaced: the map speaks only for the keys it
            // names, and a bridge take's tail join lands on a clip that
            // may already have settings of its own.
            function applySeamOptsMap(map) {
                if (!map) return 0;
                const es = tlParseSequence(seqWidget.value);
                const lines = currentLines();
                const at = entryLines();
                let n = 0;
                es.forEach((e, i) => {
                    const o = map[e.clip];
                    if (!o || !Object.keys(o).length) return;
                    // a clip that landed with empty space in front of it
                    // has no join to carry them
                    if (!seamJoinAt(i, es)) return;
                    e.seamOpts = { ...(e.seamOpts || {}), ...o };
                    lines[at[i]] = tlFormatLine(e);
                    n++;
                });
                if (n) setSequence(lines.join("\n"));
                return n;
            }

            function removeEntry(i) {
                const lines = currentLines();
                lines.splice(entryLines()[i], 1);
                setSequence(lines.join("\n"));
            }
            function reorderEntry(from, to) {
                if (from === to) return;
                const lines = currentLines();
                const [moved] = lines.splice(entryLines()[from], 1);
                // recompute entry positions after the removal
                const map = [];
                lines.forEach((raw, k) => {
                    const t = raw.trim();
                    if (tlIsEntryLine(t)) map.push(k);
                });
                const at = to >= map.length ? lines.length : map[to];
                lines.splice(at, 0, moved);
                setSequence(lines.join("\n"));
            }
            // + add: popup MENU of the scoped clip list (the dropdown
            // row is gone); click appends to the sequence end
            let addMenu = null;
            function closeAddMenu() {
                addMenu?.remove();
                addMenu = null;
            }
            function openAddMenu() {
                if (addMenu) {
                    closeAddMenu();
                    return;
                }
                const pop = document.createElement("div");
                addMenu = pop;
                Object.assign(pop.style, {
                    position: "absolute", zIndex: "11",
                    right: "8px", minWidth: "220px", maxWidth: "75%",
                    top: (header.offsetTop + header.offsetHeight + 4)
                        + "px",
                    maxHeight: "190px", overflowY: "auto",
                    background: PAL.rest, color: PAL.text,
                    border: "1px solid " + PAL.edge,
                    borderRadius: "5px", padding: "4px",
                    font: "11px sans-serif",
                    boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
                });
                pop.addEventListener("click",
                    (ev) => ev.stopPropagation());
                // base_folder may be wired, and a wired value changes
                // without firing the widget callback that normally
                // re-filters. Re-scope on open rather than trusting
                // whatever the folder was when this node was built.
                populatePicker();
                const fill = () => {
                    pop.replaceChildren();
                    const values = Array.from(picker.options)
                        .map((o) => o.value);
                    if (!values.length) {
                        const n = document.createElement("div");
                        n.textContent = "no clips in scope";
                        n.style.color = PAL.sub;
                        n.style.padding = "3px 5px";
                        pop.appendChild(n);
                    }
                    for (const v of values) {
                        const it = document.createElement("div");
                        it.textContent = v;
                        it.title = v;
                        Object.assign(it.style, {
                            padding: "3px 5px", borderRadius: "3px",
                            cursor: "pointer", whiteSpace: "nowrap",
                            overflow: "hidden", textOverflow: "ellipsis",
                        });
                        it.addEventListener("mouseenter", () =>
                            it.style.background = PAL.active);
                        it.addEventListener("mouseleave", () =>
                            it.style.background = "none");
                        it.addEventListener("click", () => {
                            const text =
                                String(seqWidget.value || "").trimEnd();
                            setSequence(text ? text + "\n" + v : v);
                            closeAddMenu();
                        });
                        pop.appendChild(it);
                    }
                };
                fill();
                // the cached list itself may be stale too (takes saved
                // since page load): rescan in the background and re-fill
                // this same menu if it is still the open one
                void rescanPickerClips().then(() => {
                    if (addMenu === pop) fill();
                }, () => {});
                container.appendChild(pop);
                setTimeout(() => document.addEventListener(
                    "click", closeAddMenu, { once: true }), 0);
            }
            addBtn.addEventListener("click", (ev) => {
                ev.stopPropagation();
                openAddMenu();
            });

            // ---- external drops into the strip ------------------------
            // OS files and Artius cards, resolved with the loader drop
            // handler's trust rules (helpers imported from
            // h3_mctx_drop.js): content already in output is referenced
            // IN PLACE (sidecar-bearing copy preferred, lineage kept);
            // anything else uploads to output/dropped. Internal reorder
            // drags carry TL_DRAG_MIME and are NOT ours here. Being a
            // DOM element, the strip sees these drops before the canvas
            // and before Artius's node-spawning bridge.
            function externalDragKind(ev) {
                const types = Array.from(ev.dataTransfer?.types ?? []);
                if (types.includes(TL_DRAG_MIME)) return null;
                if (types.includes(ARTIUS_MIME)) return "artius";
                if (types.includes("Files")) return "files";
                return null;
            }
            function gapFromX(clientX) {
                // insertion gap by block midpoints, in screen px (both
                // sides of the compare share the zoom scale)
                for (let k = 0; k < blockEls.length; k++) {
                    const r = blockEls[k].getBoundingClientRect();
                    if (clientX < r.left + r.width / 2) return k;
                }
                return blockEls.length;
            }
            function showDropLineAtGap(j) {
                let cx;
                if (!blockEls.length) {
                    cx = 4.5;
                } else if (j >= blockEls.length) {
                    const b = blockEls[blockEls.length - 1];
                    cx = b.offsetLeft + b.offsetWidth + 1.5;
                } else {
                    cx = blockEls[j].offsetLeft - 1.5;
                }
                dropLine.style.left = Math.max(0, cx - 4.5) + "px";
                dropLine.style.display = "block";
            }
            // the timeline's reference resolution: the first clip that
            // knows its dimensions. Null when the strip is empty or
            // nothing is probeable (then anything goes).
            async function timelineDims() {
                const entries = tlParseSequence(seqWidget.value);
                for (const e of entries) {
                    const d = await tlClipDims(e.clip);
                    if (d) return d;
                }
                return null;
            }
            function checkDims(dims, want, what) {
                // reject only on a CONFIRMED mismatch; an unprobeable
                // clip still gets in (assemble verifies for real)
                if (want && dims &&
                        (dims[0] !== want[0] || dims[1] !== want[1])) {
                    throw new Error(
                        `${what} is ${dims[0]}x${dims[1]} but this `
                        + `timeline is ${want[0]}x${want[1]} -- chained `
                        + `clips must share one resolution`);
                }
            }
            async function resolveArtiusClip(assets, want) {
                const asset = assets.find(isArtiusVideo);
                if (!asset) return null;
                // consumed: Artius's dragend fallback must not re-use it
                window.__tsArtiusDraggedAsset = "";
                if (asset.root_id === "output" ||
                        asset.scope === "output") {
                    const rel = artiusRelativePath(asset);
                    if (!rel) {
                        throw new Error("could not resolve the Artius "
                            + "asset's output path");
                    }
                    checkDims(await tlClipDims(rel), want,
                              rel.split("/").pop());
                    return rel;
                }
                const src = asset.file_url || (asset.id != null
                    ? `${ARTIUS_ROUTE_BASE}/file?id=` +
                        encodeURIComponent(String(asset.id))
                    : "");
                if (!src) {
                    throw new Error("Artius asset has no fetchable "
                        + "source");
                }
                const resp = await api.fetchApi(src);
                if (!resp.ok) {
                    throw new Error(`Artius fetch ${resp.status}`);
                }
                const blob = await resp.blob();
                const file = new File(
                    [blob], String(asset.filename || "clip.mp4"),
                    { type: blob.type || "video/mp4" });
                // resolution gate BEFORE the copy lands in output
                checkDims(await tlFileDims(file), want, file.name);
                const up = await uploadToOutput(file, false);
                tldbg("copied Artius asset to output:", up.name,
                      "(no sidecar travels this path)");
                return `${DROP_SUBFOLDER}/${up.name}`;
            }
            async function resolveFileClip(files, want) {
                const video = files.find((f) =>
                    DROP_VIDEO_RE.test(f.name) ||
                    f.type?.startsWith("video/"));
                if (!video) return null;
                // resolution gate first: a mismatch rejects before any
                // scan or upload happens
                checkDims(await tlFileDims(video), want, video.name);
                // in-place match: live output scan + size confirmation,
                // sidecar-bearing copy preferred
                let values = [];
                try {
                    values = await tlFetchClipList();
                } catch (err) {
                    tldbg("live clip list failed for drop match:", err);
                }
                const confirmed = [];
                for (const v of values) {
                    if (String(v).split("/").pop() !== video.name) {
                        continue;
                    }
                    if (await serverFileSize(v) === video.size) {
                        confirmed.push(v);
                    }
                }
                for (const v of confirmed) {
                    if (await serverFileSize(dropBaseName(v) +
                            DROP_SIDECAR_SUFFIX) !== null) {
                        tldbg("drop matched sidecar-bearing clip:", v);
                        return v;
                    }
                }
                if (confirmed.length) return confirmed[0];
                // fresh upload; a matching sidecar in the same drop
                // rides along with the basenames kept in lock step
                const sidecar = files.find((f) => f.name.toLowerCase()
                    .endsWith(DROP_SIDECAR_SUFFIX));
                const paired = sidecar &&
                    dropBaseName(dropBaseName(sidecar.name)) ===
                        dropBaseName(video.name);
                const up = await uploadToOutput(video, false);
                if (paired) {
                    await uploadToOutput(sidecar, true,
                        dropBaseName(up.name) + DROP_SIDECAR_SUFFIX);
                    tldbg("uploaded clip pair:", up.name);
                } else {
                    tldbg("uploaded video:", up.name, "(no sidecar in "
                          + "the drop; joins will be butt joins)");
                }
                return `${DROP_SUBFOLDER}/${up.name}`;
            }
            async function handleExternalDrop(j, artiusAssets, files) {
                try {
                    const want = await timelineDims();
                    const clip = artiusAssets
                        ? await resolveArtiusClip(artiusAssets, want)
                        : await resolveFileClip(files, want);
                    if (!clip) return;
                    const ls = currentLines();
                    const map = entryLines();
                    const at = j >= map.length ? ls.length : map[j];
                    ls.splice(at, 0, clip);
                    setSequence(ls.join("\n"));
                    tldbg("external drop inserted", clip, "at gap", j);
                } catch (err) {
                    console.error(
                        "[obvpm-h3-timeline] drop failed:", err);
                    try {
                        app.extensionManager?.toast?.add?.({
                            severity: "error",
                            summary: "H3 Timeline drop",
                            detail: String(err), life: 6000,
                        });
                    } catch (e2) { /* toast API absent */ }
                }
            }
            strip.addEventListener("dragover", (ev) => {
                if (!externalDragKind(ev)) return;
                ev.preventDefault();
                ev.stopPropagation();
                ev.dataTransfer.dropEffect = "copy";
                showDropLineAtGap(gapFromX(ev.clientX));
            });
            strip.addEventListener("dragleave", (ev) => {
                if (!externalDragKind(ev)) return;
                if (ev.relatedTarget &&
                        strip.contains(ev.relatedTarget)) {
                    return;
                }
                dropLine.style.display = "none";
            });
            strip.addEventListener("drop", (ev) => {
                const kind = externalDragKind(ev);
                if (!kind) return;
                ev.preventDefault();
                ev.stopPropagation();
                dropLine.style.display = "none";
                const j = gapFromX(ev.clientX);
                const assets = kind === "artius"
                    ? readArtiusAssets(ev) : null;
                void handleExternalDrop(j, assets,
                    Array.from(ev.dataTransfer?.files ?? []));
            });

            // ---- preview playback -------------------------------------
            let playlist = [];
            let lastPlaylistSig = "";
            let playIdx = -1;
            let act = 0;      // which of the two videos is on screen
            let rafId = null;

            // The user's mute choice, tracked separately from autoplay
            // fallback muting -- copying .muted between the twin videos
            // let one fallback mute stick forever (the self-muting bug).
            let wantMuted = false;
            for (const v of vids) {
                v.addEventListener("volumechange", () => {
                    if (v === vids[act] && !v._progMute) {
                        wantMuted = v.muted;
                    }
                });
            }
            function tryPlay(v) {
                v._progMute = true;
                v.muted = wantMuted;
                setTimeout(() => { v._progMute = false; }, 60);
                v.play().catch(() => {
                    // autoplay policy: retry muted, but never let this
                    // fallback become the remembered preference
                    v._progMute = true;
                    v.muted = true;
                    setTimeout(() => { v._progMute = false; }, 60);
                    v.play().catch(() => {});
                });
            }

            function showActive() {
                vids.forEach((v, k) => {
                    const on = k === act;
                    v.controls = on;
                    v.style.opacity = on ? "1" : "0";
                    v.style.pointerEvents = on ? "auto" : "none";
                });
            }
            function preloadInto(v, item) {
                // Whatever this element held, it is not the full build
                // any more. `_single` is how ensureSingleSrc decides the
                // build is ALREADY loaded, and it was only ever set: cut
                // a clip and drag the cut back, and the same build (same
                // content key, same URL) is adopted again, the marker
                // still matches, the src is never restored -- and the
                // scrub seeks the full build's time inside the clip.
                v._single = null;
                v._item = item ?? null;
                if (!item || item.gap != null) {   // nothing to load
                    v._item = null;
                    v.removeAttribute("src");
                    v.load();
                    return;
                }
                v.src = api.apiURL(viewRoute(item.clip));
                v.addEventListener("loadedmetadata", () => {
                    // guard: a later preload may have replaced the src
                    if (v._item === item) {
                        v.currentTime = item.enter / TL_FPS;
                    }
                }, { once: true });
            }
            function globalFrame() {
                const v = vids[act];
                if (usingSingle && single) return v.currentTime * TL_FPS;
                if (playIdx >= 0 && playlist[playIdx]) {
                    return cumStarts[playIdx] + v.currentTime * TL_FPS -
                        playlist[playIdx].enter;
                }
                return null;
            }
            // Until a seek actually lands, the video's currentTime is
            // still the OLD position -- drawing the playhead from it
            // makes the line flash backwards on every cross-clip jump.
            // seekTargetF overrides the display until the seek settles.
            let seekTargetF = null;
            for (const v of vids) {
                v.addEventListener("seeked", () => {
                    // clear the override only when the seek actually
                    // LANDED at the target -- the metadata enter-seek
                    // of a still-loading clip also fires 'seeked', and
                    // clearing on it made the playhead jump back to
                    // the stale position on every watcher tick
                    if (v !== vids[act] || seekTargetF === null) return;
                    const gf = globalFrame();
                    if (gf !== null &&
                            Math.abs(gf - seekTargetF) < 1) {
                        seekTargetF = null;
                    }
                });
            }
            function displayFrame() {
                if (seekTargetF !== null) {
                    const gf = globalFrame();
                    if (gf !== null && Math.abs(gf - seekTargetF) < 1) {
                        seekTargetF = null;
                        return gf;
                    }
                    return seekTargetF;
                }
                return globalFrame();
            }
            // Empty space in QUICK play: there is no file to roll, so the
            // videos are paused and hidden (the wrap behind them is
            // black, which is exactly what the built preview shows) and
            // the playhead is driven by the clock until the gap is spent.
            let gapTimer = null;
            let gapRaf = 0;
            function playGap(i, offset = 0, autoplay = true) {
                const item = playlist[i];
                const total = Math.max(0, (item.gap ?? 0) - offset);
                for (const v of vids) {
                    v.pause();
                    v.style.visibility = "hidden";
                }
                highlight(i);
                seekTargetF = null;
                setPlayhead(cumStarts[i] + offset);
                // Load the clip on the far side WHILE the blank runs.
                // Without this the gap ends onto an empty <video>, which
                // has to start loading from scratch -- playback stalls
                // there instead of continuing across.
                preloadInto(vids[1 - act],
                            playlist[i + 1] ?? (loopOn() ? playlist[0]
                                                        : undefined));
                if (!autoplay) return;
                const t0 = performance.now();
                const tick = () => {
                    const done = (performance.now() - t0) / 1000 * TL_FPS;
                    setPlayhead(cumStarts[i] + offset + Math.min(done, total));
                    if (done < total) gapRaf = requestAnimationFrame(tick);
                };
                gapRaf = requestAnimationFrame(tick);
                gapTimer = setTimeout(() => {
                    cancelAnimationFrame(gapRaf);
                    for (const v of vids) v.style.visibility = "";
                    playAt(i + 1);
                }, (total / TL_FPS) * 1000);
            }

            // ---- level lock, in quick playback ------------------------
            // Quick play runs the ORIGINAL files through <video>, so it
            // cannot re-encode -- but the correction is a per-frame gain,
            // and CSS brightness() is a linear multiply in the same
            // space. The server sends one luma gain per frame (the three
            // channel gains agree to ~0.001 on real joins), and this
            // rides the playhead loop that already exists.
            let seamFix = {};
            let seamFixSig = null;
            function levelFilterFor(idx, timeSec) {
                if (usingSingle) return "";      // already baked into the build
                if (widgetValue("level_lock", true) === false) return "";
                const g = seamFix[idx]?.gains;
                if (!g || !g.length) return "";
                const item = playlist[idx];
                if (!item) return "";
                const f = Math.round(timeSec * TL_FPS) - (item.enter || 0);
                if (f < 0 || f >= g.length) return "";
                const k = Number(g[f]);
                return k && Math.abs(k - 1) > 0.0005
                    ? `brightness(${k.toFixed(5)})` : "";
            }
            function paintLevelFilter() {
                const v = vids[act];
                if (!v) return;
                const want = playIdx >= 0
                    ? levelFilterFor(playIdx, v.currentTime || 0) : "";
                if (v.style.filter !== want) v.style.filter = want;
                const other = vids[1 - act];
                if (other && other.style.filter) other.style.filter = "";
            }
            function clearLevelFilter() {
                for (const v of vids) {
                    if (v.style.filter) v.style.filter = "";
                }
            }
            let seamSeq = 0;
            async function loadSeamLevels(entries) {
                // an empty strip has no joins to measure, and asking
                // anyway makes the server refuse an empty sequence --
                // once per repaint
                if (!entries.length) return;
                // one request per distinct sequence: the measurement
                // decodes clips server-side, so re-asking on every
                // repaint would be expensive for an unchanged strip
                const sig = JSON.stringify([
                    entries.map((e) => [e.clip, e.gap, e.enter, e.exit]),
                    widgetValue("level_lock_frames", 12)]);
                if (sig === seamFixSig) return;
                const mine = ++seamSeq;
                try {
                    const resp = await api.fetchApi("/obvpm/h3/seam_levels", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            sequence: seqWidget.value,
                            level_lock_frames:
                                widgetValue("level_lock_frames", 12),
                        }),
                    });
                    if (!resp.ok || mine !== seamSeq) return;
                    const raw = await resp.json();
                    if (mine !== seamSeq || raw?.error) return;
                    seamFix = {};
                    for (const [k, v] of Object.entries(raw)) {
                        seamFix[Number(k)] = v;
                    }
                    seamFixSig = sig;
                    paintSeamTitles();
                } catch (err) {
                    /* measurement is advisory: playback must not depend
                       on it, and a failed probe simply shows nothing */
                }
            }
            function seamNote(i) {
                const s = seamFix[i];
                if (!s) return "";
                const bits = [`seam: ${s.reads_as}`];
                if (s.step != null) bits.push(`step ${s.step > 0 ? "+" : ""}`
                    + `${s.step} (noise ${s.local})`);
                if (s.fixable) {
                    bits.push("fixable -- every take off this parent "
                              + "carries the same step, so re-rolling "
                              + "will not change it");
                }
                return "\n" + bits.join("\n");
            }
            function paintSeamTitles() {
                for (const el of linkEls) {
                    const i = Number(el.dataset?.obvpmSeam);
                    if (!Number.isFinite(i) || !seamFix[i]) continue;
                    const base = el.dataset.obvpmBaseTitle ?? el.title ?? "";
                    el.dataset.obvpmBaseTitle = base;
                    el.title = base + seamNote(i);
                }
            }

            function watchBoundary() {
                cancelAnimationFrame(rafId);
                const tick = () => {
                    const item = playlist[playIdx];
                    const v = vids[act];
                    if (!item) return;
                    setPlayhead(displayFrame());
                    paintLevelFilter();
                    // half a frame early beats a quarter second late
                    if (item.exit !== null &&
                        v.currentTime >= item.exit / TL_FPS -
                            0.5 / TL_FPS) {
                        playAt(playIdx + 1);
                        return;
                    }
                    rafId = requestAnimationFrame(tick);
                };
                rafId = requestAnimationFrame(tick);
            }
            function playAt(i, offset = 0, autoplay = true) {
                // a looping cut wraps: running off the end starts over,
                // through the wrap cut (already in the playlist's
                // enters/exits), which is the seam to be listening for
                if (i >= playlist.length && i > 0 && playlist.length
                        && loopOn()) {
                    playAt(0, 0, autoplay);
                    return;
                }
                if (i < 0 || i >= playlist.length) {
                    // Running off the end must STOP the picture. Without
                    // this the last clip's <video> kept rolling past its
                    // exit cut to the file's real end -- the cut looked
                    // ignored in quick play even though the model had it.
                    playIdx = -1;
                    cancelAnimationFrame(rafId);
                    clearTimeout(gapTimer);
                    for (const v of vids) v.pause();
                    clearLevelFilter();
                    return;
                }
                usingSingle = false;
                playIdx = i;
                clearTimeout(gapTimer);
                cancelAnimationFrame(gapRaf);
                for (const v of vids) v.style.visibility = "";
                if (playlist[i].gap != null) {
                    playGap(i, offset, autoplay);
                    return;
                }
                seekTargetF = cumStarts[i] + offset;
                const item = playlist[i];
                const standby = vids[1 - act];
                if (standby._item === item && standby.readyState >= 1) {
                    vids[act].pause();
                    act = 1 - act;    // the hop: already loaded + seeked
                } else if (vids[act]._item !== item) {
                    // NOT already loading this item: repeated calls
                    // while scrubbing must not re-set src (that
                    // restarts the load every pointermove and it never
                    // finishes)
                    preloadInto(vids[act], item);
                }
                const v = vids[act];
                const want = (item.enter + offset) / TL_FPS;
                if (v.readyState >= 1) {
                    if (Math.abs(v.currentTime - want) > 0.02) {
                        v.currentTime = want;
                    }
                } else {
                    // still loading: land on the requested spot once
                    // metadata is in (last queued target wins)
                    v.addEventListener("loadedmetadata", () => {
                        if (v._item === item) v.currentTime = want;
                    }, { once: true });
                }
                showActive();
                highlight(i);
                if (autoplay) tryPlay(v);
                preloadInto(vids[1 - act],
                            playlist[i + 1] ?? (loopOn() ? playlist[0]
                                                        : undefined));
                watchBoundary();
            }
            for (const v of vids) {
                v.addEventListener("ended", () => {
                    // MODE check, not existence: `single` is non-null
                    // whenever a full build was merely adopted, and
                    // that must not stop QUICK playback from advancing
                    if (vids[act] === v && playIdx >= 0 &&
                            !usingSingle) {
                        playAt(playIdx + 1);
                    }
                });
            }

            // ---- truly seamless preview: server-built temp smart-cut --
            // One real file has no seams to hide (identical content to
            // the export -- same code path). The per-clip double buffer
            // above is the instant "quick" mode and the fallback when
            // the route is unavailable.
            let single = null;       // {url, starts:[frame...]}
            let usingSingle = false; // which mode is on screen
            function setChip(text, color) {
                stateChip.textContent = text;
                stateChip.style.color = color;
            }
            function widgetValue(name, fallback) {
                return node.widgets?.find((w) => w.name === name)
                    ?.value ?? fallback;
            }
            // 19 is the Timeline widget's own default, and the crf
            // our takes are written at -- see
            // nodes_assemble.source_crf for why the two must not drift
            // apart (a mismatch loses the smart cut on every splice).
            function crfValue() { return widgetValue("crf", 19); }
            // base_folder is commonly DRIVEN rather than typed -- a
            // subgraph widget feeding several nodes at once, which is how
            // h3_obvpm_r2v_test_import.json sets the project folder. A
            // widget's value is dead the moment its socket is connected,
            // so reading it here returned the stale default: "+ add"
            // listed every folder, and previews/exports were written to
            // the output root. Follow the wire instead, exactly as the
            // Result Preview resolves a save node's folder.
            let folderUnreadable = false;
            function folderValue() {
                const v = rpWidgetValue(node, "base_folder");
                if (v == null) {
                    // Wired, but the far end cannot be read. Unscoped is
                    // the honest answer -- offering too many clips beats
                    // silently scoping to the wrong folder -- but say it
                    // once, or it just looks broken.
                    if (!folderUnreadable) {
                        folderUnreadable = true;
                        console.warn("[obvpm h3] Timeline #" + node.id +
                            ": base_folder is wired from something this "
                            + "widget cannot read, so clip scoping is "
                            + "off. Type the folder in directly to scope "
                            + "it.");
                    }
                    return "";
                }
                return String(v).trim().replace(/^\/+|\/+$/g, "");
            }
            // The seam-repair settings, sent with every build. They are
            // part of the server's cache key, so changing one rebuilds
            // rather than serving a preview made under the old setting.
            // Driven off TL_SEAM_ORDER rather than written out, so the
            // set of settings the build is told about cannot drift from
            // the set a per-seam override may name.
            function fixOpts() {
                const out = {};
                for (const k of TL_SEAM_ORDER) {
                    const d = TL_FIX_DEFAULTS[k];
                    const v = widgetValue(k, d);
                    // the fallback IS the default, so one rule reads
                    // both polarities of boolean correctly
                    out[k] = TL_SEAM_KEYS[k] === "bool"
                        ? v !== false : (Number(v) || d);
                }
                return out;
            }
            async function previewApi(probe) {
                // nothing on the strip = nothing to build. Returning
                // null rather than asking keeps an empty timeline from
                // refusing on every repaint; callers read it as "no
                // preview", which is what an empty strip has.
                if (!tlParseSequence(seqWidget.value).length) return null;
                const resp = await api.fetchApi("/obvpm/h3/preview_cut", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        sequence: seqWidget.value, crf: crfValue(),
                        base_folder: folderValue(),
                        preview_filename:
                            widgetValue("preview_filename", ""),
                        probe, ...fixOpts(),
                    }),
                });
                if (!resp.ok) throw new Error(await resp.text());
                return resp.json();
            }
            function adoptSingle(d) {
                // one overwritten file per node: the URL only changes via
                // the content key, so ?v= is what defeats the browser cache
                single = {
                    url: api.apiURL(
                        `/view?filename=${encodeURIComponent(d.filename)}` +
                        `&subfolder=${encodeURIComponent(d.subfolder)}` +
                        `&type=${encodeURIComponent(d.type ?? "output")}` +
                        `&v=${encodeURIComponent(d.v ?? "")}`),
                    starts: d.starts,
                };
                setChip("full built ✓", "#4d9960");
                updateModeButtons(); // full supersedes quick
            }
            let exportSeq = 0;
            async function doExport() {
                const seq = ++exportSeq;
                exportBtn.disabled = true;
                exportBtn.textContent = "⏳ export";
                progress.arm("preparing…");
                try {
                    const resp = await api.fetchApi("/obvpm/h3/export", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            sequence: seqWidget.value, crf: crfValue(),
                            base_folder: folderValue(),
                            preview_filename:
                                widgetValue("preview_filename", ""),
                            export_filename_prefix: widgetValue(
                                "export_filename_prefix", "full"),
                            ...fixOpts(),
                        }),
                    });
                    if (!resp.ok) throw new Error(await resp.text());
                    const d = await resp.json();
                    tldbg("exported:", d.path);
                    exportBtn.textContent = "✓ exported";
                    exportBtn.title = "Last export: " + d.path;
                    app.extensionManager?.toast?.add?.({
                        severity: "success", summary: "Cut exported",
                        detail: d.path, life: 4000,
                    });
                    // the export ensured a current build server-side;
                    // adopt it so "▶ full" is instant now
                    void probeFull();
                } catch (err) {
                    tldbg("export failed:", err);
                    exportBtn.textContent = "✗ export";
                    app.extensionManager?.toast?.add?.({
                        severity: "error", summary: "Export failed",
                        detail: String(err?.message ?? err), life: 6000,
                    });
                } finally {
                    progress.hide();
                }
                exportBtn.disabled = false;
                setTimeout(() => {
                    if (seq === exportSeq) exportBtn.textContent = "⇪ export";
                }, 2500);
            }
            exportBtn.addEventListener("click", () => void doExport());
            let probeSeq = 0;
            async function probeFull() {
                const seq = ++probeSeq;
                try {
                    const d = await previewApi(true);
                    if (seq !== probeSeq) return;
                    if (d?.cached) adoptSingle(d);
                    else setChip("", "#9aa1ac");
                } catch {
                    if (seq === probeSeq) setChip("", "#9aa1ac");
                }
            }
            function watchSingle() {
                cancelAnimationFrame(rafId);
                const tick = () => {
                    if (!single || !usingSingle) return;
                    const f = displayFrame() ?? 0;
                    setPlayhead(f);
                    let idx = 0;
                    single.starts.forEach((s, k) => {
                        if (f >= s - 0.5) idx = k;
                    });
                    highlight(idx);
                    rafId = requestAnimationFrame(tick);
                };
                rafId = requestAnimationFrame(tick);
            }
            function ensureSingleSrc(cb) {
                const v = vids[act];
                if (v._single !== single.url) {
                    v._single = single.url;
                    v._item = null;
                    v.src = single.url;
                    v.addEventListener("loadedmetadata", cb, { once: true });
                } else {
                    cb();
                }
                showActive();
            }
            // Show one clip's OWN frame, whatever the current cut says.
            // playAt seeks relative to the clip's enter, and a START cut
            // being loosened asks for a frame BEFORE it -- that would
            // clamp to zero and show the wrong picture. So let playAt
            // put the right file in the element, then set the time
            // directly. Used while dragging a cut handle.
            function seekSourceFrame(idx, srcF) {
                const item = playlist[idx];
                if (!item || item.gap != null) return;
                usingSingle = false;
                const v = vids[act];
                const t = Math.max(0, srcF) / TL_FPS;
                const put = () => {
                    // a later request may have replaced the src
                    if (v._item !== item) return;
                    if (Math.abs(v.currentTime - t) > 0.01) {
                        v.currentTime = t;
                    }
                };
                if (v._item !== item) {
                    // preloadInto seeks to the clip's enter on its own
                    // loadedmetadata; ours is registered after, so it
                    // lands second and wins
                    preloadInto(v, item);
                    v.addEventListener("loadedmetadata", put,
                                       { once: true });
                } else if (v.readyState >= 1) {
                    put();
                } else {
                    v.addEventListener("loadedmetadata", put,
                                       { once: true });
                }
                // Empty space is shown by HIDING both elements and
                // letting the black wrap through (playGap), and
                // showActive only touches opacity -- so starting a drag
                // while the cursor sat on a gap left the picture hidden
                // and the preview stayed black. playAt used to clear
                // this; nothing did once the scrub stopped calling it.
                for (const vv of vids) vv.style.visibility = "";
                showActive();
            }

            // Where the time cursor belongs once a cut lands.
            //
            // An END cut leaves the clip's start alone and only changes
            // its length, so the frame to sit on is the last one that
            // still plays: an exit of N means N does NOT play.
            //
            // A START cut is different, and this is what the first
            // attempt got wrong. Absorbing the trimmed frames into the
            // hole in FRONT of the clip moves the clip along the
            // timeline by exactly the change in that hole -- that is
            // the whole point of absorbing, it is what keeps the clip's
            // OTHER end still. Assuming the start stayed put landed the
            // cursor where the clip used to begin, which is inside the
            // hole and reads as jumping backwards.
            //
            // `nowSpace` null means no hole is involved at all (a linked
            // head cut opens none), and then nothing moves.
            function cutLanding(at0, isExit, enter, latest,
                                wasSpace, nowSpace) {
                if (isExit) return at0 + Math.max(0, latest - enter - 1);
                return at0 + ((nowSpace ?? wasSpace) - wasSpace);
            }

            function clipAt(f) {
                let k = 0;
                cumStarts.forEach((s, i) => { if (f >= s) k = i; });
                return k;
            }
            function playSingleFrame(f) {
                usingSingle = true;
                seekTargetF = f;
                ensureSingleSrc(() => {
                    const v = vids[act];
                    v.currentTime = f / TL_FPS;
                    tryPlay(v);
                    watchSingle();
                });
            }
            function playSingle(i) {
                playSingleFrame(single.starts[i] ?? 0);
            }
            async function ensureFullBuilt() {
                if (single) return true;
                fullBtn.disabled = true;
                fullBtn.textContent = "⏳ building";
                setChip("building…", "#c9973b");
                // armed at zero: a build served straight from cache
                // reports nothing at all, and a blank bar looks stalled
                progress.arm("preparing…");
                try {
                    const d = await previewApi(false);
                    if (d) adoptSingle(d);
                    else setChip("nothing to build", "#9aa1ac");
                } catch (err) {
                    tldbg("full preview failed; quick fallback:", err);
                    setChip("✗ build failed", "#de5b5b");
                } finally {
                    progress.hide();
                }
                fullBtn.disabled = false;
                fullBtn.textContent = "▶ full";
                return !!single;
            }
            async function playFull(i) {
                if (await ensureFullBuilt()) playSingle(i);
                else playAt(i);
            }
            // Start at a timeline FRAME rather than a clip: pressing play
            // should carry on from where the playhead sits.
            async function playFullFrame(f) {
                if (await ensureFullBuilt()) {
                    playSingleFrame(f);
                    return;
                }
                if (!cumStarts.length) return;
                const k = clipAt(f);
                playAt(k, f - cumStarts[k]);
            }
            // Where playback picks up. At (or one frame from) the end
            // there is nothing left to resume into, so rewind instead.
            function resumeFrame() {
                const f = displayFrame() ?? heldFrame;
                const last = Math.max(totalFrames - 1, 0);
                if (f == null || !Number.isFinite(f) || f >= last - 1) {
                    return 0;
                }
                return Math.max(0, Math.round(f));
            }
            // quick/full buttons double as play/pause toggles for their
            // own mode; pressing the other mode's button switches mode
            function activeStarted() {
                return usingSingle
                    ? (single && vids[act]._single === single.url)
                    : playIdx >= 0;
            }
            // Once a full build exists it supersedes quick -- same cut,
            // seamless -- so the lesser button just adds noise. It stays
            // while quick is the mode actually playing, or there would be
            // no way to pause it.
            function updateModeButtons() {
                const quickInUse = !usingSingle && playIdx >= 0;
                quickBtn.style.display = single && !quickInUse ? "none" : "";
            }
            function updateButtons() {
                const playing = activeStarted() && !vids[act].paused;
                quickBtn.textContent =
                    !usingSingle && playing ? "⏸ quick" : "▶ quick";
                if (!fullBtn.disabled) {
                    fullBtn.textContent =
                        usingSingle && playing ? "⏸ full" : "▶ full";
                }
                updateModeButtons();
            }
            for (const v of vids) {
                v.addEventListener("play", updateButtons);
                v.addEventListener("pause", updateButtons);
            }
            quickBtn.addEventListener("click", () => {
                const v = vids[act];
                if (!usingSingle && playIdx >= 0) {
                    if (v.paused) tryPlay(v);
                    else v.pause();
                    return;
                }
                if (!cumStarts.length) return;
                const f = resumeFrame();
                const k = clipAt(f);
                playAt(k, f - cumStarts[k]);
            });
            fullBtn.addEventListener("click", () => {
                const v = vids[act];
                if (usingSingle && single && v._single === single.url) {
                    if (v.paused) {
                        tryPlay(v);
                        watchSingle();
                    } else {
                        v.pause();
                    }
                    return;
                }
                void playFullFrame(resumeFrame());
            });

            // ---- ruler scrubbing (both modes, keeps paused state) -----
            function seekGlobal(f) {
                f = Math.max(0, Math.min(f, Math.max(totalFrames - 1, 0)));
                const wasPlaying = activeStarted() && !vids[act].paused;
                seekTargetF = f;
                if (usingSingle && single) {
                    ensureSingleSrc(() => {
                        vids[act].currentTime = f / TL_FPS;
                        watchSingle();
                    });
                    setPlayhead(f);
                    return;
                }
                let k = 0;
                cumStarts.forEach((s, i2) => { if (f >= s) k = i2; });
                const off = f - cumStarts[k];
                if (k === playIdx && vids[act].readyState >= 1) {
                    vids[act].currentTime =
                        (playlist[k].enter + off) / TL_FPS;
                } else {
                    // scrubbing must not start playback by itself
                    playAt(k, off, wasPlaying);
                }
                setPlayhead(f);
            }
            let scrubbing = false;
            function scrubTo(ev) {
                // rect is in SCREEN px but offsetLeft math is LAYOUT px;
                // the canvas zoom CSS-scales the widget, so divide it out
                const r = ruler.getBoundingClientRect();
                const scale = r.width / (ruler.offsetWidth || 1) || 1;
                const x = (ev.clientX - r.left) / scale + strip.scrollLeft;
                seekGlobal(xToFrame(x));
            }
            ruler.addEventListener("pointerdown", (ev) => {
                if (!totalFrames) return;
                ev.preventDefault();
                ev.stopPropagation();
                scrubbing = true;
                ruler.setPointerCapture(ev.pointerId);
                if (single && playIdx < 0 && !usingSingle) {
                    usingSingle = true; // prefer the seamless file
                }
                scrubTo(ev);
            });
            ruler.addEventListener("pointermove", (ev) => {
                if (scrubbing) scrubTo(ev);
            });
            ruler.addEventListener("pointerup", () => {
                scrubbing = false;
            });

            let lastHighlight = -1;
            let infoIdx = -2; // which clip the info bar currently shows
            let linkTinted = false;
            function clearLinkTint() {
                if (!linkTinted) return;
                linkTinted = false;
                highlight(lastHighlight);
            }
            function tintLinkBlock(el) {
                el.style.background = "#1e5231";
                el.style.color = "#e9f5ec";
                linkTinted = true;
            }
            function highlight(active) {
                lastHighlight = active;
                blockEls.forEach((el, i) => {
                    // Empty space carries no fill in EITHER state. This
                    // runs on every render and on every playback frame,
                    // so painting gaps here quietly put the background
                    // back however the gap block was built. Selection
                    // shows in the ink instead of a fill.
                    if (el.dataset?.obvpmGap) {
                        el.style.background = "none";
                        el.style.color = i === active ? PAL.text : PAL.sub;
                        return;
                    }
                    el.style.background =
                        i === active ? PAL.active : PAL.rest;
                    el.style.color =
                        i === active ? PAL.activeText : PAL.text;
                });
                // the highlighted clip IS the selection -- the info bar
                // follows the playhead/click, no separate outline.
                // ONLY on index change: the playback watchers call
                // highlight() every rAF frame, and rebuilding the bar
                // that often destroys its buttons between mousedown and
                // mouseup (clicks never land).
                if (infoIdx !== active) renderClipInfo();
            }

            // seam zones: the lineage connector is also a CONTROL. Hover
            // a gap between mctx clips to see the (potential) link;
            // click to pick a compatible clip -- insert one that links
            // here, or (when a link exists) replace either side while
            // keeping the link intact.
            let seamPopup = null;
            function closeSeamPopup() {
                seamPopup?.remove();
                seamPopup = null;
            }
            function linkedMeta(a, b) {
                // does b belong AFTER a? (b extends a, or a leads into
                // b) -- multi-pin aware via the lineage accessors
                if (!a || !b) return false;
                return tlExtendParent(b)?.id === a.self_id ||
                       tlPrependChild(a)?.id === b.self_id;
            }
            function drawLinks(entries) {
                linkEls.forEach((el) => el.remove());
                linkEls = [];
                closeSeamPopup();
                // the gap the NEXT RUN will attach at (extend = after
                // the pinned clip, prepend = before it) wears the pin
                // orange so the strip itself shows where generation
                // lands
                const pstD = shownPin();
                const pinGaps = new Set();
                if (pstD) {
                    const k = entries.findIndex(
                        (en) => en.clip === pstD.source);
                    if (k >= 0) {
                        pinGaps.add(pstD.role === "prepend"
                            ? k : k + 1);
                    }
                    if (pstD.role === "bridge") {
                        const k2 = entries.findIndex(
                            (en) => en.clip === pstD.source2);
                        if (k2 >= 0) pinGaps.add(k2);
                    }
                }
                // one zone builder for interior gaps AND the strip's two
                // ends (Le/Re null on the open side). A zone starts
                // inert; a linked seam enables immediately (replace
                // views always exist), an open gap enables only when
                // some picker clip could actually link there -- a popup
                // full of "none found" is not a control. The candidate
                // check is async but meta reads are cached after the
                // first pass; the zone upgrades in place unless the
                // strip was rebuilt meanwhile.
                // A seam has TWO edges meeting at it -- the left clip's
                // end and the right clip's start -- and at an open seam
                // both can move. They therefore get their own halves of
                // the zone rather than one winning: left half cuts the
                // left clip, right half cuts the right one. A seam
                // between two LINKED clips moves nothing, since that
                // join is what the sidecars record.
                const cutTargetsFor = (i) => ({
                    left: lastEntries?.[i - 1] &&
                        tlCanCut(lastEntries, i - 1, "exit", linkedMeta)
                        ? { idx: i - 1, side: "exit" } : null,
                    right: lastEntries?.[i] &&
                        tlCanCut(lastEntries, i, "enter", linkedMeta)
                        ? { idx: i, side: "enter" } : null,
                });

                // Which edge you are about to move, shown on the clip
                // itself -- a 3px bar down the edge that would travel.
                const markEdge = (target, on) => {
                    const b = blockEls?.[target?.idx];
                    if (!b) return;
                    b.style.boxShadow = on
                        ? (target.side === "exit"
                            ? `inset -3px 0 0 ${DROP_GREEN}`
                            : `inset 3px 0 0 ${DROP_GREEN}`)
                        : "";
                };

                const startCutDrag = (ev, target) => {
                    const e = lastEntries?.[target.idx];
                    const block = blockEls?.[target.idx];
                    if (!e || !block) return;
                    ev.preventDefault();
                    ev.stopPropagation();
                    // A drag is scrubbing, not playing. Stop the player
                    // first: left running, its advance logic carries on
                    // past the cut into the NEXT entry, and when that
                    // entry is empty space the picture goes black -- so
                    // the preview flickered between the clip, the cut
                    // and nothing at all.
                    cancelAnimationFrame(rafId);
                    clearTimeout(gapTimer);
                    cancelAnimationFrame(gapRaf);
                    for (const vv of vids) vv.pause();
                    const played = Math.max(1, (e.exit ?? e.frames) - e.enter);
                    // The SCALE, not the block's measured width.
                    // offsetWidth is an integer, while the real width is
                    // played * scale and fractional -- so a measured
                    // ratio is off by up to half a pixel per frame, and
                    // a long drag walks visibly away from the pointer.
                    const pxPerFrame = scaleNow()
                        || ((block.offsetWidth || 1) / played);
                    // clientX is SCREEN px and the scale is LAYOUT px;
                    // the canvas zoom is the ratio. Without dividing it
                    // out the cut ran ahead of the mouse whenever the
                    // graph was zoomed in.
                    const czoom = block.getBoundingClientRect().width
                        / (block.offsetWidth || 1) || 1;
                    const isExit = target.side === "exit";
                    const base = isExit ? (e.exit ?? e.frames) : e.enter;
                    // a cut may never cross the clip's other edge, and
                    // must leave at least one frame playing
                    const lo = isExit ? e.enter + 1 : 0;
                    const hi = isExit ? e.frames
                                      : (e.exit ?? e.frames) - 1;
                    const startX = ev.clientX;
                    let latest = base;

                    const guide = document.createElement("div");
                    Object.assign(guide.style, {
                        position: "absolute", top: "0", bottom: "0",
                        width: "2px", background: DROP_GREEN,
                        pointerEvents: "none", zIndex: "6",
                    });
                    const tag = document.createElement("div");
                    Object.assign(tag.style, {
                        position: "absolute", top: "2px",
                        transform: "translateX(-50%)",
                        background: DROP_GREEN, color: "#04220f",
                        font: "600 10px sans-serif", borderRadius: "3px",
                        padding: "0 4px", whiteSpace: "nowrap",
                        pointerEvents: "none", zIndex: "7",
                    });
                    // Show the frame the cut is ON, so a trim is judged
                    // by the picture rather than by a number. A START
                    // handle shows the clip's new FIRST frame; an END
                    // handle shows its new LAST one, which is the frame
                    // before the cut -- an exit of 60 means 60 plays and
                    // 60 does not.
                    //
                    // Seeking is expensive and pointermove is not, so
                    // the request rides one frame behind, coalesced.
                    // What the neighbouring hole will be once this cut
                    // lands, or null when no hole is involved. Mirrors
                    // commitCut exactly -- if the two ever disagree the
                    // label is lying, which is worse than not showing
                    // it at all.
                    const spaceAfterCut = () => {
                        const freed = isExit ? base - latest : latest - base;
                        const side = isExit
                            ? lastEntries[target.idx + 1]
                            : lastEntries[target.idx - 1];
                        if (side && side.gap != null) {
                            return Math.max(0, (Number(side.gap) || 0)
                                               + freed);
                        }
                        if (!isExit && freed > 0
                                && !linkedMeta(side?.meta, e.meta)) {
                            return freed;
                        }
                        return null;
                    };
                    let cutRaf = 0;
                    let cutWant = null;
                    const showCutFrame = () => {
                        cutWant = isExit ? latest - 1 : latest;
                        if (cutRaf) return;
                        cutRaf = requestAnimationFrame(() => {
                            cutRaf = 0;
                            if (cutWant == null) return;
                            seekSourceFrame(target.idx, cutWant);
                        });
                    };

                    // the valid stops, so the snap notches read as
                    // intentional rather than as a laggy drag
                    const ticks = [];
                    for (const f of tlCutStops(e.meta, e.frames, snapOn(), target.side)) {
                        if (f <= e.enter || f >= (e.exit ?? e.frames)) continue;
                        const t = document.createElement("div");
                        Object.assign(t.style, {
                            position: "absolute", top: "0", bottom: "0",
                            width: "1px", opacity: "0.35",
                            background: PAL.text, pointerEvents: "none",
                            left: ((f - e.enter) * pxPerFrame) + "px",
                        });
                        ticks.push(t);
                        block.appendChild(t);
                    }
                    const place = () => {
                        // The guide and its label live on the STRIP, not
                        // inside the block: the block clips its overflow,
                        // so a label centred on the clip's own left edge
                        // was half-eaten and looked like the clip beside
                        // it was covering it.
                        const px = (latest - e.enter) * pxPerFrame;
                        const gx = block.offsetLeft + px;
                        guide.style.left = gx + "px";
                        // the time cursor belongs ON the cut while you
                        // are placing it -- leaving it where it was made
                        // the strip look frozen mid-drag
                        cutPin = gx;
                        setPlayhead(null);
                        // Frame and time of the cut, then what it does to
                        // the empty space beside it -- in the SAME units
                        // and rounding the space itself is labelled with,
                        // so the two numbers can be compared instead of
                        // looking like they disagree.
                        // When a hole is involved, the seconds shown
                        // are THE HOLE'S -- same value, same rounding as
                        // the gap block will carry once this lands.
                        // Showing the cut's own timecode next to it read
                        // as two numbers disagreeing. With no hole in
                        // play there is nothing to match, so the cut's
                        // own position is the useful thing.
                        const space = spaceAfterCut();
                        tag.textContent = space == null
                            ? `${latest}f · ${(latest / TL_FPS).toFixed(2)}s`
                            : `${latest}f · `
                              + `${(space / TL_FPS).toFixed(1)}s space`;
                        // it is centred on the guide, so keep half of it
                        // clear of either end of the scrollable content
                        const half = (tag.offsetWidth || 60) / 2;
                        const far = Math.max(half,
                            (strip.scrollWidth || 0) - half - 2);
                        tag.style.left =
                            Math.min(Math.max(gx, half + 2), far) + "px";
                        showCutFrame();
                    };
                    strip.append(guide, tag);
                    // keep the edge marked for the whole drag, so which
                    // clip is moving stays obvious once the pointer has
                    // left the handle
                    block.style.boxShadow = isExit
                        ? `inset -3px 0 0 ${DROP_GREEN}`
                        : `inset 3px 0 0 ${DROP_GREEN}`;
                    place();

                    const onMove = (m) => {
                        const d = (m.clientX - startX) / czoom
                            / (pxPerFrame || 1);
                        const want = Math.min(hi, Math.max(lo, base + d));
                        const snapped = tlSnapCut(want, e.meta, snapOn(),
                                                  target.side);
                        latest = Math.min(hi, Math.max(lo, snapped));
                        place();
                    };
                    const onUp = () => {
                        document.removeEventListener("mousemove", onMove, true);
                        document.removeEventListener("mouseup", onUp, true);
                        cancelAnimationFrame(cutRaf);
                        cutPin = null;
                        guide.remove();
                        tag.remove();
                        ticks.forEach((t) => t.remove());
                        block.style.boxShadow = "";
                        // Land the time cursor ON the cut. refresh()
                        // restores it from displayFrame(), which reports
                        // wherever the preview was last SEEKED -- and
                        // showing the cut frame seeks to the clip's
                        // start, so without this the cursor jumped back
                        // there the moment the drop re-rendered.
                        if (latest !== base) {
                            const before = lastEntries[target.idx - 1];
                            const was = before && before.gap != null
                                ? (Number(before.gap) || 0) : 0;
                            const gcut = cutLanding(
                                cumStarts[target.idx] ?? 0, isExit,
                                e.enter, latest, was, spaceAfterCut());
                            heldFrame = gcut;
                            seekTargetF = gcut;
                        }
                        if (latest === base) {
                            // abandoned: put the time cursor back where
                            // it was, or it sits on a cut that never
                            // happened until the next render
                            setPlayhead(displayFrame() ?? heldFrame);
                            return;
                        }
                        // back at the clip's own edge = no cut at all,
                        // so the line loses its @ rather than pinning
                        // the value it already had
                        commitCut(target.idx, target.side, latest);
                    };
                    document.addEventListener("mousemove", onMove, true);
                    document.addEventListener("mouseup", onUp, true);
                };

                // Le/Re are the ENTRIES either side, null only where the
                // strip really ends. Deciding this from their metas
                // instead was the old bug: a hole has no meta, so a
                // boundary against empty space presented exactly like
                // the end of the timeline and said so.
                const addZone = (i, x, seam, Le, Re) => {
                    const hasLink = !!(seam && seam.kind !== "butt");
                    const zone = document.createElement("div");
                    Object.assign(zone.style, {
                        // starts below the blocks' top row so the zone
                        // never sits over (and steals clicks from) a
                        // block's ✕ button
                        position: "absolute", top: "18px", bottom: "0",
                        left: Math.max(0, x - 9) + "px", width: "18px",
                        zIndex: "4", cursor: "default",
                    });
                    const atPin = pinGaps.has(i);
                    // the pill is a SIBLING of the zone, centered on
                    // the strip's own middle (the zone starts below the
                    // blocks' ✕ row, so centering within it sat too
                    // low)
                    const pill = document.createElement("div");
                    Object.assign(pill.style, {
                        position: "absolute", top: "50%",
                        left: (x - 7) + "px",
                        transform: "translateY(-50%)", width: "14px",
                        height: atPin ? "13px" : "6px",
                        borderRadius: "3px", zIndex: "5",
                        // COLOUR carries what this boundary is; opacity
                        // carries nothing. It used to carry the same
                        // fact, and a translucent 6px bar over a hole
                        // with no fill of its own read as nothing there.
                        //
                        // The no-lineage fill is PAL.sub and NOT
                        // TL_COLORS.butt: butt is a fixed light grey,
                        // fine on a dark strip and invisible on a light
                        // one. PAL.sub inverts with the theme -- dark
                        // ink on a light page, light on a dark one --
                        // so it reads on both.
                        background: atPin ? TL_ROLE.extend.bg
                            : hasLink ? TL_COLORS[seam.kind] : PAL.sub,
                        boxShadow: "0 0 0 2px rgba(0,0,0,0.35)",
                        transition: "height 0.1s",
                        pointerEvents: "none",
                        display: "flex", alignItems: "center",
                        justifyContent: "center",
                    });
                    if (atPin) {
                        // the strip's "generation lands here" marker
                        pill.textContent = "+";
                        pill.style.color = "#fff";
                        pill.style.font = "bold 11px/13px sans-serif";
                    }
                    // an end zone with BOTH sides is the wrap of a loop
                    const wrapZone = !!(Le && Re && loopOn()
                        && (i === 0 || i === entries.length));
                    const edge = wrapZone
                        ? "loop: the end wraps onto the start (no link yet)"
                        : !Le ? "timeline start"
                        : !Re ? "timeline end"
                        : (Le.gap != null || Re.gap != null)
                            ? "edge of empty space"
                            : "no link";
                    const baseTitle = (hasLink
                        ? (wrapZone ? "loop: " : "") + seam.note : edge)
                        + (atPin
                           ? "\nnext run generates here" : "");
                    // always clickable: even with no linking clips on
                    // disk, "generate new" is a valid action whenever a
                    // side carries mctx (which gates zone creation)
                    zone.style.cursor = "pointer";
                    zone.title = baseTitle +
                        "\nclick: insert, swap or generate a linking "
                        + "clip";
                    // The link button is a BUTTON: it points, and it
                    // never starts a drag.
                    pill.style.cursor = "pointer";
                    pill.draggable = false;
                    pill.addEventListener("mousedown", (ev) => {
                        ev.stopPropagation();
                    });
                    // Cut handles: one per side, split above and below
                    // the pill so the link button keeps its own band.
                    const targets = cutTargetsFor(i);
                    const addHandle = (target, left, upper) => {
                        if (!target) return;
                        const own = blockEls[target.idx];
                        if (own && own.offsetWidth < CUT_HANDLE_MIN) return;
                        const h = document.createElement("div");
                        Object.assign(h.style, {
                            position: "absolute", left: left + "px",
                            width: "11px", zIndex: "5", cursor: "ew-resize",
                            top: upper ? "18px" : "calc(50% + 10px)",
                            bottom: upper ? "calc(50% + 10px)" : "0",
                        });
                        h.title = target.side === "exit"
                            ? "drag to cut the END of "
                              + lastEntries[target.idx].clip.split("/").pop()
                            : "drag to cut the START of "
                              + lastEntries[target.idx].clip.split("/").pop();
                        const hover = (on) => {
                            h.style.background = on
                                ? "rgba(47,206,104,0.20)" : "";
                            markEdge(target, on);
                        };
                        h.addEventListener("mouseenter", () => hover(true));
                        h.addEventListener("mouseleave", () => hover(false));
                        h.addEventListener("mousedown", (ev) => {
                            hover(false);
                            startCutDrag(ev, target);
                        });
                        // seam elements are absolutely positioned, so a
                        // ctrl+drag has to move them by hand -- and only
                        // the ones downstream of the clip being moved
                        h.dataset.obvpmSeam = String(i);
                        strip.appendChild(h);
                        linkEls.push(h);
                    };
                    for (const upper of [true, false]) {
                        addHandle(targets.left, Math.max(0, x - 11), upper);
                        addHandle(targets.right, x, upper);
                    }
                    // hover grows it; that is the whole affordance now
                    zone.addEventListener("mouseenter", () => {
                        pill.style.height = "11px";
                    });
                    zone.addEventListener("mouseleave", () => {
                        pill.style.height = atPin ? "13px" : "6px";
                    });
                    zone.addEventListener("click", (ev) => {
                        ev.stopPropagation();
                        void openSeamPopup(i, entries, hasLink, x);
                    });
                    zone.dataset.obvpmSeam = String(i);
                    pill.dataset.obvpmSeam = String(i);
                    strip.append(zone, pill);
                    linkEls.push(zone, pill);
                };
                entries.forEach((e, i) => {
                    if (!i) return;
                    const L = blockEls[i - 1];
                    const R = blockEls[i];
                    if (!L || !R) return;
                    const hasLink = e.seam && e.seam.kind !== "butt";
                    // EITHER side with mctx is enough: a chain ending
                    // against a no-mctx neighbor still deserves its
                    // insert/generate control on the mctx side
                    const offer = entries[i - 1].meta || e.meta;
                    if (!hasLink && !offer) return;
                    const x = (L.offsetLeft + L.offsetWidth +
                               R.offsetLeft) / 2;
                    addZone(i, x, e.seam, entries[i - 1], e);
                });
                // the strip's ends: prepend before the first clip /
                // extend after the last one ("the timeline end can't be
                // clicked" report) -- only for mctx-bearing edge clips
                if (entries.length && blockEls.length) {
                    const first = entries[0];
                    const last = entries[entries.length - 1];
                    const fB = blockEls[0];
                    const lB = blockEls[blockEls.length - 1];
                    // straddling the outer clip edge, exactly like the
                    // interior pills straddle the gap between blocks
                    // a looping cut has ONE seam here, drawn at both
                    // ends: the last clip into the first. Both pills
                    // carry the wrap's link state and open the same menu.
                    const wrap = loopOn() && first.gap == null
                        && last.gap == null && (first.meta || last.meta);
                    const wrapSeam = wrap ? (first.wrapSeam ?? null) : null;
                    if (wrap && fB) {
                        addZone(0, fB.offsetLeft, wrapSeam, last, first);
                    } else if (first.meta && fB) {
                        addZone(0, fB.offsetLeft, null,
                                null, first);
                    }
                    if (wrap && lB) {
                        addZone(entries.length,
                                lB.offsetLeft + lB.offsetWidth,
                                wrapSeam, last, first);
                    } else if (last.meta && lB) {
                        addZone(entries.length,
                                lB.offsetLeft + lB.offsetWidth,
                                null, last, null);
                    }
                }
            }
            let lastSeamX = 0, lastSeamLink = false;
            async function openSeamPopup(i, entries, hasLink, x) {
                closeSeamPopup();
                lastSeamX = x;
                lastSeamLink = hasLink;
                // edge zones have only one real side -- unless the cut
                // loops, when both ends are the SAME seam: last into
                // first (the same clip on both sides when there is one)
                const n = entries.length;
                const wrap = loopOn() && n > 0 && (i === 0 || i === n)
                    && entries[0].gap == null && entries[n - 1].gap == null;
                const L = i > 0 ? entries[i - 1]
                    : wrap ? entries[n - 1] : null;
                const R = i < n ? entries[i] : wrap ? entries[0] : null;
                const pop = document.createElement("div");
                seamPopup = pop;
                const stripTop = timelineWrap.offsetTop +
                    strip.offsetTop;
                Object.assign(pop.style, {
                    position: "absolute", zIndex: "10",
                    left: Math.max(0, Math.min(
                        timelineWrap.offsetLeft + 3 + x -
                            strip.scrollLeft - 110,
                        container.clientWidth - 240)) + "px",
                    bottom: (container.clientHeight - stripTop + 2)
                        + "px",
                    minWidth: "220px", maxWidth: "320px",
                    maxHeight: "190px", overflowY: "auto",
                    background: PAL.rest, color: PAL.text,
                    border: "1px solid " + PAL.edge,
                    borderRadius: "5px", padding: "6px",
                    font: "11px sans-serif",
                    boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
                });
                pop.addEventListener("click",
                    (ev) => ev.stopPropagation());
                pop.textContent = "scanning clips…";
                container.appendChild(pop);
                setTimeout(() => document.addEventListener("click",
                    closeSeamPopup, { once: true }), 0);

                // fresh folder scope AND a fresh file list -- this popup
                // already shows "scanning clips…", so the await is free
                await rescanPickerClips().catch(() => populatePicker());
                if (seamPopup !== pop) return;
                const values = Array.from(picker.options)
                    .map((o) => o.value);
                const metas = await Promise.all(
                    values.map((v) => tlClipMeta(v)));
                if (seamPopup !== pop) return;
                const cand = values
                    .map((v, k) => ({ clip: v, meta: metas[k] }))
                    .filter((c) => c.meta);

                const item = (parent, text, hoverTitle, onClick) => {
                    const it = document.createElement("div");
                    it.textContent = text;
                    it.title = hoverTitle;
                    Object.assign(it.style, {
                        padding: "3px 5px", borderRadius: "3px",
                        cursor: "pointer", whiteSpace: "nowrap",
                        overflow: "hidden", textOverflow: "ellipsis",
                    });
                    it.addEventListener("mouseenter", () =>
                        it.style.background = PAL.active);
                    it.addEventListener("mouseleave", () =>
                        it.style.background = "none");
                    it.addEventListener("click", () => {
                        onClick();
                        closeSeamPopup();
                    });
                    parent.appendChild(it);
                    return it;
                };
                const section = (side, parent) => {
                    const h = document.createElement("div");
                    h.textContent = side.title;
                    Object.assign(h.style, {
                        color: PAL.sub, margin: "4px 0 2px",
                        fontSize: "10px", letterSpacing: "0.05em",
                    });
                    parent.appendChild(h);
                    // The "generate new" items ARM the next generation,
                    // so upscale mode has nothing to offer them: the pin
                    // they would set is not what the next Run reads, and
                    // clicking one would silently rewrite the pin this
                    // mode is preserving. The repair items below stay --
                    // they edit the cut, which every mode still plays.
                    if (side.gen && !upscaleMode()) {
                        // pin the seam's clip so the NEXT RUN creates
                        // the take for this side; when that pin is
                        // ALREADY active the item wears the pin purple
                        // and clicking it again clears the pin
                        const cur = pinState();
                        const genActive = pinHasRole(
                            side.gen.clip, cur, side.gen.role);
                        const g = item(parent,
                            side.gen.role === "extend"
                                ? "+ extend new" : "+ prepend new",
                            (genActive
                                ? "active -- click to clear the pin"
                                : side.gen.role === "extend"
                                    ? "next run extends "
                                      + side.gen.clip
                                    : "next run prepends into "
                                      + side.gen.clip)
                            + (side.gen.pixel
                                ? " (no mctx sidecar: VAE-encoded from "
                                  + "pixels, soft continuity)" : ""),
                            () => togglePinFor(side.gen.clip,
                                              side.gen.role));
                        g.style.color = genActive
                            ? (PAL.light ? "#5e3a9e"
                                         : TL_ROLE.extend.text)
                            : PAL.text;
                        g.style.fontWeight = "600";
                        if (side.gen.pixel) {
                            const px = document.createElement("span");
                            px.textContent = " (pixels)";
                            px.style.color = PAL.sub;
                            px.style.fontWeight = "400";
                            g.appendChild(px);
                        }
                    }
                    // ... and a section left empty by that must say so,
                    // or it renders as a bare heading
                    if (!side.items.length
                            && !(side.gen && !upscaleMode())) {
                        const n = document.createElement("div");
                        n.textContent = "none found";
                        n.style.color = PAL.sub;
                        parent.appendChild(n);
                        return;
                    }
                    for (const c of side.items) {
                        item(parent, c.clip, c.clip,
                             () => side.act(c.clip));
                    }
                };

                pop.textContent = "";
                // Either END pill offers the loop, whether or not the
                // cut loops yet: "+ loop new" arms the take that CLOSES
                // the ring -- one run, extending the last clip and
                // prepending into the first -- and writes the `loop`
                // line; while the cut loops, "stop looping" removes it.
                // The loop is created and undone in one place.
                const endL = entries[n - 1], endR = entries[0];
                const atEnd = n > 0 && (i === 0 || i === n)
                    && endL.gap == null && endR.gap == null;
                if (atEnd && !upscaleMode() && tlPinGrade(endL)
                        && tlPinGrade(endR)) {
                    const cur = pinState();
                    const active = impliesLoop(cur);
                    const same = endL.clip === endR.clip;
                    const g = item(pop,
                        active ? "✓ loop take armed" : "+ loop new",
                        active ? "active -- click to clear the pin"
                            : (same
                                ? "next run bridges " + endL.clip
                                  + "'s tail to its own head"
                                : "next run bridges " + endL.clip + " → "
                                  + endR.clip)
                              + (loopOn() ? ""
                                 : " and makes the sequence loop"),
                        () => {
                            if (active) { setPinState(null); return; }
                            setBridge({ role: "bridge", source: endL.clip,
                                        source2: endR.clip,
                                        window: cur?.window ?? "22" });
                        });
                    g.style.fontWeight = "600";
                    g.style.color = active
                        ? (PAL.light ? "#5e3a9e" : TL_ROLE.extend.text)
                        : PAL.text;
                }
                if (atEnd && loopOn()) {
                    const s = item(pop, "stop looping",
                        "open the sequence again: the ends stop being a seam, "
                        + "playback stops at the end, and the export keeps "
                        + "its own top and tail. Any pin stays as it is.",
                        () => setLoop(false));
                    s.style.color = PAL.sub;
                }
                // at the wrap, "before the first" is line 0 and "after
                // the last" is the end -- not the zone's own index
                const insertAtGap = (clip, idx = i) => {
                    const ls = currentLines();
                    const map = entryLines();
                    const at = idx >= map.length ? ls.length : map[idx];
                    ls.splice(at, 0, clip);
                    setSequence(ls.join("\n"));
                };
                const replaceAt = (idx, clip) => {
                    const ls = currentLines();
                    ls[entryLines()[idx]] = clip;
                    setSequence(ls.join("\n"));
                };
                // the sides of the seam; edge zones contribute a single
                // side. gen = the "+ generate new" pin action (only
                // when that side's seam clip carries mctx to pin).
                const sides = [];
                if (hasLink) {
                    sides.push({
                        title: "replace left side",
                        items: cand.filter((c) => c.clip !== L.clip &&
                            linkedMeta(c.meta, R.meta)),
                        act: (clip) => replaceAt(wrap ? n - 1 : i - 1, clip),
                        gen: tlPinGrade(R)
                            ? { clip: R.clip, role: "prepend",
                                pixel: tlPinGrade(R) === "pixel" } : null,
                    });
                    sides.push({
                        title: "replace right side",
                        items: cand.filter((c) => c.clip !== R.clip &&
                            linkedMeta(L.meta, c.meta)),
                        act: (clip) => replaceAt(wrap ? 0 : i, clip),
                        gen: tlPinGrade(L)
                            ? { clip: L.clip, role: "extend",
                                pixel: tlPinGrade(L) === "pixel" } : null,
                    });
                } else {
                    // "insert before" (leads into the RIGHT clip) on
                    // the left, "insert after" (continues the LEFT
                    // clip) on the right -- matching reading order of
                    // where the result attaches
                    // `items` still needs R.meta -- inserting an
                    // EXISTING clip seamlessly is a lineage question --
                    // but `gen` does not: a new take can be generated
                    // from any clip, through pixels if need be.
                    if (tlPinGrade(R)) {
                        sides.push({
                            title: "insert before",
                            items: R.meta ? cand.filter((c) =>
                                linkedMeta(c.meta, R.meta)) : [],
                            act: (clip) => insertAtGap(clip, wrap ? 0 : i),
                            gen: { clip: R.clip, role: "prepend",
                                   pixel: tlPinGrade(R) === "pixel" },
                        });
                    }
                    if (tlPinGrade(L)) {
                        sides.push({
                            title: "insert after",
                            items: L.meta ? cand.filter((c) =>
                                linkedMeta(L.meta, c.meta)) : [],
                            act: (clip) => insertAtGap(clip, wrap ? n : i),
                            gen: { clip: L.clip, role: "extend",
                                   pixel: tlPinGrade(L) === "pixel" },
                        });
                    }
                }
                // two sides -> ALWAYS side-by-side columns (stacked
                // layout made the second side easy to miss, and a
                // stable layout beats one that reshapes with results);
                // edge zones keep the single column
                if (sides.length === 2) {
                    pop.style.minWidth = "340px";
                    pop.style.maxWidth = "480px";
                    pop.style.left = Math.max(0, Math.min(
                        timelineWrap.offsetLeft + 3 + x -
                            strip.scrollLeft - 170,
                        container.clientWidth - 490)) + "px";
                    const cols = document.createElement("div");
                    Object.assign(cols.style, {
                        display: "flex", gap: "10px",
                        alignItems: "flex-start",
                    });
                    for (const s of sides) {
                        const col = document.createElement("div");
                        Object.assign(col.style,
                            { flex: "1", minWidth: "0" });
                        section(s, col);
                        cols.appendChild(col);
                    }
                    pop.appendChild(cols);
                } else {
                    for (const s of sides) {
                        section(s, pop);
                    }
                }
                seamSettingsButton(i, pop);
            }

            // ---- one join's own seam settings -------------------------
            // The timeline's Seam Improvement settings are the default;
            // any join may differ. The override rides on the sequence
            // line of the clip to the RIGHT of the join, so it moves
            // with that clip -- reorder moves it, delete removes it.
            //
            // Only a REAL join gets them: both sides a clip. A boundary
            // against empty space is not a continuation at all -- the
            // level lock and the crossfade both skip it, and the server
            // would ignore anything written there -- so offering the
            // controls would promise something that cannot happen.
            function seamJoinAt(i, entries) {
                const L = i > 0 ? entries[i - 1] : null;
                const R = i < entries.length ? entries[i] : null;
                return (L && R && L.gap == null && R.gap == null)
                    ? { L, R } : null;
            }
            // ONE join's settings, in the same dialog the timeline's
            // use. `get` answers with the join's own value when it has
            // one and the timeline's otherwise, so every control shows
            // what will actually happen -- and `set` always writes an
            // override, because touching a control here IS the act of
            // giving this join its own.
            // What this join can actually be given -- see
            // tlSeamCapability, which the Result Preview asks the same
            // question of for the take it is judging.
            function seamCapability(i) {
                return tlSeamCapability(lastEntries?.[i - 1],
                                        lastEntries?.[i]);
            }

            let joinDialogClose = null;
            function openSeamJoinDialog(i) {
                if (joinDialogClose) joinDialogClose();
                const own = () => seamOptsAt(i);
                const es = tlParseSequence(seqWidget.value);
                const pair = seamJoinAt(i, es);
                if (!pair) return;
                const name = (e) => e.clip.split("/").pop();
                joinDialogClose = tlSeamDialog({
                    get: (k) => {
                        const o = own();
                        return o[k] !== undefined ? o[k] : widgetValue(k);
                    },
                    set: (k, v) => setSeamOpt(i, k, v),
                    seams: null,
                    scope: {
                        title: "This Join's Seam Settings",
                        sub: name(pair.L) + " → " + name(pair.R)
                            + ".  Anything left as inherited follows the "
                            + "timeline, so changing the timeline still "
                            + "changes this join.",
                        applies: (k) => {
                            const cap = seamCapability(i);
                            return !cap || cap[TL_SEAM_NEEDS[k]] !== false;
                        },
                        own: (k) => own()[k] !== undefined,
                        clear: (k) => setSeamOpt(i, k, undefined),
                        clearAll: () => clearSeamOpts(i),
                        count: () => Object.keys(own()).length,
                    },
                    // setSeamOpt already rewrote the
                    // sequence, which discards the built
                    // preview; this keeps the readouts
                    // honest alongside it
                    onChange: seamSettingsChanged,
                    onClose: () => { joinDialogClose = null; },
                });
            }

            function seamSettingsButton(i, parent) {
                const es = tlParseSequence(seqWidget.value);
                if (!seamJoinAt(i, es)) return;
                const own = es[i].seamOpts || {};
                const n = Object.keys(own).length;
                // withheld (SHOW_GUIDED) unless this join already has
                // settings of its own, or continues a clip that had no
                // sidecar -- the one masked join the repairs act on
                const pixel = tlPixelJoin(lastEntries?.[i - 1],
                                          lastEntries?.[i]);
                if (!SHOW_GUIDED && !n && !pixel) return;
                const b = document.createElement("button");
                b.textContent = n
                    ? `seam settings (${n} of its own)…`
                    : "seam settings…";
                Object.assign(b.style, {
                    marginTop: "8px", width: "100%",
                    borderTop: "1px solid " + PAL.edge,
                    background: n ? TL_ROLE.extend.bg : "none",
                    color: n ? "#fff" : PAL.text,
                    border: "1px solid " + (n ? "transparent" : PAL.edge),
                    borderRadius: "3px", padding: "2px 0",
                    font: "10px sans-serif", cursor: "pointer",
                });
                b.title = n
                    ? "This join has settings of its own. Open them."
                    : "Give this join its own seam settings, instead of "
                      + "following the timeline's.";
                b.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    closeSeamPopup();
                    openSeamJoinDialog(i);
                });
                parent.appendChild(b);
            }

            // ---- cut / uncut at the playhead --------------------------
            // The strip's two ends, as buttons: cut the clip the
            // playhead sits on, from the left or from the right. Only
            // the ends of a chain (and either side of an unlinked clip)
            // may move -- an interior join belongs to the sidecars.
            function playheadCut(side) {
                const f = displayFrame();
                if (f == null || !cumStarts.length) return null;
                const i = clipAt(Math.max(0, Math.round(f)));
                const e = lastEntries?.[i];
                if (!e || e.gap != null) return null;
                // where the playhead sits INSIDE the clip's own frames
                const within = Math.max(0, Math.round(f) - cumStarts[i])
                    + e.enter;
                if (!tlCanCut(lastEntries, i,
                              side === "left" ? "enter" : "exit",
                              linkedMeta)) {
                    return { idx: i, blocked: true };
                }
                return { idx: i, blocked: false, at: within };
            }

            function applyPlayheadCut(side) {
                const t = playheadCut(side);
                if (!t) return;
                const e = lastEntries[t.idx];
                if (t.blocked) {
                    app.extensionManager?.toast?.add?.({
                        severity: "warn", summary: "That side is joined",
                        detail: `${e.clip.split("/").pop()} is linked to its `
                            + `neighbour on that side -- the join is recorded `
                            + `in the sidecars. Only the ends of a chain can `
                            + `be cut.`, life: 6000,
                    });
                    return;
                }
                const snapped = tlSnapCut(t.at, e.meta, snapOn(),
                                        side === "left" ? "enter" : "exit");
                const lo = e.enter + 1, hi = (e.exit ?? e.frames) - 1;
                commitCut(t.idx, side === "left" ? "enter" : "exit",
                          side === "left"
                              ? Math.min(hi, Math.max(0, snapped))
                              : Math.max(lo, Math.min(e.frames, snapped)));
            }

            // The pin continues from the CUT, not from the clip's own
            // edge -- so pin_state has to carry it. Derived from the
            // sequence on every refresh rather than written once at
            // pin time: that way a cut dragged afterwards, or a clip
            // swapped for another take, updates the pin by itself
            // instead of leaving it pointing at a frame that moved.
            function syncPinCut(entries) {
                const st = pinState();
                if (!st || !psWidget) return;
                const cutOf = (clip, side) => {
                    const e = entries.find(
                        (x) => x.gap == null && x.clip === clip);
                    if (!e) return null;
                    const v = side === "exit" ? e.exitOverride
                                              : e.enterOverride;
                    return v == null ? null : v;
                };
                const next = { ...st };
                delete next.cut;
                delete next.cut2;
                if (st.role === "bridge") {
                    // the extend side leaves `source`, the prepend side
                    // arrives at `source2`
                    const a = cutOf(st.source, "exit");
                    const b = cutOf(st.source2, "enter");
                    if (a != null) next.cut = a;
                    if (b != null) next.cut2 = b;
                } else {
                    const a = cutOf(st.source,
                                    st.role === "prepend" ? "enter" : "exit");
                    if (a != null) next.cut = a;
                }
                const now = JSON.stringify(next);
                if (now === psWidget.value) return;
                // written directly: setPinState would re-enter refresh
                psWidget.value = now;
                tlNoteEdit();
            }

            // THE cut writer. Both the seam drag and the cut buttons go
            // through it, so "cutting a head leaves empty space" cannot
            // be true of one and not the other.
            // An adjacent hole ABSORBS a cut, in both directions: trim
            // the edge and the hole grows by what you took, restore the
            // edge and the hole gives it back. The clip's other end
            // then never moves, and neither does anything past the
            // hole.
            //
            // Both halves were wrong before. Trimming a head beside an
            // existing hole added a SECOND `~` line next to it, so the
            // strip grew a row of little holes instead of one that got
            // bigger. And dragging back the other way handed the frames
            // to the clip while leaving the hole where it was, which
            // pushed the whole rest of the timeline along -- so the
            // gesture could not be undone by reversing it.
            function absorbInto(gapIdx, freed) {
                const g = lastEntries?.[gapIdx];
                if (!g || g.gap == null || !freed) return false;
                setGapAt(gapIdx, (Number(g.gap) || 0) + freed);
                return true;
            }

            function commitCut(idx, side, at) {
                const e = lastEntries?.[idx];
                if (!e || e.gap != null) return;
                if (side === "exit") {
                    const freed = (e.exit ?? e.frames) - at;
                    setCut(idx, { exit: at >= e.frames ? null : at });
                    absorbInto(idx + 1, freed);
                    return;
                }
                const freed = at - e.enter;
                setCut(idx, { enter: at <= 0 ? null : at });
                if (absorbInto(idx - 1, freed)) return;
                // No hole to absorb it: cutting the HEAD of a clip
                // nothing leads into leaves the time it gave up behind,
                // so the clips after it hold their place -- that hole
                // is what a bridge fills.
                if (freed > 0 &&
                        !linkedMeta(lastEntries[idx - 1]?.meta, e.meta)) {
                    insertGapAt(idx, freed);
                }
            }

            // Rewrite the empty space that IS entry i, or drop the line
            // when it reaches nothing -- a `~ 0` is not something the
            // parser should have to tolerate.
            function setGapAt(i, frames) {
                const lines = currentLines();
                const map = entryLines();
                if (map[i] == null) return;
                const n = Math.max(0, Math.round(Number(frames) || 0));
                if (n <= 0) lines.splice(map[i], 1);
                else lines[map[i]] = `~ ${n}`;
                setSequence(lines.join("\n"));
            }

            function insertGapAt(i, frames) {
                const lines = currentLines();
                const map = entryLines();
                const at = i >= map.length ? lines.length : map[i];
                lines.splice(at, 0, `~ ${Math.max(1, Math.round(frames))}`);
                setSequence(lines.join("\n"));
            }

            // THE empty-space writer. A clip's position in time is fixed
            // entirely by what precedes it, so "move this clip later" is
            // one edit: the size of the hole in front of it. Setting it to
            // zero removes the hole rather than leaving a `~ 0` line the
            // parser would have to tolerate.
            function setGapBefore(i, frames) {
                const es = tlParseSequence(seqWidget.value);
                const lines = currentLines();
                const map = entryLines();
                const n = Math.max(0, Math.round(Number(frames) || 0));
                if (i > 0 && es[i - 1]?.gap != null) {
                    if (n <= 0) lines.splice(map[i - 1], 1);
                    else lines[map[i - 1]] = `~ ${n}`;
                } else {
                    if (n <= 0) return;            // already flush
                    const at = i >= map.length ? lines.length : map[i];
                    lines.splice(at, 0, `~ ${n}`);
                }
                setSequence(lines.join("\n"));
            }

            // The space in front of a clip, as the sequence currently
            // stands -- the value a ctrl+drag starts from.
            function gapBefore(entries, i) {
                return i > 0 && entries[i - 1]?.gap != null
                    ? entries[i - 1].gap : 0;
            }
            // Which strip children travel with a clip being moved in time.
            // Clip and gap blocks are flow children, so DOM order decides
            // them. The SEAM elements are absolutely positioned and are
            // appended after EVERY block, so DOM order would sweep them
            // all along: they have to be picked by the seam they belong
            // to. Seam j lies between blocks j-1 and j, so only j > i
            // travels -- and seam i is the hole the drag is resizing, so
            // it hides rather than sit in a place that is no longer true.
            // Seam furniture -- the link zones, their pills and the cut
            // handles -- floats ABOVE the blocks on the strip, and it is
            // thickest exactly at a boundary. During a reorder that is
            // where you aim: the drop line lives in the gap between two
            // clips, so the natural place to hold the pointer is the one
            // place the block underneath never saw the event. Nothing
            // updated, and the drop looked refused.
            //
            // Muting it for the duration of the drag lets dragover reach
            // the blocks. The previous values are kept rather than reset
            // to "": pills are pointer-transparent by design and would
            // otherwise start swallowing clicks afterwards.
            let seamMuted = null;
            function muteSeamFurniture() {
                if (seamMuted) return;
                seamMuted = [];
                for (const el of strip.children) {
                    if (el.dataset?.obvpmSeam === undefined) continue;
                    seamMuted.push([el, el.style.pointerEvents]);
                    el.style.pointerEvents = "none";
                }
            }
            function unmuteSeamFurniture() {
                if (!seamMuted) return;
                for (const [el, was] of seamMuted) {
                    el.style.pointerEvents = was;
                }
                seamMuted = null;
            }

            function dragCohort(children, block, i, skip) {
                // `skip` may be several elements: the drop indicator and
                // the drag's own badge both live on the strip during a
                // gesture, and neither is part of the timeline. Sweeping
                // the badge along moved it by the shift AND placed it at
                // the shifted x, so it ran away at twice the speed.
                const skips = Array.isArray(skip) ? skip : [skip];
                const move = [], hide = [];
                let seen = false;
                for (const el of children) {
                    if (el === block) seen = true;
                    if (skips.includes(el)) continue;
                    const s = el.dataset?.obvpmSeam;
                    if (s === undefined) {
                        if (seen) move.push(el);
                    } else if (Number(s) > i) {
                        move.push(el);
                    } else if (Number(s) === i) {
                        hide.push(el);
                    }
                }
                return { move, hide };
            }
            // A clip that something leads INTO cannot be slid away from
            // it: the hole would break the seam. Same rule that stops you
            // cutting an interior head, for the same reason.
            function canMoveInTime(entries, i) {
                const e = entries[i];
                if (!e || e.gap != null) return false;
                return !linkedMeta(entries[i - 1]?.meta, e.meta);
            }

            // "uncut" is offered only when the selected clip actually
            // carries a manual cut -- otherwise it is a button that does
            // nothing, which is worse than no button.
            function selectedHasCut() {
                const e = tlParseSequence(seqWidget.value)[lastHighlight];
                return !!e && e.gap == null
                    && (e.enterOverride !== null || e.exitOverride !== null);
            }

            // ---- empty space ------------------------------------------
            // A gap is a hole you have not filled yet -- black in the
            // preview and the export, and the thing a bridge generation
            // lands in. It is drawn as a hatched, unfilled block so it
            // never reads as a clip, and clicking it offers removal.
            function gapBlock(e, i, played, entries) {
                const block = document.createElement("div");
                block.dataset.obvpmFrames = String(played);
                // Marked on the ELEMENT, not looked up by index. The
                // render calls highlight() before it publishes the new
                // entry list, so an index lookup there reads the
                // PREVIOUS sequence -- and right after an edit that
                // added or moved a hole, the wrong block got painted
                // like a clip. That is the intermittent background.
                block.dataset.obvpmGap = "1";
                Object.assign(block.style, {
                    width: Math.max(1, played * (scaleNow() || 1)) + "px",
                    flexGrow: "0", flexShrink: "0", minWidth: "0",
                    boxSizing: "border-box",
                    border: "1px dashed " + PAL.edge,
                    borderRadius: "4px", position: "relative",
                    display: "flex", flexDirection: "column",
                    alignItems: "center", justifyContent: "center",
                    gap: "2px", cursor: "pointer", color: PAL.sub,
                    overflow: "hidden",
                    // No fill: empty space is the absence of a clip, so
                    // it reads better as a hole in the strip than as
                    // another kind of block. The dashed border is what
                    // marks it out.
                    background: "none",
                });
                block.title = `empty space, ${played} frames `
                    + `(${(played / TL_FPS).toFixed(1)}s)\n`
                    + "black in the preview and the export -- generate a "
                    + "bridge to fill it. Hover for its controls.";
                const cap = document.createElement("div");
                cap.textContent = `${(played / TL_FPS).toFixed(1)}s`;
                Object.assign(cap.style, {
                    font: "10px sans-serif", opacity: "0.8",
                    whiteSpace: "nowrap",
                });
                // Glyphs, not phrases: a gap can be a few pixels wide,
                // and "delete empty space" ellipsised down to "delet...".
                // Both buttons take the SAME box rather than sizing to
                // their own glyph -- "+" and "✕" are different widths,
                // so padding alone left one visibly larger than the
                // other. Revealed as flex, to centre the glyph in it.
                // The size is per glyph, not shared: "✕" is drawn much
                // larger than "+" at the same font-size, so matching the
                // numbers leaves the two looking unequal. The BOX is
                // identical; only the type size is tuned to make them
                // read as the same weight.
                const gapBtn = (glyph, fill, size) => {
                    const b = document.createElement("button");
                    b.textContent = glyph;
                    Object.assign(b.style, {
                        display: "none", alignItems: "center",
                        justifyContent: "center",
                        width: "22px", height: "18px", padding: "0",
                        background: fill, color: "#fff", border: "none",
                        borderRadius: "3px", cursor: "pointer",
                        font: size + "/1 sans-serif", whiteSpace: "nowrap",
                    });
                    return b;
                };
                const del = gapBtn("✕", "rgba(170,40,40,0.9)", "9px");
                del.title = "Delete this empty space";
                del.addEventListener("mouseenter", () => {
                    del.style.background = "rgba(200,55,55,0.95)";
                });
                del.addEventListener("mouseleave", () => {
                    del.style.background = "rgba(170,40,40,0.9)";
                });
                del.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    removeEntry(i);
                });

                // ---- fill it: pin a BRIDGE ------------------------
                // A bridge is one run that extends the clip on the left
                // AND prepends into the one on the right, so the take it
                // makes binds to both sides -- which is exactly what a
                // hole between two clips needs. The pin machinery has
                // always supported it; until now the only way to ask for
                // one was to pin opposite sides of two clips and let
                // them combine, which nobody would guess.
                const left = entries?.[i - 1];
                const right = entries?.[i + 1];
                const ends = !left || !right;
                const holes = !ends
                    && (left.gap != null || right.gap != null);
                // a pin is built from the sidecar's latents, so a clip
                // without one has nothing to bind to
                const noMctx = !ends && !holes && !(left.meta && right.meta);
                // A bridge is a PIN, so upscale mode cannot offer one:
                // the next Run refines the timeline instead of filling
                // this hole, and setting the pin anyway would rewrite
                // the one this mode is keeping.
                const canBridge = !ends && !holes && !noMctx
                    && !upscaleMode();
                const cur0 = shownPin();
                const isThis = canBridge && cur0?.role === "bridge"
                    && cur0.source === left.clip
                    && cur0.source2 === right.clip;
                const bridge = gapBtn("+", canBridge
                    ? TL_ROLE.extend.bg : "rgba(127,127,127,0.35)", "13px");
                // a little air between the two, or they read as one
                // two-storey control
                bridge.style.marginBottom = "3px";
                bridge.style.cursor = canBridge ? "pointer" : "default";
                // pressed look when this hole is already the pin
                bridge.style.boxShadow = isThis
                    ? "0 0 0 2px rgba(255,255,255,0.7)" : "none";
                bridge.title = upscaleMode()
                    ? "upscaling is on, so the next Run refines "
                      + "this timeline rather than generating anything. "
                      + "Switch to generation to bridge this space."
                    : ends
                    ? "A bridge needs a clip on BOTH sides -- this space "
                      + "is at the end of the timeline"
                    : holes
                        ? "A bridge needs a clip on both sides, not more "
                          + "empty space"
                        : noMctx
                            ? "A bridge is generated from both clips' "
                              + "mctx, and one of these does not have one"
                            : isThis
                                ? "The next run already bridges this "
                                  + "space -- click to clear the pin"
                                : "Generate a clip that fills this space: "
                                  + "the next run extends "
                                  + left.clip.split("/").pop()
                                  + " and prepends into "
                                  + right.clip.split("/").pop();
                if (canBridge) {
                    bridge.addEventListener("mouseenter", () => {
                        bridge.style.background = TL_ROLE.extend.hover;
                    });
                    bridge.addEventListener("mouseleave", () => {
                        bridge.style.background = TL_ROLE.extend.bg;
                    });
                    bridge.addEventListener("click", (ev) => {
                        ev.stopPropagation();
                        const cur = pinState();
                        setPinState(isThis ? null : {
                            role: "bridge", source: left.clip,
                            source2: right.clip,
                            window: cur?.window ?? "22",
                        });
                    });
                }

                // The controls appear on HOVER, so a narrow gap is not a
                // row of unreadable buttons and reaching them costs
                // nothing. Hover is exclusive by nature, so unlike the
                // old click-to-open there is no other gap's pair to
                // close first. mouseenter/mouseleave ignore the child
                // buttons, so moving onto one does not hide it.
                const show = (on) => {
                    for (const el of [bridge, del]) {
                        el.style.display = on ? "flex" : "none";
                    }
                    cap.style.display = on ? "none" : "block";
                };
                block.addEventListener("mouseenter", () => show(true));
                block.addEventListener("mouseleave", () => show(false));
                block.append(cap, bridge, del);
                blockEls.push(block);
                return block;
            }

            // ---- timeline rendering -----------------------------------
            let refreshSeq = 0;
            async function refresh() {
                const seq = ++refreshSeq;
                paintDur();     // the grid follows the pin's arriving mode
                const keepScroll = strip.scrollLeft;
                const keepFrame = displayFrame() ?? heldFrame;
                const entries = tlParseSequence(seqWidget.value);
                // a gap has no file behind it: no meta, no probe
                const metas = await Promise.all(
                    entries.map((e) => (e.gap != null
                        ? null : tlClipMeta(e.clip))));
                // clip lengths: sidecar header, else container metadata
                const lens = await Promise.all(entries.map((e, i) =>
                    e.gap != null ? e.gap
                        : metas[i]
                            ? Number(metas[i].delivered_frames ?? 0) || 0
                            : tlProbeClipFrames(e.clip)));
                if (seq !== refreshSeq) return;

                // enters/exits exactly like resolve_sequence()
                entries.forEach((e, i) => {
                    e.meta = metas[i];
                    e.frames = lens[i];
                    e.enter = 0;
                    e.exit = null;
                });
                for (let i = 1; i < entries.length; i++) {
                    // empty space breaks the chain: the clips either
                    // side keep their own ends rather than being joined
                    if (entries[i - 1].gap != null || entries[i].gap != null) {
                        continue;
                    }
                    const s = tlDeriveSeam(entries[i - 1].meta,
                                           entries[i].meta);
                    if (entries[i - 1].exit === null) {
                        entries[i - 1].exit = s.exitL;
                    }
                    entries[i].enter = s.enterR;
                    entries[i].seam = s;
                }
                // the wrap join of a looping cut: last into first, by
                // the same rule (mirrors resolve_sequence). A gap at
                // either end cannot wrap: the server refuses that build,
                // and the strip shows the ends unjoined rather than
                // pretending.
                if (loopOn() && entries.length && entries[0].gap == null
                        && entries[entries.length - 1].gap == null) {
                    const first = entries[0];
                    const last = entries[entries.length - 1];
                    const s = tlDeriveSeam(last.meta, first.meta);
                    if (last.exit === null) last.exit = s.exitL;
                    first.enter = s.enterR;
                    first.wrapSeam = s;
                }
                // a manual cut IS the decision -- it beats the derived seam
                for (const e of entries) {
                    if (e.enterOverride !== null) e.enter = e.enterOverride;
                    if (e.exitOverride !== null) e.exit = e.exitOverride;
                }
                syncPinCut(entries);

                playlist = entries.map((e) => ({
                    clip: e.clip, enter: e.enter, exit: e.exit,
                    gap: e.gap ?? null,
                }));
                // Only a CONTENT change invalidates playback state (a
                // theme flip re-runs refresh too, and must not). The
                // loaded twin <video>s and playIdx belong to the old
                // playlist -- without this, resuming after an edit plays
                // the stale chain until a seek forces a reload.
                const playlistSig = JSON.stringify(playlist);
                if (playlistSig !== lastPlaylistSig) {
                    lastPlaylistSig = playlistSig;
                    single = null;
                    usingSingle = false;
                    for (const v of vids) {
                        v.pause();
                        v._item = null;
                    }
                    playIdx = -1;
                    setPlayhead(null);
                    updateModeButtons(); // the build is gone; offer quick again
                }
                PAL = themePalette();
                applyChrome();
                paintSnap();
                cumStarts = [];
                playedArr = [];
                totalFrames = 0;
                entries.forEach((e) => {
                    const played =
                        Math.max(0, (e.exit ?? e.frames) - e.enter);
                    cumStarts.push(totalFrames);
                    playedArr.push(played);
                    totalFrames += played;
                });
                updateButtons();
                void probeFull();

                // First render of a timeline that has never been zoomed
                // fits, and then STAYS there. Re-fitting on every edit
                // is exactly the moving scale this replaced.
                if (!storedScale() && totalFrames) {
                    saveScale(fitScale(totalFrames));
                }
                const px = scaleNow() || 1;
                // A drop must never change the scale -- only a zoom may.
                // If this fires, whatever you saw move IS the scale; if
                // it stays quiet, the scale held and the movement is
                // layout (a clip and a hole swapping width, or the
                // scroll clamping against a shorter timeline).
                if (lastPx != null && Math.abs(px - lastPx) > 1e-9) {
                    tldbg("SCALE MOVED during a render:", lastPx, "->", px,
                          "| stored", storedScale(),
                          "| stripInnerWidth", stripInnerWidth());
                }
                lastPx = px;

                strip.replaceChildren();
                strip.appendChild(dropLine);
                blockEls = [];
                entries.forEach((e, i) => {
                    const delivered = e.frames;
                    const played = (e.exit ?? delivered) - e.enter;
                    if (e.gap != null) {
                        strip.appendChild(gapBlock(e, i, played, entries));
                        return;
                    }
                    const block = document.createElement("div");
                    // exactly its own length: no flexing, no floor. The
                    // 1px minimum is only so a degenerate zero-length
                    // clip stays clickable enough to delete.
                    block.dataset.obvpmFrames = String(played);
                    Object.assign(block.style, {
                        width: Math.max(1, played * px) + "px",
                        flexGrow: "0", flexShrink: "0", minWidth: "0",
                        boxSizing: "border-box",
                        background: PAL.rest,
                        border: "1px solid " + PAL.edge,
                        borderRadius: "4px",
                        position: "relative",
                        padding: "5px 9px",
                        display: "flex", flexDirection: "column",
                        gap: "3px", cursor: "pointer", color: PAL.text,
                        overflow: "hidden",
                    });
                    block.title = (i ? e.seam?.note + "\n" : "") +
                        (tlHasSidecar(e.meta) ? "mctx ✓" : "no mctx") +
                        (e.enter || e.exit !== null
                            ? `\nplays ${e.enter}..${e.exit ?? (delivered || "end")}`
                            : "") +
                        "\ndrag: reorder   ctrl+drag: move in time";
                    const name = document.createElement("div");
                    name.textContent = e.clip.split("/").pop();
                    Object.assign(name.style, {
                        whiteSpace: "nowrap", textOverflow: "ellipsis",
                        overflow: "hidden", font: "600 11px sans-serif",
                    });
                    const sub = document.createElement("div");
                    sub.style.opacity = "0.65";
                    sub.style.font = "10px sans-serif";
                    sub.textContent = played > 0
                        ? `${(played / TL_FPS).toFixed(1)}s` : "?";
                    const mini = (label, title, fn, active) => {
                        const b = document.createElement("button");
                        b.textContent = label;
                        b.title = title;
                        // highlight() swaps the block's text color per
                        // state (light-mode active is dark blue + light
                        // ink, resting is light + dark ink), so the
                        // controls INHERIT it for contrast in every
                        // combination: dimmed at rest, full + a neutral
                        // pill on hover. An ACTIVE pin button instead
                        // wears the this-run green permanently.
                        const restBg = active
                            ? "rgba(30,110,50,0.85)" : "none";
                        Object.assign(b.style, {
                            background: restBg, border: "none",
                            color: active ? "#fff" : "inherit",
                            opacity: active ? "1" : "0.7",
                            cursor: "pointer", borderRadius: "3px",
                            padding: "0 2px", font: "11px sans-serif",
                        });
                        b.addEventListener("mouseenter", () => {
                            b.style.opacity = "1";
                            b.style.background = active
                                ? "rgba(40,140,65,0.95)"
                                : "rgba(127,127,127,0.3)";
                        });
                        b.addEventListener("mouseleave", () => {
                            b.style.opacity = active ? "1" : "0.7";
                            b.style.background = restBg;
                        });
                        b.addEventListener("click", (ev) => {
                            ev.stopPropagation();
                            fn();
                        });
                        return b;
                    };
                    const mbadge = document.createElement("span");
                    Object.assign(mbadge.style, {
                        // inline-flex centering: a bare line-height box
                        // leaves 9px text visibly high of center
                        display: "inline-flex", alignItems: "center",
                        justifyContent: "center", height: "13px",
                        padding: "0 6px", whiteSpace: "nowrap",
                        borderRadius: "8px",
                        background: tlHasSidecar(e.meta)
                            ? "rgba(30,110,50,0.85)"
                            : "rgba(170,40,40,0.85)",
                        color: "#fff", font: "9px sans-serif",
                    });
                    // no ✓ glyph: it renders oversized and pushes the
                    // 9px text off vertical center
                    mbadge.textContent = tlHasSidecar(e.meta)
                        ? "mctx" : "no mctx";
                    mbadge.title = tlHasSidecar(e.meta)
                        ? "verified mctx sidecar"
                        : "no mctx sidecar: no latents to slice, so pins "
                          + "from this clip are VAE-encoded from its "
                          + "pixels. It can still be named as a parent -- "
                          + "takes made from it link to it and their seams "
                          + "are repairable.";
                    const subRow = document.createElement("div");
                    Object.assign(subRow.style, {
                        display: "flex", gap: "5px",
                        alignItems: "center",
                    });
                    // pin badge slot on the mctx row: width RESERVED on
                    // every block ("prepending" is the widest label) so
                    // pinning never changes any block's min size. The
                    // pin itself is toggled from the selected-clip bar
                    // or the seam popups.
                    const pst = shownPin();
                    const pinnedRole = pinRoleFor(e.clip, pst);
                    // the clip occupying the GAP(s) the pin regenerates
                    // gets replaced on add -- fade it as a preview.
                    // Gap-based, either lineage direction: prepending
                    // into S also displaces S's extend-PARENT (both are
                    // lead-ins for the same boundary). A bridge checks
                    // both of its sides.
                    const exClip = pst && pst.role !== "prepend"
                        ? pst.source : null;
                    const prClip = !pst ? null
                        : pst.role === "bridge" ? pst.source2
                        : pst.role === "prepend" ? pst.source : null;
                    const exMeta = exClip ? entries.find((en) =>
                        en.clip === exClip)?.meta : null;
                    const prMeta = prClip ? entries.find((en) =>
                        en.clip === prClip)?.meta : null;
                    // Lineage alone is NOT enough: a clip keeps its
                    // lineage wherever you drag it, so a bare
                    // linkedMeta() check left it faded after it had been
                    // moved away from the seam. A clip is only replaced
                    // when it still OCCUPIES the gap the run regenerates
                    // -- i.e. it is the immediate neighbour on that side
                    // right now, with the link live.
                    const willReplace =
                        tlOccupant(entries, exClip, "after",
                                   linkedMeta)?.entry === e ||
                        tlOccupant(entries, prClip, "before",
                                   linkedMeta)?.entry === e;
                    if (willReplace) {
                        block.style.opacity = "0.45";
                        block.title +=
                            "\nthe next run replaces this clip "
                            + "(same seam)";
                    }
                    // end bars on BOTH sides of every clip: overlay
                    // strips, NOT borders (thick borders miter into
                    // diagonal corners against the thin ones). The
                    // block's overflow:hidden clips them to its rounded
                    // corners. Green = seamless link on that side,
                    // purple = the next run regenerates on that side,
                    // gray = otherwise.
                    // at the ends of a looping cut the wrap seam is the
                    // last clip's right join and the first clip's left
                    const seamR = entries[i + 1]?.seam
                        ?? (i === entries.length - 1
                            ? entries[0]?.wrapSeam : undefined);
                    const seamL = i > 0 ? e.seam : e.wrapSeam;
                    const mkBar = (side, color) => {
                        const bar = document.createElement("div");
                        Object.assign(bar.style, {
                            position: "absolute", top: "0",
                            bottom: "0", width: "4px",
                            [side]: "0", background: color,
                            pointerEvents: "none",
                        });
                        return bar;
                    };
                    block.append(
                        // per SIDE, not per role: a clip bridged onto
                        // itself carries both
                        mkBar("left",
                            pinHasRole(e.clip, pst, "prepend")
                                ? TL_ROLE.prepend.bg
                                : seamL?.kind === "seamless"
                                    ? TL_COLORS.seamless
                                    : TL_COLORS.butt),
                        mkBar("right",
                            pinHasRole(e.clip, pst, "extend")
                                ? TL_ROLE.extend.bg
                                : seamR?.kind === "seamless"
                                    ? TL_COLORS.seamless
                                    : TL_COLORS.butt));
                    const pinSlot = document.createElement("span");
                    Object.assign(pinSlot.style, {
                        width: "68px", flexShrink: "0",
                        display: "inline-flex", alignItems: "center",
                    });
                    if (pinnedRole) {
                        const pb = document.createElement("span");
                        Object.assign(pb.style, {
                            display: "inline-flex",
                            alignItems: "center",
                            justifyContent: "center", height: "13px",
                            padding: "0 6px", whiteSpace: "nowrap",
                            borderRadius: "8px",
                            background: TL_ROLE[pinnedRole].bg,
                            color: "#fff", font: "9px sans-serif",
                        });
                        // the take that closes a loop wears its own
                        // word on both clips it touches (or the one)
                        const looping = impliesLoop(pst) &&
                            (pst.source === e.clip || pst.source2 === e.clip);
                        pb.textContent = looping ? "looping"
                            : pinnedRole === "extend"
                                ? "extending" : "prepending";
                        pb.title = looping
                            ? "the next generation closes the loop: it "
                              + "bridges the last clip's tail to the "
                              + "first clip's head"
                            : pinnedRole === "extend"
                                ? "the next generation extends from this "
                                  + "clip's tail"
                                : "the next generation prepends into this "
                                  + "clip's head";
                        pinSlot.appendChild(pb);
                    }
                    subRow.append(sub, mbadge, pinSlot);
                    const topRow = document.createElement("div");
                    Object.assign(topRow.style, {
                        display: "flex", gap: "4px",
                        alignItems: "center",
                    });
                    name.style.flex = "1";
                    const closeBtn = mini("✕",
                        "Remove from the sequence",
                        () => removeEntry(i));
                    closeBtn.style.font = "9px/13px sans-serif";
                    closeBtn.style.marginRight = "2px";
                    closeBtn.style.marginTop = "-1px";
                    topRow.append(name, closeBtn);
                    block.append(topRow, subRow);
                    block.addEventListener("click",
                        () => {
                            if (dragMoved) {   // tail of a ctrl+drag
                                dragMoved = false;
                                return;
                            }
                            const playing = activeStarted() &&
                                !vids[act].paused;
                            if (playing) {
                                (single ? playSingle : playAt)(i);
                            } else {
                                // paused: move there, don't start
                                seekGlobal(cumStarts[i]);
                                highlight(i);
                            }
                        });

                    // drag & drop reorder (custom MIME so nothing here
                    // ever looks like a file drop to the loader nodes)
                    block.draggable = true;
                    block.addEventListener("dragstart", (ev) => {
                        // ctrl is the move-in-time gesture below; a native
                        // drag here would fight its pointer capture
                        if (ev.ctrlKey || ev.metaKey) {
                            ev.preventDefault();
                            return;
                        }
                        dragFrom = i;
                        muteSeamFurniture();
                        ev.dataTransfer.setData(TL_DRAG_MIME, String(i));
                        ev.dataTransfer.effectAllowed = "move";
                    });
                    block.addEventListener("dragover", (ev) => {
                        if (!Array.from(ev.dataTransfer?.types ?? [])
                            .includes(TL_DRAG_MIME)) return;
                        ev.preventDefault();
                        ev.stopPropagation();
                        ev.dataTransfer.dropEffect = "move";
                        // which GAP: left or right half of this block
                        const r = block.getBoundingClientRect();
                        const before = ev.clientX - r.left < r.width / 2;
                        const j = before ? i : i + 1;
                        if (dragFrom === null ||
                            j === dragFrom || j === dragFrom + 1) {
                            // dropping here would not move the clip
                            dropInsertAt = null;
                            dropLine.style.display = "none";
                            clearLinkTint();
                            return;
                        }
                        dropInsertAt = j;
                        // link preview: tint the WHOLE clip(s)
                        // this drop would link with dark green
                        const dragged = entries[dragFrom];
                        const linkL = j > 0 && linkedMeta(
                            entries[j - 1]?.meta, dragged?.meta);
                        const linkR = j < entries.length && linkedMeta(
                            dragged?.meta, entries[j]?.meta);
                        clearLinkTint();
                        if (linkL) tintLinkBlock(blockEls[j - 1]);
                        if (linkR) tintLinkBlock(blockEls[j]);
                        const cx = before
                            ? block.offsetLeft - 1.5
                            : block.offsetLeft + block.offsetWidth + 1.5;
                        dropLine.style.left =
                            Math.max(0, cx - 4.5) + "px";
                        dropLine.style.display = "block";
                    });
                    block.addEventListener("drop", (ev) => {
                        dropLine.style.display = "none";
                        clearLinkTint();
                        if (dragFrom === null || dropInsertAt === null) {
                            return;
                        }
                        ev.preventDefault();
                        ev.stopPropagation();
                        // insertion index -> position after removal
                        const to = dropInsertAt > dragFrom
                            ? dropInsertAt - 1 : dropInsertAt;
                        unmuteSeamFurniture();
                        reorderEntry(dragFrom, to);
                        dragFrom = dropInsertAt = null;
                    });
                    block.addEventListener("dragend", () => {
                        dropLine.style.display = "none";
                        clearLinkTint();
                        unmuteSeamFurniture();
                        dragFrom = dropInsertAt = null;
                    });

                    // ---- ctrl+drag: move in TIME, not in order --------
                    // Plain drag reorders (above). Ctrl -- or Cmd -- drag
                    // slides the clip along the timeline, growing or
                    // shrinking the empty space in front of it. The
                    // gesture has to differ because the outcomes differ:
                    // one changes WHICH clip plays next, the other only
                    // WHEN this one plays.
                    const timeOK = () => canMoveInTime(entries, i);
                    block.addEventListener("pointerdown", (ev) => {
                        if (ev.button !== 0) return;
                        if (!(ev.ctrlKey || ev.metaKey)) return;
                        ev.preventDefault();       // no native drag, no
                        ev.stopPropagation();      // canvas panning
                        if (!timeOK()) return;
                        const startX = ev.clientX;
                        const base = gapBefore(entries, i);
                        // px per frame from the block the user is actually
                        // holding: dragging it its own width == its own
                        // length, which is the only ratio that reads right
                        const r0 = block.getBoundingClientRect();
                        // LAYOUT px per frame, taken from the scale
                        // itself. Deriving it from the measured width
                        // instead used offsetWidth, which is rounded to
                        // a whole pixel, so the ghost drifted from the
                        // pointer by that error times the frames moved
                        // and snapped straight on drop.
                        const ppf = scaleNow()
                            || (block.offsetWidth
                                / Math.max(1, (e.exit ?? delivered)
                                              - e.enter));
                        // getBoundingClientRect and clientX are SCREEN
                        // px; transforms and style.left are LAYOUT px.
                        // The canvas zoom scales between them, so the
                        // ghost drifted from the cursor and then jumped
                        // to the true position on drop -- and the drag
                        // tag flew off to the side. ppf itself is fine
                        // (screen over screen); only what we WRITE has
                        // to be converted back.
                        const czoom = r0.width /
                            (block.offsetWidth || 1) || 1;
                        // onto the strip, so it can be placed in the
                        // same coordinates as the blocks
                        strip.appendChild(dragTag);
                        const { move: sibs, hide: hidden } =
                            dragCohort(strip.children, block, i,
                                       [dropLine, dragTag]);
                        const wasShown = hidden.map(
                            (el) => el.style.visibility);
                        for (const el of hidden) el.style.visibility = "hidden";
                        // some seam elements are centred WITH a transform
                        // (translateY(-50%)); the shift composes onto it
                        // rather than replacing it
                        const baseTf = sibs.map((el) => el.style.transform);
                        let frames = base;
                        dragMoved = false;
                        block.draggable = false;
                        try {
                            block.setPointerCapture(ev.pointerId);
                        } catch (err) { /* no capture: still works */ }
                        const onMove = (m) => {
                            const dx = (m.clientX - startX) / czoom;
                            if (Math.abs(dx * czoom) > 2) dragMoved = true;
                            let f = Math.max(0, base + dx / ppf);
                            f = snapOn()
                                ? Math.round(f / TL_GROUP) * TL_GROUP
                                : Math.round(f);
                            if (f < 2) f = 0;   // snap flush to the left
                            frames = f;
                            const shift = (frames - base) * ppf;
                            sibs.forEach((el, k) => {
                                el.style.transform =
                                    `translateX(${shift}px) ${baseTf[k]}`;
                            });
                            // keep the ruler and the playhead on the
                            // same story as the blocks
                            ghostShift = { from: i, px: shift };
                            drawRuler();
                            setPlayhead(displayFrame() ?? heldFrame);
                            // On the clip's own START, not at the
                            // pointer: that edge IS the thing being
                            // moved, and it is where the cut drag puts
                            // its readout too. Seconds to one decimal,
                            // the way the empty block labels itself, so
                            // the two numbers can be compared.
                            dragTag.textContent = frames
                                ? `${frames}f · `
                                  + `${(frames / TL_FPS).toFixed(1)}s space`
                                : "no space";
                            const gx = block.offsetLeft + shift;
                            const half = (dragTag.offsetWidth || 60) / 2;
                            const far = Math.max(half,
                                (strip.scrollWidth || 0) - half - 2);
                            dragTag.style.left =
                                Math.min(Math.max(gx, half + 2), far) + "px";
                            dragTag.style.display = "block";
                        };
                        const onUp = () => {
                            document.removeEventListener(
                                "pointermove", onMove, true);
                            document.removeEventListener(
                                "pointerup", onUp, true);
                            try {
                                block.releasePointerCapture(ev.pointerId);
                            } catch (err) { /* never captured */ }
                            dragTag.style.display = "none";
                            dragTag.remove();
                            ghostShift = null;
                            sibs.forEach((el, k) => {
                                el.style.transform = baseTf[k];
                            });
                            hidden.forEach((el, k) => {
                                el.style.visibility = wasShown[k];
                            });
                            block.draggable = true;
                            // the pending click is swallowed by the guard
                            // in the click handler; clear it if none comes
                            setTimeout(() => { dragMoved = false; }, 0);
                            if (frames !== base) setGapBefore(i, frames);
                        };
                        document.addEventListener(
                            "pointermove", onMove, true);
                        document.addEventListener("pointerup", onUp, true);
                    });
                    blockEls.push(block);
                    strip.appendChild(block);
                });
                highlight(playIdx);
                // ruler + link geometry depend on freshly laid-out blocks
                lastEntries = entries;
                // a pin whose clip left the sequence is stale: clear it
                // directly (setPinState would recurse into refresh)
                const pst2 = pinState();
                const inSeq = (c) =>
                    entries.some((en) => en.clip === c);
                if (pst2 && (!inSeq(pst2.source) ||
                        (pst2.role === "bridge" &&
                         !inSeq(pst2.source2)))) {
                    if (psWidget) psWidget.value = "";
                    tldbg("pin cleared: its clip left the sequence");
                }
                // repainted on every strip render, not only on a click:
                // "load settings" writes widget values straight in, and
                // a button showing the mode it was left on is worse than
                // no button
                paintRunMode();
                // Can this sequence loop at all? Only when something
                // closes the ring: its last clip leads back into its
                // first, or the take that will do so is armed (the pin
                // "+ loop new" sets, which is why looping may be on
                // before that take exists).
                const hadRender = seenClips !== null;
                const arrived = hadRender
                    ? entries.filter((en) => en.gap == null
                                     && !seenClips.has(en.clip)) : [];
                seenClips = new Set(entries.filter((en) => en.gap == null)
                                           .map((en) => en.clip));
                const firstEn = entries[0];
                const lastEn = entries[entries.length - 1];
                const closes = entries.length > 1 && firstEn.gap == null
                    && lastEn.gap == null
                    && linkedMeta(lastEn.meta, firstEn.meta);
                loopPossible = closes || impliesLoop(pinState());
                paintLoop();
                // The line follows the strip in BOTH directions, and only
                // on an edit (there was a previous render) -- loading a
                // graph decides nothing. ON when the clip that closes the
                // ring ARRIVES as the last entry, so "stop looping"
                // sticks rather than being undone by the next repaint;
                // OFF when nothing closes it any more (that take was
                // removed, the loop pin dropped), because a `loop` line
                // nobody can see or justify renders the wrong frames.
                if (hadRender && !loopOn() && closes
                        && arrived.includes(lastEn)) {
                    tldbg("loop on: the added clip leads back into the first");
                    tlLoopToast("looping switched on: "
                        + lastEn.clip.split("/").pop()
                        + " leads back into the first clip");
                    setLoop(true);      // re-renders through setSequence
                    return;
                }
                if (hadRender && loopOn() && !loopPossible) {
                    tldbg("loop off: nothing closes the ring any more");
                    tlLoopToast("looping switched off: no clip leads back "
                        + "into the first one any more");
                    setLoop(false);
                    return;
                }
                renderNextRun();
                renderClipInfo();
                void loadSeamLevels(entries);
                // put the view back where it was: same scroll, same
                // playhead (clamped -- a cut can shorten the timeline
                // out from under the old position)
                strip.scrollLeft = keepScroll;
                heldFrame = keepFrame == null ? null
                    : Math.max(0, Math.min(keepFrame,
                                           Math.max(0, totalFrames - 1)));
                setPlayhead(heldFrame);
                updateCutRow();
                requestAnimationFrame(() => {
                    strip.scrollLeft = keepScroll;   // after layout settles
                    setPlayhead(heldFrame);
                    drawRuler();
                    drawLinks(entries);
                    paintSeamTitles();   // drawLinks rebuilt the pills
                });
                node.graph?.setDirtyCanvas(true);
            }

            // Shared pin toggle: the ONLY writers of pin_state are this
            // (via the info-bar toggles and the seam popups' "generate
            // new" items) -- the strip blocks just wear a status badge.
            function togglePinFor(clip, role) {
                const cur = pinState();
                const w = cur?.window ?? "22";
                if (pinHasRole(clip, cur, role)) {
                    // toggling an ACTIVE side off: a bridge keeps its
                    // other side as a single pin, a single pin clears
                    if (cur.role === "bridge") {
                        setPinState(role === "extend"
                            ? { source: cur.source2, role: "prepend",
                                window: w }
                            : { source: cur.source, role: "extend",
                                window: w });
                    } else {
                        setPinState(null);
                    }
                    return;
                }
                // opposite-side single pin already set on another clip
                // -> combine into a bridge (extend side = source,
                // prepend side = source2)
                // ...or on the SAME clip when it is the only one on the
                // strip: extending its tail and prepending its head in
                // one run is the take that loops it onto itself
                const alone = lastEntries?.length === 1 &&
                    lastEntries[0].clip === clip;
                if (cur && cur.role !== "bridge" &&
                        (clip !== cur.source || alone) && cur.role ===
                        (role === "extend" ? "prepend" : "extend")) {
                    setBridge(role === "extend"
                        ? { role: "bridge", source: clip,
                            source2: cur.source, window: w }
                        : { role: "bridge", source: cur.source,
                            source2: clip, window: w });
                    return;
                }
                // fresh pin. If it would REPLACE a clip that is linked
                // on BOTH sides (regenerating the middle of a chain),
                // auto-upgrade to a bridge so the new take binds to
                // both neighbors.
                const meta = lastEntries?.find(
                    (en) => en.clip === clip)?.meta;
                if (meta && lastEntries) {
                    if (role === "extend") {
                        const occ = lastEntries.find((en) =>
                            en.clip !== clip && en.meta &&
                            linkedMeta(meta, en.meta));
                        const far = occ && lastEntries.find((en) =>
                            en !== occ && en.clip !== clip &&
                            en.meta && linkedMeta(occ.meta, en.meta));
                        if (far) {
                            setPinState({ role: "bridge",
                                          source: clip,
                                          source2: far.clip,
                                          window: w });
                            return;
                        }
                    } else {
                        const occ = lastEntries.find((en) =>
                            en.clip !== clip && en.meta &&
                            linkedMeta(en.meta, meta));
                        const far = occ && lastEntries.find((en) =>
                            en !== occ && en.clip !== clip &&
                            en.meta && linkedMeta(en.meta, occ.meta));
                        if (far) {
                            setPinState({ role: "bridge",
                                          source: far.clip,
                                          source2: clip, window: w });
                            return;
                        }
                    }
                }
                setPinState({ source: clip, role, window: w });
            }

            const smallBtn = (label, title) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.title = title;
                Object.assign(b.style, {
                    // 0 vertical padding: 16px line + borders must fit
                    // inside the bars' fixed 24px height untouched
                    background: "none",
                    border: "1px solid " + PAL.edge,
                    color: PAL.text, borderRadius: "4px",
                    padding: "0 10px", cursor: "pointer",
                    font: "11px/16px sans-serif",
                });
                b.style.flexShrink = "0";
                b.addEventListener("mouseenter", () => {
                    b.style.background = b._hoverBg
                        ?? "rgba(127,127,127,0.3)";
                });
                b.addEventListener("mouseleave", () => {
                    b.style.background = b._restBg ?? "none";
                });
                return b;
            };

            // A dropdown wearing the same clothes as smallBtn. The pin
            // options used to cycle on click, which reads fine for a
            // two-value toggle and badly for a five-rung ladder: you
            // could not see what the other rungs were without walking
            // through them, and walking past the one you wanted meant
            // going round again.
            const smallSelect = (values, current, label, title, onPick) => {
                const sel = document.createElement("select");
                sel.title = title;
                Object.assign(sel.style, {
                    background: PAL.rest, color: PAL.text,
                    border: "1px solid " + PAL.edge, borderRadius: "4px",
                    padding: "0 2px", font: "11px/16px sans-serif",
                    height: "18px", boxSizing: "border-box",
                    flexShrink: "0", cursor: "pointer",
                });
                for (const v of values) {
                    const o = document.createElement("option");
                    o.value = String(v);
                    o.textContent = label(v);
                    if (v === current) o.selected = true;
                    sel.appendChild(o);
                }
                sel.addEventListener("change", () => {
                    const v = values.find(
                        (x) => String(x) === sel.value);
                    if (v !== undefined) onPick(v);
                });
                // the canvas swallows these otherwise and the menu never
                // opens (same reason the picker stops them)
                sel.addEventListener("pointerdown",
                                     (ev) => ev.stopPropagation());
                sel.addEventListener("click", (ev) => ev.stopPropagation());
                return sel;
            };

            // Advanced controls are OFF by default: the measured defaults
            // (both + ramp 10 + edge 0.40, window 39) are what a pin
            // should use, and a row of five selectors invites fiddling
            // with settings whose effect is smaller than the seed noise.
            // The preference is per-browser, like the pin mode's.
            const ADV_KEY = "obvpm.h3.pin_advanced";
            let advOpen = false;
            try { advOpen = localStorage.getItem(ADV_KEY) === "1"; }
            catch (err) { /* private mode -- stay simple */ }
            function setAdvanced(on) {
                advOpen = !!on;
                try { localStorage.setItem(ADV_KEY, advOpen ? "1" : "0"); }
                catch (err) { /* ignore */ }
                renderNextRun();
                node.setDirtyCanvas?.(true, true);
            }

            // The "next run" bar: what the pin_specs output will emit.
            // Window cycles through the everyday sizes; anything fancier
            // is H3MCtxPinSpec's job.
            const PIN_WINDOWS = ["22", "39", "56"];
            function renderNextRun() {
                const st = pinState();
                const frame = barParts(nextRunRow, "next run");
                const value = document.createElement("span");
                Object.assign(value.style, {
                    font: "600 11px/16px sans-serif",
                    whiteSpace: "nowrap", overflow: "hidden",
                    textOverflow: "ellipsis",
                    // A badge, not bare text: this line states what the
                    // NEXT GENERATION will do, which is the one thing
                    // here that changes what a queued run produces --
                    // it should not read like a caption. Same scheme as
                    // the "extending" badge a pinned clip wears on the
                    // strip (see pinnedRole below): solid role fill,
                    // white text, 8px corners. Sized to fit the row's
                    // hard-locked 24px: 16px line + 2px padding + 2px
                    // border = 20px.
                    padding: "1px 8px", borderRadius: "8px",
                    border: "1px solid transparent",
                    boxSizing: "border-box", maxWidth: "100%",
                });
                if (upscaleMode()) {
                    // The row states what a Run does, and in this mode a
                    // Run does not generate at all -- so it says so
                    // instead of describing a pin that will not be used.
                    // A FILLED badge, like a pin's: this is an active
                    // claim about the next Run, not the absence of one.
                    // The duration box is left off for the same reason
                    // the pin is: `length` feeds the generation branch,
                    // which this mode switches off.
                    value.textContent = "upscale — refine and render the full sequence";
                    value.style.background = TL_UPSCALE_BG;
                    value.style.color = "#fff";
                    value.style.borderColor = "transparent";
                    value.style.flexShrink = "1";
                    value.style.minWidth = "0";
                    value.title =
                        "upscaling is on, so a Run refines this "
                        + "whole timeline as one piece and renders the "
                        + "full sequence instead of generating a new "
                        + "take. Any pin is kept exactly as it is and "
                        + "comes back when you switch to generation.";
                    frame.append(value);
                    return;
                }
                if (!st) {
                    // no pin: the same surface as the clip-info box, so
                    // "nothing special is happening" looks like the
                    // resting chrome rather than an announcement
                    value.textContent = "root generation (no pin)";
                    value.style.color = PAL.sub;
                    value.style.background = PAL.rest;
                    value.style.borderColor = PAL.edge;
                    // duration belongs here too: a root generation has a
                    // length like any other run, and this row is where
                    // the length output is explained
                    frame.append(value, durWrap);
                    return;
                }
                const shortName = st.source.split("/").pop();
                value.textContent = st.role === "bridge"
                    ? (impliesLoop(st)
                        ? (st.source === st.source2
                            ? `loops ${shortName} onto itself`
                            : `loops: bridges ${shortName} → `
                              + st.source2.split("/").pop())
                        : `bridges ${shortName} → `
                          + st.source2.split("/").pop())
                    : st.role === "extend"
                        ? `extends ${shortName}`
                        : `prepends into ${shortName}`;
                // Exactly the clip badge's colours: a solid role fill
                // carrying white text. No light/dark branch, because
                // there is none there either -- the fill is dark enough
                // to hold white ink on either ground, which is the
                // point of filling it rather than tinting it.
                // "bridge" is not a strip role and has no colour of its
                // own; it pins the same way, so it wears the same one.
                value.style.background =
                    (TL_ROLE[st.role] ?? TL_ROLE.extend).bg;
                value.style.color = "#fff";
                value.style.borderColor = "transparent";
                value.style.flexShrink = "1";
                value.style.minWidth = "0";
                value.title = "Wire the node's pin_specs output to "
                    + "H3MCtxApplyPins; the pinned clip is hash-verified "
                    + "against its sidecar at run time.";
                const pinMode = pinModeOf(st);
                // On a bridge the controls set the arriving side; the
                // departing one is a masked extend and stays that way.
                const arriveMode = arriveModeOf(st);
                const shownMode = st.role === "bridge" ? arriveMode : pinMode;
                const modeChoosable = PIN_MODE_CHOOSABLE.has(st.role)
                    || st.role === "bridge";
                const sh = maskShapeOf(st);
                const ladder = pinModeMasks(pinMode)
                    ? PIN_WINDOWS_MASKED : PIN_WINDOWS;
                const w = ladder.includes(String(st.window ?? ""))
                    ? String(st.window)
                    : (pinModeMasks(pinMode) ? "39" : "22");
                const shapeable = PIN_ARRIVE_ROLES.has(st.role)
                    && pinModeMasks(arriveMode);

                // ---- the advanced controls -------------------------
                // Everything here has a measured default, so the row
                // shows none of it until asked. Built only when open:
                // five selectors are five DOM subtrees per repaint, and
                // this row repaints on every strip refresh.
                const adv = [];
                if (advOpen) {
                    if (SHOW_GUIDED || modeChoosable) {
                        adv.push(smallSelect(
                            modeChoosable ? PIN_MODES : ["masked", "guide"],
                            shownMode,
                            (v) => (v === "guide" ? "guided" : v),
                            MODE_HELP,
                            (side) => {
                                if (side === shownMode) return;
                                if (st.role === "bridge") {
                                    setPinState({ ...st, mode2: side });
                                    return;
                                }
                                rememberPinMode(side);
                                setPinState({ ...st, mode: side,
                                    window: pinModeMasks(side)
                                        ? maskedWindow(w) : w });
                            }));
                    }
                    adv.push(smallSelect(
                        ladder, w, (v) => "window " + v,
                        pinModeMasks(pinMode) ? WINDOW_HELP_MASKED
                                              : WINDOW_HELP_GUIDED,
                        (v) => setPinState({ ...st, window: v })));
                    if (shapeable) {
                        // Writing ONE key would leave the other two
                        // implicit, and an absent key means "whatever
                        // the default is today" -- so a state saved now
                        // could mean something else after a default
                        // moves. Touching any of the three pins all
                        // three at the values shown when it was touched.
                        const setShape = (patch) => setPinState({
                            ...st,
                            mask_ramp_frames: sh.ramp,
                            mask_ramp_edge: sh.edge,
                            mask_hold: sh.hold,
                            ...patch });
                        adv.push(smallSelect(
                            MASK_RAMPS, sh.ramp,
                            (v) => (v ? "ramp " + v + "f →"
                                        + (MASK_RUNWAY[v] ?? 0)
                                      : "ramp 0f"),
                            RAMP_HELP,
                            (v) => setShape({ mask_ramp_frames: v })));
                        if (sh.ramp) {
                            adv.push(smallSelect(
                                MASK_EDGES, sh.edge,
                                (v) => "edge " + v.toFixed(2),
                                EDGE_HELP,
                                (v) => setShape({ mask_ramp_edge: v })));
                        }
                        adv.push(smallSelect(
                            MASK_HOLDS, sh.hold,
                            (v) => "deep " + v.toFixed(2),
                            DEEP_HELP,
                            (v) => setShape({ mask_hold: v })));
                    }
                }
                const advBtn = smallBtn(advOpen ? "simple" : "advanced",
                    advOpen
                    ? "Hide the pin options and go back to the measured "
                      + "defaults' surface. Whatever is set now is KEPT "
                      + "-- this only puts the controls away."
                    : "Show the pin options: mode, window"
                      + (shapeable ? ", and the soft-hold shape (ramp, "
                                     + "edge, deep)" : "")
                      + ". They already sit at the measured-best values, "
                      + "so there is nothing you must set here.");
                advBtn.addEventListener("click", () => setAdvanced(!advOpen));
                const clearBtn = smallBtn("✕",
                    "Clear the pin (pin_specs goes empty: plain root "
                    + "generation)");
                clearBtn.addEventListener("click",
                    () => setPinState(null));
                const btns = document.createElement("span");
                Object.assign(btns.style, {
                    marginLeft: "auto", display: "flex", gap: "4px",
                    alignItems: "center", flexShrink: "0",
                });
                btns.append(...adv, advBtn, clearBtn);
                frame.append(value, durWrap, btns);
            }

            // The selected-clip info bar + its extend/prepend toggles.
            // Selection IS the highlighted clip (the one the playhead /
            // last click sits on) -- no separate selection state.
            function renderClipInfo() {
                infoIdx = lastHighlight;
                const e = lastEntries?.[lastHighlight];
                infoRow.replaceChildren();
                // the clip-surfaced box holds the text; the toggles sit
                // OUTSIDE it, to its right
                const frame = document.createElement("div");
                Object.assign(frame.style, {
                    flex: "1", minWidth: "0", display: "flex",
                    alignItems: "center", gap: "6px",
                    boxSizing: "border-box", borderRadius: "4px",
                    padding: "0 7px", overflow: "hidden",
                    background: PAL.rest,
                    border: "1px solid " + PAL.edge,
                });
                infoRow.appendChild(frame);
                if (!e) {
                    // no caption when empty -- one line in the SAME
                    // style as the "selected clip" caption
                    const d = document.createElement("span");
                    d.textContent = "no clip selected";
                    Object.assign(d.style, {
                        color: PAL.sub, fontSize: "10px",
                        letterSpacing: "0.05em",
                        whiteSpace: "nowrap",
                    });
                    frame.appendChild(d);
                    return;
                }
                const nm = document.createElement("span");
                nm.textContent = e.clip.split("/").pop();
                nm.title = e.clip;
                Object.assign(nm.style, {
                    font: "600 11px/16px sans-serif", color: PAL.text,
                    whiteSpace: "nowrap", flexShrink: "0",
                });
                const det = document.createElement("span");
                const exit = e.exit ?? e.frames;
                det.textContent =
                    `- ${e.frames} frames (plays ${e.enter}-${exit})`;
                det.title = det.textContent;
                Object.assign(det.style, {
                    color: PAL.sub, font: "11px/16px sans-serif",
                    whiteSpace: "nowrap", overflow: "hidden",
                    textOverflow: "ellipsis", flexShrink: "1",
                    minWidth: "0",
                });
                frame.append(nm, det);
                const grade = tlPinGrade(e);
                if (!grade) return;
                const pst = shownPin();
                // No sidecar = no latents, so a pin from this clip has
                // to be VAE-ENCODED from its pixels at run time. That is
                // a weaker join, not an impossible one, so the toggles
                // are offered and MARKED rather than hidden: hiding them
                // was indistinguishable from the feature being broken.
                const pixel = grade === "pixel";
                // INTEGRATED toggles: inline text that colors up when
                // active, not framed buttons -- they read as part of
                // the bar itself
                const mkToggle = (role, label, title) => {
                    const active = pinHasRole(e.clip, pst, role);
                    const b = document.createElement("button");
                    b.textContent = label;
                    b.title = pixel
                        ? title + " -- this clip has no mctx sidecar, so "
                            + "the pin is VAE-encoded from its pixels: "
                            + "soft (pixel-grade) continuity, and the new "
                            + "take is saved as a root, not a "
                            + "continuation. Connect vae + audio_vae on "
                            + "H3 MCtx Apply Pins."
                        : title;
                    Object.assign(b.style, {
                        // framed button; toggled = purple fill + white
                        background: active
                            ? TL_ROLE[role].bg : PAL.rest,
                        // dashed = the pixel route; the fill still shows
                        // which side is armed
                        border: (pixel ? "1px dashed " : "1px solid ") +
                            (active ? "rgba(255,255,255,0.55)" : PAL.edge),
                        color: active ? "#fff" : PAL.text,
                        borderRadius: "4px", cursor: "pointer",
                        // fixed width + centered text: toggling never
                        // shifts the neighbor
                        width: "88px", textAlign: "center",
                        padding: "1px 0", boxSizing: "border-box",
                        font: "11px/16px sans-serif",
                        whiteSpace: "nowrap", flexShrink: "0",
                    });
                    b.addEventListener("mouseenter", () => {
                        b.style.background = active
                            ? TL_ROLE[role].hover
                            : "rgba(127,127,127,0.3)";
                    });
                    b.addEventListener("mouseleave", () => {
                        b.style.background = active
                            ? TL_ROLE[role].bg : PAL.rest;
                    });
                    b.addEventListener("click", (ev) => {
                        ev.stopPropagation();
                        togglePinFor(e.clip, role);
                    });
                    return b;
                };
                const wrap = document.createElement("span");
                Object.assign(wrap.style, {
                    display: "flex", gap: "2px", flexShrink: "0",
                    alignItems: "center",
                });
                wrap.append(
                    // "load settings" replays the workflow out of the
                    // SIDECAR, so it has nothing to offer without one
                    ...(pixel ? [] : [mkRestoreBtn(e.clip)]),
                    // The pin toggles are about the NEXT GENERATION, so
                    // in upscale mode there is nothing for them to arm.
                    // Hidden rather than shown inactive: they would read
                    // as unarmed while a pin is in fact stored, and
                    // clicking one would clear the pin this mode is
                    // supposed to be keeping safe. "load settings" stays
                    // -- it reads a sidecar and cares about neither.
                    ...(upscaleMode() ? [] : [
                        mkToggle("prepend", "prepend clip",
                            "Next run prepends into this clip's head"),
                        mkToggle("extend", "extend clip",
                            "Next run extends from this clip's tail")]));
                infoRow.appendChild(wrap);
            }

            // ---- restore settings from the selected clip --------------
            // The take's sidecar carries the workflow that made it; this
            // replays the widget values and bypass state of the groups
            // named by `restore_groups`, plus this node's own pin
            // options. Nothing is created, rewired or moved.
            // Read at CLICK time, never captured: the keyword widget can
            // be edited long after this row was rendered, and a button
            // holding the value it saw at render is stale by definition.
            function restoreKeyword() {
                return String(
                    node.widgets?.find((w) => w.name === "restore_groups")
                        ?.value ?? "").trim();
            }
            function restoreSkipWords() {
                return String(
                    node.widgets?.find(
                        (w) => w.name === "skip_restore_nodes")?.value ?? "")
                    .trim();
            }

            function mkRestoreBtn(clip) {
                const kw = restoreKeyword();
                const b = document.createElement("button");
                b.textContent = "load settings";
                // Same frame as the prepend/extend toggles beside it --
                // it has no active state, so it wears their resting look,
                // fixed width included so the row never shifts.
                Object.assign(b.style, {
                    background: PAL.rest, color: PAL.text,
                    border: "1px solid " + PAL.edge,
                    borderRadius: "4px", cursor: "pointer",
                    width: "88px", textAlign: "center",
                    padding: "1px 0", boxSizing: "border-box",
                    font: "11px/16px sans-serif",
                    whiteSpace: "nowrap", flexShrink: "0",
                    // the one deviation: dimmed while no keyword is set,
                    // which is the only state in which it does nothing
                    opacity: kw ? "1" : "0.45",
                });
                b.addEventListener("mouseenter", () => {
                    b.style.background = "rgba(127,127,127,0.3)";
                });
                b.addEventListener("mouseleave", () => {
                    b.style.background = PAL.rest;
                });
                b.title = kw
                    ? `Load this take's settings into groups whose name `
                      + `contains "${kw}" (widget values + bypass state, `
                      + `top-level nodes only), and set this timeline's `
                      + `pin options to the ones that made it. Asks first.`
                    : "Set 'restore_groups' on this node first: settings "
                      + "are only applied to groups whose name matches it";
                b.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    void restoreFrom(clip, restoreKeyword());
                });
                return b;
            }

            async function restoreFrom(clip, keyword) {
                const say = (severity, summary, detail, life = 5000) => {
                    app.extensionManager?.toast?.add?.(
                        { severity, summary, detail, life });
                    tldbg("restore:", summary, detail);
                };
                let wf = null;
                try {
                    wf = await readSidecarBlob(sidecarValue(clip), "workflow");
                } catch (err) {
                    tldbg("sidecar workflow unreadable:", err);
                }
                // Fallback: the MP4's own container tags. Covers clips
                // with no sidecar at all, ones saved before blobs
                // existed, and videos from core's Save Video or VHS.
                if (!wf) {
                    try {
                        const r = await api.fetchApi(
                            "/obvpm/h3/video_workflow", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ path: clip }),
                            });
                        const d = r.ok ? await r.json() : null;
                        if (d?.workflow) {
                            wf = d.workflow;
                            tldbg("workflow from the mp4 itself:", d.source);
                        }
                    } catch (err) {
                        tldbg("mp4 workflow lookup failed:", err);
                    }
                }
                if (!wf) {
                    say("warn", "Nothing to load",
                        `${clip.split("/").pop()} carries no workflow — `
                        + "neither in its sidecar nor in the MP4. Only "
                        + "clips saved with metadata embedded have one.",
                        7000);
                    return;
                }
                // Top level only, explicitly: a subgraph's interior is
                // never touched, and neither is a group inside one.
                const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
                // DRY RUN first -- nothing is written until the dialog is
                // accepted, so a cancel really does leave the graph alone.
                const skipWords = restoreSkipWords();
                const dry = tlPlanRestore(wf, root, keyword, node.id,
                                          false, skipWords);

                // this node's own pin options, keyword or not
                const savedPin = tlSavedPinState(wf, node.id);
                const pinChanges = savedPin !== undefined && psWidget
                    && psWidget.value !== savedPin;
                let pinRole = "cleared";
                if (pinChanges) {
                    try {
                        pinRole = JSON.parse(savedPin || "{}")?.role
                            || "cleared";
                    } catch (err) { /* keep the raw blob, report loosely */ }
                }
                const pinNote = pinChanges ? `; pin set to ${pinRole}` : "";

                // Nothing loadable -> say why, and never show a dialog
                // offering to do nothing.
                if (dry.reason) {
                    say("warn", "Nothing to load",
                        dry.reason === "no keyword"
                            ? "Set 'restore_groups' on this node first: it "
                              + "names the groups to load settings into."
                            : `No group name contains "${keyword}"`, 7000);
                    return;
                }
                if (!dry.applied.length && !pinChanges) {
                    say("warn", "Nothing to load",
                        `Every node in ${dry.groups.length} matching `
                        + `group(s) already holds this take's values`
                        + (dry.skipped.length || dry.missing.length
                            ? ` (${dry.skipped.length + dry.missing.length} `
                              + "could not be loaded — see the console)" : ""),
                        7000);
                    if (dry.skipped.length || dry.missing.length) {
                        console.info("[obvpm-h3] load settings, not loadable:",
                            { skipped: dry.skipped, missing: dry.missing });
                    }
                    return;
                }

                const go = await tlLoadDialog({
                    clip, keyword, plan: dry,
                    pinRole: pinChanges ? pinRole : null,
                });
                if (!go) return;   // cancelled: nothing was written

                // Only the ticked rows: the same walk again, filtered.
                const plan = tlPlanRestore(wf, root, keyword, node.id,
                                           true, skipWords, go.only);
                if (pinChanges && go.pin) {
                    psWidget.value = savedPin;
                    void refresh();
                }
                const left = plan.skipped.length + plan.missing.length;
                const bits = [`${plan.changes.length} value(s) on `
                    + `${plan.applied.length} node(s)`];
                if (left) bits.push(`${left} left alone`);
                // the per-node detail was already logged before the dialog
                say("success", "Settings loaded",
                    bits.join(", ") + (go.pin ? pinNote : ""));
                app.graph?.setDirtyCanvas?.(true, true);
            }

            // ---- wiring ----------------------------------------------
            const prevCallback = seqWidget.callback;
            seqWidget.callback = function (...args) {
                const r = prevCallback?.apply(this, args);
                void refresh();
                return r;
            };
            // live re-render while typing in the textarea
            let typeTimer = null;
            (seqWidget.element ?? seqWidget.inputEl)?.addEventListener("input", () => {
                clearTimeout(typeTimer);
                typeTimer = setTimeout(() => void refresh(), 400);
            });
            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                const r = onConfigure?.apply(this, arguments);
                // a graph saved while this widget was the run_mode combo
                // restores "generation"/"upscale"; as a BOOLEAN input
                // the server would read either string as true
                if (runModeWidget && typeof runModeWidget.value === "string") {
                    runModeWidget.value = runModeWidget.value === "upscale";
                }
                void refresh();
                return r;
            };
            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                cancelAnimationFrame(rafId);
                clearInterval(durPoll);
                rulerRO.disconnect();
                themeMO.disconnect();
                TL_REGISTRY.delete(node.id);
                dropProgress();
                for (const v of vids) {
                    v.pause();
                    v.removeAttribute("src");
                    v.load();
                }
                return onRemoved?.apply(this, arguments);
            };

            // picker (and thereby seam-popup candidates) scoped by the
            // base_folder widget; re-filtered live when it changes. The
            // node's own preview file lives in that folder too -- hide
            // it, it is a cut, not a source clip.
            let allPickerClips = [];
            function populatePicker() {
                const folder = folderValue();
                const pv = (folder ? folder + "/" : "") +
                    String(widgetValue("preview_filename",
                                       "obvpm_h3_preview")).trim() +
                    ".mp4";
                // hide our own working files (this node's preview under
                // its configured name, and any obvpm_h3_* preview file
                // other nodes drop beside the clips)
                const shown = allPickerClips.filter((c) =>
                    c !== pv &&
                    !c.split("/").pop().startsWith("obvpm_h3_") &&
                    (!folder || c.startsWith(folder + "/")));
                picker.replaceChildren(...shown.map((c) => {
                    const o = document.createElement("option");
                    o.value = o.textContent = c;
                    return o;
                }));
            }
            // /object_info re-runs the picker class's INPUT_TYPES on
            // every request, so each call re-lists the folder: this is
            // how the menus learn about takes saved after page load.
            async function rescanPickerClips() {
                allPickerClips = await tlFetchClipList();
                populatePicker();
            }
            void rescanPickerClips();
            const folderWidget = node.widgets?.find(
                (w) => w.name === "base_folder");
            if (folderWidget) {
                const folderCb = folderWidget.callback;
                folderWidget.callback = function (...args) {
                    const r = folderCb?.apply(this, args);
                    populatePicker();
                    return r;
                };
            }
            // ---- lineage-aimed placement --------------------------------
            // Where a take belongs in this timeline, derived from its
            // lineage. The Result Preview owns the OFFER (its "+ add to
            // timeline" button calls TL_REGISTRY.offer below); this node
            // no longer watches for new clips itself.
            function clipBase(c) { return c.split("/").pop(); }
            function aimFor(p) {
                // multi-pin aware: a bridge take aims after its
                // extend-parent (its prepend-side neighbor is then the
                // gap occupant placeClip resolves against)
                const m = p.meta;
                const ex = tlExtendParent(m);
                const pr = tlPrependChild(m);
                let k = ex ? lastEntries.findIndex((e) =>
                    e.meta && e.meta.self_id === ex.id) : -1;
                if (k >= 0) {
                    return { at: k + 1, parentIdx: k, side: "after",
                             label: "after " + clipBase(lastEntries[k].clip),
                             tail: k === lastEntries.length - 1 };
                }
                k = pr ? lastEntries.findIndex((e) =>
                    e.meta && e.meta.self_id === pr.id) : -1;
                if (k >= 0) {
                    return { at: k, parentIdx: k, side: "before",
                             label: "before " + clipBase(lastEntries[k].clip),
                             tail: false };
                }
                return { at: lastEntries.length, parentIdx: -1,
                         side: null, label: "at the end", tail: false };
            }
            function replaceEntry(idx, clip) {
                const ls = currentLines();
                ls[entryLines()[idx]] = clip;
                setSequence(ls.join("\n"));
            }
            // Insert at the lineage-aimed position -- but when that seam
            // is already occupied by an earlier take of the SAME
            // continuation (same relation, same parent, same join
            // frame), the new take replaces it. Deliberately strict: a
            // merely-linked neighbor (the parent's own prepend, an
            // extend at a different cut point) is a different seam and
            // must not be clobbered.
            // The manual cut that seeded a take was a REQUEST; the take
            // is what actually happened. H3 slices latents on a 17-frame
            // grid, so a cut off that grid is shifted DOWN at generation
            // time -- the server logs it ("extend window shifted to end
            // at frame 141... assemble as this[:141] + new") and the
            // take's sidecar records where it really continues from.
            //
            // Left alone the stale cut plays the parent past that point,
            // so the frames between are shown twice: the take looks
            // right in the Result Preview, which reads the sidecar, and
            // wrong here, which reads the cut. The take wins -- it is
            // the thing that exists.
            //
            // Only the PIN's own cut is rewritten. An unrelated manual
            // cut on the same clip is a decision about something else
            // and has to survive being extended from.
            function reconcileParentCut(meta) {
                const ex = tlExtendParent(meta);
                if (!ex) return null;
                const i = lastEntries.findIndex(
                    (e) => e.meta && e.meta.self_id === ex.id);
                const e = i >= 0 ? lastEntries[i] : null;
                if (!e || e.exitOverride == null) return null;
                // delivered coordinates, exactly as tlDeriveSeam maps
                // them: the parent's own pinned head offsets the join
                const exitF = ex.join
                    - (Number(e.meta.pinned_head_frames) || 0);
                const delivered = Number(e.meta.delivered_frames) || 0;
                const was = e.exitOverride;
                if (was === exitF) return null;
                // Two ways to be sure this cut is the one the take was
                // asked to continue from -- it only takes one.
                const st = pinState();
                const thePinsCut = !!st && st.source === e.clip
                    && Number(st.cut) === was;
                // The pin is gone after a reload, but the arithmetic
                // still proves it: the take's join is exactly where the
                // 17-frame grid would have moved THIS cut -- on the
                // grid, below it, and inside one group of it, so no
                // other legal cut lies between the two.
                const ph = Number(e.meta.pinned_head_frames) || 0;
                const onGrid = (c) => ((ph + c - TL_WINDOW_PHASE)
                    % TL_GROUP + TL_GROUP) % TL_GROUP === 0;
                const gridMoved = exitF < was && was - exitF < TL_GROUP
                    && onGrid(exitF) && !onGrid(was);
                if (!thePinsCut && !gridMoved) {
                    return null;               // somebody else's cut
                }
                // a join at the clip's own end needs no cut at all
                setCut(i, { exit: exitF === delivered ? null : exitF });
                return `${clipBase(e.clip)} now ends at ${exitF}`
                    + ` instead of ${was}: the 17-frame grid moved the`
                    + " take's start";
            }

            function placeClip(clip, meta) {
                const moved = reconcileParentCut(meta);
                const note = (msg) => (moved ? msg + " -- " + moved : msg);
                const aim = aimFor({ clip, meta });
                if (aim.parentIdx >= 0) {
                    // The occupant is whatever sits IN the gap this
                    // take aims at -- the immediate neighbour on that
                    // side, and only while its link is live. A clip
                    // that merely shares lineage but has been dragged
                    // elsewhere is a different position in the cut now,
                    // and must not be silently removed.
                    const held = tlOccupant(
                        lastEntries, lastEntries[aim.parentIdx]?.clip,
                        aim.side === "after" ? "after" : "before",
                        linkedMeta);
                    if (held && held.entry.clip !== clip) {
                        replaceEntry(held.idx, clip);
                        return note("replaced "
                            + clipBase(held.entry.clip) + " (same gap)");
                    }
                }
                return note(insertFilling(aim, clip, meta));
            }
            // Insert the take, and when it lands in empty space let the
            // space give up the take's length -- so the take FILLS the
            // hole instead of pushing it, and everything past it, along.
            // That is what the gap-fitted duration is for, and what the
            // ruler's landing band shows. A take shorter than the hole
            // leaves the rest of it behind; a longer one uses it all and
            // pushes only the difference. A bridge whose other foot is
            // the clip right after the hole takes the whole hole either
            // way: a remainder would stand between it and the clip it
            // was generated to lead into.
            // One sequence write for both edits, so the undo step and the
            // refresh see a single change.
            function insertFilling(aim, clip, meta) {
                const es = lastEntries;
                const ls = currentLines();
                const map = entryLines();
                let at = aim.at >= map.length ? ls.length : map[aim.at];
                const gi = aim.side === "after" ? aim.at
                    : aim.side === "before" ? aim.at - 1 : -1;
                const hole = gi >= 0 ? es?.[gi] : null;
                const took = Number(meta?.delivered_frames) || 0;
                let filled = "";
                if (hole && hole.gap != null && map[gi] != null && took) {
                    const child = tlPrependChild(meta);
                    const bridged = aim.side === "after" && !!child
                        && es[gi + 1]?.meta?.self_id === child.id;
                    const left = bridged ? 0
                        : Math.max(0, Math.round(Number(hole.gap) || 0) - took);
                    if (left > 0) {
                        ls[map[gi]] = `~ ${left}`;
                        filled = `, ${(left / TL_FPS).toFixed(1)}s of empty `
                            + "space left";
                    } else {
                        ls.splice(map[gi], 1);
                        if (map[gi] < at) at -= 1;
                        filled = ", filling the empty space";
                    }
                }
                ls.splice(at, 0, clip);
                setSequence(ls.join("\n"));
                return "inserted " + aim.label + filled;
            }

            TL_REGISTRY.set(node.id, {
                folder: () => folderValue(),
                // so the Result Preview can build its clip pair the same
                // way the export will: judging a take against a
                // differently-repaired preview is judging the wrong thing
                fixOpts: () => fixOpts(),
                crf: () => crfValue(),
                has: (clip) => lastEntries.some((e) => e.clip === clip),
                // Where the SAME join sits here: those two clips, in
                // that order, next to each other. -1 when they are not,
                // which is the only safe answer -- a join is a PAIR,
                // and settings written against a different pair are
                // settings on a different seam. This is what lets the
                // Result Preview edit a join it shares with the
                // timeline in place, instead of keeping a second copy
                // that the export would then ignore.
                joinAt: (leftClip, rightClip) => {
                    for (let i = 1; i < lastEntries.length; i++) {
                        if (lastEntries[i - 1].clip === leftClip
                                && lastEntries[i].clip === rightClip) {
                            return i;
                        }
                    }
                    return -1;
                },
                seamOptsAt: (i) => seamOptsAt(i),
                setSeamOptAt: (i, key, value) => setSeamOpt(i, key, value),
                clearSeamOptsAt: (i) => clearSeamOpts(i),
                // `seamOpts` is {clipPath: {...}} -- the per-join
                // settings the take was judged under, so what was
                // approved in the Result Preview is what gets built
                // here. Applied AFTER placement, because until the take
                // is in the sequence there is no join to attach them to.
                offer: async (clip, seamOpts) => {
                    if (lastEntries.some((e) => e.clip === clip)) {
                        return "already in the timeline";
                    }
                    const meta = await tlClipMeta(clip);
                    const msg = placeClip(clip, meta);
                    const n = applySeamOptsMap(seamOpts);
                    return n
                        ? `${msg}, with ${n === 1 ? "its own seam settings"
                            : n + " joins' seam settings"}`
                        : msg;
                },
                forget: (clip) => {
                    allPickerClips = allPickerClips.filter(
                        (c) => c !== clip);
                    populatePicker();
                    const idx = lastEntries.findIndex(
                        (e) => e.clip === clip);
                    if (idx >= 0) removeEntry(idx);
                },
            });

            void refresh();
            return result;
        }
    },
});

// ================= H3ResultPreview: mini lineage timeline =============
// A run finishes -> the Python node returns {clip, parent, relation,
// sequence} -> this widget shows the pair as two blocks with the seam
// pill between them and plays the sequence as ONE server-built smart
// cut (the same /obvpm/h3/preview_cut the Timeline uses, its own
// preview file). The payload rides node.properties so it survives a
// workflow save/reload.
const RP_NODE = "H3ResultPreview";
const RP_PREVIEW_NAME = "obvpm_h3_result";

// ---- whose run was that? -------------------------------------------
// ComfyUI delivers `executed` by NODE ID into whichever workflow is open
// (app.ts: getNodeByExecutionId(rootGraph, ...)) and wipes stored outputs
// on every workflow switch (loadGraphData -> clean()). Two copies of one
// workflow share node ids, so the open copy would show the other copy's
// result while the backgrounded copy — whose nodes are not in the open
// graph — sees nothing at all. Route on the thing that actually differs:
// the output FOLDER a preview watches, read off the Save node wired into
// its `path` input. Listening to the raw event also means a run that
// finishes while its workflow is in the background is still captured.
const RP_RESULTS = [];          // recent payloads, newest last (bounded)
const RP_LIVE = new Set();      // live previews, each {accepts, show, foreign}

function rpFolderOf(clip) {
    const c = String(clip ?? "");
    const i = c.lastIndexOf("/");
    return i >= 0 ? c.slice(0, i) : "";
}

// The save nodes hold the folder and the name prefix apart; a clip path
// is the two joined. This mirrors _save_prefix() in h3/nodes_save.py and
// is the ONLY place that join is written on this side -- every comparison
// below goes through it, because a scope built by one rule and matched by
// another is silently unmatchable (which is exactly how this broke).
function rpJoinPrefix(folder, name) {
    const f = String(folder ?? "").trim().replace(/^[/\\]+|[/\\]+$/g, "");
    const n = String(name ?? "").trim().replace(/^[/\\]+/, "");
    return f && n ? `${f}/${n}` : (f || n);
}

// ---- resolving a wire-driven widget --------------------------------
// A widget's value is DEAD once its socket is connected: the real value
// arrives at execution time from upstream and never lands on the widget.
// Reading it anyway yields a stale default -- h3_r2v_new.json stores
// base_folder "h3" on a save node whose base_folder is actually fed from
// a PrimitiveString inside a subgraph. So FOLLOW THE WIRE instead, and
// only report "unknown" when the far end genuinely can't be read.
const RP_HOPS = 8;

function rpLinkById(graph, id) {
    const links = graph?.links;
    if (!links || id == null) return null;
    // an object at the root, a Map inside a subgraph
    return typeof links.get === "function" ? links.get(id) : links[id];
}

// What feeds `node`'s input slot, as a literal string. null = can't tell.
// A subgraph's INPUT boundary. An inner link whose origin is the
// boundary carries a value from OUTSIDE the subgraph, so resolving it
// means popping back OUT to that slot on the instance node -- which is
// commonly a promoted widget (h3_obvpm_r2v_test_import.json drives
// base_folder exactly that way: instance widget -> boundary ->
// PrimitiveString -> subgraph output). Descending alone is not enough;
// without this every trace dies the moment it reaches a promoted input,
// and a dead trace reads as "scope unknown", which is permissive.
const RP_IN_BOUNDARY = -10;

function rpIsInputBoundary(graph, id) {
    if (id === RP_IN_BOUNDARY) return true;
    const b = graph?.inputNode;
    return !!b && b.id === id;
}

// `ctx` is the instance node we descended through, plus ITS ctx, so a
// pop-out can itself pop out again from a nested subgraph.
function rpResolveInput(node, slot, hops, ctx) {
    if (!node || hops <= 0 || slot?.link == null) return null;
    const link = rpLinkById(node.graph, slot.link);
    if (!link) return null;
    if (rpIsInputBoundary(node.graph, link.origin_id)) {
        return ctx
            ? rpResolveSlot(ctx.node, link.origin_slot, hops - 1, ctx.parent)
            : null;
    }
    const from = node.graph?.getNodeById?.(link.origin_id);
    return from ? rpResolveOutput(from, link.origin_slot, hops - 1, ctx) : null;
}

// What arrives at `node`'s input slot `index`: the wire when one is
// connected, otherwise the widget that backs the slot. Matched by NAME,
// never by position -- only the widget-backed slots carry widgets, so
// slot index and widget index do not correspond (slot 12 of the test
// workflow's subgraph is widget 4).
function rpResolveSlot(node, index, hops, ctx) {
    const slot = node?.inputs?.[index];
    if (!slot) return null;
    if (slot.link != null) return rpResolveInput(node, slot, hops, ctx);
    const name = slot.widget?.name ?? slot.name;
    const w = node.widgets?.find((x) => x.name === name);
    if (!w || w.value == null) return null;
    return typeof w.value === "string" ? w.value : String(w.value);
}

// What `node` emits on an output slot, as a literal string.
function rpResolveOutput(node, slotIndex, hops, ctx) {
    if (!node || hops <= 0) return null;

    // A subgraph instance: descend to whatever drives that output. Two
    // shapes are accepted because this part of litegraph keeps moving --
    // SubgraphOutput.getLinks(), else the -20 boundary node's slots.
    const sub = node.subgraph;
    if (sub) {
        const out = sub.outputs?.[slotIndex];
        const viaSlot = out?.getLinks?.()?.[0];
        const boundary = sub.outputNode;
        const viaBoundary = viaSlot
            ? null : rpLinkById(sub, boundary?.inputs?.[slotIndex]?.link);
        const link = viaSlot ?? viaBoundary;
        if (!link) return null;
        // a subgraph that passes one of its own inputs straight through
        const inner = { node, parent: ctx };
        if (rpIsInputBoundary(sub, link.origin_id)) {
            return rpResolveSlot(node, link.origin_slot, hops - 1, ctx);
        }
        const from = sub.getNodeById?.(link.origin_id);
        return from
            ? rpResolveOutput(from, link.origin_slot, hops - 1, inner) : null;
    }

    // KJNodes Set/Get: the Get's widget holds the CONSTANT NAME, not
    // the value -- the single-widget fallback below would happily
    // return "base_folder" as if it were the folder, and the scope it
    // builds matches no clip ever written (silent preview, 2026-09-01,
    // the refine retrofit's Set/Get wiring). Hop to the Set node
    // carrying the same constant -- same graph, which is how the pair
    // resolves itself -- and read what feeds it.
    if (node.type === "GetNode") {
        const key = String(node.widgets?.[0]?.value ?? "");
        const setter = (node.graph?._nodes ?? []).find(
            (n) => n.type === "SetNode"
                   && String(n.widgets?.[0]?.value ?? "") === key);
        return setter
            ? rpResolveInput(setter, setter.inputs?.[0], hops - 1, ctx)
            : null;
    }
    if (node.type === "SetNode") {
        return rpResolveInput(node, node.inputs?.[0], hops - 1, ctx);
    }

    // A reroute or other pass-through: one in, one out, nothing of its own.
    if (node.inputs?.length === 1 && node.outputs?.length === 1
            && !node.widgets?.length) {
        return rpResolveInput(node, node.inputs[0], hops - 1, ctx);
    }

    // A literal: PrimitiveString and friends keep it in a `value` widget;
    // a single-widget node can only mean that widget.
    const w = node.widgets?.find((x) => x.name === "value")
        ?? (node.widgets?.length === 1 ? node.widgets[0] : null);
    if (!w) return null;
    const wired = (node.inputs ?? []).find(
        (s) => (s.widget?.name === w.name || s.name === w.name)
               && s.link != null);
    if (wired) return rpResolveInput(node, wired, hops - 1, ctx);
    // Numbers count. This read "string, else null" while its sibling
    // rpResolveSlot stringified -- written when the only things traced
    // were base_folder and filename_prefix. The moment a NUMERIC widget
    // was traced (duration_seconds, driven by a primitive whose value is
    // 8, not "8") it reported "cannot tell" for a value sitting right
    // there. Same rule both sides now: absent is unknown, anything else
    // is its string form.
    if (w.value == null) return null;
    return typeof w.value === "string" ? w.value : String(w.value);
}

//   undefined = this node has no such widget
//   null      = it exists but its value cannot be established
//   string    = the value, read off the widget or traced up the wire
function rpWidgetValue(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return undefined;
    const wired = (node.inputs ?? []).find(
        (s) => (s.widget?.name === name || s.name === name) && s.link != null);
    if (!wired) return String(w.value ?? "");
    const traced = rpResolveInput(node, wired, RP_HOPS, null);
    return traced == null ? null : String(traced);
}

// The joined save prefix of one node: undefined if it is not a save node,
// null if it is but its prefix cannot be known. H3 Joint VAE Decode and Save counts:
// it writes "<base_folder>/<prefix>_NNNNN.mp4" like the save nodes do.
function rpSavePrefix(node) {
    const name = rpWidgetValue(node, "filename_prefix");
    if (name === undefined) return undefined;
    const folder = rpWidgetValue(node, "base_folder");
    if (name === null || folder === null) return null;
    const prefix = rpJoinPrefix(folder ?? "", name);
    // %date%-style tokens can't be compared against a real path; treat
    // that as "unknown" rather than filtering everything out
    return !prefix || prefix.includes("%") ? null : prefix;
}

// The save target this preview is responsible for — the whole save prefix
// of the Save node wired into its `path` input, e.g. "space_battle/clip".
// The full prefix, not just its folder: two copies of a workflow may well
// write into one folder under different names.
// null when it cannot be determined — then nothing is filtered.
function rpScopeFolder(node) {
    let src = null;
    try {
        src = node.getInputNode?.(0) ?? null;
    } catch (err) {
        src = null; // getInputNode throws while the graph is still loading
    }
    for (let hop = 0; src && hop < 6; hop++) {
        const prefix = rpSavePrefix(src);
        if (prefix !== undefined) return prefix;
        try {
            // KJNodes Get: no input of its own -- the wire continues at
            // the Set node carrying the same constant. Without this hop
            // a preview fed through a Get reads as scope unknown, and
            // two unknown previews both show every result (2026-09-14).
            if (src.type === "GetNode") {
                const key = String(src.widgets?.[0]?.value ?? "");
                src = (src.graph?._nodes ?? []).find(
                    (n) => n.type === "SetNode"
                           && String(n.widgets?.[0]?.value ?? "") === key) ?? null;
            }
            src = src?.getInputNode?.(0) ?? null;
        } catch (err) {
            src = null;
        }
    }
    return null;
}

// "yes" | "no" | "unknown". Kept three-valued on purpose: a preview whose
// scope can't be established must not be treated the same as one that
// positively matches, or it swallows every other workflow's results.
function rpVerdict(node, d) {
    const scope = rpScopeFolder(node);
    const clip = String(d?.clip ?? "");
    if (scope === null || !clip) return "unknown";
    // the save nodes write "<prefix>_00042.mp4"; anything not shaped that
    // way falls back to the folder so nothing legitimate is dropped
    const m = clip.match(/^(.*)_\d+\.[A-Za-z0-9]+$/);
    const hit = m ? m[1] === scope : rpFolderOf(clip) === rpFolderOf(scope);
    return hit ? "yes" : "no";
}

// Permissive form, for paths where showing something beats showing
// nothing (restoring a payload saved into the workflow file).
function rpAccepts(node, d) {
    return rpVerdict(node, d) !== "no";
}

// Deliver to the previews that positively match; only if NOBODY does may
// the undecidable ones have it. That is what stops a preview whose scope
// is unreadable from displaying a clip that demonstrably belongs to
// another workflow, while still leaving it useful when it is the only
// candidate around.
function rpDeliver(d) {
    const live = [...RP_LIVE].filter((l) => {
        if (l.alive()) return true;
        RP_LIVE.delete(l);   // a workflow switch clears the graph
        return false;
    });
    const verdicts = live.map((l) => [l, l.verdict(d)]);
    const claimed = verdicts.some(([, v]) => v === "yes");
    for (const [l, v] of verdicts) {
        if (v === "yes" || (!claimed && v === "unknown")) l.show(d);
    }
    return { claimed, count: verdicts.length };
}

// Does this queued job write to `scope`? A /queue item is
// [number, prompt_id, prompt, extra_data, outputs] and the prompt is the
// whole API-format graph, so ownership is decided by the same
// filename_prefix that results are matched on. A prompt_id on its own
// says nothing about which workflow produced it.
// The joined prefix has to be rebuilt here too: in the API format the
// save node carries base_folder and filename_prefix as SEPARATE inputs,
// so comparing the bare filename_prefix against a joined scope never
// matches and every run looks like somebody else's.
// A wired input appears as [node_id, slot] rather than a string, which is
// the normal case here -- base_folder is commonly driven from a shared
// value. Follow it: the API prompt is FLAT (subgraphs are already
// expanded), so the far end is a real node whose literal sits in its own
// inputs. Returns null when it cannot be resolved.
function rpPromptValue(prompt, v, hops) {
    if (typeof v === "string") return v;
    if (hops <= 0 || !Array.isArray(v) || !v.length) return null;
    const inputs = prompt?.[String(v[0])]?.inputs;
    if (!inputs) return null;
    for (const key of ["value", "string", "text"]) {
        if (inputs[key] !== undefined) {
            return rpPromptValue(prompt, inputs[key], hops - 1);
        }
    }
    const keys = Object.keys(inputs);   // a pass-through has exactly one
    return keys.length === 1
        ? rpPromptValue(prompt, inputs[keys[0]], hops - 1) : null;
}

function rpPromptWrites(prompt, scope) {
    if (!prompt || typeof prompt !== "object") return false;
    for (const n of Object.values(prompt)) {
        if (n?.inputs?.filename_prefix === undefined) continue;
        const p = rpPromptValue(prompt, n.inputs.filename_prefix, RP_HOPS);
        if (p == null) continue;
        // ABSENT and UNREADABLE are different: a save node with no
        // base_folder really does write at the root, but one whose folder
        // can't be resolved must not be judged as if it were empty -- that
        // makes "folder/clip" look like a match for scope "clip".
        let f = "";
        if (n.inputs.base_folder !== undefined) {
            f = rpPromptValue(prompt, n.inputs.base_folder, RP_HOPS);
            if (f == null) continue;
        }
        if (rpJoinPrefix(f, p) === scope) return true;
    }
    return false;
}

// "yes" | "no" | "unknown" — unknown whenever the answer can't be
// established, so callers stay permissive rather than going silent.
async function rpQueueOwns(scope, promptId) {
    if (scope === null) return "unknown";
    let q = null;
    try {
        const r = await api.fetchApi("/queue");
        if (r.ok) q = await r.json();
    } catch (err) {
        q = null;
    }
    if (!q) return "unknown";
    const jobs = [...(q.queue_running ?? []), ...(q.queue_pending ?? [])];
    if (promptId == null) {
        return jobs.some((j) => rpPromptWrites(j?.[2], scope)) ? "yes" : "no";
    }
    const job = jobs.find((j) => j?.[1] === promptId);
    // gone from the queue already (very short run) -> can't tell
    if (!job) return "unknown";
    return rpPromptWrites(job[2], scope) ? "yes" : "no";
}

// Console diagnostic: obvpmRPDiag(). A silent preview is almost always a
// scope that resolves to something no clip can match, and the scope is
// computed from two widgets on a node several hops away -- so print the
// whole chain rather than making the user guess which end is wrong.
globalThis.obvpmRPDiag = function obvpmRPDiag() {
    const out = [...RP_LIVE].map((live) => {
        const node = live.node;
        let src = null;
        try {
            src = node?.getInputNode?.(0) ?? null;
        } catch (err) {
            src = null;
        }
        let save = null;
        for (let hop = 0; src && hop < 4; hop++) {
            if (rpSavePrefix(src) !== undefined) { save = src; break; }
            try {
                src = src.getInputNode?.(0) ?? null;
            } catch (err) {
                src = null;
            }
        }
        return {
            node: node?.id,
            scope: rpScopeFolder(node),
            saveNode: save ? `${save.type}#${save.id}` : "(none found via path)",
            base_folder: save ? rpWidgetValue(save, "base_folder") : undefined,
            filename_prefix: save
                ? rpWidgetValue(save, "filename_prefix") : undefined,
            wired: save ? (save.inputs ?? [])
                .filter((s) => s.link != null && (s.widget || s.name))
                .map((s) => s.widget?.name ?? s.name) : [],
            verdicts: RP_RESULTS.map(
                (d) => `${rpVerdict(node, d).padEnd(7)} ${d.clip}`),
        };
    });
    console.log("[obvpm-rp] previews:", out,
                "| cached results:", RP_RESULTS.map((d) => d.clip));
    return out;
};

api.addEventListener("executed", ({ detail }) => {
    const d = detail?.output?.h3_result?.[0];
    if (!d?.clip) return;
    RP_RESULTS.push(d);
    if (RP_RESULTS.length > 20) RP_RESULTS.shift();
    tldbg("result event:", d.clip, "| live previews:",
          [...RP_LIVE].map((l) => l.describe()).join(", ") || "(none)");
    rpDeliver(d);
});

app.registerExtension({
    name: "obvpm.h3_apply_socket_order",
    // The VAEs lead the node (user preference 2026-08-20). They are
    // OPTIONAL server-side -- latent-grade pins never touch a VAE, so
    // promoting them to required would refuse perfectly valid graphs --
    // and ComfyUI always renders required sockets before optional ones.
    // So the order is set here, on the INSTANCE, at creation: socket
    // order is client-side data, links serialize against the instance's
    // own arrays, and the api prompt maps inputs by NAME. Existing saved
    // nodes keep their saved order (configure restores it after
    // onNodeCreated); only freshly added nodes are touched.
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3MCtxApplyPins") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                const lead = ["video_vae", "audio_vae"];
                this.inputs.sort((a, b) =>
                    (lead.indexOf(a.name) + 1 || lead.length + 2)
                    - (lead.indexOf(b.name) + 1 || lead.length + 2));
            } catch (err) {
                console.error("[obvpm-h3] socket reorder failed:", err);
            }
            return result;
        };
    },
});

app.registerExtension({
    name: "obvpm.h3_mctx_result",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== RP_NODE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                buildResultPreview(this);
            } catch (err) {
                console.error("[obvpm-h3-result] widget build FAILED:",
                              err);
            }
            return result;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            // Fallback only: the module-level `executed` listener above is
            // the real delivery path (it sees every run, not just the open
            // workflow's). This still fires for ComfyUI's id-routed
            // delivery, so it must respect the same scope; _h3ShowResult
            // ignores a repeat of what is already on screen.
            // Strict, not permissive: this is precisely the id-routed
            // delivery that hands the OPEN workflow a payload belonging to
            // another copy with the same node ids. The module-level
            // listener has already dealt with the undecidable case.
            const d = message?.h3_result?.[0];
            if (d && rpVerdict(this, d) === "yes") this._h3ShowResult?.(d);
            return r;
        };
    },
});

function buildResultPreview(node) {
    let PAL = themePalette();

    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex", flexDirection: "column", gap: "5px",
        font: "12px sans-serif", overflow: "hidden",
    });

    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    Object.assign(video.style, {
        width: "100%", flex: "1", minHeight: "0",
        objectFit: "contain", borderRadius: "4px", display: "none",
        background: "#000",
    });

    // The LIVE picture while a run this preview owns is sampling: KJ's
    // Model Preview Override pushes each step's preview over the
    // websocket as a `kj_preview_override` event (a JPEG, an animated
    // WebP, or an MP4 when NVENC is available), tagged with ITS node id
    // and painted on that node by its own script. Listening here as
    // well puts the same picture in the box the result will take over,
    // so the two previews read as one panel and the KJ node can be
    // collapsed. No KJ node in the graph -> no events -> nothing here
    // changes. An <img> plays an animated WebP by itself; MP4 needs
    // the <video>.
    const liveImg = document.createElement("img");
    const liveVideo = document.createElement("video");
    liveVideo.muted = true;
    liveVideo.loop = true;
    liveVideo.autoplay = true;
    liveVideo.playsInline = true;
    for (const el of [liveImg, liveVideo]) {
        Object.assign(el.style, {
            width: "100%", flex: "1", minHeight: "0",
            objectFit: "contain", borderRadius: "4px", display: "none",
            background: "#000",
        });
    }
    let liveUrl = null;          // the object URL on screen, if any
    let liveStep = null;         // {step, total} of the last frame
    function liveClear() {
        liveImg.style.display = "none";
        liveVideo.style.display = "none";
        liveVideo.pause();
        liveImg.removeAttribute("src");
        liveVideo.removeAttribute("src");
        if (liveUrl) URL.revokeObjectURL(liveUrl);
        liveUrl = null;
        liveStep = null;
    }
    function liveShow(data) {
        const mime = typeof data.mime === "string" ? data.mime : "image/jpeg";
        let blob;
        try {
            const bin = atob(data.image);
            const bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
            blob = new Blob([bytes], { type: mime });
        } catch (err) {
            return;
        }
        const url = URL.createObjectURL(blob);
        const old = liveUrl;
        liveUrl = url;
        // the result of the previous run gives way to the live picture,
        // its clip strip and buttons with it; stopClock() brings them
        // back if this run leaves no result
        video.style.display = "none";
        strip.style.display = "none";
        btnRow.style.display = "none";
        if (mime === "video/mp4") {
            liveImg.style.display = "none";
            liveVideo.src = url;
            liveVideo.style.display = "block";
            void liveVideo.play?.().catch?.(() => {});
        } else {
            liveVideo.style.display = "none";
            liveImg.src = url;
            liveImg.style.display = "block";
        }
        if (old) URL.revokeObjectURL(old);
    }

    const strip = document.createElement("div");
    Object.assign(strip.style, {
        display: "none", gap: "4px", alignItems: "stretch",
    });

    const status = document.createElement("div");
    Object.assign(status.style, {
        font: "11px sans-serif", color: PAL.sub,
        whiteSpace: "nowrap", overflow: "hidden",
        textOverflow: "ellipsis",
    });
    status.textContent = "run a generation to see its result here";

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style,
        { display: "none", gap: "6px", alignItems: "center",
          justifyContent: "center" });
    const rpBtn = (label, title) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.title = title;
        Object.assign(b.style, {
            background: PAL.rest, color: PAL.text,
            border: "1px solid " + PAL.edge, borderRadius: "4px",
            padding: "2px 8px", cursor: "pointer",
            // explicit line-height: the ✕ glyph renders taller than
            // latin text and would otherwise grow its button
            font: "11px/16px sans-serif",
        });
        return b;
    };
    const addTlBtn = rpBtn("+ add to timeline",
        "Insert this take into the Timeline node at its "
        + "lineage-derived position");
    // the accept action is green like the take it accepts
    Object.assign(addTlBtn.style, {
        background: "#2e7d4f", color: "#f0f6f1",
        borderColor: "#1f5c38",
    });
    const dismissBtn = rpBtn("dismiss",
        "Hide these buttons (the preview and the take stay)");
    const delBtn = rpBtn("✕ delete",
        "Delete this take (MP4 + sidecar) -- for rejected seeds");
    // destructive action wears red; dismiss stays neutral
    Object.assign(delBtn.style, {
        background: "#8c3432", color: "#f6efef",
        borderColor: "#6d2624",
    });
    btnRow.append(addTlBtn, delBtn, dismissBtn);

    const rpProgress = tlProgressBar();
    const dropRpProgress = tlOnProgress(() => RP_PREVIEW_NAME, rpProgress);
    container.append(liveImg, liveVideo, video, strip, rpProgress.el, btnRow, status);
    const widget = node.addDOMWidget("mctx_result", "div", container,
                                     { hideOnZoom: false });
    widget.serialize = false;
    widget.options.serialize = false;
    tlPanWithMiddleButton(container);
    widget.computeLayoutSize = () => ({ minHeight: 220, minWidth: 240 });
    Object.defineProperty(widget, "width", {
        configurable: true, get: () => undefined, set: () => {},
    });

    let starts = [];
    let reqSeq = 0;
    let blockEls = [];
    // which clip is on screen: BLUE marking follows the playhead
    // (border swap + inset shadow, not background -- the green
    // identity of "this run" must survive being the active block)
    const RP_PLAY_BLUE = "#4a8fd4";
    video.addEventListener("timeupdate", () => {
        if (!blockEls.length || !starts.length) return;
        const f = video.currentTime * TL_FPS;
        let idx = 0;
        starts.forEach((s, k) => {
            if (f >= s - 0.5) idx = k;
        });
        blockEls.forEach((b, k) => {
            // inset shadow, not outline: outlines paint OUTSIDE the
            // element and the strip's edges clip them. The 1px border
            // flips to the same blue so no other edge color remains.
            b.style.boxShadow = k === idx
                ? "inset 0 0 0 2px " + RP_PLAY_BLUE : "none";
            b.style.borderColor = k === idx
                ? RP_PLAY_BLUE : (b._edgeColor ?? PAL.edge);
        });
    });
    function block(label, sub, k, isNew) {
        const b = document.createElement("div");
        Object.assign(b.style, {
            flex: "1", minWidth: "0", padding: "3px 7px",
            borderRadius: "4px", cursor: "pointer",
            overflow: "hidden",
            // the run's own take is green so the eye lands on it
            background: isNew ? "#2e7d4f" : PAL.rest,
            color: isNew ? "#f0f6f1" : PAL.text,
            border: "1px solid " + (isNew ? "#1f5c38" : PAL.edge),
        });
        // resting border color, restored when the playhead moves on
        b._edgeColor = isNew ? "#1f5c38" : PAL.edge;
        const nm = document.createElement("div");
        Object.assign(nm.style, {
            font: "600 11px sans-serif", whiteSpace: "nowrap",
            overflow: "hidden", textOverflow: "ellipsis",
        });
        nm.textContent = label.split("/").pop();
        nm.title = label;
        const s = document.createElement("div");
        Object.assign(s.style, {
            font: (isNew ? "bold " : "") + "10px sans-serif",
            opacity: isNew ? "0.9" : "0.65",
        });
        s.textContent = sub;
        b.append(nm, s);
        b.addEventListener("click", () => {
            if (video.src && starts[k] !== undefined) {
                video.currentTime = starts[k] / TL_FPS;
                void video.play().catch(() => {});
            }
        });
        return b;
    }

    let current = null;

    // ---- this take's own seam settings ------------------------------
    // A take is accepted or rejected on how its JOIN looks, so the
    // settings that decide how that join is built belong here, next to
    // the picture -- not only in the Timeline the take has not been
    // added to yet.
    //
    // Keyed by the clip to the RIGHT of the join, exactly as the
    // sequence line keys them, so handing them over is a lookup and not
    // a translation. On node.properties so a workflow save keeps them
    // alongside the h3_result payload they describe.
    function rpAllSeamOpts() {
        const o = node.properties?.h3_seam_opts;
        return (o && typeof o === "object") ? o : {};
    }
    function rpSeamOptsFor(clip) {
        return rpAllSeamOpts()[clip] || {};
    }
    function rpWriteSeamOpts(clip, own) {
        const all = { ...rpAllSeamOpts() };
        if (own && Object.keys(own).length) all[clip] = own;
        else delete all[clip];
        node.properties.h3_seam_opts = all;
    }
    function rpSetSeamOpt(clip, key, value) {
        const own = { ...rpSeamOptsFor(clip) };
        if (value === undefined || value === null) delete own[key];
        else own[key] = value;
        rpWriteSeamOpts(clip, own);
    }
    // A new take is a new question; settings named for clips that are no
    // longer on screen would otherwise accumulate in the workflow file
    // forever. Keys still in the sequence survive, so flipping back and
    // forth between takes does not lose them.
    function rpPruneSeamOpts(clips) {
        const keep = new Set(clips);
        const all = rpAllSeamOpts();
        const next = {};
        for (const [k, v] of Object.entries(all)) {
            if (keep.has(k)) next[k] = v;
        }
        if (Object.keys(next).length !== Object.keys(all).length) {
            node.properties.h3_seam_opts = next;
        }
    }
    // Once a join exists in a Timeline too, THAT is where its settings
    // live -- editing a private copy here would show a repaired picture
    // the export knows nothing about. Matched on the PAIR of clips, so
    // a bridge take's tail join (whose right-hand clip is already in the
    // timeline, but next to something else) is correctly treated as a
    // different seam and stays local until the take is added.
    function rpOwnerJoin(clips, k) {
        if (k < 1) return null;
        for (const t of tlMatchesFor(clips[k])) {
            let i = -1;
            try {
                i = t.joinAt ? t.joinAt(clips[k - 1], clips[k]) : -1;
            } catch (err) {
                i = -1;
            }
            if (i >= 0) return { t, i };
        }
        return null;
    }
    function rpJoinOpts(clips, k) {
        const o = rpOwnerJoin(clips, k);
        if (!o) return rpSeamOptsFor(clips[k]);
        try {
            return o.t.seamOptsAt(o.i) || {};
        } catch (err) {
            return {};
        }
    }
    // The sequence the build is actually asked for: the same clips, with
    // each join's own settings written into the line. The server already
    // reads that section and resolves it against the base options
    // (preview_route._seam_plan), so this needs nothing new server-side
    // -- and it means the Result Preview and the export go through one
    // format, not two.
    function rpSequenceText(d) {
        const clips = d.sequence.split("\n");
        return clips.map((c, k) => c +
            (k > 0 ? tlFormatSeamOpts(rpJoinOpts(clips, k)) : ""))
            .join("\n");
    }
    // What this join inherits when it says nothing of its own: the
    // owning Timeline's setting, or the shared default when no Timeline
    // claims the folder -- which is also what the server would apply.
    function rpInherited(clip, key) {
        const base = rpFixOpts(clip);
        return base[key] !== undefined ? base[key] : TL_FIX_DEFAULTS[key];
    }

    // The dialog writes immediately, but the REBUILD waits for it to be
    // closed. Every control is its own decision and a build is seconds
    // of work, so rebuilding per toggle spends most of its time making
    // pictures of settings that were on their way somewhere else.
    // "Done", Escape and the backdrop all close through onClose, so
    // there is one place to catch it.
    let rpJoinDialogClose = null;
    let rpJoinDialogDirty = false;
    function rpCloseSeamDialog(rebuild) {
        if (!rpJoinDialogClose) return;
        // a new take is about to be rendered anyway; rebuilding for the
        // join that just left the screen would build the wrong thing
        if (!rebuild) rpJoinDialogDirty = false;
        rpJoinDialogClose();
    }
    function rpOpenSeamDialog(d, k, clips, metas) {
        rpCloseSeamDialog(true);
        const right = clips[k];
        const own = () => rpJoinOpts(clips, k);
        // one writer, so it cannot matter to any caller which store is
        // behind it
        const put = (key, v) => {
            const o = rpOwnerJoin(clips, k);
            if (o) o.t.setSeamOptAt(o.i, key, v);
            else rpSetSeamOpt(right, key, v);
            rpJoinDialogDirty = true;
        };
        const shared = !!rpOwnerJoin(clips, k);
        const entry = (i) => ({ clip: clips[i], gap: null, meta: metas[i],
                                enter: 0, exit: null });
        const name = (c) => c.split("/").pop();
        rpJoinDialogClose = tlSeamDialog({
            get: (key) => {
                const o = own();
                return o[key] !== undefined ? o[key]
                    : rpInherited(d.clip, key);
            },
            set: put,
            seams: null,
            scope: {
                title: "This Join's Seam Settings",
                sub: name(clips[k - 1]) + " → " + name(right)
                    + ".  The preview rebuilds as you change them, and "
                    + (shared
                        ? "this join is in the timeline already, so they "
                          + "are its settings you are changing."
                        : "they go with the take when it is added to the "
                          + "timeline.")
                    + "  Anything left as inherited follows the "
                    + "timeline's own settings.",
                applies: (key) => {
                    const cap = tlSeamCapability(entry(k - 1), entry(k));
                    return !cap || cap[TL_SEAM_NEEDS[key]] !== false;
                },
                own: (key) => own()[key] !== undefined,
                clear: (key) => put(key, undefined),
                clearAll: () => {
                    const o = rpOwnerJoin(clips, k);
                    if (o) o.t.clearSeamOptsAt(o.i);
                    else rpWriteSeamOpts(right, null);
                    rpJoinDialogDirty = true;
                },
                count: () => Object.keys(own()).length,
            },
            onClose: () => {
                rpJoinDialogClose = null;
                if (!rpJoinDialogDirty) return;
                rpJoinDialogDirty = false;
                rpRebuild();
            },
        });
    }
    // Rebuilding IS re-rendering: the strip's badges have to restate the
    // new counts anyway, and one path cannot disagree with itself. The
    // dismissal survives, because changing a setting is not a new take.
    function rpRebuild() {
        if (current) void render(current, { keep: true });
    }

    async function render(d, opts) {
        PAL = themePalette();
        current = d;
        const seq = ++reqSeq;
        blockEls = [];
        strip.replaceChildren();
        strip.style.display = "flex";
        if (!opts?.keep) {
            // the open dialog describes a join that is no longer on
            // screen, holding the clips it was opened with
            rpCloseSeamDialog(false);
        }
        updateActionBtns(); // owns btnRow visibility too
        video.style.display = "none";
        // the result arrives while the run is still finishing: the live
        // picture has served its purpose the moment there is a take
        liveClear();
        status.style.color = PAL.sub;
        status.textContent = "building preview…";

        // blocks + seam pill from the sidecars (order matches sequence)
        const clips = d.sequence.split("\n");
        const metas = await Promise.all(clips.map((c) => tlClipMeta(c)));
        if (seq !== reqSeq) return;
        rpPruneSeamOpts(clips);
        clips.forEach((c, k) => {
            if (k > 0) {
                // pill = the RELATION; under it the MEASURED verdict
                // (latent step-diff at the boundary), which beats the
                // metadata one: coordinates lining up only means the
                // join is where it claims, not that motion flows
                // through it
                const s = tlDeriveSeam(metas[k - 1], metas[k]);
                const colors = { "seamless": TL_COLORS.seamless,
                                 "soft bump": TL_COLORS.cut,
                                 "hard cut": "#b3403c" };
                const wrap = document.createElement("div");
                Object.assign(wrap.style, {
                    display: "flex", flexDirection: "column",
                    alignItems: "center", alignSelf: "center",
                    gap: "2px",
                });
                // per-GAP relation + measured verdict: a bridge take
                // has two boundaries (head seam into it = extends its
                // parent; tail seam out of it = prepends into the
                // child), each with its own measurement
                let relText, gapSeam;
                if (d.relation === "bridges") {
                    const isHead = clips[k] === d.clip;
                    relText = isHead ? "extends" : "prepends";
                    gapSeam = isHead ? d.seam : d.seam2;
                } else {
                    relText = (d.relation === "extends" ||
                               d.relation === "prepends")
                        ? d.relation : "no link";
                    gapSeam = d.seam;
                }
                // relation label wears the clip bars' chrome, smaller —
                // and when the measured seam is SEAMLESS it turns the
                // same green as the "this run" block; the verdict text
                // below still carries the exact number
                const good = gapSeam?.verdict === "seamless";
                const pill = document.createElement("span");
                Object.assign(pill.style, {
                    padding: "1px 6px", borderRadius: "4px",
                    background: good ? "#2e7d4f" : PAL.rest,
                    color: good ? "#f0f6f1" : PAL.text,
                    border: "1px solid " + (good ? "#1f5c38" : PAL.edge),
                    font: "600 9px sans-serif", whiteSpace: "nowrap",
                });
                pill.textContent = relText;
                pill.title = s.note;
                if (gapSeam) {
                    const v = document.createElement("span");
                    Object.assign(v.style, {
                        font: "9px sans-serif", whiteSpace: "nowrap",
                        color: colors[gapSeam.verdict] ?? PAL.sub,
                    });
                    // where the worst point sat: at the join, or later,
                    // once the run had left the pinned window behind
                    const at = Number(gapSeam.at ?? 0);
                    const away = Math.abs(at) >= 0.05
                        ? ` @${at > 0 ? "+" : "−"}${Math.abs(at).toFixed(1)}s`
                        : "";
                    v.textContent =
                        `${gapSeam.ratio}x ${gapSeam.verdict}${away}`;
                    v.title = (gapSeam.kind === "cut"
                        ? "a jump cut in the delivered frames"
                        : gapSeam.kind === "burst"
                            ? "the run lurches away instead of "
                              + "continuing: sustained motion far above "
                              + "the rest of the clip"
                            : "worst motion break within ~1s of the "
                              + "generated↔pinned join, as a ratio of "
                              + "the clip's median motion")
                        + (away ? ` — ${Math.abs(at).toFixed(1)}s `
                            + `${at > 0 ? "after" : "before"} the start`
                            : "")
                        + (gapSeam.boundary != null
                            ? ` (at the join itself ${gapSeam.boundary}x)`
                            : "")
                        + (gapSeam.latent != null
                            ? ` (latent metric said ${gapSeam.latent}x)`
                            : "");
                    wrap.append(pill, v);
                } else {
                    wrap.append(pill);
                }
                // The ARRIVING gap also says how the take's own frames
                // meet its pinned window -- inside the take, ~half a
                // second before this join. Always drawn when measured
                // (clean included) so the row keeps one height.
                const arriving = d.relation === "bridges"
                    ? clips[k] !== d.clip : d.relation === "prepends";
                if (arriving && d.arrival) {
                    const a = document.createElement("span");
                    const bad = d.arrival.verdict !== "clean";
                    Object.assign(a.style, {
                        font: (bad ? "600 " : "") + "9px sans-serif",
                        whiteSpace: "nowrap",
                        color: d.arrival.verdict === "local jump"
                            ? "#b3403c"
                            : bad ? TL_COLORS.cut : PAL.sub,
                    });
                    a.textContent = `arrival ${d.arrival.ratio}x`
                        + (bad ? ` ${d.arrival.verdict}` : "");
                    a.title = "Where this take's own frames give way to "
                        + "the window it arrives in (frame "
                        + d.arrival.frame + "): the worst-changing parts "
                        + "of the picture there, against the same moment "
                        + "one latent step (4 frames) earlier and later. "
                        + "About 1x is clean."
                        + (bad ? " Above 1.3x something moves or sharpens "
                            + "in a single frame -- the take arrives "
                            + "slower or sharper than the clip it lands "
                            + "in. It can pass at this size and show after "
                            + "an upscale, which keeps motion as it is. "
                            + "Another seed is the clean cure. To keep "
                            + "this take, upscale with a refine amount "
                            + "around 0.6: it re-draws through the step, "
                            + "but it re-designs detail too, so anything "
                            + "that must stay as it is needs a reference "
                            + "image." : "");
                    wrap.append(a);
                }
                // ⚙ = this join's own seam settings. It sits with the
                // seam it governs rather than in a corner of the widget,
                // because there can be two of them (a bridge take has a
                // head join and a tail join) and a single control could
                // not say which one it meant.
                const n = Object.keys(rpJoinOpts(clips, k)).length;
                const cog = document.createElement("button");
                cog.textContent = n ? `⚙ ${n}` : "⚙";
                Object.assign(cog.style, {
                    background: n ? TL_ROLE.extend.bg : "none",
                    color: n ? "#fff" : PAL.sub,
                    border: "1px solid " + (n ? "transparent" : PAL.edge),
                    borderRadius: "3px", padding: "0 4px",
                    font: "9px/13px sans-serif", cursor: "pointer",
                });
                cog.title = n
                    ? `This join has ${n} setting${n === 1 ? "" : "s"} of `
                      + "its own. The preview below is built with them, "
                      + "and they go with the take when it is added to "
                      + "the timeline."
                    : "Change how this join is repaired, and rebuild the "
                      + "preview to see it.";
                cog.addEventListener("click", (ev) => {
                    ev.stopPropagation();
                    rpOpenSeamDialog(d, k, clips, metas);
                });
                // withheld (SHOW_GUIDED) unless this join already has
                // settings of its own, or continues a clip that had no
                // sidecar (see tlPixelJoin)
                const pixel = tlPixelJoin(
                    { clip: clips[k - 1], gap: null, meta: metas[k - 1] },
                    { clip: clips[k], gap: null, meta: metas[k] });
                if (SHOW_GUIDED || n || pixel) wrap.appendChild(cog);
                strip.appendChild(wrap);
            }
            const role = clips.length === 1 ? "alone"
                : (c === d.clip ? "this run" : "lineage");
            const bEl = block(c, role, k, c === d.clip);
            blockEls.push(bEl);
            strip.appendChild(bEl);
        });

        rpProgress.arm("preparing…");
        try {
            const resp = await api.fetchApi("/obvpm/h3/preview_cut", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    sequence: rpSequenceText(d), crf: rpCrf(d.clip),
                    base_folder: d.clip.includes("/")
                        ? d.clip.slice(0, d.clip.lastIndexOf("/")) : "",
                    preview_filename: RP_PREVIEW_NAME,
                    ...rpFixOpts(d.clip),
                }),
            });
            if (!resp.ok) throw new Error(await resp.text());
            const r = await resp.json();
            if (seq !== reqSeq) return;
            starts = r.starts;
            video.src = api.apiURL(
                `/view?filename=${encodeURIComponent(r.filename)}` +
                `&subfolder=${encodeURIComponent(r.subfolder)}` +
                `&type=${encodeURIComponent(r.type ?? "output")}` +
                `&v=${encodeURIComponent(r.v ?? "")}`);
            video.style.display = "block";
            // The strip already says who relates to whom and how well, so
            // the status line normally stays quiet. It does have to
            // resolve one apparent contradiction though: the seam pill
            // reports the RAW take (measured when it was generated) while
            // the video now plays with the repair applied, so a "soft
            // bump" pill can sit above a picture that looks clean.
            // counted over the JOINS on screen, not the store, so a
            // join whose settings live in the timeline still counts
            let tuned = 0;
            for (let k = 1; k < clips.length; k++) {
                if (Object.keys(rpJoinOpts(clips, k)).length) tuned++;
            }
            status.textContent = !d.parent
                ? `${d.clip.split("/").pop()} (no lineage in folder)`
                : (rpRepairsOn(d)
                    ? (tuned ? "This join has settings of its own." : "")
                    : (tuned
                        ? "Seam repair is switched off at this join."
                        : ""));
        } catch (err) {
            if (seq !== reqSeq) return;
            status.style.color = "#de5b5b";
            status.textContent = "preview failed: " +
                String(err?.message ?? err).slice(0, 120);
        } finally {
            // a superseded request must not clear the CURRENT build's
            // bar, so only the live one puts it away
            if (seq === reqSeq) rpProgress.hide();
        }
        node.graph?.setDirtyCanvas(true);
    }

    function applyRpChrome() {
        // addTlBtn's green and delBtn's red are theme-exempt; the
        // neutral dismiss button follows the palette
        dismissBtn.style.background = PAL.rest;
        dismissBtn.style.borderColor = PAL.edge;
        dismissBtn.style.color = PAL.text;
        status.style.color = PAL.sub;
    }
    // live theme following, same pattern as the Timeline: debounced
    // whole-palette compare (the var() strings alone never change)
    let rpThemeTimer = null;
    const rpThemeMO = new MutationObserver(() => {
        clearTimeout(rpThemeTimer);
        rpThemeTimer = setTimeout(() => {
            const next = themePalette();
            if (JSON.stringify(next) === JSON.stringify(PAL)) return;
            PAL = next;
            applyRpChrome();
            if (current) void render(current);
        }, 150);
    });
    for (const t of [document.documentElement, document.body]) {
        rpThemeMO.observe(t, { attributes: true,
            attributeFilter: ["class", "style", "data-theme"] });
    }
    // palette changes that arrive as injected or replaced stylesheets
    // never touch html/body attributes -- watch the head too. The
    // debounce + computed-palette compare make unrelated firings
    // (other extensions adding styles) no-ops.
    rpThemeMO.observe(document.head,
        { childList: true, subtree: true, characterData: true });

    // a running generation shows a live elapsed clock in the status
    // line until its result (or failure) takes the line over
    let runStartedAt = null;
    let runTimer = null;
    let runPromptId = null;
    // false when we joined a run already in progress: the elapsed time is
    // unknowable then, so don't invent one
    let runElapsedKnown = true;
    function runTick() {
        if (runStartedAt === null) return;
        status.style.color = PAL.sub;
        if (!runElapsedKnown) {
            status.textContent = "generation running…";
            return;
        }
        const s = Math.floor((Date.now() - runStartedAt) / 1000);
        status.textContent = "new generation running… " +
            Math.floor(s / 60) + ":" +
            String(s % 60).padStart(2, "0") +
            (liveStep && liveStep.total
                ? " · step " + liveStep.step + "/" + liveStep.total : "");
    }
    // a live frame is shown only for the run being timed: the event is
    // global, and another workflow's sampling must not paint here
    // Per node, saved in the workflow: a preview on the refine branch
    // shares the one event stream with the generation branch's, and
    // its owner may not want the sampler's frames there at all.
    const liveWanted = () => node.properties?.h3_live_preview !== false;
    const onLivePreview = (e) => {
        const data = e?.detail;
        if (!data || runStartedAt === null || !liveWanted()) return;
        if (typeof data.step === "number" && typeof data.total === "number") {
            liveStep = { step: data.step, total: data.total };
            runTick();
        }
        if (typeof data.image === "string" && data.image) liveShow(data);
    };
    function startClock(promptId, elapsedKnown) {
        runPromptId = promptId;
        runElapsedKnown = elapsedKnown;
        runStartedAt = Date.now();
        clearInterval(runTimer);
        runTimer = setInterval(runTick, 1000);
        runTick();
    }
    // The ownership probe is ASYNC, so a run can end before it answers.
    // Cancelling straight after queueing did exactly that: the interrupt
    // arrived first and cleared the (not yet started) clock, then the
    // probe resolved and started one for a run that was already over --
    // leaving "new generation running…" ticking forever.
    let pendingPromptId = null;
    const onRunStart = (e) => {
        const id = e?.detail?.prompt_id ?? null;
        pendingPromptId = id;
        // The event is global. Ask the queue whether that job actually
        // writes where this preview is watching; only stay quiet on a
        // definite "no", so an unanswerable case behaves as before.
        void rpQueueOwns(rpScopeFolder(node), id).then((owns) => {
            if (pendingPromptId !== id) return;   // ended while we asked
            pendingPromptId = null;
            if (owns === "no") return;
            startClock(id, true);
        });
    };
    function stopClock() {
        runPromptId = null;
        pendingPromptId = null;
        runStartedAt = null;
        clearInterval(runTimer);
        runTimer = null;
        // the result's own render overwrites the line; failures and
        // interrupts just stop the clock
        if (status.textContent.startsWith("new generation running") ||
            status.textContent.startsWith("generation running")) {
            status.textContent = "";
        }
        // the live picture goes with the run; the previous result comes
        // back until (and unless) a new one renders over it
        if (liveUrl) {
            liveClear();
            if (video.getAttribute("src")) video.style.display = "block";
            if (current) strip.style.display = "flex";
            updateActionBtns();
        }
    }
    const onRunEnd = (e) => {
        const id = e?.detail?.prompt_id ?? null;
        // only the run being timed may stop its own clock -- another
        // workflow finishing says nothing about this one. A run whose
        // probe is still in flight counts as the one being timed.
        const timed = runPromptId ?? pendingPromptId;
        if (timed != null && id != null && id !== timed) return;
        stopClock();
    };
    // Backstop for every cancel path that emits no interrupt at all --
    // clearing the queue, or deleting a job that had not started. The
    // status payload carries pending + running, so zero means there is
    // nothing left that could be our run, whatever events did or did not
    // fire.
    const onQueueStatus = (e) => {
        const left = e?.detail?.exec_info?.queue_remaining;
        if (left === 0 && (runStartedAt !== null || pendingPromptId !== null)) {
            stopClock();
        }
    };
    api.addEventListener("execution_start", onRunStart);
    api.addEventListener("execution_success", onRunEnd);
    api.addEventListener("execution_error", onRunEnd);
    api.addEventListener("execution_interrupted", onRunEnd);
    api.addEventListener("status", onQueueStatus);
    api.addEventListener("kj_preview_override", onLivePreview);

    // The Result Preview plays the take with the SAME seam repair the
    // export will apply, taken from the Timeline that owns this folder.
    // Otherwise you would accept or reject a take on the strength of a
    // picture that is not the one you are going to ship.
    //
    // No Timeline in the graph -> send nothing, and the server's own
    // defaults apply. There is no export in that case either, so there
    // is nothing for it to disagree with.
    function rpFixOpts(clip) {
        const t = tlMatchesFor(clip).find((x) => x.fixOpts);
        try {
            return t ? t.fixOpts() : {};
        } catch (err) {
            return {};
        }
    }
    // The owning timeline's quality, for the same reason as its repair
    // settings: the Result Preview should build what the export will.
    // 23 when no timeline claims the clip -- the widget's own default.
    function rpCrf(clip) {
        const t = tlMatchesFor(clip).find((x) => x.crf);
        try {
            return t ? Number(t.crf()) || 23 : 23;
        } catch (err) {
            return 23;
        }
    }
    // Is ANY join in this preview being repaired? Resolved per join, the
    // same way the build resolves it: a per-seam override can switch
    // both repairs off at the only join there is, and the status line
    // must not then claim the picture has been repaired.
    function rpRepairsOn(d) {
        const clips = String(d?.sequence ?? "").split("\n");
        for (let k = 1; k < clips.length; k++) {
            const own = rpJoinOpts(clips, k);
            const on = (key) => own[key] !== undefined
                ? own[key] : rpInherited(d.clip, key);
            if (on("level_lock") || on("crossfade")) return true;
        }
        return false;
    }

    function tlMatchesFor(clip) {
        const folderOf = clip.includes("/")
            ? clip.slice(0, clip.lastIndexOf("/")) : "";
        return [...TL_REGISTRY.values()].filter((t) => {
            const f = t.folder();
            return !f || f === folderOf;
        });
    }
    // The decision taken on a take (added, or dismissed) is kept BY CLIP
    // in the node's properties, so it survives a reload, a workflow
    // switch and the restore of the same result: the same take never
    // asks twice. A new take is a new decision; a deleted one is gone.
    const decided = () => !!current &&
        node.properties?.h3_decided === current.clip;
    function decide() {
        node.properties = node.properties || {};
        node.properties.h3_decided = current?.clip ?? null;
    }
    function updateActionBtns() {
        if (decided()) {
            btnRow.style.display = "none";
            return;
        }
        // a take that already sits in a timeline is neither addable
        // nor safe to delete from here; a rendered cut is the final
        // product, not a take for a timeline
        const inTl = current && [...TL_REGISTRY.values()]
            .some((t) => t.has?.(current.clip));
        const canAdd = !!(current && !inTl && current.render !== "joint" &&
            tlMatchesFor(current.clip).length);
        const canDel = !!(current && !inTl);
        addTlBtn.style.display = canAdd ? "" : "none";
        delBtn.style.display = canDel ? "" : "none";
        // with nothing to accept or reject the row would hold only
        // dismiss -- nothing worth dismissing, so hide the whole row
        btnRow.style.display = canAdd || canDel ? "flex" : "none";
    }
    // timelines register/unregister (and their sequences change) as
    // the user works; re-check on hover so the row is honest at the
    // moment it is looked at. On the CONTAINER, not the row: a hidden
    // row gets no mouse events, and it must be able to come back when
    // its take leaves a timeline again
    container.addEventListener("mouseenter", updateActionBtns);
    addTlBtn.addEventListener("click", async () => {
        if (!current) return;
        const match = tlMatchesFor(current.clip);
        if (!match.length) {
            updateActionBtns();
            return;
        }
        // the settings this take was judged under travel with it, so
        // the timeline builds the join that was approved here
        const own = rpAllSeamOpts();
        for (const t of match) {
            status.textContent = await t.offer(current.clip, own);
        }
        // the timeline's own refresh is async, so has() still answers
        // with the OLD sequence here -- but the offer just succeeded,
        // so the take is in a timeline now: hide directly, and remember
        // the decision so the row stays down for this take
        decide();
        addTlBtn.style.display = "none";
        delBtn.style.display = "none";
        btnRow.style.display = "none"; // dismiss alone is no row
    });
    dismissBtn.addEventListener("click", () => {
        // just tuck the buttons away; the preview itself stays. Sticky
        // for this take (hover re-checks and reloads must not resurrect
        // an explicitly dismissed row)
        decide();
        btnRow.style.display = "none";
    });
    delBtn.addEventListener("click", async () => {
        if (!current) return;
        if (!confirm(`Delete ${current.clip}? Its sidecar, recorded \nreferences and conditioning cache go with it.`)) return;
        try {
            const resp = await api.fetchApi("/obvpm/h3/delete_take", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: current.clip }),
            });
            if (!resp.ok) throw new Error(await resp.text());
            const gone = current.clip;
            // module caches would otherwise serve a ghost if the save
            // counter reuses the name
            tlMetaCache.delete(gone);
            tlFramesCache.delete(gone);
            for (const t of TL_REGISTRY.values()) t.forget(gone);
            current = null;
            node.properties.h3_result = null;
            video.pause();
            video.removeAttribute("src");
            video.load();
            video.style.display = "none";
            strip.replaceChildren();
            strip.style.display = "none";
            btnRow.style.display = "none";
            status.style.color = PAL.sub;
            status.textContent = "deleted " + gone;
        } catch (err) {
            status.style.color = "#de5b5b";
            status.textContent = "delete failed: " +
                String(err?.message ?? err).slice(0, 120);
        }
    });

    let lastShownKey = null;
    let lastShownAt = 0;
    node._h3ShowResult = (d) => {
        // Both delivery paths can land the same payload in the same tick;
        // re-rendering would restart the preview build and the video for
        // nothing. Only collapse near-simultaneous repeats: an identical
        // payload arriving later is a genuine re-run, since the save
        // counter reuses a filename once that take has been deleted.
        const key = JSON.stringify(d);
        const now = Date.now();
        if (key === lastShownKey && now - lastShownAt < 2000) return;
        lastShownKey = key;
        lastShownAt = now;
        node.properties = node.properties || {};
        node.properties.h3_result = d;
        void render(d);
        void backfillArrival(d);
    };

    // A payload saved before the `arrival` reading existed has no such
    // key at all (a measured "nothing to report" is null), and restoring
    // it runs nothing on the server. Ask once, keep the answer in the
    // payload, draw again.
    async function backfillArrival(d) {
        if (!d?.clip || d.arrival !== undefined
                || (d.relation !== "prepends" && d.relation !== "bridges")) {
            return;
        }
        try {
            const resp = await api.fetchApi("/obvpm/h3/arrival", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ clip: d.clip }),
            });
            if (!resp.ok) return;       // older server: leave it unasked
            const got = await resp.json();
            if (node.properties?.h3_result !== d) return;   // moved on
            d.arrival = got.arrival ?? null;
            lastShownKey = JSON.stringify(d);
            void render(d);
        } catch (err) {
            tldbg("result preview: arrival backfill failed", err);
        }
    }

    // Pick up a run that finished while this workflow was in the
    // background: its `executed` event never reached this node (the node
    // wasn't in the open graph) and ComfyUI's own output store was wiped
    // by the workflow switch, so RP_RESULTS is the only surviving copy.
    // Restoring a payload from an earlier session means its clips may be
    // gone -- deleting a rejected take is routine, and the preview build
    // would fail with a "clip not found" error for something the user did
    // on purpose. Check first and say it plainly instead.
    async function restoreResult(d) {
        if (!d?.clip) return;
        const clips = String(d.sequence || d.clip)
            .split("\n").map((c) => c.trim()).filter(Boolean);
        const sizes = await Promise.all(
            clips.map((c) => serverFileSize(c).catch(() => null)));
        const missing = clips.filter((c, i) => sizes[i] === null);
        if (!missing.length) {
            node._h3ShowResult(d);
            return;
        }
        node.properties = node.properties || {};
        if (missing.includes(d.clip)) {
            // the take itself is gone: nothing left to show, and keeping
            // the payload would resurrect this every time it loads
            node.properties.h3_result = null;
            status.style.color = PAL.sub;
            status.textContent =
                d.clip.split("/").pop() + " was deleted";
            tldbg("result preview: take", d.clip, "no longer exists");
            return;
        }
        // only a lineage neighbour went away -- still show the take alone
        tldbg("result preview: lineage clip(s) gone:", missing.join(", "));
        node._h3ShowResult({
            ...d, parent: null, parent2: null, relation: "",
            sequence: d.clip, seam: null, seam2: null,
        });
    }

    function adoptCachedResult() {
        if (rpScopeFolder(node) === null) return; // can't tell whose result
        // newest first, matched by the same rule that gates live events
        for (let i = RP_RESULTS.length - 1; i >= 0; i--) {
            if (rpAccepts(node, RP_RESULTS[i])) {
                void restoreResult(RP_RESULTS[i]);
                return;
            }
        }
    }
    const prevMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
        const r = prevMenu?.apply(this, arguments);
        options.push({
            content: (liveWanted() ? "✓ " : "") + "Live preview while sampling",
            callback: () => {
                this.properties ??= {};
                this.properties.h3_live_preview = !liveWanted();
                if (!liveWanted()) {
                    liveClear();
                    if (video.getAttribute("src")) video.style.display = "block";
                    if (current) strip.style.display = "flex";
                    updateActionBtns();
                }
                this.setDirtyCanvas?.(true, true);
            },
        });
        return r;
    };

    const live = {
        alive: () => !!node.graph,
        describe: () => `#${node.id} watches ${rpScopeFolder(node) ?? "(any)"}`,
        accepts: (d) => rpAccepts(node, d),
        verdict: (d) => rpVerdict(node, d),
        show: (d) => node._h3ShowResult(d),
        node,
    };
    RP_LIVE.add(live);

    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        // Deferred as a whole: links aren't wired up yet during configure,
        // so the scope can't be read until the graph settles.
        setTimeout(() => {
            const saved = this.properties?.h3_result;
            if (saved && !rpAccepts(this, saved)) {
                // The payload is saved INTO the workflow file, so a result
                // captured from another workflow before scope checking
                // existed comes back on every load. Drop it for good.
                tldbg("result preview: dropping saved result", saved.clip,
                      "— this node watches", rpScopeFolder(this));
                this.properties.h3_result = null;
            } else if (saved) {
                void restoreResult(saved);
            }
            adoptCachedResult();
            // Joining a run that is already in flight: the start event
            // fired before this node existed, so ask the queue instead.
            void rpQueueOwns(rpScopeFolder(this), null).then((owns) => {
                if (owns === "yes" && runStartedAt === null) {
                    startClock(null, false);
                }
            });
        }, 0);
        return r;
    };
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        RP_LIVE.delete(live);
        rpThemeMO.disconnect();
        clearInterval(runTimer);
        api.removeEventListener("execution_start", onRunStart);
        api.removeEventListener("execution_success", onRunEnd);
        api.removeEventListener("execution_error", onRunEnd);
        api.removeEventListener("execution_interrupted", onRunEnd);
        api.removeEventListener("status", onQueueStatus);
        api.removeEventListener("kj_preview_override", onLivePreview);
        liveClear();
        dropRpProgress();
        video.pause();
        video.removeAttribute("src");
        video.load();
        return onRemoved?.apply(this, arguments);
    };
}
