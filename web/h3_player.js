// Our own video controls, BELOW the picture instead of over it: the
// browser's controls cover the bottom of the frame with a dark shade --
// exactly where a join or a subtitle-safe detail is being judged.
//
// One bar can drive one clip or a GROUP played as one: several clips
// side by side, seeked and paused together. The group's clock
// is its longest member; a shorter member holds its last frame until the
// clock comes round again, and a `stretch` member (a sampler's live
// preview, a handful of frames for the whole clip) is mapped onto the
// clock proportionally instead.
//
// Members are anything with the HTMLMediaElement surface the bar uses
// (currentTime, duration, paused, play(), pause(), muted, volume, plus
// "loadedmetadata" / "play" / "pause" events): a <video>, or a LiveClip.
//
// Sound and loop are per viewer conveniences, kept in this browser
// (localStorage; a private window simply starts from the defaults).

const KEY = "obvpm.h3.player";
const FPS_DEFAULT = 24;
// Keeping members on the clock. A seek is costly here -- our files have
// few keyframes, so it decodes everything from the last one -- and under
// load a seek can make the member fall behind again: re-seeked, stalled,
// re-seeked, a picture that holds a frame and jumps. So a member a little
// off is brought back by playing it a little faster or slower, and only
// one far off is seeked.
const DRIFT = 0.35;            // seconds off before a member is re-seeked
const NUDGE = 0.02;            // ...and before its speed is nudged
const NUDGE_GAIN = 2;          // speed change per second off
const NUDGE_MAX = 0.2;         // at most 0.8x .. 1.2x

function prefs() {
    try {
        const v = JSON.parse(localStorage.getItem(KEY) || "{}");
        return v && typeof v === "object" ? v : {};
    } catch (err) {
        return {};
    }
}
function savePrefs(patch) {
    try {
        localStorage.setItem(KEY, JSON.stringify({ ...prefs(), ...patch }));
    } catch (err) { /* storage blocked: the setting lasts this session */ }
}

// ---- icons: 16x16 SVG paths, drawn in the bar's ink --------------------
const ICONS = {
    play: "M4 2.5v11l9-5.5z",
    pause: "M4 2.5h3v11H4zM9 2.5h3v11H9z",
    back: "M3 3h2v10H3zM13 3v10L6 8z",
    fwd: "M11 3h2v10h-2zM3 3v10l7-5z",
    sound: "M2 6h3l4-3.5v11L5 10H2zM11 5.2a4 4 0 0 1 0 5.6l-.9-.9a2.7 2.7 0 0 0 0-3.8zM12.8 3.4a6.5 6.5 0 0 1 0 9.2l-.9-.9a5.2 5.2 0 0 0 0-7.4z",
    mute: "M2 6h3l4-3.5v11L5 10H2zM10.6 5.6l1.4 1.4 1.4-1.4.9.9L12.9 8l1.4 1.4-.9.9L12 8.9l-1.4 1.4-.9-.9L11.1 8 9.7 6.5z",
    loop: "M4 5h7V3l3 3-3 3V7H5v2H3V6a1 1 0 0 1 1-1zM12 11H5v2l-3-3 3-3v2h6V7h2v3a1 1 0 0 1-1 1z",
    full: "M2 2h4v1.5H3.5V6H2zM10 2h4v4h-1.5V3.5H10zM2 10h1.5v2.5H6V14H2zM12.5 10H14v4h-4v-1.5h2.5z",
};
function icon(name, size = 14) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.style.display = "block";
    svg.style.pointerEvents = "none";
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", ICONS[name]);
    path.setAttribute("fill", "currentColor");
    svg.appendChild(path);
    return svg;
}

function fmt(t) {
    if (!Number.isFinite(t) || t < 0) t = 0;
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

const dur = (m) => (Number.isFinite(m?.duration) ? m.duration : 0);

/**
 * The bar. opts:
 *   pal()        -> {rest, text, sub, edge, active}: the theme, read on paint
 *   fps          frame step size (default 24)
 *   fullscreen() -> the element to make fullscreen (omit: no button)
 *   seek         false: no seek bar (a player with a scrubber of its own)
 *   loopKey      remember loop under this name (omit: not remembered)
 *   loop         initial loop when nothing is remembered (default true)
 *   hover        true: sound plays only from the member hear() names (a
 *                grid: hover a take to hear it); the bar's mute and
 *                volume still decide whether and how loud
 *   dock         true: in fullscreen the bar stays below the picture
 *                (a view with more under the picture than the bar)
 *   host         a player with a transport of its own (the Timeline: a
 *                playlist of cuts, empty space, a built file). The bar
 *                then shows the ONE element it is given -- its whole
 *                file on the seek bar -- and hands play / pause and
 *                frame steps to the host; it never seeks, starts or
 *                loops the element by itself, and has no loop button.
 *                { playing() -> bool, toggle(), step(frames),
 *                  enabled() -> bool, onSeek() (the bar moved the
 *                  element: a scrub on the seek bar) }
 * A member may be { m, offset, stretch, primary }: `primary` is the one
 * watched (a large picture over a row of small ones) -- the clock is read
 * from it when it covers the whole span, so a small one that stutters
 * does not drag the watched one along with it.
 * Returns { el, track, total, setMembers(list), hear(m), setLoop(on),
 *           loop, play(), pause(), toggle(), seek(t), setRange(a, b),
 *           refresh(), attach(surface, {click}), destroy() }.
 * track: the seek bar's element, for lining things up with it; total:
 * the clock's length in seconds.
 * attach(): the picture the bar plays -- a click on it plays / pauses
 * and a double click is fullscreen (click: false leaves clicks to the
 * surface, e.g. a grid whose tiles open a take), and a sideways scroll
 * over it jogs, as over the bar itself. In fullscreen the bar lies over
 * the bottom of the picture and slides away while the mouse rests
 * (unless `dock`).
 */
export function tlPlayerBar(opts = {}) {
    const pal = () => (opts.pal ? opts.pal() : {
        rest: "#2a2d33", text: "#ddd", sub: "#999", edge: "#444",
        active: "#3557b0" });
    const fps = opts.fps || FPS_DEFAULT;
    const host = opts.host ?? null;
    let members = [];           // [{m, stretch}]
    let heard = null;
    let destroyed = false;
    let raf = 0;
    let scrubbing = false;
    let range = null;
    // the clock's last good reading: a member whose source is swapped
    // (the same <video>, another file) restarts at 0 and is put back here
    // -- and when it is the member the clock is read from, the clock
    // itself would otherwise jump to 0 with it
    let clockT = 0;

    const p0 = prefs();
    let volume = Number.isFinite(p0.volume) ? p0.volume : 1;
    let muted = !!p0.muted;
    let loop = opts.loopKey && typeof p0.loops?.[opts.loopKey] === "boolean"
        ? p0.loops[opts.loopKey] : (opts.loop ?? true);

    const el = document.createElement("div");
    Object.assign(el.style, {
        display: "flex", alignItems: "center", gap: "4px",
        height: "24px", flexShrink: "0", padding: "0 2px",
        userSelect: "none", font: "10px/14px sans-serif",
    });
    const btn = (name, title) => {
        const b = document.createElement("button");
        b.title = title;
        b.appendChild(icon(name));
        Object.assign(b.style, {
            width: "22px", height: "20px", padding: "0", flexShrink: "0",
            display: "flex", alignItems: "center", justifyContent: "center",
            borderRadius: "4px", cursor: "pointer", border: "1px solid",
        });
        b.addEventListener("pointerdown", (ev) => ev.stopPropagation());
        return b;
    };
    const setIcon = (b, name) => b.replaceChildren(icon(name));
    const playBtn = btn("play", "Play / pause (Space)");
    const backBtn = btn("back", "One frame back");
    const fwdBtn = btn("fwd", "One frame forward");

    // the seek bar: the whole clip, what has played filled, an optional
    // highlighted range (the part a cut uses) under it
    const track = document.createElement("div");
    Object.assign(track.style, {
        position: "relative", flex: "1", height: "14px", cursor: "pointer",
        minWidth: "40px",
    });
    const rail = document.createElement("div");
    const rangeEl = document.createElement("div");
    const fill = document.createElement("div");
    const knob = document.createElement("div");
    Object.assign(rail.style, { position: "absolute", left: "0", right: "0",
        top: "5px", height: "4px", borderRadius: "2px" });
    // under the rail rather than on it: the played fill would cover it
    Object.assign(rangeEl.style, { position: "absolute", top: "10px",
        height: "3px", borderRadius: "1.5px", display: "none",
        background: "rgba(232,176,64,0.95)" });
    Object.assign(fill.style, { position: "absolute", left: "0", top: "5px",
        height: "4px", borderRadius: "2px" });
    Object.assign(knob.style, { position: "absolute", top: "2px",
        width: "10px", height: "10px", borderRadius: "5px",
        marginLeft: "-5px", boxShadow: "0 0 2px rgba(0,0,0,0.6)" });
    track.append(rail, rangeEl, fill, knob);
    const time = document.createElement("div");
    Object.assign(time.style, { flexShrink: "0", whiteSpace: "nowrap",
        fontVariantNumeric: "tabular-nums", minWidth: "74px",
        textAlign: "center" });

    // sound: the button mutes; the volume is a VERTICAL slider that pops
    // up over the picture above it while the pointer is on either (and
    // the wheel over it turns the volume)
    const soundBtn = btn("sound", "Mute / unmute");
    const soundBox = document.createElement("div");
    Object.assign(soundBox.style, { position: "relative", flexShrink: "0" });
    const volPop = document.createElement("div");
    Object.assign(volPop.style, {
        position: "absolute", bottom: "100%", left: "50%",
        transform: "translateX(-50%)", width: "26px", height: "104px",
        paddingBottom: "4px", display: "none", zIndex: "5",
    });
    const volPanel = document.createElement("div");
    Object.assign(volPanel.style, {
        width: "100%", height: "100%", borderRadius: "6px",
        boxSizing: "border-box", border: "1px solid",
        boxShadow: "0 4px 14px rgba(0,0,0,0.45)", position: "relative",
        cursor: "pointer",
    });
    const volRail = document.createElement("div");
    const volFill = document.createElement("div");
    const volKnob = document.createElement("div");
    Object.assign(volRail.style, { position: "absolute", left: "11px",
        width: "4px", top: "10px", bottom: "10px", borderRadius: "2px" });
    Object.assign(volFill.style, { position: "absolute", left: "11px",
        width: "4px", bottom: "10px", borderRadius: "2px",
        background: "#4a8fd4" });
    Object.assign(volKnob.style, { position: "absolute", left: "8px",
        width: "10px", height: "10px", borderRadius: "5px",
        marginBottom: "-5px", background: "#e8eef8",
        boxShadow: "0 0 2px rgba(0,0,0,0.6)" });
    volPanel.append(volRail, volFill, volKnob);
    volPanel.title = "Volume";
    volPop.appendChild(volPanel);
    soundBox.append(volPop, soundBtn);
    let volDrag = false;
    let volHide = 0;
    const showVol = (on) => {
        clearTimeout(volHide);
        if (on) volPop.style.display = "block";
        else volHide = setTimeout(() => {
            if (!volDrag) volPop.style.display = "none";
        }, 250);
    };
    soundBox.addEventListener("mouseenter", () => showVol(true));
    soundBox.addEventListener("mouseleave", () => showVol(false));
    function paintVol() {
        const v = muted ? 0 : volume;
        const h = 104 - 4 - 20;              // the rail's length
        volFill.style.height = (v * h) + "px";
        volKnob.style.bottom = (10 + v * h) + "px";
    }
    function setVolume(v) {
        volume = Math.max(0, Math.min(1, v));
        muted = volume === 0;
        savePrefs({ muted, volume });
        applySound();
    }
    const volAt = (ev) => {
        const r = volRail.getBoundingClientRect();
        setVolume(1 - (ev.clientY - r.top) / (r.height || 1));
    };
    volPanel.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        ev.preventDefault();
        ev.stopPropagation();
        volDrag = true;
        volPanel.setPointerCapture?.(ev.pointerId);
        volAt(ev);
    });
    volPanel.addEventListener("pointermove", (ev) => {
        if (volDrag) volAt(ev);
    });
    const volEnd = () => {
        volDrag = false;
        if (!soundBox.matches(":hover")) showVol(false);
    };
    volPanel.addEventListener("pointerup", volEnd);
    volPanel.addEventListener("pointercancel", volEnd);
    const loopBtn = btn("loop", "Loop");
    const fullBtn = opts.fullscreen ? btn("full", "Fullscreen") : null;

    el.append(playBtn, backBtn, fwdBtn);
    if (opts.seek !== false) el.append(track);
    else time.style.marginRight = "auto";
    el.append(time, soundBox);
    if (!host) el.append(loopBtn);
    if (fullBtn) el.append(fullBtn);
    for (const ev of ["pointerdown", "wheel", "keydown"]) {
        // the canvas must not see presses on the bar (the wheel still
        // reaches it from anywhere else on the panel)
        if (ev !== "wheel") el.addEventListener(ev, (e) => e.stopPropagation());
    }

    // ---- the group clock --------------------------------------------------
    // A member may sit LATER on the clock (`offset`, seconds): a finished
    // take inside the latent timeline of the live preview, after its
    // pinned head. Before its offset it holds its first frame, after its
    // end its last. The clock is the member reaching furthest (offset 0
    // preferred on a tie), so it covers the whole span.
    const off = (x) => x.offset || 0;
    const extent = (x) => off(x) + dur(x.m);
    // A <video> loading another file reads 0 and has no length yet: as the
    // clock it would say "at the start" -- or "at the end", and loop the
    // whole group back to its start. It runs the clock again once loaded.
    // Loaded is not enough: until the bar has PUT it at the clock's moment
    // (its seek done) it still reads its file's start.
    const placing = new WeakSet();
    const ready = (x) => !(x.m instanceof HTMLMediaElement)
        || (!placing.has(x.m) && x.m.readyState >= 1
            && Number.isFinite(x.m.duration));
    const master = () => {
        let best = null;
        const any = members.some((x) => !x.stretch && ready(x));
        for (const x of members) {
            if (x.stretch || (any && !ready(x))) continue;
            const tie = best && Math.abs(extent(x) - extent(best)) <= 1e-3;
            if (!best || extent(x) > extent(best) + 1e-3
                    || (tie && !!x.primary !== !!best.primary && x.primary)
                    || (tie && !!x.primary === !!best.primary
                        && off(x) < off(best))) {
                best = x;
            }
        }
        return best ?? members[0] ?? null;
    };
    // a member back at its own speed (it was nudged onto the clock)
    const unNudge = (m) => {
        if (m instanceof HTMLMediaElement && m.playbackRate !== 1) {
            m.playbackRate = 1;
        }
    };
    let seeks = 0;                // re-seeks made to keep members on the clock
    const total = () => {
        const x = master();
        return x ? extent(x) : 0;
    };
    const now = () => {
        const x = master();
        if (!x) return 0;
        // nothing loaded to read: where the clock last was
        return ready(x) ? x.m.currentTime + off(x) : clockT;
    };
    const playing = () => {
        if (host) return !!host.playing();
        const x = master();
        return !!x && !x.m.paused;
    };
    // where member x should be when the clock reads t
    function targetOf(x, t) {
        const d = dur(x.m);
        if (!d) return 0;
        if (x.stretch) {
            const T = total();
            return T ? Math.min(d, (t / T) * d) % d : 0;
        }
        return Math.max(0, Math.min(t - off(x), Math.max(0, d - 0.04)));
    }
    function applySound() {
        for (const { m } of members) {
            m.volume = volume;
            m.muted = muted || (opts.hover ? m !== heard : false);
        }
        setIcon(soundBtn, muted || volume === 0 ? "mute" : "sound");
        paintVol();
    }
    function seekAll(t) {
        clockT = t;
        for (const x of members) {
            unNudge(x.m);
            try {
                x.m.currentTime = targetOf(x, t);
            } catch (err) { /* not seekable yet */ }
        }
        host?.onSeek?.();
    }
    // Whether the group is MEANT to play: set by play, cleared by pause
    // (and a frame step, a jog). A member that becomes ready while it is
    // set starts by itself -- a video whose src was just swapped (a
    // joined preview arriving) lost its play request in the load and sat
    // frozen, holding the whole clock at 0.
    let wanted = false;
    function play() {
        wanted = true;
        if (!members.length) return;
        // at the end with no loop: start over, as a player does
        if (total() && now() >= total() - 0.05) seekAll(0);
        const t = now();
        for (const x of members) {
            const d = dur(x.m);
            if (!x.stretch && d && t - off(x) > d) continue;  // holds its end
            if (!x.stretch && t < off(x)) continue;        // not yet on
            startMember(x.m);
        }
        tick();
    }
    // A browser refuses to START sound nobody asked for (no click on the
    // page yet): then it plays muted rather than not at all.
    function startMember(m) {
        void m.play?.()?.catch?.((err) => {
            if (err?.name !== "NotAllowedError" || m.muted) return;
            m.muted = true;
            void m.play?.()?.catch?.(() => {});
        });
    }
    // Paused, every member shows the clock's frame -- exactly: while
    // playing they may be a frame or so apart, and stopped like that the
    // pictures side by side (and a take switched to) would not be the
    // same moment. Aimed at the middle of the frame, so rounding never
    // lands on its neighbour.
    function pause() {
        wanted = false;
        for (const { m } of members) m.pause?.();
        const T = total();
        if (members.length && T) {
            const f = Math.floor(Math.min(now(), T) * fps + 1e-6);
            seekAll(Math.min(T, (f + 0.5) / fps));
        }
        paint();
    }
    // start what should be playing now and is not, without seeking
    function resume() {
        const t = now();
        for (const x of members) {
            if (!x.m.paused) continue;
            const d = dur(x.m);
            if (!d) continue;
            if (!x.stretch && (t < off(x) || t - off(x) >= d)) continue;
            startMember(x.m);
        }
    }
    function toggle() {
        if (host) {
            host.toggle();
            paint();
            tick();
            return;
        }
        if (playing()) pause();
        else play();
    }
    function step(frames) {
        if (host) {
            host.step(frames);
            paint();
            return;
        }
        pause();
        const t = Math.max(0, Math.min(total(), now() + frames / fps));
        seekAll(t);
        paint();
    }
    // keep the members on the clock while it runs; the clock's end loops
    // (or stops) everything together -- members never loop on their own
    function tick() {
        cancelAnimationFrame(raf);
        if (destroyed) return;
        if (host) {
            // the host keeps its own clock: the face only follows it
            paint();
            if (playing()) raf = requestAnimationFrame(tick);
            return;
        }
        const x0 = master();
        if (x0) {
            const t = now();
            const T = total();
            if (T && (x0.m.ended || t >= T - 0.02) && !scrubbing) {
                if (loop) {
                    seekAll(0);
                    for (const x of members) {
                        void x.m.play?.()?.catch?.(() => {});
                    }
                } else {
                    pause();
                }
            } else if (!x0.m.paused && !scrubbing) {
                for (const x of members) {
                    if (x === x0) continue;
                    // a stretch member (a sampler's live preview) runs
                    // free at its own speed and loops by itself: pulled
                    // onto the clock it was re-seeked many times a second,
                    // which is what made it stutter
                    if (x.stretch) {
                        if (x.m.paused) void x.m.play?.()?.catch?.(() => {});
                        continue;
                    }
                    const d = dur(x.m);
                    if (!d) continue;
                    const local = t - off(x);
                    if (local < 0) {
                        // not on yet: its first frame, waiting
                        if (!x.m.paused) x.m.pause();
                        if (x.m.currentTime > 0.05) {
                            try { x.m.currentTime = 0; } catch (err) { /* */ }
                        }
                        continue;
                    }
                    if (local >= d) {
                        if (!x.m.paused) x.m.pause();
                        continue;
                    }
                    const want = targetOf(x, t);
                    const off_ = want - x.m.currentTime;
                    if (Math.abs(off_) > DRIFT) {
                        unNudge(x.m);
                        seeks++;
                        try { x.m.currentTime = want; } catch (err) { /* */ }
                    } else if (x.m instanceof HTMLMediaElement) {
                        const rate = Math.abs(off_) < NUDGE ? 1
                            : 1 + Math.max(-NUDGE_MAX,
                                           Math.min(NUDGE_MAX, off_ * NUDGE_GAIN));
                        if (Math.abs(x.m.playbackRate - rate) > 0.01) {
                            x.m.playbackRate = rate;
                        }
                    }
                    if (x.m.paused) void x.m.play?.()?.catch?.(() => {});
                }
            }
        }
        paint();
        if (playing()) raf = requestAnimationFrame(tick);
    }

    // something to play: a member -- or, under a host, whatever it says
    // (empty space on its timeline plays with no element at all)
    const enabled = () => (host ? (host.enabled?.() ?? true)
                                : members.length > 0);
    function paint() {
        // over a fullscreen picture: light on dark, whatever the theme
        const P = overlay ? { rest: "rgba(255,255,255,0.12)", text: "#f2f2f2",
                              sub: "#d8d8d8", edge: "rgba(255,255,255,0.18)",
                              active: "#3557b0", activeText: "#fff" } : pal();
        el.style.color = P.text;
        volPanel.style.background = overlay ? "rgba(20,20,20,0.9)"
            : (pal().restRgb ?? P.rest);
        volPanel.style.borderColor = P.edge;
        volRail.style.background = P.edge;
        paintVol();
        for (const b of [playBtn, backBtn, fwdBtn, soundBtn, loopBtn, fullBtn]) {
            if (!b) continue;
            b.style.background = P.rest;
            b.style.borderColor = P.edge;
            b.style.color = P.text;
        }
        loopBtn.style.background = loop ? P.active : P.rest;
        loopBtn.style.color = loop ? (P.activeText ?? "#fff") : P.text;
        loopBtn.title = loop ? "Loop: on" : "Loop: off";
        setIcon(playBtn, playing() ? "pause" : "play");
        rail.style.background = P.edge;
        fill.style.background = "#4a8fd4";
        knob.style.background = "#e8eef8";
        time.style.color = P.sub;
        const T = total();
        const t = Math.min(now(), T || 0);
        const x0 = master();
        if (x0 && (x0.m.readyState ?? 1) >= 1) clockT = t;
        const f = T ? t / T : 0;
        fill.style.width = (f * 100) + "%";
        knob.style.left = (f * 100) + "%";
        if (range && T) {
            rangeEl.style.display = "";
            rangeEl.style.left = (Math.max(0, range[0] / T) * 100) + "%";
            rangeEl.style.width = (Math.max(0, (Math.min(T, range[1]) - range[0]) / T) * 100) + "%";
        } else {
            rangeEl.style.display = "none";
        }
        time.textContent = `${fmt(t)} / ${fmt(T)}`;
        const any = enabled();
        for (const b of [playBtn, backBtn, fwdBtn]) b.disabled = !any;
        el.style.opacity = any ? "1" : "0.5";
    }

    // ---- input ------------------------------------------------------------
    playBtn.addEventListener("click", toggle);
    backBtn.addEventListener("click", () => step(-1));
    fwdBtn.addEventListener("click", () => step(1));
    soundBtn.addEventListener("click", () => {
        muted = !muted;
        if (!muted && volume === 0) volume = 0.5;
        savePrefs({ muted, volume });
        applySound();
    });
    loopBtn.addEventListener("click", () => setLoop(!loop, true));
    function toggleFullscreen() {
        const target = opts.fullscreen?.();
        if (!target) return;
        if (document.fullscreenElement) void document.exitFullscreen?.();
        else void target.requestFullscreen?.()?.catch?.(() => {});
    }
    fullBtn?.addEventListener("click", toggleFullscreen);

    // ---- the picture: click plays / pauses, double click is fullscreen,
    // a sideways scroll over it (or over the bar) jogs --------------------
    // A double click is two clicks first, so it toggles play twice --
    // back where it was -- and then goes fullscreen, as players do.
    const surfaces = new Set();
    function attach(surface, { click = true } = {}) {
        if (!surface || surfaces.has(surface)) return;
        surfaces.add(surface);
        if (!click) return;
        surface.addEventListener("click", (ev) => {
            if (ev.button !== 0 || ev.target.closest?.("button, input, a")) return;
            ev.stopPropagation();
            toggle();
        });
        surface.addEventListener("dblclick", (ev) => {
            if (ev.target.closest?.("button, input, a")) return;
            ev.preventDefault();
            ev.stopPropagation();
            toggleFullscreen();
        });
    }
    // Listened for on WINDOW, capture phase (as the Timeline's jog): in
    // Nodes 2.0 the node hands a wheel over it to the canvas before any
    // listener on the panel sees it. Only a mostly SIDEWAYS scroll is
    // taken (shift+wheel is the browser's sideways scroll); up/down still
    // zooms the graph.
    let jogAcc = 0;
    const JOG_GAIN = 0.125;          // frames per pixel scrolled
    const onWheel = (ev) => {
        if (destroyed) return;
        const t = ev.target;
        if (!(t instanceof Node)) return;
        // the wheel over the volume slider turns the volume
        if (volPop.contains(t)) {
            ev.preventDefault();
            ev.stopImmediatePropagation();
            const d = ev.deltaY || -ev.deltaX;
            if (d) setVolume((muted ? 0 : volume) - Math.sign(d) * 0.05);
            return;
        }
        if (!enabled()) return;
        const fs = document.fullscreenElement;
        const over = el.contains(t) || [...surfaces].some((s) => s.contains(t))
            || (fs && fs.contains(el) && fs.contains(t));
        if (!over) return;
        const dx = ev.shiftKey && !ev.deltaX ? ev.deltaY : ev.deltaX;
        const dy = ev.shiftKey && !ev.deltaX ? 0 : ev.deltaY;
        if (!dx || Math.abs(dx) <= Math.abs(dy)) return;
        ev.preventDefault();
        ev.stopImmediatePropagation();
        const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1;
        jogAcc += dx * unit * JOG_GAIN;
        const frames = Math.trunc(jogAcc);
        if (!frames) return;
        jogAcc -= frames;
        step(frames);
    };
    window.addEventListener("wheel", onWheel, { passive: false, capture: true });

    // ---- fullscreen: the bar over the bottom of the picture, sliding
    // away when the mouse rests ------------------------------------------
    const IDLE_MS = 2500;
    let overlay = false;
    let idleTimer = 0;
    let restStyle = null;           // the bar's own style, back on exit
    function wake() {
        if (!overlay) return;
        el.style.transform = "translateY(0)";
        const fs = document.fullscreenElement;
        if (fs) fs.style.cursor = "";
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            if (!overlay || el.matches(":hover") || volDrag || scrubbing) {
                wake();
                return;
            }
            el.style.transform = "translateY(110%)";
            const f = document.fullscreenElement;
            if (f) f.style.cursor = "none";
        }, IDLE_MS);
    }
    const onFsMove = () => wake();
    function onFullscreen() {
        const fs = document.fullscreenElement;
        const on = !!fs && fs.contains(el) && !opts.dock;
        if (on === overlay) return;
        overlay = on;
        if (on) {
            restStyle = {};
            for (const k of ["position", "left", "right", "bottom", "zIndex",
                             "padding", "height", "background", "transform",
                             "transition"]) {
                restStyle[k] = el.style[k];
            }
            Object.assign(el.style, {
                position: "absolute", left: "0", right: "0", bottom: "0",
                zIndex: "10", padding: "8px 14px 10px", height: "auto",
                background: "linear-gradient(transparent, rgba(0,0,0,0.75))",
                transition: "transform 0.25s ease",
            });
            if (getComputedStyle(fs).position === "static") {
                fs.style.position = "relative";
                fs.dataset.tlPlayerPos = "1";
            }
            fs.addEventListener("mousemove", onFsMove);
            fs.addEventListener("pointerdown", onFsMove);
            wake();
        } else {
            clearTimeout(idleTimer);
            if (restStyle) Object.assign(el.style, restStyle);
            for (const f of document.querySelectorAll("[data-tl-player-pos]")) {
                f.style.position = "";
                f.style.cursor = "";
                delete f.dataset.tlPlayerPos;
                f.removeEventListener("mousemove", onFsMove);
                f.removeEventListener("pointerdown", onFsMove);
            }
            const t = opts.fullscreen?.();
            if (t) {
                t.style.cursor = "";
                t.removeEventListener("mousemove", onFsMove);
                t.removeEventListener("pointerdown", onFsMove);
            }
        }
        paint();
    }
    document.addEventListener("fullscreenchange", onFullscreen);
    let wasPlaying = false;
    const seekAt = (ev) => {
        const r = track.getBoundingClientRect();
        const f = Math.max(0, Math.min(1, (ev.clientX - r.left) / (r.width || 1)));
        seekAll(f * total());
        paint();
    };
    track.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0 || !members.length) return;
        ev.preventDefault();
        ev.stopPropagation();
        scrubbing = true;
        wasPlaying = playing();
        for (const { m } of members) m.pause?.();
        track.setPointerCapture?.(ev.pointerId);
        seekAt(ev);
    });
    track.addEventListener("pointermove", (ev) => {
        if (scrubbing) seekAt(ev);
    });
    const endScrub = () => {
        if (!scrubbing) return;
        scrubbing = false;
        if (!wasPlaying) paint();
        else if (host) {
            // the element carries on from the new place; the host's own
            // watcher takes it from there (past a cut: on to the next)
            for (const { m } of members) startMember(m);
            tick();
        } else play();
    };
    track.addEventListener("pointerup", endScrub);
    track.addEventListener("pointercancel", endScrub);

    function setLoop(on, byUser) {
        loop = !!on;
        if (opts.loopKey) {
            const loops = { ...(prefs().loops ?? {}), [opts.loopKey]: loop };
            savePrefs({ loops });
        }
        if (byUser && loop && !playing() && total() && now() >= total() - 0.05) {
            play();
        }
        paint();
    }

    // members' own events keep the face honest (a play() from elsewhere,
    // metadata arriving, a src swap)
    // a <video> given another file: it is being placed from now on
    const onLoadStart = (ev) => {
        if (host) return;           // the host places its own element
        if (ev.target instanceof HTMLMediaElement) placing.add(ev.target);
    };
    const onSeeked = (ev) => {
        // a host seeks with the element paused: the face follows it
        if (host) {
            paint();
            return;
        }
        if (placing.has(ev.target) && !ev.target.seeking) {
            placing.delete(ev.target);
            paint();
        }
    };
    const onMedia = (ev) => {
        if (host) {
            paint();
            if (playing()) tick();
            return;
        }
        // ...put where the group is, not at its start -- by the clock the
        // others read now (it is not one of them until it is placed). A
        // LiveClip keeps its own place across its swaps.
        if (ev?.type === "loadedmetadata"
                && ev.target instanceof HTMLMediaElement) {
            const x = members.find((m) => m.m === ev.target);
            if (x && !x.stretch && !scrubbing) {
                placing.add(x.m);
                const want = targetOf(x, now());
                if (Math.abs(x.m.currentTime - want) > 0.01) {
                    try {
                        x.m.currentTime = want;     // placed on "seeked"
                    } catch (err) {
                        placing.delete(x.m);
                    }
                } else {
                    placing.delete(x.m);
                }
            } else {
                placing.delete(ev.target);
            }
        }
        if (wanted && ev?.type !== "pause" && !scrubbing) resume();
        paint();
        if (playing()) tick();
    };
    function setMembers(list) {
        const before = members.map((x) => x.m);
        const t = now();
        const was = playing();
        for (const m of before) {
            m.removeEventListener?.("play", onMedia);
            m.removeEventListener?.("pause", onMedia);
            m.removeEventListener?.("loadedmetadata", onMedia);
            m.removeEventListener?.("canplay", onMedia);
            m.removeEventListener?.("loadstart", onLoadStart);
            m.removeEventListener?.("seeked", onSeeked);
        }
        members = (list ?? []).filter(Boolean).map((x) => (x.m
            ? { m: x.m, stretch: !!x.stretch, offset: Number(x.offset) || 0,
                primary: !!x.primary }
            : { m: x, stretch: false, offset: 0 }));
        for (const x of members) {
            // the clock loops the members, except a free-running one
            if (!host) x.m.loop = x.stretch && x !== master();
            x.m.addEventListener?.("play", onMedia);
            x.m.addEventListener?.("pause", onMedia);
            x.m.addEventListener?.("loadedmetadata", onMedia);
            x.m.addEventListener?.("canplay", onMedia);
            x.m.addEventListener?.("loadstart", onLoadStart);
            x.m.addEventListener?.("seeked", onSeeked);
            // a newcomer joins the group where it is
            if (!host && !before.includes(x.m) && before.length) {
                try { x.m.currentTime = targetOf(x, t); } catch (err) { /* */ }
                if (was) void x.m.play?.()?.catch?.(() => {});
                else x.m.pause?.();
            }
        }
        if (heard && !members.some((x) => x.m === heard)) heard = null;
        applySound();
        paint();
        // the clock runs whenever the group plays -- also when a member
        // was already playing as it joined (its "play" fired before we
        // listened), or nothing would loop it at its end
        if (was || playing()) tick();
    }

    paint();
    return {
        el,
        setMembers,
        hear(m) {
            heard = m;
            applySound();
        },
        setLoop: (on) => setLoop(on, false),
        get loop() { return loop; },
        // the viewer's own choice (a host starting an element with sound)
        get muted() { return muted || volume === 0; },
        // console diagnostic
        get state() {
            return { members: members.length, scrubbing, overlay,
                     master: master()?.m?.constructor?.name ?? null,
                     primary: !!master()?.primary,
                     t: now(), total: total(), seeks };
        },
        get playing() { return playing(); },
        // meant to play (play() asked, no pause since) -- true before
        // the media itself has started
        get wanted() { return wanted; },
        get total() { return total(); },
        track,
        play, pause, toggle,
        seek(t) { seekAll(t); paint(); },
        setRange(a, b) {
            range = Number.isFinite(a) && Number.isFinite(b) ? [a, b] : null;
            paint();
        },
        refresh: paint,
        attach,
        destroy() {
            destroyed = true;
            cancelAnimationFrame(raf);
            clearTimeout(idleTimer);
            window.removeEventListener("wheel", onWheel, { capture: true });
            document.removeEventListener("fullscreenchange", onFullscreen);
            setMembers([]);
            el.remove();
        },
    };
}

/**
 * A sampler's live preview (KJ's Model Preview Override sends one short
 * clip per sampling step) as a member the bar can drive. Each new step
 * REPLACES the picture but keeps the place it had reached and whether it
 * was playing, so the preview no longer jumps back to its first frame at
 * every step -- and the swap itself is seamless:
 *   MP4 -> two <video>s: the new step loads in the hidden one, is put at
 *     the place the visible one has reached by then, and only takes over
 *     once its frame is ready (one <video> reloading in place showed a
 *     hitch at every step)
 *   animated WebP -> decoded frame by frame (ImageDecoder) onto a canvas
 *     and drawn by our own clock; where ImageDecoder is missing it falls
 *     back to an <img>, which plays by itself and cannot be controlled
 *   JPEG -> a still on the canvas
 * `el` is the element to put on screen; it is styled to fill its box.
 */
// how far ahead a step's new video starts (see updateVideo)
const LIVE_LEAD = 0.05;

export class LiveClip extends EventTarget {
    constructor() {
        super();
        this.el = document.createElement("div");
        Object.assign(this.el.style, { position: "relative", width: "100%",
            height: "100%" });
        this.videos = [0, 1].map(() => {
            const v = document.createElement("video");
            v.muted = true;
            v.playsInline = true;
            v.preload = "auto";
            return v;
        });
        this.active = 0;        // which of the two is on screen
        this.urls = [null, null];
        this.canvas = document.createElement("canvas");
        this.img = document.createElement("img");
        this.imgUrl = null;
        for (const e of [...this.videos, this.canvas, this.img]) {
            Object.assign(e.style, { position: "absolute", inset: "0",
                width: "100%", height: "100%", objectFit: "contain",
                display: "none" });
            this.el.appendChild(e);
        }
        this.mode = null;       // "video" | "frames" | "img"
        this.frames = [];       // [{bitmap, t, d}] for "frames"
        this.length = 0;
        this.t = 0;             // the frames' clock, in seconds
        this.playingFrames = false;
        this.lastTs = 0;
        this.raf = 0;
        this.seq = 0;
        this.wantPlay = true;
        this.muted = true;
        this.volume = 1;
        this._loop = false;
        this.videos.forEach((v, k) => {
            for (const ev of ["play", "pause", "loadedmetadata"]) {
                v.addEventListener(ev, () => {
                    if (this.mode === "video" && k === this.active) {
                        this.dispatchEvent(new Event(ev));
                    }
                });
            }
        });
    }

    get video() {
        return this.videos[this.active];
    }
    get loop() {
        return this._loop;
    }
    set loop(on) {
        this._loop = !!on;
        for (const v of this.videos) v.loop = this._loop;
    }
    get width() {
        return this.mode === "video" ? this.video.videoWidth
            : (this.frames[0]?.bitmap.width ?? this.img.naturalWidth ?? 0);
    }
    get height() {
        return this.mode === "video" ? this.video.videoHeight
            : (this.frames[0]?.bitmap.height ?? this.img.naturalHeight ?? 0);
    }
    get duration() {
        if (this.mode === "video") return this.video.duration || 0;
        return this.mode === "frames" ? this.length : 0;
    }
    get currentTime() {
        if (this.mode === "video") return this.video.currentTime;
        return this.mode === "frames" ? this.t : 0;
    }
    set currentTime(v) {
        if (this.mode === "video") {
            this.video.currentTime = v;
            return;
        }
        this.t = Math.max(0, Math.min(this.length, Number(v) || 0));
        this.draw();
    }
    get paused() {
        if (this.mode === "video") return this.video.paused;
        return !this.playingFrames;
    }
    get ended() {
        return this.mode === "video" ? this.video.ended : false;
    }
    play() {
        this.wantPlay = true;
        if (this.mode === "video") return this.video.play();
        if (this.mode === "frames" && !this.playingFrames && this.length) {
            this.playingFrames = true;
            this.lastTs = performance.now();
            this.loopFrames();
            this.dispatchEvent(new Event("play"));
        }
        return Promise.resolve();
    }
    pause() {
        this.wantPlay = false;
        if (this.mode === "video") {
            this.video.pause();
            return;
        }
        if (this.playingFrames) {
            this.playingFrames = false;
            cancelAnimationFrame(this.raf);
            this.dispatchEvent(new Event("pause"));
        }
    }
    loopFrames() {
        cancelAnimationFrame(this.raf);
        if (!this.playingFrames) return;
        const ts = performance.now();
        this.t += (ts - this.lastTs) / 1000;
        this.lastTs = ts;
        // the frames cycle by themselves (a group's clock may also seek
        // them; it no longer pulls them along every frame)
        if (this.length && this.t >= this.length) this.t %= this.length;
        this.draw();
        this.raf = requestAnimationFrame(() => this.loopFrames());
    }
    draw() {
        if (this.mode !== "frames" || !this.frames.length) return;
        let f = this.frames[0];
        for (const x of this.frames) {
            if (x.t <= this.t + 1e-6) f = x;
            else break;
        }
        const c = this.canvas;
        if (c.width !== f.bitmap.width || c.height !== f.bitmap.height) {
            c.width = f.bitmap.width;
            c.height = f.bitmap.height;
        }
        c.getContext("2d").drawImage(f.bitmap, 0, 0);
    }
    show(which) {
        this.videos.forEach((v, k) => {
            v.style.display = which === "video" && k === this.active
                ? "block" : "none";
        });
        this.canvas.style.display = which === "frames" ? "block" : "none";
        this.img.style.display = which === "img" ? "block" : "none";
    }

    /** One step's preview: {image: base64, mime}. Resolves when shown. */
    async update(data) {
        const seq = ++this.seq;
        // H3 Live Preview (nodes_live) sends its clip on the take's own
        // timeline, with how many frames are pinned at each end; KJ's
        // Model Preview Override sends neither
        this.meta = { synced: !!data?.synced, head: Number(data?.head) || 0,
                      tail: Number(data?.tail) || 0,
                      frames: Number(data?.frames) || 0 };
        const mime = typeof data?.mime === "string" ? data.mime : "image/jpeg";
        let bytes;
        try {
            const bin = atob(data.image);
            bytes = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        } catch (err) {
            return;
        }
        const place = this.currentTime;
        // play on if it was MEANT to play: a sampling pass opens with one
        // still (KJ's starting-noise picture), which cannot play -- read
        // as "paused", the frames after it stayed stopped
        const was = this.wantPlay;
        if (mime === "video/mp4") {
            await this.updateVideo(bytes, mime, seq, place, was);
            return;
        }
        // stills and animated WebP: decoded to frames we draw ourselves
        let frames = null;
        if (typeof ImageDecoder !== "undefined") {
            try {
                frames = await decodeFrames(bytes, mime);
            } catch (err) {
                frames = null;
            }
        }
        if (seq !== this.seq) {
            for (const f of frames ?? []) f.bitmap.close?.();
            return;
        }
        this.stopVideos();
        if (!frames) {
            // no decoder: the <img> plays an animation by itself
            const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
            const old = this.imgUrl;
            this.imgUrl = url;
            this.stopFrames();
            this.mode = "img";
            this.img.src = url;
            this.show("img");
            if (old) URL.revokeObjectURL(old);
            this.dispatchEvent(new Event("loadedmetadata"));
            return;
        }
        const oldFrames = this.frames;
        const playing = this.mode === "frames" && this.playingFrames;
        this.frames = frames;
        this.length = frames.length > 1
            ? frames[frames.length - 1].t + frames[frames.length - 1].d : 0;
        this.mode = "frames";
        // a playing preview carried on while the new frames decoded
        if (!playing) this.t = this.length ? place % this.length : 0;
        else if (this.length) this.t %= this.length;
        this.draw();
        this.show("frames");
        for (const f of oldFrames) f.bitmap.close?.();
        this.dispatchEvent(new Event("loadedmetadata"));
        if (was && this.length && !this.playingFrames) void this.play();
    }
    async updateVideo(bytes, mime, seq, place, was) {
        const fresh = this.mode !== "video";
        const k = fresh ? this.active : 1 - this.active;
        const v = this.videos[k];
        const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
        const old = this.urls[k];
        this.urls[k] = url;
        v.loop = this._loop;
        await new Promise((resolve) => {
            const done = () => {
                v.removeEventListener("loadeddata", done);
                v.removeEventListener("error", done);
                resolve();
            };
            v.addEventListener("loadeddata", done);
            v.addEventListener("error", done);
            v.src = url;
        });
        if (old) URL.revokeObjectURL(old);
        if (seq !== this.seq) return;
        // where the one on screen has got to by now (it kept playing
        // while this loaded), wrapped into the new clip
        const on = this.videos[this.active];
        const now = fresh ? place : on.currentTime;
        const d = v.duration || 0;
        const at = d ? now % d : 0;
        if (Math.abs(v.currentTime - at) > 0.01) {
            await new Promise((resolve) => {
                const done = () => {
                    v.removeEventListener("seeked", done);
                    resolve();
                };
                v.addEventListener("seeked", done);
                try { v.currentTime = at; } catch (err) { resolve(); }
                setTimeout(done, 300);
            });
        }
        if (seq !== this.seq) return;
        this.stopFrames();
        if (was) {
            // started before it is shown, so the first frame on screen is
            // already moving -- and a little AHEAD of the old one, which
            // keeps playing while this one starts (lined up exactly, the
            // swap held a frame or two)
            try {
                if (!fresh && !on.paused) {
                    v.currentTime = d ? (on.currentTime + LIVE_LEAD) % d : 0;
                }
            } catch (err) { /* */ }
            await v.play().catch(() => {});
        }
        if (seq !== this.seq) return;
        const prev = this.active;
        this.active = k;
        const fromVideo = this.mode === "video";
        this.mode = "video";
        this.show("video");
        if (!fresh && fromVideo) this.videos[prev].pause();
        this.dispatchEvent(new Event("loadedmetadata"));
        // it started while hidden, so its own "play" never reached a
        // listener: the bar's clock must hear it to loop it at its end
        if (!v.paused) this.dispatchEvent(new Event("play"));
    }
    stopVideos() {
        for (const v of this.videos) v.pause();
    }
    stopFrames() {
        cancelAnimationFrame(this.raf);
        this.playingFrames = false;
        for (const f of this.frames) f.bitmap.close?.();
        this.frames = [];
        this.length = 0;
    }
    clear() {
        ++this.seq;
        this.stopFrames();
        for (const v of this.videos) {
            v.pause();
            v.removeAttribute("src");
            v.load();
        }
        this.urls.forEach((u) => u && URL.revokeObjectURL(u));
        this.urls = [null, null];
        this.img.removeAttribute("src");
        if (this.imgUrl) URL.revokeObjectURL(this.imgUrl);
        this.imgUrl = null;
        this.mode = null;
        this.t = 0;
        this.show(null);
    }
}

async function decodeFrames(bytes, mime) {
    const dec = new ImageDecoder({ data: bytes, type: mime });
    await dec.tracks.ready;
    const n = dec.tracks.selectedTrack?.frameCount ?? 1;
    const frames = [];
    let t = 0;
    for (let i = 0; i < Math.max(1, n); i++) {
        const { image } = await dec.decode({ frameIndex: i });
        const d = (image.duration ?? 0) / 1e6 || 1 / 12;
        const bitmap = await createImageBitmap(image);
        image.close();
        frames.push({ bitmap, t, d });
        t += d;
    }
    dec.close();
    return frames;
}
