"""Level lock: match a clip's opening to the clip it continues.

WHY THIS EXISTS
---------------
An extend hands the model a latent slice of its parent, the model
re-renders that same window as its `pinned_head`, then continues past
it. We discard the re-render and keep the parent's original pixels, so
the join lands exactly where the parent's rendering of a moment meets
the model's continuation of ITS rendering of that same moment. The two
renderings do not agree on level, and the difference is what you see.

Measured over 38 parent->child joins in one project:

  * the step is 3.7x more consistent WITHIN a parent (four independently
    sampled takes off one clip opened +3.78, +3.18, +3.38, +3.59) than
    BETWEEN parents. It is a bias of the model given that context, not
    sampling noise -- which is why re-rolling a take never fixed it.
  * it is confined to roughly the first 12 frames and then self-corrects
  * median |step| 2.87 against a median frame-to-frame level noise of
    0.44, i.e. the step is ~6x anything else happening on screen.

Two shapes turn up. A STEP: the clip opens at a different level and
holds it. A SWING: the opening frames oscillate, which reads as flicker.
Both are handled here; see `gains`.

WHAT IT DOES
------------
Per channel, over the first N frames of the continuing clip:

  1. predict where the parent was HEADING (a line through its last
     `fit` frames, extrapolated one frame past the end) -- not simply
     its last frame, or a clip legitimately walking into shade would be
     "corrected" back out of its own lighting change
  2. fit the child's opening to a straight line: the target the wobble
     is straightened onto
  3. delta = predicted - that line's start, ramped to zero by frame N
  4. gain = desired / actual, per channel per frame

Multiplicative, not additive: a gain leaves black at black and scales
midtones proportionally, which is how exposure behaves. An offset would
lift the blacks and read as haze.

Decaying to identity by frame N is what makes it safe to chain. Each
clip is back at its native grade within half a second, so a fifty-clip
chain accumulates nothing. A constant per-clip grade -- the obvious
alternative -- multiplies down the chain and eventually clips.

SCOPE
-----
Extends only. A prepend has the same problem mirrored (its TAIL is the
re-render, so the artefact sits at its closing frames), and the maths
mirrors cleanly, but every join measured here was an extend and shipping
an unvalidated correction is worse than shipping none. `measure` reports
prepend joins; `gains` returns None for them.

Standalone: numpy + av, no torch and no comfy imports, so the maths can
be tested without a running ComfyUI.
"""

import logging
import os

_LOG = logging.getLogger("obvpm.h3")

# Frames the correction spans. From the data: the transient decays over
# ~12 frames and the child is back at the parent's level by then. Too
# short reads as a ramp; too long flattens real lighting change.
FRAMES = 12
# Frames of the parent used to predict where it was heading.
FIT = 12
# Correct the step only when it beats the local frame-to-frame noise by
# this much -- below it there is nothing to fix and a "correction" would
# only add a slow drift of its own.
STEP_OVER_NOISE = 3.0
# ... and never act on a step smaller than this outright, however quiet
# the footage is (levels are 0..255).
STEP_FLOOR = 0.6
# A swing has to clear the same bar before the straightening kicks in.
SWING_OVER_NOISE = 3.0
SWING_FLOOR = 1.5
# Gains above 1 can crush highlights that were already near white. No
# join measured needed more than ~1.04, so this is a guard, not a limit
# anything real runs into.
MAX_GAIN = 1.06
# Level means only: this is about level, not detail.
SIZE = (160, 90)

_MEANS = {}
_MEANS_CAP = 256


def _luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def channel_means(path, size=SIZE):
    """Per-frame, per-channel mean of a clip: ndarray (n, 3), 0..255.

    Cached on (path, size, mtime): a preview build asks for the same
    clip once per join it takes part in, and our takes carry a single
    keyframe so every decode is a decode of the whole file.
    """
    import av
    import numpy as np

    try:
        st = os.stat(path)
        key = (os.path.abspath(path), size, st.st_mtime_ns, st.st_size)
    except OSError:
        key = None
    if key is not None and key in _MEANS:
        return _MEANS[key]

    rows = []
    with av.open(path) as c:
        for frame in c.decode(video=0):
            a = (frame.reformat(width=size[0], height=size[1],
                                format="rgb24").to_ndarray()
                 .astype(np.float32))
            rows.append(a.mean(axis=(0, 1)))
    out = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 3))
    if key is not None:
        if len(_MEANS) >= _MEANS_CAP:
            _MEANS.clear()
        _MEANS[key] = out
    return out


def forget(path=None):
    """Drop cached means (a rebuilt clip must not be measured stale)."""
    if path is None:
        _MEANS.clear()
        return
    ap = os.path.abspath(path)
    for k in [k for k in _MEANS if k[0] == ap]:
        del _MEANS[k]


def _linked(left_header, right_header):
    """Does `right` continue `left`? Returns "extends", "prepends" or None.

    Uses nodes_assemble's lineage accessors so there is ONE definition of
    what a link is; a second one here would drift from the seam logic.
    """
    from . import nodes_assemble as na
    lh, rh = left_header or None, right_header or None
    if not lh or not rh:
        return None
    ex = na._extend_parent(rh)
    if ex and ex[0] == lh.get("self_id"):
        return "extends"
    pr = na._prepend_child(lh)
    if pr and pr[0] == rh.get("self_id"):
        return "prepends"
    return None


def _fit_line(y):
    """(slope, intercept) of a least-squares line through y."""
    import numpy as np
    n = len(y)
    x = np.arange(float(n))
    return np.polyfit(x, y, 1)


def measure(left_path, left_exit, right_path, right_enter,
            left_header=None, right_header=None,
            frames=FRAMES, fit=FIT):
    """What is happening at one join. None when it cannot be measured.

    `left_exit` is the parent's cut point (None = its own end) and
    `right_enter` the child's -- the correction must be computed against
    the footage actually IN the timeline, not against the files' ends.
    """
    import numpy as np

    kind = _linked(left_header, right_header)
    lm = channel_means(left_path)
    rm = channel_means(right_path)
    if len(lm) < 4 or len(rm) < 4:
        return None
    end = len(lm) if left_exit is None else max(0, min(int(left_exit),
                                                      len(lm)))
    start = max(0, min(int(right_enter or 0), len(rm)))
    tail = lm[max(0, end - fit):end]
    head = rm[start:start + frames]
    if len(tail) < 4 or len(head) < 4:
        return None

    predicted, fitted, delta = [], [], []
    for c in range(3):
        slope, icept = _fit_line(tail[:, c])
        p = slope * len(tail) + icept
        s2, i2 = _fit_line(head[:, c])
        predicted.append(float(p))
        fitted.append(float(i2))
        delta.append(float(p - i2))

    lt, lh_ = _luma(tail), _luma(head)
    slope, icept = _fit_line(lt)
    pred_luma = float(slope * len(lt) + icept)
    local = float(np.abs(np.diff(lt)).mean()) if len(lt) > 1 else 0.0
    # Wobble is deviation from the opening's OWN trend, not its
    # peak-to-peak range. A clip descending steeply through a real
    # lighting change has a huge range and no wobble at all, and
    # measuring range (or subtracting the parent's range, which was the
    # first attempt) scored exactly the wrong clips: a genuinely
    # flickering opening whose parent was mid-pan came out NEGATIVE.
    s2, i2 = _fit_line(lh_)
    resid = lh_ - (s2 * np.arange(float(len(lh_))) + i2)
    swing = float(resid.max() - resid.min())
    settle = (float(lh_[min(len(lh_) - 1, frames):].mean() - pred_luma)
              if len(lh_) > frames else None)

    return {
        "kind": kind,
        "step": float(lh_[:3].mean() - pred_luma),
        "local": local,
        "swing": swing,
        "settle": settle,
        "predicted": predicted,
        "fitted": fitted,
        "delta": delta,
        "frames": int(min(frames, len(head))),
        "head": head,
        # the LEFT clip's fitted tail. An extend corrects the right
        # clip's opening; a prepend corrects this instead, working
        # backwards from the join -- same measurement, other end.
        "tail": tail,
    }


def verdict(m, step_over_noise=STEP_OVER_NOISE,
            swing_over_noise=SWING_OVER_NOISE):
    """What the numbers say, as flags the UI and `gains` both read."""
    if not m:
        return {"measurable": False, "step": False, "swing": False}
    bar_s = max(STEP_FLOOR, step_over_noise * m["local"])
    bar_w = max(SWING_FLOOR, swing_over_noise * m["local"])
    return {
        "measurable": True,
        "linked": m["kind"] is not None,
        "step": abs(m["step"]) > bar_s,
        "swing": m["swing"] > bar_w,
        "step_bar": bar_s,
        "swing_bar": bar_w,
    }


def gains(m, fix_step=True, fix_swing=True, max_gain=MAX_GAIN,
          step_over_noise=STEP_OVER_NOISE,
          swing_over_noise=SWING_OVER_NOISE):
    """Per-frame, per-channel gains for the opening: (n, 3) or None.

    None means "leave this join alone" -- not linked, not measurable, or
    nothing worth correcting. Returning None rather than a unit array
    matters: it lets the caller skip the re-encode entirely.
    """
    import numpy as np

    if not m or m["kind"] != "extends":
        return None
    v = verdict(m, step_over_noise, swing_over_noise)
    do_step = fix_step and v["step"]
    do_swing = fix_swing and v["swing"]
    if not (do_step or do_swing):
        return None

    n = m["frames"]
    head = m["head"][:n]
    ramp = 1.0 - np.arange(float(n)) / float(n)
    out = np.ones((n, 3), dtype=np.float64)
    for c in range(3):
        actual = head[:, c]
        if do_swing:
            s2, i2 = _fit_line(actual)
            base = s2 * np.arange(float(n)) + i2
        else:
            base = actual
        desired = base + (m["delta"][c] * ramp if do_step else 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(actual) > 1e-6, desired / actual, 1.0)
        out[:, c] = np.clip(g, 1.0 / max_gain, max_gain)
    return out


def gains_closing(m, fix_step=True, fix_swing=True, max_gain=MAX_GAIN,
                  step_over_noise=STEP_OVER_NOISE,
                  swing_over_noise=SWING_OVER_NOISE):
    """Per-frame, per-channel gains for the CLOSING frames: (n, 3) or None.

    The mirror of `gains`, for a prepend. There the new take is the LEFT
    clip and the thing it must match is the RIGHT one, which already
    exists and may already have children -- so the correction goes on
    the take, at its tail, and works backwards from the join.

    Same `delta` either way: (what the left tail extrapolates to at the
    join) minus (what the right head actually opens at). An extend adds
    it to the right's opening; a prepend subtracts it from the left's
    close. The ramp runs the other way too -- nothing at the start of
    the window, full correction on the last frame, so the clip leaves
    its own footage untouched until it has to arrive.
    """
    import numpy as np

    if not m or m["kind"] != "prepends":
        return None
    v = verdict(m, step_over_noise, swing_over_noise)
    do_step = fix_step and v["step"]
    do_swing = fix_swing and v["swing"]
    if not (do_step or do_swing):
        return None

    n = int(min(m["frames"], len(m["tail"])))
    if n < 2:
        return None
    tail = m["tail"][-n:]
    # 1/n .. 1.0: the last frame carries the whole correction
    ramp = (np.arange(float(n)) + 1.0) / float(n)
    out = np.ones((n, 3), dtype=np.float64)
    for c in range(3):
        actual = tail[:, c]
        if do_swing:
            s2, i2 = _fit_line(actual)
            base = s2 * np.arange(float(n)) + i2
        else:
            base = actual
        desired = base - (m["delta"][c] * ramp if do_step else 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(np.abs(actual) > 1e-6, desired / actual, 1.0)
        out[:, c] = np.clip(g, 1.0 / max_gain, max_gain)
    return out


def apply_gains(images, g):
    """Scale the first len(g) frames of an IMAGE tensor (n,h,w,3), 0..1.

    Returns a new tensor; the input is left alone because the caller may
    still splice untouched frames from it.
    """
    if g is None or images is None or images.shape[0] == 0:
        return images
    out = images.clone()
    n = min(len(g), int(out.shape[0]))
    for t in range(n):
        for c in range(3):
            out[t, ..., c] = out[t, ..., c] * float(g[t][c])
    return out.clamp_(0.0, 1.0)


def join_pairs(entries, loop=False):
    """(left index, right index) for every join in playback order.

    A looping cut has one more join than it has boundaries between
    lines: its last entry into its first. Callers key their results by
    entry index, so the wrap join lands on entry 0's opening and the
    last entry's closing with no special case of its own.
    """
    pairs = [(i - 1, i) for i in range(1, len(entries))]
    if loop and entries:
        pairs.append((len(entries) - 1, 0))
    return pairs


def plan(entries, frames=FRAMES, fit=FIT, fix_step=True, fix_swing=True,
         opts_for=None, loop=False):
    """{entry index: gains} for a resolved sequence.

    Only joins where one clip genuinely continues another are corrected:
    across a gap, a cut, or between unrelated clips there is no
    continuity to preserve and matching them would invent one.

    `opts_for(i)` gives one join its own settings -- {"frames", "fix_step",
    "fix_swing"}, any subset -- or None to skip the join entirely. A join
    the caller does not want is skipped rather than measured and thrown
    away: measuring is the expensive half.
    """
    out, closing = {}, {}
    for j, i in join_pairs(entries, loop):
        left, right = entries[j], entries[i]
        if left.get("gap") or right.get("gap"):
            continue
        one = opts_for(i) if opts_for else {}
        if one is None:
            continue
        frames_i = int(one.get("frames", frames))
        step_i = bool(one.get("fix_step", fix_step))
        swing_i = bool(one.get("fix_swing", fix_swing))
        try:
            m = measure(left["path"], left.get("exit"),
                        right["path"], right.get("enter") or 0,
                        left.get("header"), right.get("header"),
                        frames=frames_i, fit=fit)
            g = gains(m, fix_step=step_i, fix_swing=swing_i)
            gc = gains_closing(m, fix_step=step_i, fix_swing=swing_i)
        except Exception:
            _LOG.exception("obvpm.h3: level lock failed at %s -> %s",
                           left.get("clip"), right.get("clip"))
            continue
        if g is not None:
            out[i] = g
            _LOG.debug("obvpm.h3: level lock %s -> %s: step %+.2f, "
                       "swing %.2f, gain %.4f..%.4f (opening %s)",
                       left.get("clip"), right.get("clip"),
                       m["step"], m["swing"], g[0].mean(), g[-1].mean(),
                       right.get("clip"))
        if gc is not None:
            # keyed by the entry that gets corrected, exactly as `out`
            # is: there it is the right clip, here the left one
            closing[j] = gc
            _LOG.debug("obvpm.h3: level lock %s -> %s: step %+.2f, "
                       "swing %.2f, gain %.4f..%.4f (closing %s)",
                       left.get("clip"), right.get("clip"),
                       m["step"], m["swing"], gc[0].mean(), gc[-1].mean(),
                       left.get("clip"))
    return out, closing


def report(entries, frames=FRAMES, fit=FIT, with_gains=False, loop=False):
    """Per-join measurements for the UI: one entry per seam index.

    `with_gains` adds the correction as a list of per-frame LUMA gains.
    Quick preview cannot re-encode, but it can put a CSS brightness()
    filter on the <video> element -- and brightness() is a linear
    multiply in the same space these gains are, so the approximation is
    exact up to the per-channel spread (measured at ~0.001 on real
    joins, i.e. invisible). The export still applies all three.
    """
    out = {}
    for j, i in join_pairs(entries, loop):
        left, right = entries[j], entries[i]
        if left.get("gap") or right.get("gap"):
            continue
        try:
            m = measure(left["path"], left.get("exit"),
                        right["path"], right.get("enter") or 0,
                        left.get("header"), right.get("header"),
                        frames=frames, fit=fit)
        except Exception:
            continue
        if not m:
            continue
        v = verdict(m)
        out[i] = {
            "kind": m["kind"], "step": round(m["step"], 2),
            "local": round(m["local"], 2), "swing": round(m["swing"], 2),
            "settle": (None if m["settle"] is None
                       else round(m["settle"], 2)),
            "fixable": bool(m["kind"] in ("extends", "prepends")
                            and (v["step"] or v["swing"])),
            "reads_as": ("step" if v["step"] and not v["swing"] else
                         "flicker" if v["swing"] and not v["step"] else
                         "step + flicker" if v["step"] else "clean"),
        }
        if with_gains:
            g = (gains_closing(m) if m["kind"] == "prepends" else gains(m))
            out[i]["gains"] = (
                None if g is None
                else [round(float(row.mean()), 5) for row in g])
    return out
