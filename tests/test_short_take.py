"""A very short take can still be continued: the window reaches back
into the frames it held from the clip it continues.

A take made by extending another clip stores the window it pinned from
that clip (its pinned head) ahead of what it delivered: 35 + 33 = 68 raw
frames for a 33-frame take. A 39-frame window ending at its end does not
fit in the 33 delivered frames, but it fits in the 68 -- the take's own
latents, continuous with what it delivered. The join does not move: the
new take still starts at the cut. Likewise a take made by prepending
keeps the frames it led into (its pinned tail), which a prepend's window
may run on into.

A pin from the take's pixels (clip_pixels) reads the take's FILE, which
only has the delivered frames; there the frames before them are read from
the parent's file, which shows exactly those frames -- and a prepend's
frames after them from the file of the clip the take leads into.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
import folder_paths  # noqa: E402
from obvpm_tl_test import frames as fr  # noqa: E402
from obvpm_tl_test import mctx  # noqa: E402
from obvpm_tl_test import nodes_assemble as na  # noqa: E402
from obvpm_tl_test import nodes_encode as ne  # noqa: E402
from obvpm_tl_test import nodes_joint as nj  # noqa: E402
from obvpm_tl_test import nodes_load  # noqa: E402
from obvpm_tl_test import nodes_pins  # noqa: E402

EXTEND = "extend (pin tail)"
PREPEND = "prepend (pin head)"


def take(head=35, delivered=33, tail=0, self_id="T", clip="t.mp4",
         pins=None):
    raw = head + delivered + tail
    steps = next(s for s in range(1, 200) if fr.pixel_frames(s) == raw)
    return {"self_id": self_id, "clip": clip,
            "meta": {"raw_frames": str(raw), "delivered_frames": str(delivered),
                     "pinned_head_frames": str(head),
                     "pinned_tail_frames": str(tail),
                     "pins": json.dumps(pins or [])},
            "video_latent": torch.arange(steps, dtype=torch.float32)
                                 .view(1, 1, steps, 1, 1).expand(1, 4, steps, 2, 2)
                                 .clone(),
            "audio_latent": torch.zeros(1, 4, 2, round(raw / 24 * 40))}


def pin_of(bundle, role=EXTEND, at_frame=None, mode="masked"):
    specs = nodes_load._create_pins(bundle, "t.mp4", role, "39",
                                    at_frame=at_frame, pin_mode=mode)
    return nodes_pins._prepare_clip_pin(0, specs[0], snap_down=False)


class LatentRoute(unittest.TestCase):
    def test_a_33_frame_take_extends_by_reaching_into_its_held_head(self):
        p = pin_of(take(35, 33))
        spec = p["spec"]
        # the window ends on the latent grid at delivered frame 21 (as the
        # tail of any pinned-head take does) and starts 18 frames before
        # the take's first delivered frame: raw 17..56
        self.assertEqual((spec["take_from"], spec["take_from_frame"]),
                         ("at_frame", 21))
        self.assertEqual((spec["source_start"], spec["source_frames"]),
                         (17, 39))
        self.assertEqual(p["covered"], 39)
        # the slice is the stored latent's own steps from step 5 on
        self.assertEqual(p["video"][0, 0, :, 0, 0].tolist(),
                         list(range(5, 5 + fr.steps_for_frames(39))))

    def test_the_join_stays_at_the_cut(self):
        spec = pin_of(take(35, 33))["spec"]
        # the window's END, in the take's delivered frames: where the new
        # take continues from
        self.assertEqual(spec["source_start"] + spec["source_frames"] - 35,
                         21)
        rel, sid, join = mctx.summarize_pins([spec])
        self.assertEqual((rel, sid), ("extends", "T"))
        self.assertEqual(join - 35, 21 - 4)      # handover a step inside

    def test_a_cut_close_to_the_start_too(self):
        # an explicit cut at delivered frame 4: raw 39 = 35 + 4, a window
        # of 39 from raw 0
        p = pin_of(take(35, 33), at_frame=4)
        self.assertEqual(p["spec"]["source_start"], 0)

    def test_apply_alone_takes_the_room_too(self):
        # a spec asking for 39 frames ending at frame 21 (as a timeline
        # pin is handed over): "only 21 frames are available" before
        (specs,) = nodes_pins.H3MCtxPinSpec().build(
            mctx=take(35, 33), window=39, take_from="at_frame",
            take_from_frame=21, place="before", place_at_frame=0,
            audio_window=0, mode="masked")
        p = nodes_pins._prepare_clip_pin(0, specs[0], snap_down=False)
        self.assertEqual(p["spec"]["source_start"], 17)

    def test_a_root_this_short_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, "too short to extend"):
            pin_of(take(0, 34))

    def test_not_even_with_its_held_head(self):
        # 5 held + 29 delivered: no 39-frame window ends inside
        with self.assertRaisesRegex(ValueError, "even reaching 5 frames"):
            pin_of(take(5, 29))

    def test_a_long_take_is_untouched(self):
        p = pin_of(take(35, 106))
        self.assertEqual(p["spec"]["source_start"], 102)     # 141 - 39

    def test_a_prepend_runs_on_into_the_held_tail(self):
        # a take made by prepending: 29 delivered, then the 39 frames it
        # led into
        p = pin_of(take(0, 29, 39), role=PREPEND, mode="both")
        self.assertEqual((p["spec"]["source_start"],
                          p["spec"]["source_frames"]), (0, 39))

    def test_a_short_prepend_without_a_held_tail_is_refused(self):
        with self.assertRaises(ValueError):
            pin_of(take(0, 34), role=PREPEND, mode="both")


def header(self_id, head_f=0, delivered=141, pin=None, join=0):
    return {"self_id": self_id, "raw_frames": str(head_f + delivered),
            "delivered_frames": str(delivered),
            "pinned_head_frames": str(head_f), "pinned_tail_frames": "0",
            "relation": "extends" if pin else "",
            "parent_id": pin["source_id"] if pin else "",
            "parent_join_frame": str(join),
            "pins": json.dumps([pin] if pin else [])}


def held(source_id, start):
    return {"source_id": source_id, "source_kind": "clip", "place": "before",
            "source_start": start, "source_frames": 39, "mode": "masked",
            "handover_edge": True}


class JointLayout(unittest.TestCase):
    """G, then P extending G (35 held + 106), cut at P's frame 21, then C
    extending P from raw 17: C's held head runs back through P's into G's
    frames. The upscale lays C by its pin and the cut decides the rows."""

    def layout(self):
        g = header("G")
        p = header("P", 35, 106, held("G", 85), join=85 + 39 - 4)
        c = header("C", 35, 106, held("P", 17), join=17 + 39 - 4)
        # as the timeline cuts a parent where its new take joins it: the
        # window ended at P's frame 21, the join is a step inside (17)
        lines = ["g.mp4", "p.mp4 @ 0..17", "c.mp4"]
        items = nj.sequence_items(chr(10).join(lines))
        hs = [g, p, c]
        segs = nj.split_segments(items, hs)
        cuts = nj.shown_cuts(lines, hs, nj.derive_flags(items))
        ticks = [fr.audio_total(int(h["raw_frames"])) for h in hs]
        return segs, nj.timeline_layout(lines, hs, ticks, cuts=cuts)

    def test_one_stretch_and_each_clip_where_its_pin_says(self):
        segs, lay = self.layout()
        self.assertEqual([s["clips"] for s in segs], [[0, 1, 2]])
        self.assertEqual(lay["starts"], [0, 85, 102])
        self.assertEqual(lay["hard_hold"], [False, True, True])

    def test_the_rows_are_what_the_cut_shows(self):
        segs, lay = self.layout()
        own = lay["owner_steps"]
        at = {fr.frame_at_latent(t): o for t, o in enumerate(own)}

        def owner(lo, hi):
            return {o for f, o in at.items() if lo <= f < hi}
        # G up to its join with P (joint frame 120), P's shown frames
        # 0..17 at joint 120..137, C after its join
        self.assertEqual(owner(0, 115), {0})
        self.assertEqual(owner(121, 136), {1})
        self.assertEqual(owner(141, 230), {2})


class OutputFolder(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="obvpm_tl_short_")
        self._old_out = folder_paths.get_output_directory()
        folder_paths.set_output_directory(self.root)
        nodes_load._HASH_CACHE.clear()

    def tearDown(self):
        folder_paths.set_output_directory(self._old_out)
        shutil.rmtree(self.root, ignore_errors=True)

    def video(self, rel, content):
        path = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def sidecar(self, video_path, **meta):
        full = {"format": mctx.FORMAT, "self_id": mctx.hash_file(video_path),
                "relation": "", "parent_id": "", "pins": "[]"}
        full.update({k: (v if isinstance(v, str) else str(v))
                     for k, v in meta.items()})
        save_file({"video": torch.zeros(1)}, mctx.sidecar_path(video_path),
                  metadata=full)
        return full

    def family(self, parent_head=0, take_dir="proj", recorded=None,
               start=61):
        """clip_00046 (100 delivered after `parent_head` held) and a
        33-frame take extending it from parent raw `start`..`start + 39`
        (35 of them held)."""
        parent = self.video("proj/clip_00046.mp4", b"parent frames")
        pmeta = self.sidecar(parent, delivered_frames=100,
                             pinned_head_frames=parent_head,
                             raw_frames=100 + parent_head)
        p = {"place": "before", "source_kind": "clip",
             "source_id": pmeta["self_id"], "source_start": start,
             "source_frames": 39, "mode": "masked"}
        if recorded:
            p["source_path"] = recorded
        child = self.video(take_dir + "/clip_00047.mp4", b"take frames")
        cmeta = self.sidecar(child, delivered_frames=33,
                             pinned_head_frames=35, raw_frames=68,
                             audio_phase="0", audio_synced="1",
                             pins=json.dumps([p]))
        return parent, child, cmeta

    def prepend_family(self, child_head=0, tail=39, start=0,
                       take_dir="proj", recorded=None, kind="clip"):
        """clip_00050 (100 delivered after `child_head` held) and a
        29-frame take made by prepending to it: its window was the
        clip's raw `start`..`start + 39`, `tail` of them held (39 for a
        hard hold, fewer when it handed some over)."""
        child = self.video("proj/clip_00050.mp4", b"child frames")
        cmeta = self.sidecar(child, delivered_frames=100,
                             pinned_head_frames=child_head,
                             raw_frames=100 + child_head)
        p = {"place": "after", "source_kind": kind,
             "source_id": cmeta["self_id"], "source_start": start,
             "source_frames": 39, "mode": "both"}
        if recorded:
            p["source_path"] = recorded
        take = self.video(take_dir + "/clip_00049.mp4", b"prepended frames")
        tmeta = self.sidecar(take, delivered_frames=29,
                             pinned_head_frames=0, pinned_tail_frames=tail,
                             raw_frames=29 + tail,
                             audio_phase="0", audio_synced="1",
                             pins=json.dumps([p]))
        return child, take, tmeta


class ParentFrames(OutputFolder):
    def test_found_beside_the_take(self):
        parent, child, header = self.family()
        # the take's delivered frame 0 is its raw 35 = the parent's raw 96
        self.assertEqual(nodes_pins._parent_frames(child, header, 18),
                         (os.path.abspath(parent), 96))

    def test_the_parents_own_held_head_is_counted_out(self):
        parent, child, header = self.family(parent_head=10)
        self.assertEqual(nodes_pins._parent_frames(child, header, 18)[1], 86)

    def test_found_by_the_recorded_path_elsewhere(self):
        parent, child, header = self.family(take_dir="other",
                                            recorded="proj/clip_00046.mp4")
        got = nodes_pins._parent_frames(child, header, 18)
        self.assertEqual(os.path.normcase(got[0]),
                         os.path.normcase(os.path.abspath(parent)))

    def test_not_found(self):
        parent, child, header = self.family(take_dir="other")
        self.assertIsNone(nodes_pins._parent_frames(child, header, 18))

    def test_no_further_back_than_the_held_head(self):
        parent, child, header = self.family()
        self.assertIsNone(nodes_pins._parent_frames(child, header, 36))


class ChildFrames(OutputFolder):
    def test_found_beside_the_take(self):
        child, take, header = self.prepend_family(start=17)
        # the take's file ends where its held tail starts: the clip's 17
        self.assertEqual(nodes_pins._child_frames(take, header, 10),
                         (os.path.abspath(child), 17))

    def test_after_the_frames_it_handed_over(self):
        # 4 of the 39 handed over (delivered): the file ends at raw 21
        child, take, header = self.prepend_family(start=17, tail=35)
        self.assertEqual(nodes_pins._child_frames(take, header, 10)[1], 21)

    def test_the_clips_own_held_head_is_counted_out(self):
        child, take, header = self.prepend_family(child_head=35, start=51)
        self.assertEqual(nodes_pins._child_frames(take, header, 10)[1], 16)

    def test_a_pixel_recipe_counts_the_files_frames(self):
        child, take, header = self.prepend_family(child_head=35, start=51,
                                                  kind="clip_pixels")
        self.assertEqual(nodes_pins._child_frames(take, header, 10)[1], 51)

    def test_found_by_the_recorded_path_elsewhere(self):
        child, take, header = self.prepend_family(
            take_dir="other", recorded="proj/clip_00050.mp4")
        got = nodes_pins._child_frames(take, header, 10)
        self.assertEqual(os.path.normcase(got[0]),
                         os.path.normcase(os.path.abspath(child)))

    def test_not_found(self):
        child, take, header = self.prepend_family(take_dir="other")
        self.assertIsNone(nodes_pins._child_frames(take, header, 10))

    def test_no_further_on_than_the_held_tail(self):
        child, take, header = self.prepend_family()
        self.assertIsNone(nodes_pins._child_frames(take, header, 40))

    def test_not_past_the_clips_end(self):
        child, take, header = self.prepend_family(start=85)
        self.assertIsNone(nodes_pins._child_frames(take, header, 20))

    def test_an_extend_made_take_has_none(self):
        parent, child, header = self.family()
        self.assertIsNone(nodes_pins._child_frames(child, header, 10))


class PixelRoute(OutputFolder):
    """The pixel pin of the short take, with the file reads and
    the encode stood in for: which frames, of which file, in what order."""

    SR = 32000

    def setUp(self):
        super().setUp()
        self.parent, self.child, self.header = self.family()
        self.stub_files({self.parent: (100, 1000), self.child: (33, 2000)})

    def stub_files(self, files):
        """Stand in for the file reads and the encode: each file is
        (frames, tag), its frame f decodes to pixels of value tag + f and
        its sound to the tag."""
        frames = {os.path.abspath(p): v[0] for p, v in files.items()}
        tag = {os.path.abspath(p): v[1] for p, v in files.items()}
        sr = self.SR
        self.encoded = {}
        self._saved = (ne.probe_clip, ne.decode_window, ne._decode_audio_range,
                       ne.encode_bundle)

        def probe(path):
            return {"frames": frames[os.path.abspath(path)], "fps": 24.0,
                    "src_frames": frames[os.path.abspath(path)],
                    "width": 64, "height": 32}

        def decode(path, start, count, info=None):
            t = tag[os.path.abspath(path)]
            img = torch.zeros(count, 32, 64, 3)
            for k in range(count):
                img[k] = t + start + k
            return img, self.sound(path, start / 24, (start + count) / 24)

        def sound(path, a, b):
            return {"waveform": torch.full(
                (1, 1, round((b - a) * sr)),
                float(tag[os.path.abspath(path)])), "sample_rate": sr}
        self.sound = sound

        def encode(images, vae, audio_vae, **kw):
            self.encoded = {"images": images, "audio": kw.get("audio")}
            n = int(images.shape[0])
            steps = next(s for s in range(1, 200) if fr.pixel_frames(s) == n)
            return {"self_id": "enc", "meta": {
                        "raw_frames": str(n), "delivered_frames": str(n),
                        "pinned_head_frames": "0"},
                    "video_latent": torch.zeros(1, 4, steps, 2, 4),
                    "audio_latent": torch.zeros(1, 4, 2, round(n / 24 * 40))}
        ne.probe_clip, ne.decode_window = probe, decode
        ne._decode_audio_range = sound
        ne.encode_bundle = encode

    def tearDown(self):
        (ne.probe_clip, ne.decode_window, ne._decode_audio_range,
         ne.encode_bundle) = self._saved
        super().tearDown()

    def pin(self, cut=21):
        spec = {"source_kind": "clip_pixels",
                "source_path": "proj/clip_00047.mp4",
                "take_from": "at_frame", "take_from_frame": cut,
                "requested_window": 39, "place": "before", "mode": "masked",
                "fit": "stretch"}
        ctx = {"vae": object(), "audio_vae": object(),
               "width": 64, "height": 32}
        return nodes_pins._prepare_pixel_pin(0, spec, False, ctx)

    def test_the_parents_frames_then_the_takes(self):
        self.pin()
        firsts = self.encoded["images"][:, 0, 0, 0].tolist()
        # 39 frames ending at the take's frame 21: the parent's 78..95
        # (its frames under the take's held 17..34), then the take's 0..20
        self.assertEqual(firsts, [1000 + f for f in range(78, 96)]
                         + [2000 + f for f in range(0, 21)])

    def test_the_sound_follows_the_same_files(self):
        self.pin()
        wave = self.encoded["audio"]["waveform"][0, 0]
        self.assertAlmostEqual(len(wave) / self.SR, 39 / 24, delta=0.01)
        self.assertEqual(float(wave[0]), 1000.0)
        self.assertEqual(float(wave[-1]), 2000.0)
        split = round(18 / 24 * self.SR)
        self.assertEqual(float(wave[split - 10]), 1000.0)
        self.assertEqual(float(wave[split + 10]), 2000.0)

    def test_a_window_that_fits_reads_the_take_alone(self):
        # a 22-frame window fits in the take's 33 frames: nothing borrowed
        spec = {"source_kind": "clip_pixels",
                "source_path": "proj/clip_00047.mp4",
                "take_from": "at_frame", "take_from_frame": 33,
                "requested_window": 22, "place": "before", "mode": "guide",
                "fit": "stretch"}
        nodes_pins._prepare_pixel_pin(0, spec, False, {
            "vae": object(), "audio_vae": object(), "width": 64, "height": 32})
        firsts = self.encoded["images"][:, 0, 0, 0].tolist()
        self.assertEqual(firsts, [2000 + f for f in range(11, 33)])

    def test_without_the_parent_it_is_refused(self):
        os.remove(self.parent)
        os.remove(mctx.sidecar_path(self.parent))
        with self.assertRaisesRegex(ValueError, "only offers 21 frames"):
            self.pin()


class PrependPixelRoute(PixelRoute):
    """The same, the other way round: the pixel pin for a
    prepend to a 29-frame take made by prepending to clip_00050 (whose
    raw 17.. it held)."""

    def setUp(self):
        OutputFolder.setUp(self)
        self.child, self.take, self.header = self.prepend_family(start=17)
        self.stub_files({self.child: (100, 1000), self.take: (29, 2000)})

    def pin(self, take_from="head", cut=0):
        spec = {"source_kind": "clip_pixels",
                "source_path": "proj/clip_00049.mp4",
                "take_from": take_from, "take_from_frame": cut,
                "requested_window": 39, "place": "after", "mode": "both",
                "fit": "stretch"}
        ctx = {"vae": object(), "audio_vae": object(),
               "width": 64, "height": 32}
        return nodes_pins._prepare_pixel_pin(0, spec, False, ctx)

    def test_the_takes_frames_then_the_clips(self):
        self.pin()
        firsts = self.encoded["images"][:, 0, 0, 0].tolist()
        # 39 frames from the take's frame 0: its 29, then the clip's 17..26
        self.assertEqual(firsts, [2000 + f for f in range(0, 29)]
                         + [1000 + f for f in range(17, 27)])

    def test_a_window_ending_past_the_file_too(self):
        # as a prepend at a cut is handed over: the window ENDS at
        # cut + 39, here the take's frame 39 -- 10 past its file
        self.pin("at_frame", 39)
        firsts = self.encoded["images"][:, 0, 0, 0].tolist()
        self.assertEqual(firsts, [2000 + f for f in range(0, 29)]
                         + [1000 + f for f in range(17, 27)])

    def test_the_sound_follows_the_same_files(self):
        self.pin()
        wave = self.encoded["audio"]["waveform"][0, 0]
        self.assertAlmostEqual(len(wave) / self.SR, 39 / 24, delta=0.01)
        self.assertEqual(float(wave[0]), 2000.0)
        self.assertEqual(float(wave[-1]), 1000.0)
        split = round(29 / 24 * self.SR)
        self.assertEqual(float(wave[split - 10]), 2000.0)
        self.assertEqual(float(wave[split + 10]), 1000.0)

    def test_the_recipe_names_the_take(self):
        pin = self.pin()
        self.assertEqual((pin["spec"]["source_start"],
                          pin["spec"]["source_frames"]), (0, 39))

    def test_a_window_that_fits_reads_the_take_alone(self):
        spec = {"source_kind": "clip_pixels",
                "source_path": "proj/clip_00049.mp4",
                "take_from": "head", "take_from_frame": 0,
                "requested_window": 22, "place": "after", "mode": "both",
                "fit": "stretch"}
        nodes_pins._prepare_pixel_pin(0, spec, False, {
            "vae": object(), "audio_vae": object(), "width": 64, "height": 32})
        firsts = self.encoded["images"][:, 0, 0, 0].tolist()
        self.assertEqual(firsts, [2000 + f for f in range(0, 22)])

    def test_without_the_clip_it_is_refused(self):
        os.remove(self.child)
        os.remove(mctx.sidecar_path(self.child))
        with self.assertRaisesRegex(ValueError, "only offers 29 frames"):
            self.pin()

    def test_a_cut_past_the_file_without_the_clip_is_refused(self):
        os.remove(self.child)
        os.remove(mctx.sidecar_path(self.child))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.pin("at_frame", 39)

    # the extend cases do not apply here
    test_the_parents_frames_then_the_takes = None
    test_without_the_parent_it_is_refused = None


# A, a 124-frame root, and X, the take that extends it: X held A's raw
# 85..123 (35 held, 4 handed over), so X's raw r is A's raw 85 + r and X
# joins A at its frame 120. Its held sound started on A's audio step 142.
A_BUNDLE = dict(clip="proj/a.mp4", self_id="A")
X_PIN = {"place": "before", "source_kind": "clip", "source_id": "A",
         "source_start": 85, "source_frames": 39, "mode": "masked",
         "handover_edge": True, "audio_start_step": 142, "audio_steps": 65}


def a_and_x(x_pin=X_PIN):
    return (take(0, 124, **A_BUNDLE),
            take(35, 106, self_id="X", clip="proj/x.mp4", pins=[x_pin]))


def through(at_frame=102, x_pin=X_PIN):
    a, x = a_and_x(x_pin)
    return nodes_load.prepend_through(a, x, "proj/x.mp4", PREPEND, "39",
                                      at_frame=at_frame, pin_mode="both")


class PrependThrough(unittest.TestCase):
    """A prepend at a cut close to a clip's end: the window runs on past
    the clip, into the take that continues it."""

    def test_the_window_end(self):
        a, _ = a_and_x()
        self.assertEqual(nodes_load.prepend_end(a["meta"], "39", 102), 141)
        # off the latent grid: moved on to the next group, as _create_pins
        self.assertEqual(nodes_load.prepend_end(a["meta"], "39", 100), 141)

    def test_it_is_taken_from_the_take(self):
        (spec,) = through()
        # the take's raw 17 is A's 102; its delivered frames start at 35
        self.assertEqual((spec["take_from"], spec["take_from_frame"]),
                         ("at_frame", 21))
        self.assertEqual(spec["source_id"], "X")
        self.assertEqual(spec["via"]["raw_offset"], 85)
        self.assertEqual(spec["via"]["file_offset"], 120)

    def test_sliced_from_the_take_and_recorded_as_the_clip(self):
        (spec,) = through()
        (pin,) = nodes_pins._prepare_pins([spec], False)
        plain = nodes_pins._prepare_clip_pin(
            0, {k: v for k, v in spec.items() if k != "via"}, False)
        # the take's latent from its raw 17 (its step 5)
        self.assertEqual(pin["video"][0, 0, :, 0, 0].tolist(),
                         list(range(5, 5 + fr.steps_for_frames(39))))
        s = pin["spec"]
        self.assertEqual((pin["source_id"], s["source_id"]), ("A", "A"))
        self.assertEqual((s["source_start"], s["source_frames"]), (102, 39))
        self.assertEqual(s["source_path"], "proj/a.mp4")
        self.assertNotIn("via", s)
        # its sound in A's audio steps: where the take's held sound began
        self.assertEqual(s["audio_start_step"],
                         142 + plain["spec"]["audio_start_step"])

    def test_the_new_take_leads_into_the_clip_at_the_cut(self):
        (spec,) = through()
        (pin,) = nodes_pins._prepare_pins([spec], False)
        rel, sid, join = mctx.summarize_pins([pin["spec"]])
        self.assertEqual((rel, sid), ("prepends", "A"))
        self.assertGreaterEqual(join, 102)
        self.assertLess(join, 120)          # before the take's join

    def test_an_older_takes_sound_is_placed_by_rule(self):
        x_pin = {k: v for k, v in X_PIN.items()
                 if k not in ("audio_start_step", "audio_steps")}
        (spec,) = through(x_pin=x_pin)
        (pin,) = nodes_pins._prepare_pins([spec], False)
        self.assertNotIn("audio_start_step", pin["spec"])

    def test_not_before_what_the_take_holds(self):
        self.assertIsNone(through(at_frame=68))

    def test_not_a_take_of_another_clip(self):
        self.assertIsNone(through(x_pin=dict(X_PIN, source_id="B")))


class NextOnTimeline(unittest.TestCase):
    SEQ = chr(10).join(["g.mp4 [id=g1]", "proj/a.mp4 @ 102..120 [id=a1]",
                        "proj/x.mp4 [id=x1]", "~ 24", "c.mp4 [id=c1]"])

    def test_by_part(self):
        self.assertEqual(na.next_on_timeline(self.SEQ, "a1", "proj/a.mp4",
                                             102), "proj/x.mp4")

    def test_by_clip_and_cut(self):
        self.assertEqual(na.next_on_timeline(self.SEQ, None, "proj/a.mp4",
                                             102), "proj/x.mp4")

    def test_a_gap_or_the_end_is_nothing(self):
        self.assertIsNone(na.next_on_timeline(self.SEQ, "x1", "", 0))
        self.assertIsNone(na.next_on_timeline(self.SEQ, "c1", "", 0))


class TimelinePrepend(unittest.TestCase):
    """The Timeline's prepend side: through the take after the cut on
    the timeline, or refused saying why."""

    def setUp(self):
        a, x = a_and_x()
        self.bundles = {"proj/a.mp4": a, "proj/x.mp4": x}
        self._saved = nodes_load.verified_bundle_or_none
        nodes_load.verified_bundle_or_none = self.bundles.get

    def tearDown(self):
        nodes_load.verified_bundle_or_none = self._saved

    def specs(self, seq, cut=102, part="a1"):
        return na.H3Timeline()._specs_for(
            "proj/a.mp4", "prepend", "39", cut, "both", sequence=seq,
            part=part)

    def test_through_the_take_after_it(self):
        (spec,) = self.specs(chr(10).join(["proj/a.mp4 @ 102..120 [id=a1]",
                                           "proj/x.mp4 [id=x1]"]))
        self.assertEqual(spec["source_id"], "X")
        self.assertEqual(spec["via"]["source_id"], "A")

    def test_a_cut_that_fits_takes_the_clip_alone(self):
        (spec,) = self.specs(chr(10).join(["proj/a.mp4 @ 68 [id=a1]",
                                           "proj/x.mp4 [id=x1]"]), cut=68)
        self.assertEqual(spec["source_id"], "A")
        self.assertNotIn("via", spec)

    def test_nothing_after_it_is_refused_saying_why(self):
        with self.assertRaisesRegex(
                ValueError, "a prepend at frame 102 of proj/a.mp4 needs 39 "
                            "frames from there, but only 22 are left"):
            self.specs("proj/a.mp4 @ 102 [id=a1]")


class PrependThroughPixels(PixelRoute):
    """The pixel pin of the same: the take's file, which reaches back
    into the clip's for the frames before its own. clip_00047 extends
    clip_00046 from raw 51 (joining it at 86); a prepend at 68 runs past
    00046's 100 frames."""

    def setUp(self):
        OutputFolder.setUp(self)
        self.parent, self.child, self.header = self.family(start=51)
        self.stub_files({self.parent: (100, 1000), self.child: (33, 2000)})

    def pin(self):
        spec = {"source_kind": "clip_pixels",
                "source_path": "proj/clip_00047.mp4",
                "take_from": "at_frame", "take_from_frame": 21,
                "requested_window": 39, "place": "after", "mode": "both",
                "fit": "stretch",
                "via": {"source_id": mctx.hash_file(self.parent),
                        "raw_offset": 51, "file_offset": 86,
                        "audio_offset": None,
                        "source_path": "proj/clip_00046.mp4"}}
        ctx = {"vae": object(), "audio_vae": object(),
               "width": 64, "height": 32}
        (pin,) = nodes_pins._prepare_pins([spec], False, ctx)
        return pin

    def test_the_clips_frames_then_the_takes(self):
        self.pin()
        firsts = self.encoded["images"][:, 0, 0, 0].tolist()
        self.assertEqual(firsts, [1000 + f for f in range(68, 86)]
                         + [2000 + f for f in range(0, 21)])

    def test_recorded_as_the_clips(self):
        s = self.pin()["spec"]
        self.assertEqual((s["source_kind"], s["source_start"],
                          s["source_frames"]), ("clip_pixels", 68, 39))
        self.assertEqual(s["source_id"], mctx.hash_file(self.parent))
        self.assertEqual(s["source_path"], "proj/clip_00046.mp4")

    test_the_parents_frames_then_the_takes = None
    test_the_sound_follows_the_same_files = None
    test_a_window_that_fits_reads_the_take_alone = None
    test_without_the_parent_it_is_refused = None


if __name__ == "__main__":
    unittest.main()
