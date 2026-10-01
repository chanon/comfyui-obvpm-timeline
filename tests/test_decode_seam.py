"""Decode seams: the frames a decode EDGE got wrong at a masked join are
swapped for the newer take's own decode of them (see decodeseam).

Pins three things: which window edges a take keeps (only hard-held
masked/both windows), which joins get a swap (only where the older
clip's played edge IS its raw decode edge), and the swap itself
(aligned at the join, refused when the verify frames disagree).

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
from obvpm_tl_test import decodeseam as ds  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RAMP = {"mask_ramp_frames": 10, "mask_ramp_edge": 0.4}


def prepared(place, mode="masked", covered=39, kind="clip", **shape):
    """A pin as Apply hands it to Save: covered + its spec."""
    return {"place": place, "covered": covered,
            "spec": dict({"place": place, "mode": mode, "source_kind": kind},
                         **shape)}


class CaptureSides(unittest.TestCase):
    def test_hard_masked_extend_keeps_its_head(self):
        for mode in ("masked", "both"):
            self.assertEqual(ds.capture_sides([prepared("before", mode)]),
                             {"before"})

    def test_hard_arrival_keeps_its_tail(self):
        self.assertEqual(ds.capture_sides([prepared("after", "both")]),
                         {"after"})

    def test_bridge_keeps_both(self):
        self.assertEqual(ds.capture_sides([prepared("before"),
                                           prepared("after", "both")]),
                         {"before", "after"})

    def test_soft_arrival_keeps_nothing(self):
        # the take delivers the ramped frames itself; the target enters
        # past its decode edge
        self.assertEqual(ds.capture_sides([prepared("after", "both", **RAMP)]),
                         set())

    def test_guided_and_pixel_windows_keep_nothing(self):
        self.assertEqual(ds.capture_sides([prepared("before", "guide")]), set())
        self.assertEqual(ds.capture_sides(
            [prepared("before", kind="clip_pixels")]), set())

    def test_window_too_short(self):
        self.assertEqual(ds.capture_sides(
            [prepared("before", covered=ds.END_SWAP + ds.VERIFY - 1)]), set())


class Slices(unittest.TestCase):
    def test_head_frames_are_the_end_of_the_head_window(self):
        raw = torch.arange(60).float().view(60, 1, 1, 1)
        got = ds.head_frames(raw, 39)
        self.assertEqual(got[:, 0, 0, 0].tolist(), list(range(31, 39)))

    def test_tail_frames_are_the_start_of_the_tail_window(self):
        raw = torch.arange(60).float().view(60, 1, 1, 1)
        got = ds.tail_frames(raw, 39)          # window = raw 21..59
        self.assertEqual(got[:, 0, 0, 0].tolist(), list(range(21, 30)))


def header(self_id, delivered, pins=(), head=0, tail=0):
    return {"self_id": self_id, "relation": "",
            "delivered_frames": str(delivered),
            "pinned_head_frames": str(head), "pinned_tail_frames": str(tail),
            "pins": json.dumps(list(pins))}


def pin(source_id, place, start, mode="masked", frames=39, **shape):
    return dict({"source_id": source_id, "source_kind": "clip",
                 "place": place, "mode": mode, "source_start": start,
                 "source_frames": frames}, **shape)


def entry(name, hdr, enter=0, exit=None):
    return {"clip": name, "path": name, "header": hdr, "enter": enter,
            "exit": exit}


class Plan(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(ds, "_stored", lambda path, tail: 9)
        p.start()
        self.addCleanup(p.stop)
        self.A = header("A", 124)
        self.B = header("B", 102, [pin("A", "before", 85)], head=39)

    def test_natural_extend_swaps_the_parents_end(self):
        got = ds.plan([entry("a", self.A), entry("b", self.B)])
        self.assertEqual(got, {0: [("end", ds.END_SWAP, "b")]})

    def test_cut_moves_the_join_off_the_edge(self):
        self.assertEqual(ds.plan([entry("a", self.A, exit=100),
                                  entry("b", self.B)]), {})
        self.assertEqual(ds.plan([entry("a", self.A),
                                  entry("b", self.B, enter=5)]), {})

    def test_parent_with_a_trimmed_tail_is_not_at_its_edge(self):
        A = header("A", 124, tail=26)
        self.assertEqual(ds.plan([entry("a", A), entry("b", self.B)]), {})

    def test_no_stored_frames_no_swap(self):
        with mock.patch.object(ds, "_stored", lambda path, tail: 0):
            self.assertEqual(ds.plan([entry("a", self.A),
                                      entry("b", self.B)]), {})

    def test_hard_prepend_swaps_the_targets_start(self):
        T = header("T", 124)
        P = header("P", 100, [pin("T", "after", 0, mode="both")], tail=39)
        got = ds.plan([entry("p", P), entry("t", T)])
        self.assertEqual(got, {1: [("start", ds.START_SWAP, "p")]})

    def test_soft_prepend_enters_past_the_edge(self):
        T = header("T", 124)
        P = header("P", 113, [pin("T", "after", 0, mode="both", **RAMP)],
                   tail=26)
        e = [entry("p", P), entry("t", T)]
        # the target enters where the lineage says, past the ramped frames
        e[1]["enter"] = ds_enter(e)
        self.assertGreater(e[1]["enter"], 0)
        self.assertEqual(ds.plan(e), {})

    def test_bridge_loop_swaps_both_edges_of_the_root(self):
        A = header("A", 90)
        B = header("B", 113, [pin("A", "before", 51),
                              pin("A", "after", 0, mode="both")],
                   head=39, tail=39)
        got = ds.plan([entry("a", A), entry("b", B)], loop=True)
        self.assertEqual(got, {0: [("end", ds.END_SWAP, "b"),
                                   ("start", ds.START_SWAP, "b")]})
        # not looping: the wrap is not a join
        self.assertEqual(ds.plan([entry("a", A), entry("b", B)]),
                         {0: [("end", ds.END_SWAP, "b")]})

    def test_gaps_are_not_joins(self):
        gap = {"gap": 12, "clip": None, "path": None, "header": None,
               "enter": 0, "exit": None}
        self.assertEqual(ds.plan([entry("a", self.A), gap,
                                  entry("b", self.B)]), {})


def ds_enter(e):
    from obvpm_tl_test import nodes_assemble as na
    return na._derive_seam(e[0], e[1])[1]


def planes(n, h=4, w=6, base=0):
    y = (np.arange(n, dtype=np.uint8)[:, None, None] * 3 + base) \
        * np.ones((1, h, w), np.uint8)
    u = np.full((n, h // 2, w // 2), 128, np.uint8)
    return y, u, u.copy()


class Swap(unittest.TestCase):
    def _stored(self, got):
        return mock.patch.object(ds, "_blob_file",
                                 lambda path, tail, reader: got)

    def test_end_swap_aligns_at_the_join(self):
        clip = planes(20)
        stored = [p.copy() for p in planes(8, base=3 * 12)]  # = clip 12..19
        stored[0][4:] = 200                                    # the new edge
        with self._stored(tuple(stored)):
            ranges = ds.apply_planes(clip, [("end", 4, "b")], 0, 20)
        self.assertEqual(ranges, [(16, 20)])
        self.assertTrue((clip[0][16:] == 200).all())
        self.assertEqual(int(clip[0][15, 0, 0]), 45)          # untouched

    def test_start_swap_aligns_at_the_join(self):
        clip = planes(20)
        stored = [p.copy() for p in planes(9)]                 # = clip 0..8
        stored[0][:5] = 7
        with self._stored(tuple(stored)):
            ranges = ds.apply_planes(clip, [("start", 5, "p")], 0, 20)
        self.assertEqual(ranges, [(0, 5)])
        self.assertTrue((clip[0][:5] == 7).all())
        self.assertEqual(int(clip[0][5, 0, 0]), 15)

    def test_disagreeing_verify_frames_refuse_the_swap(self):
        clip = planes(20)
        before = clip[0].copy()
        stored = [p.copy() for p in planes(8, base=3 * 12)]
        stored[0][:4] = 0                                      # not this clip
        with self._stored(tuple(stored)):
            ranges = ds.apply_planes(clip, [("end", 4, "b")], 0, 20)
        self.assertEqual(ranges, [])
        self.assertTrue((clip[0] == before).all())

    def test_size_mismatch_refuses_the_swap(self):
        clip = planes(20)
        with self._stored(planes(8, h=8, w=8)):
            self.assertEqual(ds.apply_planes(clip, [("end", 4, "b")], 0, 20), [])

    def test_rgb_path_matches(self):
        imgs = torch.arange(20).float().view(20, 1, 1, 1).expand(20, 2, 2, 3) / 255.0
        imgs = imgs.clone()
        stored = (torch.arange(12, 20).float().view(8, 1, 1, 1)
                  .expand(8, 2, 2, 3) / 255.0).clone()
        stored[4:] = 1.0
        with self._stored(stored):
            ranges = ds.apply_images(imgs, [("end", 4, "b")], 0, 20, None)
        self.assertEqual(ranges, [(16, 20)])
        self.assertTrue(bool((imgs[16:] == 1.0).all()))


if __name__ == "__main__":
    unittest.main()
