"""Looping cuts: the `loop` directive and the wrap seam it adds.

Runs against the real modules with ComfyUI on sys.path (folder_paths and
torch import for real); the output folder and the sidecar reader are
patched, so no clip files or sidecars are needed.

Run from the pack folder:
  C:/AI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe -s -m unittest discover -s tests -v
"""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)

pack = types.ModuleType("obvpm_tl_test")
pack.__path__ = [ROOT]
sys.modules[pack.__name__] = pack
na = importlib.import_module("obvpm_tl_test.nodes_assemble")
levellock = importlib.import_module("obvpm_tl_test.levellock")
preview_route = importlib.import_module("obvpm_tl_test.preview_route")
import folder_paths  # noqa: E402  (ComfyUI's, on the path above)


def _header(self_id, delivered, pins=(), pinned_head=0):
    return {"self_id": self_id, "relation": "", "parent_id": "",
            "parent_join_frame": 0, "pinned_head_frames": pinned_head,
            "pinned_tail_frames": 0, "delivered_frames": delivered,
            "pins": json.dumps(list(pins))}


# Clip A, 100 delivered frames, a root. Take T is the take that LOOPS
# it: a bridge extending A's tail (before-pin) and prepending into A's
# head (after-pin) in one run. The after-pin is held softly so the
# junction is not the window edge -- the wrap must move A's enter.
A = _header("A", 100)
T = _header("T", 60, pins=[
    {"source_id": "A", "source_kind": "clip", "source_start": 61,
     "source_frames": 39, "place": "before", "audio_window": 0,
     "mode": "masked"},
    {"source_id": "A", "source_kind": "clip", "source_start": 0,
     "source_frames": 39, "place": "after", "audio_window": 0,
     "mode": "both", "mask_ramp_frames": 12, "mask_ramp_edge": 0.4},
])
HEADERS = {"a.mp4": A, "t.mp4": T}


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name in HEADERS:
            open(os.path.join(self.tmp.name, name), "wb").close()
        self.patches = [
            patch.object(folder_paths, "get_output_directory",
                         lambda: self.tmp.name),
            patch.object(na, "_read_verified_header",
                         lambda p: HEADERS[os.path.basename(p)]),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    # -- the directive -------------------------------------------------

    def test_directive_is_read_and_is_not_an_entry(self):
        self.assertTrue(na.sequence_loops("loop\na.mp4"))
        self.assertTrue(na.sequence_loops("a.mp4\n  LOOP  \n"))
        self.assertFalse(na.sequence_loops("a.mp4\n# loop\n"))
        self.assertFalse(na.sequence_loops("loop.mp4"))
        self.assertEqual([e["clip"] for e in na._parse_sequence("loop\na.mp4\nt.mp4")],
                         ["a.mp4", "t.mp4"])

    # -- the wrap seam -------------------------------------------------

    def test_wrap_seam_is_the_linear_seam_of_take_before_clip(self):
        looped = na.resolve_sequence("loop\na.mp4\nt.mp4")
        linear = na.resolve_sequence("t.mp4\na.mp4")
        # entry 0 is now the CHILD of the take at the end: it enters
        # past the frames the take delivers itself, exactly as it would
        # if the take sat before it on an open timeline
        self.assertGreater(linear[1]["enter"], 0)
        self.assertEqual(looped[0]["enter"], linear[1]["enter"])
        self.assertEqual(looped[1]["exit"], linear[0]["exit"])
        # ...and the take still extends A on its left, as before
        self.assertEqual(looped[0]["exit"],
                         na.resolve_sequence("a.mp4\nt.mp4")[0]["exit"])

    def test_without_the_directive_the_ends_stay_open(self):
        plain = na.resolve_sequence("a.mp4\nt.mp4")
        self.assertEqual(plain[0]["enter"], 0)
        self.assertIsNone(plain[1]["exit"])

    def test_manual_cut_beats_the_wrap(self):
        looped = na.resolve_sequence("loop\na.mp4 @ 5\nt.mp4")
        self.assertEqual(looped[0]["enter"], 5)

    def test_single_clip_loops_onto_itself_as_a_butt_join(self):
        one = na.resolve_sequence("loop\na.mp4")
        self.assertEqual(one[0]["enter"], 0)
        self.assertIsNone(one[0]["exit"])

    def test_gap_at_either_end_cannot_wrap(self):
        with self.assertRaisesRegex(ValueError, "empty space"):
            na.resolve_sequence("loop\na.mp4\n~ 20")
        with self.assertRaisesRegex(ValueError, "empty space"):
            na.resolve_sequence("loop\n~ 20\na.mp4")
        # a gap in the MIDDLE is fine: it breaks that chain, not the wrap
        na.resolve_sequence("loop\na.mp4\n~ 20\nt.mp4")

    def test_is_changed_skips_gaps(self):
        na.H3Assemble.IS_CHANGED("~ 10\na.mp4")

    # -- the repairs see the wrap as a join ----------------------------

    def test_join_pairs_add_the_wrap_once(self):
        es = [{}, {}, {}]
        self.assertEqual(levellock.join_pairs(es), [(0, 1), (1, 2)])
        self.assertEqual(levellock.join_pairs(es, loop=True),
                         [(0, 1), (1, 2), (2, 0)])
        self.assertEqual(levellock.join_pairs([{}], loop=True), [(0, 0)])
        self.assertEqual(levellock.join_pairs([], loop=True), [])

    def test_level_lock_plans_the_wrap_on_entry_zero(self):
        es = na.resolve_sequence("loop\na.mp4\nt.mp4")
        seen = []

        def fake_measure(*args, **kwargs):
            seen.append((args[0], args[2]))
            return None                     # nothing to correct
        with patch.object(levellock, "measure", fake_measure):
            levellock.plan(es, loop=True)
        self.assertEqual([tuple(os.path.basename(p) for p in pair)
                          for pair in seen],
                         [("a.mp4", "t.mp4"), ("t.mp4", "a.mp4")])

    # -- the joint render keeps the cut's wrap ---------------------------

    def test_joint_render_crops_to_the_wrap(self):
        nj = importlib.import_module("obvpm_tl_test.nodes_joint")
        # the joint's own parser must skip the directive like the cut's
        self.assertEqual(nj.parse_sequence("loop\na.mp4\nt.mp4"),
                         ["a.mp4", "t.mp4"])
        self.assertEqual(nj.parse_sequence_lines("LOOP\na.mp4 @ 5\nt.mp4"),
                         ["a.mp4 @ 5", "t.mp4"])
        # A at raw start 0 with no head; T (60 raw frames, 25 delivered,
        # 39-frame pinned head) at raw start 61 -- its delivered frames
        # sit at 61 + 39 + f. The joint runs to 141 frames, so T's tail
        # rows past its delivered end are the opening again.
        layout = {"starts": [0, 61], "heads": [0, 39], "total_frames": 141}
        headers = [{"delivered_frames": 100}, {"delivered_frames": 25}]
        entries = [{"enter": 13, "exit": None}, {"enter": 0, "exit": None}]
        self.assertEqual(nj.wrap_keep_frames(layout, entries, headers),
                         [13, 61 + 39 + 25])
        # a manual exit on the last entry is honoured; a hi past the
        # joint is clamped; nothing left is refused
        entries[-1]["exit"] = 20
        self.assertEqual(nj.wrap_keep_frames(layout, entries, headers)[1], 120)
        layout["total_frames"] = 110
        self.assertEqual(nj.wrap_keep_frames(layout, entries, headers)[1], 110)
        with self.assertRaisesRegex(ValueError, "leaves nothing"):
            nj.wrap_keep_frames(layout, [{"enter": 200, "exit": None},
                                         entries[-1]], headers)

    def test_crop_blocks_slices_across_block_edges(self):
        import torch
        nj = importlib.import_module("obvpm_tl_test.nodes_joint")
        blocks = [torch.arange(10).view(10, 1, 1, 1),
                  torch.arange(10, 20).view(10, 1, 1, 1),
                  torch.arange(20, 30).view(10, 1, 1, 1)]
        got = torch.cat(list(nj.crop_blocks(iter(blocks), 7, 23))).flatten()
        self.assertEqual(got.tolist(), list(range(7, 23)))
        # a block wholly inside is passed through, wholly outside dropped
        got = torch.cat(list(nj.crop_blocks(iter(blocks), 10, 20))).flatten()
        self.assertEqual(got.tolist(), list(range(10, 20)))
        self.assertEqual(list(nj.crop_blocks(iter(blocks), 30, 40)), [])

    def test_seam_plan_treats_entry_zero_as_the_wrap_child(self):
        es = na.resolve_sequence("loop\na.mp4\nt.mp4")
        base = {"level_lock": True, "crossfade": True, "audio_declick": True}
        open_ = preview_route._seam_plan(es, base)
        ring = preview_route._seam_plan(es, base, loop=True)
        # the take's arrival at A is a masked join, so entry 0's repairs
        # default off ONLY once the cut wraps
        self.assertTrue(open_[0]["level_lock"])
        self.assertFalse(ring[0]["level_lock"])
        self.assertFalse(ring[0]["crossfade"])
        self.assertTrue(ring[0]["audio_declick"])


if __name__ == "__main__":
    unittest.main()
