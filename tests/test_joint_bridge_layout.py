"""A bridge that leaves a CUT parent must land on the latent grid, and the
rows it plays must be its own.

Seen live 2026-09-19 (demo: clip 5 | clip 6 | clip 2). Clip 6 bridges from
clip 5 -- cut at frame 107 of its 115 -- onto clip 2, which it hands over
to at frame 25. Neither number is in the sequence text: the widget writes
a marker only for a cut made by hand, and these come from the takes'
recipes. The joint placed the two-pin bridge "by the cut" from the text
alone, so it sat at frame 76 instead of 68 (off the grid, refused), and
once placed right it still lost its first ten steps to clip 5's cut-away
tail, because ownership knew no junction for a two-pin take either.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import importlib
import json
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
nj = importlib.import_module("obvpm_tl_test.nodes_joint")
fr = importlib.import_module("obvpm_tl_test.frames")

RAMP = {"mask_ramp_frames": 10, "mask_ramp_edge": 0.4}


def head(self_id, raw, head_f, tail_f, pins, relation="", parent="", join=0):
    return {"self_id": self_id, "raw_frames": str(raw),
            "delivered_frames": str(raw - head_f - tail_f),
            "pinned_head_frames": str(head_f),
            "pinned_tail_frames": str(tail_f),
            "relation": relation, "parent_id": parent,
            "parent_join_frame": str(join), "pins": json.dumps(pins)}


def pin(source, place, start, mode, **shape):
    return dict({"source_id": source, "source_kind": "clip", "place": place,
                 "source_start": start, "source_frames": 39, "mode": mode},
                **shape)


def demo():
    # clip 5 prepends a clip that is NOT on this timeline
    c5 = head("C5", 141, 0, 26, [pin("ELSEWHERE", "after", 0, "both", **RAMP)],
              "prepends", "ELSEWHERE", 13)
    # the bridge: leaves clip 5 at raw 68..107, arrives on clip 2 at 51..90
    c6 = head("C6", 192, 39, 26, [pin("C5", "before", 68, "masked"),
                                  pin("C2", "after", 51, "both", **RAMP)])
    # clip 2 extends a clip that is not on this timeline either
    c2 = head("C2", 141, 39, 0, [pin("ELSEWHERE", "before", 68, "masked")],
              "extends", "ELSEWHERE", 107)
    return ["demo/c5.mp4", "demo/c6.mp4", "demo/c2.mp4"], [c5, c6, c2]


class BridgeFromACut(unittest.TestCase):
    def test_the_cut_is_derived_where_the_text_is_silent(self):
        lines, headers = demo()
        self.assertEqual(nj.shown_cuts(lines, headers),
                         [[None, 107], [None, None], [25, None]])

    def test_a_marker_written_by_hand_wins(self):
        lines, headers = demo()
        lines[0] = "demo/c5.mp4 @ ..90"
        lines[2] = "demo/c2.mp4 @ 30"
        cuts = nj.shown_cuts(lines, headers)
        self.assertEqual(cuts[0], [None, 90])
        self.assertEqual(cuts[2], [30, None])

    def test_everything_lands_on_the_latent_grid(self):
        lines, headers = demo()
        starts = nj.raw_starts(lines, headers)
        self.assertEqual(starts, [0, 68, 170])
        for s in starts:
            self.assertIsNotNone(fr.steps_for_frames(s))

    def test_each_clip_owns_the_rows_it_plays(self):
        lines, headers = demo()
        layout = nj.timeline_layout(lines, headers, [235, 320, 235])
        owner = layout["owner_steps"]
        # clip 5 plays frames 0..107, the bridge 107..234, clip 2 from 234
        a = fr.steps_for_frames(107)
        b = fr.steps_for_frames(234)
        self.assertEqual((a, b), (32, 69))
        self.assertEqual(set(owner[:a]), {0})
        self.assertEqual(set(owner[a:b]), {1})
        self.assertEqual(set(owner[b:]), {2})
        # and nothing is left unowned
        self.assertNotIn(-1, owner)

    def test_a_head_held_from_elsewhere_is_not_checked_as_exact(self):
        lines, headers = demo()
        layout = nj.timeline_layout(lines, headers, [235, 320, 235])
        self.assertEqual(layout["hard_hold"], [False, False, False])


class PlainChainsAreUnchanged(unittest.TestCase):
    def test_an_extend_from_the_clip_end(self):
        root = head("A", 141, 0, 0, [])
        child = head("B", 141, 39, 0, [pin("A", "before", 102, "masked")],
                     "extends", "A", 141)
        lines = ["p/a.mp4", "p/b.mp4"]
        self.assertEqual(nj.raw_starts(lines, [root, child]), [0, 102])
        layout = nj.timeline_layout(lines, [root, child], [235, 235])
        join = fr.steps_for_frames(141)
        self.assertEqual(set(layout["owner_steps"][:join]), {0})
        self.assertEqual(set(layout["owner_steps"][join:]), {1})
        self.assertEqual(layout["hard_hold"], [False, True])


if __name__ == "__main__":
    unittest.main()
