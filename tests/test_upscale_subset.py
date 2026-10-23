"""Upscaling only some of the timeline: nodes_joint.upscale_subset.

The Timeline's upscale_skip widget names the clips an upscale Run leaves
out (part ids, else clip paths). The sequence that goes down the refine
branch keeps the others, each with the cut it plays in the FULL timeline
-- lineage-derived ends included, so a clip whose parent is skipped
still shows the same frames -- and splits into segments where a clip is
left out.

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
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
from obvpm_tl_test import nodes_joint as nj  # noqa: E402
from obvpm_tl_test import nodes_assemble as na  # noqa: E402


def head(self_id, raw, head_f=0, pins=(), relation="", parent="", join=0):
    return {"self_id": self_id, "raw_frames": str(raw),
            "delivered_frames": str(raw - head_f),
            "pinned_head_frames": str(head_f), "pinned_tail_frames": "0",
            "relation": relation, "parent_id": parent,
            "parent_join_frame": str(join), "pins": json.dumps(list(pins))}


def extend_of(parent_id, parent_raw, self_id, raw=141, covered=39):
    start = parent_raw - covered
    return head(self_id, raw, head_f=covered,
                pins=[{"source_id": parent_id, "source_kind": "clip",
                       "place": "before", "source_start": start,
                       "source_frames": covered, "mode": "masked"}],
                relation="extends", parent=parent_id, join=parent_raw)


# a.mp4 -> b.mp4 extends a -> c.mp4 extends b; d.mp4 unrelated
HEADERS = {
    "p/a.mp4": head("A", 141),
    "p/b.mp4": extend_of("A", 141, "B"),
    "p/c.mp4": extend_of("B", 141, "C"),
    "p/d.mp4": head("D", 90),
}


def subset(text, skip):
    with patch.object(nj, "clip_headers",
                      lambda clips: [HEADERS[c] for c in clips]):
        return nj.upscale_subset(text, set(skip))


def segments(text):
    items = nj.sequence_items(text)
    lines = [line for kind, line in items if kind == "clip"]
    headers = [HEADERS[nj.clip_of_line(line)] for line in lines]
    return [[nj.clip_of_line(lines[i]) for i in s["clips"]]
            for s in nj.split_segments(items, headers)]


def cuts(text):
    items = nj.sequence_items(text)
    lines = [line for kind, line in items if kind == "clip"]
    headers = [HEADERS[nj.clip_of_line(line)] for line in lines]
    return nj.shown_cuts(lines, headers, nj.derive_flags(items))


FULL = "p/a.mp4\np/b.mp4\np/c.mp4\n~ 24\np/d.mp4"


class SubsetTests(unittest.TestCase):
    def test_nothing_skipped_is_unchanged(self):
        self.assertEqual(subset(FULL, []), FULL)
        self.assertEqual(subset(FULL, ["p/zzz.mp4"]), FULL)

    def test_everything_skipped_is_refused(self):
        with self.assertRaises(ValueError):
            subset(FULL, ["p/a.mp4", "p/b.mp4", "p/c.mp4", "p/d.mp4"])

    def test_kept_clips_keep_their_cuts(self):
        full = dict(zip(["a", "b", "c", "d"], cuts(FULL)))
        out = subset(FULL, ["p/a.mp4"])
        got = dict(zip(["b", "c", "d"], cuts(out)))
        for k in ("b", "c", "d"):
            self.assertEqual([got[k][0] or 0, got[k][1]],
                             [full[k][0] or 0, full[k][1]], k)

    def test_a_skipped_middle_clip_splits_the_upscale(self):
        out = subset(FULL, ["p/b.mp4"])
        self.assertEqual(segments(out), [["p/a.mp4"], ["p/c.mp4"],
                                         ["p/d.mp4"]])

    def test_linked_clips_stay_one_segment(self):
        out = subset(FULL, ["p/d.mp4"])
        self.assertEqual(segments(out), [["p/a.mp4", "p/b.mp4", "p/c.mp4"]])
        self.assertNotIn("~", out)      # no gap left dangling at the end

    def test_gap_between_kept_clips_stays(self):
        out = subset(FULL, ["p/a.mp4"])
        self.assertIn("~ 24", out.splitlines())

    def test_part_ids_and_seam_settings(self):
        text = ("p/a.mp4 [id=aaaa]\np/b.mp4 [id=bbbb crossfade=on]\n"
                "p/b.mp4 @ 60 [id=cccc crossfade=on]")
        out = subset(text, ["bbbb"]).splitlines()
        self.assertEqual(len(out), 2)
        self.assertEqual(na.line_part(out[0]), "aaaa")
        self.assertEqual(na.line_part(out[1]), "cccc")
        # its join in front was with the skipped part: settings dropped
        self.assertEqual(na.split_seam_opts(out[1])[1], {})

    def test_loop_line_dropped(self):
        out = subset("loop\n" + FULL, ["p/d.mp4"])
        self.assertFalse(any(na.LOOP_RE.match(x) for x in out.splitlines()))


class SkipKeysTests(unittest.TestCase):
    def test_parse(self):
        k = na.H3Timeline._skip_keys
        self.assertEqual(k('["a", "p/b.mp4", 3, ""]'), {"a", "p/b.mp4"})
        self.assertEqual(k(""), set())
        self.assertEqual(k("not json"), set())
        self.assertEqual(k('{"a": 1}'), set())


if __name__ == "__main__":
    unittest.main()
