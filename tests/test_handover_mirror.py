"""web/h3_handover.js must give the numbers nodes_masked.handover_frames gives.

The strip derives multi-pin seams in the browser from this arithmetic and
the server derives them again for the build; a drift between the two is a
quick preview that plays a join the export does not have (seen live on a
bridge, 2026-09-16). Runs the JS under node over a grid of window sizes,
places and mask shapes and demands identical integers. Skips without node.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
from obvpm_tl_test import nodes_masked as nm
from obvpm_tl_test import frames as fr
GRID = [(covered, place, ramp, edge, deep, decode_edge)
        for covered in (1, 5, 22, 39, 56, 90, 100)
        for place in ("before", "after")
        for ramp in (0, 3, 6, 10, 14, 19, 23, 40)
        for edge, deep in ((0.0, 0.0), (0.4, 0.0), (1.0, 0.0), (0.4, 0.3),
                           (0.0, 1.0), (0.2, 1.0))
        for decode_edge in (False, True)]

SCRIPT = """
import { readFileSync } from "node:fs";
import { handoverFrames, frameAtLatent, stepsForFrames } from %s;
// the grid comes in a file: on the command line it outgrows Windows'
// 32k argument limit
const grid = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = grid.map(([c, p, r, e, d, x]) => handoverFrames(c, p, r, e, d, x));
const frames = [0, 1, 2, 5, 6, 7, 12, 37, 41, 72].map((k) => frameAtLatent(k));
const steps = [0, 1, 5, 9, 13, 17, 22, 39, 56, 90, 100, 141].map((n) => stepsForFrames(n));
process.stdout.write(JSON.stringify({ out, frames, steps }));
"""


class HandoverMirrorTests(unittest.TestCase):
    def test_js_handover_matches_python(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH")
        js_path = os.path.join(ROOT, "web", "h3_handover.js").replace("\\", "/")
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "probe.mjs")
            with open(script, "w", encoding="utf-8") as f:
                f.write(SCRIPT % json.dumps("file:///" + js_path))
            grid = os.path.join(tmp, "grid.json")
            with open(grid, "w", encoding="utf-8") as f:
                json.dump(GRID, f)
            got = json.loads(subprocess.check_output(
                [node, script, grid], text=True))
        want = [nm.handover_frames(c, p, r, e, d, decode_edge=x)
                for c, p, r, e, d, x in GRID]
        bad = [(g, w, v) for g, w, v in zip(GRID, want, got["out"]) if w != v]
        self.assertEqual(bad, [], "JS handover drifted from Python at: %s" % bad[:6])
        self.assertEqual(got["frames"],
                         [fr.frame_at_latent(k) for k in (0, 1, 2, 5, 6, 7, 12, 37, 41, 72)])
        self.assertEqual(got["steps"],
                         [fr.steps_for_frames(n) for n in (0, 1, 5, 9, 13, 17, 22, 39, 56, 90, 100, 141)])


class DecodeEdgeHandover(unittest.TestCase):
    """The recipe flag moves a HARD hold's join one latent step inside
    the window -- and nothing else."""

    def test_hard_extend_hands_over_at_the_last_step(self):
        self.assertEqual(nm.handover_frames(39, "before"), 39)
        self.assertEqual(nm.handover_frames(39, "before", decode_edge=True), 35)
        self.assertEqual(nm.handover_frames(90, "before", decode_edge=True), 86)

    def test_hard_arrival_hands_over_past_the_first_two_steps(self):
        self.assertEqual(nm.handover_frames(39, "after"), 0)
        self.assertEqual(nm.handover_frames(39, "after", decode_edge=True), 5)

    def test_soft_holds_are_unchanged(self):
        for place in ("before", "after"):
            for shape in ((10, 0.4, 0.0), (23, 0.6, 0.0)):
                self.assertEqual(
                    nm.handover_frames(39, place, *shape, decode_edge=True),
                    nm.handover_frames(39, place, *shape), (place, shape))

    def test_open_and_partial_windows_are_unchanged(self):
        for place in ("before", "after"):
            for shape in ((0, 0.0, 1.0), (0, 0.0, 0.3)):
                self.assertEqual(
                    nm.handover_frames(39, place, *shape, decode_edge=True),
                    nm.handover_frames(39, place, *shape), (place, shape))

    def test_handover_lands_on_a_latent_step(self):
        for covered in (22, 39, 56, 90, 141):
            for place in ("before", "after"):
                h = nm.handover_frames(covered, place, decode_edge=True)
                self.assertIsNotNone(fr.steps_for_frames(h) if h else 0,
                                     (covered, place, h))


if __name__ == "__main__":
    unittest.main()
