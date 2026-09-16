"""web/h3_handover.js must give the numbers nodes_masked.handover_frames gives.

The strip derives multi-pin seams in the browser from this arithmetic and
the server derives them again for the build; a drift between the two is a
quick preview that plays a join the export does not have (seen live on a
bridge, 2026-09-16). Runs the JS under node over a grid of window sizes,
places and mask shapes and demands identical integers. Skips without node.

Run from the pack folder:
  C:/AI/ComfyUI/ComfyUI_windows_portable/python_embeded/python.exe -s -m unittest discover -s tests -v
"""
import importlib
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
nm = importlib.import_module("obvpm_tl_test.nodes_masked")
fr = importlib.import_module("obvpm_tl_test.frames")

GRID = [(covered, place, ramp, edge, deep)
        for covered in (1, 5, 22, 39, 56, 90, 100)
        for place in ("before", "after")
        for ramp in (0, 3, 6, 10, 14, 19, 23, 40)
        for edge, deep in ((0.0, 0.0), (0.4, 0.0), (1.0, 0.0), (0.4, 0.3),
                           (0.0, 1.0), (0.2, 1.0))]

SCRIPT = """
import { handoverFrames, frameAtLatent, stepsForFrames } from %s;
const grid = JSON.parse(process.argv[2]);
const out = grid.map(([c, p, r, e, d]) => handoverFrames(c, p, r, e, d));
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
            got = json.loads(subprocess.check_output(
                [node, script, json.dumps(GRID)], text=True))
        want = [nm.handover_frames(c, p, r, e, d) for c, p, r, e, d in GRID]
        bad = [(g, w, v) for g, w, v in zip(GRID, want, got["out"]) if w != v]
        self.assertEqual(bad, [], "JS handover drifted from Python at: %s" % bad[:6])
        self.assertEqual(got["frames"],
                         [fr.frame_at_latent(k) for k in (0, 1, 2, 5, 6, 7, 12, 37, 41, 72)])
        self.assertEqual(got["steps"],
                         [fr.steps_for_frames(n) for n in (0, 1, 5, 9, 13, 17, 22, 39, 56, 90, 100, 141)])


if __name__ == "__main__":
    unittest.main()
