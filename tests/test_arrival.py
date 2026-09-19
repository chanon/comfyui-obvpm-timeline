"""The local-jump reading where an arriving window starts.

Run: python_embeded/python.exe -s -m unittest discover -s tests
"""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest

import av
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pack = types.ModuleType("obvpm_tl_arrival")
pack.__path__ = [ROOT]
sys.modules.setdefault(pack.__name__, pack)
sr = importlib.import_module(pack.__name__ + ".seam_report")

W, H, N, AT = 320, 192, 48, 24


def write_clip(path, jump):
    """A textured field drifting 1 px a frame with a bright square that
    drifts with it -- and, when `jump`, leaps 24 px going into frame AT."""
    rng = np.random.default_rng(7)
    field = rng.integers(40, 90, size=(H, W * 2), dtype=np.uint8)
    with av.open(path, "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "12"}
        for i in range(N):
            img = np.repeat(field[:, i:i + W, None], 3, axis=2).copy()
            x = 40 + i + (24 if jump and i >= AT else 0)
            img[80:112, x:x + 32] = 230
            for packet in stream.encode(
                    av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


class Arrival(unittest.TestCase):
    def test_a_local_leap_is_flagged_and_steady_motion_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            steady, leap = (os.path.join(tmp, n) for n in ("s.mp4", "l.mp4"))
            write_clip(steady, False)
            write_clip(leap, True)
            clean = sr.measure_arrival(steady, AT)
            flagged = sr.measure_arrival(leap, AT)
        self.assertEqual(clean["verdict"], "clean")
        self.assertLess(clean["ratio"], sr.ARRIVAL_BUMP)
        self.assertGreater(flagged["ratio"], sr.ARRIVAL_JUMP)
        self.assertEqual(flagged["verdict"], "local jump")
        self.assertEqual(flagged["frame"], AT)

    def test_no_reading_without_neighbours_on_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "s.mp4")
            write_clip(clip, False)
            self.assertIsNone(sr.measure_arrival(clip, 2))
            self.assertIsNone(sr.measure_arrival(clip, N - 1))

    def test_the_window_start_comes_from_the_recipe(self):
        pins = [{"place": "before", "source_frames": 39},
                {"place": "after", "source_frames": 39}]
        meta = {"raw_frames": "192", "pinned_head_frames": "39",
                "pins": json.dumps(pins)}
        self.assertEqual(sr.arrival_frame(meta), 114)
        meta["pins"] = json.dumps(pins[:1])
        self.assertIsNone(sr.arrival_frame(meta))


if __name__ == "__main__":
    unittest.main()
