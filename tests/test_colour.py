"""Takes are BT.709: the matrix, TV range and all four colour tags.

Every take used to be converted with ffmpeg's default matrix (BT.601) and
carry no tags. ComfyUI decodes an untagged file as BT.601, so it looked
right there -- but browsers (the timeline's own preview), players and
editors assume BT.709 for HD and showed our takes shifted (measured
2026-09-26: green 255 -> 216 in Edge). Takes are now written like core's
Save Video. Pieces spliced into OLD takes are still written the old way,
because the tags are codec extradata and a splice needs it to match.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import os
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
import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from obvpm_tl_test import nodes_save as ns  # noqa: E402
from obvpm_tl_test import nodes_assemble as na  # noqa: E402
from obvpm_tl_test import yuv  # noqa: E402

RED = (1.0, 0.0, 0.0)


def _frames(rgb=RED, n=24, size=64):
    return torch.tensor(rgb).view(1, 1, 1, 3).expand(n, size, size, 3).clone()


def _tags(path):
    with av.open(path) as c:
        cc = c.streams.video[0].codec_context
        return (int(cc.color_range), int(cc.colorspace),
                int(cc.color_primaries), int(cc.color_trc))


def _yuv_at_centre(path):
    with av.open(path) as c:
        f = next(c.decode(video=0))
        planes = [np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)
                  for p in f.planes]
        h, w = f.height, f.width
        return (int(planes[0][h // 2, w // 2]), int(planes[1][h // 4, w // 4]),
                int(planes[2][h // 4, w // 4]))


def _rgb_at_centre(path):
    with av.open(path) as c:
        a = next(c.decode(video=0)).to_ndarray(format="rgb24")
        return tuple(int(x) for x in a[a.shape[0] // 2, a.shape[1] // 2])


# tag values (libavutil pixfmt.h): range 1 = MPEG/tv, space 1 = BT709,
# primaries 1 = BT709, trc 13 = IEC61966-2-1 (sRGB); 2 = unspecified
BT709_TAGS = (1, 1, 1, 13)


@unittest.skipUnless(ns._ffmpeg_exe(), "needs an ffmpeg binary")
class ColourTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _enc(self, name, colour=ns.COLOUR_BT709, rgb=RED):
        p = os.path.join(self.tmp.name, name)
        ns.H3SaveVideoWithMCtx._encode_mp4(p, _frames(rgb), None, 19,
                                           colour=colour)
        return p

    def test_a_take_is_bt709_with_all_four_tags(self):
        p = self._enc("new.mp4")
        self.assertEqual(_tags(p), BT709_TAGS)
        # pure red through the BT.709 matrix into TV range
        y, cb, cr = _yuv_at_centre(p)
        self.assertLessEqual(abs(y - 63), 1)
        self.assertLessEqual(abs(cb - 102), 1)
        self.assertLessEqual(abs(cr - 240), 1)
        self.assertEqual(ns.source_colour(p), ns.COLOUR_BT709)

    def test_the_legacy_encode_is_unchanged(self):
        p = self._enc("old.mp4", ns.COLOUR_LEGACY)
        self.assertEqual(_tags(p)[1:], (2, 2, 2))     # untagged
        y, cb, cr = _yuv_at_centre(p)
        self.assertLessEqual(abs(y - 81), 1)          # BT.601
        self.assertLessEqual(abs(cb - 90), 1)
        self.assertEqual(ns.source_colour(p), ns.COLOUR_LEGACY)

    def test_both_decode_back_to_the_colour_that_went_in(self):
        rgb = (34 / 255.0, 139 / 255.0, 34 / 255.0)   # foliage
        for colour in (ns.COLOUR_BT709, ns.COLOUR_LEGACY):
            got = _rgb_at_centre(self._enc(colour + ".mp4", colour, rgb))
            for a, b in zip(got, (34, 139, 34)):
                self.assertLessEqual(abs(a - b), 3, (colour, got))

    def test_planes_are_tagged_like_their_source_never_converted(self):
        y = np.full((12, 64, 64), 100, np.uint8)
        u = np.full((12, 32, 32), 90, np.uint8)
        v = np.full((12, 32, 32), 170, np.uint8)
        with self.assertRaisesRegex(ValueError, "source clip's colour"):
            yuv.encode(os.path.join(self.tmp.name, "x.mp4"), (y, u, v), 19)
        for colour in (ns.COLOUR_BT709, ns.COLOUR_LEGACY):
            p = os.path.join(self.tmp.name, "planes_%s.mp4" % colour)
            yuv.encode(p, (y, u, v), 19, colour=colour)
            self.assertEqual(ns.source_colour(p), colour)
            got = _yuv_at_centre(p)
            for a, b in zip(got, (100, 90, 170)):
                self.assertLessEqual(abs(a - b), 1, (colour, got))

    def test_pieces_splice_with_takes_of_their_own_kind_only(self):
        for colour in (ns.COLOUR_BT709, ns.COLOUR_LEGACY):
            take = self._enc("take_%s.mp4" % colour, colour)
            bridge = self._enc("bridge_%s.mp4" % colour,
                               ns.source_colour(take))
            out = os.path.join(self.tmp.name, "cut_%s.mp4" % colour)
            na._mux_pieces([{"path": take, "from": 0, "to": 24},
                            {"path": bridge, "from": 0, "to": 24}], out, None)
            with av.open(out) as c:
                self.assertEqual(sum(1 for _ in c.decode(video=0)), 48)
        old = os.path.join(self.tmp.name, "take_legacy.mp4")
        new = os.path.join(self.tmp.name, "take_bt709.mp4")
        with self.assertRaises(na._StreamMismatch):
            na._mux_pieces([{"path": old, "from": 0, "to": 24},
                            {"path": new, "from": 0, "to": 24}],
                           os.path.join(self.tmp.name, "mixed.mp4"), None)

    def test_the_in_process_fallback_tags_too(self):
        p = os.path.join(self.tmp.name, "pyav.mp4")
        ns._encode_with_pyav(p, _frames(), None, 19)
        self.assertEqual(_tags(p)[1:3], (1, 1))       # BT.709 matrix + primaries
        self.assertEqual(ns.source_colour(p), ns.COLOUR_BT709)
        self.assertLessEqual(abs(_yuv_at_centre(p)[0] - 63), 1)


if __name__ == "__main__":
    unittest.main()
