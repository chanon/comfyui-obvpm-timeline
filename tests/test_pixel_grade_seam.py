"""A window encoded from PIXELS is not a latent-identical join.

Seam repairs default off on a masked or both-mode join, because both
sides decode from the same latents and a "correction" there only corrects
noise. That reasoning does not hold when the clip being continued had no
sidecar: its window was VAE-encoded from the file's pixels, so the
timeline plays the original file on one side and the decode of a round
trip on the other. nodes_assemble.masked_continuation makes that one
exception, and these tests pin both halves of it -- the exception itself,
and that a latent-grade join is judged exactly as before.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
from obvpm_tl_test import nodes_assemble as na
from obvpm_tl_test import preview_route as pr
def header(self_id, pins=()):
    return {"self_id": self_id, "pins": json.dumps(list(pins))}


def pin(source_id, place, kind="clip", mode="masked", frames=39, **shape):
    return dict({"source_id": source_id, "source_kind": kind,
                 "place": place, "mode": mode, "source_start": 0,
                 "source_frames": frames}, **shape)


RAMP = {"mask_ramp_frames": 10, "mask_ramp_edge": 0.4}


class MaskedContinuation(unittest.TestCase):
    def test_latent_extend_is_latent_identical(self):
        for mode in ("masked", "both"):
            for shape in ({}, RAMP):
                child = header("B", [pin("A", "before", mode=mode, **shape)])
                self.assertTrue(
                    na.masked_continuation(header("A"), child),
                    (mode, shape))

    def test_pixel_extend_is_not(self):
        for mode in ("masked", "both"):
            for shape in ({}, RAMP):
                child = header("B", [pin("A", "before", kind="clip_pixels",
                                         mode=mode, **shape)])
                self.assertFalse(
                    na.masked_continuation(header("A"), child),
                    (mode, shape))

    def test_pixel_prepend_is_not(self):
        take = header("P", [pin("A", "after", kind="clip_pixels", **RAMP)])
        self.assertFalse(na.masked_continuation(take, header("A")))
        take = header("P", [pin("A", "after", **RAMP)])
        self.assertTrue(na.masked_continuation(take, header("A")))

    def test_bridge_is_judged_per_side(self):
        # arrives from a clip with latents, lands on one without
        bridge = header("M", [pin("A", "before", **RAMP),
                              pin("C", "after", kind="clip_pixels", **RAMP)])
        self.assertTrue(na.masked_continuation(header("A"), bridge))
        self.assertFalse(na.masked_continuation(bridge, header("C")))

    def test_the_next_generation_is_latent_again(self):
        # B extended a clip with no sidecar; C then extends B from B's
        # own latents, and that join is latent-identical as usual
        b = header("B", [pin("A", "before", kind="clip_pixels", **RAMP)])
        c = header("C", [pin("B", "before", **RAMP)])
        self.assertFalse(na.masked_continuation(header("A"), b))
        self.assertTrue(na.masked_continuation(b, c))


class SeamPlan(unittest.TestCase):
    BASE = {"level_lock": True, "crossfade": True}

    def plan(self, kind, seam_opts=None):
        entries = [
            {"header": header("A")},
            {"header": header("B", [pin("A", "before", kind=kind, **RAMP)]),
             "seam_opts": seam_opts or {}},
        ]
        return pr._seam_plan(entries, self.BASE)[1]

    def test_repairs_default_on_for_a_pixel_join_only(self):
        self.assertTrue(self.plan("clip_pixels")["level_lock"])
        self.assertTrue(self.plan("clip_pixels")["crossfade"])
        self.assertFalse(self.plan("clip")["level_lock"])
        self.assertFalse(self.plan("clip")["crossfade"])

    def test_a_join_can_still_say_otherwise(self):
        self.assertFalse(
            self.plan("clip_pixels", {"level_lock": False})["level_lock"])
        self.assertTrue(self.plan("clip", {"level_lock": True})["level_lock"])


class OverlapKept(unittest.TestCase):
    """The re-render is stored for a pixel-sourced window, and only then."""

    @staticmethod
    def wired(place, kind="clip", mode="masked"):
        return {"place": place,
                "spec": {"place": place, "source_kind": kind, "mode": mode}}

    def test_sides(self):
        from obvpm_tl_test import nodes_save as ns
        w = self.wired
        self.assertEqual(ns.rerendered_sides([w("before")]), set())
        self.assertEqual(ns.rerendered_sides([w("before", mode="both")]),
                         set())
        self.assertEqual(
            ns.rerendered_sides([w("before", kind="clip_pixels")]),
            {"before"})
        self.assertEqual(
            ns.rerendered_sides([w("after", kind="clip_pixels",
                                   mode="both")]), {"after"})
        # a bridge keeps the pixel side and not the latent one
        self.assertEqual(
            ns.rerendered_sides([w("before"),
                                 w("after", kind="clip_pixels")]),
            {"after"})
        # guided is kept as it always was
        self.assertEqual(ns.rerendered_sides([w("before", mode="guide")]),
                         {"before"})
        self.assertEqual(ns.guided_sides([w("before", kind="clip_pixels")]),
                         set())


class FadeAtACut(unittest.TestCase):
    """A pixel window sits wherever the footage was cut; a latent one
    keeps the strict rule it always had."""

    def setUp(self):
        from obvpm_tl_test import crossfade
        self.xf = crossfade
        self._available = self.xf.available
        self.xf.available = lambda path, tail=False: 39

    def tearDown(self):
        self.xf.available = self._available

    def join(self, kind, exit_f):
        # the parent has 200 frames; the child's window is [61, 100), so
        # the lineage puts the parent's exit at frame 100
        parent = dict(header("A"), delivered_frames="200",
                      pinned_head_frames="0")
        child = dict(header("B", [pin("A", "before", kind=kind)]),
                     relation="extends", parent_id="A",
                     parent_join_frame="100")
        left = {"clip": "a.mp4", "path": "a.mp4", "header": parent,
                "exit": exit_f}
        right = {"clip": "b.mp4", "path": "b.mp4", "header": child,
                 "enter": 0}
        return self.xf.usable(left, right)

    def test_pixel_join_fades_at_its_derived_exit(self):
        self.assertEqual(self.join("clip_pixels", 100), 39)

    def test_pixel_join_moved_by_hand_does_not(self):
        self.assertEqual(self.join("clip_pixels", 90), 0)

    def test_latent_join_at_a_cut_is_refused_as_before(self):
        self.assertEqual(self.join("clip", 100), 0)

    def test_fade_never_exceeds_what_the_parent_plays(self):
        parent = dict(header("A"), delivered_frames="200",
                      pinned_head_frames="0")
        child = dict(header("B", [pin("A", "before", kind="clip_pixels")]),
                     relation="extends", parent_id="A",
                     parent_join_frame="20")
        n = self.xf.usable(
            {"clip": "a.mp4", "path": "a.mp4", "header": parent, "exit": 20},
            {"clip": "b.mp4", "path": "b.mp4", "header": child, "enter": 0})
        self.assertEqual(n, 20)


if __name__ == "__main__":
    unittest.main()
