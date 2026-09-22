"""H3 VRAM Headroom: sized from the run, raised for the run, put back after.

The node exists because a spill is silent -- so the things worth pinning
are the ones that would fail silently too: tokens counted from the wrong
dims, a headroom LOWERED below what the user launched with, and a headroom
left raised after an interrupted run (which would quietly slow every
later run in the session).

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import os
import sys
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
from obvpm_tl_test import nodes_budget as nb
from obvpm_tl_test import nodes_joint as nj
def lat(t, h, w):
    return {"latent": torch.zeros(1, 24, t, h, w)}


class FakeControl:
    def __init__(self, now):
        self.now = now
        self.calls = []

    def get_simple_vram_headroom(self):
        return self.now

    def set_simple_vram_headroom(self, v):
        self.calls.append(v)
        self.now = v


class Tokens(unittest.TestCase):
    def test_a_both_mode_window_at_960x544(self):
        # 39 frames = 12 latent steps on a 34x60 grid -> 17x30 patches
        conds = {"positive": [{"minimax_keyframes": [lat(12, 34, 60)]}]}
        self.assertEqual(nb.extra_tokens(conds), (12 * 17 * 30, 0))

    def test_references_and_keyframes_are_counted_apart(self):
        conds = {"positive": [{"minimax_keyframes": [lat(12, 34, 60)],
                               "minimax_refs": [lat(1, 34, 60),
                                                lat(1, 33, 59)]}]}
        # the odd-sized reference pads up to whole patches: 17 x 30
        self.assertEqual(nb.extra_tokens(conds), (6120, 510 + 17 * 30))

    def test_the_larger_conditioning_set_wins(self):
        conds = {"positive": [{"minimax_refs": [lat(1, 34, 60)]}],
                 "negative": [{}]}
        self.assertEqual(nb.extra_tokens(conds), (0, 510))

    def test_entries_without_a_picture_cost_nothing(self):
        conds = {"positive": [{"minimax_refs": [{"audio_latent": 1}, None],
                               "minimax_keyframes": None}]}
        self.assertEqual(nb.extra_tokens(conds), (0, 0))

    def test_bytes(self):
        conds = {"positive": [{"minimax_keyframes": [lat(12, 34, 60)]}]}
        want, kf, refs = nb.headroom_bytes(3.0, conds)
        extra = 6120 * nj._ACTIVATION_MB_PER_TOKEN * 1024 * 1024
        self.assertEqual((kf, refs), (6120, 0))
        self.assertEqual(want, int(3.0 * nb.GB + extra))
        # the switch for the automatic part leaves the asked amount alone
        self.assertEqual(nb.headroom_bytes(3.0, conds, False)[0],
                         int(3.0 * nb.GB))


class RaiseAndRestore(unittest.TestCase):
    def setUp(self):
        self.state = nb._Headroom()
        self.control = FakeControl(now=256 * 1024 * 1024)
        self.state._aimdo = lambda: self.control

    def test_raised_then_put_back(self):
        self.assertEqual(self.state.raise_to(3 * nb.GB), "dynamic")
        self.assertEqual(self.control.now, 3 * nb.GB)
        self.state.restore()
        self.assertEqual(self.control.now, 256 * 1024 * 1024)

    def test_never_lowers_what_the_user_launched_with(self):
        self.control.now = 6 * nb.GB          # --reserve-vram 6
        self.assertIsNone(self.state.raise_to(3 * nb.GB))
        self.assertEqual(self.control.calls, [])
        self.state.restore()                  # nothing to put back
        self.assertEqual(self.control.now, 6 * nb.GB)

    def test_restore_is_safe_twice(self):
        self.state.raise_to(3 * nb.GB)
        self.state.restore()
        self.state.restore()
        self.assertEqual(self.control.calls,
                         [3 * nb.GB, 256 * 1024 * 1024])

    def test_a_second_raise_does_not_lose_the_original(self):
        self.state.raise_to(3 * nb.GB)
        self.state.raise_to(5 * nb.GB)
        self.state.restore()
        self.assertEqual(self.control.now, 256 * 1024 * 1024)


class Wrappers(unittest.TestCase):
    def setUp(self):
        self.control = FakeControl(now=256 * 1024 * 1024)
        self._saved = nb._STATE
        nb._STATE = nb._Headroom()
        nb._STATE._aimdo = lambda: self.control

    def tearDown(self):
        nb._STATE = self._saved

    def run_once(self, options, fail=False):
        seen = {}

        def prepare(model, noise_shape, conds, **kw):
            seen["during"] = self.control.now
            if fail:
                raise KeyboardInterrupt
            return "prepared"

        def outer(*a, **k):
            return nb.prepare_sampling_with_headroom(
                prepare, "model", [1, 1, 8], {"positive": [{}]},
                model_options={nb.OPTION_KEY: options} if options else {})

        try:
            out = nb.outer_sample_restoring_headroom(outer)
        except KeyboardInterrupt:
            out = "interrupted"
        return out, seen["during"]

    def test_raised_during_the_run_and_restored_after(self):
        out, during = self.run_once({"headroom_gb": 4.0})
        self.assertEqual(out, "prepared")
        self.assertEqual(during, 4 * nb.GB)
        self.assertEqual(self.control.now, 256 * 1024 * 1024)

    def test_restored_after_an_interrupt(self):
        out, during = self.run_once({"headroom_gb": 4.0}, fail=True)
        self.assertEqual(out, "interrupted")
        self.assertEqual(during, 4 * nb.GB)
        self.assertEqual(self.control.now, 256 * 1024 * 1024)

    def test_a_model_without_the_node_is_left_alone(self):
        out, during = self.run_once(None)
        self.assertEqual(out, "prepared")
        self.assertEqual(self.control.calls, [])


class Node(unittest.TestCase):
    def test_off_is_a_pass_through(self):
        model = object()
        out, = nb.H3VramHeadroom().patch(model, False, 3.0, True)
        self.assertIs(out, model)


if __name__ == "__main__":
    unittest.main()
