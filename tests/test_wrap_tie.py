"""A looping cut's opening sits on the joint twice; the refine keeps the
two copies identical.

The joint latent is a line. The take that closes a loop holds a window of
the first clip at its tail, so the opening is there twice -- copy A at the
start (what the file shows after the wrap) and copy B at the end (cropped
from the render). Refined as strangers they come out with different
detail, and the wrap jumps. These tests pin where the copies are found,
which one leads on each side of the handover, and -- with a toy sampler --
that the copies END identical although their noise never was.

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
from obvpm_tl_test import nodes_joint as nj
from obvpm_tl_test import frames as fr
def first_clip():
    return {"self_id": "FIRST", "raw_frames": "141", "pins": "[]"}


def loop_take(shape=None, source="FIRST", start=0):
    # a bridge: leaves the last clip, arrives on the first
    after = dict({"source_id": source, "source_kind": "clip",
                  "place": "after", "mode": "both", "source_start": start,
                  "source_frames": 39}, **(shape or {}))
    before = {"source_id": "LAST", "source_kind": "clip", "place": "before",
              "mode": "masked", "source_start": 102, "source_frames": 39}
    return {"self_id": "TAKE", "raw_frames": "141",
            "pins": json.dumps([before, after])}


def layout(take_start=102):
    # the first clip at frame 0; the take's raw frame 0 where its held
    # head overlays the clip before it
    total = take_start + 141
    return {"starts": [0, take_start],
            "total_steps": fr.steps_for_frames(total),
            "total_ticks": fr.audio_total(total)}


RAMP = {"mask_ramp_frames": 10, "mask_ramp_edge": 0.4}


class WhereTheCopiesAre(unittest.TestCase):
    def test_a_soft_hold(self):
        tie = nj.wrap_tie(layout(), [first_clip(), loop_take(RAMP)])
        # copy A: the first clip's frames 0..39 = steps 0..12
        # copy B: the take's last 39 frames, 204..243 = steps 60..72
        self.assertEqual((tie["a"], tie["b"], tie["n"]), (0, 60, 12))
        # a ramp of 10 hands over 13 frames in = 4 latent steps
        self.assertEqual(tie["hand"], 4)
        self.assertEqual(tie["ta"], 0)
        self.assertEqual(tie["tb"], fr.audio_total(204))
        self.assertEqual(tie["tn"], fr.audio_span(0, 39))

    def test_a_hard_hold_hands_over_at_the_window_start(self):
        tie = nj.wrap_tie(layout(), [first_clip(), loop_take()])
        self.assertEqual(tie["hand"], 0)
        self.assertEqual(tie["thand"], 0)

    def test_a_lone_clip_looping_onto_itself(self):
        clip = dict(first_clip())
        take = loop_take(RAMP)
        tie = nj.wrap_tie(layout(), [clip, take])
        self.assertIsNotNone(tie)

    def test_no_tie_when_the_last_clip_does_not_arrive_on_the_first(self):
        self.assertIsNone(nj.wrap_tie(
            layout(), [first_clip(), loop_take(RAMP, source="OTHER")]))
        self.assertIsNone(nj.wrap_tie(layout(), [first_clip(), first_clip()]))

    def test_off_the_latent_grid_is_refused_not_guessed(self):
        self.assertIsNone(nj.wrap_tie(
            layout(take_start=103), [first_clip(), loop_take(RAMP)]))


class WhichCopyLeads(unittest.TestCase):
    def test_before_the_handover_b_leads_and_after_it_a(self):
        x = torch.zeros(1, 1, 80, 1, 1)
        x[:, :, 0:12] = 1.0      # copy A
        x[:, :, 60:72] = 2.0     # copy B
        nj.H3WindowHandler._mirror(x, 2, 0, 60, 12, 4)
        flat = x.flatten()
        # the take re-drew the first four steps and delivers them
        self.assertTrue(bool((flat[0:4] == 2.0).all()))
        self.assertTrue(bool((flat[60:64] == 2.0).all()))
        # from the handover the first clip is what the cut shows
        self.assertTrue(bool((flat[4:12] == 1.0).all()))
        self.assertTrue(bool((flat[64:72] == 1.0).all()))
        # nothing else moved
        self.assertTrue(bool((flat[12:60] == 0.0).all()))

    def test_audio_runs_on_the_last_dim(self):
        x = torch.zeros(1, 2, 3, 100)
        x[..., 0:10] = 1.0
        x[..., 50:60] = 3.0
        nj.H3WindowHandler._mirror(x, -1, 0, 50, 10, 0)
        self.assertTrue(bool((x[..., 50:60] == 1.0).all()))

    def test_predictions_are_shared(self):
        x = torch.zeros(1, 1, 80, 1, 1)
        x[:, :, 0:12] = 1.0
        x[:, :, 60:72] = 3.0
        nj.H3WindowHandler._share(x, 2, 0, 60, 12)
        a, b = x.flatten()[0:12], x.flatten()[60:72]
        self.assertTrue(bool((a == b).all()), "both copies get the SAME prediction")
        # B's (3.0) where the span follows the take, A's (1.0) where it
        # runs on into the first clip, a straight fade between
        self.assertEqual(float(b[0]), 3.0)
        self.assertEqual(float(b[-1]), 1.0)
        steps = b[:-1] - b[1:]
        self.assertTrue(bool((steps > 0).all()))
        self.assertLess(float((steps - steps[0]).abs().max()), 1e-6)
        # rows outside the two spans are not touched
        self.assertTrue(bool((x.flatten()[12:60] == 0.0).all()))

    def test_each_end_is_its_neighbours_own_prediction(self):
        # a prediction that is continuous with the take at B's start and
        # with the first clip's continuation at A's end stays so
        x = torch.zeros(1, 1, 80)
        x[..., 48:60] = 5.0            # the take's free rows, before B
        x[..., 60:72] = 5.0            # B, predicted as their continuation
        x[..., 0:12] = 2.0             # A
        x[..., 12:24] = 2.0            # the first clip running on, after A
        nj.H3WindowHandler._share(x, -1, 0, 60, 12)
        self.assertEqual(float(x[..., 60] - x[..., 59]), 0.0)
        self.assertEqual(float(x[..., 12] - x[..., 11]), 0.0)

    def test_a_single_row_span_is_the_mean(self):
        x = torch.tensor([[[1.0, 0.0, 3.0]]])
        nj.H3WindowHandler._share(x, -1, 0, 2, 1)
        self.assertEqual(x.flatten().tolist(), [2.0, 0.0, 2.0])


class TheCopiesEndIdentical(unittest.TestCase):
    """A toy sampler around the two handler steps.

    The 'model' predicts each row from itself and its neighbours, so a
    copy's prediction depends on its context -- which is exactly why the
    untied copies drift apart. Euler steps down to sigma 0.
    """

    SIGMAS = [0.2, 0.12, 0.06, 0.02, 0.0]
    A, B, N, HAND = 0, 60, 12, 4

    @staticmethod
    def model(x):
        pad = torch.nn.functional.pad(x, (1, 1), mode="replicate")
        return 0.25 * pad[..., :-2] + 0.5 * x + 0.25 * pad[..., 2:]

    def sample(self, tied):
        g = torch.Generator().manual_seed(7)
        prior = torch.linspace(0, 1, 80).view(1, 1, 80).clone()
        prior[..., self.B:self.B + self.N] = prior[..., self.A:self.A + self.N]
        x = prior + self.SIGMAS[0] * torch.randn(1, 1, 80, generator=g)
        H = nj.H3WindowHandler
        for sigma, nxt in zip(self.SIGMAS[:-1], self.SIGMAS[1:]):
            seen = x.clone()
            if tied:
                H._mirror(seen, -1, self.A, self.B, self.N, self.HAND)
            denoised = self.model(seen)
            if tied:
                H._share(denoised, -1, self.A, self.B, self.N)
            x = denoised + (nxt / sigma) * (x - denoised)
        return x[..., self.A:self.A + self.N], x[..., self.B:self.B + self.N]

    def test_untied_copies_drift_apart(self):
        a, b = self.sample(tied=False)
        self.assertGreater(float((a - b).abs().max()), 1e-3)

    def test_tied_copies_end_identical(self):
        a, b = self.sample(tied=True)
        self.assertEqual(float((a - b).abs().max()), 0.0)


class TheHandlerReadsItsTable(unittest.TestCase):
    def test_a_tie_that_does_not_fit_is_dropped(self):
        h = nj.H3WindowHandler(37, 9)
        table = {"wrap_tie": {"a": 0, "b": 60, "n": 12, "hand": 4,
                              "ta": 0, "tb": 340, "tn": 65, "thand": 22}}
        self.assertIsNone(h._tie(table, 70, 400))
        self.assertIsNone(h._tie({}, 80, 400))
        self.assertIsNone(h._tie(None, 80, 400))
        ok = h._tie(table, 72, 400)
        self.assertEqual((ok["a"], ok["b"], ok["n"]), (0, 60, 12))

    def test_audio_is_clamped_to_the_latent(self):
        h = nj.H3WindowHandler(37, 9)
        table = {"wrap_tie": {"a": 0, "b": 60, "n": 12, "hand": 4,
                              "ta": 0, "tb": 340, "tn": 65, "thand": 22}}
        ok = h._tie(table, 72, 360)
        self.assertEqual(ok["tn"], 20)
        self.assertEqual(ok["thand"], 20)


if __name__ == "__main__":
    unittest.main()
