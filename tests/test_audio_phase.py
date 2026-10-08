"""Audio phase: a take's sound stays on its parent's audio grid in the
latents, and is cut on the take's own frames where it becomes a file.

Audio steps (40 Hz) meet frames (24 fps) every third frame. A pin window
starting between two steps is sliced from the nearest one, so the take's
sound sits a third of a step off its picture. The latents keep that --
the joint upscale lays every clip on one grid and needs it -- and the
trim cuts the delivered sound that much earlier or later instead, so the
MP4, and every preview and export made of MP4s, plays in sync.

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
from obvpm_tl_test import frames as fr  # noqa: E402
from obvpm_tl_test import nodes_pins  # noqa: E402

THIRD = 1 / 3
SR = 32000


def source(frames, phase=None, head=0):
    steps = fr.frames_to_latents(frames)
    t = round(frames / 24 * 40)
    meta = {"delivered_frames": str(frames - head),
            "pinned_head_frames": str(head)}
    if phase is not None:
        meta["audio_phase"] = "%.6f" % phase
    return {"self_id": "P", "clip": "p.mp4", "meta": meta,
            "video_latent": torch.zeros(1, 4, steps, 2, 2),
            "audio_latent": torch.zeros(1, 4, 2, t)}


def pin(frames, place="before", take_from="tail", at=0, **kw):
    (specs,) = nodes_pins.H3MCtxPinSpec().build(
        mctx=source(frames, **kw), window=39, take_from=take_from,
        take_from_frame=at, place=place, place_at_frame=0, audio_window=0,
        mode="masked")
    return nodes_pins._prepare_clip_pin(0, specs[0], snap_down=False)


class PinPhase(unittest.TestCase):
    def test_a_window_between_steps(self):
        # 124 frames: window 85..124, frame 85 = step 141.67, sliced at 142
        self.assertAlmostEqual(pin(124)["audio_phase"], THIRD)
        # 362 frames: window 323..362, frame 323 = step 538.33, sliced at 538
        self.assertAlmostEqual(pin(362)["audio_phase"], -THIRD)

    def test_a_window_on_a_step(self):
        self.assertAlmostEqual(pin(141)["audio_phase"], 0.0)

    def test_a_chain_carries_its_parents_phase(self):
        # a take of 141 raw frames whose own sound is a third off: the
        # window at its end starts on a step, so the third carries on
        self.assertAlmostEqual(pin(141, phase=THIRD)["audio_phase"], THIRD,
                               places=5)    # stored to 6 decimals
        # ...and a window off a step adds its own
        self.assertAlmostEqual(
            pin(362, phase=THIRD)["audio_phase"], 0.0, places=5)  # -1/3 + 1/3

    def test_the_latents_are_the_parents_own(self):
        p = pin(124)
        self.assertEqual(p["audio_steps"], 65)                # nothing moved
        self.assertNotIn("audio_regrid_ms", p)


class Bounded(unittest.TestCase):
    """The slice starts on the step nearest the window's TRUE time, so a
    phase never grows past half a step along a chain."""

    def test_two_thirds_become_minus_one(self):
        # a source a third off, a window a third off the same way: sliced
        # on its own grid that was +2/3; nearest in true time it is -1/3
        p = pin(124, phase=THIRD)
        self.assertAlmostEqual(p["audio_phase"], -THIRD, places=5)
        self.assertEqual(p["spec"]["audio_start_step"], 141)   # not 142

    def test_a_long_chain_never_drifts(self):
        phase = 0.0
        for k in range(30):
            frames = (124, 362, 107, 141, 56 + 51)[k % 5]
            if frames < 40:
                continue
            phase = pin(frames, phase=phase)["audio_phase"]
            self.assertLessEqual(abs(phase), 0.5 + 1e-6, k)

    def test_the_recipe_records_the_slice(self):
        p = pin(124)
        self.assertEqual(p["spec"]["audio_start_step"], 142)
        self.assertEqual(p["spec"]["audio_steps"], 65)

    def test_the_sidecar_keeps_it(self):
        # what the save writes is what the upscale reads back
        from obvpm_tl_test import mctx
        p = pin(124, phase=THIRD)
        (saved,) = json.loads(mctx.serialize_pins([p["spec"]]))
        self.assertEqual(saved["audio_start_step"], 141)
        self.assertEqual(saved["audio_steps"], 65)


class UpscaleLineage(unittest.TestCase):
    """The joint lays a pinned clip's sound by its lineage."""

    def head(self, sid, raw, ph, pins):
        return {"self_id": sid, "raw_frames": str(raw),
                "delivered_frames": str(raw - ph),
                "pinned_head_frames": str(ph), "pins": json.dumps(pins)}

    def chain(self, b_step=None):
        root = self.head("R", 124, 0, [])
        a = self.head("A", 141, 39, [{"source_id": "R", "source_kind": "clip",
                                      "place": "before", "source_start": 85,
                                      "source_frames": 39}])
        pb = {"source_id": "A", "source_kind": "clip", "place": "before",
              "source_start": 52, "source_frames": 39}      # an A cut at 91
        if b_step is not None:
            pb["audio_start_step"] = b_step
        b = self.head("B", 141, 39, [pb])
        return [root, a, b]

    def test_an_off_grid_cut_of_an_off_grid_take(self):
        from obvpm_tl_test import nodes_joint as nj
        # each clip's own start rounded: R 0, A 142 (85 -> 141.67),
        # B 228 (137 -> 228.33)
        rounded = [0, fr.audio_total(85), fr.audio_total(85 + 52)]
        self.assertEqual(rounded, [0, 142, 228])
        # an OLD recipe was sliced at A's step 87 (52 -> 86.67): its sound
        # belongs at 142 + 87 = 229, where it continues A's exactly
        self.assertEqual(nj.lineage_ticks(self.chain(), rounded),
                         [0, 142, 229])
        # a new one records the step it took (86: nearest in true time)
        self.assertEqual(nj.lineage_ticks(self.chain(b_step=86), rounded),
                         [0, 142, 228])

    def test_a_clip_from_elsewhere_keeps_its_rounded_start(self):
        from obvpm_tl_test import nodes_joint as nj
        hs = self.chain()
        hs[1]["pins"] = json.dumps([{"source_id": "ELSEWHERE",
                                     "source_kind": "clip", "place": "before",
                                     "source_start": 85, "source_frames": 39}])
        self.assertEqual(nj.lineage_ticks(hs, [0, 142, 228])[1], 142)


class TakePhase(unittest.TestCase):
    def test_an_extend_takes_its_windows_phase(self):
        self.assertAlmostEqual(
            nodes_pins.pins_audio_phase([pin(124)], 141), THIRD)

    def test_a_prepend_on_the_av_grid_into_a_clip_head(self):
        # held from the target's head (frame 0, on a step), at the end of a
        # 141 frame run (on the AV grid): nothing off
        p = pin(124, place="after", take_from="head")
        self.assertAlmostEqual(nodes_pins.pins_audio_phase([p], 141), 0.0)

    def test_a_root_has_none(self):
        self.assertEqual(nodes_pins.pins_audio_phase([], 124), 0.0)


class HeaderPhase(unittest.TestCase):
    def header(self, **kw):
        pins = [dict({"place": "before", "source_kind": "clip",
                      "source_start": 85, "source_frames": 39}, **kw)]
        return {"raw_frames": "141", "pins": json.dumps(pins)}

    def test_recorded(self):
        self.assertAlmostEqual(nodes_pins.header_audio_phase(
            {"audio_phase": "-0.333333"}), -THIRD, places=5)

    def test_an_older_take_from_its_recipe(self):
        self.assertAlmostEqual(
            nodes_pins.header_audio_phase(self.header()), THIRD)

    def test_an_older_take_encoded_from_pixels(self):
        self.assertEqual(nodes_pins.header_audio_phase(
            self.header(source_kind="clip_pixels")), 0.0)

    def test_synced(self):
        self.assertTrue(nodes_pins.header_audio_synced({"audio_synced": "1"}))
        self.assertFalse(nodes_pins.header_audio_synced(self.header()))


class Trim(unittest.TestCase):
    """The delivered sound, cut on the take's own frames."""

    def trim(self, phase, raw=141, head=39):
        # each sample's value is its own index, so the cut is readable
        n = round(raw / 24 * SR) + 300
        wave = torch.arange(n, dtype=torch.float32).view(1, 1, -1)
        pins = [{"place": "before", "covered": head, "audio_phase": phase,
                 "spec": {"place": "before"}}]
        images = torch.zeros(raw, 2, 2, 3)
        _, audio = nodes_pins.H3TrimPinned().trim(
            images, pins, audio={"waveform": wave, "sample_rate": SR})
        return audio["waveform"][0, 0]

    def test_on_a_step_nothing_moves(self):
        w = self.trim(0.0)
        self.assertEqual(float(w[0]), round(39 / 24 * SR))
        self.assertEqual(len(w), round(102 / 24 * SR))

    def test_sound_from_later_than_the_picture_is_cut_earlier(self):
        w = self.trim(THIRD)
        self.assertEqual(float(w[0]), round(39 / 24 * SR) - 267)
        self.assertEqual(len(w), round(102 / 24 * SR))       # same length

    def test_sound_from_earlier_is_cut_later(self):
        w = self.trim(-THIRD)
        self.assertEqual(float(w[0]), round(39 / 24 * SR) + 267)
        self.assertEqual(len(w), round(102 / 24 * SR))

    def test_past_the_decode_it_mirrors_never_silence(self):
        out = nodes_pins.shifted_samples(
            torch.arange(10, dtype=torch.float32).view(1, -1), -3, 13)
        self.assertEqual(out[0].tolist(),
                         [3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6])


if __name__ == "__main__":
    unittest.main()
