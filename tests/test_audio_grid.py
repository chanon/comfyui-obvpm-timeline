"""The audio grid at a clip's end, as core sizes it.

Core gives a clip round(seconds * 40) audio steps -- the nearest step, so
on one 17k+5 length in three (56, 107, ..., 362 frames) the audio ends a
third of a step BEFORE the last frame. Continuing such a clip from its
end used to log "unexpected audio grid ... assuming no overhang".

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
from obvpm_tl_test import frames as fr  # noqa: E402
from obvpm_tl_test import nodes_pins  # noqa: E402


def core_audio_steps(frames):
    # comfy_extras/nodes_minimax_h3.py temporal_shape()
    return round(frames / 24 * 40)


def source(frames):
    steps = fr.frames_to_latents(frames)
    return {"self_id": "P", "clip": "p.mp4",
            "meta": {"delivered_frames": str(frames),
                     "pinned_head_frames": "0"},
            "video_latent": torch.zeros(1, 4, steps, 2, 2),
            "audio_latent": torch.zeros(1, 4, 2, core_audio_steps(frames))}


def tail_pin(frames, window=39):
    (specs,) = nodes_pins.H3MCtxPinSpec().build(
        mctx=source(frames), window=window, take_from="tail",
        take_from_frame=0, place="before", place_at_frame=0,
        audio_window=0, mode="masked")
    return nodes_pins._prepare_clip_pin(0, specs[0], snap_down=False)


class CoreGrid(unittest.TestCase):
    def test_every_clip_length_is_an_expected_grid(self):
        for k in range(120):
            n = 17 * k + 5
            oh = fr.audio_overhang(core_audio_steps(n), n)
            self.assertIsNotNone(oh, n)
            self.assertLess(abs(oh), 0.5, n)

    def test_audio_that_ends_short_of_the_last_frame(self):
        self.assertAlmostEqual(fr.audio_overhang(603, 362), -1 / 3)
        self.assertAlmostEqual(fr.audio_overhang(207, 124), 1 / 3)
        self.assertIsNone(fr.audio_overhang(602, 362))


class TailPin(unittest.TestCase):
    def test_a_362_frame_clip_continues_without_a_grid_warning(self):
        with self.assertNoLogs("obvpm.h3", level="WARNING"):
            pin = tail_pin(362)
        self.assertAlmostEqual(pin["overhang"], -1 / 3)
        self.assertEqual(pin["audio_steps"], 65)    # 39 frames, exactly

    def test_a_clip_whose_audio_runs_past_its_end(self):
        pin = tail_pin(124)
        self.assertAlmostEqual(pin["overhang"], 1 / 3)
        self.assertEqual(pin["audio_steps"], 65)

    def test_a_clip_on_the_shared_grid_has_none(self):
        pin = tail_pin(141)
        self.assertEqual(pin["overhang"], 0.0)


if __name__ == "__main__":
    unittest.main()
