"""An extend's crossfade must not end on the overlap's AAC padding.

The overlap is stored as a small MP4; its AAC sound decodes padded to the
next 1024-sample block with near-silence. The crossfade reads the overlap
from its END, so that padding sat on the join: a few to 30 ms of silence
right before it in every preview and export (issue #22).

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import math
import os
import sys
import tempfile
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
from obvpm_tl_test import crossfade as xf  # noqa: E402
from obvpm_tl_test import frames as fr  # noqa: E402
from obvpm_tl_test import nodes_save  # noqa: E402
from obvpm_tl_test import preview_route as pr  # noqa: E402

RATE = 32000


def tone(frames, rate=RATE):
    """A steady 440 Hz tone a little longer than the frames, as H3 ships it."""
    n = int(round(frames / fr.FPS * rate)) + 300
    t = torch.arange(n, dtype=torch.float32) / rate
    wave = 0.3 * torch.sin(2 * math.pi * 440.0 * t)
    return {"waveform": wave.expand(1, 2, n).clone(), "sample_rate": rate}


def round_trip(frames):
    """Store an extend's overlap as the save does, read it as a crossfade does."""
    images = torch.full((frames + 10, 64, 64, 3), 0.5)
    blob = nodes_save._overlap_bytes(images, tone(frames + 10), frames, 19)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "overlap.mp4")
        with open(p, "wb") as f:
            f.write(blob)
        return xf._fit_audio(pr._decode_images(p), pr._decode_audio(p))


def rms(x):
    return float(torch.sqrt(torch.mean(x.float() ** 2)))


class OverlapAudio(unittest.TestCase):
    def check(self, frames):
        images, audio = round_trip(frames)
        self.assertEqual(int(images.shape[0]), frames)
        wave = audio["waveform"]
        self.assertEqual(int(wave.shape[-1]),
                         xf.audio_frames_samples(frames, RATE))
        # the last 10 ms is the tone, not padding: about 0.3 / sqrt(2)
        end = rms(wave[..., -int(0.010 * RATE):])
        middle = rms(wave[..., wave.shape[-1] // 2:wave.shape[-1] // 2 + 320])
        self.assertGreater(end, 0.5 * middle)

    def test_39_frame_overlap(self):        # 7 ms of padding
        self.check(39)

    def test_90_frame_overlap(self):        # 26 ms of padding
        self.check(90)

    def test_audio_already_exact_is_kept(self):
        images = torch.zeros(22, 8, 8, 3)
        n = xf.audio_frames_samples(22, RATE)
        audio = {"waveform": torch.ones(1, 2, n), "sample_rate": RATE}
        self.assertIs(xf._fit_audio(images, audio)[1], audio)

    def test_no_sound_or_picture_passes(self):
        self.assertEqual(xf._fit_audio(None, None), (None, None))


if __name__ == "__main__":
    unittest.main()
