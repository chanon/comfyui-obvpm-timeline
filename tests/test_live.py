"""H3 Live Preview (nodes_live): the sampler's picture on the take's timeline.

Pinned here:
- the tiny decoder is built from any TAESD-style state dict (core's own
  64-channel layout here; taeh3 runs at 96);
- each latent step's picture is held for the frames its step covers, so a
  clip has exactly fr.pixel_frames(T) frames at 24 fps;
- the pins Apply Pins prepared for a latent of length T tell the preview
  how many frames are pinned at each end (and none for another length);
- the sampling wrapper: an H3 audio+video latent packed the way core packs
  it, one message per step at most, the newest step's clip on the take's
  timeline.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import base64
import io
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
from obvpm_tl_test import nodes_live as nl  # noqa: E402
from obvpm_tl_test import nodes_pins  # noqa: E402


class Decoder(unittest.TestCase):
    def test_builds_from_a_taesd_state_dict(self):
        from comfy.taesd.taesd import Decoder as CoreDecoder
        core = CoreDecoder(latent_channels=24)
        sd = {k: torch.randn_like(v) * 0.05 for k, v in core.state_dict().items()}
        model, channels = nl._tiny_decoder(sd)
        self.assertEqual(channels, 24)
        x = torch.randn(2, 24, 4, 6)
        with torch.no_grad():
            # core's layout upsamples 8x (taeh3's 16x)
            self.assertEqual(tuple(model(x).shape), (2, 3, 32, 48))
            # the same layers in the same places: the same picture
            core.load_state_dict(sd)
            self.assertTrue(torch.allclose(model(x), core(x), atol=1e-5))

    def test_a_prefixed_state_dict(self):
        from comfy.taesd.taesd import Decoder as CoreDecoder
        sd = {"decoder." + k: v for k, v in
              CoreDecoder(latent_channels=24).state_dict().items()}
        _model, channels = nl._tiny_decoder(sd)
        self.assertEqual(channels, 24)

    def test_latent_colours_without_a_file(self):
        import comfy.latent_formats as lf
        dec = nl._Decoder(nl.LATENT2RGB, lf.MiniMaxH3Video(), "cpu")
        pics, holds = dec(torch.randn(24, 7, 9, 16), 640)
        self.assertEqual(dec.kind, "rgb")
        self.assertEqual(pics.dtype, torch.uint8)
        self.assertEqual(tuple(pics.shape)[:1], (7,))
        self.assertEqual(pics.shape[-1], 3)
        self.assertEqual(holds, nl.hold_counts(7))


class WhichDecoder(unittest.TestCase):
    """Both tiny VAEs for H3 are published as taeh3.safetensors: what a
    file IS decides how it is used, never its name."""

    def setUp(self):
        import tempfile
        import folder_paths
        self.dir = tempfile.mkdtemp(prefix="obvpm_vae_approx_")
        self._paths = folder_paths.folder_names_and_paths.get("vae_approx")
        folder_paths.folder_names_and_paths["vae_approx"] = (
            [self.dir], {".safetensors"})
        folder_paths.filename_list_cache.clear()
        nl._Decoder._cache.clear()

    def tearDown(self):
        import shutil
        import folder_paths
        folder_paths.folder_names_and_paths["vae_approx"] = self._paths
        folder_paths.filename_list_cache.clear()
        nl._Decoder._cache.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, sd):
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()},
                  os.path.join(self.dir, "taeh3.safetensors"))

    def test_a_2d_one_is_held_per_step(self):
        import comfy.latent_formats as lf
        from comfy.taesd.taesd import Decoder as CoreDecoder
        self.write(CoreDecoder(latent_channels=24).state_dict())
        dec = nl._Decoder("taeh3.safetensors", lf.MiniMaxH3Video(), "cpu")
        self.assertEqual(dec.kind, "2d")
        pics, holds = dec(torch.randn(24, 7, 4, 6), 640)
        self.assertEqual(pics.shape[0], 7)
        self.assertEqual(holds, nl.hold_counts(7))

    def test_a_temporal_one_gives_every_frame(self):
        import comfy.latent_formats as lf
        from comfy.taesd.taehv import TAEHV
        self.write(TAEHV(latent_channels=24).state_dict())
        dec = nl._Decoder("taeh3.safetensors", lf.MiniMaxH3Video(), "cpu")
        self.assertEqual(dec.kind, "temporal")
        t = 7                                        # 22 frames
        pics, holds = dec(torch.randn(24, t, 4, 6), 640)
        self.assertEqual(pics.shape[0], fr.pixel_frames(t))
        self.assertEqual(holds, [1] * fr.pixel_frames(t))
        self.assertEqual(pics.dtype, torch.uint8)


class Timeline(unittest.TestCase):
    def test_holds_sum_to_the_grid(self):
        for t in (2, 7, 12, 37, 42):
            self.assertEqual(sum(nl.hold_counts(t)), fr.pixel_frames(t))

    def test_a_clip_has_the_takes_frames(self):
        import av
        t = 12                                       # 39 frames
        pics = torch.randint(0, 255, (t, 32, 48, 3), dtype=torch.uint8)
        clip = nl.encode_clip(pics, nl.hold_counts(t))
        with av.open(io.BytesIO(clip)) as box:
            n = sum(1 for _ in box.decode(video=0))
            rate = float(box.streams.video[0].average_rate)
        self.assertEqual(n, fr.pixel_frames(t))
        self.assertEqual(rate, fr.FPS)


class Pins(unittest.TestCase):
    def setUp(self):
        nl._PINS.clear()

    def test_the_pins_of_this_length(self):
        pins = [{"covered": 22, "place": "before", "spec": {}},
                {"covered": 22, "place": "after", "spec": {}}]
        nl.note_pins(42, pins)
        self.assertEqual(nl.pinned_frames(42), nodes_pins.pins_trim_totals(pins))
        self.assertEqual(nl.pinned_frames(37), (0, 0))

    def test_no_pins_is_nothing_pinned(self):
        nl.note_pins(37, [])
        self.assertEqual(nl.pinned_frames(37), (0, 0))


class Wrapper(unittest.TestCase):
    """The OUTER_SAMPLE wrapper around a stand-in sampler."""

    def setUp(self):
        nl._PINS.clear()
        self.sent = []
        self._send = nl._send
        nl._send = self.sent.append

    def tearDown(self):
        nl._send = self._send

    def run_pass(self, sigma0, steps=3, t=12, end=0.0):
        import comfy.latent_formats as lf
        import comfy.utils
        video = torch.randn(1, 24, t, 5, 8)
        audio = torch.randn(1, 32, 2, 20)
        packed, shapes = comfy.utils.pack_latents([video, audio])

        class Patcher:
            load_device = "cpu"
            model = types.SimpleNamespace(latent_format=lf.MiniMaxH3AV())

        class Exec:
            class_obj = types.SimpleNamespace(model_patcher=Patcher())

            def __call__(self, noise, latent, sampler, sigmas, mask, callback,
                         disable_pbar, seed, latent_shapes=None):
                for i in range(steps):
                    callback(i, packed, packed, steps)
                return packed

        seen = []
        w = nl._LiveWrapper("77", nl.LATENT2RGB, 256)
        w(Exec(), packed, packed, None, torch.tensor([sigma0, end]), None,
          lambda *a: seen.append(a[0]), True, 0, shapes)
        return seen

    def test_one_clip_per_step_at_most_the_newest_last(self):
        nl.note_pins(12, [{"covered": 5, "place": "before", "spec": {}}])
        seen = self.run_pass(1.0)
        self.assertEqual(seen, [0, 1, 2])            # the sampler's own callback
        self.assertTrue(1 <= len(self.sent) <= 3)
        last = self.sent[-1]
        self.assertEqual(last["step"], 3)
        self.assertEqual(last["frames"], fr.pixel_frames(12))
        self.assertEqual((last["head"], last["tail"]), nl.pinned_frames(12))
        self.assertEqual(last["node"], "77")
        self.assertTrue(last["synced"])
        import av
        with av.open(io.BytesIO(base64.b64decode(last["image"]))) as box:
            self.assertEqual(sum(1 for _ in box.decode(video=0)),
                             fr.pixel_frames(12))


if __name__ == "__main__":
    unittest.main()
