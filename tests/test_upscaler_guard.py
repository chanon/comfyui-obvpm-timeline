"""A Run that would latent-upscale with temporal chunking off is refused
before the upscaled latent is sampled.

The upscaler (MinimaxH3LatentUpscaler3D) with enable_temporal_chunking
off was seen inventing details the clip never had. The first node of
ours after it -- H3 Joint Audio Mask in the upscale -- reads the queued
prompt and walks back up its inputs to the upscaler.

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
from obvpm_tl_test import nodes_joint as nj  # noqa: E402


def upscale_prompt(chunking=True, mask_id="1363", up_id="1351"):
    """The 010 upscale path as the API prompt has it: upscaler ->
    LTXVConcatAVLatent -> CleanVRAM -> H3 Joint Audio Mask."""
    return {
        "1350": {"class_type": "H3JointLatent", "inputs": {}},
        up_id: {"class_type": nj.UPSCALER_CLASS, "inputs": {
            "samples": ["1350", 0], "enable_temporal_chunking": chunking}},
        "1352": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": [up_id, 0], "audio_latent": ["1350", 1]}},
        "1365": {"class_type": "CleanVRAM (obvpm)", "inputs": {
            "anything": ["1352", 0]}},
        mask_id: {"class_type": "H3JointAudioMask", "inputs": {
            "latent": ["1365", 0], "audio_denoise": 0.0}},
    }


class Guard(unittest.TestCase):
    def test_chunking_on_runs(self):
        self.assertEqual(nj.unchunked_upscalers(upscale_prompt(), "1363"), [])

    def test_chunking_off_is_found_through_the_nodes_between(self):
        self.assertEqual(
            nj.unchunked_upscalers(upscale_prompt(False), "1363"), ["1351"])

    def test_refused_with_the_node_named(self):
        with self.assertRaisesRegex(
                ValueError, r"H3 Joint Audio Mask: the latent upscaler "
                            r"\(node #1351\) has enable_temporal_chunking off"):
            nj.refuse_unchunked_upscale(upscale_prompt(False), "1363",
                                        "H3 Joint Audio Mask")

    def test_a_muted_upscaler_is_not_in_the_prompt(self):
        p = upscale_prompt(False)
        del p["1351"]
        p["1352"]["inputs"]["video_latent"] = ["1350", 0]
        self.assertEqual(nj.unchunked_upscalers(p, "1363"), [])

    def test_a_value_on_a_wire_is_not_known_here(self):
        p = upscale_prompt()
        p["1351"]["inputs"]["enable_temporal_chunking"] = ["1400", 0]
        self.assertEqual(nj.unchunked_upscalers(p, "1363"), [])

    def test_an_upscaler_not_feeding_this_node_is_not_its_business(self):
        p = upscale_prompt()
        p["1553"] = {"class_type": nj.UPSCALER_CLASS, "inputs": {
            "enable_temporal_chunking": False}}
        self.assertEqual(nj.unchunked_upscalers(p, "1363"), [])

    def test_inside_a_subgraph(self):
        p = upscale_prompt(False, mask_id="1359:7", up_id="1359:3")
        self.assertEqual(nj.unchunked_upscalers(p, "1359:7"), ["1359:3"])

    def test_no_prompt_no_check(self):
        self.assertEqual(nj.unchunked_upscalers(None, "1363"), [])
        self.assertEqual(nj.unchunked_upscalers(upscale_prompt(False), None),
                         [])

    def test_the_audio_mask_node_refuses(self):
        latent = {"samples": torch.zeros(1, 4, 2, 2, 2)}
        with self.assertRaisesRegex(ValueError, "enable_temporal_chunking"):
            nj.H3JointAudioMask().mask(latent, 0.0,
                                       prompt=upscale_prompt(False),
                                       unique_id="1363")


if __name__ == "__main__":
    unittest.main()
