"""The joint refine in SEGMENTS: split at hard cuts, cropped to the cut.

Three reported bugs had one cause -- H3 Join Latents laid the timeline as
one line from the first clip's raw frame 0 to the last clip's raw end,
and the render output all of it:
  - a take trimmed to a few seconds was upscaled and refined whole;
  - the last clip's cut was not respected;
  - a timeline with a hard cut (two unrelated clips, a gray seam) could
    not be upscaled (off the latent grid, or joined mid-cycle).
Now the cut is split into segments at every join that is not a seamless
lineage link as the cut plays it (a hand cut that moves a green seam is
a jump cut, so it splits too), each segment is cropped to what the cut
shows plus a little context, the nodes after H3 Join Latents run once
per segment, and the render writes each segment and joins them.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
import torch  # noqa: E402
import folder_paths  # noqa: E402
from obvpm_tl_test import nodes_joint as nj  # noqa: E402
from obvpm_tl_test import nodes_render as nr  # noqa: E402
from obvpm_tl_test import frames as fr  # noqa: E402
from obvpm_tl_test import avpack  # noqa: E402


def head(self_id, raw, head_f=0, tail_f=0, pins=(), relation="", parent="",
         join=0):
    return {"self_id": self_id, "raw_frames": str(raw),
            "delivered_frames": str(raw - head_f - tail_f),
            "pinned_head_frames": str(head_f),
            "pinned_tail_frames": str(tail_f),
            "relation": relation, "parent_id": parent,
            "parent_join_frame": str(join), "pins": json.dumps(list(pins))}


def extend_of(parent_id, parent_raw, raw=141, covered=39):
    """A take extending `parent_id` from its last `covered` raw frames."""
    start = parent_raw - covered
    return head("C_" + parent_id, raw, head_f=covered,
                pins=[{"source_id": parent_id, "source_kind": "clip",
                       "place": "before", "source_start": start,
                       "source_frames": covered, "mode": "masked"}],
                relation="extends", parent=parent_id, join=parent_raw)


def layout_of(text, headers_by_clip):
    """What build() computes per segment, without loading any latent."""
    items = nj.sequence_items(text)
    lines = [line for kind, line in items if kind == "clip"]
    clips = [nj.clip_of_line(line) for line in lines]
    headers = [headers_by_clip[c] for c in clips]
    segs = nj.split_segments(items, headers)
    cuts = nj.shown_cuts(lines, headers, nj.derive_flags(items))
    out = []
    for seg in segs:
        idx = seg["clips"]
        hs = [headers[i] for i in idx]
        cs_ = [cuts[i] for i in idx]
        ticks = [fr.audio_total(int(h["raw_frames"])) for h in hs]
        lay = nj.timeline_layout([lines[i] for i in idx], hs, ticks, cuts=cs_)
        lo, hi = nj.shown_range(lay, hs, cs_)
        cs, ce = nj.crop_range(lay["total_steps"], lo, hi)
        cropped = nj.crop_layout(lay, cs, ce)
        f0 = cropped["crop"]["frames"][0]
        out.append(dict(cropped, keep=[lo - f0, hi - f0], seg=seg,
                        full=lay))
    return out


class SplitTests(unittest.TestCase):
    def test_seam_settings_are_not_part_of_the_path(self):
        self.assertEqual(nj.parse_sequence("a.mp4 @ 5 [audio_declick=off]\n"
                                           "b.mp4 [crossfade=on]"),
                         ["a.mp4", "b.mp4"])

    def test_gaps_are_items_not_clips(self):
        items = nj.sequence_items("loop\na.mp4\n~ 24\n# note\nb.mp4 @ 3")
        self.assertEqual(items, [("clip", "a.mp4"), ("gap", 24),
                                 ("clip", "b.mp4 @ 3")])
        self.assertEqual(nj.parse_sequence_lines("a.mp4\n~ 24\nb.mp4"),
                         ["a.mp4", "b.mp4"])

    def test_unrelated_clips_are_separate_segments(self):
        hs = [head("A", 124), head("B", 124)]
        segs = nj.split_segments(nj.sequence_items("a\nb"), hs)
        self.assertEqual([s["clips"] for s in segs], [[0], [1]])

    def test_a_green_seam_played_as_is_is_one_segment(self):
        p = head("P", 141)
        c = extend_of("P", 141)
        segs = nj.split_segments(nj.sequence_items("p\nc"), [p, c])
        self.assertEqual([s["clips"] for s in segs], [[0, 1]])

    def test_a_hand_cut_on_a_green_seam_splits_it(self):
        # before: the take was placed by its pin and the joint silently
        # played the natural join instead of the hand cut
        p = head("P", 141)
        c = extend_of("P", 141)
        for text in ("p @ ..100\nc", "p\nc @ 12"):
            segs = nj.split_segments(nj.sequence_items(text), [p, c])
            self.assertEqual([s["clips"] for s in segs], [[0], [1]], text)
        # a marker that says what the lineage says is no jump
        segs = nj.split_segments(nj.sequence_items("p @ ..141\nc @ 0"), [p, c])
        self.assertEqual([s["clips"] for s in segs], [[0, 1]])

    def test_empty_space_splits_and_is_kept(self):
        p = head("P", 141)
        c = extend_of("P", 141)
        items = nj.sequence_items("~ 10\np\n~ 24\nc\n~ 5")
        segs = nj.split_segments(items, [p, c])
        self.assertEqual(segs, [
            {"clips": [0], "gap_before": 10, "gap_after": 0},
            {"clips": [1], "gap_before": 24, "gap_after": 5}])
        # and no seam is derived across it: both clips play whole
        self.assertEqual(nj.derive_flags(items), [True, False])


class CropTests(unittest.TestCase):
    def test_a_trimmed_take_is_cropped_to_what_the_cut_shows(self):
        segs = layout_of("a @ 60..100", {"a": head("A", 141)})
        self.assertEqual(len(segs), 1)
        s = segs[0]
        # 42 steps laid out; 37 kept: frames 0..124, the shown 60..100 plus
        # a cycle past it, clip-shaped (5k+2)
        self.assertEqual(s["full"]["total_steps"], 42)
        self.assertEqual(s["crop"]["steps"], [0, 37])
        self.assertEqual(s["total_frames"], 124)
        self.assertEqual(s["keep"], [60, 100])
        self.assertEqual(s["total_ticks"], fr.audio_total(124))

    def test_a_crop_deep_into_a_long_take(self):
        segs = layout_of("a @ 200..260", {"a": head("A", 294)})
        s = segs[0]
        # the head is cropped too: on a 51-frame boundary at least a
        # cycle before the shown start (153 = 3 x 51), where audio ticks
        # are whole numbers (255 = 3 x 85)
        self.assertEqual(s["crop"]["steps"], [45, 82])
        self.assertEqual(s["crop"]["frames"], [153, 277])
        self.assertEqual(s["crop"]["ticks"], [255, fr.audio_total(277)])
        self.assertEqual(s["keep"], [47, 107])
        self.assertEqual(s["total_steps"] % 5, 2)
        # the tick count is what a clip of that many frames has
        self.assertEqual(s["total_ticks"], fr.audio_total(s["total_frames"]))
        self.assertEqual(s["steps"], [(-45, 87)])
        self.assertEqual(s["owner_steps"], [0] * 37)

    def test_a_whole_chain_keeps_its_layout(self):
        p = head("P", 141)
        c = extend_of("P", 141)
        s = layout_of("p\nc", {"p": p, "c": c})[0]
        self.assertEqual(s["crop"]["steps"], [0, s["full"]["total_steps"]])
        self.assertEqual(s["steps"], s["full"]["steps"])
        self.assertEqual(s["owner_steps"], s["full"]["owner_steps"])
        self.assertEqual(s["keep"], [0, 243])

    def test_the_last_clips_cut_is_kept(self):
        p = head("P", 141)
        c = extend_of("P", 141)
        s = layout_of("p\nc @ ..60", {"p": p, "c": c})[0]
        # c's delivered frame 60 sits at its raw start (102) + head (39)
        self.assertEqual(s["keep"], [0, 102 + 39 + 60])
        self.assertLess(s["total_frames"], s["full"]["total_frames"])

    def test_two_unrelated_takes_each_start_on_the_cycle(self):
        # 124 frames is 5 mod 17: back to back on one line the second
        # take started mid-cycle
        segs = layout_of("a\nb", {"a": head("A", 124), "b": head("B", 124)})
        self.assertEqual(len(segs), 2)
        for s in segs:
            self.assertEqual(s["steps"], [(0, 37)])
            self.assertEqual(s["keep"], [0, 124])

    def test_a_hand_cut_take_enters_where_the_cut_says(self):
        p = head("P", 141)
        c = extend_of("P", 141)
        segs = layout_of("p @ ..100\nc @ 12", {"p": p, "c": c})
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["keep"], [0, 100])
        # c on its own: raw 0, 39 held frames, then delivered 12..102
        self.assertEqual(segs[1]["keep"], [39 + 12, 39 + 102])

    def test_the_off_grid_error_names_the_clip(self):
        hs = [head("A", 122), head("B", 124)]
        with self.assertRaisesRegex(ValueError, r"b\.mp4.*nearest frames"):
            nj.timeline_layout(["a.mp4", "b.mp4"], hs, [200, 200])


class AssembleTests(unittest.TestCase):
    def test_rows_land_where_the_crop_puts_them(self):
        s = layout_of("a @ 200..260", {"a": head("A", 294)})[0]
        steps = torch.arange(87, dtype=torch.float32)
        v = steps.view(1, 1, 87, 1, 1).expand(1, 2, 87, 2, 2).clone()
        a = torch.arange(fr.audio_total(294), dtype=torch.float32)
        a = a.view(1, 1, 1, -1).expand(1, 32, 2, -1).clone()
        video, audio = nj.assemble([v], [a], s)
        self.assertEqual(video.shape[2], 37)
        self.assertEqual(video[0, 0, :, 0, 0].tolist(), list(range(45, 82)))
        self.assertEqual(float(audio[0, 0, 0, 0]), 255.0)
        self.assertEqual(audio.shape[-1], s["total_ticks"])


class WindowTests(unittest.TestCase):
    def test_a_clip_before_the_crop_is_clamped_and_one_outside_skipped(self):
        # clip 0 starts before the crop and owns nothing in it; clip 1
        # starts 20 steps before it and owns every row
        spans = [(-60, 42), (-20, 87)]
        owners = [1] * 42
        wins = nj.anchored_windows(42, 37, 7, spans, owners)
        self.assertTrue(wins)
        self.assertTrue(all(j == 1 for _, _, j in wins))
        self.assertEqual(wins[0][0], 0)
        self.assertEqual(wins[-1][1], 42)

    def test_timeline_steps_reads_the_owners(self):
        table = {"spans": [(-20, 87)], "owners": [0] * 42}
        self.assertEqual(nj.timeline_steps([[{nj.COND_KEY: table}]]), 42)


class BuildTests(unittest.TestCase):
    HEADERS = {"a.mp4": head("A", 124), "b.mp4": head("B", 141)}

    def _fake_load(self, clips):
        videos, audios = [], []
        for c in clips:
            raw = int(self.HEADERS[c]["raw_frames"])
            videos.append(torch.zeros(1, 2, fr.frames_to_latents(raw), 2, 2))
            audios.append(torch.zeros(1, 32, 2, fr.audio_total(raw)))
        return videos, audios

    def test_build_hands_out_one_latent_per_segment(self):
        self.assertEqual(nj.H3JointLatent.OUTPUT_IS_LIST, (True, True))
        with patch.object(nj, "clip_headers",
                          lambda clips: [self.HEADERS[c] for c in clips]), \
                patch.object(nj.H3JointLatent, "_load",
                             staticmethod(self._fake_load)):
            latents, layouts = nj.H3JointLatent().build(
                "a.mp4 @ 10\n~ 12\nb.mp4 @ ..90")
        self.assertEqual(len(latents), 2)
        self.assertEqual([l["segment"] for l in layouts], [0, 1])
        self.assertEqual([l["segments"] for l in layouts], [2, 2])
        self.assertEqual(layouts[1]["gap_before"], 12)
        self.assertEqual(layouts[0]["keep_frames"], [10, 124])
        self.assertEqual(layouts[1]["keep_frames"], [0, 90])
        for lat, lay in zip(latents, layouts):
            v, a = avpack.unpack_av(lat)
            self.assertEqual(int(v.shape[2]), lay["total_steps"])
            self.assertEqual(int(a.shape[-1]), lay["total_ticks"])


class JoinTests(unittest.TestCase):
    RATE = 24000

    def _sound(self, frames, value=1.0):
        n = int(round(frames / fr.FPS * self.RATE))
        return {"waveform": torch.full((1, 2, n), value), "sample_rate": self.RATE}

    def test_join_audio_sizes_every_part_and_tapers_the_cuts(self):
        parts = [{"path": "x", "frames": 10, "sound": self._sound(10)},
                 {"path": None, "frames": 5, "sound": None},
                 {"path": "y", "frames": 12, "sound": self._sound(11),
                  "declick": True}]
        got = nr.join_audio(parts)
        wave = got["waveform"][0]
        self.assertEqual(wave.shape[-1], int(round(27 / fr.FPS * self.RATE)))
        at10 = int(round(10 / fr.FPS * self.RATE))
        at15 = int(round(15 / fr.FPS * self.RATE))
        self.assertEqual(float(wave[:, at10:at15].abs().max()), 0.0)  # silence
        self.assertEqual(float(wave[0, 0]), 1.0)          # the cut's start: untouched
        self.assertLess(float(wave[0, at10 - 1]), 0.01)    # tapered into the gap
        self.assertLess(float(wave[0, at15]), 0.01)        # tapered out of it
        self.assertEqual(float(wave[0, -1]), 0.0)          # 11 frames of sound padded to 12

    def test_declick_can_be_turned_off_per_join(self):
        parts = [{"path": "x", "frames": 10, "sound": self._sound(10)},
                 {"path": "y", "frames": 10, "sound": self._sound(10),
                  "declick": False}]
        wave = nr.join_audio(parts)["waveform"][0]
        at10 = int(round(10 / fr.FPS * self.RATE))
        self.assertEqual(float(wave[0, at10 - 1]), 1.0)
        self.assertEqual(float(wave[0, at10]), 1.0)
        self.assertFalse(nr._declick_into({"lines": ["b.mp4 [audio_declick=off]"]}))
        self.assertTrue(nr._declick_into({"lines": ["b.mp4"]}))


class _FakeVAE:
    """Decodes a latent of T steps into pixel_frames(T) grey frames."""
    def decode(self, z):
        t, h, w = int(z.shape[2]), int(z.shape[3]), int(z.shape[4])
        return torch.full((1, fr.pixel_frames(t), h * 16, w * 16, 3), 0.5)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(folder_paths, "get_output_directory",
                         lambda: self.tmp.name),
            patch.object(nr, "decode_audio", self._audio),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    @staticmethod
    def _audio(audio_vae, audio):
        ticks = int(audio.shape[-1])
        n = int(round(ticks / 40.0 * 24000))
        return {"waveform": torch.zeros(1, 2, n), "sample_rate": 24000}

    @staticmethod
    def _latent(steps):
        frames = fr.pixel_frames(steps)
        return avpack.pack_av(torch.zeros(1, 2, steps, 2, 2),
                              torch.zeros(1, 32, 2, fr.audio_total(frames)))

    def _frames(self, path):
        import av
        with av.open(path) as c:
            return sum(1 for _ in c.decode(video=0))

    def test_one_segment_is_one_take_cropped_to_the_cut(self):
        out = nr.H3JointRender().render(
            samples=[self._latent(7)], vae=[_FakeVAE()], audio_vae=[None],
            base_folder=["t"], filename_prefix=["up"], crf=[30],
            window_seconds=[5.0],
            layout=[{"total_steps": 7, "keep_frames": [3, 20], "gap_before": 0,
                     "gap_after": 0, "clips": ["a.mp4"], "lines": ["a.mp4"]}])
        path = out[0]
        self.assertTrue(path.endswith("up_00001.mp4"))
        self.assertEqual(self._frames(path), 17)
        header = nr.mctx.read_header(nr.mctx.sidecar_path(path))
        self.assertEqual(int(header["delivered_frames"]), 17)
        self.assertEqual(int(header["pinned_head_frames"]), 3)

    def test_segments_are_written_and_joined(self):
        lay = lambda k, gb, ga, keep: {
            "total_steps": 7, "keep_frames": keep, "gap_before": gb,
            "gap_after": ga, "segment": k, "segments": 2,
            "clips": ["x.mp4"], "lines": ["x.mp4"]}
        out = nr.H3JointRender().render(
            samples=[self._latent(7), self._latent(7)], vae=[_FakeVAE()],
            audio_vae=[None], base_folder=["t"], filename_prefix=["up"],
            crf=[30], window_seconds=[5.0],
            layout=[lay(0, 0, 0, [0, 22]), lay(1, 6, 4, [5, 15])])
        path = out[0]
        folder = os.path.dirname(path)
        self.assertTrue(path.endswith("up_00001.mp4"))
        for k in (1, 2):
            seg = os.path.join(folder, "up_00001_s%02d.mp4" % k)
            self.assertTrue(os.path.isfile(seg))
            self.assertTrue(os.path.isfile(nr.mctx.sidecar_path(seg)))
            meta = json.loads(nr.mctx.read_header(
                nr.mctx.sidecar_path(seg))["user_meta"])
            self.assertEqual(meta["segment"], [k, 2])
            self.assertEqual(meta["cut"], "up_00001.mp4")
        # 22 + 6 of empty space + 10 + 4 of empty space
        self.assertEqual(self._frames(path), 42)
        import av
        with av.open(path) as c:
            self.assertEqual(len(c.streams.audio), 1)
        # the next render takes the next number, not one of the segments'
        out2 = nr.H3JointRender().render(
            samples=[self._latent(7)], vae=[_FakeVAE()], audio_vae=[None],
            base_folder=["t"], filename_prefix=["up"], crf=[30],
            window_seconds=[5.0], layout=[lay(0, 0, 0, [0, 22])])
        self.assertTrue(out2[0].endswith("up_00002.mp4"))


if __name__ == "__main__":
    unittest.main()
