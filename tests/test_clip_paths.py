"""Finding a clip's sidecar and parent from wherever the clip lives.

Three fixes, pinned here:
- the server's clip_meta answers a sidecar that PAIRS with its full
  header (issue #15: the browser's own ranged read failed, the server said
  nothing, and every clip showed "no mctx"), and says why when there is
  no usable one;
- a pin taken from a clip with a sidecar records where that clip was, so
  a take extended from a clip OUTSIDE its project folder still finds its
  parent (Result Preview used to scan only the take's own folder);
- a base_folder typed with backslashes is the same folder as with "/".

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.abspath(os.path.join(ROOT, "..", ".."))
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
import folder_paths  # noqa: E402
from obvpm_tl_test import mctx  # noqa: E402
from obvpm_tl_test import nodes_load  # noqa: E402
from obvpm_tl_test import nodes_pins  # noqa: E402
from obvpm_tl_test import nodes_result  # noqa: E402
from obvpm_tl_test import nodes_save  # noqa: E402
from obvpm_tl_test import preview_route as pr  # noqa: E402
from obvpm_tl_test import seam_report  # noqa: E402


class OutputFolder(unittest.TestCase):
    """A throwaway output folder with fake clips: any bytes are a "video"
    as far as hashing goes, and a sidecar is a safetensors file whose
    metadata carries the format marker and the video's hash."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="obvpm_tl_paths_")
        self._old_out = folder_paths.get_output_directory()
        folder_paths.set_output_directory(self.root)
        nodes_load._HASH_CACHE.clear()

    def tearDown(self):
        folder_paths.set_output_directory(self._old_out)
        shutil.rmtree(self.root, ignore_errors=True)

    def video(self, rel, content=b"frames"):
        path = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def sidecar(self, video_path, **meta):
        full = {"format": mctx.FORMAT, "self_id": mctx.hash_file(video_path),
                "relation": "", "parent_id": "", "pins": "[]"}
        full.update({k: str(v) for k, v in meta.items()})
        save_file({"video": torch.zeros(1)}, mctx.sidecar_path(video_path),
                  metadata=full)
        return full


class ClipMeta(OutputFolder):
    def setUp(self):
        super().setUp()
        # a plain video's identity needs ffprobe on a real file; these are
        # byte blobs, so stand in for it (what it returns is not under test)
        self._synth = nodes_load.synthetic_header
        nodes_load.synthetic_header = lambda path: {
            "format": mctx.FORMAT, "self_id": "SYNTH", "no_sidecar": "1"}

    def tearDown(self):
        nodes_load.synthetic_header = self._synth
        super().tearDown()

    def test_paired_sidecar_is_answered_in_full(self):
        # issue #15: this used to be None, which the browser could only
        # read as "no latents"
        meta = self.sidecar(self.video("V_Project/clip_00010.mp4"),
                            delivered_frames=124)
        got = pr.plain_clip_meta("V_Project/clip_00010.mp4")
        self.assertEqual(got, meta)
        self.assertNotIn("no_sidecar", got)

    def test_missing_sidecar_says_so(self):
        self.video("V_Project/plain.mp4")
        got = pr.plain_clip_meta("V_Project/plain.mp4")
        self.assertEqual(got["no_sidecar"], "1")
        self.assertEqual(got["no_sidecar_reason"], "missing")
        self.assertEqual(got["self_id"], "SYNTH")

    def test_changed_video_is_a_mismatch(self):
        path = self.video("V_Project/clip_00011.mp4")
        self.sidecar(path)
        with open(path, "ab") as f:
            f.write(b"re-encoded")
        got = pr.plain_clip_meta("V_Project/clip_00011.mp4")
        self.assertEqual(got["no_sidecar_reason"], "mismatch")
        self.assertEqual(got["no_sidecar"], "1")

    def test_unreadable_sidecar_says_so(self):
        path = self.video("V_Project/clip_00012.mp4")
        with open(mctx.sidecar_path(path), "wb") as f:
            f.write(b"not a safetensors file")
        got = pr.plain_clip_meta("V_Project/clip_00012.mp4")
        self.assertEqual(got["no_sidecar_reason"], "unreadable")

    def test_without_identity_no_hash_of_a_plain_video(self):
        self.video("V_Project/plain.mp4")
        calls = []
        real = nodes_load._cached_hash
        nodes_load._cached_hash = lambda p: calls.append(p) or real(p)
        try:
            got = pr.plain_clip_meta("V_Project/plain.mp4", identity=False)
        finally:
            nodes_load._cached_hash = real
        self.assertEqual(got, {"no_sidecar": "1",
                               "no_sidecar_reason": "missing"})
        self.assertEqual(calls, [])

    def test_without_identity_a_paired_sidecar_is_still_answered(self):
        meta = self.sidecar(self.video("V_Project/clip_00013.mp4"))
        self.assertEqual(
            pr.plain_clip_meta("V_Project/clip_00013.mp4", identity=False),
            meta)


class LatentPinsRecordTheirSource(unittest.TestCase):
    def bundle(self, clip):
        return mctx.make_mctx("PARENT", torch.zeros(1, 24, 12, 2, 2),
                              torch.zeros(1, 32, 2, 40),
                              {"delivered_frames": "39"}, clip=clip)

    def build(self, bundle):
        (specs,) = nodes_pins.H3MCtxPinSpec().build(
            mctx=bundle, window=22, take_from="tail", take_from_frame=0,
            place="before", place_at_frame=0, audio_window=0)
        return specs[-1]

    def test_a_loaded_clip_records_its_path(self):
        spec = self.build(self.bundle("elsewhere/clip_00001.mp4"))
        self.assertEqual(spec["source_path"], "elsewhere/clip_00001.mp4")
        self.assertEqual(spec["source_kind"], "clip")   # still a latent pin
        entry = json.loads(mctx.serialize_pins([spec]))[0]
        self.assertEqual(entry["source_path"], "elsewhere/clip_00001.mp4")

    def test_backslashes_are_written_as_slashes(self):
        spec = self.build(self.bundle("elsewhere\\clip_00001.mp4"))
        self.assertEqual(spec["source_path"], "elsewhere/clip_00001.mp4")

    def test_a_wired_bundle_has_no_path_to_record(self):
        self.assertNotIn("source_path", self.build(self.bundle(None)))


class ResultPreviewFindsAParentElsewhere(OutputFolder):
    def setUp(self):
        super().setUp()
        # the seam numbers need real latents and frames; not under test
        self._seam = nodes_result._measure_seam
        self._cuts = seam_report.measure_cuts
        nodes_result._measure_seam = lambda *a, **k: None
        seam_report.measure_cuts = lambda *a, **k: None

    def tearDown(self):
        nodes_result._measure_seam = self._seam
        seam_report.measure_cuts = self._cuts
        super().tearDown()

    def take(self, parent_rel, recorded):
        parent = self.video(parent_rel, b"parent frames")
        pid = self.sidecar(parent)["self_id"]
        pin = {"source_id": pid, "source_kind": "clip", "place": "before",
               "source_start": 0, "source_frames": 22}
        if recorded:
            pin["source_path"] = parent_rel
        take = self.video("V_Project/clip_00002.mp4", b"take frames")
        self.sidecar(take, relation="extends", parent_id=pid,
                     pins=json.dumps([pin]))
        return nodes_result.H3ResultPreview().show(
            "V_Project/clip_00002.mp4")["ui"]["h3_result"][0]

    def test_parent_in_another_folder_is_found_by_its_recorded_path(self):
        got = self.take("elsewhere/clip_00001.mp4", recorded=True)
        self.assertEqual(got["parent"], "elsewhere/clip_00001.mp4")
        self.assertEqual(got["sequence"],
                         "elsewhere/clip_00001.mp4\nV_Project/clip_00002.mp4")

    def test_nothing_wired_shows_nothing_new(self):
        # the Save node feeding it is muted for this Run: no path arrives,
        # and an empty ui sends no `executed`, so the preview keeps its take
        self.assertIn("path", nodes_result.H3ResultPreview.INPUT_TYPES()["optional"])
        self.assertEqual(nodes_result.H3ResultPreview().show(), {"ui": {}})
        self.assertTrue(nodes_result.H3ResultPreview.IS_CHANGED())

    def test_without_a_recorded_path_it_is_not(self):
        # an older take (saved before the path was recorded)
        got = self.take("elsewhere/clip_00001.mp4", recorded=False)
        self.assertIsNone(got["parent"])

    def test_a_recorded_path_to_a_different_file_is_refused(self):
        got = self.take("elsewhere/clip_00001.mp4", recorded=True)
        self.assertEqual(got["parent"], "elsewhere/clip_00001.mp4")
        # replace the parent with other content: the hash decides
        with open(os.path.join(self.root, "elsewhere", "clip_00001.mp4"),
                  "wb") as f:
            f.write(b"something else")
        nodes_load._HASH_CACHE.clear()
        again = nodes_result.H3ResultPreview().show(
            "V_Project/clip_00002.mp4")["ui"]["h3_result"][0]
        self.assertIsNone(again["parent"])

    def test_same_folder_parent_still_found_without_a_path(self):
        got = self.take("V_Project/clip_00001.mp4", recorded=False)
        self.assertEqual(got["parent"], "V_Project/clip_00001.mp4")

    def test_the_payload_carries_the_takes_file_time(self):
        # by the server's clock, in ms: a cleared preview compares it
        got = self.take("V_Project/clip_00001.mp4", recorded=False)
        st = os.stat(os.path.join(self.root, "V_Project", "clip_00002.mp4"))
        self.assertEqual(got["mtime_ms"], st.st_mtime_ns // 1_000_000)


class BackslashFolders(OutputFolder):
    def test_folder_text(self):
        for typed in ("projects\\V_Project", "projects/V_Project",
                      " \\projects\\V_Project\\ ", "/projects/V_Project/"):
            self.assertEqual(nodes_save.folder_text(typed),
                             "projects/V_Project", typed)
        self.assertEqual(nodes_save.folder_text(None), "")

    def test_save_prefix(self):
        self.assertEqual(
            nodes_save._save_prefix("projects\\V_Project", "clip"),
            "projects/V_Project/clip")
        self.assertEqual(
            nodes_save._save_prefix("projects/V_Project", "takes\\clip"),
            "projects/V_Project/takes/clip")

    def test_preview_lands_in_the_same_folder_either_way(self):
        a = pr._preview_paths("projects\\V_Project", "obvpm_h3_preview")
        b = pr._preview_paths("projects/V_Project", "obvpm_h3_preview")
        self.assertEqual(a, b)
        self.assertEqual(a[3], "projects/V_Project")

    def test_escape_is_still_refused(self):
        with self.assertRaises(ValueError):
            nodes_save._save_prefix("..\\..\\elsewhere", "clip")


def _link_folder(link, target):
    """A folder link: a Windows junction (no admin rights needed) or a
    symlink. False when this host can make neither."""
    if os.name == "nt":
        import subprocess
        subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True)
        if os.path.isdir(link):
            return True
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False


class LinkedFolders(OutputFolder):
    """A project folder inside output that is a link to another drive:
    takes saved there must load, extend and upscale like any other (they
    used to be listed, then refused as "escaping the output folder")."""

    def setUp(self):
        super().setUp()
        self.elsewhere = tempfile.mkdtemp(prefix="obvpm_tl_elsewhere_")
        with open(os.path.join(self.elsewhere, "clip_00001.mp4"), "wb") as f:
            f.write(b"frames")
        self.links = []
        for link, target in ((os.path.join(self.root, "proj"), self.elsewhere),
                             (os.path.join(self.elsewhere, "back"), self.root)):
            if not _link_folder(link, target):
                self.skipTest("this host can make no folder links")
            self.links.append(link)

    def tearDown(self):
        for link in self.links:       # the link itself, never what it points at
            os.rmdir(link)
        shutil.rmtree(self.elsewhere, ignore_errors=True)
        super().tearDown()

    def test_a_take_in_a_linked_folder_resolves(self):
        path = nodes_load.resolve_clip_path("proj/clip_00001.mp4")
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.normcase(path), os.path.normcase(
            os.path.join(self.root, "proj", "clip_00001.mp4")))

    def test_paths_written_to_leave_are_still_refused(self):
        self.video("own.mp4")
        outside = os.path.join(self.elsewhere, "clip_00001.mp4")
        for clip in ("../" + os.path.basename(self.root) + "x/a.mp4",
                     "proj/../../clip_00001.mp4", outside, "", "proj"):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                nodes_load.resolve_clip_path(clip)
        self.assertTrue(os.path.isfile(nodes_load.resolve_clip_path("own.mp4")))

    def test_the_listing_walks_links_once(self):
        self.video("own.mp4")
        clips = nodes_load._list_clips()
        self.assertIn("proj/clip_00001.mp4", clips)
        self.assertIn("own.mp4", clips)
        # proj/back leads to the output folder again: walked once only
        self.assertEqual(len(clips), len(set(clips)))
        self.assertFalse([c for c in clips if c.startswith("proj/back/")])

    def test_a_link_never_takes_a_real_folders_place(self):
        # "a_alias" sorts before the real "z_takes" it points to: the real
        # folder is listed, the alias adds nothing
        self.video("z_takes/clip_00002.mp4")
        alias = os.path.join(self.root, "a_alias")
        self.assertTrue(_link_folder(alias, os.path.join(self.root, "z_takes")))
        self.links.insert(0, alias)
        clips = nodes_load._list_clips()
        self.assertIn("z_takes/clip_00002.mp4", clips)
        self.assertNotIn("a_alias/clip_00002.mp4", clips)
        # and the alias still loads, if a workflow names it
        self.assertTrue(os.path.isfile(nodes_load.resolve_clip_path("a_alias/clip_00002.mp4")))


if __name__ == "__main__":
    unittest.main()
