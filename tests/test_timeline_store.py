"""timeline_store: path safety and round-trip persistence."""
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

pack = types.ModuleType("obvpm_tl_test")
pack.__path__ = [ROOT]
sys.modules[pack.__name__] = pack
from obvpm_tl_test import timeline_store as ts  # noqa: E402
import folder_paths  # noqa: E402


class TimelineStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = patch.object(
            folder_paths, "get_output_directory", lambda: self.tmp.name)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_round_trip(self):
        seq = "loop\nproject1/clip_00001.mp4\nproject1/clip_00002.mp4"
        ts.save_state("project1", seq)
        self.assertEqual(ts.load_state("project1"), seq)

    def test_missing_returns_none(self):
        self.assertIsNone(ts.load_state("nothing_here"))

    def test_traversal_refused(self):
        with self.assertRaises(ValueError):
            ts.state_path("../outside")

    def test_list_folders_finds_video_dir(self):
        d = os.path.join(self.tmp.name, "myproj")
        os.makedirs(d)
        open(os.path.join(d, "clip.mp4"), "wb").close()
        self.assertIn("myproj", ts.list_output_folders())

    def test_list_folders_finds_saved_state(self):
        ts.save_state("saved_only", "a.mp4")
        self.assertIn("saved_only", ts.list_output_folders())

    def test_search_finds_deep_folder(self):
        deep = os.path.join(self.tmp.name, "a", "b", "c", "d")
        os.makedirs(deep)
        open(os.path.join(deep, "clip.mp4"), "wb").close()
        self.assertNotIn("a/b/c/d", ts.list_output_folders())
        self.assertIn("a/b/c/d", ts.search_output_folders("c/d"))

    def test_browse_lists_children_and_up(self):
        os.makedirs(os.path.join(self.tmp.name, "a", "b"))
        os.makedirs(os.path.join(self.tmp.name, "a", "c"))
        listing = ts.browse_folder("a")
        self.assertEqual(listing["folder"], "a")
        self.assertEqual(listing["parent"], "")
        names = [c["name"] for c in listing["children"]]
        self.assertEqual(names, ["b", "c"])
        self.assertEqual(ts.browse_folder("a/b")["parent"], "a")
        root = ts.browse_folder("")
        self.assertIsNone(root["parent"])
        self.assertIn("a", [c["name"] for c in root["children"]])

    def test_ensure_folder_creates_and_refuses_traversal(self):
        self.assertEqual(ts.ensure_folder("a/b/c"), "a/b/c")
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name, "a", "b", "c")))
        with self.assertRaises(ValueError):
            ts.ensure_folder("../outside")

    def test_browse_traversal_refused(self):
        with self.assertRaises(ValueError):
            ts.browse_folder("../outside")


if __name__ == "__main__":
    unittest.main()
