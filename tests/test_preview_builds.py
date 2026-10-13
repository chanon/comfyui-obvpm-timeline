"""Preview builds, one at a time per file, and what a Result Preview
shows when its workflow is opened.

- two requests for the same preview file never build at once (they used
  to share one temp file: the first renamed it away and the second failed
  with FileNotFoundError); different files still build side by side, a
  probe never waits, and a failing build releases the lock;
- latest_result picks the newest `<prefix>_NNNNN.mp4` -- not a joint
  render's `_s01` segment, not an HQ copy -- judges it with show(), caches
  that per file state, and refuses a prefix outside the output folder.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest

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
from obvpm_tl_test import nodes_result as nr  # noqa: E402
from obvpm_tl_test import preview_route as pr  # noqa: E402


class OneBuildPerFile(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="obvpm_tl_builds_")
        self._old_out = folder_paths.get_output_directory()
        folder_paths.set_output_directory(self.root)
        self._real = pr._build_preview
        self.active = {}
        self.overlap = []
        self.calls = []
        guard = threading.Lock()

        def fake(sequence, crf, probe=False, base_folder="",
                 preview_filename=None, **opts):
            key = (base_folder, preview_filename)
            with guard:
                self.calls.append((key, probe))
                self.active[key] = self.active.get(key, 0) + 1
                if self.active[key] > 1:
                    self.overlap.append(key)
            try:
                if sequence == "boom":
                    raise RuntimeError("build failed")
                time.sleep(0.15)
                return (preview_filename, base_folder, [], 0, False, "k")
            finally:
                with guard:
                    self.active[key] -= 1
        pr._build_preview = fake

    def tearDown(self):
        pr._build_preview = self._real
        folder_paths.set_output_directory(self._old_out)
        shutil.rmtree(self.root, ignore_errors=True)

    def run_all(self, jobs):
        errors = []

        def one(args, kw):
            try:
                pr.build_preview(*args, **kw)
            except Exception as exc:        # noqa: BLE001 -- recorded
                errors.append(exc)
        threads = [threading.Thread(target=one, args=j) for j in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        return errors

    def test_same_file_never_builds_twice_at_once(self):
        job = (("a.mp4", 19, False, "proj", "obvpm_h3_result_clip"), {})
        self.assertEqual(self.run_all([job] * 4), [])
        self.assertEqual(self.overlap, [])
        self.assertEqual(len(self.calls), 4)

    def test_different_files_still_build_side_by_side(self):
        start = time.monotonic()
        self.run_all([
            (("a.mp4", 19, False, "proj", "obvpm_h3_result_clip"), {}),
            (("a.mp4", 19, False, "proj", "obvpm_h3_result_upscale"), {}),
        ])
        self.assertLess(time.monotonic() - start, 0.28,
                        "two different previews waited for each other")

    def test_a_probe_never_waits(self):
        lock = pr._build_lock(pr._preview_paths("proj", "p")[0])
        lock.acquire()
        try:
            t0 = time.monotonic()
            pr.build_preview("a.mp4", 19, True, "proj", "p")
            self.assertLess(time.monotonic() - t0, 0.5)
        finally:
            lock.release()

    def test_a_failed_build_releases_the_lock(self):
        with self.assertRaises(RuntimeError):
            pr.build_preview("boom", 19, False, "proj", "p")
        lock = pr._build_lock(pr._preview_paths("proj", "p")[0])
        self.assertTrue(lock.acquire(timeout=0.5))
        lock.release()

    def test_one_lock_per_file_whatever_the_spelling(self):
        a = pr._build_lock(os.path.join(self.root, "proj", "x.mp4"))
        b = pr._build_lock(os.path.join(self.root, "proj", ".", "x.mp4"))
        c = pr._build_lock(os.path.join(self.root, "proj", "y.mp4"))
        self.assertIs(a, b)
        self.assertIsNot(a, c)


class LatestResult(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="obvpm_tl_latest_")
        self._old_out = folder_paths.get_output_directory()
        folder_paths.set_output_directory(self.root)
        os.makedirs(os.path.join(self.root, "proj"))
        self.shown = []
        self._show = nr.H3ResultPreview.show
        test = self

        def show(self, path):
            test.shown.append(path)
            return {"ui": {"h3_result": [{"clip": path, "sequence": path}]}}
        nr.H3ResultPreview.show = show
        nr._LATEST_CACHE.clear()

    def tearDown(self):
        nr.H3ResultPreview.show = self._show
        folder_paths.set_output_directory(self._old_out)
        shutil.rmtree(self.root, ignore_errors=True)

    def touch(self, name, age):
        p = os.path.join(self.root, "proj", name)
        with open(p, "wb") as f:
            f.write(b"x")
        t = time.time() - age
        os.utime(p, (t, t))

    def test_newest_take_of_the_prefix(self):
        self.touch("clip_00001.mp4", 30)
        self.touch("clip_00002.mp4", 20)
        self.touch("clip_00002_s01.mp4", 1)        # a segment
        self.touch("clip_hq_00003.mp4", 1)         # an HQ copy
        self.touch("upscale_00004.mp4", 1)         # another preview's
        self.assertEqual(nr.latest_result("proj/clip")["clip"],
                         "proj/clip_00002.mp4")
        self.assertEqual(nr.latest_result("proj/upscale")["clip"],
                         "proj/upscale_00004.mp4")

    def test_newest_by_time_not_by_number(self):
        # the counter reuses a deleted take's number
        self.touch("clip_00009.mp4", 50)
        self.touch("clip_00003.mp4", 5)
        self.assertEqual(nr.latest_result("proj/clip")["clip"],
                         "proj/clip_00003.mp4")

    def test_judged_once_per_file_state(self):
        self.touch("clip_00001.mp4", 30)
        nr.latest_result("proj/clip")
        nr.latest_result("proj/clip")
        self.assertEqual(self.shown, ["proj/clip_00001.mp4"])
        self.touch("clip_00001.mp4", 10)           # re-saved
        nr.latest_result("proj/clip")
        self.assertEqual(len(self.shown), 2)

    def test_nothing_there(self):
        self.assertIsNone(nr.latest_result("proj/clip"))
        self.assertIsNone(nr.latest_result("nowhere/clip"))
        self.assertIsNone(nr.latest_result(""))

    def test_outside_the_output_folder_is_refused(self):
        with self.assertRaises(ValueError):
            nr.latest_result("../../somewhere/clip")


if __name__ == "__main__":
    unittest.main()
