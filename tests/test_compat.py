"""The install floor: each check names what is wrong and how to fix it, a
broken check never blocks, and the timeline node refuses on a problem.
(The full per-workflow list lives in comfyui-obvpm's Compatibility Check.)

Run from the pack folder:
  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pack = sys.modules.get("obvpm_tl_test")
if pack is None:
    pack = types.ModuleType("obvpm_tl_test")
    pack.__path__ = [ROOT]
    sys.modules[pack.__name__] = pack
from obvpm_tl_test import compat


def stub_registry(**classes):
    mod = types.ModuleType("nodes")
    mod.NODE_CLASS_MAPPINGS = dict(classes)
    sys.modules["nodes"] = mod
    return mod


def module_at(name, path, **attrs):
    """A module the checks can locate: the version is read from the file
    the class's module names, so a stand-in needs only __file__."""
    mod = types.ModuleType(name)
    mod.__file__ = path
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class Versions(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(compat.parse_version("0.2.3"), (0, 2, 3))
        self.assertEqual(compat.parse_version("v0.36"), (0, 36, 0))
        self.assertEqual(compat.parse_version("0.36.0.dev1"), (0, 36, 0))
        self.assertIsNone(compat.parse_version("main"))
        self.assertIsNone(compat.parse_version(None))

    def test_pack_version_walks_up_to_the_pyproject(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "nodes"))
            with open(os.path.join(tmp, "pyproject.toml"), "w") as fh:
                fh.write('[project]\nname = "x"\nversion = "0.2.2"\n')
            mod = module_at("obvpm_tl_test._fake", os.path.join(tmp, "nodes", "mod.py"))
            N = type("N", (), {"__module__": mod.__name__})
            self.assertEqual(compat.pack_version(N), (0, 2, 2))
        # a pack without a pyproject must not answer with ComfyUI's:
        # the walk stops at the folder under custom_nodes
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pyproject.toml"), "w") as fh:
                fh.write('version = "9.9.9"\n')     # stands for ComfyUI's
            os.makedirs(os.path.join(tmp, "custom_nodes", "pack", "nodes"))
            mod = module_at("obvpm_tl_test._nopy",
                            os.path.join(tmp, "custom_nodes", "pack", "nodes", "mod.py"))
            N = type("N", (), {"__module__": mod.__name__})
            self.assertIsNone(compat.pack_version(N))
        # a class whose module has no file has no version
        cls = type("Nowhere", (), {})
        cls.__module__ = "obvpm_tl_test._does_not_exist"
        self.assertIsNone(compat.pack_version(cls))


class ObvpmCheck(unittest.TestCase):
    def test_missing(self):
        stub_registry()
        p = compat.check_obvpm()
        self.assertEqual(p.key, "obvpm")
        self.assertIn("not installed", p.title)
        self.assertTrue(any("github.com/chanon/comfyui-obvpm" in l["url"] for l in p.links))

    def test_too_old_by_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pyproject.toml"), "w") as fh:
                fh.write('version = "0.1.9"\n')
            mod = module_at("obvpm_tl_test._old", os.path.join(tmp, "presets.py"))
            stub_registry(**{"ValuePresets (obvpm)":
                             type("ValuePresets", (), {"__module__": mod.__name__})})
            p = compat.check_obvpm()
            self.assertIsNotNone(p)
            self.assertIn("0.1.9 is installed", p.detail)
            self.assertIn(compat.version_text(compat.MIN_OBVPM) + " or newer", p.detail)
            self.assertIn("Update", p.fix)
            self.assertEqual([l["label"] for l in p.links],
                             ["comfyui-obvpm releases", "comfyui-obvpm on the Comfy Registry"])

    def test_new_enough_by_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pyproject.toml"), "w") as fh:
                fh.write('version = "%s"\n' % compat.version_text(compat.MIN_OBVPM))
            mod = module_at("obvpm_tl_test._new", os.path.join(tmp, "presets.py"))
            stub_registry(**{"ValuePresets (obvpm)":
                             type("ValuePresets", (), {"__module__": mod.__name__})})
            self.assertIsNone(compat.check_obvpm())

    def test_no_version_is_not_a_problem(self):
        # a copy without a pyproject: the version cannot be read, so it
        # is let through rather than refused on a guess
        cls = type("ValuePresets", (), {"__module__": "obvpm_tl_test._nowhere"})
        stub_registry(**{"ValuePresets (obvpm)": cls})
        self.assertIsNone(compat.check_obvpm())


    def test_the_real_installed_pack_passes(self):
        # the sibling checkout, when it is there
        sibling = os.path.join(os.path.dirname(ROOT), "comfyui-obvpm")
        if not os.path.isfile(os.path.join(sibling, "presets.py")):
            self.skipTest("no sibling comfyui-obvpm checkout")
        cls = type("ValuePresets", (), {})
        # the version is read from the FILE the class came from, so a
        # class whose module is the sibling's presets.py
        fake = types.ModuleType("obvpm_tl_test._sibling_presets")
        fake.__file__ = os.path.join(sibling, "presets.py")
        sys.modules[fake.__name__] = fake
        cls.__module__ = fake.__name__
        stub_registry(**{"ValuePresets (obvpm)": cls})
        have = compat.pack_version(cls)
        self.assertIsNotNone(have)
        self.assertEqual(compat.check_obvpm() is None, have >= compat.MIN_OBVPM)


class ComfyCheck(unittest.TestCase):
    def test_versions(self):
        for text, expect in (("0.34.0", True), ("0.35.0", False), ("0.37.0", False), ("main", False)):
            mod = types.ModuleType("comfyui_version")
            mod.__version__ = text
            sys.modules["comfyui_version"] = mod
            try:
                p = compat.check_comfyui()
            finally:
                del sys.modules["comfyui_version"]
            self.assertEqual(p is not None, expect, text)
            if p:
                self.assertIn("0.34.0", p.detail)
                self.assertIn("0.35.0", p.detail)


class Running(unittest.TestCase):
    def test_problems_skips_a_check_that_raises(self):
        def boom():
            raise RuntimeError("the check itself is broken")
        stub_registry()
        found = compat.problems((boom, compat.check_obvpm))
        self.assertEqual([p.key for p in found], ["obvpm"])

    def test_all_checks_pass_on_a_complete_install(self):
        stub_registry()
        cls = type("ValuePresets", (), {"__module__": "obvpm_tl_test._ok"})
        sys.modules["nodes"].NODE_CLASS_MAPPINGS["ValuePresets (obvpm)"] = cls
        self.assertEqual(compat.problems(), [])
        compat.require()                    # does not raise

    def test_require_names_every_problem_and_its_fix(self):
        stub_registry()
        with self.assertRaises(RuntimeError) as caught:
            compat.require()
        text = str(caught.exception)
        self.assertIn("cannot run on this install yet", text)
        self.assertIn("comfyui-obvpm is not installed", text)
        self.assertIn("Fix:", text)
        self.assertIn("https://github.com/", text)

    def test_dict_shape(self):
        stub_registry()
        d = compat.check_obvpm().as_dict()
        self.assertEqual(sorted(d), ["detail", "fix", "key", "links", "title"])
        self.assertEqual(sorted(d["links"][0]), ["label", "url"])


if __name__ == "__main__":
    unittest.main()
