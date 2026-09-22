"""The install checks: each names what is wrong and how to fix it, a
broken check never blocks, and the timeline node refuses on a problem.

Run from the pack folder:
  python -m unittest discover -s tests -v
"""
import importlib
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
compat = importlib.import_module("obvpm_tl_test.compat")


def stub_registry(**classes):
    mod = types.ModuleType("nodes")
    mod.NODE_CLASS_MAPPINGS = dict(classes)
    sys.modules["nodes"] = mod
    return mod


def node_with_inputs(*names, module=None):
    def INPUT_TYPES():
        return {"required": {n: ("INT",) for n in names}}
    cls = type("Node", (), {"INPUT_TYPES": staticmethod(INPUT_TYPES)})
    if module:
        cls.__module__ = module
    return cls


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
            path = os.path.join(tmp, "nodes", "mod.py")
            with open(path, "w") as fh:
                fh.write("class N:\n    pass\n")
            spec = importlib.util.spec_from_file_location("obvpm_tl_test._fake", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            self.assertEqual(compat.pack_version(mod.N), (0, 2, 2))
        # a pack without a pyproject must not answer with ComfyUI's:
        # the walk stops at the folder under custom_nodes
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pyproject.toml"), "w") as fh:
                fh.write('version = "9.9.9"\n')     # stands for ComfyUI's
            os.makedirs(os.path.join(tmp, "custom_nodes", "pack", "nodes"))
            path = os.path.join(tmp, "custom_nodes", "pack", "nodes", "mod.py")
            with open(path, "w") as fh:
                fh.write("class N:\n    pass\n")
            spec = importlib.util.spec_from_file_location("obvpm_tl_test._nopy", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            self.assertIsNone(compat.pack_version(mod.N))
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
                fh.write('version = "0.2.2"\n')
            path = os.path.join(tmp, "presets.py")
            with open(path, "w") as fh:
                fh.write("class ValuePresets:\n    pass\n")
            spec = importlib.util.spec_from_file_location("obvpm_tl_test._old", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            stub_registry(**{"ValuePresets (obvpm)": mod.ValuePresets})
            p = compat.check_obvpm()
            self.assertIsNotNone(p)
            self.assertIn("0.2.2 is installed", p.detail)
            self.assertIn("0.2.3 or newer", p.detail)
            self.assertIn("Update", p.fix)
            self.assertEqual([l["label"] for l in p.links],
                             ["comfyui-obvpm releases", "comfyui-obvpm on the Comfy Registry"])

    def test_new_enough_by_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "pyproject.toml"), "w") as fh:
                fh.write('version = "0.2.3"\n')
            path = os.path.join(tmp, "presets.py")
            with open(path, "w") as fh:
                fh.write("class ValuePresets:\n    pass\n")
            spec = importlib.util.spec_from_file_location("obvpm_tl_test._new", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            stub_registry(**{"ValuePresets (obvpm)": mod.ValuePresets})
            self.assertIsNone(compat.check_obvpm())

    def test_no_version_falls_back_to_asking_the_parser(self):
        # a copy without a pyproject: the parser is asked whether it
        # reads a hint
        for answer, expect_problem in (([{"name": "x", "hint": "h"}], False),
                                       ([{"name": "x"}], True)):
            mod = types.ModuleType("obvpm_tl_test._nover")
            mod.describe = lambda schema, _a=answer: _a
            cls = type("ValuePresets", (), {})
            cls.__module__ = mod.__name__
            sys.modules[mod.__name__] = mod
            stub_registry(**{"ValuePresets (obvpm)": cls})
            p = compat.check_obvpm()
            self.assertEqual(p is not None, expect_problem, answer)
            if p:
                self.assertIn("cannot read", p.detail)

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


class UpscalerCheck(unittest.TestCase):
    def test_original(self):
        stub_registry(MinimaxH3LatentUpscaler3D=node_with_inputs(
            "latent", "model_name", "mode", "align", "enable_temporal_chunking",
            "force_unload", "device", "precision"))
        self.assertIsNone(compat.check_upscaler())

    def test_plus_fork(self):
        stub_registry(MinimaxH3LatentUpscaler3D=node_with_inputs(
            "latent", "model_name", "mode", "align", "keep_proportion",
            "device", "precision", "offload_after_upscale"))
        p = compat.check_upscaler()
        self.assertEqual(p.title, "The wrong latent upscaler is installed")
        self.assertIn("Plus", p.detail)
        self.assertIn("device shows 'true'", p.detail)
        self.assertIn("LBH-123-AI", p.fix)
        self.assertEqual(p.links[0]["url"], "https://github.com/" + compat.UPSCALER_REPO)

    def test_some_other_version_without_chunking(self):
        stub_registry(MinimaxH3LatentUpscaler3D=node_with_inputs("latent", "scale"))
        p = compat.check_upscaler()
        self.assertEqual(p.title, "The latent upscaler has no temporal chunking")

    def test_missing(self):
        stub_registry()
        self.assertIn("not installed", compat.check_upscaler().title)

    def test_a_node_that_will_not_describe_itself_is_not_a_problem(self):
        def INPUT_TYPES():
            raise RuntimeError("no models folder here")
        stub_registry(MinimaxH3LatentUpscaler3D=type("N", (), {"INPUT_TYPES": staticmethod(INPUT_TYPES)}))
        self.assertIsNone(compat.check_upscaler())


class ComfyCheck(unittest.TestCase):
    def test_versions(self):
        for text, expect in (("0.32.0", True), ("0.33.0", False), ("0.36.0", False), ("main", False)):
            mod = types.ModuleType("comfyui_version")
            mod.__version__ = text
            sys.modules["comfyui_version"] = mod
            try:
                p = compat.check_comfyui()
            finally:
                del sys.modules["comfyui_version"]
            self.assertEqual(p is not None, expect, text)
            if p:
                self.assertIn("0.32.0", p.detail)
                self.assertIn("0.33.0", p.detail)


class Running(unittest.TestCase):
    def test_problems_skips_a_check_that_raises(self):
        def boom():
            raise RuntimeError("the check itself is broken")
        stub_registry()
        found = compat.problems((boom, compat.check_rgthree))
        self.assertEqual([p.key for p in found], ["rgthree"])

    def test_all_checks_pass_on_a_complete_install(self):
        stub_registry(**{
            "ModelPreviewOverrideKJ": object, "MiniMaxH3TurboLoRA": object,
            "SpectrumApplyMiniMaxH3": object, "Power Lora Loader (rgthree)": object,
            "MinimaxH3LatentUpscaler3D": node_with_inputs("enable_temporal_chunking"),
        })
        # obvpm answered through the parser fallback
        mod = types.ModuleType("obvpm_tl_test._ok")
        mod.describe = lambda schema: [{"name": "x", "hint": "h"}]
        sys.modules[mod.__name__] = mod
        cls = type("ValuePresets", (), {})
        cls.__module__ = mod.__name__
        sys.modules["nodes"].NODE_CLASS_MAPPINGS["ValuePresets (obvpm)"] = cls
        self.assertEqual(compat.problems(), [])
        compat.require()                    # does not raise

    def test_require_names_every_problem_and_its_fix(self):
        stub_registry()
        with self.assertRaises(RuntimeError) as caught:
            compat.require()
        text = str(caught.exception)
        self.assertIn("cannot run on this install yet", text)
        for pack in ("comfyui-obvpm", "Comfyui_Minimax_h3_latent_Upscaler",
                     "ComfyUI-KJNodes", "ComfyUI-MiniMax-H3-Turbo",
                     "ComfyUI-Spectrum-MiniMax-H3", "rgthree-comfy"):
            self.assertIn(pack, text)
        self.assertIn("Fix:", text)
        self.assertIn("https://github.com/", text)

    def test_dict_shape_for_the_widget(self):
        stub_registry()
        d = compat.check_kjnodes().as_dict()
        self.assertEqual(sorted(d), ["detail", "fix", "key", "links", "title"])
        self.assertEqual(sorted(d["links"][0]), ["label", "url"])


if __name__ == "__main__":
    unittest.main()
