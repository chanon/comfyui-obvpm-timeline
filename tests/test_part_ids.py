"""Part ids: the `[id=...]` a sequence line may carry, and the pin fields
that carry it into a take's recipe.

- the line grammar reads and writes the id without disturbing the cut or
  the seam settings, and the JS mirror (web/h3_mctx_ui.js, run under node)
  reads every line exactly as Python does -- a drift there is a cut that
  silently becomes a different cut;
- pin_state's `part` / `part2` are accepted, checked, and stamped with the
  queued cut onto that side's pin specs, and from there into the recipe;
- nothing is written for a pin without a part, so every other recipe is
  byte-for-byte what it was.

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
from obvpm_tl_test import mctx  # noqa: E402
from obvpm_tl_test import nodes_assemble as na  # noqa: E402

LINES = [
    "a.mp4",
    "a.mp4 [id=k3x9]",
    "a.mp4 @ 40..80 [id=a2]",
    "a.mp4 @ ..40 [id=a1 crossfade=on]",
    "a.mp4 @ 80 [id=a_3-x crossfade_frames=6 audio_declick=off]",
    "a.mp4 @ 12..250 [crossfade=off level_lock_frames=18]",
    "h3/take one.mp4 @ 5 [id=Zz]",
    # malformed ids are dropped, the line still reads
    "a.mp4 @ 40 [id=bad!id crossfade=on]",
    "a.mp4 [id=]",
]


class LineGrammar(unittest.TestCase):
    def test_part_is_read_and_nothing_else_moves(self):
        (e,) = na._parse_sequence("a.mp4 @ 40..80 [id=a2 crossfade=on]")
        self.assertEqual(e["part"], "a2")
        self.assertEqual((e["clip"], e["enter_override"], e["exit_override"]),
                         ("a.mp4", 40, 80))
        self.assertEqual(e["seam_opts"], {"crossfade": True})
        (plain,) = na._parse_sequence("a.mp4 @ 40..80")
        self.assertIsNone(plain["part"])

    def test_an_id_is_not_a_seam_setting(self):
        with self.assertNoLogs("obvpm.h3", level="WARNING"):
            head, seam = na.split_seam_opts("a.mp4 @ 40 [id=a2]")
        self.assertEqual((head, seam), ("a.mp4 @ 40", {}))

    def test_malformed_ids_are_dropped(self):
        self.assertIsNone(na.line_part("a.mp4 [id=bad!id]"))
        self.assertIsNone(na.line_part("a.mp4 [id=]"))
        self.assertIsNone(na.line_part("a.mp4 [id=%s]" % ("x" * 25)))
        self.assertEqual(na.line_part("a.mp4 [crossfade=on id=k9]"), "k9")

    def test_round_trip(self):
        for line in ("a.mp4 [id=k3x9]", "a.mp4 @ 40..80 [id=a2]",
                     "a.mp4 @ ..40 [id=a1 crossfade=on]", "a.mp4 @ 7",
                     "a.mp4 @ 12..250 [id=q level_lock_frames=18 crossfade=off]"):
            (e,) = na._parse_sequence(line)
            self.assertEqual(na.format_sequence_line(
                e["clip"], e["enter_override"], e["exit_override"],
                seam=e["seam_opts"], part=e["part"]), line)

    def test_the_joint_reads_the_cut_through_an_id(self):
        from obvpm_tl_test import nodes_joint as nj
        self.assertEqual(nj.text_cuts(["a.mp4 @ 40..80 [id=a2]"]), [[40, 80]])
        self.assertEqual(nj.clip_of_line("a.mp4 @ 40..80 [id=a2]"), "a.mp4")


JS_PROBE = r"""
import { readFileSync } from "node:fs";
const SRC = readFileSync(%s, "utf8");
const lift = (name) => {
    const at = SRC.indexOf("function " + name + "(");
    let depth = 0, i = SRC.indexOf("{", at);
    for (;; i += 1) {
        if (SRC[i] === "{") depth += 1;
        if (SRC[i] === "}" && --depth === 0) break;
    }
    return SRC.slice(at, i + 1);
};
const liftConst = (name) => {
    const at = SRC.indexOf("\nconst " + name + " =");
    let depth = 0, i = at + 1;
    for (;; i += 1) {
        const c = SRC[i];
        if ("{[(".includes(c)) depth += 1;
        if ("}])".includes(c)) depth -= 1;
        if (c === ";" && depth === 0) break;
    }
    return SRC.slice(at + 1, i + 1);
};
const body = [
    ...["TL_LINE_RE", "TL_GAP_RE", "TL_LOOP_RE", "TL_OPTS_RE", "TL_SEAM_KEYS",
        "TL_SEAM_ORDER", "TL_TRUE", "TL_FALSE", "TL_PART_KEY", "TL_PART_RE"]
        .map(liftConst),
    ...["tlIsEntryLine", "tlSplitSeamOpts", "tlFormatSeamOpts", "tlParseSequence",
        "tlFormatLine"].map(lift),
    "return { tlParseSequence, tlFormatLine };",
].join("\n");
const { tlParseSequence, tlFormatLine } = new Function(body)();
const lines = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(lines.map((l) => {
    const [e] = tlParseSequence(l);
    return { clip: e.clip, enter: e.enterOverride, exit: e.exitOverride,
             seam: e.seamOpts, part: e.part, line: tlFormatLine(e) };
})));
"""


class JsMirror(unittest.TestCase):
    def test_js_reads_and_writes_lines_as_python_does(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH")
        js = os.path.join(ROOT, "web", "h3_mctx_ui.js").replace("\\", "/")
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "probe.mjs")
            with open(script, "w", encoding="utf-8") as f:
                f.write(JS_PROBE % json.dumps(js))
            got = json.loads(subprocess.check_output(
                [node, script, json.dumps(LINES)], text=True))
        for line, js_e in zip(LINES, got):
            (py,) = na._parse_sequence(line)
            want = {"clip": py["clip"], "enter": py["enter_override"],
                    "exit": py["exit_override"], "seam": py["seam_opts"],
                    "part": py["part"],
                    "line": na.format_sequence_line(
                        py["clip"], py["enter_override"], py["exit_override"],
                        seam=py["seam_opts"], part=py["part"])}
            self.assertEqual(js_e, want, line)


class PinState(unittest.TestCase):
    def parse(self, **state):
        return na.H3Timeline._parse_pin_state(json.dumps(
            dict({"source": "a.mp4", "role": "extend", "window": "22"},
                 **state)))

    def test_parts_are_read(self):
        self.assertEqual(self.parse(part="a2")["part"], "a2")
        st = self.parse(role="bridge", source2="b.mp4", part="a2", part2="b1")
        self.assertEqual((st["part"], st["part2"]), ("a2", "b1"))
        self.assertNotIn("part", self.parse())

    def test_a_bad_part_is_refused(self):
        with self.assertRaises(ValueError):
            self.parse(part="no spaces")

    def test_stamp(self):
        specs = [{"source_id": "a", "place": "before"}]
        na._stamp_part(specs, "a2", 80)
        self.assertEqual((specs[0]["tl_part"], specs[0]["tl_cut"]), ("a2", 80))
        edge = na._stamp_part([{}], "a3", None)
        self.assertEqual((edge[0]["tl_part"], edge[0]["tl_cut"]), ("a3", None))
        self.assertEqual(na._stamp_part([{}], None, 80), [{}])

    def test_the_recipe_carries_it_and_only_then(self):
        base = {"source_id": "a", "source_kind": "clip", "source_start": 17,
                "source_frames": 22, "place": "before", "audio_window": 0}
        plain = json.loads(mctx.serialize_pins([dict(base)]))[0]
        self.assertNotIn("tl_part", plain)
        self.assertNotIn("tl_cut", plain)
        rec = json.loads(mctx.serialize_pins(
            [dict(base, tl_part="a2", tl_cut=None)]))[0]
        self.assertEqual((rec["tl_part"], rec["tl_cut"]), ("a2", None))
        rec = json.loads(mctx.serialize_pins(
            [dict(base, tl_part="a2", tl_cut=39)]))[0]
        self.assertEqual(rec["tl_cut"], 39)
        # the lineage summary does not care
        self.assertEqual(mctx.summarize_pins([dict(base, tl_part="a2")])[:2],
                         ("extends", "a"))


if __name__ == "__main__":
    unittest.main()
