"""handover_edge: a hard-held window hands over a latent step inside the
older clip's decode edge (see nodes_masked.handover_frames).

H3's VAE draws a file's last latent step with nothing after it and its
first two with nothing before them; the newer take decoded the same held
latents mid-file. So new latent-grade masked pins record handover_edge,
and every reader of the recipe moves the join by one step: the extend
delivers 4 more frames and its parent exits 4 earlier; a hard-held
arrival delivers 5 more and its target enters at 5. Takes made before the
flag keep the join they were cut at, and the joint render lays out the
very same latent line either way (the held latents are verbatim).

Run from the pack folder:
  python -m unittest discover -s tests -v
(with the Python that runs ComfyUI -- on the Windows portable build,
python_embeded/python.exe -s)
"""
import json
import os
import sys
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
from obvpm_tl_test import nodes_joint as nj  # noqa: E402
from obvpm_tl_test import nodes_pins  # noqa: E402


def bundle(delivered=124, head=0):
    return {"self_id": "P", "clip": "p.mp4",
            "meta": {"delivered_frames": str(delivered),
                     "pinned_head_frames": str(head)}}


def spec_for(mode, place="before", take_from="tail"):
    (specs,) = nodes_pins.H3MCtxPinSpec().build(
        mctx=bundle(), window=39, take_from=take_from, take_from_frame=0,
        place=place, place_at_frame=0, audio_window=0, mode=mode)
    return specs[0]


class Recipe(unittest.TestCase):
    def test_latent_masked_pins_carry_the_flag(self):
        self.assertTrue(spec_for("masked")["handover_edge"])
        self.assertTrue(spec_for("both", "after", "head")["handover_edge"])
        self.assertFalse(spec_for("guide")["handover_edge"])

    def test_recorded_only_when_set(self):
        on = json.loads(mctx.serialize_pins([spec_for("masked")]))[0]
        off = json.loads(mctx.serialize_pins([spec_for("guide")]))[0]
        self.assertIs(on.get("handover_edge"), True)
        self.assertNotIn("handover_edge", off)

    def test_lineage_summary_moves_the_join_a_step_in(self):
        s = dict(spec_for("masked"), source_start=85, source_frames=39)
        self.assertEqual(mctx.summarize_pins([s]), ("extends", "P", 120))
        s["handover_edge"] = False
        self.assertEqual(mctx.summarize_pins([s]), ("extends", "P", 124))
        a = dict(spec_for("both", "after", "head"), source_start=0,
                 source_frames=39)
        self.assertEqual(mctx.summarize_pins([a]), ("prepends", "P", 5))

    def test_trim_delivers_the_extra_frames(self):
        before = {"place": "before", "covered": 39, "spec": spec_for("masked")}
        after = {"place": "after", "covered": 39,
                 "spec": spec_for("both", "after", "head")}
        self.assertEqual(nodes_pins.pins_trim_totals([before]), (35, 0))
        self.assertEqual(nodes_pins.pins_trim_totals([after]), (0, 34))
        old = {"place": "before", "covered": 39,
               "spec": dict(spec_for("masked"), handover_edge=False)}
        self.assertEqual(nodes_pins.pins_trim_totals([old]), (39, 0))


def head(self_id, raw, head_f=0, tail_f=0, pins=(), relation="", parent="",
         join=0):
    return {"self_id": self_id, "raw_frames": str(raw),
            "delivered_frames": str(raw - head_f - tail_f),
            "pinned_head_frames": str(head_f),
            "pinned_tail_frames": str(tail_f),
            "relation": relation, "parent_id": parent,
            "parent_join_frame": str(join), "pins": json.dumps(list(pins))}


def extend_of(parent_id, parent_raw, raw=141, covered=39, edge=True):
    start = parent_raw - covered
    hand = covered - 4 if edge else covered
    pin = {"source_id": parent_id, "source_kind": "clip", "place": "before",
           "source_start": start, "source_frames": covered, "mode": "masked"}
    if edge:
        pin["handover_edge"] = True
    return head("C_" + parent_id, raw, head_f=hand, pins=[pin],
                relation="extends", parent=parent_id, join=start + hand)


class Seam(unittest.TestCase):
    def test_parent_exits_a_step_short_and_it_is_still_seamless(self):
        p = head("P", 124)
        exit_f, enter, note = na._derive_seam(
            {"clip": "p", "header": p}, {"clip": "c", "header": extend_of("P", 124)})
        self.assertEqual((exit_f, enter), (120, 0))
        self.assertIn("seamless", note)

    def test_old_takes_keep_their_join(self):
        p = head("P", 124)
        exit_f, enter, note = na._derive_seam(
            {"clip": "p", "header": p},
            {"clip": "c", "header": extend_of("P", 124, edge=False)})
        self.assertEqual((exit_f, enter, note), (None, 0, "extends: seamless"))


class Joint(unittest.TestCase):
    def test_the_derived_exit_is_not_a_jump_cut(self):
        p, c = head("P", 141), extend_of("P", 141)
        for text in ("p\nc", "p @ ..137\nc"):
            segs = nj.split_segments(nj.sequence_items(text), [p, c])
            self.assertEqual([s["clips"] for s in segs], [[0, 1]], text)

    def test_the_joint_line_is_the_same_either_way(self):
        lay = {}
        for edge in (False, True):
            p, c = head("P", 141), extend_of("P", 141, edge=edge)
            lines = ["p", "c"]
            ticks = [nj.fr.audio_total(141)] * 2
            cuts = nj.shown_cuts(lines, [p, c], [True, True])
            lay[edge] = nj.timeline_layout(lines, [p, c], ticks, cuts=cuts)
        for key in ("total_steps", "total_frames", "steps"):
            self.assertEqual(lay[True][key], lay[False][key], key)
        # ...and the step the take now hands over in is the take's: the
        # same held latents either way, its conditioning follows the join
        old, new = lay[False]["owner_steps"], lay[True]["owner_steps"]
        moved = [k for k, (a, b) in enumerate(zip(old, new)) if a != b]
        self.assertEqual(len(moved), 1)
        self.assertEqual((old[moved[0]], new[moved[0]]), (0, 1))
        self.assertEqual(old[moved[0] + 1], 1)


if __name__ == "__main__":
    unittest.main()
