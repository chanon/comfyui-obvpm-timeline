"""Getting a take's CONDITIONING back, for a refine pass.

One source: `clip_00086.cond.safetensors`, the encoded tensors the take
was actually sampled with. Exact, and needs no model. Without it the
take is refused BY NAME: a refine pass that quietly conditions on the
wrong thing is the failure this whole area exists to prevent, and the
take is still perfectly playable -- it simply cannot be refined.

WHY NOT REBUILD. Two recipes were considered and both decay.

Re-running the prompt graph (the sidecar carries it) fails months later:
a node class the tracer does not know, one of our own loaders changing
its resize, `pasted/image.png` overwritten by the next paste.

Re-encoding the reference PIXELS a recorder node had saved beside the
take (this module's second source until 2026-09-14) was exact to 2.6e-5 for
image and audio references, but it could never cover video references,
and reference adapters -- "refmods", pooled or trained latents that
other packs append to `minimax_refs` -- have no pixels to record at
all. A conditioning that is partly recipe and partly unrecoverable is
worse than none: it rebuilds something that looks right and is missing
a block. The `.cond` stores the whole `minimax_refs` list as it was,
refmod blocks included, so it is the one record that stays faithful
whatever the conditioning was built from.
"""

import logging
import os

from . import condstore

_LOG = logging.getLogger("obvpm.h3")


def has_conditioning(video_path):
    """Whether this take can be refined at all."""
    return condstore.has_conditioning(video_path)


def load_for_clip(video_path):
    """The saved conditioning of a take, or a refusal that names it."""
    name = os.path.basename(video_path)
    if not has_conditioning(video_path):
        raise ValueError(
            "%s has no .cond beside it, so there is nothing to condition a "
            "refine on. It was saved with save_conditioning off or with "
            "no conditioning wired, or the .cond was deleted. The take is "
            "fine; it just cannot be refined." % name)
    conditioning, _meta = condstore.load_conditioning(
        condstore.cond_path(video_path))
    _LOG.info("obvpm.h3: %s -- conditioning from its .cond", name)
    return conditioning
