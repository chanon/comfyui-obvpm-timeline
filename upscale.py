"""Profile bookkeeping for the joint refine pass.

A PROFILE is one joint refine of one timeline. It owns an output folder,
and that folder is the whole state: the sampled timeline latent
(`joint.mctx.safetensors`, written by H3 Joint Store) lives in it, the
refined clips are sliced out of that latent into it, and a `profile.json`
records which refined clip came from which source and which sampling it
was cut from.

Why a folder rather than a suffix on each clip:

- **The source stays immutable.** A refine reads the original takes and
  writes somewhere else, so the pass can be re-run, abandoned, or run
  twice at different settings without ever touching what it read.
- **The existing timeline works unchanged.** Frame counts are identical,
  so pointing `base_folder` at the profile folder plays the refined cut
  with no other edit.
- **Resume is a lookup.** The pass is one long sampling plus a decode
  and save per clip, any of which can be interrupted; the manifest names
  each refined clip by its SOURCE and by the joint sampling it was cut
  from, so "where was I" is answered from the record, not a scan.

The joint file's stamp is what stops a profile mixing two samplings: a
clip cut from an earlier joint latent in the same folder does not count
as done once the timeline has been sampled again.
"""

import json
import os
import re

FORMAT = "obvpm_upscale_v1"
MANIFEST = "profile.json"
SUBFOLDER = "_upscale"


def profile_folder(base_folder, profile):
    """Output-relative folder for one profile of one project."""
    base = str(base_folder or "").strip().strip("/\\")
    name = str(profile or "").strip().strip("/\\")
    if not name:
        raise ValueError(
            "H3 upscale: the profile needs a name -- it is the folder the "
            "refined clips are written to.")
    if any(part in ("..", "") for part in name.replace("\\", "/").split("/")):
        raise ValueError("H3 upscale: %r is not a usable profile name." % name)
    parts = [p for p in (base, SUBFOLDER, name) if p]
    return "/".join(parts)


def manifest_path(folder):
    return os.path.join(folder, MANIFEST)


def read_manifest(folder):
    """The profile's record, or a fresh empty one."""
    path = manifest_path(folder)
    if not os.path.isfile(path):
        return {"format": FORMAT, "entries": []}
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("format") != FORMAT:
        raise ValueError(
            "%s: unknown upscale profile format %r -- written by a newer "
            "build?" % (path, document.get("format")))
    document.setdefault("entries", [])
    return document


def write_manifest(folder, document):
    """Replace the manifest atomically.

    Written AFTER the clip it describes, so a crash between the two
    leaves a refined clip the manifest does not know about -- which the
    next run simply slices again, overwriting nothing. The other order
    would leave the manifest claiming a clip that is not there.
    """
    os.makedirs(folder, exist_ok=True)
    path = manifest_path(folder)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def parse_sequence(text):
    """The timeline's `sequence` widget -> clip paths, in delivery order.

    The same shape H3Assemble reads: one output-relative clip per line,
    `# ...` ignored, and an optional ` @ N` cut marker which is about
    where a clip ENTERS the cut and is not part of its identity.
    """
    clips = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        clips.append(line.split("@")[0].strip())
    return [c for c in clips if c]


def parse_sequence_lines(text):
    """The same lines `parse_sequence` reads, markers still attached.

    Aligned with it by construction -- same filter, same order -- so
    position N in one is position N in the other. The assembly needs the
    markers back: a refined clip has the same frame count as its source,
    so a cut that entered at frame 141 still enters at frame 141.
    """
    out = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.split("@")[0].strip():
            out.append(line)
    return out


def entry_for(document, source):
    """The refined clip written for one SOURCE clip, or None.

    Entries are keyed by source rather than by delivery position: a
    position is a fact about the current timeline, a refined clip is a
    fact about the take it was made from.
    """
    for entry in document.get("entries") or []:
        if entry.get("source") == source:
            return entry
    return None


def refined_for(document, folder, clips, deps, stamp):
    """Which positions of the CURRENT timeline are already refined.

    Returns {index: entry}. A position counts when its source has an
    entry cut from THIS joint sampling (`stamp`), whose file is still
    there, AND every held join the timeline needs was made against the
    refined output its neighbour has NOW: for each neighbour j it pins
    to, the entry was pinned to j's source and to j's current output.
    So re-slicing a clip invalidates exactly the clips whose lineage
    pointed at its old rendering, and nothing else is redone.

    `deps` is {index: [neighbour indices]} in the current timeline;
    it is acyclic, so validity resolves neighbours-first.
    """
    valid = {}

    def ok(i, stack=()):
        if i in valid:
            return True
        if i in stack:
            return False
        entry = entry_for(document, clips[i])
        if not entry or not entry.get("output"):
            return False
        if entry.get("joint") != stamp:
            return False
        if not os.path.isfile(os.path.join(folder, str(entry["output"]))):
            return False
        pinned = entry.get("pinned_to") or {}
        for j in deps.get(i, []):
            j = j[0] if isinstance(j, (tuple, list)) else int(j)
            if not ok(j, stack + (i,)):
                return False
            if pinned.get(clips[j]) != valid[j]["output"]:
                return False
        valid[i] = entry
        return True

    for i in range(len(clips)):
        ok(i)
    return valid


def sequence_text(document, rel_folder):
    """The refined clips as a playable sequence, in DELIVERY order.

    The timeline's own lines with each source replaced by its refined
    rendering, markers verbatim: a refined clip covers its source's raw
    span and carries its pins, so the cut is the source's cut. Only the
    leading run of finished positions is returned -- a profile stopped
    halfway has a hole in the middle of the delivery, and splicing
    across it would quietly hand back a cut with a clip missing. Stopping
    early looks like what it is.
    """
    out = []
    for line in document.get("lines") or []:
        source = line.split("@")[0].strip()
        entry = entry_for(document, source)
        if not entry:
            break
        suffix = line[len(line.split("@")[0].rstrip()):]
        out.append("%s/%s%s" % (rel_folder, entry["output"], suffix))
    return "\n".join(out)


# a sequence line's tail: ` @ enter`, ` @ enter..exit`, ` @ ..exit`
_MARKER = re.compile(r"^(\s*@\s*)(\d*)(\.\.)?(\d*)(?=\s|\[|$)")


def record(document, source, output, stamp, pinned_to=None):
    """Add or replace one refined clip. Callers write the manifest afterwards.

    Keyed by source: slicing a clip again replaces its entry (the old
    file is left where it is). `stamp` names the joint sampling the clip
    was cut from; `pinned_to` maps each neighbour SOURCE this rendering's
    lineage points at onto the neighbour's refined OUTPUT at the time,
    which is what `refined_for` checks continuity against.
    """
    document["format"] = FORMAT
    entry = {"source": source, "output": output, "joint": stamp,
             "pinned_to": dict(pinned_to or {})}
    entries = document.setdefault("entries", [])
    for k, old in enumerate(entries):
        if old.get("source") == source:
            entries[k] = entry
            break
    else:
        entries.append(entry)
    return document
