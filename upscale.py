"""Profile bookkeeping for the upscale/refine pass.

A PROFILE is one named upscale configuration applied to one timeline. It
owns an output folder, and that folder is the whole state: the refined
clips live in it, in delivery order, alongside a `profile.json` recording
what settings produced them and which source each came from.

Why a folder rather than a suffix on each clip:

- **The source stays immutable.** A refine reads the original take and
  writes somewhere else, so the pass can be re-run, abandoned, or run
  twice at different settings without ever touching what it read.
- **The existing timeline works unchanged.** Frame counts are identical,
  so pointing `base_folder` at the profile folder plays the refined cut
  with no other edit.
- **Resume is a lookup.** The pass costs roughly a full re-render of a
  timeline, so it WILL be interrupted; the manifest names each refined
  clip by its SOURCE and by the neighbours it was pinned to, so "where
  was I" -- and "what did this timeline edit change" -- are answered
  from the record, not a scan or a heuristic.

The config hash is what stops a half-finished profile being completed
under different settings -- a timeline whose first six clips were refined
at one sigma and whose last six were refined at another is a subtle,
expensive kind of broken, and the folder cannot tell you that happened
unless it recorded what it was doing.
"""

import hashlib
import json
import os
import re

FORMAT = "obvpm_upscale_v1"
MANIFEST = "profile.json"
SUBFOLDER = "_upscale"

# what the pass IS, for the resume check. Anything here changes the
# picture the pass produces, so changing one mid-timeline is the mistake
# the hash exists to catch. The SEQUENCE is deliberately absent: a
# refined clip is a property of its source clip and the neighbours it was
# pinned to, not of where the timeline happens to put it, so editing the
# timeline must reuse what is already refined and redo only what the edit
# actually changed (see `refined_for`).
CONFIG_KEYS = ("first_sigma", "upscale")
# hashed only when set (see config_hash): `junction` = [ramp, edge, deep]
# the refine holds its junction windows with, in place of the recipe's
OPTIONAL_CONFIG_KEYS = ("junction", "junction_mode")


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


AUTO_PREFIX = "refine"


def choose_profile(base_dir, config, clips=()):
    """The profile folder a configuration belongs to: (name, existing).

    Automatic profile management, for a Loop Start left without a name.
    A profile IS its settings hash, so the folder whose manifest carries
    this config's hash is this config's profile -- one still missing some
    of `clips` is preferred over one that has them all, and a complete
    one is still returned rather than duplicated, so the caller can say
    "already done" instead of quietly refining a second copy. With no
    match the next free `refineNN` is created: changing a setting
    therefore lands in a new folder by itself, which is what the hash
    check used to refuse by hand. The timeline is not part of the match:
    the same settings over an edited timeline is the same profile, with
    only the changed clips left to refine.

    `base_dir` is the project's ABSOLUTE folder; the name comes back
    bare, for `profile_folder`.
    """
    want = config_hash(config)
    root = os.path.join(base_dir, SUBFOLDER)
    names = sorted(os.listdir(root)) if os.path.isdir(root) else []
    matching = []
    for name in names:
        folder = os.path.join(root, name)
        if not os.path.isfile(manifest_path(folder)):
            continue
        try:
            document = read_manifest(folder)
        except (ValueError, OSError):
            continue
        if document_hash(document) == want:
            matching.append((name, document))
    for name, document in matching:
        if any(entry_for(document, c) is None for c in clips):
            return name, True
    if matching:
        return matching[0][0], True
    n = 1
    while True:
        name = "%s%02d" % (AUTO_PREFIX, n)
        if not os.path.exists(os.path.join(root, name)):
            return name, False
        n += 1


def config_hash(config):
    """A stable digest of the settings that define a profile's output.

    Settings added later join the payload only when SET, so a profile
    written before they existed keeps hashing the same -- an unramped
    junction is the recorded hold, exactly what those profiles did.
    """
    payload = {k: config.get(k) for k in CONFIG_KEYS}
    for key in OPTIONAL_CONFIG_KEYS:
        if config.get(key):
            payload[key] = config[key]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_path(folder):
    return os.path.join(folder, MANIFEST)


def read_manifest(folder):
    """The profile's record, or a fresh empty one."""
    path = manifest_path(folder)
    if not os.path.isfile(path):
        return {"format": FORMAT, "hash": None, "config": {}, "entries": []}
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
    next run simply refines again, overwriting nothing. The other order
    would leave the manifest claiming a clip that is not there, and the
    junction pin would then load a file that does not exist.
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


def document_hash(document):
    """The settings hash a manifest stands for, old manifests included.

    Manifests written before 2026-09-02 hashed the sequence in as well;
    their recorded hash can never match a settings-only one, so it is
    recomputed from the config they stored, which is the same thing under
    the current definition.
    """
    config = document.get("config") or {}
    if "sequence" in config:
        return config_hash(config)
    return document.get("hash")


def validate(document, config):
    """Refuse to continue a profile whose settings have moved.

    Returns the hash to record. A profile with no entries yet has nothing
    to be inconsistent with, so it simply adopts whatever it is given --
    that is what makes editing settings before the first clip free.
    """
    want = config_hash(config)
    have = document_hash(document)
    if not document.get("entries") or have is None:
        return want
    if have != want:
        raise ValueError(
            "H3 upscale: profile already holds %d refined clip(s) made "
            "with different settings (recorded %s, now %s). Finish it as "
            "it is, start a new profile, or delete the folder -- mixing "
            "two settings across one timeline is the failure this check "
            "exists to prevent."
            % (len(document["entries"]), have[:12], want[:12]))
    return want


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


def refined_for(document, folder, clips, deps):
    """Which positions of the CURRENT timeline are already refined.

    Returns {index: entry}. A position counts when its source has an
    entry whose file is still there AND every junction the current
    timeline needs was the one it was refined with: for each neighbour
    j it must pin to, the entry was pinned to j's source and to the
    refined output j has NOW. So re-refining a clip invalidates
    exactly the clips that were pinned to its old rendering, and
    inserting a clip that creates a new held join invalidates the one
    clip that join lands on -- nothing else is redone.

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

    Only the leading run of finished positions of the timeline the
    profile was last run with. A profile stopped halfway has a hole in
    the middle of the delivery, and splicing across it would quietly hand
    back a cut with a clip missing -- which looks like a finished cut.
    Stopping early looks like what it is.
    """
    lines = document.get("lines") or []
    sources = [line.split("@")[0].strip() for line in lines]
    entries = []
    for path in sources:
        entry = entry_for(document, path)
        if not entry:
            break
        entries.append(entry)
    out = []
    for k, entry in enumerate(entries):
        suffix = lines[k][len(sources[k]):]
        # A junction the refine made is authoritative for where the two
        # refined clips change hands: the seam is derived from their
        # sidecars, and a ramped hold moves it. A marker copied from the
        # source would put the cut back where the SOURCE joined, so the
        # marker on that side of a held junction is dropped. Markers on
        # cuts (no junction) are the user's decision and stay.
        nxt = entries[k + 1] if k + 1 < len(entries) else None
        prev = entries[k - 1] if k > 0 else None
        # the next clip pins to this one (an extend): this clip's exit is
        # derived, so its exit marker goes. The previous clip pins to
        # this one (a prepend / a bridge's departing half): this clip's
        # enter is derived, so its enter marker goes.
        drop_exit = bool(nxt and sources[k] in (nxt.get("pinned_to") or {}))
        drop_enter = bool(prev and sources[k] in (prev.get("pinned_to") or {}))
        suffix = shift_markers(suffix, int(entry.get("head_shift", 0) or 0),
                               drop_enter=drop_enter, drop_exit=drop_exit)
        out.append("%s/%s%s" % (rel_folder, entry["output"], suffix))
    return "\n".join(out)


_MARKER = re.compile(r"^(\s*@\s*)(\d*)(\.\.)?(\d*)(?=\s|\[|$)")


def shift_markers(suffix, shift, drop_enter=False, drop_exit=False):
    """A sequence line's tail (` @ a..b [opts]`) with the cut moved.

    Cut markers index a clip's DELIVERED frames; a refined clip whose
    head starts `shift` frames earlier than its source's needs every
    marker moved by `shift`. Dropped markers are removed outright; a
    marker left with nothing to say disappears with its `@`.
    """
    m = _MARKER.match(suffix or "")
    if not m:
        return suffix or ""
    enter, dots, exit_ = m.group(2), m.group(3), m.group(4)
    rest = suffix[m.end():]
    # "@ N" alone is an ENTER marker
    if enter and not dots:
        enter, exit_ = enter, ""
    if drop_enter:
        enter = ""
    if drop_exit:
        exit_ = ""
    if enter:
        enter = str(max(0, int(enter) + int(shift)))
    if exit_:
        exit_ = str(max(0, int(exit_) + int(shift)))
    if not enter and not exit_:
        return rest.lstrip() and " " + rest.lstrip() or ""
    if exit_:
        marker = " @ %s..%s" % (enter, exit_)
    else:
        marker = " @ %s" % enter
    return marker + rest


def record(document, source, output, config, hash_value, pinned_to=None,
           head_shift=0):
    """Add or replace one refined clip. Callers write the manifest afterwards.

    Keyed by source: refining a clip again replaces its entry (the old
    file is left where it is). `pinned_to` maps each neighbour SOURCE
    this rendering was pinned to onto the neighbour's refined OUTPUT at
    the time, which is what `refined_for` checks continuity against.
    `head_shift` is how many more delivered frames this rendering has
    at its head than its source (a ramped junction ships part of the
    window); `sequence_text` moves the source's cut markers by it.
    """
    document["format"] = FORMAT
    document["hash"] = hash_value
    document["config"] = {k: config.get(k) for k in CONFIG_KEYS}
    for key in OPTIONAL_CONFIG_KEYS:
        if config.get(key):
            document["config"][key] = config[key]
    entry = {"source": source, "output": output,
             "pinned_to": dict(pinned_to or {})}
    if int(head_shift or 0):
        entry["head_shift"] = int(head_shift)
    entries = document.setdefault("entries", [])
    for k, old in enumerate(entries):
        if old.get("source") == source:
            entries[k] = entry
            break
    else:
        entries.append(entry)
    return document
