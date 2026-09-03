"""The wire types this subpackage DEFINES. One place, because they are API.

A type name is the most public thing a node pack has: the moment someone
else's workflow holds a link of that type, the name cannot change without
breaking it. Everything here is therefore prefixed `OBVPM_` -- decided
2026-08-20, before the first release, which is the only moment the rename
was free.

That REVERSES an earlier decision for `MCTX` (DESIGN.md 4), which argued
for an unbranded name so other packs could emit and accept the same wire
-- "the collision risk that motivates prefixed names is here the goal".
The counter-argument won: a user looking at a graph should be able to see
where a type comes from, and an unbranded `MCTX` in someone's node list
says nothing about who is responsible for it. The openness the old
argument wanted lives in the FILE format instead, which is the durable
artifact: `mctx_v1` sidecars stay unbranded and any pack may read or
write them. Interop through files, attribution through wires.

Why `OBVPM_H3_` and not just `OBVPM_`: all three carry H3's own latents.
`MCTX` holds [1,24,T,h/16,w/16] video and [1,32,2,T40] audio; `PINS`
holds slices of them plus step/overhang bookkeeping off the 17k+5 grid;
and `PINSPECS` looks like pure data but its `source` field IS the bundle,
so that wire transports H3 latents too. The pack root already hosts
model-agnostic nodes, so a second family is plausible -- and a bare
`OBVPM_MCTX` would be the wrong name to be stuck with when it arrives.
`OBVPM_BUNDLE` keeps the short form precisely because it is generic: it
carries {name: value} and nothing about any model.

Note what is NOT renamed: `mctx.FORMAT` ("mctx_v1") and the
`.mctx.safetensors` suffix. Those identify the on-disk format, not our
wires, and changing them would orphan every sidecar already written.
"""

# latents + verified identity/lineage for one clip (DESIGN.md 4)
MCTX = "OBVPM_H3_MCTX"

# a list of pin DESCRIPTIONS -- pure data, chainable like LoRA stacks
PINSPECS = "OBVPM_H3_PINSPECS"

# the RESOLVED pins Apply produces: what was actually sliced
PINS = "OBVPM_H3_PINS"

# the reference PIXELS a take was conditioned on, on their way past
# H3RecordReferences to the save node -- see refstore.py
REFS = "OBVPM_H3_REFS"

# the upscale loop's open->close handshake: which node opened it, and the
# iteration state it carries (see nodes_loop.py)
LOOP = "OBVPM_H3_LOOP"
