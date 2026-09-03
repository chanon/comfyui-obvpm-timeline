# comfyui-obvpm-h3

Non-linear clip composition for **MiniMax H3** in ComfyUI. Build a long
piece out of several generations, without the joins showing.

Every take is saved as a **clip pair** — the MP4 plus a
`.mctx.safetensors` sidecar holding its full latents and its lineage.
Because the latents survive, the next generation can be conditioned on a
window of the previous one's actual latents rather than on re-encoded
pixels: continuation is exact, not approximate, with no VAE round trip.
From any saved take you can **extend** forward, **prepend** a lead-in,
or **bridge** the gap between two clips — today, or months later.
Clips with no sidecar (imports, older footage) still work, through a
VAE-encoded fallback at pixel grade, and the pack always records which
grade a join is.

A **timeline widget** then composes the takes: drag to reorder, cut,
open gaps, preview the real cut, and export it. Assembly is a smart-cut
— video packets are stream-copied bit for bit wherever possible, and
only the frames around a mid-clip seam are re-encoded. The next
generation can be aimed from the timeline itself, so composing and
generating are one activity rather than two.

Seams are **measured and repairable**: each join reports whether motion
actually flows through it, and level lock / crossfade / audio de-click
can be set per join.

Full guide: **[docs/h3.md](docs/h3.md)** · per-node reference:
**[docs/h3-nodes.md](docs/h3-nodes.md)**

## Installation

Clone (or copy) this folder into `ComfyUI/custom_nodes`:

```
cd ComfyUI/custom_nodes
git clone https://github.com/chanon/comfyui-obvpm-h3
```

Restart ComfyUI. No extra Python dependencies are required. A ComfyUI
from 2026-08-13 or later is needed: the pack pins through core's native
MiniMax H3 keyframe anchoring, and older cores are refused with a clear
message at apply time.

The example workflows also use the companion pack
**[comfyui-obvpm](https://github.com/chanon/comfyui-obvpm)** for the
reference-image loader (Load Images & Compose), Value Presets, gates,
lazy switches and bundles. Install both for the full experience; this
pack's nodes run without it.

Every node in this pack is listed with **(obvpm)** after its name, so
searching the node menu for `obvpm` finds all of them. The names used
throughout this documentation leave that suffix off.

## License

[GPL-3.0](LICENSE). See
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md) for acknowledgments.
