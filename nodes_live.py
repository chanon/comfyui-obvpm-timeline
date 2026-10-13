"""H3 Live Preview: the sampler's picture, every step, as a short clip on
the take's own timeline.

A model patch (an OUTER_SAMPLE wrapper, so it sees every sampling pass of
the model it is applied to). At each step the denoised estimate's video
latent is decoded one picture per latent step -- by a tiny 2D decoder from
models/vae_approx (taeh3) or, without one, by the latent format's RGB
factors -- and each picture is HELD for exactly the frames its step covers
on H3's grid (1, 4, 4, 4, 4 per five steps; frames.py). So the clip has the
take's own frame count at 24 fps: no fps to set, and it lines up frame for
frame with what the save node will write. A TEMPORAL tiny VAE (madebyollin's
taeh3, a TAEHV) is decoded by core's own H3 code instead and gives every
frame by itself. Both are published as taeh3.safetensors, so a file is told
apart by its layers, never by its name.

The whole latent is shown, pinned frames included: the parent's frames at
the head of an extend (and the target's at the tail of a prepend) are where
a seam is made, and watching them while sampling is how a bad join is seen
early. How many frames are pinned at each end rides along with every
message (head / tail), taken from the pins the H3 MCtx Apply Pins node
prepared for a latent of this length (note_pins), so the Result Preview can
place the finished takes inside this timeline.

Encoded as H.264 on the CPU (libx264), on a worker thread: sampling never
waits for it, and a step that arrives while the last one is still encoding
replaces it (only the newest picture matters). Sent over the websocket as
`obvpm.h3.live`.
"""

import base64
import io
import logging
import os
import threading

import torch

import folder_paths

from . import frames as fr

_LOG = logging.getLogger("obvpm.h3")

EVENT = "obvpm.h3.live"
WRAPPER_KEY = "obvpm_h3_live"
LATENT2RGB = "latent RGB (no tiny VAE)"
CRF = 26

# ---- the pins Apply Pins prepared, by latent length ------------------------
# The sampler hands the wrapper a latent, not the pins; Apply Pins runs
# before it in the same Run and notes what it prepared. Keyed by the video
# latent's step count: the pass sampling a latent of that length is the one
# those pins were prepared for (an upscale's refine samples the whole cut,
# a different length, and gets none). Kept across Runs too: an Apply Pins
# served from cache did not run again, and it had exactly these inputs.
_PINS = {}
_PINS_MAX = 32
_PINS_GUARD = threading.Lock()


def note_pins(latent_t, pins):
    """Apply Pins: these pins were prepared for a video latent of
    `latent_t` steps."""
    try:
        t = int(latent_t)
    except (TypeError, ValueError):
        return
    with _PINS_GUARD:
        _PINS.pop(t, None)
        _PINS[t] = list(pins or [])
        while len(_PINS) > _PINS_MAX:
            _PINS.pop(next(iter(_PINS)))


def pinned_frames(latent_t):
    """(head, tail) pixel frames the save node trims from a take sampled
    on a latent of `latent_t` steps; (0, 0) when no pins are known."""
    with _PINS_GUARD:
        pins = _PINS.get(int(latent_t))
    if not pins:
        return 0, 0
    from .nodes_pins import pins_trim_totals
    try:
        head, tail = pins_trim_totals(pins)
    except Exception as exc:
        _LOG.warning("obvpm.h3 live preview: pins unreadable (%s)", exc)
        return 0, 0
    return int(head), int(tail)


# ---- decoders --------------------------------------------------------------
def _tiny_decoder(sd):
    """A TAESD-style 2D decoder (Clamp, conv, ReLU, Blocks, Upsamples, ...)
    built from the layer layout of its state dict, so any width works --
    core's TAESD decoder is fixed at 64 channels and taeh3's runs at 96.
    Layer i is a conv when it has `i.weight`, a Block when it has
    `i.conv.0.weight`; the gaps are the parameter-free layers: the input
    clamp first, a ReLU right after the first conv, an Upsample elsewhere."""
    from torch import nn
    from comfy.taesd.taesd import Block, Clamp
    import comfy.ops

    prefix = ""
    first = next(iter(sd))
    if not first.split(".")[0].isdigit():
        prefix = first.split(".")[0] + "."
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    last = max(int(k.split(".")[0]) for k in sd)
    layers = []
    first_conv = None
    for i in range(last + 1):
        if "%d.conv.0.weight" % i in sd:
            n_out, n_in = sd["%d.conv.0.weight" % i].shape[:2]
            layers.append(Block(int(n_in), int(n_out)))
        elif "%d.weight" % i in sd:
            n_out, n_in = sd["%d.weight" % i].shape[:2]
            layers.append(comfy.ops.disable_weight_init.Conv2d(
                int(n_in), int(n_out), 3, padding=1,
                bias=("%d.bias" % i) in sd))
            if first_conv is None:
                first_conv = i
        elif i == 0:
            layers.append(Clamp())
        elif first_conv is not None and i == first_conv + 1:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Upsample(scale_factor=2))
    model = nn.Sequential(*layers)
    model.load_state_dict(sd)
    latent_channels = int(sd["%d.weight" % first_conv].shape[1])
    return model, latent_channels


class _Decoder:
    """Video latent [C, T, H, W] -> uint8 pictures [n, h, w, 3] on the CPU,
    at most `max_size` on their long side, and how many frames each one is
    shown for (`holds`, summing to fr.pixel_frames(T)).

    Which decoder a file is comes from its CONTENT, not its name: both
    tiny VAEs for H3 are published as taeh3.safetensors.
      temporal (madebyollin's TAEHV, "decoder.22.bias"): core's own H3
        decode, every frame of the take -- each shown once
      2D (Kijai's quick TAE): one picture per latent step, each held for
        the frames its step covers
      neither (or no file): the latent's RGB factors, one per step"""

    _cache = {}

    def __init__(self, name, latent_format, device):
        self.device = device
        self.kind = None
        self.model = None
        self.factors = None
        if name and name != LATENT2RGB:
            path = folder_paths.get_full_path("vae_approx", name)
            # the file itself, not just its name: swapping one taeh3 for
            # the other under the same name must not serve the old model
            try:
                st = os.stat(path) if path else None
            except OSError:
                st = None
            key = (path, str(device),
                   (st.st_size, st.st_mtime_ns) if st else None)
            if key in _Decoder._cache:
                self.kind, self.model = _Decoder._cache[key]
            elif path:
                try:
                    import comfy.utils
                    sd = comfy.utils.load_torch_file(path, safe_load=True)
                    dtype = torch.float16 if torch.device(device).type == "cuda" \
                        else torch.float32
                    if "decoder.22.bias" in sd:
                        from comfy.taesd.taehv import TAEHV
                        channels = int(sd["decoder.1.weight"].shape[1])
                        model = TAEHV(latent_channels=channels)
                        model.load_state_dict(sd)
                        if not model.is_h3:
                            raise ValueError("a temporal tiny VAE for %d-channel "
                                             "latents, not H3's 24" % channels)
                        self.kind = "temporal"
                    else:
                        model, channels = _tiny_decoder(sd)
                        self.kind = "2d"
                    self.model = (model.eval().to(device=device, dtype=dtype),
                                  channels, dtype)
                    _Decoder._cache[key] = (self.kind, self.model)
                except Exception as exc:
                    _LOG.warning("obvpm.h3 live preview: %s cannot be used "
                                 "(%s); showing latent colours instead",
                                 name, exc)
                    self.kind = self.model = None
        if self.model is None:
            f = getattr(latent_format, "latent_rgb_factors", None)
            if f is not None:
                self.kind = "rgb"
                self.factors = (torch.tensor(f, dtype=torch.float32),
                                getattr(latent_format, "latent_rgb_factors_bias",
                                        None))

    def __call__(self, video, max_size):
        c, t, h, w = video.shape
        frames = fr.pixel_frames(t)
        holds = hold_counts(t)
        if self.kind in ("temporal", "2d"):
            model, channels, dtype = self.model
            if c != channels:
                raise ValueError("the tiny VAE decodes %d channels, this latent "
                                 "has %d" % (channels, c))
        if self.kind == "temporal":
            with torch.no_grad():
                y = model.decode(video[None].to(device=self.device, dtype=dtype))
            rgb = y[0].float().movedim(0, -1).cpu()      # [n, h, w, 3]
            # core's H3 decode gives the take's own frame count; were it ever
            # a frame off, the timeline still has to be the take's
            if rgb.shape[0] >= frames:
                rgb = rgb[:frames]
            else:
                rgb = torch.cat([rgb, rgb[-1:].expand(frames - rgb.shape[0],
                                                      *rgb.shape[1:])], 0)
            holds = [1] * frames
        elif self.kind == "2d":
            out = []
            x = video.movedim(1, 0)                       # [T, C, H, W]
            with torch.no_grad():
                for i in range(0, t, 8):
                    y = model(x[i:i + 8].to(device=self.device, dtype=dtype))
                    out.append(y.float().clamp(0, 1).cpu())
            rgb = torch.cat(out, 0).movedim(1, -1)        # [T, h, w, 3]
        elif self.kind == "rgb":
            factors, bias = self.factors
            x = video.float().cpu().movedim(0, -1)        # [T, H, W, C]
            rgb = torch.nn.functional.linear(
                x, factors.transpose(0, 1),
                None if bias is None else torch.tensor(bias, dtype=torch.float32))
            rgb = ((rgb + 1) / 2).clamp(0, 1)
        else:
            raise ValueError("no way to decode this latent")
        th, tw = rgb.shape[1:3]
        scale = min(1.0, float(max_size) / max(th, tw))
        nh = max(2, int(round(th * scale / 2)) * 2)
        nw = max(2, int(round(tw * scale / 2)) * 2)
        if (nh, nw) != (th, tw):
            rgb = torch.nn.functional.interpolate(
                rgb.movedim(-1, 1), size=(nh, nw), mode="bilinear",
                antialias=True, align_corners=False).movedim(1, -1)
        return rgb.mul(255).round().clamp(0, 255).to(torch.uint8), holds


# ---- the clip ----------------------------------------------------------------
def hold_counts(latent_t):
    """How many frames each latent step's picture is shown for: the frames
    that step covers on H3's grid. Sums to fr.pixel_frames(latent_t)."""
    return [fr.FRAME_PER_TOKEN[k % fr.LATENTS_PER_GROUP]
            for k in range(int(latent_t))]


def encode_clip(pictures, holds):
    """uint8 [n, h, w, 3] -> MP4 bytes at 24 fps, picture k shown for
    holds[k] frames (libx264; the repeats cost next to nothing)."""
    import av
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as box:
        s = box.add_stream("libx264", rate=fr.FPS)
        s.width = int(pictures.shape[2])
        s.height = int(pictures.shape[1])
        s.pix_fmt = "yuv420p"
        s.options = {"crf": str(CRF), "preset": "veryfast"}
        arr = pictures.numpy()
        for k, n in enumerate(holds):
            frame = av.VideoFrame.from_ndarray(arr[min(k, len(arr) - 1)],
                                               format="rgb24")
            for _ in range(n):
                for packet in s.encode(frame):
                    box.mux(packet)
        for packet in s.encode():
            box.mux(packet)
    return buf.getvalue()


class _Sender:
    """One worker per sampling pass: encodes and sends the NEWEST step; a
    step that arrives while one is encoding replaces the one waiting."""

    def __init__(self):
        self.cv = threading.Condition()
        self.job = None
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="obvpm_h3_live")
        self.thread.start()

    def submit(self, job):
        with self.cv:
            self.job = job
            self.cv.notify()

    def close(self, timeout=10.0):
        with self.cv:
            self.stop = True
            self.cv.notify()
        self.thread.join(timeout)

    def _run(self):
        while True:
            with self.cv:
                while self.job is None and not self.stop:
                    self.cv.wait()
                job, self.job = self.job, None
                if job is None and self.stop:
                    return
            try:
                job()
            except Exception:
                _LOG.exception("obvpm.h3 live preview: a step could not be sent")


def run_id():
    """The id of the Run executing now (its prompt id), or "" outside one."""
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
    except Exception:
        return ""
    return str(getattr(ctx, "prompt_id", "") or "") if ctx else ""


def _send(payload):
    try:
        from server import PromptServer
        srv = PromptServer.instance
        srv.send_sync(EVENT, payload, srv.client_id)
    except Exception as exc:
        _LOG.debug("obvpm.h3 live preview: not sent (%s)", exc)


# prompt id -> sampling passes so far
_PASSES = {}


class _LiveWrapper:
    def __init__(self, node_id, tiny_vae, max_size):
        self.node_id = str(node_id) if node_id is not None else None
        self.tiny_vae = tiny_vae
        self.max_size = int(max_size)

    def __call__(self, executor, noise, latent_image, sampler, sigmas,
                 denoise_mask, callback, disable_pbar, seed, latent_shapes):
        import comfy.utils
        guider = executor.class_obj
        patcher = guider.model_patcher
        rid = run_id()
        _PASSES[rid] = _PASSES.get(rid, 0) + 1
        while len(_PASSES) > 64:
            _PASSES.pop(next(iter(_PASSES)))
        pass_no = _PASSES[rid]
        decoder = None
        try:
            decoder = _Decoder(self.tiny_vae, patcher.model.latent_format,
                               patcher.load_device)
        except Exception as exc:
            _LOG.warning("obvpm.h3 live preview: no decoder (%s)", exc)
        sender = _Sender()
        shapes = latent_shapes
        max_size = self.max_size
        node_id = self.node_id

        def video_of(x0):
            if getattr(x0, "is_nested", False):
                x0 = x0.tensors[0]
            elif shapes and len(shapes) > 1 and x0.ndim == 3:
                x0 = comfy.utils.unpack_latents(x0, shapes)[0]
            return x0[0] if x0.ndim == 5 else None

        def new_callback(step, x0, x, total_steps):
            if callback is not None:
                callback(step, x0, x, total_steps)
            if decoder is None:
                return
            try:
                video = video_of(x0)
                if video is None:
                    return
                t = int(video.shape[1])
                pictures, holds = decoder(video.detach(), max_size)
            except Exception as exc:
                _LOG.warning("obvpm.h3 live preview: step %d not decoded (%s)",
                             step + 1, exc)
                return
            head, tail = pinned_frames(t)

            def job():
                clip = encode_clip(pictures, holds)
                _send({
                    "prompt_id": rid, "node": node_id,
                    "image": base64.b64encode(clip).decode("ascii"),
                    "mime": "video/mp4", "step": step + 1,
                    "total": total_steps, "pass": pass_no,
                    "frames": fr.pixel_frames(t), "fps": fr.FPS,
                    "head": head, "tail": tail,
                    "w": int(pictures.shape[2]), "h": int(pictures.shape[1]),
                    # on the take's own timeline: the Result Preview plays
                    # it in step with the finished takes
                    "synced": True,
                })
            sender.submit(job)

        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            new_callback, disable_pbar, seed,
                            latent_shapes=latent_shapes)
        finally:
            sender.close()


def _tiny_choices():
    names = [n for n in folder_paths.get_filename_list("vae_approx")
             if n.lower().endswith((".safetensors", ".pth", ".pt"))]
    h3 = [n for n in names if n.lower().startswith("taeh3")]
    rest = [n for n in names if n not in h3]
    return h3 + [LATENT2RGB] + rest


class H3LivePreview:
    CATEGORY = "obvpm/h3"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Live preview while sampling: put it on the model the sampler uses, "
        "and the H3 MCtx Result Preview shows each step as a short clip on "
        "the take's own timeline -- in step with the finished takes, the "
        "pinned frames at the ends included so a join can be judged while "
        "it is being made. No frame rate to set: every latent step is held "
        "for the frames it covers. Decoded with a tiny VAE from "
        "models/vae_approx (taeh3) or, without one, the latent's colours; "
        "encoded on the CPU so sampling never waits for it."
    )
    OUTPUT_TOOLTIPS = ("The same model, previewing while it samples.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "tiny_vae": (_tiny_choices(), {
                    "tooltip": "The decoder for the picture, from "
                               "models/vae_approx: a tiny VAE for H3 "
                               "(taeh3) -- madebyollin's temporal one shows "
                               "every frame, Kijai's 2D one a picture per "
                               "latent step; either works under any name. "
                               "Or the latent's own colours (coarse, no "
                               "file needed)."}),
                "max_size": ("INT", {
                    "default": 640, "min": 128, "max": 2048, "step": 32,
                    "tooltip": "The preview's long side in pixels. Smaller "
                               "is quicker to encode and to send."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def patch(self, model, tiny_vae, max_size, unique_id=None):
        import comfy.patcher_extension
        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
            _LiveWrapper(unique_id, tiny_vae, max_size))
        return (m,)
