"""Pixel-space mask batch to latent-resolution mask for Set Latent Noise Mask.

Core's reshape_mask trilinear-interpolates video masks across time, which
drifts from causal VAE frame grouping and mean-blends coverage. This node
reduces with the VAE's real geometry and selectable reduction semantics.
Generalized from WanMaskToLatentSpace in ComfyUI-WanVaceAdvanced.
"""

import logging
import math

import torch
import torch.nn.functional as F

from comfy import model_management
from comfy_api.latest import io

CATEGORY = "MaskVidExperiments"


_TEMPORAL_REDUCE = {
    "max": lambda g: g.amax(0),
    "min": lambda g: g.amin(0),
    "mean": lambda g: g.mean(0),
    "first": lambda g: g[0],
    "last": lambda g: g[-1],
}


def _grow_spatial(mask, steps):
    """Grey dilate (+) or erode (-) each frame with a cross kernel, one pixel
    per step. Borders act as subject for erosion so edge contact is kept."""
    x = mask[:, None]
    for _ in range(abs(steps)):
        if steps > 0:
            x = torch.maximum(F.max_pool2d(x, (3, 1), 1, (1, 0)),
                              F.max_pool2d(x, (1, 3), 1, (0, 1)))
        else:
            x = -torch.maximum(F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                               F.max_pool2d(-x, (1, 3), 1, (0, 1)))
    return x[:, 0]


def _grow_temporal(mask, steps):
    """Grey dilate (+) or erode (-) along time, one frame per step."""
    t, h, w = mask.shape
    x = mask.reshape(t, h * w).transpose(0, 1)[None]
    for _ in range(abs(steps)):
        if steps > 0:
            x = F.max_pool1d(x, 3, 1, 1)
        else:
            x = -F.max_pool1d(-x, 3, 1, 1)
    return x[0].transpose(0, 1).reshape(t, h, w)


def _token_snap(x, token, method):
    """Unify each token block of latent pixels so the whole block carries one
    value, replicate-padding odd edges the way the model does."""
    t, h, w = x.shape
    x = F.pad(x[:, None], (0, -w % token, 0, -h % token), mode="replicate")
    if method == "max":
        x = F.max_pool2d(x, token)
    elif method == "min":
        x = -F.max_pool2d(-x, token)
    elif method == "mean":
        x = F.avg_pool2d(x, token)
    else:
        x = F.interpolate(x, (x.shape[-2] // token, x.shape[-1] // token), mode="nearest-exact")
    x = x.repeat_interleave(token, dim=-2).repeat_interleave(token, dim=-1)
    return x[:, 0, :h, :w]


def _manual_frame_formula(head_frames, head_latents, chunk_frames, chunk_latents):
    """Latent frame count for the first t pixel frames on a head + repeating
    chunk grid. Partial trailing chunks complete no latent frame, matching the
    chunked encoders (their leftover frames merge into the last group)."""
    def formula(t):
        if head_frames > 0 and t <= head_frames:
            return max(1, math.ceil(t * head_latents / head_frames))
        return head_latents + (t - head_frames) // chunk_frames * chunk_latents
    return formula


def _manual_frame_inverse(head_frames, head_latents, chunk_frames, chunk_latents):
    """Pixel frame count for the first latents latent frames, on the same grid
    as _manual_frame_formula. Off-grid latent counts round up to a whole chunk."""
    def inverse(latents):
        if head_latents > 0 and latents <= head_latents:
            return max(1, math.ceil(latents * head_frames / head_latents))
        return head_frames + math.ceil((latents - head_latents) / chunk_latents) * chunk_frames
    return inverse


def _pattern_parse(text):
    try:
        pattern = [int(p) for p in text.replace(" ", "").split(",") if p]
    except ValueError:
        pattern = []
    if not pattern or any(p < 1 for p in pattern):
        raise ValueError("frames_per_latent must be comma-separated positive frame counts, e.g. 1,4,4,4,4")
    return pattern


def _pattern_starts(pattern):
    starts = [0]
    for p in pattern:
        starts.append(starts[-1] + p)
    return starts


def _pattern_groups(pattern, t, drop=0):
    """Frame ranges feeding each latent frame on an exact repeating cycle
    from frame 0. A positive drop models encoders that pad to whole chunks
    and trim that many trailing latents (MiniMax H3): frames from trimmed
    latents merge into the last kept group, and groups lying entirely in
    padding cover the final frame, which the padding repeats."""
    starts = _pattern_starts(pattern)
    chunk, n = starts[-1], len(pattern)
    total = n * math.ceil(t / chunk)
    if drop > 0:
        kept = max(1, total - drop)
    else:
        kept = sum(1 for k in range(total)
                   if (k // n) * chunk + starts[k % n] < t)
    groups = []
    for k in range(kept):
        s = (k // n) * chunk + starts[k % n]
        groups.append((min(s, t - 1), min(s + pattern[k % n], t)))
    if drop > 0:
        groups[-1] = (groups[-1][0], t)
    return groups


def _pattern_inverse(pattern, drop=0):
    starts = _pattern_starts(pattern)
    chunk, n = starts[-1], len(pattern)

    def inverse(latents):
        if drop > 0:
            chunks = math.ceil((latents + drop) / n)
            phase = latents - (chunks - 1) * n
            if 1 <= phase <= n:
                return (chunks - 1) * chunk + starts[phase]
        return (latents // n) * chunk + starts[latents % n]
    return inverse


def _pattern_geometry(spatial, pattern, drop=0):
    if max(pattern) == 1 and drop == 0:
        return spatial, None, None
    return (spatial,
            lambda t: _pattern_groups(pattern, t, drop),
            _pattern_inverse(pattern, drop))


def _resolve_geometry(compression, vae):
    """Spatial factor, frames-to-groups function, latents-to-frame-count
    inverse and latent pixels per model token, from the connected VAE in auto
    mode or the widgets in manual. Chunked causal encoders (MiniMax H3) expose
    their chunk parameters on the VAE, from which auto derives the exact
    frames-per-latent cycle; that family's DiT reads the mask per 2x2 latent
    patch, so auto also sets the token size to 2 there. The function pair is
    (None, None) when there is no temporal compression."""
    token = 1
    if compression["compression"] == "auto":
        if vae is None:
            raise ValueError("auto compression requires a VAE to be connected")
        spatial = vae.spacial_compression_encode()
        inner = getattr(vae, "first_stage_model", None)
        ratio_t = getattr(inner, "vae_ratio_t", None)
        chunk_latents = getattr(inner, "tokens_chunk_size", None)
        if ratio_t is not None and chunk_latents is not None and ratio_t > 1:
            pre_pad = getattr(inner, "frame_pre_padding", 0)
            drop = getattr(inner, "token_drop", 0) or 0
            pattern = [ratio_t - pre_pad] + [ratio_t] * (chunk_latents - 1)
            return _pattern_geometry(spatial, pattern, drop) + (2,)
        formula = inverse = None
        down = getattr(vae, "downscale_ratio", None)
        if isinstance(down, (tuple, list)) and down and callable(down[0]):
            formula = down[0]
            up = getattr(vae, "upscale_ratio", None)
            if isinstance(up, (tuple, list)) and up and callable(up[0]):
                inverse = up[0]
        else:
            temporal = vae.temporal_compression_decode()
            if temporal is None or temporal <= 1:
                return spatial, None, None, token
            grid = (1, 1, temporal, 1)
            formula = _manual_frame_formula(*grid)
            inverse = _manual_frame_inverse(*grid)
    else:
        spatial = compression["spatial"]
        token = compression.get("token_spatial", 1)
        text = compression.get("frames_per_latent", "")
        if text.strip():
            return _pattern_geometry(spatial, _pattern_parse(text)) + (token,)
        grid = (compression["head_frames"], compression["head_latents"],
                compression["chunk_frames"], compression["chunk_latents"])
        formula = _manual_frame_formula(*grid)
        inverse = _manual_frame_inverse(*grid)
    return spatial, (lambda t: _temporal_groups(formula, t)), inverse, token


def _temporal_groups(formula, t):
    """Frame ranges feeding each latent frame. Wherever the formula completes
    d new latent frames, the pixel frames since the previous completion split
    uniformly among them. Trailing frames that complete no latent frame merge
    into the last group so their coverage is kept."""
    groups = []
    prev_t = 0
    prev_l = 0
    for tp in range(1, t + 1):
        latents = formula(tp)
        if latents <= prev_l:
            continue
        d = latents - prev_l
        span = tp - prev_t
        for j in range(d):
            s = prev_t + round(j * span / d)
            e = prev_t + round((j + 1) * span / d)
            if e > s:
                groups.append((s, e))
        prev_t = tp
        prev_l = latents
    if not groups:
        return [(0, t)]
    if prev_t < t:
        groups[-1] = (groups[-1][0], t)
    return groups


class MVEx_MaskToLatentSpaceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_MaskToLatentSpace",
            display_name="MVEx Mask To Latent Space",
            category=CATEGORY,
            description="Reduces a pixel-space mask batch to latent resolution using the VAE's spatial and temporal compression, aligned to causal video VAE frame grouping and to the model's token grid (2x2 latent pixels for MiniMax H3). Feed the result to Set Latent Noise Mask.",
            inputs=[
                io.Mask.Input("masks", tooltip="Pixel-space masks, one per frame."),
                # vae stays top-level rather than inside the auto option: a
                # DynamicCombo whose default option adds only sockets (no
                # widgets) crashes frontend node construction, because the
                # combo's own socket does not exist yet when the default
                # option's inputs are inserted
                io.Vae.Input("vae", optional=True,
                             tooltip="The VAE used to encode the latents this mask will be applied to. Required when compression is auto."),
                io.DynamicCombo.Input(
                    "compression",
                    tooltip="auto: read the compression geometry from the connected VAE, including the exact frame cycle and 2x2 token grid of chunked models like MiniMax H3. manual: enter it directly.",
                    options=[
                        io.DynamicCombo.Option("auto", []),
                        io.DynamicCombo.Option("manual", [
                            io.Int.Input("spatial", default=8, min=1,
                                         tooltip="Pixels per latent pixel on each axis."),
                            io.Int.Input("token_spatial", default=1, min=1,
                                         tooltip="Latent pixels per model token on each axis; the mask is unified over each token block. 2 for MiniMax H3, 1 for models that read the mask per latent pixel."),
                            io.Int.Input("head_frames", default=1, min=0,
                                         tooltip="Pixel frames in the model's leading frame group. 1 for Wan, Hunyuan and LTX. 0 when the model has no leading group."),
                            io.Int.Input("head_latents", default=1, min=0,
                                         tooltip="Latent frames produced by the leading group. 1 for Wan, Hunyuan and LTX."),
                            io.Int.Input("chunk_frames", default=4, min=1,
                                         tooltip="Pixel frames in each repeating group after the leading group. 4 for Wan and Hunyuan, 8 for LTX. 1 disables temporal reduction (image models)."),
                            io.Int.Input("chunk_latents", default=1, min=1,
                                         tooltip="Latent frames produced by each repeating group. 1 for Wan, Hunyuan and LTX."),
                            io.String.Input("frames_per_latent", default="",
                                            tooltip="Exact pixel frames covered by each latent frame, comma-separated, repeating from the first frame. Overrides the grid above when set. 1,4,4,4,4 with spatial 16 for MiniMax H3. For frame counts off the model's grid, use auto."),
                        ]),
                    ],
                ),
                io.Combo.Input("spatial_method", options=["max", "min", "mean", "nearest"], default="max",
                               tooltip="How a block of pixels reduces to one latent pixel. max marks the cell if any pixel is masked, min only if all are."),
                io.Combo.Input("temporal_method", options=["max", "min", "mean", "first", "last"], default="max",
                               tooltip="How a group of frames reduces to one latent frame. max marks the frame if any grouped frame is masked."),
                io.Int.Input("grow_spatial", default=0, min=-256, max=256,
                             tooltip="Grow (+) or shrink (-) the mask this many pixels before reduction."),
                io.Int.Input("grow_temporal", default=0, min=-64, max=64,
                             tooltip="Grow (+) or shrink (-) the mask this many frames before reduction."),
            ],
            outputs=[io.Mask.Output(display_name="mask", tooltip="Latent-resolution mask for Set Latent Noise Mask.")],
        )

    @classmethod
    def execute(cls, masks, compression, spatial_method, temporal_method, grow_spatial, grow_temporal, vae=None) -> io.NodeOutput:
        spatial, groups, _, token = _resolve_geometry(compression, vae)
        out = None
        device = model_management.get_torch_device()
        if device.type != "cpu":
            # any accelerator failure (OOM, missing op on MPS or DirectML) falls
            # back to the CPU path, which is slow but always correct
            try:
                out = cls._reduce(masks.to(device), spatial, groups, token, spatial_method, temporal_method, grow_spatial, grow_temporal)
            except Exception as e:
                logging.warning(f"MVEx Mask To Latent Space: mask reduction on {device} failed ({e}), retrying on CPU")
                model_management.soft_empty_cache()
        if out is None:
            out = cls._reduce(masks, spatial, groups, token, spatial_method, temporal_method, grow_spatial, grow_temporal)
        return io.NodeOutput(out.cpu())

    @staticmethod
    def _reduce(x, spatial, groups, token, spatial_method, temporal_method, grow_spatial, grow_temporal):
        if grow_spatial != 0:
            x = _grow_spatial(x, grow_spatial)
        if grow_temporal != 0:
            x = _grow_temporal(x, grow_temporal)

        t, h, w = x.shape
        lh = max(1, h // spatial)
        lw = max(1, w // spatial)

        x = x[:, None]
        if spatial_method == "max":
            x = F.adaptive_max_pool2d(x, (lh, lw))
        elif spatial_method == "min":
            x = -F.adaptive_max_pool2d(-x, (lh, lw))
        elif spatial_method == "mean":
            x = F.adaptive_avg_pool2d(x, (lh, lw))
        else:
            x = F.interpolate(x, (lh, lw), mode="nearest-exact")
        x = x[:, 0]

        if token > 1:
            x = _token_snap(x, token, spatial_method)

        if groups is None or t <= 1:
            return x

        reduce = _TEMPORAL_REDUCE[temporal_method]
        return torch.stack([reduce(x[s:e]) for s, e in groups(t)])


class MVEx_LatentMaskToMaskNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_LatentMaskToMask",
            display_name="MVEx Latent Mask To Mask",
            category=CATEGORY,
            description="Expands a latent-resolution mask back to pixel resolution for visualization or testing. Each latent frame paints every pixel frame in its group and each token block paints as one value, so detail collapsed by the latent reduction stays collapsed.",
            inputs=[
                io.Mask.Input("mask", tooltip="Latent-resolution masks, one per latent frame."),
                # vae stays top-level for the same frontend reason as in
                # MVEx Mask To Latent Space
                io.Vae.Input("vae", optional=True,
                             tooltip="The VAE whose latents this mask matches. Required when compression is auto."),
                io.DynamicCombo.Input(
                    "compression",
                    tooltip="auto: read the compression geometry from the connected VAE, including the exact frame cycle and 2x2 token grid of chunked models like MiniMax H3. manual: enter it directly.",
                    options=[
                        io.DynamicCombo.Option("auto", []),
                        io.DynamicCombo.Option("manual", [
                            io.Int.Input("spatial", default=8, min=1,
                                         tooltip="Pixels per latent pixel on each axis."),
                            io.Int.Input("token_spatial", default=1, min=1,
                                         tooltip="Latent pixels per model token on each axis; the strongest value in each token block paints the whole block, matching how the model reads the mask. 2 for MiniMax H3, 1 for models that read the mask per latent pixel."),
                            io.Int.Input("head_frames", default=1, min=0,
                                         tooltip="Pixel frames in the model's leading frame group. 1 for Wan, Hunyuan and LTX. 0 when the model has no leading group."),
                            io.Int.Input("head_latents", default=1, min=0,
                                         tooltip="Latent frames produced by the leading group. 1 for Wan, Hunyuan and LTX."),
                            io.Int.Input("chunk_frames", default=4, min=1,
                                         tooltip="Pixel frames in each repeating group after the leading group. 4 for Wan and Hunyuan, 8 for LTX. 1 disables temporal expansion (image models)."),
                            io.Int.Input("chunk_latents", default=1, min=1,
                                         tooltip="Latent frames produced by each repeating group. 1 for Wan, Hunyuan and LTX."),
                            io.String.Input("frames_per_latent", default="",
                                            tooltip="Exact pixel frames covered by each latent frame, comma-separated, repeating from the first frame. Overrides the grid above when set. 1,4,4,4,4 with spatial 16 for MiniMax H3. For frame counts off the model's grid, use auto."),
                        ]),
                    ],
                ),
                io.Int.Input("frames", default=0, min=0, max=16384,
                             tooltip="Pixel frame count to paint. 0 uses the count the compression grid implies for the incoming latent frames. Set it explicitly to match a source video that is off the model's frame grid."),
            ],
            outputs=[io.Mask.Output(display_name="masks", tooltip="Pixel-space masks, one per frame.")],
        )

    @classmethod
    def execute(cls, mask, compression, frames, vae=None) -> io.NodeOutput:
        spatial, groups, inverse, token = _resolve_geometry(compression, vae)

        if token > 1:
            # the model reads the strongest value per token, show that
            mask = _token_snap(mask, token, "max")

        latents, lh, lw = mask.shape
        x = F.interpolate(mask[:, None], (lh * spatial, lw * spatial), mode="nearest-exact")[:, 0]

        if groups is None:
            return io.NodeOutput(x)

        if frames <= 0:
            if inverse is None:
                raise ValueError("the connected VAE does not expose a latent-to-frame formula, set frames explicitly")
            frames = inverse(latents)
        if frames <= 1:
            return io.NodeOutput(x[:1])

        out = x.new_zeros((frames, x.shape[1], x.shape[2]))
        for i, (s, e) in enumerate(groups(frames)):
            out[s:e] = x[min(i, latents - 1)]
        return io.NodeOutput(out)
