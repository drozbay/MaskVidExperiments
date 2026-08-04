"""Pixel-space mask batch to latent-resolution mask for Set Latent Noise Mask.

Core's reshape_mask trilinear-interpolates video masks across time, which
drifts from causal VAE frame grouping and mean-blends coverage. This node
reduces with the VAE's real geometry and selectable reduction semantics.
Generalized from WanMaskToLatentSpace in ComfyUI-WanVaceAdvanced.
"""

import math

import torch
import torch.nn.functional as F

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


def _manual_frame_formula(head_frames, head_latents, chunk_frames, chunk_latents):
    """Latent frame count for the first t pixel frames on a head + repeating
    chunk grid. Partial trailing chunks complete no latent frame, matching the
    chunked encoders (their leftover frames merge into the last group)."""
    def formula(t):
        if head_frames > 0 and t <= head_frames:
            return max(1, math.ceil(t * head_latents / head_frames))
        return head_latents + (t - head_frames) // chunk_frames * chunk_latents
    return formula


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
            description="Reduces a pixel-space mask batch to latent resolution using the VAE's spatial and temporal compression, aligned to causal video VAE frame grouping. Feed the result to Set Latent Noise Mask.",
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
                    tooltip="auto: read the compression geometry from the connected VAE. manual: enter it directly.",
                    options=[
                        io.DynamicCombo.Option("auto", []),
                        io.DynamicCombo.Option("manual", [
                            io.Int.Input("spatial", default=8, min=1,
                                         tooltip="Pixels per latent pixel on each axis."),
                            io.Int.Input("head_frames", default=1, min=0,
                                         tooltip="Pixel frames in the model's leading frame group. 1 for Wan, Hunyuan and LTX, 5 for MiniMax H3. 0 when the model has no leading group."),
                            io.Int.Input("head_latents", default=1, min=0,
                                         tooltip="Latent frames produced by the leading group. 1 for Wan, Hunyuan and LTX, 2 for MiniMax H3."),
                            io.Int.Input("chunk_frames", default=4, min=1,
                                         tooltip="Pixel frames in each repeating group after the leading group. 4 for Wan and Hunyuan, 8 for LTX, 17 for MiniMax H3. 1 disables temporal reduction (image models)."),
                            io.Int.Input("chunk_latents", default=1, min=1,
                                         tooltip="Latent frames produced by each repeating group. 1 for Wan, Hunyuan and LTX, 5 for MiniMax H3."),
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
        formula = None
        if compression["compression"] == "auto":
            if vae is None:
                raise ValueError("auto compression requires a VAE to be connected")
            spatial = vae.spacial_compression_encode()
            ratio = getattr(vae, "downscale_ratio", None)
            if isinstance(ratio, (tuple, list)) and ratio and callable(ratio[0]):
                formula = ratio[0]
            else:
                temporal = vae.temporal_compression_decode()
                if temporal is not None and temporal > 1:
                    formula = _manual_frame_formula(1, 1, temporal, 1)
        else:
            spatial = compression["spatial"]
            formula = _manual_frame_formula(
                compression["head_frames"], compression["head_latents"],
                compression["chunk_frames"], compression["chunk_latents"])

        x = masks
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

        if formula is None or t <= 1:
            return io.NodeOutput(x)

        reduce = _TEMPORAL_REDUCE[temporal_method]
        out = torch.stack([reduce(x[s:e]) for s, e in _temporal_groups(formula, t)])
        return io.NodeOutput(out)
