"""Pixel-space mask batch to latent-resolution mask for Set Latent Noise Mask.

Core's reshape_mask trilinear-interpolates video masks across time, which
drifts from causal VAE frame grouping (frame 0 alone, then groups of N) and
mean-blends coverage. This node reduces with the VAE's real geometry and
selectable reduction semantics. Generalized from WanMaskToLatentSpace in
ComfyUI-WanVaceAdvanced.
"""

import torch
import torch.nn.functional as F

from comfy_api.latest import io

CATEGORY = "MaskVidExperiments"


_TEMPORAL_REDUCE = {
    "max": lambda g: g.amax(1),
    "min": lambda g: g.amin(1),
    "mean": lambda g: g.mean(1),
    "first": lambda g: g[:, 0],
    "last": lambda g: g[:, -1],
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
                    tooltip="auto: read the compression factors from the connected VAE. manual: enter them directly.",
                    options=[
                        io.DynamicCombo.Option("auto", []),
                        io.DynamicCombo.Option("manual", [
                            io.Int.Input("spatial", default=8, min=1,
                                         tooltip="Pixels per latent pixel on each axis."),
                            io.Int.Input("temporal", default=4, min=1,
                                         tooltip="Frames per latent frame. 1 disables temporal reduction (image models)."),
                            io.Boolean.Input("first_frame_special", default=True,
                                             tooltip="First frame maps to its own latent frame and the rest group by the temporal factor, as in causal video VAEs (Wan, Hunyuan, LTX)."),
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
            temporal = vae.temporal_compression_decode()
            first_special = True
            ratio = getattr(vae, "downscale_ratio", None)
            if isinstance(ratio, (tuple, list)) and ratio and callable(ratio[0]):
                formula = ratio[0]
        else:
            spatial = compression["spatial"]
            temporal = compression["temporal"]
            first_special = compression["first_frame_special"]

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

        if temporal is None or temporal <= 1 or t <= 1:
            return io.NodeOutput(x)

        head = x[:1] if first_special else x[:0]
        rest = x[head.shape[0]:]
        if rest.shape[0]:
            remainder = rest.shape[0] % temporal
            if remainder:
                rest = torch.cat([rest, rest[-1:].expand(temporal - remainder, -1, -1)], 0)
            rest = _TEMPORAL_REDUCE[temporal_method](rest.reshape(-1, temporal, lh, lw))
        out = torch.cat([head, rest], 0)

        if formula is not None:
            # The VAE's own frame-count formula is authoritative; the grouping
            # above only approximates it for exotic patterns
            lt = formula(t)
            if isinstance(lt, int) and 0 < lt < out.shape[0]:
                out = out[:lt]
            elif isinstance(lt, int) and lt > out.shape[0]:
                out = torch.cat([out, out[-1:].expand(lt - out.shape[0], -1, -1)], 0)
        return io.NodeOutput(out)
