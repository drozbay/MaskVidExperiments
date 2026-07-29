"""Differential Diffusion with soft per-step thresholds.

Stock Differential Diffusion turns a grayscale denoise mask into a family
of hard binary masks that grow as sampling proceeds, so every intermediate
mask has a razor-sharp edge regardless of how feathered the input was.
This variant replaces the hard threshold with a ramp in mask-value space,
so each intermediate mask keeps a soft edge whose spatial width tracks the
blur already present in the input mask (band width = softness / |grad|):
heavily feathered masks stay feathered at every step, sharp masks stay
sharp. Per-pixel and blur-free, so it is temporally stable on video.
"""

import torch

from comfy_api.latest import io

CATEGORY = "MaskVidExperiments"


class MVEx_DifferentialDiffusionSoftNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_DifferentialDiffusionSoft",
            display_name="MVEx Differential Diffusion (Soft)",
            category=CATEGORY,
            description="Differential Diffusion whose per-step masks keep a soft edge proportional to the blur in the original mask, instead of the stock hard binary threshold. softness 0 reproduces the stock node; softness 1 degenerates to plain static mask blending.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "softness", default=0.2, min=0.0, max=1.0, step=0.01,
                    tooltip="Width of the soft threshold band in mask-value units. 0 is the stock hard threshold; larger values keep wider soft edges on each per-step mask (spatial width scales with the input mask's blur). 1 disables the schedule entirely and blends by the mask as-is.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Blend between the thresholded per-step mask (1) and the raw input mask (0), as in the stock node.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, softness, strength=1.0) -> io.NodeOutput:
        model = model.clone()
        model.set_model_denoise_mask_function(
            lambda *args, **kwargs: cls.forward(*args, **kwargs, softness=softness, strength=strength))
        return io.NodeOutput(model)

    @classmethod
    def forward(cls, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict,
                softness: float, strength: float):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        # 1 at the first step, falling to 0 at the last, as in stock DD
        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        if softness <= 0:
            mask = (denoise_mask >= threshold).to(denoise_mask.dtype)
        else:
            # Ramp of width `softness` in mask-value space. The ramp center
            # traverses [softness/2, 1 - softness/2] rather than [0, 1] so
            # mask==1 pixels denoise fully from the first step and mask==0
            # pixels never do, matching the stock schedule's endpoints.
            mask = ((denoise_mask - threshold * (1.0 - softness)) / softness).clamp(0.0, 1.0)

        if strength and strength < 1:
            mask = strength * mask + (1 - strength) * denoise_mask
        return mask
