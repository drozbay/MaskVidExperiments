"""Subject-tracked batch crop/uncrop for video mask workflows. The crop
boxes come from planner.py, which plans position, size, and shape jointly
over the whole clip."""

import logging
import math

import numpy as np
import torch
from scipy import ndimage

import comfy.utils
from comfy_api.latest import io

from . import planner

CATEGORY = "MaskVidExperiments"


def _clean_mask_stack(masks, value_thresh, min_pixels, min_frames):
    """Binarize a mask batch and drop noise blobs.

    Connected components are labelled in 3D (time + space). A blob survives if
    its peak single-frame area clears min_pixels and it is either subject-sized
    (peak >= 50% of the largest blob, so fast motion that splits the subject
    into per-frame components doesn't delete it) or persists min_frames frames.
    """
    binary = masks.detach().cpu().numpy() > value_thresh
    if not binary.any():
        return binary

    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return binary

    span = np.zeros(count + 1, dtype=np.int64)
    for label, location in enumerate(ndimage.find_objects(labels), start=1):
        if location is not None:
            span[label] = location[0].stop - location[0].start

    peak = np.zeros(count + 1, dtype=np.int64)
    for frame in labels:
        peak = np.maximum(peak, np.bincount(frame.ravel(), minlength=count + 1))

    span[0] = 0
    peak[0] = 0
    subject = peak >= 0.5 * peak.max() if peak.max() > 0 else np.zeros_like(peak, dtype=bool)
    keep = (peak >= min_pixels) & (subject | (span >= min_frames))
    keep[0] = False
    return keep[labels]


def _open_reconstruct(masks, value_thresh, radius, min_frames):
    """Morphological opening by reconstruction on a mask batch.

    Erodes each frame spatially so blobs thinner than ~2*radius vanish, then
    instead of dilating back, keeps the original shape of every blob with a
    surviving core, so the subject's silhouette (thin protrusions included) is
    untouched. Blobs are 3D components (time + space): a core surviving in any
    frame keeps the whole track. border_value=1 so subjects cut off by the
    frame edge are not eroded from that side.
    """
    binary = masks.detach().cpu().numpy() > value_thresh
    if not binary.any():
        return binary

    spatial = np.zeros((3, 3, 3), dtype=bool)
    spatial[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    eroded = ndimage.binary_erosion(binary, structure=spatial, iterations=radius, border_value=1)

    labels, count = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        return binary

    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[eroded])] = True
    if min_frames > 1:
        for label, location in enumerate(ndimage.find_objects(labels), start=1):
            if location is not None and location[0].stop - location[0].start < min_frames:
                keep[label] = False
    keep[0] = False
    return keep[labels]


def _resize_image(img_hwc, w, h, method="lanczos"):
    if img_hwc.shape[0] == h and img_hwc.shape[1] == w:
        return img_hwc
    return comfy.utils.common_upscale(
        img_hwc.movedim(-1, 0).unsqueeze(0), w, h, method, "disabled"
    ).squeeze(0).movedim(0, -1)


def _resize_mask(mask_hw, w, h, method="bilinear"):
    if mask_hw.shape[0] == h and mask_hw.shape[1] == w:
        return mask_hw
    return comfy.utils.common_upscale(
        mask_hw[None, None], w, h, method, "disabled"
    )[0, 0]


def _upscale_size(w, h, megapixels, divisible_by):
    """Size upscaling a w x h crop to about megapixels pixels on the
    divisible_by grid. None when already at or above the floor."""
    scale = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    if scale <= 1.0:
        return None
    return (max(1, round(w * scale / divisible_by)) * divisible_by,
            max(1, round(h * scale / divisible_by)) * divisible_by)


def _cell_params(mode):
    """Raw planner dials for the standard node's (padding, prefer) cell.
    The advanced node feeds the same dials directly. guaranteed floors the
    full promise against every raw frame (floor window 1): a true
    guarantee, so mask noise must be cleaned upstream. firm floors 70%
    against the sustained tracks; flexible has no floor. In tracked mode
    the crop never rescales, so prefer differentiates through the oversize
    rent instead: stillness holds a bigger box to move less. In zoomed
    mode the rent is never scaled by prefer (cheapening it tips whole
    clips into one constant maximum size, killing zoom-follow); prefer
    differentiates through the resize cost."""
    sel = mode["mode"]
    pad_level = mode.get("padding", "firm")
    prefer = mode.get("prefer", "stillness")
    floor = {"guaranteed": 1.0, "firm": 0.7, "flexible": 0.0}[pad_level]
    still = prefer == "stillness"
    if sel == "tracked":
        oversize = 32.0 if still else 8.0
    else:
        oversize = float(mode.get("pad_surplus_tol", 16))
    return {
        "crop_scale": mode["crop_scale"],
        "min_padding_allowed": floor,
        "min_padding_allowed_window": 1 if pad_level == "guaranteed" else 16,
        "pad_deficit_tol": 16.0,
        "pad_surplus_tol": oversize,
        "resize_cost": 2.0 if still else 1.0,
        "movement_cost": 1.0,
        "center_pull": 1e-4,
        "end_tightening": 0.0,
        "end_tightening_window": 80 if floor > 0 else 0,
        "zoom_step": mode.get("zoom_step", 1.0),
        "max_zoom_rate": 0.0,
        "aspect_ratio": mode["aspect_ratio"],
        "seamless_loop": mode.get("seamless_loop", False),
    }


def _debug_summary(binary, boxes, scale, sel, info):
    """Brief per-run report: what shape the planner chose and how well the
    plan delivered the padding promise."""
    n = len(boxes)
    worst = []
    img_h, img_w = binary.shape[1:]
    for i in range(n):
        ys, xs = binary[i].nonzero()
        if len(xs) == 0:
            worst.append(np.nan)
            continue
        b = boxes[i]
        bx1 = b["x"] + b["width"] - 1
        by1 = b["y"] + b["height"] - 1
        wx = (scale - 1) / 2 * (xs.max() - xs.min() + 1)
        wy = (scale - 1) / 2 * (ys.max() - ys.min() + 1)
        fr = []
        if b["x"] > 0: fr.append((xs.min() - b["x"]) / max(wx, 1e-6))
        if bx1 < img_w - 1: fr.append((bx1 - xs.max()) / max(wx, 1e-6))
        if b["y"] > 0: fr.append((ys.min() - b["y"]) / max(wy, 1e-6))
        if by1 < img_h - 1: fr.append((by1 - ys.max()) / max(wy, 1e-6))
        worst.append(min(fr) if fr else np.nan)
    worst = np.array(worst)
    v = worst[~np.isnan(worst)]
    cx = np.array([b["x"] + b["width"] / 2 for b in boxes])
    cy = np.array([b["y"] + b["height"] / 2 for b in boxes])
    travel = np.hypot(np.diff(cx), np.diff(cy)).sum()

    ar = info["aspect"]
    if info.get("swept"):
        shape = f"{ar:.2f} (chosen from {info['candidates']} candidates)"
    elif sel == "zoomed":
        shape = f"{ar:.2f} (manual)"
    else:
        shape = f"{ar:.2f}"
    ws = [b["width"] for b in boxes]
    hs = [b["height"] for b in boxes]
    lines = [
        f"mode: {sel}",
        f"frames: {n} ({img_w}x{img_h} source)",
        f"aspect_ratio: {shape}",
        f"smallest_box: {min(ws)}x{min(hs)}",
        f"largest_box: {max(ws)}x{max(hs)}",
    ]
    if sel == "zoomed":
        lines.append(f"zoom_ratio: {max(hs) / max(min(hs), 1):.2f} "
                     "(largest box / smallest box)")
    lines.append(f"movement: {travel:.0f}px "
                 "(total distance the box center travels)")
    if len(v) and scale > 1.0:
        lines.append(f"padding_promised: {(scale - 1) / 2:.0%} of the "
                     f"subject's size on each side (crop_scale {scale:.2f})")
        lines.append(f"padding_worst: {max(v.min(), 0):.0%} of promised")
        lines.append(f"padding_typical: {max(np.median(v), 0):.0%} of promised")
    return "\n".join(lines)


def _plan_and_crop(original_images, masks, sel, p, divisible_by,
                   mask_threshold, upscale_megapixels=0.0):
    if original_images.shape[0] != masks.shape[0]:
        raise ValueError(f"original_images ({original_images.shape[0]}) and masks ({masks.shape[0]}) must have the same frame count")
    if original_images.shape[1:3] != masks.shape[1:3]:
        raise ValueError(f"original_images ({original_images.shape[2]}x{original_images.shape[1]}) and masks ({masks.shape[2]}x{masks.shape[1]}) must have the same dimensions")

    binary = masks.detach().cpu().numpy() > mask_threshold
    if not binary.any():
        raise ValueError("all masks are empty, nothing to crop")

    boxes, info = planner.plan(binary, sel, p, divisible_by)
    debug = _debug_summary(binary, boxes, p["crop_scale"], sel, info)

    size = None
    img_method, mask_method = "lanczos", "bilinear"
    if sel == "zoomed":
        # Target sized to the largest planned box: that stretch is ~1:1
        # and everything else upscales rather than losing detail.
        th = math.ceil(max(b["height"] for b in boxes) / divisible_by) * divisible_by
        size = (math.ceil(th * info["aspect"] / divisible_by) * divisible_by, th)
    if upscale_megapixels > 0:
        up = _upscale_size(*(size or (max(b["width"] for b in boxes),
                                      max(b["height"] for b in boxes))),
                           upscale_megapixels, divisible_by)
        if up is not None:
            size = up
            img_method, mask_method = "bicubic", "nearest-exact"

    if size is not None:
        tw, th = size
        debug += f"\noutput_width: {tw}\noutput_height: {th}"
        cropped_images = torch.stack([
            _resize_image(original_images[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"], :], tw, th, img_method)
            for i, b in enumerate(boxes)
        ])
        cropped_masks = torch.stack([
            _resize_mask(masks[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]], tw, th, mask_method)
            for i, b in enumerate(boxes)
        ])
    else:
        cropped_images = torch.stack([
            original_images[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"], :]
            for i, b in enumerate(boxes)
        ])
        cropped_masks = torch.stack([
            masks[i, b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]]
            for i, b in enumerate(boxes)
        ])
    return io.NodeOutput(cropped_images, cropped_masks,
                         [[b] for b in boxes], debug)


_UPSCALE_MEGAPIXELS = io.Float.Input(
    "upscale_megapixels", default=0.0, min=0.0, max=16.0, step=0.05,
    tooltip="Upscale every crop to at least about this many megapixels, on the divisible_by grid, keeping its shape. Bicubic for images, nearest exact for masks. Never downscales. 0 disables.")


_OUTPUTS = [
    io.Image.Output(display_name="cropped_images", tooltip="Constant-size crops, one per frame."),
    io.Mask.Output(display_name="cropped_masks", tooltip="The input masks cropped to the same boxes. Feed to Subject Uncrop to confine the paste to the subject."),
    io.BoundingBox.Output(display_name="bboxes", tooltip="One box per frame, for Subject Uncrop."),
    io.String.Output(display_name="debug", tooltip="Summary of the plan: chosen shape, box sizes, movement, and how much of the padding promise was kept."),
]


class MVEx_MaskCleanupNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_MaskCleanup",
            display_name="MVEx Mask Cleanup",
            category=CATEGORY,
            description="Removes specks and brief flickering blobs from a video mask batch while keeping the real subject, including its soft edges.",
            inputs=[
                io.Mask.Input("masks", tooltip="Mask batch to clean, one mask per frame."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Mask values above this count as subject when finding blobs."),
                io.DynamicCombo.Input(
                    "method",
                    tooltip="shrink_grow: shrinks the mask so thin specks vanish, then restores the surviving blobs' exact shapes. components: drops blobs that are both small and short-lived.",
                    options=[
                        io.DynamicCombo.Option("shrink_grow", [
                            io.Int.Input("shrink", default=4, min=1,
                                         tooltip="Shrink distance in pixels. Blobs thinner than about twice this are removed."),
                            io.Int.Input("min_frames", default=1, min=1,
                                         tooltip="Surviving blobs must also persist at least this many frames. 1 disables the temporal check."),
                        ]),
                        io.DynamicCombo.Option("components", [
                            io.Int.Input("min_pixels", default=32, min=1,
                                         tooltip="Blobs whose largest single-frame area is below this many pixels are removed."),
                            io.Int.Input("min_frames", default=2, min=1,
                                         tooltip="Small blobs must persist at least this many frames to survive."),
                        ]),
                    ],
                ),
                io.Int.Input("edge_grow", default=4, min=0,
                             tooltip="Grow kept regions by this many pixels so the subject's soft edges are preserved."),
            ],
            outputs=[io.Mask.Output(display_name="masks", tooltip="Cleaned masks, same size and soft values as the input.")],
        )

    @classmethod
    def execute(cls, masks, threshold, method, edge_grow) -> io.NodeOutput:
        if method["method"] == "shrink_grow":
            keep = _open_reconstruct(masks, threshold, method["shrink"], method["min_frames"])
        else:
            keep = _clean_mask_stack(masks, threshold, method["min_pixels"], method["min_frames"])
        if edge_grow > 0:
            spatial = np.zeros((3, 3, 3), dtype=bool)
            spatial[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            keep = ndimage.binary_dilation(keep, structure=spatial, iterations=edge_grow)
        gate = torch.from_numpy(keep).to(dtype=masks.dtype, device=masks.device)
        return io.NodeOutput(masks * gate)


class MVEx_SubjectCropNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_SubjectCrop",
            display_name="MVEx Subject Crop",
            category=CATEGORY,
            description="Crops a region around the masked subject from every frame, sized so the whole batch stacks into one tensor. Position, size, and shape are chosen together over the whole clip, so the crop holds still through mask jitter and follows only sustained motion.",
            inputs=[
                io.Image.Input("original_images", tooltip="Video frames as a batch."),
                io.Mask.Input("masks", tooltip="Subject masks, one per frame. Clean them first (Mask Cleanup) if the segmentation is noisy."),
                io.DynamicCombo.Input(
                    "mode",
                    tooltip="In order of increasing complexity. combined: one static crop with padding around the subject's whole travel. tracked: a constant-size crop that stays still until the subject would leave it, then moves as little as possible. zoomed: the crop also follows the subject's size, planned over the whole clip, every crop resampled to a fixed output resolution.",
                    options=[
                        io.DynamicCombo.Option("combined", [
                            io.Float.Input("crop_scale", default=1.5, min=1.0, max=4.0, step=0.05,
                                           tooltip="Crop size as a multiple of the subject's combined extent across all frames: 1.5 leaves a third of the crop as margin around the whole travel. 1.0 hugs the travel exactly."),
                            io.Float.Input("aspect_ratio", default=0.0, min=0.0, max=10.0, step=0.01,
                                           tooltip="Crop shape as width divided by height (e.g. 1.78), reached by growing the smaller dimension. 0 sizes width and height from the mask's whole travel."),
                        ]),
                        io.DynamicCombo.Option("tracked", [
                            io.Float.Input("crop_scale", default=1.5, min=1.0, max=4.0, step=0.05,
                                           tooltip="Crop size as a multiple of the subject's size: 1.5 keeps the subject at two thirds of the crop with the rest as padding, at any resolution. 1.0 crops tight, leaving no padding for the padding setting to keep; 4.0 surrounds the subject with three times its own size."),
                            io.Combo.Input("padding", options=["guaranteed", "firm", "flexible"], default="firm",
                                           tooltip="How firmly the crop_scale padding is kept. guaranteed: never less than promised on any frame, whatever it costs in size or movement; only the image edge can break it, and mask noise counts as subject, so clean the masks first. firm: at least 70% is always kept, the rest yields briefly during fast motion. flexible: padding yields first whenever keeping it would cost stillness or tightness."),
                            io.Combo.Input("prefer", options=["stillness", "tightness"], default="stillness",
                                           tooltip="What pays for keeping the padding when the subject moves. stillness: the crop runs larger so it can move less. tightness: the crop stays as small as the padding allows and moves as much as needed instead."),
                            io.Float.Input("aspect_ratio", default=0.0, min=0.0, max=10.0, step=0.01,
                                           tooltip="Crop shape as width divided by height (e.g. 1.78). 0 lets the planner choose the shape that best balances padding, stillness, and size for this clip."),
                            io.Boolean.Input("seamless_loop", default=False,
                                             tooltip="Plan the crop path so the last frame wraps seamlessly into the first. Only for clips that genuinely repeat."),
                        ]),
                        io.DynamicCombo.Option("zoomed", [
                            io.Float.Input("crop_scale", default=1.5, min=1.0, max=4.0, step=0.05,
                                           tooltip="Crop size as a multiple of the subject's size: 1.5 keeps the subject at two thirds of the crop with the rest as padding, so the subject holds a constant share of the output at any resolution. 1.0 crops tight, leaving no padding for the padding setting to keep; 4.0 surrounds the subject with three times its own size."),
                            io.Combo.Input("padding", options=["guaranteed", "firm", "flexible"], default="firm",
                                           tooltip="How firmly the crop_scale padding is kept. guaranteed: never less than promised on any frame, whatever it costs in size, movement, or rescaling; only the image edge can break it, and mask noise counts as subject, so clean the masks first. firm: at least 70% is always kept, the rest yields briefly during fast motion. flexible: padding yields first whenever keeping it would cost stillness or tightness."),
                            io.Combo.Input("prefer", options=["stillness", "tightness"], default="stillness",
                                           tooltip="What pays for keeping the padding when the subject moves. stillness: the crop moves and rescales as little as possible, running larger instead. tightness: the crop and the output resolution stay as small as the padding allows, moving and rescaling as much as needed instead."),
                            io.Int.Input("pad_surplus_tol", default=16, min=1,
                                         tooltip="Extra padding (a crop bigger than needed) lasting shorter than this many frames is kept rather than trimmed, riding out occlusions and dips at a steady size. At 1 the crop follows every size change tightly; very high holds one constant size for the whole clip."),
                            io.Float.Input("zoom_step", default=1.0, min=1.0, max=4.0, step=0.05,
                                           tooltip="Whether crop size changes smoothly or snaps to discrete zoom levels this ratio apart. At 1.0 size changes continuously; at 2.0 every zoom level doubles the previous one."),
                            io.Float.Input("aspect_ratio", default=0.0, min=0.0, max=10.0, step=0.01,
                                           tooltip="Shape of every crop and of the output resolution, as width divided by height (e.g. 1.78). 0 lets the planner choose the shape that best balances padding, stillness, and rescaling for this clip. Output size is automatic: the largest crop stays at 1:1 pixel scale."),
                            io.Boolean.Input("seamless_loop", default=False,
                                             tooltip="Plan crop position and size so the last frame wraps seamlessly into the first. Only for clips that genuinely repeat."),
                        ]),
                    ],
                ),
                io.Int.Input("divisible_by", default=16, min=1,
                             tooltip="Crop width and height are rounded up to a multiple of this. Match the model's resolution requirement."),
                _UPSCALE_MEGAPIXELS,
            ],
            outputs=_OUTPUTS,
        )

    @classmethod
    def execute(cls, original_images, masks, mode, divisible_by, upscale_megapixels=0.0) -> io.NodeOutput:
        # extents count pixels at least 10% present: feathered edges stay
        # in the box, a blur's numerically-nonzero tail does not
        sel = mode["mode"]
        if sel == "combined":
            p = {"crop_scale": mode["crop_scale"],
                 "aspect_ratio": mode["aspect_ratio"]}
        else:
            p = _cell_params(mode)
        return _plan_and_crop(original_images, masks, sel, p,
                              divisible_by, 0.1, upscale_megapixels)


class MVEx_SubjectCropAdvancedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        def dial(name, **over):
            build = {
                "crop_scale": lambda: io.Float.Input("crop_scale", default=1.5, min=1.0, max=4.0, step=0.05,
                                                     tooltip="How much padding the crop promises around the subject, as a multiple of its size. At 1.0 the crop is tight with no padding; at 4.0 the subject sits in three times its own size of padding."),
                "min_padding_allowed": lambda: io.Float.Input("min_padding_allowed", default=0.7, min=0.0, max=1.0, step=0.05,
                                                        tooltip="The padding you always get, as a fraction of the promise. At 0 padding may be sacrificed entirely (flexible); at 1 the full promise can only be broken by the image edge (guaranteed); 0.7 is firm."),
                "min_padding_allowed_window": lambda: io.Int.Input("min_padding_allowed_window", default=16, min=1, max=256,
                                                     tooltip="How long motion must last (about half this many frames) before the minimum protects it. At 1 every raw frame is protected, mask noise included; at 256 brief excursions may dip below the minimum."),
                "pad_deficit_tol": lambda: io.Int.Input("pad_deficit_tol", default=16, min=1, max=999,
                                                          tooltip="Padding dips shorter than this many frames are ignored instead of chased, keeping the crop calm. At 1 every dip is corrected immediately; at 999 lost padding is never restored."),
                "pad_surplus_tol": lambda default=16: io.Int.Input("pad_surplus_tol", default=default, min=1, max=999,
                                                                       tooltip="Extra padding (a crop bigger than needed) lasting shorter than this many frames is kept rather than trimmed, riding out occlusions and dips at a steady size. At 1 the crop follows every size change tightly; at 999 it holds one constant size for the whole clip."),
                "resize_cost": lambda: io.Float.Input("resize_cost", default=2.0, min=0.05, max=16.0, step=0.05,
                                                      tooltip="How much the crop resists changing size, relative to moving (movement_cost). Near 0 it rescales freely instead of moving; at 16 it almost never rescales, moving and running large instead."),
                "movement_cost": lambda: io.Float.Input("movement_cost", default=1.0, min=0.05, max=16.0, step=0.05,
                                                    tooltip="How much the crop resists moving: the yardstick the other costs trade against. Near 0 it glues itself to the subject; at 16 it approaches one static box around the subject's whole travel."),
                "center_pull": lambda: io.Float.Input("center_pull", default=0.0001, min=0.0, max=0.5, step=0.0001,
                                                      tooltip="How strongly the subject is kept centered when the plan has a choice. At 0 the crop parks wherever is cheapest, which can look arbitrary; at 0.5 it recenters constantly at the price of extra movement."),
                "end_tightening": lambda: io.Float.Input("end_tightening", default=0.0, min=0.0, max=1.0, step=0.05,
                                                            tooltip="Fraction of the normal shrink pressure that still applies inside the end window (end_tightening_window). At 0 the crop holds its size through the ends; at 1 the ends tighten to the subject like mid-clip and the window only steadies position."),
                "end_tightening_window": lambda: io.Int.Input("end_tightening_window", default=80, min=0, max=256,
                                                        tooltip="How many frames at each end the crop is held settled, so the clip starts and ends on a steady crop. At 0 the crop may drift and rescale right through the first and last frame; at 256 a long stretch at each end is held, shrinking only as much as end_tightening allows."),
                "zoom_step": lambda: io.Float.Input("zoom_step", default=1.0, min=1.0, max=4.0, step=0.05,
                                                    tooltip="Whether crop size changes smoothly or snaps to discrete zoom levels this ratio apart. At 1.0 size changes continuously; at 2.0 every zoom level doubles the previous one."),
                "max_zoom_rate": lambda: io.Float.Input("max_zoom_rate", default=0.0, min=0.0, max=2.0, step=0.01,
                                                        tooltip="Caps how fast the crop may rescale between adjacent frames. At 0 or 1 the speed is uncapped; at 1.02 the crop zooms at most about 2% per frame, lagging smoothly behind fast growth."),
                "aspect_ratio": lambda: io.Float.Input("aspect_ratio", default=0.0, min=0.0, max=10.0, step=0.01,
                                                       tooltip="Fixes the crop shape as width divided by height (1.78 is 16:9). At 0 the planner chooses the shape that best balances padding, stillness, and size for this clip."),
                "seamless_loop": lambda: io.Boolean.Input("seamless_loop", default=False,
                                                          tooltip="Plans the crop path so the last frame wraps seamlessly into the first, for clips that genuinely repeat. Disables the end holds."),
            }[name]
            return build(**over)

        tracked_dials = [dial("crop_scale"), dial("min_padding_allowed"),
                         dial("min_padding_allowed_window"), dial("pad_deficit_tol"),
                         dial("pad_surplus_tol", default=32),
                         dial("movement_cost"), dial("center_pull"),
                         dial("end_tightening"), dial("end_tightening_window"),
                         dial("aspect_ratio"), dial("seamless_loop")]
        zoomed_dials = [dial("crop_scale"), dial("min_padding_allowed"),
                        dial("min_padding_allowed_window"), dial("pad_deficit_tol"),
                        dial("pad_surplus_tol"), dial("resize_cost"),
                        dial("movement_cost"), dial("center_pull"),
                        dial("end_tightening"), dial("end_tightening_window"),
                        dial("zoom_step"), dial("max_zoom_rate"),
                        dial("aspect_ratio"), dial("seamless_loop")]
        return io.Schema(
            node_id="MVEx_SubjectCropAdvanced",
            display_name="MVEx Subject Crop (Advanced)",
            category=CATEGORY,
            description="Subject Crop with every planner cost exposed. The standard node's padding and prefer settings are presets over these dials; the defaults here reproduce its firm/stillness cell.",
            inputs=[
                io.Image.Input("original_images", tooltip="Video frames as a batch."),
                io.Mask.Input("masks", tooltip="Subject masks, one per frame."),
                io.DynamicCombo.Input(
                    "mode",
                    tooltip="tracked: one constant-size crop, pixel-exact slices, its shape part of the plan. zoomed: position and size planned jointly, the shape swept as a candidate set, crops resampled to one resolution.",
                    options=[
                        io.DynamicCombo.Option("tracked", tracked_dials),
                        io.DynamicCombo.Option("zoomed", zoomed_dials),
                    ],
                ),
                io.Float.Input("mask_threshold", default=0.1, min=0.0, max=1.0, step=0.01,
                               tooltip="Mask values above this count as subject when measuring extents. Near 0 even faint feathered edges steer the crop; near 1 only solid mask cores do."),
                io.Int.Input("divisible_by", default=16, min=1,
                             tooltip="Crop width and height are rounded up to a multiple of this."),
                _UPSCALE_MEGAPIXELS,
            ],
            outputs=_OUTPUTS,
        )

    @classmethod
    def execute(cls, original_images, masks, mode, mask_threshold, divisible_by, upscale_megapixels=0.0) -> io.NodeOutput:
        p = dict(mode)
        for key, val in (("resize_cost", 2.0), ("zoom_step", 1.0),
                         ("max_zoom_rate", 0.0)):
            p.setdefault(key, val)
        return _plan_and_crop(original_images, masks, mode["mode"], p,
                              divisible_by, mask_threshold, upscale_megapixels)


class MVEx_SubjectUncropNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_SubjectUncrop",
            display_name="MVEx Subject Uncrop",
            category=CATEGORY,
            description="Pastes processed crops back into the original frames with a feathered border, optionally confined by a mask. The paste is pixel-exact when the crop size was not changed.",
            inputs=[
                # cropped_images first: bypass routes the first IMAGE input to
                # the output, and bypassing crop+uncrop must yield the
                # processed full frames, not the untouched originals
                io.Image.Input("cropped_images", tooltip="Processed cropped frames."),
                io.Image.Input("original_images", tooltip="Original frames to paste into."),
                io.BoundingBox.Input("bboxes", force_input=True,
                                     tooltip="Boxes from Subject Crop, one per frame. A single box or single frame is applied to every frame; extra boxes within a frame are ignored. When inputs disagree on frame count, extra frames are dropped."),
                io.Mask.Input("cropped_masks", optional=True,
                              tooltip="Confines the paste to the mask, one per crop at any resolution. A single mask is applied to every frame. Used exactly as given; pre-blur the mask for a soft edge."),
                io.Int.Input("feather", default=16, min=0,
                             tooltip="Blend width in pixels, feathered inward from the crop border. Sides touching the image edge are not feathered."),
            ],
            outputs=[io.Image.Output(display_name="images", tooltip="Original frames with the crops blended back in.")],
        )

    @classmethod
    def execute(cls, cropped_images, original_images, bboxes, feather, cropped_masks=None) -> io.NodeOutput:
        if isinstance(bboxes, dict):
            frames = None
        elif isinstance(bboxes, list) and bboxes and all(isinstance(f, list) and f for f in bboxes):
            frames = bboxes
        else:
            raise ValueError("bboxes must be a single box or a per-frame list of box lists")

        # single-frame bboxes and masks broadcast, so they don't cap n
        counts = {"cropped_images": cropped_images.shape[0],
                  "original_images": original_images.shape[0]}
        if frames is not None and len(frames) > 1:
            counts["bboxes"] = len(frames)
        if cropped_masks is not None and cropped_masks.shape[0] > 1:
            counts["cropped_masks"] = cropped_masks.shape[0]
        n = min(counts.values())
        if n < max(counts.values()):
            got = ", ".join(f"{k}={v}" for k, v in counts.items())
            logging.warning(f"MVEx Subject Uncrop: frame counts differ ({got}), using the first {n} frames of each")

        cropped_images = cropped_images[:n]
        original_images = original_images[:n]
        if cropped_masks is not None:
            cropped_masks = cropped_masks.expand(n, -1, -1) if cropped_masks.shape[0] == 1 else cropped_masks[:n]
        if frames is None:
            boxes = [bboxes] * n
        else:
            boxes = [f[0] for f in (frames * n if len(frames) == 1 else frames[:n])]

        img_h, img_w = original_images.shape[1], original_images.shape[2]
        out = original_images.clone()
        for i, b in enumerate(boxes):
            # BOUNDING_BOX payloads are convention, not schema; SAM3 emits float coords
            x, y, w, h = (int(round(b[k])) for k in ("x", "y", "width", "height"))
            # a crop resampled anywhere along the way (Subject Crop's
            # upscale_megapixels, zoomed mode, or a resize node) scales back here
            crop = _resize_image(cropped_images[i], w, h)

            alpha = torch.ones(h, w, dtype=out.dtype, device=out.device)
            fx, fy = min(feather, w // 2), min(feather, h // 2)
            if fy > 0:
                ramp = torch.linspace(0, 1, fy + 2, dtype=out.dtype, device=out.device)[1:-1]
                if y > 0:
                    alpha[:fy, :] *= ramp[:, None]
                if y + h < img_h:
                    alpha[h - fy:, :] *= ramp.flip(0)[:, None]
            if fx > 0:
                ramp = torch.linspace(0, 1, fx + 2, dtype=out.dtype, device=out.device)[1:-1]
                if x > 0:
                    alpha[:, :fx] *= ramp[None, :]
                if x + w < img_w:
                    alpha[:, w - fx:] *= ramp.flip(0)[None, :]

            if cropped_masks is not None:
                m = _resize_mask(cropped_masks[i], w, h)
                alpha = alpha * m.clamp(0.0, 1.0).to(dtype=out.dtype, device=out.device)

            alpha = alpha[..., None]
            region = out[i, y:y + h, x:x + w, :]
            out[i, y:y + h, x:x + w, :] = crop.to(region) * alpha + region * (1 - alpha)

        return io.NodeOutput(out)
