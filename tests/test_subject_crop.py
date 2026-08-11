"""Tests for Subject Crop planner invariants (containment, padding tiers,
stillness, loops) through the standard and advanced nodes."""
import importlib
import importlib.util
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))  # ComfyUI root

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "MaskVidExperiments", os.path.join(PACK, "__init__.py"),
    submodule_search_locations=[PACK])
pack = importlib.util.module_from_spec(spec)
sys.modules["MaskVidExperiments"] = pack
spec.loader.exec_module(pack)
sc = importlib.import_module("MaskVidExperiments.nodes_subject_crop")

Node = sc.MVEx_SubjectCropNode

H, W = 240, 320


def masks_from_rects(rects):
    """rects: per frame (y0, y1, x0, x1) or None for an empty frame."""
    m = torch.zeros(len(rects), H, W)
    for i, r in enumerate(rects):
        if r is not None:
            y0, y1, x0, x1 = r
            m[i, y0:y1, x0:x1] = 1.0
    return m


def run_tracked(masks, crop_scale=1.5, padding="firm", prefer="stillness",
                aspect_ratio=0.0, seamless_loop=False):
    images = torch.zeros(masks.shape[0], H, W, 3)
    out = Node.execute(images, masks,
                       {"mode": "tracked", "crop_scale": crop_scale,
                        "padding": padding, "prefer": prefer,
                        "aspect_ratio": aspect_ratio,
                        "seamless_loop": seamless_loop},
                       divisible_by=16)
    return [f[0] for f in out.args[2]]


def run_combined(masks, crop_scale=1.5, aspect_ratio=0.0):
    images = torch.zeros(masks.shape[0], H, W, 3)
    out = Node.execute(images, masks,
                       {"mode": "combined", "crop_scale": crop_scale,
                        "aspect_ratio": aspect_ratio},
                       divisible_by=16)
    return [f[0] for f in out.args[2]]


def run_zoomed(masks, crop_scale=1.5, padding="firm", prefer="stillness",
               zoom_step=1.0, aspect_ratio=0.0, seamless_loop=False):
    images = torch.zeros(masks.shape[0], H, W, 3)
    out = Node.execute(images, masks,
                       {"mode": "zoomed", "crop_scale": crop_scale,
                        "padding": padding, "prefer": prefer,
                        "zoom_step": zoom_step,
                        "aspect_ratio": aspect_ratio,
                        "seamless_loop": seamless_loop},
                       divisible_by=16)
    return out


def contained(masks, boxes, pad=0):
    for i in range(masks.shape[0]):
        ys, xs = (masks[i] > 0.1).numpy().nonzero()
        if len(xs) == 0:
            continue
        b = boxes[i]
        if xs.min() < b["x"] or xs.max() >= b["x"] + b["width"]:
            return False
        if ys.min() < b["y"] or ys.max() >= b["y"] + b["height"]:
            return False
    return True


failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def test_jitter_is_stationary():
    rng = np.random.default_rng(3)
    rects = []
    for _ in range(40):
        dy, dx = rng.integers(-3, 4, 2)
        rects.append((100 + dy, 160 + dy, 130 + dx, 190 + dx))
    boxes = run_tracked(masks_from_rects(rects))
    assert contained(masks_from_rects(rects), boxes)
    assert len({(b["x"], b["y"]) for b in boxes}) == 1, "box moved under jitter"


def test_containment_during_motion():
    rects = [(80, 140, 10 + 3 * i, 70 + 3 * i) for i in range(50)]
    masks = masks_from_rects(rects)
    boxes = run_tracked(masks)
    assert contained(masks, boxes)


def test_padding_restored_after_cut():
    rects = [(60, 120, 40, 100)] * 30 + [(140, 220, 200, 290)] * 30
    masks = masks_from_rects(rects)
    boxes = run_tracked(masks)
    assert contained(masks, boxes)
    positions = [(b["x"], b["y"]) for b in boxes]
    assert len(set(positions)) == 2, f"expected one jump, got {len(set(positions))} positions"
    b = boxes[-1]
    left_pad = 200 - b["x"]
    right_pad = b["x"] + b["width"] - 1 - 289
    assert min(left_pad, right_pad) >= 8, (left_pad, right_pad)


def test_zero_padding_still_contains():
    rects = [(100, 160, 130, 190), (98, 158, 133, 193)] * 10
    masks = masks_from_rects(rects)
    boxes = run_tracked(masks, crop_scale=1.0)
    assert contained(masks, boxes)


def test_empty_frames_interpolated():
    rects = [(100, 160, 60, 120)] * 10 + [None] * 8 + [(100, 160, 180, 240)] * 10
    masks = masks_from_rects(rects)
    boxes = run_tracked(masks)
    assert len(boxes) == 28
    assert contained(masks, boxes)
    assert all(b["width"] == boxes[0]["width"] for b in boxes)


def test_constant_size():
    rects = [(80, 140, 10 + 3 * i, 70 + 3 * i) for i in range(50)]
    boxes = run_tracked(masks_from_rects(rects))
    assert len({(b["width"], b["height"]) for b in boxes}) == 1


def test_debug_summary_present():
    rects = [(100 - i, 160 + i, 130 - i, 190 + i) for i in range(20)]
    out = run_zoomed(masks_from_rects(rects))
    debug = out.args[3]
    for key in ("mode: zoomed", "aspect_ratio:", "zoom_ratio:", "movement:",
                "padding_worst:", "output_width:"):
        assert key in debug, (key, debug)


def test_zoomed_containment_and_stacking():
    rects = [(100 - i, 160 + i, 130 - i, 190 + i) for i in range(40)]
    masks = masks_from_rects(rects)
    out = run_zoomed(masks)
    boxes = [f[0] for f in out.args[2]]
    assert contained(masks, boxes)
    imgs = out.args[0]
    assert imgs.shape[0] == 40
    assert imgs.shape[1] % 16 == 0 and imgs.shape[2] % 16 == 0
    assert out.args[1].shape == imgs.shape[:3]


def test_zoomed_ignores_occlusion_dip():
    rects = [(80, 180, 100, 220)] * 30 + [(80, 105, 100, 220)] * 15 + [(80, 180, 100, 220)] * 30
    masks = masks_from_rects(rects)
    boxes = [f[0] for f in run_zoomed(masks).args[2]]
    assert contained(masks, boxes)
    assert len({(b["width"], b["height"]) for b in boxes}) == 1, "size reacted to the dip"


def test_zoomed_growth_takes_few_steps():
    rects = [(120 - i, 120 + i, 160 - i, 160 + i) for i in range(2, 81)]
    masks = masks_from_rects(rects)
    boxes = [f[0] for f in run_zoomed(masks, zoom_step=1.5).args[2]]
    assert contained(masks, boxes)
    sizes = [boxes[0]["height"]]
    for b in boxes[1:]:
        if b["height"] != sizes[-1]:
            sizes.append(b["height"])
    assert len(sizes) <= 4, f"too many size levels: {sizes}"
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), "growth should only step up"


def test_zoomed_aspect_ratio_shapes_output():
    rects = [(100, 160, 130, 190)] * 12
    out = run_zoomed(masks_from_rects(rects), aspect_ratio=2.0)
    imgs = out.args[0]
    boxes = [f[0] for f in out.args[2]]
    assert contained(masks_from_rects(rects), boxes)
    # output resolution and every box carry the requested shape
    assert abs(imgs.shape[2] / imgs.shape[1] - 2.0) < 0.35
    for b in boxes:
        assert abs(b["width"] / b["height"] - 2.0) < 0.05


def test_tracked_aspect_ratio_is_honored():
    rects = [(100, 160, 130, 190)] * 12
    masks = masks_from_rects(rects)
    base = run_tracked(masks)[0]
    wide = run_tracked(masks, aspect_ratio=2.0)[0]
    assert contained(masks, [wide] * 12)
    assert wide["width"] > base["width"]
    # ratio approximate to the divisibility grid
    assert abs(wide["width"] - 2.0 * wide["height"]) <= 32


def test_firm_restores_end_padding():
    rects = [(90, 150, 10 + 3 * i, 70 + 3 * i) for i in range(60)]
    masks = masks_from_rects(rects)
    least = run_tracked(masks, padding="flexible")
    kept = run_tracked(masks)
    assert contained(masks, kept)
    # flexible spends the trailing pad at both clip ends; the default
    # restores it
    def side_pads(boxes, i):
        left = rects[i][2] - boxes[i]["x"]
        right = boxes[i]["x"] + boxes[i]["width"] - rects[i][3]
        return left, right
    assert min(side_pads(least, -1)) <= 8, side_pads(least, -1)
    assert min(side_pads(kept, -1)) >= 12, side_pads(kept, -1)
    assert min(side_pads(kept, 0)) > min(side_pads(least, 0))


def test_firm_zoomed_grows_to_padded_end():
    rects = [(120 - i // 2, 120 + i // 2, 160 - i, 160 + i) for i in range(4, 84)]
    masks = masks_from_rects(rects)
    least = [f[0] for f in run_zoomed(masks, padding="flexible").args[2]]
    kept = [f[0] for f in run_zoomed(masks).args[2]]
    assert contained(masks, kept)
    need = (rects[-1][3] - rects[-1][2]) * 1.5
    assert least[-1]["width"] < 0.85 * need
    assert kept[-1]["width"] >= 0.9 * need, (kept[-1]["width"], need)


def test_firm_never_tightens_at_ends():
    # subject drifts left and shrinks ~2x; leading (left) edge is where
    # flexible spends the padding at the end
    rects = []
    for i in range(60):
        x = 250 - 3 * i
        hh, hw = 30 - i // 4, 22 - i // 6
        rects.append((120 - hh, 120 + hh, x - hw, x + hw))
    masks = masks_from_rects(rects)
    least = [f[0] for f in run_zoomed(masks, padding="flexible").args[2]]
    kept = [f[0] for f in run_zoomed(masks).args[2]]
    assert contained(masks, kept)
    lead = lambda boxes: rects[-1][2] - boxes[-1]["x"]
    # padding scales with the subject: the shrunken end subject earns
    # (1.5 - 1) / 2 of its own width, ~6px here
    end_pad = (rects[-1][3] - rects[-1][2]) * 0.25
    assert lead(least) <= 0.4 * end_pad, (lead(least), end_pad)
    assert lead(kept) >= 0.7 * end_pad, (lead(kept), end_pad)
    # padding comes from re-centering the roomy box, not from shrinking it
    assert kept[-1]["width"] >= least[-1]["width"] - 1


def swing_rects(out=16, hold=24, x0=100, step=6, half=25):
    """Stationary, excursion right over `out` frames, hold, return, rest."""
    xs = [x0] * 30
    xs += [x0 + step * (i + 1) for i in range(out)]
    xs += [x0 + step * out] * hold
    xs += [x0 + step * (out - 1 - i) for i in range(out)]
    xs += [x0] * 30
    return [(120 - half, 120 + half, x - half, x + half) for x in xs]


def min_side_pad(rects, boxes):
    return min(min(r[2] - b["x"], b["x"] + b["width"] - 1 - (r[3] - 1))
               for r, b in zip(rects, boxes))


def test_guaranteed_holds_padding_through_swing():
    rects = swing_rects()
    masks = masks_from_rects(rects)
    pad = (rects[0][3] - rects[0][2]) * 0.25
    soft = run_tracked(masks, padding="flexible", prefer="tightness")
    hard = run_tracked(masks, padding="guaranteed", prefer="tightness")
    assert contained(masks, hard)
    # the excursion is sustained, so flexible sheds padding while
    # guaranteed delivers the full promise on every frame (minus rounding)
    assert min_side_pad(rects, soft) <= 0.5 * pad, (min_side_pad(rects, soft), pad)
    assert min_side_pad(rects, hard) >= pad - 2, (min_side_pad(rects, hard), pad)


def test_guaranteed_zoomed_holds_padding_through_swing():
    rects = swing_rects()
    masks = masks_from_rects(rects)
    pad = (rects[0][3] - rects[0][2]) * 0.25
    boxes = [f[0] for f in run_zoomed(masks, padding="guaranteed",
                                      prefer="tightness").args[2]]
    assert contained(masks, boxes)
    assert min_side_pad(rects, boxes) >= pad - 2, (min_side_pad(rects, boxes), pad)


def test_single_frame_spike_firm_rides_guaranteed_pads():
    rects = [(100, 160, 130, 190)] * 40
    rects[20] = (100, 160, 130, 270)  # one-frame protrusion to the right
    masks = masks_from_rects(rects)
    firm = run_tracked(masks)
    assert contained(masks, firm)
    assert len({(b["x"], b["y"]) for b in firm}) == 1, "firm chased a blip"
    # guaranteed treats even the blip as subject: the spike frame gets the
    # full promise, which costs a visibly bigger box than firm
    guar = run_tracked(masks, padding="guaranteed")
    assert contained(masks, guar)
    b = guar[20]
    spike_pad = min(130 - b["x"], b["x"] + b["width"] - 270)
    assert spike_pad >= 0.25 * (270 - 130) - 2, spike_pad
    assert guar[0]["width"] > firm[0]["width"]


def test_prefer_tightness_trades_size_for_motion():
    rects = swing_rects()
    masks = masks_from_rects(rects)
    still = run_tracked(masks, prefer="stillness")
    tight = run_tracked(masks, prefer="tightness")
    assert contained(masks, still) and contained(masks, tight)
    assert tight[0]["width"] <= still[0]["width"]
    travel = lambda boxes: sum(abs(boxes[i + 1]["x"] - boxes[i]["x"])
                               for i in range(len(boxes) - 1))
    assert travel(still) <= travel(tight)


def test_loop_is_seamless_at_the_wrap():
    n = 72
    rects = []
    for i in range(n):
        cx = 160 + int(70 * np.cos(2 * np.pi * i / n))
        cy = 120 + int(70 * np.sin(2 * np.pi * i / n))
        rects.append((cy - 20, cy + 20, cx - 20, cx + 20))
    masks = masks_from_rects(rects)
    for boxes in (run_tracked(masks, seamless_loop=True),
                  [f[0] for f in run_zoomed(masks, seamless_loop=True).args[2]]):
        assert contained(masks, boxes)
        step = max(abs(boxes[i + 1]["x"] - boxes[i]["x"]) + abs(boxes[i + 1]["y"] - boxes[i]["y"])
                   for i in range(n - 1))
        seam = (abs(boxes[0]["x"] - boxes[-1]["x"]) + abs(boxes[0]["y"] - boxes[-1]["y"])
                + abs(boxes[0]["width"] - boxes[-1]["width"])
                + abs(boxes[0]["height"] - boxes[-1]["height"]))
        assert seam <= step + 1, (seam, step)


def test_advanced_dials_reproduce_standard_cells():
    rects = swing_rects()
    masks = masks_from_rects(rects)
    images = torch.zeros(masks.shape[0], H, W, 3)
    Adv = sc.MVEx_SubjectCropAdvancedNode
    base = {"crop_scale": 1.5, "min_padding_allowed": 0.7,
            "min_padding_allowed_window": 16,
            "pad_deficit_tol": 16,
            "movement_cost": 1.0, "center_pull": 0.0001,
            "end_tightening": 0.0, "end_tightening_window": 80,
            "aspect_ratio": 0.0, "seamless_loop": False}
    # firm/stillness zoomed == advanced zoomed defaults
    std = [f[0] for f in run_zoomed(masks).args[2]]
    adv = [f[0] for f in Adv.execute(
        images, masks,
        {"mode": "zoomed", **base, "pad_surplus_tol": 16,
         "resize_cost": 2.0, "zoom_step": 1.0, "max_zoom_rate": 0.0},
        mask_threshold=0.1, divisible_by=16).args[2]]
    assert adv == std
    # firm/stillness tracked == advanced tracked defaults (surplus 32)
    std = run_tracked(masks)
    adv = [f[0] for f in Adv.execute(
        images, masks,
        {"mode": "tracked", **base, "pad_surplus_tol": 32},
        mask_threshold=0.1, divisible_by=16).args[2]]
    assert adv == std


def test_combined_is_static_union_with_padding():
    rects = [(80, 140, 10 + 3 * i, 70 + 3 * i) for i in range(50)]
    masks = masks_from_rects(rects)
    boxes = run_combined(masks)
    assert contained(masks, boxes)
    assert len({(b["x"], b["y"], b["width"], b["height"]) for b in boxes}) == 1
    b = boxes[0]
    # union spans x 10..216: padding must be funded into the static box
    assert b["x"] <= 10 - 8 and b["x"] + b["width"] >= 217 + 8


def test_megapixels_budget_keeps_shape():
    rects = [(100, 160, 130, 190)] * 12
    masks = masks_from_rects(rects)
    images = torch.zeros(12, H, W, 3)
    for mode, aspect in (("tracked", 2.0), ("combined", 0.0), ("zoomed", 0.0)):
        cell = {"mode": mode, "crop_scale": 1.5, "aspect_ratio": aspect}
        if mode != "combined":
            cell.update(padding="firm", prefer="stillness", seamless_loop=False)
        if mode == "zoomed":
            cell.update(zoom_step=1.0, pad_surplus_tol=16)
        plain = Node.execute(images, masks, cell, divisible_by=16)
        out = Node.execute(images, masks, cell, divisible_by=16, megapixels=0.25)
        imgs, msks = out.args[0], out.args[1]
        assert imgs.shape[0] == 12 and msks.shape == imgs.shape[:3]
        assert imgs.shape[1] % 16 == 0 and imgs.shape[2] % 16 == 0, mode
        pixels = imgs.shape[1] * imgs.shape[2]
        assert abs(pixels / 250_000 - 1) < 0.15, (mode, imgs.shape)
        # the planned boxes are untouched and the shape survives the resample
        assert out.args[2] == plain.args[2], mode
        before = plain.args[0].shape[2] / plain.args[0].shape[1]
        assert abs(imgs.shape[2] / imgs.shape[1] - before) < 0.1, mode
        assert f"output_width: {imgs.shape[2]}" in out.args[3]


def test_megapixels_round_trips_through_uncrop():
    rects = [(100, 160, 130, 190)] * 6
    masks = masks_from_rects(rects)
    images = torch.rand(6, H, W, 3)
    cell = {"mode": "tracked", "crop_scale": 1.5, "padding": "firm",
            "prefer": "stillness", "aspect_ratio": 0.0, "seamless_loop": False}
    out = Node.execute(images, masks, cell, divisible_by=16, megapixels=0.25)
    b = out.args[2][0][0]
    assert out.args[0].shape[1:3] != (b["height"], b["width"]), "test needs a real resample"
    back = sc.MVEx_SubjectUncropNode.execute(
        out.args[0], images, out.args[2], feather=0).args[0]
    assert back.shape == images.shape
    # the paste lands in the planned box and leaves everything else untouched
    outside = torch.ones(H, W, dtype=torch.bool)
    outside[b["y"]:b["y"] + b["height"], b["x"]:b["x"] + b["width"]] = False
    assert torch.equal(back[:, outside], images[:, outside])


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
    check(name, fn)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nall tests passed")
