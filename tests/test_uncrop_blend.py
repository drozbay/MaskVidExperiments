"""Tests for Subject Uncrop unified blend (rectangle feather x optional mask)."""
import importlib
import importlib.util
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))  # ComfyUI root

import torch

spec = importlib.util.spec_from_file_location(
    "MaskVidExperiments", os.path.join(PACK, "__init__.py"),
    submodule_search_locations=[PACK])
pack = importlib.util.module_from_spec(spec)
sys.modules["MaskVidExperiments"] = pack
spec.loader.exec_module(pack)
sc = importlib.import_module("MaskVidExperiments.nodes_subject_crop")

Node = sc.MVEx_SubjectUncropNode

N, H, W = 2, 64, 64
BOX = {"x": 16, "y": 16, "width": 24, "height": 24}


def run(feather, masks=None):
    images = torch.zeros(N, H, W, 3)
    crops = torch.ones(N, 24, 24, 3)
    out = Node.execute(cropped_images=crops, original_images=images, bboxes=[[BOX]] * N,
                       feather=feather, cropped_masks=masks)
    return out.args[0]


failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def test_no_mask_hard_paste():
    out = run(0)
    region = out[0, 16:40, 16:40, 0]
    assert region.min() == 1.0
    assert out[0].sum() == region.numel() * 3


def test_no_mask_feather_ramps():
    out = run(8)
    # center of paste fully opaque, border attenuated
    assert out[0, 28, 28, 0] == 1.0
    assert out[0, 16, 28, 0] < 0.2  # top edge of box heavily feathered
    assert out[0, 28, 16, 0] < 0.2  # left edge


def test_mask_confines_paste():
    masks = torch.zeros(N, 24, 24)
    masks[:, 4:20, 4:20] = 1.0
    out = run(0, masks)
    assert out[0, 28, 28, 0] == 1.0        # inside mask
    assert out[0, 17, 17, 0] == 0.0        # inside box, outside mask
    assert out[0, 10, 10, 0] == 0.0        # outside box


def test_mask_used_exactly_no_blur():
    masks = torch.zeros(N, 24, 24)
    masks[:, 4:20, 4:20] = 1.0
    out = run(0, masks)
    # hard mask edge stays hard: neighboring pixels are exactly 1 and 0
    assert out[0, 16 + 4, 28, 0] == 1.0 and out[0, 16 + 3, 28, 0] == 0.0


def test_feather_multiplies_with_mask():
    masks = torch.ones(N, 24, 24)
    assert torch.equal(run(8, masks), run(8))  # full mask == no mask
    soft = torch.full((N, 24, 24), 0.5)
    out = run(0, soft)
    assert abs(out[0, 28, 28, 0].item() - 0.5) < 1e-6


def test_mask_resized_from_other_resolution():
    masks = torch.zeros(N, 48, 48)  # 2x the crop size
    masks[:, 8:40, 8:40] = 1.0
    out = run(0, masks)
    assert out[0, 28, 28, 0] == 1.0


def test_mismatched_counts_trim_to_shortest():
    images = torch.zeros(N + 3, H, W, 3)
    crops = torch.ones(N + 1, 24, 24, 3)
    masks = torch.ones(N + 2, 24, 24)
    out = Node.execute(cropped_images=crops, original_images=images,
                       bboxes=[[BOX]] * N, feather=0, cropped_masks=masks)
    assert out.args[0].shape[0] == N


def test_single_mask_broadcasts():
    masks = torch.zeros(1, 24, 24)
    masks[:, 4:20, 4:20] = 1.0
    out = run(0, masks)
    assert out.shape[0] == N
    for i in range(N):
        assert out[i, 28, 28, 0] == 1.0    # inside mask
        assert out[i, 17, 17, 0] == 0.0    # inside box, outside mask


def test_broadcast_bboxes_do_not_cap_frames():
    images = torch.zeros(N, H, W, 3)
    crops = torch.ones(N, 24, 24, 3)
    for bb in (BOX, [[BOX]]):
        out = Node.execute(cropped_images=crops, original_images=images,
                           bboxes=bb, feather=0, cropped_masks=None)
        assert out.args[0].shape[0] == N


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
    check(name, fn)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nall tests passed")
