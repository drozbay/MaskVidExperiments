"""Functional tests for Differential Diffusion (Soft)."""
import importlib.util
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))  # ComfyUI root

import torch

spec = importlib.util.spec_from_file_location(
    "nodes_differential_soft", os.path.join(PACK, "nodes_differential_soft.py"))
dds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dds)

Node = dds.MVEx_DifferentialDiffusionSoftNode

from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion as StockNode


class FakeModelSampling:
    sigma_min = torch.tensor(0.01)

    def timestep(self, sigma):
        return sigma * 1000.0


class FakeInner:
    model_sampling = FakeModelSampling()


class FakeModel:
    inner_model = FakeInner()


SIGMAS = torch.linspace(1.0, 0.0, 11)
EXTRA = {"model": FakeModel(), "sigmas": SIGMAS}


def sigma_at(threshold):
    """Sigma whose stock DD threshold equals the given value."""
    ms = FakeModelSampling()
    ts_from = ms.timestep(SIGMAS[0])
    ts_to = ms.timestep(ms.sigma_min)  # SIGMAS[-1]=0 < sigma_min
    return torch.tensor([(threshold * (ts_from - ts_to) + ts_to) / 1000.0])


def run(mask, threshold, softness, strength=1.0):
    return Node.forward(sigma_at(threshold), mask, EXTRA, softness=softness, strength=strength)


failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def test_softness_zero_matches_stock():
    torch.manual_seed(0)
    mask = torch.rand(2, 1, 16, 16)
    for th in (0.9, 0.5, 0.1):
        ours = run(mask, th, softness=0.0)
        stock = StockNode.forward(sigma_at(th), mask, EXTRA, strength=1.0)
        assert torch.equal(ours, stock), th


def test_strength_matches_stock_at_softness_zero():
    torch.manual_seed(1)
    mask = torch.rand(1, 1, 8, 8)
    ours = run(mask, 0.5, softness=0.0, strength=0.6)
    stock = StockNode.forward(sigma_at(0.5), mask, EXTRA, strength=0.6)
    assert torch.allclose(ours, stock)


def test_endpoints():
    mask = torch.tensor([0.0, 0.5, 1.0])
    first = run(mask, 1.0, softness=0.3)  # first step
    assert first[2] == 1.0, first  # fully masked pixels denoise from step one
    assert first[0] == 0.0, first
    last = run(mask, 0.0, softness=0.3)  # last step
    assert last[0] == 0.0, last  # unmasked pixels never denoise
    assert last[1] == 1.0 and last[2] == 1.0, last


def test_monotonic_over_schedule():
    torch.manual_seed(2)
    mask = torch.rand(64)
    prev = run(mask, 1.0, softness=0.25)
    for th in torch.linspace(0.9, 0.0, 10):
        cur = run(mask, float(th), softness=0.25)
        assert (cur >= prev - 1e-6).all(), th  # per-step masks only grow
        prev = cur


def test_band_width_tracks_input_blur():
    # Two edges blurred over 8 px vs 64 px; the soft band of the per-step
    # mask should be ~8x wider on the blurrier edge.
    sharp = torch.linspace(0, 1, 8)
    blurry = torch.linspace(0, 1, 64)
    band = lambda m: (((m > 0.001) & (m < 0.999)).sum()).item()
    b_sharp = band(run(sharp, 0.5, softness=0.2))
    b_blurry = band(run(blurry, 0.5, softness=0.2))
    assert b_sharp > 0, b_sharp
    ratio = b_blurry / b_sharp
    assert 6 <= ratio <= 10, (b_sharp, b_blurry)


def test_softness_one_is_static_mask():
    torch.manual_seed(3)
    mask = torch.rand(32)
    for th in (1.0, 0.5, 0.0):
        out = run(mask, th, softness=1.0)
        assert torch.allclose(out, mask), th


def test_strength_blend():
    torch.manual_seed(4)
    mask = torch.rand(32)
    soft = run(mask, 0.5, softness=0.3, strength=1.0)
    blended = run(mask, 0.5, softness=0.3, strength=0.5)
    assert torch.allclose(blended, 0.5 * soft + 0.5 * mask)


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
    check(name, fn)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nall tests passed")
