"""Functional tests for Audio Mask To Latent."""
import importlib.util
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))  # ComfyUI root

import torch

spec = importlib.util.spec_from_file_location(
    "nodes_audio_mask", os.path.join(PACK, "nodes_audio_mask.py"))
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)

import comfy.nested_tensor

Node = am.MVEx_AudioMaskToLatentNode
DebugNode = am.MVEx_AudioMaskDebugNode


class _H3Inner:
    latents_per_second = 40


class FakeH3AudioVAE:
    first_stage_model = _H3Inner()
    audio_sample_rate = 32000
    downscale_ratio = 800


class _LTXInner:
    latents_per_second = 25.0
    mel_bins = 64


class FakeLTXAudioVAE:
    first_stage_model = _LTXInner()
    audio_sample_rate = 44100
    downscale_ratio = 4096


class FakeFallbackAudioVAE:
    first_stage_model = object()
    audio_sample_rate = 48000
    downscale_ratio = 1920


AUTO = {"timing": "auto"}
MANUAL_H3 = {"timing": "manual", "latents_per_second": 40.0, "layout": "time last"}
MANUAL_LTX = {"timing": "manual", "latents_per_second": 25.0, "layout": "time then bins"}


def h3_latent(t=200):
    return {"samples": torch.zeros(1, 32, 2, t)}


def ltx_latent(t=125):
    return {"samples": torch.zeros(1, 128, t, 16)}


def joint_latent(t=200):
    video = torch.zeros(1, 24, 10, 8, 8)
    audio = torch.zeros(1, 32, 2, t)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def run(latent, timing=AUTO, start=0.0, end=1.0, existing="keep", mask=None, ranges="", vae=None):
    out = Node.execute(latent, dict(timing), start, end, existing,
                       mask=mask, time_ranges=ranges, vae=vae)
    return out.args[0]


failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def test_h3_start_end():
    out = run(h3_latent(), start=1.0, end=2.0, vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert nm.shape == (1, 1, 2, 200), nm.shape
    assert nm[..., 40:80].min() == 1.0
    assert nm.sum() == 40 * 2, nm.sum()


def test_ltx_time_axis():
    out = run(ltx_latent(), start=1.0, end=2.0, vae=FakeLTXAudioVAE())
    nm = out["noise_mask"]
    assert nm.shape == (1, 1, 125, 16), nm.shape
    assert nm[:, :, 25:50].min() == 1.0
    assert nm.sum() == 25 * 16, nm.sum()


def test_joint_nested():
    out = run(joint_latent(), start=1.0, end=2.0, vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert nm.is_nested
    vmask, amask = nm.unbind()
    assert vmask.shape == (1, 1, 10, 8, 8), vmask.shape
    assert vmask.max() == 0.0
    assert amask[..., 40:80].min() == 1.0
    assert amask.sum() == 40 * 2


def test_video_mask_passes_through():
    latent = joint_latent()
    prior = torch.ones(1, 1, 10, 8, 8)
    latent["noise_mask"] = prior
    out = run(latent, start=1.0, end=2.0, vae=FakeH3AudioVAE())
    vmask, amask = out["noise_mask"].unbind()
    assert torch.equal(vmask, prior)
    assert amask.sum() == 40 * 2


def test_time_ranges_priority():
    out = run(h3_latent(), start=2.0, end=3.0, ranges="0,0.5", vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert nm[..., :20].min() == 1.0
    assert nm.sum() == 20 * 2, nm.sum()


def test_timeline_mask():
    m = torch.zeros(1, 8, 100)
    m[:, :, :50] = 1.0
    out = run(h3_latent(), mask=m, vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert nm[..., :100].min() == 1.0
    assert nm[..., 100:].max() == 0.0


def test_soft_mask_survives():
    m = torch.full((1, 8, 100), 0.5)
    out = run(h3_latent(), mask=m, vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert abs(nm.max().item() - 0.5) < 1e-5
    assert abs(nm.min().item() - 0.5) < 1e-5


def test_chain_union():
    first = run(h3_latent(), start=0.0, end=1.0, vae=FakeH3AudioVAE())
    out = run(first, start=2.0, end=3.0, vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert nm[..., :40].min() == 1.0
    assert nm[..., 80:120].min() == 1.0
    assert nm.sum() == 80 * 2, nm.sum()


def test_replace():
    first = run(h3_latent(), start=0.0, end=1.0, vae=FakeH3AudioVAE())
    out = run(first, start=2.0, end=3.0, existing="replace", vae=FakeH3AudioVAE())
    nm = out["noise_mask"]
    assert nm[..., :40].max() == 0.0
    assert nm[..., 80:120].min() == 1.0


def test_auto_requires_vae():
    try:
        run(h3_latent(), start=1.0, end=2.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError without a VAE in auto timing")


def test_manual_timing():
    out = run(h3_latent(), timing=MANUAL_H3, start=1.0, end=2.0)
    assert out["noise_mask"].sum() == 40 * 2
    out = run(ltx_latent(), timing=MANUAL_LTX, start=1.0, end=2.0)
    nm = out["noise_mask"]
    assert nm[:, :, 25:50].min() == 1.0
    assert nm.sum() == 25 * 16


def test_fallback_rate():
    out = run(h3_latent(), start=1.0, end=2.0, vae=FakeFallbackAudioVAE())
    nm = out["noise_mask"]
    assert nm[..., 25:50].min() == 1.0
    assert nm.sum() == 25 * 2, nm.sum()


def test_audio_only_3dim():
    latent = {"samples": torch.zeros(1, 64, 200)}
    out = run(latent, timing=MANUAL_H3, start=1.0, end=2.0)
    nm = out["noise_mask"]
    assert nm.shape == (1, 1, 200), nm.shape
    assert nm[..., 40:80].min() == 1.0
    assert nm.sum() == 40


def test_debug_report():
    masked = run(h3_latent(), start=1.0, end=2.0, vae=FakeH3AudioVAE())
    report = DebugNode.execute(masked, dict(AUTO), vae=FakeH3AudioVAE()).args[0]
    assert "1.00-2.00s = 1 (generate)" in report, report
    assert "0.00-1.00s = 0 (keep)" in report, report

    report = DebugNode.execute(h3_latent(), dict(AUTO), vae=FakeH3AudioVAE()).args[0]
    assert "no noise mask" in report, report

    latent = joint_latent()
    latent["noise_mask"] = torch.ones(1, 1, 10, 8, 8)
    report = DebugNode.execute(latent, dict(AUTO), vae=FakeH3AudioVAE()).args[0]
    assert "video-only" in report, report


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
    check(name, fn)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nall tests passed")
