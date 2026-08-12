"""Functional tests for Mask To Latent Space."""
import importlib.util
import math
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))  # ComfyUI root

import torch

spec = importlib.util.spec_from_file_location(
    "nodes_mask_to_latent", os.path.join(PACK, "nodes_mask_to_latent.py"))
m2l = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2l)

Node = m2l.MVEx_MaskToLatentSpaceNode
InverseNode = m2l.MVEx_LatentMaskToMaskNode


class FakeWanVAE:
    def __init__(self):
        self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
        self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)

    def spacial_compression_encode(self):
        return self.downscale_ratio[-1]

    def temporal_compression_decode(self):
        return round(self.upscale_ratio[0](8192) / 8192)


class _FakeMiniMaxInner:
    # chunk parameters as exposed by comfy.ldm.minimax.vae.VideoVAE
    vae_ratio_t = 4
    tokens_chunk_size = 5
    frame_pre_padding = 3
    token_drop = 3


class FakeMiniMaxVAE:
    def __init__(self):
        self.first_stage_model = _FakeMiniMaxInner()
        self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
        self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)

    def spacial_compression_encode(self):
        return self.downscale_ratio[-1]

    def temporal_compression_decode(self):
        return round(self.upscale_ratio[0](8192) / 8192)


class FakeImageVAE:
    downscale_ratio = 8
    upscale_ratio = 8

    def spacial_compression_encode(self):
        return 8

    def temporal_compression_decode(self):
        return None


def run(masks, compression, spatial_method="max", temporal_method="max",
        grow_spatial=0, grow_temporal=0, vae=None):
    if compression.get("compression") == "auto" and vae is None and "vae" in compression:
        vae = compression.pop("vae")
    out = Node.execute(masks, compression, spatial_method, temporal_method,
                       grow_spatial, grow_temporal, vae=vae)
    return out.args[0]


AUTO = lambda: {"compression": "auto", "vae": FakeWanVAE()}
failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def test_auto_wan_shape():
    masks = torch.zeros(9, 64, 64)
    out = run(masks, AUTO())
    assert out.shape == (3, 8, 8), out.shape  # formula(9) = 3, 64/8 = 8


def test_auto_wan_temporal_alignment():
    # frame 0 -> latent 0; frames 1-4 -> latent 1; frames 5-8 -> latent 2
    masks = torch.zeros(9, 64, 64)
    masks[0, :16, :16] = 1.0
    out = run(masks, AUTO())
    assert out[0].max() == 1.0 and out[1].max() == 0.0 and out[2].max() == 0.0

    masks = torch.zeros(9, 64, 64)
    masks[6, :16, :16] = 1.0  # frame 6 is in group 5-8 -> latent 2
    out = run(masks, AUTO())
    assert out[2].max() == 1.0 and out[0].max() == 0.0 and out[1].max() == 0.0


def test_max_vs_min_spatial():
    masks = torch.zeros(1, 64, 64)
    masks[0, 0:4, 0:8] = 1.0  # half of the top-left 8x8 cell
    out_max = run(masks, AUTO())
    out_min = run(masks, AUTO(), spatial_method="min")
    assert out_max[0, 0, 0] == 1.0, out_max[0, 0, 0]
    assert out_min[0, 0, 0] == 0.0, out_min[0, 0, 0]
    full = torch.zeros(1, 64, 64)
    full[0, 0:8, 0:8] = 1.0
    assert run(full, AUTO(), spatial_method="min")[0, 0, 0] == 1.0


def test_temporal_max_vs_mean():
    masks = torch.zeros(5, 64, 64)
    masks[2, :8, :8] = 1.0  # 1 of 4 frames in the group
    assert run(masks, AUTO())[1, 0, 0] == 1.0
    assert abs(run(masks, AUTO(), temporal_method="mean")[1, 0, 0] - 0.25) < 1e-5


def MANUAL(spatial=8, head_frames=1, head_latents=1, chunk_frames=4, chunk_latents=1,
           frames_per_latent=""):
    return {"compression": "manual", "spatial": spatial,
            "head_frames": head_frames, "head_latents": head_latents,
            "chunk_frames": chunk_frames, "chunk_latents": chunk_latents,
            "frames_per_latent": frames_per_latent}


def test_manual_no_temporal():
    masks = torch.zeros(9, 64, 64)
    out = run(masks, MANUAL(chunk_frames=1))
    assert out.shape == (9, 8, 8), out.shape


def test_manual_uniform_grouping():
    masks = torch.zeros(8, 64, 64)
    out = run(masks, MANUAL(head_frames=0, head_latents=0))
    assert out.shape == (2, 8, 8), out.shape


def test_manual_wan_matches_auto():
    masks = torch.rand(9, 64, 64)
    assert torch.equal(run(masks, MANUAL()), run(masks, AUTO()))


def test_auto_minimax_shape():
    # 124 frames on the 17k+5 grid -> 5k+2 = 37 latent frames, 16x spatial
    masks = torch.zeros(124, 64, 64)
    out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape == (37, 4, 4), out.shape


def test_auto_minimax_tail_alignment():
    # a mask only in the final frames must land in the final latent frame
    masks = torch.zeros(124, 64, 64)
    masks[121:, :16, :16] = 1.0
    out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out[36].max() == 1.0
    assert out[:36].max() == 0.0

    masks = torch.zeros(124, 64, 64)
    masks[0, :16, :16] = 1.0
    out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out[0].max() == 1.0 and out[1:].max() == 0.0


def test_auto_minimax_off_grid_keeps_tail():
    # 38 frames is off the 17k+5 grid: the encoder pads to 3 chunks and trims,
    # yielding 12 latents; the last real frame lands in the last kept group
    masks = torch.zeros(38, 64, 64)
    masks[37, :16, :16] = 1.0
    out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape[0] == 12, out.shape
    assert out[11].max() == 1.0


def test_auto_minimax_matches_encoder_count_all_lengths():
    # encode_temporal pads to whole 17-frame clips and drops the last 3
    # tokens, so the latent always has 5*ceil(t/17) - 3 frames
    for t in [2, 6, 17, 18, 23, 30, 34, 51, 124, 150, 375]:
        masks = torch.zeros(t, 64, 64)
        out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
        expected = 5 * math.ceil(t / 17) - 3
        assert out.shape[0] == expected, (t, out.shape[0], expected)


def test_auto_minimax_trimmed_tail_merges():
    # 23 frames: frame 22 falls in a trimmed token, so its coverage merges
    # into the last kept latent instead of creating an extra group
    masks = torch.zeros(23, 64, 64)
    masks[22, :16, :16] = 1.0
    out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape[0] == 7, out.shape
    assert out[6].max() == 1.0 and out[:6].max() == 0.0


def test_auto_minimax_padding_latent_repeats_last_frame():
    # 18 frames: the 7th latent covers only encoder padding, which repeats
    # frame 17, so it takes frame 17's mask
    masks = torch.zeros(18, 64, 64)
    masks[17, :16, :16] = 1.0
    out = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape[0] == 7, out.shape
    assert out[5].max() == 1.0 and out[6].max() == 1.0
    assert out[:5].max() == 0.0


def test_inverse_auto_minimax_off_grid_counts():
    # 7 latents decode from the canonical 22-frame source (17 + 5)
    lat = torch.zeros(7, 4, 4)
    out = run_inverse(lat, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape[0] == 22, out.shape
    lat = torch.zeros(2, 4, 4)
    out = run_inverse(lat, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape[0] == 5, out.shape


def test_manual_minimax_shape():
    masks = torch.zeros(124, 64, 64)
    out = run(masks, MANUAL(spatial=16, head_frames=5, head_latents=2,
                            chunk_frames=17, chunk_latents=5))
    assert out.shape == (37, 4, 4), out.shape


def PATTERN(spatial=16, frames_per_latent="1,4,4,4,4"):
    return MANUAL(spatial=spatial, frames_per_latent=frames_per_latent)


def test_pattern_minimax_shape():
    # 124 frames = 7*17+5 -> 37 latent frames, same count as the encoder
    masks = torch.zeros(124, 64, 64)
    out = run(masks, PATTERN())
    assert out.shape == (37, 4, 4), out.shape


def test_pattern_minimax_exact_groups():
    # the VAE's causal cycle: frame 0 alone, then fours; frame 16 closes
    # latent 4, frame 17 opens latent 5 alone
    masks = torch.zeros(124, 64, 64)
    masks[16, :16, :16] = 1.0
    out = run(masks, PATTERN())
    assert out[4].max() == 1.0 and out[:4].max() == 0.0 and out[5:].max() == 0.0

    masks = torch.zeros(124, 64, 64)
    masks[17, :16, :16] = 1.0
    out = run(masks, PATTERN())
    assert out[5].max() == 1.0 and out[:5].max() == 0.0 and out[6:].max() == 0.0


def test_pattern_fixes_uniform_grid_blind_spot():
    # the head 5->2 chunk 17->5 grid assigns frame 16 to latent 5's group, so
    # latent 4 (which actually encodes it) gets no coverage; pattern mode does
    masks = torch.zeros(124, 64, 64)
    masks[16, :16, :16] = 1.0
    old = run(masks, MANUAL(spatial=16, head_frames=5, head_latents=2,
                            chunk_frames=17, chunk_latents=5))
    new = run(masks, PATTERN())
    assert old[4].max() == 0.0
    assert new[4].max() == 1.0


def test_pattern_inverse_roundtrip():
    lat = torch.zeros(37, 4, 4)
    lat[4] = 1.0
    out = run_inverse(lat, PATTERN())
    assert out.shape == (124, 64, 64), out.shape
    tprofile = out.amax(dim=(1, 2))
    assert tprofile.tolist() == [0] * 13 + [1] * 4 + [0] * 107, tprofile.tolist()


def test_manual_pattern_matches_auto_minimax():
    masks = torch.rand(124, 64, 64)
    assert torch.equal(run(masks, PATTERN()),
                       run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()}))


def test_pattern_off_grid_tail():
    # 120 frames ends mid-cycle: the last latent group is clipped to the video
    masks = torch.zeros(120, 64, 64)
    masks[119, :16, :16] = 1.0
    out = run(masks, PATTERN())
    assert out.shape[0] == 36, out.shape  # latent 36 would start at frame 120
    assert out[35].max() == 1.0


def test_image_vae():
    masks = torch.rand(6, 64, 64)
    out = run(masks, {"compression": "auto", "vae": FakeImageVAE()})
    assert out.shape == (6, 8, 8), out.shape


def test_grow_spatial():
    masks = torch.zeros(1, 64, 64)
    masks[0, 32, 32] = 1.0
    base = run(masks, AUTO())
    grown = run(masks, AUTO(), grow_spatial=8)
    assert grown.sum() > base.sum(), (base.sum(), grown.sum())
    shrunk = run(masks, AUTO(), grow_spatial=-2)
    assert shrunk.sum() == 0, shrunk.sum()


def test_grow_temporal():
    masks = torch.zeros(9, 64, 64)
    masks[5, :8, :8] = 1.0
    base = run(masks, AUTO())
    grown = run(masks, AUTO(), grow_temporal=1)  # spreads to frames 4 and 6
    assert base[1].max() == 0.0
    assert grown[1].max() == 1.0  # frame 4 is in group 1-4 -> latent 1


def test_auto_without_vae_raises():
    try:
        run(torch.zeros(5, 64, 64), {"compression": "auto"})
    except ValueError:
        return
    raise AssertionError("expected ValueError when vae is not connected")


def run_inverse(mask, compression, frames=0, vae=None):
    if compression.get("compression") == "auto" and vae is None and "vae" in compression:
        vae = compression.pop("vae")
    out = InverseNode.execute(mask, compression, frames, vae=vae)
    return out.args[0]


def test_inverse_auto_wan():
    # latent frame 1 covers pixel frames 1-4
    lat = torch.zeros(3, 8, 8)
    lat[1, 0, 0] = 1.0
    out = run_inverse(lat, AUTO())
    assert out.shape == (9, 64, 64), out.shape
    tprofile = out.amax(dim=(1, 2))
    assert tprofile.tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0], tprofile.tolist()
    assert out[1, :8, :8].min() == 1.0 and out[1, 8:, :].max() == 0.0


def test_inverse_auto_minimax():
    # latent 36 is chunk 7 phase 1: pixel frames 120-123
    lat = torch.zeros(37, 4, 4)
    lat[36] = 1.0
    out = run_inverse(lat, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert out.shape == (124, 64, 64), out.shape
    tprofile = out.amax(dim=(1, 2))
    assert tprofile[:120].max() == 0.0 and tprofile[120:].min() == 1.0


def test_inverse_explicit_frames_off_grid():
    # latent 6 covers frames 18-21; later groups reuse the last latent frame,
    # so its paint extends through the remaining frames
    lat = torch.zeros(7, 4, 4)
    lat[6] = 1.0
    out = run_inverse(lat, {"compression": "auto", "vae": FakeMiniMaxVAE()}, frames=38)
    assert out.shape == (38, 64, 64), out.shape
    tprofile = out.amax(dim=(1, 2))
    assert tprofile[18:].min() == 1.0
    assert tprofile[:18].max() == 0.0


def test_inverse_manual_matches_auto():
    lat = torch.rand(3, 8, 8)
    assert torch.equal(run_inverse(lat, MANUAL()), run_inverse(lat, AUTO()))


def test_inverse_image_vae():
    lat = torch.rand(6, 8, 8)
    out = run_inverse(lat, {"compression": "auto", "vae": FakeImageVAE()})
    assert out.shape == (6, 64, 64), out.shape


def test_roundtrip_minimax():
    # forward then inverse: mask painted over exactly the frames whose group was hit
    masks = torch.zeros(124, 64, 64)
    masks[60, :32, :32] = 1.0
    lat = run(masks, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    back = run_inverse(lat, {"compression": "auto", "vae": FakeMiniMaxVAE()})
    assert back.shape == (124, 64, 64), back.shape
    assert back[60, :32, :32].min() == 1.0  # original coverage survives
    assert back[0].max() == 0.0 and back[123].max() == 0.0


def test_soft_mask_values_survive():
    masks = torch.full((5, 64, 64), 0.5)
    out = run(masks, AUTO())
    assert abs(out.max().item() - 0.5) < 1e-5


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
    check(name, fn)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nall tests passed")
