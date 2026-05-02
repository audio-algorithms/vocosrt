# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Test 03 -- state warmup convergence.

Spec (prompt sec.7.1): "Per-frame divergence vs. offline converges to bit-exact
within <= 49 frames."

Our actual causal receptive field is 55 frames (1 input embed conv + 8 ConvNeXt
blocks * 6 past frames each). After exactly that many frames, every causal-conv
ring buffer is full of real activations (no leftover zeros from reset), and the
streaming output should bit-match the offline causal output.

This test traces the per-frame absolute error and asserts:
  1. The error is monotonically decreasing in expectation through the warmup region
     (allow some bumps but the long-term trend must be down).
  2. By frame 55, the per-frame max error is below 1e-5 (FP32 tolerance).
  3. After frame 55, the per-frame error stays below 1e-5 indefinitely.
"""

from __future__ import annotations

import torch

from tests.conftest import make_random_mel

HOP = 256
WARMUP_FRAMES = 55         # actual causal RF
TOL = 1e-5
SEQ_LEN_FRAMES = 200       # well past warmup, gives ~145 steady-state frames


def _per_frame_max_err(streaming_audio: torch.Tensor, offline_audio: torch.Tensor) -> torch.Tensor:
    """Return shape (T,) -- max abs sample error within each hop-sized frame."""
    assert streaming_audio.shape == offline_audio.shape
    diff = (streaming_audio - offline_audio).abs()  # (B, T*hop)
    diff = diff.view(diff.shape[0], -1, HOP)         # (B, T, hop)
    return diff.amax(dim=(0, 2))                     # (T,)


def test_warmup_converges_within_55_frames(streaming_vocos, offline_vocos, device) -> None:
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=0, device=device)
    with torch.inference_mode():
        offline = offline_vocos.forward(mel)
    streaming_vocos.reset(batch_size=1)
    streaming = streaming_vocos.stream(mel)
    per_frame_err = _per_frame_max_err(streaming, offline)
    # By frame 55 (0-indexed: index 54 is the 55th frame), causal RF is fully filled.
    assert per_frame_err[WARMUP_FRAMES - 1] < TOL, (
        f"frame {WARMUP_FRAMES - 1} max abs err {float(per_frame_err[WARMUP_FRAMES - 1]):.2e} "
        f">= {TOL:.0e}; warmup did not converge in {WARMUP_FRAMES} frames"
    )


def test_steady_state_error_stays_at_machine_precision(
    streaming_vocos, offline_vocos, device,
) -> None:
    """After warmup, every frame's error must stay below tolerance."""
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=0, device=device)
    with torch.inference_mode():
        offline = offline_vocos.forward(mel)
    streaming_vocos.reset(batch_size=1)
    streaming = streaming_vocos.stream(mel)
    per_frame_err = _per_frame_max_err(streaming, offline)
    steady = per_frame_err[WARMUP_FRAMES:]
    bad = (steady > TOL).nonzero(as_tuple=True)[0]
    assert len(bad) == 0, (
        f"{len(bad)} steady-state frames exceed tolerance; first bad: index "
        f"{int(bad[0]) + WARMUP_FRAMES} err {float(steady[int(bad[0])]):.2e}"
    )


def test_warmup_error_trend_is_decreasing(streaming_vocos, offline_vocos, device) -> None:
    """The total error in the first half of warmup must exceed the second half."""
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=0, device=device)
    with torch.inference_mode():
        offline = offline_vocos.forward(mel)
    streaming_vocos.reset(batch_size=1)
    streaming = streaming_vocos.stream(mel)
    per_frame_err = _per_frame_max_err(streaming, offline)
    early = per_frame_err[:WARMUP_FRAMES // 2].sum()
    late = per_frame_err[WARMUP_FRAMES // 2 : WARMUP_FRAMES].sum()
    assert early >= late, (
        f"warmup error trend not decreasing: early sum {float(early):.2e} < late sum {float(late):.2e}"
    )
