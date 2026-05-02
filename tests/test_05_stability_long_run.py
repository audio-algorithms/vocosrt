# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Test 05 -- long-run stability.

Spec (prompt sec.7.1): ">= 1 hour stream, no NaN/Inf, drift bounded (running RMS
error stationary, not accumulating)."

Implementation: 1 hour at 24 kHz with hop=256 == 337,500 mel frames. At our
streaming RTF ~0.5 on the target RTX 3050, that's ~30 minutes wall-clock for
this single test. Pytest will not run it by default -- it is gated behind the
``@pytest.mark.long`` marker. To run:

    pytest tests/test_05_stability_long_run.py -m long -v

A short variant (1 minute = 5,625 frames, ~3 s wall-clock) runs in the default
suite and catches anything that breaks within a typical session length.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.conftest import make_random_mel

HOP = 256
SAMPLE_RATE = 24_000


def _frames_for_seconds(seconds: float) -> int:
    return int(math.ceil(seconds * SAMPLE_RATE / HOP))


def _stream_long(streaming_vocos, offline_vocos, device, n_frames: int, chunk_frames: int) -> dict:
    """Run a long stream in chunks; return summary diagnostics.

    Computes per-chunk RMS error vs offline as a stationarity proxy. Tracks
    NaN/Inf occurrence and amplitude bounds.
    """
    streaming_vocos.reset(batch_size=1)
    err_rms_per_chunk: list[float] = []
    out_rms_per_chunk: list[float] = []
    nan_seen = False
    inf_seen = False
    out_min = float("inf")
    out_max = float("-inf")

    seed = 0
    n_remaining = n_frames
    while n_remaining > 0:
        this_chunk = min(chunk_frames, n_remaining)
        mel = make_random_mel(this_chunk, seed=seed, device=device)
        with torch.inference_mode():
            offline = offline_vocos.forward(mel)
        streaming = streaming_vocos.stream(mel)

        if torch.isnan(streaming).any():
            nan_seen = True
        if torch.isinf(streaming).any():
            inf_seen = True

        out_min = min(out_min, float(streaming.min()))
        out_max = max(out_max, float(streaming.max()))
        out_rms_per_chunk.append(float(streaming.pow(2).mean().sqrt()))

        # Compare in steady state region only -- offline is computed independently per
        # chunk (no carry-over), so the first ~55 frames of each chunk's offline differ
        # from streaming's continuation. We only compare the deep middle of each chunk.
        if this_chunk > 110:
            mid_lo = 55 * HOP
            mid_hi = (this_chunk - 10) * HOP  # tail margin for OLA tail
            err = (streaming[..., mid_lo:mid_hi] - offline[..., mid_lo:mid_hi]).abs()
            err_rms_per_chunk.append(float(err.pow(2).mean().sqrt()))

        seed += 1
        n_remaining -= this_chunk

    return {
        "nan_seen": nan_seen,
        "inf_seen": inf_seen,
        "out_min": out_min,
        "out_max": out_max,
        "err_rms_per_chunk": err_rms_per_chunk,
        "out_rms_per_chunk": out_rms_per_chunk,
        "n_chunks": len(out_rms_per_chunk),
    }


def _assert_stable(diag: dict) -> None:
    assert not diag["nan_seen"], "NaN detected in streaming output"
    assert not diag["inf_seen"], "Inf detected in streaming output"
    # Output amplitude should stay in a sane range for randomly generated mel inputs.
    # The streaming wrapper produces audio in the un-normalized range -- typical peak
    # for noisy mel input is ~1-50 for the as-is causal weights. We only assert no runaway.
    assert diag["out_max"] < 1e4, f"output max {diag['out_max']:.2e} suggests runaway"
    assert diag["out_min"] > -1e4, f"output min {diag['out_min']:.2e} suggests runaway"

    # Per-chunk RMS error must NOT grow over the run -- proves no slow drift.
    if len(diag["err_rms_per_chunk"]) >= 4:
        early = diag["err_rms_per_chunk"][: len(diag["err_rms_per_chunk"]) // 2]
        late = diag["err_rms_per_chunk"][len(diag["err_rms_per_chunk"]) // 2 :]
        early_mean = sum(early) / len(early)
        late_mean = sum(late) / len(late)
        # Allow late mean to be up to 10x the early mean -- this is *very* generous;
        # in practice both are at FP-noise floor. Anything beyond 10x means real drift.
        assert late_mean <= early_mean * 10 + 1e-5, (
            f"per-chunk RMS error grew during the run: early {early_mean:.2e} -> "
            f"late {late_mean:.2e}; suggests numeric drift in streaming state"
        )


def test_short_1_minute_stream_is_stable(streaming_vocos, offline_vocos, device) -> None:
    """1-minute stream, runs in the default suite -- catches obvious drift quickly."""
    n_frames = _frames_for_seconds(60.0)
    diag = _stream_long(
        streaming_vocos, offline_vocos, device,
        n_frames=n_frames, chunk_frames=512,
    )
    _assert_stable(diag)


@pytest.mark.long
def test_full_1_hour_stream_is_stable(streaming_vocos, offline_vocos, device) -> None:
    """Full 1-hour stream per spec. Marked ``long``; not in default suite."""
    n_frames = _frames_for_seconds(60.0 * 60.0)
    diag = _stream_long(
        streaming_vocos, offline_vocos, device,
        n_frames=n_frames, chunk_frames=2048,
    )
    _assert_stable(diag)
