# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Test 02 -- chunk invariance.

Spec (prompt sec.7.1): "Identical input streamed at chunk sizes {1, 4, 16, 64,
256} produces identical outputs after warm-up."

Property: for any chunk size K, calling ``stream(mel[:, :, kK:(k+1)K])`` in
sequence (without reset between chunks) produces the same total output as
``stream(mel)`` in one shot. This proves the streaming state perfectly carries
across call boundaries -- no resets, no re-warmups, no off-by-ones in the ring
buffer or OLA accumulator.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import make_random_mel

CHUNK_SIZES = [1, 4, 16, 64, 256]
HOP = 256
WARMUP_FRAMES = 55
TOL = 1e-5
SEQ_LEN_FRAMES = 600  # >= 256 so chunk size 256 produces multiple chunks


def _stream_in_chunks(streaming_vocos, mel: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Reset, then stream the mel in fixed-size chunks; return concatenated audio."""
    streaming_vocos.reset(batch_size=mel.shape[0])
    outs = []
    for start in range(0, mel.shape[-1], chunk_size):
        end = min(start + chunk_size, mel.shape[-1])
        outs.append(streaming_vocos.stream(mel[:, :, start:end]))
    return torch.cat(outs, dim=-1)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_chunk_invariance_random_mel(streaming_vocos, device, chunk_size: int) -> None:
    """Chunked streaming equals one-shot streaming, after warmup, for random mel."""
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=0, device=device)
    one_shot = _stream_in_chunks(streaming_vocos, mel, chunk_size=SEQ_LEN_FRAMES)
    chunked = _stream_in_chunks(streaming_vocos, mel, chunk_size=chunk_size)
    assert one_shot.shape == chunked.shape, (one_shot.shape, chunked.shape)
    edge = WARMUP_FRAMES * HOP
    err = float((one_shot[..., edge:] - chunked[..., edge:]).abs().max())
    assert err < TOL, (
        f"chunk_size={chunk_size} diverged from one-shot: max abs err {err:.2e}. "
        "State is not carrying cleanly across stream() boundaries."
    )


def test_chunk_invariance_warmup_region_too(streaming_vocos, device) -> None:
    """Chunk size shouldn't affect even the warmup region (state starts identical = zeros)."""
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=1, device=device)
    one_shot = _stream_in_chunks(streaming_vocos, mel, chunk_size=SEQ_LEN_FRAMES)
    chunked = _stream_in_chunks(streaming_vocos, mel, chunk_size=4)
    err = float((one_shot - chunked).abs().max())
    assert err < TOL, (
        f"warmup-region chunked output diverged from one-shot: {err:.2e}. "
        "Initial-state behavior is not chunk-invariant."
    )


def test_chunk_invariance_uneven_final_chunk(streaming_vocos, device) -> None:
    """Last chunk is shorter than the others -- must still produce identical output."""
    n_frames = 130  # 130 = 4*32 + 2 -- last chunk of size 4 will be only 2 frames
    mel = make_random_mel(n_frames, seed=2, device=device)
    one_shot = _stream_in_chunks(streaming_vocos, mel, chunk_size=n_frames)
    chunked = _stream_in_chunks(streaming_vocos, mel, chunk_size=4)
    err = float((one_shot - chunked).abs().max())
    assert err < TOL, f"uneven trailing chunk diverged: {err:.2e}"
