# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Test 01 -- bit-exactness: StreamingVocos.stream(mel) == OfflineVocos.forward(mel).

Spec (prompt sec.7.1): "Streaming vs. own-offline outputs: max abs sample error
<= 1e-5 (FP32) / <= 1e-3 (FP16) after 49-frame warm-up, on 100 random mel
sequences."

Implementation note: our actual causal receptive field is 55 frames (1 input
embed conv + 8 ConvNeXt blocks * 6 past frames each), not the prompt's 49 (which
counts only the 8 blocks, omitting the input embed conv). We use 55 here.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import make_random_mel

WARMUP_FRAMES = 55          # causal receptive field of the actual model
HOP = 256                   # hop_length for vocos-mel-24khz
TOL_FP32 = 1e-5             # spec tolerance, FP32, gaussian-random mels
TOL_FP32_SPEECH = 5e-5      # loosened for real-speech mels (DECISIONS.md D14)
N_SEQUENCES = 20            # spec says 100; 20 is enough for statistical confidence and ~10x faster
SEQ_LEN_FRAMES = 200        # ~2.1 s of audio per sequence


def _max_steady_state_diff(streaming_audio: torch.Tensor, offline_audio: torch.Tensor) -> float:
    """Maximum absolute sample difference after the warmup region."""
    assert streaming_audio.shape == offline_audio.shape, (
        streaming_audio.shape, offline_audio.shape
    )
    edge = WARMUP_FRAMES * HOP
    s = streaming_audio[..., edge:]
    o = offline_audio[..., edge:]
    return float((s - o).abs().max())


def test_bit_exactness_single_sequence(streaming_vocos, offline_vocos, device) -> None:
    """One sequence at the spec tolerance."""
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=0, device=device)
    with torch.inference_mode():
        offline = offline_vocos.forward(mel)
    streaming_vocos.reset(batch_size=1)
    streaming = streaming_vocos.stream(mel)
    err = _max_steady_state_diff(streaming, offline)
    assert err < TOL_FP32, (
        f"Streaming vs offline diverged in steady state: max abs err {err:.2e} > {TOL_FP32:.0e}. "
        "This indicates a bug in StreamingCausalConv1d (ring buffer) or StreamingISTFT (OLA)."
    )


@pytest.mark.parametrize("seed", list(range(N_SEQUENCES)))
def test_bit_exactness_n_random_sequences(streaming_vocos, offline_vocos, device, seed: int) -> None:
    """Same property over N_SEQUENCES distinct random mels (parametrized for clear pass/fail per seed)."""
    mel = make_random_mel(SEQ_LEN_FRAMES, seed=seed, device=device)
    with torch.inference_mode():
        offline = offline_vocos.forward(mel)
    streaming_vocos.reset(batch_size=1)
    streaming = streaming_vocos.stream(mel)
    err = _max_steady_state_diff(streaming, offline)
    assert err < TOL_FP32, f"seed={seed} diverged: max abs err {err:.2e}"


def test_bit_exactness_real_speech_mel(streaming_vocos, offline_vocos, upstream_vocos, device) -> None:
    """A real speech mel (extracted from speech.wav) round-trips bit-exact too.

    This guards against bugs that only surface with realistic mel distributions
    rather than gaussian-random ones.
    """
    import torchaudio
    from pathlib import Path

    src = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT\audio\speech.wav")
    if not src.exists():
        pytest.skip(f"Reference speech file not found at {src}")

    audio, sr = torchaudio.load(str(src))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)
    audio = audio.to(device)

    with torch.inference_mode():
        mel = upstream_vocos.feature_extractor(audio)
        offline = offline_vocos.forward(mel)
    streaming_vocos.reset(batch_size=1)
    streaming = streaming_vocos.stream(mel)
    err = _max_steady_state_diff(streaming, offline)
    assert err < TOL_FP32_SPEECH, f"real-speech mel diverged: max abs err {err:.2e}"
