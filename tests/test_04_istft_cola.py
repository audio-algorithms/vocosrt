# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Test 04 -- ISTFT COLA correctness.

Per prompt sec.7.1: "Synthetic identity-mel passthrough; reconstruction error
<= -80 dB."

This test does NOT involve the neural backbone -- it isolates the streaming
inverse-STFT primitive. The setup:

  audio -> torch.stft -> frames -> StreamingISTFT.step (per frame) -> reconstructed

For Hann window with 75% overlap (hop_length = n_fft / 4) the perfect-reconstruction
COLA condition is satisfied, so reconstruction error should be at machine precision
in the steady-state region (after warmup). The -80 dB target is generous; we
typically see well below -100 dB.

Test cases cover three input families: tonal (sinusoid sweep), transient (impulse
train), and broadband (gaussian noise). The transient case exercises the OLA
boundary handling most aggressively -- a single sample's worth of energy must be
reconstructed across the 4 OLA-overlapping frames.
"""

from __future__ import annotations

import math

import pytest
import torch

from vocos_rt.streaming_stft import StreamingISTFT

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP = 256
WIN = 1024
DURATION_S = 1.0  # 1 second of audio = 24,000 samples = ~93 STFT frames

# Skip the warmup region from comparison: the first 4 hops have partial OLA.
# We trim the first WARMUP_FRAMES * HOP samples and the last WARMUP_FRAMES * HOP samples.
WARMUP_FRAMES = 4

# Target reconstruction error: -80 dB per spec; we expect to far exceed this.
TARGET_ERR_DB = -80.0


def _stft_frames(audio_padded: torch.Tensor) -> torch.Tensor:
    """Compute STFT of pre-padded audio. Returns complex (B, n_freq, T)."""
    window = torch.hann_window(WIN, dtype=audio_padded.dtype)
    return torch.stft(
        audio_padded,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=False,           # caller pre-padded
        return_complex=True,
    )


def _round_trip_compare(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """STFT -> StreamingISTFT round-trip. Returns aligned (reference, recon).

    The streaming ISTFT reconstructs ``audio_padded`` directly (its time index
    starts at audio_padded[0]). We compare against audio_padded itself, both
    trimmed of the warmup region at the start and the unreconstructed tail at
    the end.

    Trimming math:
      - audio_padded has length N + 2*pad
      - We feed T frames where T = (N + 2*pad - WIN) // HOP + 1
      - Streaming emits T*HOP samples reconstructing audio_padded[0:T*HOP]
      - The first WARMUP_FRAMES*HOP samples of recon are partial-OLA warmup
      - The last (n_fft - hop) samples of audio_padded never get reconstructed
        (they would need future frames that don't exist), so we trim
        WARMUP_FRAMES*HOP from the end too as a margin.
    """
    pad = (WIN - HOP) // 2
    audio_padded = torch.nn.functional.pad(audio, (pad, pad), mode="reflect")
    frames = _stft_frames(audio_padded)
    T_hop = frames.shape[-1] * HOP

    istft = StreamingISTFT(n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    istft.reset(batch_size=audio.shape[0])
    recon = istft.stream(frames)

    assert recon.shape[-1] == T_hop, (recon.shape[-1], T_hop)

    edge = WARMUP_FRAMES * HOP
    ref = audio_padded[..., edge : T_hop - edge]
    rec = recon[..., edge : T_hop - edge]
    return ref, rec


def _err_db(reference: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """20 * log10(||err|| / ||ref||) in dB."""
    err = reference - reconstructed
    err_rms = torch.sqrt(torch.mean(err ** 2))
    ref_rms = torch.sqrt(torch.mean(reference ** 2))
    return 20.0 * math.log10(float(err_rms) / float(ref_rms) + 1e-30)


# ------------------------------- tests -------------------------------


def test_streaming_istft_construct_invalid_args() -> None:
    with pytest.raises(ValueError):
        StreamingISTFT(n_fft=1024, hop_length=256, win_length=512)  # win != n_fft
    with pytest.raises(ValueError):
        StreamingISTFT(n_fft=1024, hop_length=300)                  # 300 doesn't divide 1024


def test_step_rejects_wrong_freq_bins() -> None:
    istft = StreamingISTFT()
    istft.reset(batch_size=1)
    bad = torch.zeros(1, 100, dtype=torch.complex64)  # should be 513
    with pytest.raises(ValueError):
        istft.step(bad)


def test_step_rejects_batch_size_change() -> None:
    istft = StreamingISTFT()
    istft.reset(batch_size=1)
    istft.step(torch.zeros(1, N_FFT // 2 + 1, dtype=torch.complex64))
    with pytest.raises(ValueError):
        istft.step(torch.zeros(2, N_FFT // 2 + 1, dtype=torch.complex64))


def test_algorithmic_latency_and_warmup() -> None:
    istft = StreamingISTFT(n_fft=1024, hop_length=256, win_length=1024)
    assert istft.algorithmic_latency_samples == 768
    assert istft.warmup_frames == 4


def test_cola_sinusoid() -> None:
    """A 440 Hz sinusoid round-trips below -80 dB after warmup."""
    n_samples = int(SAMPLE_RATE * DURATION_S)
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    audio = (0.5 * torch.sin(2 * math.pi * 440.0 * t)).unsqueeze(0)
    a, r = _round_trip_compare(audio)
    err_db = _err_db(a, r)
    assert err_db < TARGET_ERR_DB, f"sinusoid reconstruction error {err_db:.1f} dB > target {TARGET_ERR_DB:.1f} dB"


def test_cola_gaussian_noise() -> None:
    """Broadband noise round-trips below -80 dB after warmup."""
    torch.manual_seed(0)
    n_samples = int(SAMPLE_RATE * DURATION_S)
    audio = (0.3 * torch.randn(1, n_samples))
    a, r = _round_trip_compare(audio)
    err_db = _err_db(a, r)
    assert err_db < TARGET_ERR_DB, f"noise reconstruction error {err_db:.1f} dB > target {TARGET_ERR_DB:.1f} dB"


def test_cola_impulse_train() -> None:
    """A periodic impulse exercises OLA boundary handling. Below -80 dB after warmup."""
    n_samples = int(SAMPLE_RATE * DURATION_S)
    audio = torch.zeros(1, n_samples)
    audio[0, 1000::2000] = 0.5  # impulses every ~83 ms
    a, r = _round_trip_compare(audio)
    err_db = _err_db(a, r)
    assert err_db < TARGET_ERR_DB, f"impulse train reconstruction error {err_db:.1f} dB > target {TARGET_ERR_DB:.1f} dB"


def test_cola_chirp() -> None:
    """A frequency sweep covers the spectrum; should also round-trip below -80 dB."""
    n_samples = int(SAMPLE_RATE * DURATION_S)
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    f0, f1 = 100.0, 8000.0
    phase = 2 * math.pi * (f0 * t + (f1 - f0) / (2 * DURATION_S) * t * t)
    audio = (0.4 * torch.sin(phase)).unsqueeze(0)
    a, r = _round_trip_compare(audio)
    err_db = _err_db(a, r)
    assert err_db < TARGET_ERR_DB, f"chirp reconstruction error {err_db:.1f} dB > target {TARGET_ERR_DB:.1f} dB"


def test_n_frames_seen_increments_then_resets() -> None:
    istft = StreamingISTFT()
    istft.reset(batch_size=1)
    assert istft.n_frames_seen == 0
    for k in range(5):
        istft.step(torch.zeros(1, N_FFT // 2 + 1, dtype=torch.complex64))
        assert istft.n_frames_seen == k + 1
    istft.reset(batch_size=1)
    assert istft.n_frames_seen == 0


def test_stream_matches_per_step() -> None:
    """stream(frames) yields identical output to a sequence of step(frame_t)."""
    torch.manual_seed(1)
    pad = (WIN - HOP) // 2
    audio = 0.3 * torch.randn(1, int(SAMPLE_RATE * DURATION_S))
    audio_padded = torch.nn.functional.pad(audio, (pad, pad), mode="reflect")
    frames = _stft_frames(audio_padded)

    a = StreamingISTFT()
    a.reset(batch_size=1)
    out_a = a.stream(frames)

    b = StreamingISTFT()
    b.reset(batch_size=1)
    out_b_chunks = [b.step(frames[:, :, t]) for t in range(frames.shape[-1])]
    out_b = torch.cat(out_b_chunks, dim=-1)

    assert torch.equal(out_a, out_b), "stream() must equal sequential step() calls"


def test_state_dict_does_not_leak_per_stream_state() -> None:
    """The accumulator and frame counter are per-stream, not per-model, and must
    NOT appear in state_dict (would corrupt restored streams from another stream's
    mid-state)."""
    istft = StreamingISTFT()
    istft.reset(batch_size=1)
    istft.step(torch.zeros(1, N_FFT // 2 + 1, dtype=torch.complex64))
    keys = list(istft.state_dict().keys())
    # Allow window/window_sq if persistent; we registered them non-persistent so they're absent
    forbidden_substrings = ["accum", "env", "n_frames"]
    for key in keys:
        for forbidden in forbidden_substrings:
            assert forbidden not in key.lower(), f"per-stream state {key!r} leaked into state_dict"
