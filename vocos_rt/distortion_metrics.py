# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Objective distortion metrics for vocoder output.

Designed to detect the artifacts a listener would call:
- **clicks** : single-sample large jumps. Caught by ``max_jump``, ``jump_count_above``.
- **pops** : brief amplitude bursts spanning several samples. Caught by
  ``rms_envelope_outlier_count`` (sliding-RMS spikes vs the local norm).
- **distorted voice** : long-term spectral envelope drift away from a reference.
  Caught by ``log_mel_l1_vs_reference`` and ``spectral_centroid_drift``.
- **clipping / saturation** : caught by ``clip_count``.
- **DC offset / weird bias** : caught by ``dc_offset``.
- **NaN / Inf** : caught by ``nonfinite_count``.

All metrics are pure functions of a 1-D audio tensor (or pair, for the
reference-based ones). They do not require the model -- just the produced WAV.
This means we can run them on any audio file from any source and compare across
checkpoints, vocoders, or recording conditions.

Each metric returns a numeric value with units documented in its docstring;
the test harness asserts ranges relative to a reference (typically the upstream
non-causal Vocos output on the same mel input).
"""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor


# --------------------------------------------------------------- click metrics


def sample_diff_stats(audio: Tensor) -> dict[str, float]:
    """Per-sample first-difference distribution. Larger = more high-freq energy / clicks."""
    if audio.dim() != 1:
        raise ValueError(f"expected 1-D audio, got shape {tuple(audio.shape)}")
    diff = (audio[1:] - audio[:-1]).abs()
    return {
        "max_jump": float(diff.max()),
        "p9999_jump": float(diff.quantile(0.9999)),
        "p999_jump": float(diff.quantile(0.999)),
        "p99_jump": float(diff.quantile(0.99)),
        "mean_jump": float(diff.mean()),
    }


def jump_count_above(audio: Tensor, threshold: float) -> int:
    """Number of consecutive-sample jumps with abs value > threshold.

    A click is typically a single-sample swing of >0.1 in normalized audio.
    Pretrained streaming vs upstream gold both produce ~5/sec of these for natural
    speech; thousands per file would indicate distortion.
    """
    diff = (audio[1:] - audio[:-1]).abs()
    return int((diff > threshold).sum())


def jumps_per_second(audio: Tensor, threshold: float, sample_rate: int = 24_000) -> float:
    n = audio.shape[-1]
    duration_s = n / sample_rate
    return jump_count_above(audio, threshold) / max(duration_s, 1e-6)


# --------------------------------------------------------------- pop metrics


def rms_envelope_outlier_count(
    audio: Tensor,
    window_samples: int = 256,
    hop_samples: int = 64,
    n_sigma: float = 6.0,
) -> int:
    """Count of sliding-RMS frames whose value exceeds local mean + n_sigma * local stddev.

    Pops are brief bursts that don't show as single-sample jumps but as
    several-sample bumps that elevate the local RMS far above the surrounding norm.
    """
    if audio.shape[-1] < window_samples * 4:
        return 0
    # (T,) -> sliding windows -> RMS per window
    a2 = audio.pow(2)
    # Use unfold for non-overlapping-but-strided RMS
    frames = a2.unfold(-1, window_samples, hop_samples)  # (n_frames, window_samples)
    rms = frames.mean(dim=-1).sqrt()  # (n_frames,)
    # Local statistics with a wider window
    if rms.shape[-1] < 16:
        return 0
    local_window = max(8, rms.shape[-1] // 32)
    rms_pad = torch.nn.functional.pad(rms.unsqueeze(0).unsqueeze(0), (local_window, local_window), mode="replicate").squeeze(0).squeeze(0)
    rolling = rms_pad.unfold(0, 2 * local_window + 1, 1)[: rms.shape[-1]]
    local_mean = rolling.mean(dim=-1)
    local_std = rolling.std(dim=-1).clamp(min=1e-6)
    z = (rms - local_mean) / local_std
    return int((z > n_sigma).sum())


# --------------------------------------------------------------- spectral metrics


def log_mel_l1_vs_reference(
    audio: Tensor,
    reference: Tensor,
    sample_rate: int = 24_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 100,
) -> float:
    """L1 distance between log-mel spectrograms. Catches voice timbre distortion.

    Both inputs must be the same length; trimmed to min length if not.
    Returns the L1 distance per (mel-bin, frame). Lower is better; values for
    very-similar audio (e.g. upstream-gold vs pretrained-streaming) should be
    < 0.5; > 1.0 indicates significant timbre drift.
    """
    n = min(audio.shape[-1], reference.shape[-1])
    a = audio[..., :n].unsqueeze(0) if audio.dim() == 1 else audio[..., :n]
    r = reference[..., :n].unsqueeze(0) if reference.dim() == 1 else reference[..., :n]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        center=True, power=1.0,
    ).to(audio.device)
    log_a = (mel(a) + 1e-7).log()
    log_r = (mel(r) + 1e-7).log()
    return float((log_a - log_r).abs().mean())


def spectral_centroid_drift(
    audio: Tensor, reference: Tensor, sample_rate: int = 24_000,
) -> float:
    """Mean abs difference of the spectral centroid (Hz) between audio and reference."""
    n = min(audio.shape[-1], reference.shape[-1])
    a = audio[..., :n].unsqueeze(0) if audio.dim() == 1 else audio[..., :n]
    r = reference[..., :n].unsqueeze(0) if reference.dim() == 1 else reference[..., :n]
    sc = torchaudio.transforms.SpectralCentroid(sample_rate=sample_rate).to(audio.device)
    ca = sc(a)
    cr = sc(r)
    return float((ca - cr).abs().mean())


# --------------------------------------------------------------- amplitude / hygiene


def clip_count(audio: Tensor, threshold: float = 0.999) -> int:
    return int((audio.abs() >= threshold).sum())


def dc_offset(audio: Tensor) -> float:
    return float(audio.mean())


def nonfinite_count(audio: Tensor) -> int:
    return int((~torch.isfinite(audio)).sum())


def crest_factor(audio: Tensor) -> float:
    """Peak / RMS. Speech naturally 10-20; clicks push above that; flatness pulls below."""
    peak = float(audio.abs().max())
    rms = float(audio.pow(2).mean().sqrt())
    return peak / max(rms, 1e-9)


# --------------------------------------------------------------- tube / comb-filter metric


def hop_rate_ripple_db(
    audio: Tensor,
    reference: Tensor,
    sample_rate: int = 24_000,
    hop_length: int = 256,
    envelope_window_samples: int = 192,  # ~8 ms at 24 kHz
    envelope_hop_samples: int = 64,
) -> float:
    """Detect 'tube'/comb-filter artifact at the hop rate.

    Per adversarial agent: causal masking + constant OLA normalization leaks
    frame-correlated mag/phase errors as a periodic ripple at the hop rate
    (=24000/256 = 93.75 Hz). Periodic gain ripple in time = comb filter in
    frequency = 'tube' sound.

    Method: take the log-magnitude envelope of (audio / reference), DFT it,
    measure the SNR at the hop frequency vs. the median of neighboring bins.

    Returns SNR in dB at the hop frequency. Clean speech: < 3 dB. Tube: > 10 dB.
    """
    n = min(audio.shape[-1], reference.shape[-1])
    a = audio[..., :n].squeeze() if audio.dim() > 1 else audio[..., :n]
    r = reference[..., :n].squeeze() if reference.dim() > 1 else reference[..., :n]

    # Sliding RMS for both
    a2 = a.pow(2)
    r2 = r.pow(2)
    if a2.shape[-1] < envelope_window_samples * 4:
        return 0.0
    a_frames = a2.unfold(-1, envelope_window_samples, envelope_hop_samples)
    r_frames = r2.unfold(-1, envelope_window_samples, envelope_hop_samples)
    a_env = a_frames.mean(dim=-1).sqrt().clamp(min=1e-7).log()  # (n_frames,)
    r_env = r_frames.mean(dim=-1).sqrt().clamp(min=1e-7).log()
    ratio = a_env - r_env  # log-ratio; should be roughly constant if no comb
    ratio = ratio - ratio.mean()  # remove DC

    # Sample rate of the envelope signal
    env_sample_rate = sample_rate / envelope_hop_samples  # 24000/64 = 375 Hz
    hop_rate_hz = sample_rate / hop_length                # 24000/256 = 93.75 Hz

    # FFT of the envelope
    n_fft = ratio.shape[-1]
    spec = torch.fft.rfft(ratio, n=n_fft).abs()
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / env_sample_rate)

    # Find the bin closest to hop_rate_hz
    hop_bin = int(torch.argmin((freqs - hop_rate_hz).abs()))
    if hop_bin < 3 or hop_bin >= len(spec) - 3:
        return 0.0

    # SNR: amplitude at hop bin vs median of neighboring bins (excluding immediate neighbors)
    neighbor_lo = max(1, hop_bin - 10)
    neighbor_hi = min(len(spec) - 1, hop_bin + 10)
    neighborhood = torch.cat([spec[neighbor_lo:hop_bin - 2], spec[hop_bin + 3:neighbor_hi]])
    if neighborhood.numel() == 0:
        return 0.0
    noise_floor = float(neighborhood.median().clamp(min=1e-12))
    peak_amp = float(spec[hop_bin].clamp(min=1e-12))
    snr_db = 20.0 * (torch.tensor(peak_amp / noise_floor)).log10().item()
    return snr_db


# --------------------------------------------------------------- bundle


def all_metrics(audio: Tensor, reference: Tensor | None = None,
                sample_rate: int = 24_000) -> dict[str, float]:
    """Compute the full distortion bundle. ``reference`` enables the relative metrics."""
    out: dict[str, float] = {}
    out.update(sample_diff_stats(audio))
    out["jumps_per_sec_0p05"] = jumps_per_second(audio, 0.05, sample_rate)
    out["jumps_per_sec_0p10"] = jumps_per_second(audio, 0.10, sample_rate)
    out["jumps_per_sec_0p20"] = jumps_per_second(audio, 0.20, sample_rate)
    out["pop_count_6sigma"] = float(rms_envelope_outlier_count(audio, n_sigma=6.0))
    out["clip_count"] = float(clip_count(audio))
    out["dc_offset"] = dc_offset(audio)
    out["nonfinite_count"] = float(nonfinite_count(audio))
    out["crest_factor"] = crest_factor(audio)
    if reference is not None:
        out["log_mel_l1_vs_ref"] = log_mel_l1_vs_reference(audio, reference, sample_rate)
        out["spectral_centroid_drift_hz"] = spectral_centroid_drift(audio, reference, sample_rate)
        out["hop_rate_ripple_db"] = hop_rate_ripple_db(audio, reference, sample_rate)
    return out


def hop_rate_envelope_flatness_loss(
    audio_hat: Tensor,
    audio_real: Tensor,
    sample_rate: int = 24_000,
    hop_length: int = 256,
    env_window_samples: int = 192,
    env_hop_samples: int = 64,
    n_neighbor_bins: int = 1,
) -> Tensor:
    """Differentiable training loss: penalize hop-rate periodic ripple in the
    log-power-envelope ratio (audio_hat / audio_real).

    Targets the comb-filter ("tube") artifact directly. Per the DSP-specialist
    review: the hop-rate ripple is a frame-correlated mag/phase error pattern
    that mel/STFT-magnitude losses are blind to; this loss penalizes it
    explicitly in the modulation domain.

    Returns a scalar loss value. Differentiable through unfold, log, and FFT.
    Forced fp32 internally for numerical stability of the FFT peak read.

    Args:
        audio_hat: (B, T) generator output
        audio_real: (B, T) ground truth
        sample_rate: input sample rate (default 24000 → hop rate 93.75 Hz)
        hop_length: STFT hop in samples
        env_window_samples: sliding-RMS window for envelope extraction
        env_hop_samples: sliding-RMS hop
        n_neighbor_bins: how many FFT bins around hop_rate to include in the penalty
    """
    # Force fp32 -- bf16 FFT precision is poor at low magnitudes and can hide
    # the hop-rate peak we're trying to penalize.
    a = audio_hat.float()
    r = audio_real.float()
    n = min(a.shape[-1], r.shape[-1])
    a = a[..., :n]
    r = r[..., :n]

    if a.shape[-1] < env_window_samples * 8:  # need at least a few env frames for FFT
        return torch.zeros((), device=audio_hat.device, dtype=audio_hat.dtype)

    # Sliding RMS via unfold
    a2 = a.pow(2)
    r2 = r.pow(2)
    a_frames = a2.unfold(-1, env_window_samples, env_hop_samples)  # (..., n_env, win)
    r_frames = r2.unfold(-1, env_window_samples, env_hop_samples)
    a_env = a_frames.mean(dim=-1).clamp(min=1e-7)
    r_env = r_frames.mean(dim=-1).clamp(min=1e-7)

    # Log-ratio (per-element); subtract per-sample mean so DC doesn't leak in
    ratio = a_env.log() - r_env.log()
    ratio = ratio - ratio.mean(dim=-1, keepdim=True)

    # FFT the envelope ratio
    n_fft_env = ratio.shape[-1]
    spec = torch.fft.rfft(ratio, n=n_fft_env, dim=-1).abs()

    # Sample rate of the envelope signal & target hop-rate bin
    env_sr = sample_rate / env_hop_samples              # 24000/64 = 375 Hz
    hop_rate_hz = sample_rate / hop_length               # 24000/256 = 93.75 Hz
    freq_bin_size = env_sr / n_fft_env
    hop_bin = int(round(hop_rate_hz / freq_bin_size))
    if hop_bin < 1 or hop_bin >= spec.shape[-1] - 1:
        return torch.zeros((), device=audio_hat.device, dtype=audio_hat.dtype)

    lo = max(1, hop_bin - n_neighbor_bins)
    hi = min(spec.shape[-1], hop_bin + n_neighbor_bins + 1)
    # Penalize the squared magnitude (energy) at the hop-rate bin
    loss = spec[..., lo:hi].pow(2).mean()
    return loss.to(audio_hat.dtype)


__all__ = [
    "all_metrics",
    "clip_count",
    "crest_factor",
    "dc_offset",
    "hop_rate_envelope_flatness_loss",
    "hop_rate_ripple_db",
    "jump_count_above",
    "jumps_per_second",
    "log_mel_l1_vs_reference",
    "nonfinite_count",
    "rms_envelope_outlier_count",
    "sample_diff_stats",
    "spectral_centroid_drift",
]
