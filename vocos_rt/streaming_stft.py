# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Streaming inverse-STFT primitive for the vocoder output head.

The upstream Vocos ISTFT (vocos.spectral_ops.ISTFT, padding="same") consumes a
full sequence of complex spectral frames in one shot and reconstructs the audio
via overlap-add with a hann window. For real-time use we need the equivalent
mechanism, but driven one frame at a time, with bounded memory and a single
emitted hop of audio per call.

Algorithm (one call to ``step``):

  1. Receive one frequency-domain frame, shape (B, n_fft//2 + 1), complex.
  2. Inverse FFT -> (B, n_fft) real, then multiply by the hann window.
  3. Add the n_fft windowed samples into a sliding accumulator of length n_fft
     and a parallel envelope accumulator (running sum of window^2). The
     accumulator's leftmost ``hop_length`` samples now have all the
     contributions they will ever receive (no future frame's window reaches
     back that far).
  4. Emit ``self._accum[:, :hop_length] / self._env[:hop_length]`` -- the OLA
     normalization. Envelope is clamped to a small floor to avoid div-by-zero
     at the very start of the stream where the hann window is near-zero.
  5. Shift both accumulators left by ``hop_length`` and zero-fill the rightmost
     ``hop_length`` samples (those positions will be filled by future frames).

Algorithmic latency: ``n_fft - hop_length`` samples (768 = 32 ms for the Vocos
defaults). This is the irreducible delay of a 4-frame OLA at 75% overlap.

The first ``n_fft // hop_length - 1`` (= 3) emitted hops are warm-up: their
envelope sums are smaller than the steady-state value, so the output mirrors
the boundary behavior of the offline ISTFT with ``padding="same"`` -- the
first samples are attenuated by the partial overlap. This is what makes the
streaming output bit-exact with the offline reference (for matching frame
sequences), modulo floating-point accumulation order.

The class is an ``nn.Module`` so the hann window registers as a buffer and
moves with ``.to(device)``. Per-stream state (``_accum``, ``_env``, frame
counter) is held as plain attributes -- it is per-stream, not per-model,
so it must NOT participate in ``state_dict``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StreamingISTFT(nn.Module):
    """Streaming inverse-STFT with a sliding overlap-add accumulator.

    Args:
        n_fft: FFT size. Must equal ``win_length`` for the upstream-matching path.
        hop_length: Frame stride. Must divide ``n_fft`` evenly.
        win_length: Window length. Must equal ``n_fft``.
        device: Initial device for the hann window buffer.
        dtype: Float dtype for the accumulator and window.

    Example:
        >>> istft = StreamingISTFT(n_fft=1024, hop_length=256, win_length=1024)
        >>> istft.reset(batch_size=1)
        >>> for frame in frames:                  # frame: (1, 513) complex
        ...     audio_chunk = istft.step(frame)   # audio_chunk: (1, 256) real
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if win_length != n_fft:
            raise ValueError(
                f"win_length ({win_length}) must equal n_fft ({n_fft}) for the "
                "upstream-matching path"
            )
        if n_fft % hop_length != 0:
            raise ValueError(
                f"hop_length ({hop_length}) must divide n_fft ({n_fft}) evenly"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = torch.hann_window(win_length, device=device, dtype=dtype)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("window_sq", window.square(), persistent=False)

        # Per-stream state -- NOT registered as buffers so they don't leak into state_dict.
        # Initialized lazily in reset(), since batch size is not known at construction.
        self._accum: Tensor | None = None
        self._env: Tensor | None = None
        self._n_frames: int = 0
        self._dtype = dtype

    # ------------------------------------------------------------------ public

    def reset(self, batch_size: int = 1) -> None:
        """Clear the per-stream state. Call before the first ``step`` of a new stream."""
        device = self.window.device
        self._accum = torch.zeros(batch_size, self.n_fft, device=device, dtype=self._dtype)
        self._env = torch.zeros(self.n_fft, device=device, dtype=self._dtype)
        self._n_frames = 0

    @property
    def algorithmic_latency_samples(self) -> int:
        """Irreducible output-side delay in samples."""
        return self.n_fft - self.hop_length

    @property
    def warmup_frames(self) -> int:
        """Number of frames after which OLA reaches steady state."""
        return self.n_fft // self.hop_length

    @property
    def n_frames_seen(self) -> int:
        """Number of ``step`` calls since the last ``reset``."""
        return self._n_frames

    def step(self, frame: Tensor) -> Tensor:
        """Feed one frequency-domain frame; return one hop of time-domain samples.

        Args:
            frame: complex tensor of shape ``(B, n_fft // 2 + 1)``.

        Returns:
            Real tensor of shape ``(B, hop_length)``.
        """
        if self._accum is None:
            self.reset(batch_size=frame.shape[0])
        # Type narrowing for the static checker
        assert self._accum is not None and self._env is not None

        if frame.shape[0] != self._accum.shape[0]:
            raise ValueError(
                f"batch size changed mid-stream: frame batch {frame.shape[0]} "
                f"!= accumulator batch {self._accum.shape[0]} (call reset() first)"
            )
        expected_freq = self.n_fft // 2 + 1
        if frame.shape[-1] != expected_freq:
            raise ValueError(
                f"expected {expected_freq} frequency bins, got {frame.shape[-1]}"
            )

        # 1. Inverse FFT and window
        time_domain = torch.fft.irfft(frame, n=self.n_fft, dim=-1, norm="backward")
        time_domain = time_domain * self.window  # broadcasts over batch dim

        # 2. Add to accumulators
        self._accum = self._accum + time_domain
        self._env = self._env + self.window_sq
        self._n_frames += 1

        # 3. Emit leftmost hop samples (envelope-normalized)
        # Floor on envelope to handle the very-near-zero hann window samples at the
        # start of stream; the corresponding samples in _accum are also near-zero so
        # the result is dominated by the floor only at a few boundary samples.
        emit_env = self._env[: self.hop_length].clamp(min=1e-11)
        emit = self._accum[:, : self.hop_length] / emit_env

        # 4. Shift left by hop, zero-fill rightmost hop positions
        new_accum = torch.zeros_like(self._accum)
        new_accum[:, : -self.hop_length] = self._accum[:, self.hop_length :]
        self._accum = new_accum

        new_env = torch.zeros_like(self._env)
        new_env[: -self.hop_length] = self._env[self.hop_length :]
        self._env = new_env

        return emit

    def stream(self, frames: Tensor) -> Tensor:
        """Convenience wrapper: feed many frames, return concatenated hops.

        Args:
            frames: complex tensor of shape ``(B, n_freq, T)``.

        Returns:
            Real tensor of shape ``(B, T * hop_length)``.
        """
        if frames.dim() != 3:
            raise ValueError(f"expected 3D (B, n_freq, T), got shape {tuple(frames.shape)}")
        outputs = []
        for t in range(frames.shape[-1]):
            outputs.append(self.step(frames[:, :, t]))
        return torch.cat(outputs, dim=-1)


__all__ = ["StreamingISTFT"]
