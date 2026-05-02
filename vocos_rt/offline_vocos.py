# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Offline-causal Vocos: same model as ``StreamingVocos`` but full-sequence forward.

Two roles:

1. **Reference oracle for the bit-exactness tests (01/02/03).** ``OfflineVocos.forward(mel)``
   is the ground truth that ``StreamingVocos.stream(mel)`` must match within FP tolerance
   after warmup. Any divergence indicates a bug in the streaming wrapper (ring buffer,
   ISTFT accumulator, or padding alignment).

2. **Training target for Phase 2 fine-tune.** Per-frame streaming is ~100x too slow to
   run inside an SGD loop. The fine-tune script computes losses on
   ``OfflineVocos.forward(mel)`` and backpropagates; ``StreamingVocos`` then deploys the
   resulting weights. Test 01 guarantees the deployed inference matches what the loss
   function saw.

Architectural identity with StreamingVocos:
- Same backbone: causalized input embed conv + 8 causalized ConvNeXt blocks + final LN
- Same head: Linear(dim -> n_fft+2) + ISTFT
- Same weights (shares the wrapped upstream module references)
- Same causal-conv math (left-pad with zeros, padding=0 conv, identical kernels)
- Same OLA + steady-state-envelope normalization

The differences from StreamingVocos are purely structural: full-sequence forward instead
of per-frame, no ring buffer, vectorized OLA via ``torch.nn.functional.fold``.
"""

from __future__ import annotations

import torch
import vocos
from torch import Tensor, nn

from vocos_rt.causal_conv import StreamingCausalConv1d, StreamingCausalConvNeXtBlock


class OfflineVocos(nn.Module):
    """Full-sequence causal Vocos. Same weights and math as ``StreamingVocos``."""

    def __init__(self, upstream: vocos.Vocos):
        super().__init__()
        backbone = upstream.backbone
        head = upstream.head

        self.input_channels: int = backbone.embed.in_channels
        self.dim: int = backbone.embed.out_channels
        self.n_fft: int = head.istft.n_fft
        self.hop_length: int = head.istft.hop_length
        self.win_length: int = head.istft.win_length

        # Same causal modules as StreamingVocos (reuses the StreamingCausalConv1d
        # constructor that clones the upstream weights). We use the .forward_offline
        # methods below instead of .step.
        self.embed = StreamingCausalConv1d(backbone.embed)
        self.input_norm = backbone.norm
        self.blocks = nn.ModuleList(
            [StreamingCausalConvNeXtBlock(blk) for blk in backbone.convnext]
        )
        self.final_layer_norm = backbone.final_layer_norm
        self.head_out = head.out

        # Hann window for the offline ISTFT; non-persistent so it doesn't bloat state_dict
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

        # Steady-state envelope -- same as StreamingISTFT. Constant 1.5 for Hann + 75%.
        n_overlap = self.n_fft // self.hop_length
        steady = torch.zeros(self.hop_length)
        for i in range(self.hop_length):
            for k in range(n_overlap):
                idx = i + k * self.hop_length
                if idx < self.n_fft:
                    steady[i] = steady[i] + window[idx] ** 2
        self.register_buffer("steady_env", steady, persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "charactr/vocos-mel-24khz",
        device: str | torch.device = "cpu",
    ) -> OfflineVocos:
        upstream = vocos.Vocos.from_pretrained(repo_id)
        upstream.to(device)
        upstream.eval()
        m = cls(upstream)
        m.to(device)
        return m

    @property
    def causal_receptive_field_frames(self) -> int:
        return 1 + (1 + len(self.blocks)) * 6

    def forward(self, mel: Tensor) -> Tensor:
        """Full-sequence causal forward. (B, input_channels, T) -> (B, T*hop_length)."""
        if mel.dim() != 3 or mel.shape[1] != self.input_channels:
            raise ValueError(
                f"expected (B, {self.input_channels}, T), got {tuple(mel.shape)}"
            )
        # Embed (causal): (B, 100, T) -> (B, 512, T)
        x = self.embed.forward_offline(mel)
        # Per-frame LayerNorm
        x = x.transpose(1, 2)  # (B, T, 512)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        # ConvNeXt blocks
        for blk in self.blocks:
            x = blk.forward_offline(x)
        # Final norm
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)
        # Head Linear: (B, T, dim) -> (B, T, n_fft+2)
        x = self.head_out(x)
        x = x.transpose(1, 2)  # (B, n_fft+2, T)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        S = mag * (torch.cos(p) + 1j * torch.sin(p))  # (B, n_freq, T) complex
        # ISTFT (full sequence)
        return self._istft_full(S)

    def _istft_full(self, spec: Tensor) -> Tensor:
        """OLA-add T frames into a single audio signal of length T*hop_length.

        Bit-equivalent (modulo floating-point summation order) to
        ``StreamingISTFT.stream(spec)``: one windowed irfft per frame, sum at
        time-aligned offsets, divide by the steady-state envelope (constant 1.5
        for Hann + 75% overlap).
        """
        B, n_freq, T = spec.shape
        if n_freq != self.n_fft // 2 + 1:
            raise ValueError(f"expected {self.n_fft // 2 + 1} frequency bins, got {n_freq}")

        # 1. Inverse FFT each frame: (B, n_fft, T) real
        ifft = torch.fft.irfft(spec, n=self.n_fft, dim=1, norm="backward")
        # 2. Window: (B, n_fft, T)
        ifft = ifft * self.window[None, :, None]
        # 3. OLA via fold: contributes ifft[:, :, k] to output positions [k*hop, k*hop + n_fft)
        fold_out_len = (T - 1) * self.hop_length + self.n_fft
        audio = torch.nn.functional.fold(
            ifft,
            output_size=(1, fold_out_len),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]  # (B, fold_out_len)

        # 4. Truncate to T*hop_length to match what streaming emits (streaming never emits
        # the unreconstructed tail of (n_fft - hop) samples).
        out_len = T * self.hop_length
        audio = audio[:, :out_len]

        # 5. Steady-state envelope normalization. The steady_env is a length-hop_length
        # vector; we tile it across T frames to broadcast against the full audio.
        # Reshape (B, T*hop) -> (B, T, hop), divide, reshape back.
        audio = audio.view(B, T, self.hop_length) / self.steady_env
        audio = audio.reshape(B, out_len)
        return audio


__all__ = ["OfflineVocos"]
