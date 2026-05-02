# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Public streaming-Vocos API.

``StreamingVocos`` wraps an upstream ``vocos.Vocos`` pretrained checkpoint and
runs it one mel frame at a time, producing one hop of audio per call.

Usage:
    >>> v = StreamingVocos.from_pretrained("charactr/vocos-mel-24khz")
    >>> v.reset(batch_size=1)
    >>> for t in range(mel.shape[-1]):
    ...     audio_hop = v.step(mel[:, :, t : t + 1])  # (1, 100, 1) -> (1, 256)
    ...     play(audio_hop)
"""

from __future__ import annotations

import torch
import vocos
from torch import Tensor, nn

from vocos_rt.causal_conv import StreamingCausalConv1d, StreamingCausalConvNeXtBlock
from vocos_rt.streaming_stft import StreamingISTFT


class StreamingVocos(nn.Module):
    """Causal streaming wrapper around the upstream Vocos mel-24khz model.

    Hot-swaps the input embed conv and each ConvNeXt block's depthwise conv to
    causal-streaming variants. All other modules (LayerNorms, pointwise convs,
    GELU, gamma, head's Linear) are reused unchanged. The ISTFT head is
    replaced with our ``StreamingISTFT``.

    Causal receptive field: 1 + 9*(K-1) = 55 frames (~587 ms at hop=256, sr=24kHz).

    Per-stream state lives inside each ``StreamingCausalConv1d`` and
    ``StreamingISTFT``; ``reset(batch_size)`` clears all of them.
    """

    def __init__(self, upstream: vocos.Vocos):
        super().__init__()
        backbone = upstream.backbone
        head = upstream.head

        # Discover dims from the loaded model rather than hardcoding.
        self.input_channels: int = backbone.embed.in_channels   # 100 for mel
        self.dim: int = backbone.embed.out_channels             # 512
        self.n_fft: int = head.istft.n_fft
        self.hop_length: int = head.istft.hop_length
        self.win_length: int = head.istft.win_length

        # Causalize the input embed conv (K=7, padding=3 -> padding=0 + ring buffer)
        self.embed = StreamingCausalConv1d(backbone.embed)

        # First per-frame LayerNorm (shared with upstream module)
        self.input_norm = backbone.norm

        # Causalize each ConvNeXt block's dwconv; reuse the rest of each block
        self.blocks = nn.ModuleList(
            [StreamingCausalConvNeXtBlock(blk) for blk in backbone.convnext]
        )

        # Final per-frame LayerNorm
        self.final_layer_norm = backbone.final_layer_norm

        # Head's projection (Linear: dim -> n_fft + 2). Shared with upstream.
        self.head_out = head.out

        # Streaming ISTFT replacing the upstream offline-only ISTFT
        self.istft = StreamingISTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        # Move the streaming ISTFT's window to the same device as the model
        self.istft.to(next(self.head_out.parameters()).device)

        # Number of layers that own per-stream state (1 embed + N blocks)
        self.num_causal_layers = 1 + len(self.blocks)

    # ------------------------- construction -------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "charactr/vocos-mel-24khz",
        device: str | torch.device = "cpu",
    ) -> StreamingVocos:
        """Load upstream Vocos from HuggingFace and wrap it. ``upstream.eval()`` is invoked."""
        upstream = vocos.Vocos.from_pretrained(repo_id)
        upstream.to(device)
        upstream.eval()
        return cls(upstream)

    # ------------------------- streaming control -------------------------

    def reset(self, batch_size: int = 1) -> None:
        """Clear all per-stream state. Call before the first ``step`` of a new stream."""
        self.embed.reset(batch_size)
        for blk in self.blocks:
            blk.reset(batch_size)
        self.istft.reset(batch_size)

    @property
    def algorithmic_latency_samples(self) -> int:
        """ISTFT structural delay only (the ConvNeXt depth is causal, so no extra delay)."""
        return self.istft.algorithmic_latency_samples

    @property
    def algorithmic_latency_ms(self) -> float:
        return self.algorithmic_latency_samples * 1000.0 / 24_000.0

    @property
    def causal_receptive_field_frames(self) -> int:
        """Frames of past mel context that influence the current output frame."""
        # 1 (embed) + 8 (ConvNeXt blocks) layers * (K-1=6) past frames each
        return 1 + self.num_causal_layers * 6

    # ------------------------- inference -------------------------

    @torch.inference_mode()
    def step(self, mel_frame: Tensor) -> Tensor:
        """One mel frame in -> ``hop_length`` audio samples out.

        Args:
            mel_frame: (B, input_channels, 1)

        Returns:
            (B, hop_length) real audio
        """
        if mel_frame.dim() != 3 or mel_frame.shape[-1] != 1:
            raise ValueError(f"step expects (B, {self.input_channels}, 1), got {tuple(mel_frame.shape)}")
        # Backbone embed (B, 100, 1) -> (B, 512, 1)
        x = self.embed.step(mel_frame)
        # First LayerNorm: per-frame, applied along channel dim
        x = x.transpose(1, 2)  # (B, 1, 512)
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # (B, 512, 1)
        # 8 ConvNeXt blocks
        for blk in self.blocks:
            x = blk.step(x)
        # Final LayerNorm
        x = x.transpose(1, 2)  # (B, 1, 512)
        x = self.final_layer_norm(x)
        # Project to spectral coefficients (B, 1, n_fft+2)
        x = self.head_out(x)
        # Reshape to (B, n_fft+2, 1) and split mag/phase
        x = x.transpose(1, 2)
        mag, p = x.chunk(2, dim=1)              # each (B, n_fft//2+1, 1)
        mag = torch.exp(mag).clamp(max=1e2)
        S = mag * (torch.cos(p) + 1j * torch.sin(p))   # (B, n_fft//2+1, 1) complex
        spectral_frame = S.squeeze(-1)           # (B, n_fft//2+1) complex
        # Streaming ISTFT -> hop samples
        audio = self.istft.step(spectral_frame)
        return audio

    @torch.inference_mode()
    def stream(self, mel_chunk: Tensor) -> Tensor:
        """Convenience: feed many mel frames in one call.

        Args:
            mel_chunk: (B, input_channels, T)

        Returns:
            (B, T * hop_length) real audio
        """
        if mel_chunk.dim() != 3:
            raise ValueError(f"stream expects (B, {self.input_channels}, T), got {tuple(mel_chunk.shape)}")
        outs = [self.step(mel_chunk[:, :, t : t + 1]) for t in range(mel_chunk.shape[-1])]
        return torch.cat(outs, dim=-1)


__all__ = ["StreamingVocos"]
