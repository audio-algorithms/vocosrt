# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Causal-streaming wrappers for ``nn.Conv1d`` and ``ConvNeXtBlock``.

The upstream Vocos backbone has nine convolutional layers with kernel size 7
and ``padding=3`` (one input embed conv plus eight ConvNeXt blocks' depthwise
convs). All of them are bidirectional in time. The mechanical streaming
conversion for any such conv is:

    Original (non-causal): output[t] = sum_{k=-3..+3} W[k+3] * x[t+k]
    Causal:                output[t] = sum_{k=-6..0}  W[k+6] * x[t+k]

Implementation:
- Build a copy of the conv with ``padding=0`` (kernel weights and bias unchanged).
- Maintain a ring buffer of the last ``K-1 = 6`` past input frames.
- On each ``step(frame)``: concatenate ``[buffer, frame]`` -> shape (B, C, K),
  apply ``conv``, get a single output frame (B, C', 1). Update buffer.

Parity vs. offline causal mode (Phase 1 exit gate test 01) is bit-exact because
the convolution kernel weights are unchanged; only the receptive field's
position relative to ``t`` is shifted.

Quality vs. the original non-causal model is *not* preserved -- the kernel
weights were trained for bidirectional context, so the causal interpretation
will sound smeared/dull. Phase 2 fine-tunes the same architecture under causal
masking to recover most of the lost quality.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from vocos.modules import ConvNeXtBlock


class StreamingCausalConv1d(nn.Module):
    """Wrap a non-causal ``nn.Conv1d`` (K>=2, ``padding=(K-1)//2``) for streaming.

    Holds a per-stream ring buffer of the last ``K-1`` input frames. The wrapped
    conv has ``padding=0`` and identical weights to ``base``.

    State:
        ``_buffer``: (B, in_channels, K-1) -- the last K-1 input frames.
                     Initialized to zeros; updated each ``step``. Per-stream,
                     not registered as a buffer (does not appear in state_dict).

    Limitations: only stride=1, dilation=1 are supported.
    """

    def __init__(self, base: nn.Conv1d) -> None:
        super().__init__()
        if base.stride != (1,):
            raise ValueError(f"Only stride=1 is supported (got {base.stride})")
        if base.dilation != (1,):
            raise ValueError(f"Only dilation=1 is supported (got {base.dilation})")
        K = base.kernel_size[0]
        if K < 2:
            raise ValueError(f"Kernel size must be >= 2 (got {K})")
        # Replicate the conv with padding=0; copy weights and bias (no clone needed --
        # nn.Conv1d's __init__ allocates fresh tensors which we overwrite immediately
        # with assigned references).
        new_conv = nn.Conv1d(
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            kernel_size=K,
            stride=1,
            padding=0,
            dilation=1,
            groups=base.groups,
            bias=base.bias is not None,
        )
        new_conv.weight = nn.Parameter(base.weight.detach().clone())
        if base.bias is not None:
            new_conv.bias = nn.Parameter(base.bias.detach().clone())
        self.conv = new_conv
        self.kernel_size = K
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels

        # Per-stream ring buffer, lazily allocated in reset().
        self._buffer: Tensor | None = None

    # --------- streaming API ---------

    def reset(self, batch_size: int = 1) -> None:
        device = self.conv.weight.device
        dtype = self.conv.weight.dtype
        self._buffer = torch.zeros(
            batch_size, self.in_channels, self.kernel_size - 1,
            device=device, dtype=dtype,
        )

    def step(self, x: Tensor) -> Tensor:
        """Push one frame through the causal conv.

        Args:
            x: (B, in_channels, 1)

        Returns:
            (B, out_channels, 1)
        """
        if self._buffer is None:
            self.reset(batch_size=x.shape[0])
        assert self._buffer is not None
        if x.dim() != 3 or x.shape[-1] != 1:
            raise ValueError(f"step() expects (B, C, 1), got {tuple(x.shape)}")
        # Concatenate buffer with new frame: (B, C, K)
        combined = torch.cat([self._buffer, x], dim=-1)
        # Conv with padding=0 collapses K time positions to 1
        out = self.conv(combined)
        # Update buffer: drop oldest, append new (slide left by 1)
        self._buffer = combined[:, :, 1:]
        return out

    # --------- offline API (used for the reference path / parity tests) ---------

    def forward_offline(self, x: Tensor) -> Tensor:
        """Apply the causal conv to a full sequence in one pass.

        Args:
            x: (B, in_channels, T)

        Returns:
            (B, out_channels, T) -- left-padded with zeros, conv applied with padding=0.
        """
        K = self.kernel_size
        x = torch.nn.functional.pad(x, (K - 1, 0))
        return self.conv(x)


class StreamingCausalConvNeXtBlock(nn.Module):
    """ConvNeXt block where the depthwise conv is causal-streaming; rest is unchanged.

    Reuses the upstream block's LayerNorm, pointwise convs, GELU, and gamma
    parameter as-is (no copies; we share the modules so weight updates from a
    fine-tune flow through). Only the depthwise conv is replaced with a
    StreamingCausalConv1d.
    """

    def __init__(self, base: ConvNeXtBlock) -> None:
        super().__init__()
        if base.adanorm:
            raise NotImplementedError("AdaLayerNorm path not used by the mel-24khz model")
        self.dwconv = StreamingCausalConv1d(base.dwconv)
        self.norm = base.norm
        self.pwconv1 = base.pwconv1
        self.act = base.act
        self.pwconv2 = base.pwconv2
        self.gamma = base.gamma  # nn.Parameter or None

    def reset(self, batch_size: int = 1) -> None:
        self.dwconv.reset(batch_size)

    def step(self, x: Tensor) -> Tensor:
        """One frame in -> one frame out. Shape (B, dim, 1)."""
        residual = x
        x = self.dwconv.step(x)
        x = x.transpose(1, 2)  # (B, 1, dim)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, dim, 1)
        x = residual + x
        return x

    def forward_offline(self, x: Tensor) -> Tensor:
        """Full-sequence causal evaluation. Shape (B, dim, T) -> (B, dim, T)."""
        residual = x
        x = self.dwconv.forward_offline(x)
        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, dim, T)
        return residual + x


__all__ = ["StreamingCausalConv1d", "StreamingCausalConvNeXtBlock"]
