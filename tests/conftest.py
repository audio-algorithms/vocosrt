# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Shared pytest fixtures for the vocos_rt test suite.

Loads the upstream ``charactr/vocos-mel-24khz`` checkpoint once per test session
and shares it across all parity tests. Tests that need fresh state should use
``streaming.reset(batch_size=...)`` rather than constructing new wrappers.
"""

from __future__ import annotations

import warnings

import pytest
import torch

# Suppress upstream's noisy warnings on Windows
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
warnings.filterwarnings("ignore", category=UserWarning, module="vocos.*")


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Test device: CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def upstream_vocos(device: torch.device):  # type: ignore[no-untyped-def]
    """Loaded upstream Vocos model (charactr/vocos-mel-24khz). Shared across the session."""
    import vocos
    m = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")
    m.to(device)
    m.eval()
    return m


@pytest.fixture(scope="session")
def streaming_vocos(upstream_vocos, device: torch.device):  # type: ignore[no-untyped-def]
    """A StreamingVocos wrapping the shared upstream. Caller must reset() before each use."""
    from vocos_rt.streaming_vocos import StreamingVocos
    s = StreamingVocos(upstream_vocos)
    s.to(device)
    return s


@pytest.fixture(scope="session")
def offline_vocos(upstream_vocos, device: torch.device):  # type: ignore[no-untyped-def]
    """An OfflineVocos wrapping the shared upstream. Bit-exact reference for streaming."""
    from vocos_rt.offline_vocos import OfflineVocos
    o = OfflineVocos(upstream_vocos)
    o.to(device)
    return o


def make_random_mel(
    n_frames: int,
    *,
    batch_size: int = 1,
    n_mels: int = 100,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a deterministic random log-mel tensor in a realistic range.

    Vocos was trained on log-mel features in roughly [-12, +5]; we sample from a
    Gaussian centered at -3 with stddev 2 to stay in distribution.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    mel = torch.randn(batch_size, n_mels, n_frames, generator=g) * 2.0 - 3.0
    return mel.to(device)
