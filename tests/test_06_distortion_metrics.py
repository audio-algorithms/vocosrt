# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Test 06 -- audio distortion suite.

Catches the perceptual artifacts that mel/STFT losses don't penalize:
clicks, pops, voice-quality distortion. Calibrated against EMPIRICAL data
from the upstream non-causal Vocos output (the gold reference).

Thresholds are set so:
- upstream_gold passes everything (trivial baseline)
- streaming_pretrained (Phase 1, "smeared but clean") FAILS log_mel_l1 (spectral
  smearing IS the distortion of this output) but PASSES click metrics
- D15/D16/D17 fine-tunes FAIL on click signatures (max_jump and crest_factor
  significantly higher than upstream) but pass spectral
- A model that passes ALL tests has both no clicks AND no smearing -- the goal

Two threshold types per metric:
  ABS  - absolute upper bound (catches catastrophic regardless of input variance)
  REL  - relative to upstream-gold value on the same mel input
         (catches degradation even on naturally-transient content)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
import torchaudio

warnings.filterwarnings("ignore")

from vocos_rt.distortion_metrics import (  # noqa: E402
    crest_factor,
    jumps_per_second,
    log_mel_l1_vs_reference,
    nonfinite_count,
    rms_envelope_outlier_count,
    sample_diff_stats,
)

WORKSPACE_AUDIO = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT\audio")
SAMPLE_RATE = 24_000
WARMUP_SKIP = SAMPLE_RATE // 10

SPEECH_INPUTS = [
    "speech.wav",
    "s05942-callirrhoe-female-en-us.wav",
    "s100680-algieba-male-en-us.wav",
    "s99387-callirrhoe-female-en-us.wav",
]

# Empirically calibrated 2026-05-02 from baseline_distortion.py against the four
# inputs above. Tightened slightly so a discriminator-trained model still has
# room to pass while the click-affected D15/D16/D17 fine-tunes fail.
MAX_JUMP_REL_RATIO = 1.4          # >1.4x upstream's max sample jump = audible click
CREST_FACTOR_REL_RATIO = 1.5      # >1.5x upstream's peak/rms = peaky/click signature
JUMPS_PER_SEC_0P10_REL_RATIO = 1.6
LOG_MEL_L1_ABS = 0.50             # >0.50 vs upstream = audible spectral smearing
POP_COUNT_ABS = 10                # >10 RMS-envelope 6-sigma outliers per file
NONFINITE_ABS = 0                 # any NaN/Inf is a fatal defect


# -------------------------------- per-input fixtures --------------------------------


@pytest.fixture(scope="module")
def speech_mel_pairs(upstream_vocos, device):
    """For each input: (audio_24k_1d, mel, upstream_gold_audio_1d)."""
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name in SPEECH_INPUTS:
        path = WORKSPACE_AUDIO / name
        if not path.exists():
            continue
        audio, sr = torchaudio.load(str(path))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.to(device)
        with torch.inference_mode():
            mel = upstream_vocos.feature_extractor(audio)
            gold = upstream_vocos.decode(mel).squeeze(0)
        pairs[name] = (audio.squeeze(0), mel, gold)
    return pairs


def _check_distortion(model_audio: torch.Tensor, gold_audio: torch.Tensor,
                      name: str, model_label: str,
                      check_log_mel: bool = True) -> None:
    """Apply the full distortion test bundle. Collects all failures, asserts non-empty."""
    body = model_audio[WARMUP_SKIP:] if model_audio.dim() == 1 else model_audio[..., WARMUP_SKIP:].squeeze(0)
    gold_body = gold_audio[WARMUP_SKIP:] if gold_audio.dim() == 1 else gold_audio[..., WARMUP_SKIP:].squeeze(0)

    failures: list[str] = []

    nf = nonfinite_count(body)
    if nf > NONFINITE_ABS:
        failures.append(f"nonfinite={nf} (max {NONFINITE_ABS})")

    stats = sample_diff_stats(body)
    gold_stats = sample_diff_stats(gold_body)
    max_jump_limit = gold_stats["max_jump"] * MAX_JUMP_REL_RATIO
    if stats["max_jump"] > max_jump_limit:
        failures.append(
            f"max_jump={stats['max_jump']:.3f} > {max_jump_limit:.3f} "
            f"({MAX_JUMP_REL_RATIO}x upstream {gold_stats['max_jump']:.3f})"
        )

    j10_model = jumps_per_second(body, 0.10, SAMPLE_RATE)
    j10_gold = jumps_per_second(gold_body, 0.10, SAMPLE_RATE)
    j10_limit = max(j10_gold * JUMPS_PER_SEC_0P10_REL_RATIO, 5.0)
    if j10_model > j10_limit:
        failures.append(
            f"jumps_per_sec_0p10={j10_model:.1f} > {j10_limit:.1f} "
            f"({JUMPS_PER_SEC_0P10_REL_RATIO}x upstream {j10_gold:.1f})"
        )

    cf_model = crest_factor(body)
    cf_gold = crest_factor(gold_body)
    cf_limit = cf_gold * CREST_FACTOR_REL_RATIO
    if cf_model > cf_limit:
        failures.append(
            f"crest_factor={cf_model:.1f} > {cf_limit:.1f} "
            f"({CREST_FACTOR_REL_RATIO}x upstream {cf_gold:.1f})"
        )

    pops = rms_envelope_outlier_count(body)
    if pops > POP_COUNT_ABS:
        failures.append(f"pop_count_6sigma={pops} > {POP_COUNT_ABS}")

    if check_log_mel:
        d = log_mel_l1_vs_reference(body, gold_body, SAMPLE_RATE)
        if d > LOG_MEL_L1_ABS:
            failures.append(f"log_mel_l1_vs_upstream={d:.3f} > {LOG_MEL_L1_ABS} (spectral smearing)")

    assert not failures, f"{model_label} on {name} -- failed gates: {'; '.join(failures)}"


# -------------------------------- gold baseline --------------------------------


@pytest.mark.parametrize("name", SPEECH_INPUTS)
def test_upstream_gold_passes_distortion(speech_mel_pairs, name):
    """upstream non-causal Vocos must pass all distortion gates trivially.

    If this fails, the metric thresholds are too tight relative to the natural
    transient content of speech.
    """
    if name not in speech_mel_pairs:
        pytest.skip(f"{name} not present")
    _, _, gold = speech_mel_pairs[name]
    # Compare gold to itself for the spectral check -> trivially 0
    _check_distortion(gold, gold, name, "upstream_gold")


# -------------------------------- documentation tests --------------------------------
# These intentionally show what each EXISTING variant fails on, so the test
# suite is itself a record of what we observed across attempts. They use
# pytest.xfail (expected to fail in their current state).


@pytest.mark.xfail(reason="Phase 1 streaming_pretrained: spectral smearing IS the distortion of this output")
@pytest.mark.parametrize("name", SPEECH_INPUTS)
def test_streaming_pretrained_no_distortion(streaming_vocos, speech_mel_pairs, name):
    if name not in speech_mel_pairs:
        pytest.skip(f"{name} not present")
    _, mel, gold = speech_mel_pairs[name]
    streaming_vocos.reset(batch_size=1)
    a = streaming_vocos.stream(mel).squeeze(0)
    _check_distortion(a, gold, name, "streaming_pretrained")


# Click-only check (no spectral) so the streaming_pretrained passes click gates
# even though it fails spectral gates. Documents that pretrained has no clicks.
@pytest.mark.parametrize("name", SPEECH_INPUTS)
def test_streaming_pretrained_no_clicks(streaming_vocos, speech_mel_pairs, name):
    if name not in speech_mel_pairs:
        pytest.skip(f"{name} not present")
    _, mel, gold = speech_mel_pairs[name]
    streaming_vocos.reset(batch_size=1)
    a = streaming_vocos.stream(mel).squeeze(0)
    _check_distortion(a, gold, name, "streaming_pretrained", check_log_mel=False)
