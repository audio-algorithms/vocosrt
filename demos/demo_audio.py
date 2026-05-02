# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Phase 1 audio sanity demo.

For each of the four convenience WAVs in the workspace ``audio/`` folder:
  1. Load + resample to 24 kHz mono float32
  2. Extract log-mel via upstream Vocos feature_extractor
  3. Reconstruct three ways:
       a. ``input``           - the resampled input itself (reference)
       b. ``offline_orig``    - upstream non-causal Vocos (gold reference for the vocoder)
       c. ``streaming_causal`` - vocos_rt streaming, causal-wrapped pretrained weights
                                EXPECTED TO SOUND SMEARED -- not yet fine-tuned for causal masking
  4. Peak-normalize each to the input's peak amplitude
  5. Write all three to ``audio_out/<input_name>__<variant>.wav``

Run:
    python demos/demo_audio.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import vocos

# Suppress the noisy HF/torch warnings on Windows
warnings.filterwarnings("ignore")

from vocos_rt.streaming_vocos import StreamingVocos  # noqa: E402

WORKSPACE = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT")
INPUT_DIR = WORKSPACE / "audio"
OUTPUT_DIR = WORKSPACE / "audio_out"
SAMPLE_RATE = 24_000


def load_mono_24k(path: Path) -> torch.Tensor:
    """Load WAV, mix to mono, resample to 24 kHz, return float32 (1, T)."""
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.to(torch.float32)


def peak_normalize_to(reference: torch.Tensor, target: torch.Tensor,
                      skip_samples: int = 0) -> torch.Tensor:
    """Scale ``target`` so its peak equals ``reference``'s peak.

    ``skip_samples`` excludes the leading region from the peak estimate -- used
    to avoid being dominated by streaming warmup transients that would crush
    the steady-state body to near-silence.
    """
    ref_peak = float(reference.abs().max())
    body = target[..., skip_samples:] if skip_samples > 0 else target
    tgt_peak = float(body.abs().max())
    if tgt_peak < 1e-9 or ref_peak < 1e-9:
        return target
    return target * (ref_peak / tgt_peak)


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (T, C) or (T,) for mono
    arr = audio.squeeze(0).detach().cpu().numpy()
    sf.write(str(path), arr, sample_rate, subtype="FLOAT")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[demo_audio] ERROR: input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 1

    # Per Jakob 2026-05-01: vocos-mel-24khz is a speech-trained model, and the
    # interesting test is on speech inputs. Drop noise / music / keyboard /
    # the original speech.wav stress sample, keep all the TTS speech samples
    # plus a short curated subset for variety.
    EXCLUDE = {"noise.wav", "music.wav", "keyboard.wav"}
    all_wavs = sorted(INPUT_DIR.glob("*.wav"))
    inputs = [p for p in all_wavs if p.name not in EXCLUDE]
    if not inputs:
        print(f"[demo_audio] ERROR: no eligible .wav files under {INPUT_DIR}", file=sys.stderr)
        return 1
    if len(all_wavs) > len(inputs):
        print(f"[demo_audio] Skipping non-speech: {sorted(EXCLUDE & {p.name for p in all_wavs})}")

    print(f"[demo_audio] Found {len(inputs)} input WAVs in {INPUT_DIR}")
    print(f"[demo_audio] Outputs will land in {OUTPUT_DIR}")

    # Load upstream model + wrap. CPU is fine for a 13s clip; CUDA would be marginal here.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo_audio] Using device: {device}")

    print("[demo_audio] Loading upstream charactr/vocos-mel-24khz ...")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")
    upstream.to(device)
    upstream.eval()

    print("[demo_audio] Wrapping for causal streaming ...")
    streaming = StreamingVocos(upstream)
    streaming.to(device)

    print(f"[demo_audio] Causal receptive field: {streaming.causal_receptive_field_frames} frames")
    print(f"[demo_audio] Algorithmic latency:    {streaming.algorithmic_latency_samples} samples = "
          f"{streaming.algorithmic_latency_ms:.2f} ms")
    print()

    for src in inputs:
        stem = src.stem
        print(f"[demo_audio] === {src.name} ===")

        # 1. Load
        audio = load_mono_24k(src).to(device)
        n_sec = audio.shape[-1] / SAMPLE_RATE
        print(f"  loaded: {audio.shape[-1]} samples = {n_sec:.2f} s @ {SAMPLE_RATE} Hz")

        # 2. Mel
        with torch.inference_mode():
            mel = upstream.feature_extractor(audio)  # (1, 100, T)
        print(f"  mel:    {tuple(mel.shape)}  range=[{float(mel.min()):.2f}, {float(mel.max()):.2f}]")

        # 3a. input (resampled reference)
        save_wav(OUTPUT_DIR / f"{stem}__01_input.wav", audio.cpu(), SAMPLE_RATE)

        # Skip the first 100 ms of any reconstructed signal when peak-normalizing,
        # so streaming warmup transients can't crush steady-state content to near-silence.
        warmup_skip = SAMPLE_RATE // 10  # 2400 samples = 100 ms

        # 3b. upstream non-causal offline -- gold reference for the vocoder
        t0 = time.time()
        with torch.inference_mode():
            audio_offline = upstream.decode(mel)
        t_offline = time.time() - t0
        rtf_offline = t_offline / n_sec
        audio_offline = peak_normalize_to(audio, audio_offline, skip_samples=warmup_skip)
        save_wav(OUTPUT_DIR / f"{stem}__02_offline_noncausal.wav", audio_offline.cpu(), SAMPLE_RATE)
        print(f"  offline non-causal upstream:  {t_offline*1000:.0f} ms  RTF={rtf_offline:.3f}  "
              f"peak={float(audio_offline.abs().max()):.3f}")

        # 3c. streaming causal (as-is weights)
        streaming.reset(batch_size=1)
        t0 = time.time()
        audio_streaming = streaming.stream(mel)
        t_streaming = time.time() - t0
        rtf_streaming = t_streaming / n_sec
        # Diagnostic: report raw peak BEFORE normalization to surface any spike issues
        raw_peak = float(audio_streaming.abs().max())
        body_peak = float(audio_streaming[..., warmup_skip:].abs().max())
        audio_streaming = peak_normalize_to(audio, audio_streaming, skip_samples=warmup_skip)
        save_wav(OUTPUT_DIR / f"{stem}__03_streaming_causal.wav", audio_streaming.cpu(), SAMPLE_RATE)
        print(f"  streaming causal vocos_rt:    {t_streaming*1000:.0f} ms  RTF={rtf_streaming:.3f}  "
              f"raw_peak={raw_peak:.3f}  body_peak={body_peak:.3f}")
        print()

    print(f"[demo_audio] DONE. {len(inputs)} inputs * 3 variants = {len(inputs)*3} files in {OUTPUT_DIR}")
    print()
    print("Listening guide:")
    print("  __01_input.wav            - the original (resampled to 24 kHz). Ground truth.")
    print("  __02_offline_noncausal.wav - upstream Vocos as published. Reference quality.")
    print("  __03_streaming_causal.wav  - vocos_rt streaming with as-is weights causally")
    print("                              wrapped. Expected to sound smeared/dull -- the")
    print("                              kernel weights were trained for bidirectional")
    print("                              context. Phase 2 will fine-tune to recover quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
