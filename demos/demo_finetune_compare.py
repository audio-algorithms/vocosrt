# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Phase 2.4 comparison demo: A/B speech samples with fine-tuned causal weights.

For each input speech WAV in the workspace ``audio/`` folder, write four variants
to ``audio_out_finetune/``:
  1. ``__01_input.wav``                  - resampled reference
  2. ``__02_offline_noncausal.wav``      - upstream Vocos (gold reference)
  3. ``__03_streaming_causal_pretrained.wav`` - vocos_rt streaming with as-is
                                              pretrained weights (Phase 1 baseline)
  4. ``__04_streaming_causal_finetuned.wav``  - vocos_rt streaming with the
                                              causal-fine-tuned weights from
                                              ``checkpoints/finetune/final.pt``

The interesting comparison is 03 vs 04: how much quality the fine-tune recovers
relative to the as-is causal masking.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import vocos

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vocos_rt.streaming_vocos import StreamingVocos  # noqa: E402

WORKSPACE = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT")
INPUT_DIR = WORKSPACE / "audio"
OUTPUT_DIR = WORKSPACE / "audio_out_finetune"
FINETUNE_CHECKPOINT = REPO_ROOT / "checkpoints" / "finetune" / "step_035000.pt"  # D17 best (training crashed before step 50k)
SAMPLE_RATE = 24_000
SKIP_NON_SPEECH = {"noise.wav", "music.wav", "keyboard.wav"}
WARMUP_SKIP = SAMPLE_RATE // 10  # 100 ms


def load_mono_24k(path: Path) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.to(torch.float32)


def peak_normalize_to(reference: torch.Tensor, target: torch.Tensor, skip_samples: int = 0) -> torch.Tensor:
    ref_peak = float(reference.abs().max())
    body = target[..., skip_samples:] if skip_samples > 0 else target
    tgt_peak = float(body.abs().max())
    if tgt_peak < 1e-9 or ref_peak < 1e-9:
        return target
    return target * (ref_peak / tgt_peak)


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.squeeze(0).detach().cpu().numpy(), sample_rate, subtype="FLOAT")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = [p for p in sorted(INPUT_DIR.glob("*.wav")) if p.name not in SKIP_NON_SPEECH]
    if not inputs:
        print(f"[demo_finetune_compare] ERROR: no eligible inputs in {INPUT_DIR}", file=sys.stderr)
        return 1

    if not FINETUNE_CHECKPOINT.exists():
        print(f"[demo_finetune_compare] ERROR: fine-tune checkpoint not found at {FINETUNE_CHECKPOINT}", file=sys.stderr)
        print("[demo_finetune_compare]   Run scripts/finetune_causal.py first to produce it.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo_finetune_compare] device={device}, {len(inputs)} input WAVs")

    # Two streaming wrappers: one with as-is pretrained weights, one with fine-tuned weights
    print("[demo_finetune_compare] loading upstream + pretrained streaming wrapper ...")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")
    upstream.to(device).eval()
    streaming_pretrained = StreamingVocos(upstream).to(device)

    print("[demo_finetune_compare] loading fine-tune checkpoint into a separate wrapper ...")
    upstream_for_ft = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")
    upstream_for_ft.to(device).eval()
    streaming_finetuned = StreamingVocos(upstream_for_ft).to(device)
    ckpt = torch.load(FINETUNE_CHECKPOINT, map_location=device, weights_only=False)
    # The checkpoint stores OfflineVocos.state_dict, which has the same key names
    # as StreamingVocos because both wrap the same modules.
    incompatible = streaming_finetuned.load_state_dict(ckpt["generator"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[demo_finetune_compare] WARNING: state_dict load mismatches:")
        if incompatible.missing_keys:
            print(f"  missing: {incompatible.missing_keys[:5]} (and {len(incompatible.missing_keys)} more)")
        if incompatible.unexpected_keys:
            print(f"  unexpected: {incompatible.unexpected_keys[:5]} (and {len(incompatible.unexpected_keys)} more)")
    print(f"[demo_finetune_compare] fine-tune checkpoint: step {ckpt.get('step', '?')}")

    for src in inputs:
        stem = src.stem
        print(f"[demo_finetune_compare] === {src.name} ===")

        audio = load_mono_24k(src).to(device)
        with torch.inference_mode():
            mel = upstream.feature_extractor(audio)
            audio_offline = upstream.decode(mel)

        # Variant 1: input
        save_wav(OUTPUT_DIR / f"{stem}__01_input.wav", audio.cpu(), SAMPLE_RATE)
        # Variant 2: upstream non-causal offline (gold)
        audio_offline = peak_normalize_to(audio, audio_offline, skip_samples=WARMUP_SKIP)
        save_wav(OUTPUT_DIR / f"{stem}__02_offline_noncausal.wav", audio_offline.cpu(), SAMPLE_RATE)

        # Variant 3: streaming with pretrained weights (Phase 1 baseline)
        streaming_pretrained.reset(batch_size=1)
        a_pretrained = streaming_pretrained.stream(mel)
        a_pretrained = peak_normalize_to(audio, a_pretrained, skip_samples=WARMUP_SKIP)
        save_wav(OUTPUT_DIR / f"{stem}__03_streaming_causal_pretrained.wav", a_pretrained.cpu(), SAMPLE_RATE)

        # Variant 4: streaming with fine-tuned weights (Phase 2 result)
        streaming_finetuned.reset(batch_size=1)
        a_finetuned = streaming_finetuned.stream(mel)
        a_finetuned = peak_normalize_to(audio, a_finetuned, skip_samples=WARMUP_SKIP)
        save_wav(OUTPUT_DIR / f"{stem}__04_streaming_causal_finetuned.wav", a_finetuned.cpu(), SAMPLE_RATE)

    print(f"[demo_finetune_compare] DONE. {len(inputs)} inputs * 4 variants = {len(inputs)*4} files in {OUTPUT_DIR}")
    print()
    print("Listening guide:")
    print("  __01 -> __02:  upstream Vocos's reconstruction quality on this input")
    print("  __02 -> __03:  cost of causal masking with as-is weights (Phase 1 baseline)")
    print("  __03 -> __04:  recovery from causal fine-tune (Phase 2 result)")
    print("  __01 -> __04:  the bottom line -- streaming causal fine-tuned vs ground truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
