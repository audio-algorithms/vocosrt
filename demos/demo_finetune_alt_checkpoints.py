# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Generate audio from alternative checkpoints for A/B comparison.

Phase 2.4 revealed audible clicks in the step_050000 streaming output. The clicks
are NOT a streaming wrapper bug (streaming == offline to 5e-6) but a property of
the fine-tuned weights themselves -- without a GAN discriminator, the model
learned to satisfy mel L1 + STFT magnitude L1 with larger time-domain transients
than the pretrained baseline.

This script writes streaming audio at three checkpoints (5k / 35k / 50k) for the
user to A/B and pick the best, or to confirm that all checkpoints are clicky and
a fix is needed.

Output: workspace/audio_out_finetune_alt/<speech_stem>__<step>.wav
"""

from __future__ import annotations

import warnings
import sys
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
OUTPUT_DIR = WORKSPACE / "audio_out_finetune_alt"
SAMPLE_RATE = 24_000
WARMUP_SKIP = SAMPLE_RATE // 10
SKIP_NON_SPEECH = {"noise.wav", "music.wav", "keyboard.wav"}

# A few representative inputs for fast comparison; user can run with all if useful.
SPEECH_PICKS = [
    "speech.wav",
    "s05942-callirrhoe-female-en-us.wav",
    "s100680-algieba-male-en-us.wav",
    "s99387-callirrhoe-female-en-us.wav",
]

CHECKPOINTS_TO_TRY = [
    ("pretrained", None),
    ("step_005000", REPO_ROOT / "checkpoints" / "finetune" / "step_005000.pt"),
    ("step_035000", REPO_ROOT / "checkpoints" / "finetune" / "step_035000.pt"),
    ("step_050000", REPO_ROOT / "checkpoints" / "finetune" / "step_050000.pt"),
]


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
    inputs = [INPUT_DIR / name for name in SPEECH_PICKS if (INPUT_DIR / name).exists()]
    if not inputs:
        print("[demo_finetune_alt_checkpoints] No selected inputs exist", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    streaming = StreamingVocos(upstream).to(device)

    for src in inputs:
        stem = src.stem
        audio = load_mono_24k(src).to(device)
        with torch.inference_mode():
            mel = upstream.feature_extractor(audio)
        save_wav(OUTPUT_DIR / f"{stem}__00_input.wav", audio.cpu(), SAMPLE_RATE)
        for label, ckpt_path in CHECKPOINTS_TO_TRY:
            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                streaming.load_state_dict(ckpt["generator"])
            else:
                # Reset to pretrained: reload from upstream
                streaming.load_state_dict(StreamingVocos(vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()).state_dict())
            streaming.reset(batch_size=1)
            a = streaming.stream(mel)
            a = peak_normalize_to(audio, a, skip_samples=WARMUP_SKIP)
            save_wav(OUTPUT_DIR / f"{stem}__{label}.wav", a.cpu(), SAMPLE_RATE)
        print(f"  done: {stem}")

    print(f"\nDONE. {len(inputs) * (1 + len(CHECKPOINTS_TO_TRY))} files in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
