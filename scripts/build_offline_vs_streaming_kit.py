# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Generate streaming vs offline audio from the SAME D17 weights for A/B.

If clicks exist in BOTH streaming and offline output, they are model-side
(in the fine-tuned weights) -- the streaming wrapper is innocent.
If clicks exist only in streaming, the wrapper has a defect that test 01
(streaming==offline to 5e-6) somehow missed.

Output: workspace/audio_offline_vs_streaming/<input>__<variant>.wav
Variants per input:
  A_input               : ground truth (resampled)
  B_streaming_pretrained: clean baseline (Phase 1, no fine-tune)
  C_offline_D17_FT      : offline forward of D17 fine-tune weights
  D_streaming_D17_FT    : streaming forward of D17 fine-tune weights
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import shutil
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

from vocos_rt.offline_vocos import OfflineVocos  # noqa: E402
from vocos_rt.streaming_vocos import StreamingVocos  # noqa: E402

WORKSPACE = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT")
INPUT_DIR = WORKSPACE / "audio"
OUT_DIR = WORKSPACE / "audio_offline_vs_streaming"
SAMPLE_RATE = 24_000
WARMUP_SKIP = SAMPLE_RATE // 10
D17_CKPT = REPO_ROOT / "checkpoints" / "finetune" / "step_035000.pt.d17"
PICKS = [
    ("01_long_speech",  "speech.wav"),
    ("02_med_female",   "s99387-callirrhoe-female-en-us.wav"),
    ("03_short_male",   "s47652-algieba-male-en-us.wav"),
]


def load_mono_24k(path: Path) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.to(torch.float32)


def peak_normalize_to(reference: torch.Tensor, target: torch.Tensor, skip: int = 0) -> torch.Tensor:
    ref_peak = float(reference.abs().max())
    body = target[..., skip:] if skip > 0 else target
    tgt_peak = float(body.abs().max())
    if tgt_peak < 1e-9 or ref_peak < 1e-9:
        return target
    return target * (ref_peak / tgt_peak)


def save_wav(path: Path, audio: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.squeeze(0).detach().cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")


def main() -> int:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    device = torch.device("cpu")  # CPU to avoid contending with the running training
    print(f"[offline_vs_streaming] using device={device}")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    streaming_pre = StreamingVocos(upstream).to(device).eval()

    fresh_for_str = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    streaming_ft = StreamingVocos(fresh_for_str).to(device).eval()
    fresh_for_off = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    offline_ft = OfflineVocos(fresh_for_off).to(device).eval()

    ckpt = torch.load(D17_CKPT, map_location=device, weights_only=False)
    streaming_ft.load_state_dict(ckpt["generator"])
    offline_ft.load_state_dict(ckpt["generator"])

    for prefix, filename in PICKS:
        src = INPUT_DIR / filename
        if not src.exists():
            print(f"  SKIP missing: {src}", file=sys.stderr)
            continue
        audio = load_mono_24k(src).to(device)
        with torch.inference_mode():
            mel = upstream.feature_extractor(audio)

        # A. input
        save_wav(OUT_DIR / f"{prefix}__A_input.wav", audio)

        # B. streaming pretrained (Phase 1 baseline, clean per Jakob)
        streaming_pre.reset(batch_size=1)
        b = streaming_pre.stream(mel)
        b = peak_normalize_to(audio, b, skip=WARMUP_SKIP)
        save_wav(OUT_DIR / f"{prefix}__B_streaming_pretrained_CLEAN.wav", b)

        # C. OFFLINE D17 fine-tune (full-sequence forward, NO streaming)
        with torch.inference_mode():
            c = offline_ft.forward(mel)
        c = peak_normalize_to(audio, c, skip=WARMUP_SKIP)
        save_wav(OUT_DIR / f"{prefix}__C_OFFLINE_D17_finetune.wav", c)

        # D. STREAMING D17 fine-tune (per-frame, what user heard before)
        streaming_ft.reset(batch_size=1)
        d = streaming_ft.stream(mel)
        d = peak_normalize_to(audio, d, skip=WARMUP_SKIP)
        save_wav(OUT_DIR / f"{prefix}__D_STREAMING_D17_finetune.wav", d)
        print(f"  done: {prefix}")

    readme = OUT_DIR / "README.txt"
    readme.write_text("""LISTENING KIT -- is the click model-side or wrapper-side?

Per group (01, 02, 03), 4 files A/B/C/D.

A  input                                  -- ground truth
B  streaming_pretrained                   -- Phase 1 baseline you confirmed CLEAN
C  OFFLINE D17 fine-tune                  -- full-sequence forward (no streaming)
D  STREAMING D17 fine-tune                -- per-frame streaming (you heard clicks)

Both C and D use the same D17 fine-tuned weights.
The training engineer's claim (test 01 confirms streaming == offline to 5e-6):
  if C also has clicks -> the FINE-TUNED MODEL produces clicks regardless
                          of streaming/offline; the wrapper is innocent;
                          discriminator training (now running) targets this
  if C is CLEAN but D has clicks -> there's a wrapper bug test 01 missed,
                                    and the wrapper itself needs fixing first

Listen to C vs D for any of the 3 inputs and tell the engineer which case
you observe.
""", encoding="utf-8")
    print(f"\n[offline_vs_streaming] DONE. Listen at {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
