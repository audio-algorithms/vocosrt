# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Build a small curated A/B listening kit in workspace/audio_to_listen/.

Picks 3 representative speech inputs (short / medium / long) and writes 4
variants per input with maximally-clear naming. Includes a README.txt with
listening instructions.
"""

from __future__ import annotations

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
from vocos_rt.streaming_vocos import StreamingVocos  # noqa: E402

WORKSPACE = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT")
INPUT_DIR = WORKSPACE / "audio"
OUT_DIR = WORKSPACE / "audio_to_listen_D18_step5k"
SAMPLE_RATE = 24_000
WARMUP_SKIP = SAMPLE_RATE // 10
D17_CKPT = REPO_ROOT / "checkpoints" / "finetune" / "step_005000.pt"  # D18 step 5k (best available)

# 3 picks: long (13s), medium (~5s, female), short (~1s, male)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    streaming_pre = StreamingVocos(upstream).to(device)

    # Separate wrapper for D17
    upstream_ft = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    streaming_ft = StreamingVocos(upstream_ft).to(device)
    ckpt = torch.load(D17_CKPT, map_location=device, weights_only=False)
    streaming_ft.load_state_dict(ckpt["generator"])

    for prefix, filename in PICKS:
        src = INPUT_DIR / filename
        if not src.exists():
            print(f"  SKIP missing: {src}", file=sys.stderr)
            continue
        audio = load_mono_24k(src).to(device)
        with torch.inference_mode():
            mel = upstream.feature_extractor(audio)
            audio_offline = upstream.decode(mel)

        # A. input
        save_wav(OUT_DIR / f"{prefix}__A_input.wav", audio.cpu())
        # B. upstream gold reference
        ao = peak_normalize_to(audio, audio_offline, skip=WARMUP_SKIP)
        save_wav(OUT_DIR / f"{prefix}__B_upstream_gold_offline.wav", ao.cpu())
        # C. Phase 1 pretrained streaming (smeared but clean)
        streaming_pre.reset(batch_size=1)
        ap = streaming_pre.stream(mel)
        ap = peak_normalize_to(audio, ap, skip=WARMUP_SKIP)
        save_wav(OUT_DIR / f"{prefix}__C_streaming_pretrained_SMEARED_BUT_CLEAN.wav", ap.cpu())
        # D. D17 fine-tuned streaming (sharper but possibly some clicks)
        streaming_ft.reset(batch_size=1)
        af = streaming_ft.stream(mel)
        af = peak_normalize_to(audio, af, skip=WARMUP_SKIP)
        save_wav(OUT_DIR / f"{prefix}__D_streaming_D18_step5k_GAN.wav", af.cpu())
        print(f"  done: {prefix}")

    readme = OUT_DIR / "README.txt"
    readme.write_text("""LISTENING KIT -- choose the deliverable

You have 3 numbered groups (01, 02, 03), each with 4 files A/B/C/D.

Recommended listen order PER GROUP:
  A  the input (resampled ground truth)
  B  upstream Vocos non-causal -- gold reference (best quality possible from
     this vocoder family, but NOT real-time; for context only)
  C  Phase 1 streaming pretrained -- causal masking on as-is weights.
     SMEARED/DULL but CLEAN. You previously called this acceptable.
  D  D17 streaming fine-tuned -- the new fine-tune attempt. SHARPER than C,
     but click metrics suggest some isolated single clicks may still be
     audible. Listen for: does it sound substantially better than C overall?
     If yes -> D is the new deliverable.
     If no  -> C is the deliverable; document that we cannot beat it without
               a discriminator-capable GPU.

The decision: C or D.
""", encoding="utf-8")
    print(f"\nDONE. Listen kit at: {OUT_DIR}")
    print("Per group: A -> B -> C -> D, then decide C vs D.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
