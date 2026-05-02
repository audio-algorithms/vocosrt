# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Click forensics — answer the week-old question: where do D17 clicks live?

Per adversarial agent's #6 priority. If clicks are periodic at:
  - 256-sample (hop) intervals  -> training/inference mel-frame alignment issue
  - 1024-sample (n_fft) intervals -> STFT frame alignment issue
  - irregular -> generic model drift, not periodic artifact

We use the D17 step_035000 weights (best fine-tune we have) on speech.wav.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import vocos

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vocos_rt.streaming_vocos import StreamingVocos  # noqa: E402

D17_CKPT = REPO_ROOT / "checkpoints" / "finetune" / "step_035000.pt.d17"
SPEECH_WAV = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT\audio\speech.wav")
SAMPLE_RATE = 24_000
HOP = 256
N_FFT = 1024


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    streaming = StreamingVocos(upstream).to(device)

    if not D17_CKPT.exists():
        print(f"D17 checkpoint not found at {D17_CKPT}", file=sys.stderr)
        return 1
    ckpt = torch.load(D17_CKPT, map_location=device, weights_only=False)
    streaming.load_state_dict(ckpt["generator"])

    audio, sr = torchaudio.load(str(SPEECH_WAV))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio.to(device)
    with torch.inference_mode():
        mel = upstream.feature_extractor(audio)
    streaming.reset(batch_size=1)
    a = streaming.stream(mel).cpu().squeeze(0).numpy()  # (N,)

    diff = np.abs(np.diff(a))  # (N-1,)
    threshold = np.quantile(diff, 0.999)  # top 0.1%
    click_indices = np.where(diff > threshold)[0]
    n = len(click_indices)
    print(f"Audio length: {len(a)} samples = {len(a)/SAMPLE_RATE:.2f} s")
    print(f"Top-0.1% click threshold: |diff| > {threshold:.4f}")
    print(f"Click count: {n}")
    print()
    print("=== Click positions modulo candidate periods ===")
    print("If clicks are periodic at period P, mod-P should be concentrated.")
    print("If random, mod-P should be uniformly distributed (mean ~ P/2, std ~ P/sqrt(12)).")
    print()
    for period in [HOP, N_FFT, HOP*2, HOP*3, 100, 50]:
        mods = click_indices % period
        # Expected uniform mean = (period-1)/2, expected std = period * sqrt(1/12)
        expected_mean = (period - 1) / 2
        expected_std = period * np.sqrt(1/12)
        # Observed
        obs_mean = float(mods.mean())
        obs_std = float(mods.std())
        # Concentration check: count clicks in 10% window centered at each integer-multiple of period (i.e. near 0)
        threshold_zone = period // 10  # 10% of period
        zone_count = int(((mods < threshold_zone) | (mods > period - threshold_zone)).sum())
        zone_pct = 100 * zone_count / n
        random_pct = 100 * (2 * threshold_zone) / period  # what we'd expect at random
        marker = " <-- CONCENTRATED" if zone_pct > random_pct * 2 else ""
        print(f"  period={period:>5} (=mod_{period}): obs_mean={obs_mean:>6.1f} obs_std={obs_std:>6.1f} "
              f"(uniform: {expected_mean:>6.1f}/{expected_std:>5.1f}) "
              f"clicks_within_10%={zone_pct:>4.1f}% (random={random_pct:>4.1f}%){marker}")
    print()
    print("=== Inter-click distance histogram (top distances) ===")
    if n >= 2:
        diffs = np.diff(click_indices)
        print(f"Total inter-click distances: {len(diffs)}")
        # Most common distances
        from collections import Counter
        # Round to nearest multiple of 32 to find approximate periodicity
        rounded = (diffs // 16) * 16
        most_common = Counter(rounded.tolist()).most_common(10)
        print("Top 10 inter-click distances (rounded to multiples of 16):")
        for dist, count in most_common:
            div256 = dist / HOP
            div1024 = dist / N_FFT
            print(f"  distance {dist:>5} samples (hop x {div256:>5.2f}, n_fft x {div1024:>5.2f}): {count} occurrences")
    print()
    print("=== Bottom line ===")
    print("If a 'CONCENTRATED' marker appears for period 256 -> chunk-boundary artifact")
    print("If concentrated at 1024 -> STFT frame artifact")
    print("If no concentration -> model-drift artifact (not wrapper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
