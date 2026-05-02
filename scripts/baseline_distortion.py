# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Print distortion metrics for upstream / pretrained / D15 / D16 / D17 across the
curated speech inputs. Used to calibrate test thresholds and to compare runs.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
import torchaudio
import vocos

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vocos_rt.distortion_metrics import all_metrics  # noqa: E402
from vocos_rt.streaming_vocos import StreamingVocos  # noqa: E402

INPUT_DIR = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT\audio")
SAMPLE_RATE = 24_000
WARMUP_SKIP = SAMPLE_RATE // 10
INPUTS = [
    "speech.wav",
    "s05942-callirrhoe-female-en-us.wav",
    "s100680-algieba-male-en-us.wav",
    "s99387-callirrhoe-female-en-us.wav",
]
MODELS = [
    ("upstream_gold",        None),
    ("streaming_pretrained", None),  # special: pretrained streaming wrapper, no checkpoint
    ("D15_step_50000",       REPO_ROOT / "checkpoints" / "finetune" / "step_050000.pt.d15"),
    ("D16_step_50000",       REPO_ROOT / "checkpoints" / "finetune" / "step_050000.pt.d16"),
    ("D17_step_35000",       REPO_ROOT / "checkpoints" / "finetune" / "step_035000.pt"),
]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    print(f"{'input':40} {'model':25} {'max_jump':>9} {'p9999':>7} {'jps>0.20':>9} {'pops':>6} {'log_mel':>8} {'crest':>6}")
    print("-" * 130)

    for name in INPUTS:
        path = INPUT_DIR / name
        if not path.exists():
            continue
        audio, sr = torchaudio.load(str(path))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.to(device)
        with torch.inference_mode():
            mel = upstream.feature_extractor(audio)
            gold = upstream.decode(mel).squeeze(0)
        gold_body = gold[WARMUP_SKIP:]

        for model_name, ckpt_path in MODELS:
            if model_name == "upstream_gold":
                a = gold
            else:
                fresh = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
                s = StreamingVocos(fresh).to(device)
                if ckpt_path is not None and ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    s.load_state_dict(ckpt["generator"])
                s.reset(batch_size=1)
                a = s.stream(mel).squeeze(0)
            body = a[WARMUP_SKIP:]
            m = all_metrics(body, reference=gold_body, sample_rate=SAMPLE_RATE)
            print(f"{name:40} {model_name:25} {m['max_jump']:>9.4f} {m['p9999_jump']:>7.4f} "
                  f"{m['jumps_per_sec_0p20']:>9.1f} {m['pop_count_6sigma']:>6.0f} "
                  f"{m.get('log_mel_l1_vs_ref', 0):>8.3f} {m['crest_factor']:>6.1f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
