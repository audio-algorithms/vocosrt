# Copyright (c) 2026 AudioAlgorithms LLC. All rights reserved.
"""Phase 2 -- causal fine-tune of vocos-mel-24khz on LibriTTS train-clean-100.

Per DECISIONS.md D15: this fine-tune uses RECONSTRUCTION LOSSES ONLY (mel L1 +
multi-resolution STFT magnitude L1). Discriminators are not loaded by default
because they don't fit on the target 4 GB RTX 3050; pass --use-discriminators
to opt in if a larger GPU is available.

The "training" model is OfflineVocos (full-sequence causal forward); per-frame
streaming would be ~100x too slow for SGD. Test 01 (committed in Phase 1.C)
guarantees that StreamingVocos.stream(mel) == OfflineVocos.forward(mel) to FP32
tolerance, so the resulting weights deploy correctly via the streaming API.

Resumability:
- Checkpoints written every --checkpoint-every steps to checkpoints/finetune/
- On startup, resumes from the latest checkpoint if present
- A .training_complete marker is written when num_steps is reached;
  re-running a completed config exits cleanly with no work done

Sample generation:
- Every --sample-every steps, runs the current generator on a fixed mel
  extracted from speech.wav and writes a WAV to checkpoints/finetune/samples/
- Useful for listening to fine-tune progress over time

Usage:
    python scripts/finetune_causal.py --num-steps 50000 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import vocos
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# Suppress upstream warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vocos_rt.forensic_log import log  # noqa: E402
from vocos_rt.offline_vocos import OfflineVocos  # noqa: E402

LIBRITTS_TRAIN = REPO_ROOT / "datasets" / "libritts" / "LibriTTS" / "train-clean-100"
LIBRITTS_DEV = REPO_ROOT / "datasets" / "libritts" / "LibriTTS" / "dev-clean"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "finetune"
SAMPLE_DIR = CHECKPOINT_DIR / "samples"
SPEECH_REF = Path(r"C:\Users\jakob\Desktop\google drive\VOCOSRT\audio\speech.wav")

SAMPLE_RATE = 24_000


# ---------------------------------------------------------------- data


class LibriTTSCropDataset(Dataset[Tensor]):
    """Random-crop fixed-length 24 kHz audio segments from LibriTTS .wav files."""

    def __init__(self, root: Path, segment_samples: int = 32_768, seed: int = 0):
        self.root = root
        self.segment_samples = segment_samples
        self.files = sorted(root.rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav files under {root}")
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tensor:
        path = self.files[idx]
        info = sf.info(str(path))
        n_total = info.frames
        if n_total >= self.segment_samples:
            start = self.rng.randint(0, n_total - self.segment_samples)
            audio, sr = sf.read(str(path), start=start, frames=self.segment_samples, dtype="float32")
        else:
            audio, sr = sf.read(str(path), dtype="float32")
            # Pad with zeros to reach segment length
            pad = self.segment_samples - audio.shape[0]
            import numpy as np
            audio = np.pad(audio, (0, pad))
        if sr != SAMPLE_RATE:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
            return audio_t.squeeze(0)
        return torch.from_numpy(audio)


# ---------------------------------------------------------------- losses


class MultiResolutionSTFTLoss(nn.Module):
    """Sum of L1 distances on log-magnitude STFTs at multiple FFT resolutions.

    Standard vocoder loss (Yamamoto et al., Parallel WaveGAN). Captures
    spectral structure at coarse and fine time-frequency tradeoffs, which
    pure mel L1 does not.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (2048, 1024, 512),
        hop_sizes: tuple[int, ...] = (512, 256, 128),
        win_sizes: tuple[int, ...] = (2048, 1024, 512),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.windows = nn.ParameterList(
            [nn.Parameter(torch.hann_window(w), requires_grad=False) for w in win_sizes]
        )

    def _spec_l1(self, y: Tensor, y_hat: Tensor, n_fft: int, hop: int, win: int, window: Tensor) -> Tensor:
        # (B, T) -> (B, n_fft//2+1, T') complex
        S_y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                         center=True, return_complex=True)
        S_h = torch.stft(y_hat, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                         center=True, return_complex=True)
        # Log-magnitude L1 + linear-magnitude L1 (Parallel WaveGAN style)
        log_l1 = (torch.log(S_y.abs().clamp(min=1e-7)) - torch.log(S_h.abs().clamp(min=1e-7))).abs().mean()
        mag_l1 = (S_y.abs() - S_h.abs()).abs().mean()
        return log_l1 + mag_l1

    def forward(self, y: Tensor, y_hat: Tensor) -> Tensor:
        loss = torch.zeros((), device=y.device, dtype=y.dtype)
        for n_fft, hop, win, window in zip(self.fft_sizes, self.hop_sizes, self.win_sizes, self.windows):
            loss = loss + self._spec_l1(y, y_hat, n_fft, hop, win, window)
        return loss / len(self.fft_sizes)


# ---------------------------------------------------------------- training


@dataclass
class TrainConfig:
    num_steps: int = 50_000
    batch_size: int = 2
    grad_accum_steps: int = 4              # effective batch = batch_size * grad_accum_steps
    segment_samples: int = 32_768          # ~1.37 s at 24 kHz
    lr_initial: float = 1e-4
    lr_final: float = 1e-5
    weight_decay: float = 0.01
    mel_loss_weight: float = 15.0
    stft_loss_weight: float = 2.5
    # Click-targeted losses (D17): wav_l2 penalizes outlier samples quadratically;
    # diff_match aligns the model's first-derivative envelope to the target's,
    # directly preventing sharper-than-natural transients (clicks).
    waveform_l2_weight: float = 1000.0     # MSE; smoke test shows raw value ~0.005 -> contribution ~5 (mel contribution ~6)
    diff_match_weight: float = 500.0       # first-difference L1 -- raw ~0.005 -> contribution ~2.5
    use_discriminators: bool = False       # DECISIONS.md D15
    precision: str = "bf16"                # "fp32" / "fp16" / "bf16"
    checkpoint_every: int = 2_500
    validate_every: int = 2_500
    sample_every: int = 2_500
    log_every: int = 50
    num_workers: int = 4
    seed: int = 42
    wall_clock_budget_h: float = 60.0      # halve num_steps if projected to exceed
    grad_clip: float = 1.0


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    files = sorted(checkpoint_dir.glob("step_*.pt"))
    return files[-1] if files else None


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_initial, weight_decay=cfg.weight_decay, betas=(0.8, 0.99),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_steps, eta_min=cfg.lr_final,
    )


def precision_to_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def load_reference_mel(upstream: vocos.Vocos, device: torch.device) -> Tensor | None:
    if not SPEECH_REF.exists():
        return None
    audio, sr = torchaudio.load(str(SPEECH_REF))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio.to(device)
    with torch.inference_mode():
        return upstream.feature_extractor(audio)


@torch.inference_mode()
def write_progress_sample(generator: OfflineVocos, ref_mel: Tensor, step: int) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    audio = generator.forward(ref_mel)
    out_path = SAMPLE_DIR / f"step_{step:06d}_speech.wav"
    sf.write(str(out_path), audio.squeeze(0).cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--segment-samples", type=int, default=32_768)
    parser.add_argument("--lr-initial", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=2_500)
    parser.add_argument("--validate-every", type=int, default=2_500)
    parser.add_argument("--sample-every", type=int, default=2_500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--use-discriminators", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wall-clock-budget-h", type=float, default=60.0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only 100 steps; do not write 'training_complete' marker.")
    args = parser.parse_args()

    cfg = TrainConfig(
        num_steps=100 if args.smoke_test else args.num_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        segment_samples=args.segment_samples,
        lr_initial=args.lr_initial,
        lr_final=args.lr_final,
        precision=args.precision,
        use_discriminators=args.use_discriminators,
        checkpoint_every=args.checkpoint_every,
        validate_every=args.validate_every,
        sample_every=args.sample_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
        seed=args.seed,
        wall_clock_budget_h=args.wall_clock_budget_h,
    )
    if cfg.use_discriminators:
        log.error("--use-discriminators is not yet implemented; ignoring (per DECISIONS.md D15)")
        cfg.use_discriminators = False

    log.info("vocos_rt fine-tune starting; config=%s", json.dumps(asdict(cfg), indent=2))

    # ---- prerequisite checks ----
    if not LIBRITTS_TRAIN.exists():
        log.error("LibriTTS train-clean-100 not found at %s", LIBRITTS_TRAIN)
        log.error("Run: python scripts/download_corpora.py --subsets train-clean-100 dev-clean")
        return 1
    if not LIBRITTS_DEV.exists():
        log.warning("LibriTTS dev-clean not found at %s -- validation will be skipped", LIBRITTS_DEV)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotency: skip if already complete
    completion_marker = CHECKPOINT_DIR / ".training_complete"
    if completion_marker.exists() and not args.smoke_test:
        log.info("Training already complete (%s exists). Nothing to do.", completion_marker)
        return 0

    # ---- seeding ----
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)
    if device.type == "cuda":
        log.info("gpu=%s mem=%.1fGB", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))

    # ---- model ----
    log.info("loading upstream model and wrapping in OfflineVocos ...")
    upstream = vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")
    upstream.to(device)
    upstream.eval()  # only matters if there were BN/Dropout (there aren't)
    generator = OfflineVocos(upstream).to(device)
    generator.train()
    n_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    log.info("generator trainable params: %d (%.1f MB FP32)", n_params, n_params * 4 / 1e6)

    # Mel feature extractor (frozen) -- we use upstream's directly
    mel_extractor = upstream.feature_extractor
    for p in mel_extractor.parameters():
        p.requires_grad_(False)

    # ---- losses ----
    from vocos.loss import MelSpecReconstructionLoss
    mel_loss_fn = MelSpecReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    # ---- optimizer + scheduler ----
    optimizer = make_optimizer(generator, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    # ---- AMP ----
    amp_dtype = precision_to_dtype(cfg.precision)
    use_amp = cfg.precision != "fp32"
    # bf16 doesn't need GradScaler; fp16 does
    use_scaler = cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ---- data ----
    log.info("scanning train-clean-100 ...")
    train_set = LibriTTSCropDataset(LIBRITTS_TRAIN, segment_samples=cfg.segment_samples, seed=cfg.seed)
    log.info("train_set has %d files", len(train_set))
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda", drop_last=True,
    )

    # ---- resume ----
    start_step = 0
    latest = latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        log.info("resuming from %s", latest.name)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if use_scaler:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        log.info("resumed at step %d", start_step)

    ref_mel = load_reference_mel(upstream, device)

    # ---- training loop ----
    log.info("starting training loop at step %d / %d", start_step, cfg.num_steps)
    step = start_step
    t_start = time.time()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    last_log_loss = {"mel": 0.0, "stft": 0.0, "wav_l2": 0.0, "diff": 0.0, "total": 0.0}

    while step < cfg.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        # batch: (B, T) audio
        audio = batch.to(device, non_blocking=True)

        # Extract mel (frozen)
        with torch.inference_mode():
            mel = mel_extractor(audio)
        mel = mel.detach().clone()  # detach from inference_mode so backward works

        # Forward generator
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            audio_hat = generator(mel)
            # Match length (offline forward returns T*hop samples; trim audio to match)
            n = min(audio.shape[-1], audio_hat.shape[-1])
            audio = audio[..., :n]
            audio_hat = audio_hat[..., :n]

            mel_loss = mel_loss_fn(audio_hat, audio)
            stft_loss = stft_loss_fn(audio, audio_hat)
            # Waveform L2 -- penalize large per-sample errors quadratically
            wav_l2 = (audio - audio_hat).pow(2).mean()
            # First-difference matching -- model's d/dt should match target's d/dt;
            # mismatch here is exactly what produces audible clicks
            target_diff = audio[..., 1:] - audio[..., :-1]
            output_diff = audio_hat[..., 1:] - audio_hat[..., :-1]
            diff_match = (target_diff - output_diff).abs().mean()
            total_loss = (
                cfg.mel_loss_weight * mel_loss
                + cfg.stft_loss_weight * stft_loss
                + cfg.waveform_l2_weight * wav_l2
                + cfg.diff_match_weight * diff_match
            )
            loss_for_backward = total_loss / cfg.grad_accum_steps

        if use_scaler:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        accum_count += 1

        if accum_count >= cfg.grad_accum_steps:
            # Gradient clip
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            step += 1

            last_log_loss["mel"] = float(mel_loss.detach())
            last_log_loss["stft"] = float(stft_loss.detach())
            last_log_loss["wav_l2"] = float(wav_l2.detach())
            last_log_loss["diff"] = float(diff_match.detach())
            last_log_loss["total"] = float(total_loss.detach())

            if step % cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                wall = time.time() - t_start
                steps_per_s = (step - start_step) / max(wall, 1e-3)
                eta_h = (cfg.num_steps - step) / max(steps_per_s, 1e-3) / 3600.0
                log.info(
                    "step=%d/%d lr=%.2e mel=%.4f stft=%.4f wav2=%.5f diff=%.5f total=%.4f sps=%.2f eta_h=%.1f",
                    step, cfg.num_steps, lr, last_log_loss["mel"], last_log_loss["stft"],
                    last_log_loss["wav_l2"], last_log_loss["diff"],
                    last_log_loss["total"], steps_per_s, eta_h,
                )

            # Wall-clock budget check at regular intervals (after warmup so sps is stable)
            if step == 200:
                wall_s = time.time() - t_start
                projected_h = (cfg.num_steps / max(step - start_step, 1)) * wall_s / 3600.0
                if projected_h > cfg.wall_clock_budget_h:
                    log.warning(
                        "projected wall-clock %.1f h exceeds budget %.1f h; halving num_steps "
                        "from %d to %d (DECISIONS.md will note this)",
                        projected_h, cfg.wall_clock_budget_h, cfg.num_steps, cfg.num_steps // 2,
                    )
                    cfg.num_steps = cfg.num_steps // 2
                    # Re-set scheduler T_max
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=cfg.num_steps, eta_min=cfg.lr_final, last_epoch=step,
                    )

            if step % cfg.sample_every == 0 and ref_mel is not None and not args.smoke_test:
                log.info("writing progress sample for step %d", step)
                generator.eval()
                write_progress_sample(generator, ref_mel, step)
                generator.train()

            if step % cfg.checkpoint_every == 0 and not args.smoke_test:
                ckpt_path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
                log.info("saving checkpoint %s", ckpt_path.name)
                ckpt = {
                    "step": step,
                    "generator": generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": asdict(cfg),
                }
                if use_scaler:
                    ckpt["scaler"] = scaler.state_dict()
                torch.save(ckpt, ckpt_path)

    # ---- finalize ----
    final_path = CHECKPOINT_DIR / "final.pt"
    log.info("training complete; saving final checkpoint %s", final_path.name)
    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "config": asdict(cfg),
    }, final_path)

    if not args.smoke_test:
        completion_marker.write_text(f"completed at step {step}\n")

    total_h = (time.time() - t_start) / 3600.0
    log.info("DONE. wall-clock=%.2f h, final losses: mel=%.4f stft=%.4f wav_l2=%.5f diff=%.5f",
             total_h, last_log_loss["mel"], last_log_loss["stft"],
             last_log_loss["wav_l2"], last_log_loss["diff"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
