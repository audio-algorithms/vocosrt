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
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

# Per adversarial agent's advice: expandable CUDA allocator helps fragmentation
# OOMs that empty_cache() can only mask. Set BEFORE torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

from vocos_rt.distortion_metrics import hop_rate_envelope_flatness_loss  # noqa: E402
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
    # Loss weights -- different defaults for GAN vs reconstruction-only training.
    # When use_discriminators=True, follow upstream Vocos's recipe: mel=45 + adv + fm.
    # When False, use D17-era reconstruction-only fallback weights.
    mel_loss_weight: float = 20.0          # D19: GAN-agent rebalanced from 45 (mel-floor was pinning quality)
    stft_loss_weight: float = 0.0
    waveform_l2_weight: float = 0.0
    diff_match_weight: float = 0.0
    envelope_flatness_weight: float = 5.0  # D19: hop-rate envelope flatness loss (tube/comb attack)
    fm_weight: float = 2.0                 # D19: GAN-agent uplift from 1 (fm is the early-training ladder)
    mrd_loss_coeff: float = 1.0            # upstream default
    use_discriminators: bool = False       # see DECISIONS.md D18
    disc_warmup_steps: int = 5_000         # D19: GAN-agent recommendation; let mel converge before adv
    grad_checkpoint_generator: bool = True # save activation memory at the cost of recompute
    detach_fmap_real: bool = True          # D19: Memory-agent's biggest single win (~400 MB)
    drop_mpd_high_periods: bool = True     # D19: Memory-agent drop periods 7,11 (~150 MB)
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
    parser.add_argument("--disc-warmup-steps", type=int, default=5_000,
                        help="Train mel-only for N steps before enabling discriminators (avoids early instability)")
    parser.add_argument("--mel-loss-weight", type=float, default=20.0)
    parser.add_argument("--fm-weight", type=float, default=2.0)
    parser.add_argument("--stft-loss-weight", type=float, default=0.0)
    parser.add_argument("--waveform-l2-weight", type=float, default=0.0)
    parser.add_argument("--diff-match-weight", type=float, default=0.0)
    parser.add_argument("--mrd-loss-coeff", type=float, default=1.0)
    parser.add_argument("--envelope-flatness-weight", type=float, default=5.0,
                        help="D19: hop-rate envelope flatness loss weight (direct tube/comb attack)")
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
        disc_warmup_steps=args.disc_warmup_steps,
        mel_loss_weight=args.mel_loss_weight,
        stft_loss_weight=args.stft_loss_weight,
        waveform_l2_weight=args.waveform_l2_weight,
        diff_match_weight=args.diff_match_weight,
        envelope_flatness_weight=args.envelope_flatness_weight,
        fm_weight=args.fm_weight,
        mrd_loss_coeff=args.mrd_loss_coeff,
        checkpoint_every=args.checkpoint_every,
        validate_every=args.validate_every,
        sample_every=args.sample_every,
        log_every=args.log_every,
        num_workers=args.num_workers,
        seed=args.seed,
        wall_clock_budget_h=args.wall_clock_budget_h,
    )

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
    from vocos.loss import (
        DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss,
    )
    mel_loss_fn = MelSpecReconstructionLoss(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100,
    ).to(device)
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    # ---- discriminators (D18) ----
    mpd = mrd = disc_loss_fn = gen_adv_loss_fn = fm_loss_fn = None
    optimizer_disc = scheduler_disc = None
    if cfg.use_discriminators:
        from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
        log.info("constructing MPD + MRD discriminators ...")
        # D19: drop MPD periods 7,11 per memory-agent for ~150 MB activation savings
        mpd_periods = (2, 3, 5) if cfg.drop_mpd_high_periods else (2, 3, 5, 7, 11)
        log.info("MPD periods: %s", mpd_periods)
        mpd = MultiPeriodDiscriminator(periods=mpd_periods).to(device)
        mrd = MultiResolutionDiscriminator().to(device)
        # D19: checkpoint MPD sub-disc forwards per memory-agent (~200 MB savings)
        for sub in mpd.discriminators:
            orig_fn = sub.forward
            def make_ckpt(fn):
                def wrapped(x, cond_embedding_id=None):
                    if torch.is_grad_enabled() and x.requires_grad:
                        return torch.utils.checkpoint.checkpoint(fn, x, cond_embedding_id, use_reentrant=False)
                    return fn(x, cond_embedding_id)
                return wrapped
            sub.forward = make_ckpt(orig_fn)
        mpd.train()
        mrd.train()
        n_mpd = sum(p.numel() for p in mpd.parameters())
        n_mrd = sum(p.numel() for p in mrd.parameters())
        log.info("MPD params: %d (%.1f MB FP32); MRD params: %d (%.1f MB FP32); total disc: %.1f MB",
                 n_mpd, n_mpd * 4 / 1e6, n_mrd, n_mrd * 4 / 1e6, (n_mpd + n_mrd) * 4 / 1e6)
        disc_loss_fn = DiscriminatorLoss().to(device)
        gen_adv_loss_fn = GeneratorLoss().to(device)
        fm_loss_fn = FeatureMatchingLoss().to(device)
        optimizer_disc = torch.optim.AdamW(
            list(mpd.parameters()) + list(mrd.parameters()),
            lr=cfg.lr_initial, weight_decay=cfg.weight_decay, betas=(0.8, 0.99),
        )
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_disc, T_max=cfg.num_steps, eta_min=cfg.lr_final,
        )

    # ---- optimizer + scheduler (generator) ----
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
    last_log_loss: dict[str, float] = {"mel": 0.0, "total": 0.0}

    while step < cfg.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        # batch: (B, T) audio
        audio = batch.to(device, non_blocking=True)

        # Extract mel (frozen). torch.no_grad (not inference_mode) so the resulting
        # tensor is a normal autograd-eligible tensor without needing .clone().
        with torch.no_grad():
            mel = mel_extractor(audio)

        disc_active = cfg.use_discriminators and step >= cfg.disc_warmup_steps

        # Disc-input crop budget (D18 fix per agent): MPD on full segment is the
        # peak-memory hot spot. Crop to <=8192 samples for both real and fake.
        DISC_MAX_SAMPLES = 8192

        def _crop_for_disc(real: Tensor, fake: Tensor) -> tuple[Tensor, Tensor]:
            n_d = min(real.shape[-1], fake.shape[-1])
            if n_d <= DISC_MAX_SAMPLES:
                return real[..., :n_d], fake[..., :n_d]
            start = (n_d - DISC_MAX_SAMPLES) // 2  # center crop is deterministic
            stop = start + DISC_MAX_SAMPLES
            return real[..., start:stop], fake[..., start:stop]

        # ============= Discriminator step (skipped during disc warmup) =============
        if disc_active:
            assert mpd is not None and mrd is not None and disc_loss_fn is not None
            assert optimizer_disc is not None
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    audio_hat_detached = generator(mel).detach()
                audio_for_d, audio_hat_d = _crop_for_disc(audio, audio_hat_detached)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                real_mp, fake_mp, _, _ = mpd(audio_for_d, audio_hat_d)
                real_mrd, fake_mrd, _, _ = mrd(audio_for_d, audio_hat_d)
                loss_d_mp, _, _ = disc_loss_fn(real_mp, fake_mp)
                loss_d_mrd, _, _ = disc_loss_fn(real_mrd, fake_mrd)
                loss_d = (loss_d_mp / max(len(real_mp), 1) +
                          cfg.mrd_loss_coeff * loss_d_mrd / max(len(real_mrd), 1))
            optimizer_disc.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss_d).backward()
                scaler.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(mrd.parameters()), cfg.grad_clip)
                scaler.step(optimizer_disc)
            else:
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(mrd.parameters()), cfg.grad_clip)
                optimizer_disc.step()
            assert scheduler_disc is not None
            scheduler_disc.step()
            last_log_loss["d_mp"] = float(loss_d_mp.detach())
            last_log_loss["d_mrd"] = float(loss_d_mrd.detach())
            del audio_hat_detached, audio_hat_d, audio_for_d, real_mp, fake_mp, real_mrd, fake_mrd

        # ============= Generator step =============
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            audio_hat = generator(mel)
            n = min(audio.shape[-1], audio_hat.shape[-1])
            audio = audio[..., :n]
            audio_hat = audio_hat[..., :n]

            mel_loss = mel_loss_fn(audio_hat, audio)

            # Optional reconstruction terms (D17 fallbacks; default 0 with GAN)
            stft_loss = stft_loss_fn(audio, audio_hat) if cfg.stft_loss_weight > 0 else torch.zeros((), device=device)
            wav_l2 = (audio - audio_hat).pow(2).mean() if cfg.waveform_l2_weight > 0 else torch.zeros((), device=device)
            if cfg.diff_match_weight > 0:
                target_diff = audio[..., 1:] - audio[..., :-1]
                output_diff = audio_hat[..., 1:] - audio_hat[..., :-1]
                diff_match = (target_diff - output_diff).abs().mean()
            else:
                diff_match = torch.zeros((), device=device)

            # D19: hop-rate envelope flatness -- direct attack on the comb-filter "tube"
            # artifact. Penalizes 93.75 Hz periodic ripple in (audio_hat/audio_real)
            # log-envelope, which is the fingerprint of frame-correlated mag/phase
            # errors that mel/STFT losses cannot see.
            if cfg.envelope_flatness_weight > 0:
                env_flat = hop_rate_envelope_flatness_loss(
                    audio_hat, audio,
                    sample_rate=SAMPLE_RATE, hop_length=256,
                )
            else:
                env_flat = torch.zeros((), device=device)

            # Adversarial + feature matching (when disc active)
            if disc_active:
                assert mpd is not None and mrd is not None
                assert gen_adv_loss_fn is not None and fm_loss_fn is not None
                audio_d_g, audio_hat_d_g = _crop_for_disc(audio, audio_hat)
                # D19 (memory-agent biggest win): for FM loss, the real-side fmaps
                # are constants (target). Forward them under no_grad so the
                # autograd graph through MPD/MRD only retains the fake-side path.
                # Saves ~400 MB peak.
                if cfg.detach_fmap_real:
                    with torch.no_grad():
                        _, _, fmap_r_mp, _ = mpd(audio_d_g, audio_d_g)
                        _, _, fmap_r_mrd, _ = mrd(audio_d_g, audio_d_g)
                    _, fake_mp_g, _, fmap_g_mp = mpd(audio_d_g, audio_hat_d_g)
                    _, fake_mrd_g, _, fmap_g_mrd = mrd(audio_d_g, audio_hat_d_g)
                else:
                    _, fake_mp_g, fmap_r_mp, fmap_g_mp = mpd(audio_d_g, audio_hat_d_g)
                    _, fake_mrd_g, fmap_r_mrd, fmap_g_mrd = mrd(audio_d_g, audio_hat_d_g)
                loss_adv_mp, list_adv_mp = gen_adv_loss_fn(fake_mp_g)
                loss_adv_mrd, list_adv_mrd = gen_adv_loss_fn(fake_mrd_g)
                loss_adv_mp = loss_adv_mp / max(len(list_adv_mp), 1)
                loss_adv_mrd = loss_adv_mrd / max(len(list_adv_mrd), 1)
                loss_fm_mp = fm_loss_fn(fmap_r_mp, fmap_g_mp) / max(len(fmap_r_mp), 1)
                loss_fm_mrd = fm_loss_fn(fmap_r_mrd, fmap_g_mrd) / max(len(fmap_r_mrd), 1)
            else:
                loss_adv_mp = loss_adv_mrd = loss_fm_mp = loss_fm_mrd = torch.zeros((), device=device)

            total_loss = (
                cfg.mel_loss_weight * mel_loss
                + cfg.stft_loss_weight * stft_loss
                + cfg.waveform_l2_weight * wav_l2
                + cfg.diff_match_weight * diff_match
                + cfg.envelope_flatness_weight * env_flat
                + loss_adv_mp + cfg.mrd_loss_coeff * loss_adv_mrd
                + cfg.fm_weight * (loss_fm_mp + cfg.mrd_loss_coeff * loss_fm_mrd)
            )
            loss_for_backward = total_loss / cfg.grad_accum_steps

        if use_scaler:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()
        accum_count += 1

        # NB (D18 agent fix): empty_cache() removed. With expandable_segments:True
        # the allocator manages its own bookkeeping and empty_cache fights it.

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
            last_log_loss["envf"] = float(env_flat.detach())
            last_log_loss["total"] = float(total_loss.detach())
            if disc_active:
                last_log_loss["adv_mp"] = float(loss_adv_mp.detach())
                last_log_loss["adv_mrd"] = float(loss_adv_mrd.detach())
                last_log_loss["fm_mp"] = float(loss_fm_mp.detach())
                last_log_loss["fm_mrd"] = float(loss_fm_mrd.detach())

            if step % cfg.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                wall = time.time() - t_start
                steps_per_s = (step - start_step) / max(wall, 1e-3)
                eta_h = (cfg.num_steps - step) / max(steps_per_s, 1e-3) / 3600.0
                if disc_active:
                    log.info(
                        "step=%d/%d lr=%.2e mel=%.4f envf=%.5f advMP=%.3f advMR=%.3f fmMP=%.3f fmMR=%.3f dMP=%.3f dMR=%.3f total=%.2f sps=%.2f eta_h=%.1f",
                        step, cfg.num_steps, lr, last_log_loss["mel"], last_log_loss["envf"],
                        last_log_loss.get("adv_mp", 0), last_log_loss.get("adv_mrd", 0),
                        last_log_loss.get("fm_mp", 0), last_log_loss.get("fm_mrd", 0),
                        last_log_loss.get("d_mp", 0), last_log_loss.get("d_mrd", 0),
                        last_log_loss["total"], steps_per_s, eta_h,
                    )
                else:
                    log.info(
                        "step=%d/%d lr=%.2e mel=%.4f envf=%.5f total=%.4f (mel-warmup) sps=%.2f eta_h=%.1f",
                        step, cfg.num_steps, lr, last_log_loss["mel"], last_log_loss["envf"],
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
    log.info("DONE. wall-clock=%.2f h, final mel=%.4f total=%.4f",
             total_h, last_log_loss["mel"], last_log_loss["total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
