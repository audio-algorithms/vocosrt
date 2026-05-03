# Remote H100 / A100 training for vocos_rt

The 4 GB RTX 3050 Laptop GPU cannot complete GAN fine-tuning at sufficient
batch size for stable convergence (see DECISIONS.md D15-D19). This guide gets
the same code running on a rented H100 or A100 in ~45 minutes total wall-clock.

## Cost estimate

| Provider | H100 spot | A100 40GB spot | Setup ease |
|---|---|---|---|
| **Vast.ai** | $0.80-1.50/hr | $0.40-0.80/hr | medium (community marketplace) |
| **RunPod** | $2-3/hr | $1.50-2/hr | easy (pre-built PyTorch templates) |
| **Lambda Labs** | $2.50/hr | $1.10/hr | easiest (managed) |

Total expected cost for a complete fine-tune + listening kit: **$1-3**.

## Step-by-step

### 1. Push this repo to GitHub (one-time, ~2 min)

```bash
# On your local machine in C:\dev\vocos_rt
gh auth login                                # if not already authenticated
gh repo create vocos_rt --private --source=. --push
# Note the URL: e.g. https://github.com/<your_user>/vocos_rt.git
```

If you don't want to use GitHub: skip this and use scp/rsync at step 3 instead
(`rsync -avz --exclude='.venv' --exclude='datasets' --exclude='checkpoints' . user@remote:~/vocos_rt`).

### 2. Rent the GPU

**RunPod (recommended for first time)**:
1. Sign up at https://runpod.io
2. Browse → select "RTX A6000" (48GB, ~$0.80/hr) or A100 80GB (~$2/hr) or H100 80GB (~$3/hr)
3. Choose template: "RunPod Pytorch 2.1" or any image with Python 3.10+, CUDA 12.x
4. Set disk: 50 GB minimum (LibriTTS is 8 GB extracted)
5. Deploy → connect via Web Terminal or SSH

**Vast.ai (cheapest)**:
1. Sign up at https://vast.ai
2. Search → filter for `H100` or `A100`, sort by `$/hr` ascending
3. Pick a spot instance with PyTorch image
4. SSH instructions appear in the dashboard

### 3. On the remote machine

```bash
# Clone or copy the repo
git clone https://github.com/<your_user>/vocos_rt.git
cd vocos_rt

# One-command bootstrap: installs deps, downloads model + corpus
bash scripts/setup_remote.sh

# Train (auto-detects VRAM, picks batch size accordingly)
bash scripts/train_remote.sh
```

The training script auto-sizes:
- H100 80GB → batch 32, segment 32768, no grad accum
- A100 40GB → batch 16, segment 32768
- A6000 48GB → batch 16
- RTX 4090 24GB → batch 8 with grad_accum 2 (effective 16)

100k steps at batch 16+ takes:
- H100: ~25 min
- A100 40GB: ~45 min
- A6000 48GB: ~50 min
- RTX 4090 24GB: ~80 min

### 4. Download the trained weights back

```bash
# On your local machine
scp -r user@remote:~/vocos_rt/checkpoints/finetune ./remote_checkpoints

# Or via rsync (handles partial transfers)
rsync -avz user@remote:~/vocos_rt/checkpoints/finetune ./remote_checkpoints
```

### 5. Use locally

```bash
# Copy the final checkpoint into your local checkpoints dir
cp remote_checkpoints/final.pt C:/dev/vocos_rt/checkpoints/finetune/final.pt

# Re-run the listening kit locally
.venv/Scripts/python.exe demos/demo_finetune_compare.py
```

## What you should hear

With the proper batch size (>= 8) and 100k steps:
- Voice personality preserved (the failure mode at batch=1 is fixed)
- Tube artifact substantially reduced or eliminated
- Quality close to upstream non-causal Vocos

If it still has artifacts after a proper H100 run, the next pivot would be a
HiFi-GAN family architecture (BigVGAN). But that's weeks of work; the H100
attempt should be tried first because the existing code already supports it
and it's a $2 experiment.

## Troubleshooting

- **OOM on the rental**: lower `--batch-size` in `scripts/train_remote.sh`
- **`nvidia-smi: command not found`**: pick a different image (must have CUDA drivers)
- **HF download slow**: pre-download the checkpoint locally and scp it to `checkpoints/`
- **Disk full when extracting LibriTTS**: provision ≥ 50 GB on the rental
- **Want to train longer**: bump `--num-steps 100000` to `200000` (still under 1h on H100)
