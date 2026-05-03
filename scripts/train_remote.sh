#!/usr/bin/env bash
# vocos_rt training command sized for a real GPU (H100 80GB / A100 40GB).
# Runs the full Vocos GAN recipe at the scale the original paper used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

# Detect available VRAM and pick batch size accordingly.
# H100 80GB -> batch 32 segment 32768
# A100 40GB -> batch 16 segment 32768
# Anything < 16 GB shouldn't be doing this; fall back to the 4 GB recipe.
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
echo "Detected ${VRAM_GB} GB VRAM"

if [ "$VRAM_GB" -ge 70 ]; then
    BATCH=32; SEGMENT=32768; ACCUM=1
elif [ "$VRAM_GB" -ge 35 ]; then
    BATCH=16; SEGMENT=32768; ACCUM=1
elif [ "$VRAM_GB" -ge 22 ]; then
    BATCH=8;  SEGMENT=32768; ACCUM=2   # eff batch 16
elif [ "$VRAM_GB" -ge 14 ]; then
    BATCH=4;  SEGMENT=24576; ACCUM=4   # eff batch 16
else
    echo "WARN: < 14 GB VRAM detected. Falling back to the 4 GB local recipe."
    BATCH=1; SEGMENT=8192; ACCUM=1
fi

echo "Effective batch=${BATCH} grad_accum=${ACCUM} segment=${SEGMENT}"

# Disable our memory-saving tricks (they hurt throughput and aren't needed)
# Keep gradient checkpointing on the generator (cheap, deterministic).
# Keep the envelope-flatness loss (D19 added; targets the hop-rate comb).
# RESTORE full MPD periods (2,3,5,7,11) since memory permits.

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/finetune_causal.py \
    --num-steps 100000 \
    --batch-size "$BATCH" \
    --grad-accum-steps "$ACCUM" \
    --segment-samples "$SEGMENT" \
    --num-workers 4 \
    --precision bf16 \
    --use-discriminators \
    --disc-warmup-steps 5000 \
    --mel-loss-weight 20.0 \
    --fm-weight 2.0 \
    --envelope-flatness-weight 5.0 \
    --log-every 100 \
    --checkpoint-every 5000 \
    --sample-every 5000 \
    --validate-every 5000 \
    --wall-clock-budget-h 4.0 \
    2>&1 | tee checkpoints/finetune/training.log

# Generate listening kit + final A/B audio
python scripts/build_listen_kit.py
python demos/demo_finetune_compare.py

echo
echo "============================================================"
echo " Training complete. Checkpoints in checkpoints/finetune/"
echo " Audio outputs in audio_to_listen_*/ and audio_out_finetune/"
echo " To download back: scp -r remote:vocos_rt/checkpoints/finetune ./remote_checkpoints"
echo "============================================================"
