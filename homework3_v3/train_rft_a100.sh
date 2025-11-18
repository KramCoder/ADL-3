#!/bin/bash
#
# A100 GPU Optimized RFT Training Script
# This script maximizes training throughput on A100 GPUs
#

set -e  # Exit on error

echo "=========================================="
echo "A100 GPU RFT Model Training"
echo "=========================================="
echo ""

# A100 training optimizations
export A100_BATCH_SIZE=32        # Increased from 16 (A100 can handle it)
export A100_GRAD_ACCUM=1         # Reduced from 2 since batch is larger
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if RFT dataset exists
if [ ! -f "data/rft.json" ]; then
    echo "ERROR: data/rft.json not found!"
    echo "Please run generate_rft_a100.sh first to create the dataset."
    exit 1
fi

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ $GPU_NAME == *"A100"* ]]; then
    echo "✓ A100 GPU detected - using optimized training settings"
    echo "  Batch Size: $A100_BATCH_SIZE"
    echo "  Gradient Accumulation: $A100_GRAD_ACCUM"
    echo "  Effective Batch Size: $((A100_BATCH_SIZE * A100_GRAD_ACCUM))"
else
    echo "⚠ Non-A100 GPU detected ($GPU_NAME)"
    # Use more conservative settings
    export A100_BATCH_SIZE=16
    export A100_GRAD_ACCUM=2
fi

echo ""
echo "Starting RFT training..."
echo ""

START_TIME=$(date +%s)

python -m homework.rft train

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Time elapsed: ${ELAPSED}s"
echo "Model saved to: homework3_v3/homework/rft_model/"
echo ""
