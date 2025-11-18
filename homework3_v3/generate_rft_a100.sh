#!/bin/bash
#
# A100 GPU Optimized RFT Data Generation Script
# This script maximizes throughput on A100 GPUs for RFT dataset generation
#

set -e  # Exit on error

echo "=========================================="
echo "A100 GPU RFT Data Generation"
echo "=========================================="
echo ""

# A100 optimizations via environment variables
export MICRO_BATCH_SIZE=256      # Increased from 32 (A100 has 40-80GB VRAM)
export CHUNK_SIZE=10             # Increased from 3 for sequence generation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Detect A100 and set optimal parameters
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ $GPU_NAME == *"A100"* ]]; then
    echo "✓ A100 GPU detected - using aggressive optimization settings"
    OVERSAMPLE=30      # Increased from 15 for better data quality
    BATCH_SIZE=8       # Process 8 questions in parallel
    USE_BFLOAT16=true  # A100 has excellent bfloat16 support
else
    echo "⚠ Non-A100 GPU detected ($GPU_NAME) - using moderate settings"
    OVERSAMPLE=20
    BATCH_SIZE=4
    USE_BFLOAT16=true
fi

# Allow overrides from command line
OVERSAMPLE=${1:-$OVERSAMPLE}
BATCH_SIZE=${2:-$BATCH_SIZE}
TEMPERATURE=${3:-0.7}

echo ""
echo "Configuration:"
echo "  Oversample: $OVERSAMPLE sequences per question"
echo "  Batch Size: $BATCH_SIZE questions in parallel"
echo "  Temperature: $TEMPERATURE"
echo "  Use BFloat16: $USE_BFLOAT16"
echo "  Micro Batch Size: $MICRO_BATCH_SIZE"
echo "  Chunk Size: $CHUNK_SIZE"
echo ""

# Clear any existing RFT data
OUTPUT_FILE="data/rft.json"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing $OUTPUT_FILE"
    rm "$OUTPUT_FILE"
fi

# Generate RFT dataset with A100 optimizations
echo "Starting RFT data generation..."
echo "This will be MUCH faster than standard generation on A100!"
echo ""

START_TIME=$(date +%s)

python -m homework.datagen \
    "$OUTPUT_FILE" \
    --oversample="$OVERSAMPLE" \
    --temperature="$TEMPERATURE" \
    --batch_size="$BATCH_SIZE" \
    --use_bfloat16="$USE_BFLOAT16"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "=========================================="
echo "Time elapsed: ${ELAPSED}s"
echo "Output file: $OUTPUT_FILE"

# Show statistics
if [ -f "$OUTPUT_FILE" ]; then
    NUM_EXAMPLES=$(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))")
    echo "Generated examples: $NUM_EXAMPLES"
    
    if [ "$NUM_EXAMPLES" -gt 0 ]; then
        RATE=$(echo "scale=2; $NUM_EXAMPLES / $ELAPSED" | bc)
        echo "Generation rate: ${RATE} examples/second"
        
        # Calculate speedup vs sequential processing
        SEQUENTIAL_ESTIMATE=$((900 * 3))  # ~3 seconds per question sequential
        SPEEDUP=$(echo "scale=1; $SEQUENTIAL_ESTIMATE / $ELAPSED" | bc)
        echo "Estimated speedup: ${SPEEDUP}x vs sequential"
    fi
    
    # Quality check
    if [ "$NUM_EXAMPLES" -lt 850 ]; then
        echo ""
        echo "⚠ WARNING: Only $NUM_EXAMPLES examples generated."
        echo "Target is 850-900+ for good generalization."
        echo "Consider increasing --oversample parameter or checking generation quality."
    else
        echo ""
        echo "✓ Dataset size looks good for training!"
    fi
else
    echo "ERROR: Output file not created!"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Train RFT model: python -m homework.rft train"
echo "  2. Test RFT model: python -m homework.rft test"
echo ""
