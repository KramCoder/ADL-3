#!/bin/bash
# Quick training script - run this to train your SFT model

set -e  # Exit on error

echo "=========================================="
echo "Starting SFT Training"
echo "=========================================="
echo ""

cd /workspace/homework3_v3

# Train the model
python -m homework.sft train

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: homework/sft_model/"
echo ""
echo "Next steps:"
echo "1. Check that accuracy > 40% above"
echo "2. (Optional) Train RFT for extra credit:"
echo "   python -m homework.datagen data/rft.json"
echo "   python -m homework.rft train"
echo "3. Create submission:"
echo "   python bundle.py homework YOUR_UT_ID"
echo ""
