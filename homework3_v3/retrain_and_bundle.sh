#!/bin/bash
# Script to retrain models with reduced rank and create submission bundle

echo "=== Step 1: Removing old model files ==="
rm -f homework/sft_model/adapter_model.safetensors
rm -f homework/sft_model/adapter_model.bin
rm -f homework/sft_model/adapter_config.json
rm -f homework/rft_model/adapter_model.safetensors
rm -f homework/rft_model/adapter_model.bin
rm -f homework/rft_model/adapter_config.json
echo "Old models removed."

echo ""
echo "=== Step 2: Training SFT model with rank 8 ==="
python -m homework.sft train

echo ""
echo "=== Step 3: Training RFT model with rank 16 ==="
python -m homework.rft train

echo ""
echo "=== Step 4: Creating submission bundle ==="
python3 bundle.py homework dlk929

echo ""
echo "=== Step 5: Checking bundle size ==="
BUNDLE_SIZE=$(du -h dlk929.zip | cut -f1)
echo "Bundle size: $BUNDLE_SIZE"

echo ""
echo "Done! Please verify your bundle with:"
echo "python3 -m grader dlk929.zip"
