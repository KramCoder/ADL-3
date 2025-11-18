# Submission Size Reduction Guide

This guide explains how to reduce your submission size from ~91 MB to under 50 MB while maintaining model integrity.

## Problem

The adapter model files (`adapter_model.safetensors`) are stored in FP32 format, which takes up significant space. A typical adapter file can be 30-40 MB each, and with both SFT and RFT models, this easily exceeds 50 MB.

## Solution

We use **FP16 quantization** to reduce file size by approximately 50% while maintaining model integrity. FP16 (half precision) is standard for inference and does not significantly impact model performance.

## Quick Start

### Option 1: Automated (Recommended)

Use the `prepare_submission.py` script which handles both quantization and bundling:

```bash
python3 prepare_submission.py homework dlk929
```

This will:
1. Quantize adapter models to FP16
2. Create backups of original files
3. Create the submission bundle

### Option 2: Manual Steps

1. **Quantize the adapter models:**
   ```bash
   python3 quantize_adapters.py homework/sft_model homework/rft_model
   ```

2. **Create the bundle:**
   ```bash
   python3 bundle.py homework dlk929
   ```

## What Gets Changed

- **Adapter weights**: Converted from FP32 to FP16 (50% size reduction)
- **Backup files**: Original files saved as `.backup` (excluded from bundle)
- **README.md files**: Already excluded from bundle (not needed for grading)

## File Integrity

- ✅ Model weights are preserved (just lower precision)
- ✅ Model configuration files unchanged
- ✅ All Python code unchanged
- ✅ Original files backed up automatically

## Restoring Original Files

If you need to restore the original FP32 files:

```bash
# For SFT model
mv homework/sft_model/adapter_model.safetensors.backup homework/sft_model/adapter_model.safetensors

# For RFT model  
mv homework/rft_model/adapter_model.safetensors.backup homework/rft_model/adapter_model.safetensors
```

## Expected Results

- **Before**: ~91 MB zip file
- **After**: ~45-50 MB zip file (under the 50 MB limit)
- **Size reduction**: ~45-50% reduction

## Troubleshooting

If the bundle is still too large:
1. Check that both models were quantized (look for `.backup` files)
2. Verify README.md files are excluded (they're in the blacklist)
3. Ensure no other large files are being included
