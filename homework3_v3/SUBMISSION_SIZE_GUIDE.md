# Reducing Submission Size Guide

## Current Status
Your submission zip file should be under 50 MB. The main contributors to size are:
- `sft_model/adapter_model.safetensors` (~33 MB)
- `rft_model/adapter_model.safetensors` (if exists, ~33 MB)

## Optimization Strategies

### 1. Use the Updated Bundle Scripts

I've updated `bundle.py` with:
- Maximum compression level (compresslevel=9)
- Better size reporting and warnings
- Updated size limit to 50 MB

**Option A: Use the standard bundle script (recommended)**
```bash
python3 bundle.py homework dlk929
```

**Option B: Use the minimal bundle script (only includes exact files needed)**
```bash
python3 bundle_minimal.py homework dlk929
```

### 2. Quantize Model Weights (If Still Too Large)

If your zip file is still over 50 MB, you can quantize the model weights from float32 to float16, which reduces size by ~50%.

**Install safetensors (if needed):**
```bash
pip install safetensors
```

**Quantize models:**
```bash
# Check current sizes
python3 optimize_models.py homework/sft_model --check
python3 optimize_models.py homework/rft_model --check

# Quantize to float16 (reduces size by ~50%)
python3 optimize_models.py homework/sft_model --quantize
python3 optimize_models.py homework/rft_model --quantize
```

**Note:** Make sure your models work correctly with float16 before submitting!

### 3. Reduce LoRA Rank (If Possible)

If you can reduce the LoRA rank in your training, this will significantly reduce model size:
- Current SFT rank: 16 (from `sft.py`)
- Current RFT rank: 32 (from `rft.py`, which is `max(16*2, 16)`)

Reducing rank reduces the number of parameters quadratically.

### 4. Verify What's Included

The bundle scripts will print all files being included. Make sure you're not accidentally including:
- Checkpoint directories
- Training logs
- Optimizer states
- README files (already excluded)
- __pycache__ directories (already excluded)

## Files That Must Be Included

Based on your requirements:
- `rft.py`
- `datagen.py`
- `cot.py`
- `sft_model/` (with `adapter_model.safetensors` and `adapter_config.json`)
- `data.py`
- `rft_model/` (with `adapter_model.safetensors` and `adapter_config.json`)
- `__init__.py`
- `base_llm.py`
- `conversion_utils.py`
- `sft.py`

## Testing Your Bundle

After creating the bundle, verify it:
```bash
# Check the size
ls -lh dlk929.zip

# List contents
unzip -l dlk929.zip

# Test extraction (optional)
unzip -t dlk929.zip
```

## Expected Results

- **With one model (~33 MB)**: Zip should be ~30-35 MB ✅
- **With two models (~66 MB uncompressed)**: Zip might be ~60-65 MB ⚠️
  - Solution: Quantize models to float16 → ~30-35 MB zip ✅

## Troubleshooting

If your zip is still too large:
1. Check if both model directories have safetensors files
2. Verify no extra files are being included
3. Use `bundle_minimal.py` to ensure only required files
4. Quantize models if necessary
5. Consider reducing LoRA rank if accuracy allows
