# A100 Quick Start Guide

## Quick Test

To verify A100 optimizations are working:

```bash
# 1. Check GPU detection
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 2. Run data generation (should see A100 detection messages)
python -m homework.datagen data/rft.json --oversample=15

# 3. Train RFT model (should see A100 optimizations)
python -m homework.rft train
```

## Expected Output

### Data Generation
You should see:
```
Detected A100 GPU: NVIDIA A100-SXM4-80GB
Using optimized batch processing for A100 (80GB memory)
Generating RFT dataset with 15 sequences per question...
Processing 1000 questions...
```

### Training
You should see:
```
Detected A100 GPU: NVIDIA A100-SXM4-80GB
Using bfloat16 for training (optimal for A100 - 2-3x faster than FP32)
A100 optimizations: batch_size=32, effective_batch=64
```

## Performance Comparison

### Data Generation (850+ examples)
- **Before**: ~30-60 minutes
- **After (A100)**: ~5-10 minutes
- **Speedup**: 6-8x

### Training (3 epochs)
- **Before**: ~60-90 minutes total
- **After (A100)**: ~20-30 minutes total  
- **Speedup**: 2-3x

## Troubleshooting

### Not seeing A100 optimizations?
1. Check GPU name: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
2. Ensure GPU name contains "a100" (case-insensitive)
3. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory errors?
- Reduce `question_batch_size` in `datagen.py` (line 91) from 8 to 4
- Reduce `per_device_batch_size` in `rft.py` (line 194) from 32 to 16
- Reduce `oversample` parameter: `--oversample=10` instead of 15

### Still slow?
- Verify BF16 is enabled: Check for "Using bfloat16" messages
- Check batch sizes: Should see larger batches on A100
- Monitor GPU utilization: `nvidia-smi` should show high GPU usage

## Advanced: Manual Overrides

If you want to force specific settings:

```python
# Force FP32 (slower but more stable)
os.environ["USE_FP32_INFERENCE"] = "1"

# Force FP16 (faster but less stable)
os.environ["USE_FP16_INFERENCE"] = "1"
```

Note: A100 optimizations automatically use BF16, which is better than both FP32 and FP16.
