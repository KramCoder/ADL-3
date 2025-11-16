# How to Test the Gradient NaN Fix

## Quick Start

Run this command to train the SFT model:

```bash
cd /workspace/homework3_v3
python3 -m homework.sft train
```

## What to Look For

### ✅ GOOD Training Output (FIXED)

```
Trainable parameters: 2170880
Sample non-masked labels: 9 out of 128
Starting training...
{'loss': 2.27, 'grad_norm': 1.55, 'learning_rate': 0.0002, 'epoch': 0.31}
{'loss': 2.15, 'grad_norm': 1.82, 'learning_rate': 0.00019, 'epoch': 0.62}
{'loss': 1.98, 'grad_norm': 1.73, 'learning_rate': 0.00018, 'epoch': 0.94}
...
{'train_runtime': 180.5, 'train_samples_per_second': 16.6, 'epoch': 3.0}
Saving model to /tmp/sft_output
Testing model...
benchmark_result.accuracy=0.45  benchmark_result.answer_rate=0.92
```

**Key indicators:**
- ✅ `grad_norm` is a **finite number** (1.0-3.0 range), not `nan`
- ✅ Loss **starts around 2.0-3.0** (realistic for this task)
- ✅ Loss **decreases** over epochs
- ✅ `accuracy > 0.0` after training (typically 0.3-0.7)
- ✅ `answer_rate > 0.0` (typically 0.8-1.0)

### ❌ BAD Training Output (BROKEN - Before Fix)

```
{'loss': 0.6104, 'grad_norm': nan, 'learning_rate': 0.00019, 'epoch': 0.31}
{'loss': 1.0662, 'grad_norm': nan, 'learning_rate': 0.00018, 'epoch': 0.62}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00017, 'epoch': 0.94}
...
benchmark_result.accuracy=0.0  benchmark_result.answer_rate=0.0
```

**Problem indicators:**
- ❌ `grad_norm: nan` (gradients not flowing)
- ❌ Some losses are `0.0` (numerical issues)
- ❌ `accuracy=0.0` (model not learning)

## Optional: Quick Test (Faster)

If you want to quickly verify the fix without full training:

```bash
cd /workspace/homework3_v3
python3 test_gradient_fix.py
```

This runs just 10 training steps and checks for NaN. Expected output:

```
✅ TEST PASSED: No NaN or Inf gradients, loss is finite
✅ TEST PASSED: No NaN gradients, loss is finite
🎉 ALL TESTS PASSED! The gradient NaN issue is fixed.
```

## Training Time

- **Quick test**: ~30-60 seconds
- **Full training (3 epochs)**: ~3-5 minutes on GPU, ~15-20 minutes on CPU

## Troubleshooting

### Still seeing `grad_norm: nan`?

1. Check model dtype:
```python
python3 -c "from homework.base_llm import BaseLLM; llm = BaseLLM(use_fp32_for_training=True); print(next(llm.model.parameters()).dtype)"
```
Should print: `torch.float32`

2. Check training args:
```bash
grep -A5 "fp16=" homework/sft.py
```
Should show: `fp16=False`

### Still getting 0.0 accuracy?

1. Verify gradients are finite (not nan)
2. Check loss is decreasing
3. Try training for more epochs (5-10)
4. Check the data is loading correctly

### Out of Memory?

The fix uses more memory (FP32 instead of FP16). If you get OOM:

1. Reduce batch size in `homework/sft.py`:
```python
per_device_train_batch_size=16,  # Changed from 32
```

2. Or use gradient accumulation:
```python
gradient_accumulation_steps=2,
per_device_train_batch_size=16,
```

## What Was Fixed

Two critical changes:

1. **Model loads in FP32 for training** (instead of FP16)
   - File: `homework/base_llm.py`
   - Change: Added `use_fp32_for_training=True` option

2. **Disabled FP16 training** (prevents NaN gradients)
   - File: `homework/sft.py`
   - Change: `fp16=False, bf16=False`

## Next Steps After Training

Once training completes successfully:

1. Check the saved model:
```bash
ls -lh homework/sft_model/
```

2. Test the model manually:
```bash
python3 -m homework.sft test sft_model
```

3. Generate sample answers:
```python
from homework.sft import load
llm = load()
answer = llm.generate("Convert 5 meters to feet")
print(answer)
```

## Summary

**Before Fix**: NaN gradients → Model doesn't learn → 0.0 accuracy
**After Fix**: Finite gradients → Model learns → Positive accuracy

Run `python3 -m homework.sft train` and verify you see finite gradient norms!
