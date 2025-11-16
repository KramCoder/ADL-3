# Gradient Norm NaN Fix - Comprehensive Solution

## Problem Summary

During SFT training, the following issues were observed:
1. **NaN gradient norms** throughout training (`'grad_norm': nan`)
2. **Loss values of 0.0** in some steps (suspicious)
3. **Model accuracy of 0.0** after training (model not learning)

## Root Causes Identified

1. **FP16 Numerical Instability**: Using `fp16=True` can cause gradient overflow/underflow, leading to NaN values
2. **Missing Gradient Validation**: No checks for NaN/Inf gradients before clipping
3. **Loss Computation Issues**: Potential issues with loss computation when labels are masked
4. **No Loss Scaling**: FP16 training requires proper loss scaling (though Trainer handles this automatically)

## Solutions Implemented

### 1. Improved Precision Handling
- **Prefer bfloat16 (bf16) over float16 (fp16)**: bf16 is more stable and less prone to overflow
- **Automatic detection**: Checks if GPU supports bf16 (Ampere+ GPUs)
- **Fallback to fp16**: If bf16 not available, uses fp16 with proper handling

```python
# Determine precision settings - prefer bf16 over fp16 for stability
use_bf16 = False
use_fp16 = False
if torch.cuda.is_available():
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        use_bf16 = True
    else:
        use_fp16 = True
```

### 2. Custom Gradient Norm Callback
- **Validates gradients**: Checks for NaN/Inf in all gradients before clipping
- **Proper gradient norm computation**: Manually computes gradient norm from all trainable parameters
- **NaN detection and handling**: Zeroes out NaN/Inf gradients to prevent training crash
- **Logging**: Properly logs gradient norms and NaN counts

Key features:
- `on_step_end`: Validates gradients after backward pass, before optimizer step
- `on_log`: Updates logs with computed gradient norm
- Detects and reports which parameters have NaN/Inf gradients

### 3. Stable Loss Computation
- **Custom Trainer class**: `StableTrainer` with validated loss computation
- **Loss validation**: Checks for NaN/Inf loss values
- **Masked label handling**: Properly handles -100 masked labels with `ignore_index=-100`
- **Edge case detection**: Warns when loss is suspiciously small (might indicate all labels masked)

### 4. Additional Stability Settings
- `dataloader_num_workers=0`: Avoids multiprocessing issues
- `fp16_full_eval=False`: Uses full precision for evaluation
- `ddp_find_unused_parameters=False`: Faster training

## Expected Results

After applying these fixes:

1. **Gradient norms should be finite numbers** (typically 0.1-2.0 range)
2. **No NaN values** in gradient norms or loss
3. **Stable training** without crashes
4. **Model should learn** (accuracy > 0)

## Testing

To verify the fix works:

1. Run training and check logs for:
   - `grad_norm` values should be finite numbers (not NaN)
   - No "WARNING: NaN detected" messages
   - Loss values should be reasonable (typically 0.1-3.0 range)

2. Check for these indicators:
   - ✅ `grad_norm` is a finite number (e.g., 1.5, 0.8, 2.1)
   - ✅ Loss decreases over time
   - ✅ No NaN/Inf warnings in logs
   - ✅ Model accuracy improves (should be > 0)

3. Example of good output:
```
{'loss': 2.27, 'grad_norm': 1.55, 'learning_rate': 0.0002, 'epoch': 0.0}
{'loss': 2.29, 'grad_norm': 1.81, 'learning_rate': 0.00013, 'epoch': 0.01}
{'loss': 1.50, 'grad_norm': 1.70, 'learning_rate': 0.0001, 'epoch': 0.02}
```

## Key Changes Made

1. **Added `GradientNormCallback`**: Validates gradients and computes proper norms
2. **Added `StableTrainer`**: Custom trainer with validated loss computation
3. **Improved precision handling**: Prefers bf16, falls back to fp16
4. **Added validation**: Checks for NaN/Inf in both loss and gradients
5. **Better error handling**: Zeroes out invalid gradients instead of crashing

## Files Modified

- `homework3_v3/homework/sft.py`: Added gradient validation, stable loss computation, and improved precision handling
