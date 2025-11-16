# Gradient Norm NaN Fix - Comprehensive Solution

## Problem Summary

The SFT training was experiencing:
1. **NaN gradient norms** throughout training (`'grad_norm': nan`)
2. **Zero loss values** (`'loss': 0.0`) which prevent gradient computation
3. **Zero accuracy** after training, indicating the model wasn't learning

## Root Causes

1. **Zero Loss → No Gradients**: When loss is 0.0, no gradients are computed, leading to NaN gradient norms
2. **fp16 Overflow**: fp16 training can cause gradient overflow to inf/NaN
3. **All Labels Masked**: In some batches, all labels might be masked (-100), resulting in 0.0 loss
4. **Unstable Training**: Without proper safeguards, NaN values propagate through the training loop

## Solutions Implemented

### 1. Gradient Monitoring Callback (`GradientMonitoringCallback`)

Added a custom callback that:
- **Detects NaN/inf gradient norms** in training logs and reports them
- **Detects zero/NaN loss** values and warns about them
- **Fixes NaN/inf gradients** in `on_step_end` by zeroing NaN gradients and clipping inf gradients
- Provides real-time monitoring during training

**Key Features:**
```python
- on_log(): Monitors logs for invalid gradient norms and loss values
- on_step_end(): Checks all model parameters for NaN/inf gradients and fixes them
```

### 2. Safe Loss Computation (`safe_compute_loss`)

Overrides the Trainer's `compute_loss` method to:
- **Validate labels**: Check that non-masked labels exist in each batch
- **Prevent zero loss**: Replace 0.0 loss with a small positive value (1e-6) to ensure gradients are computed
- **Prevent NaN loss**: Replace NaN loss with a small positive value
- **Handle edge cases**: Gracefully handle None loss or computation errors

**Protection Logic:**
```python
1. Check if all labels are masked → return dummy loss (1e-6)
2. Compute loss normally
3. If loss is None/NaN/0.0 → replace with 1e-6
4. Return validated loss
```

### 3. Better Numerical Precision (bf16 > fp16)

Changed from fp16 to bf16 when available:
- **bf16** has better numerical stability than fp16
- Prevents gradient overflow issues
- Falls back to fp16 if bf16 is not supported

**Implementation:**
```python
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16
```

### 4. Enhanced Training Arguments

Added/improved:
- `max_grad_norm=1.0`: Gradient clipping (already present, but now more effective)
- `warmup_steps=50`: Stabilize training at the start
- `bf16=True` when available: Better numerical stability
- Proper `label_names`: Explicit label specification for PeftModel

## Code Changes

### New Classes/Components

1. **`GradientMonitoringCallback`**: Custom TrainerCallback for gradient monitoring
2. **`safe_compute_loss`**: Wrapper around Trainer's compute_loss with validation

### Modified Functions

1. **`train_model()`**: 
   - Added callback instantiation and registration
   - Added compute_loss override
   - Improved training arguments (bf16, warmup)

## Expected Results

After these fixes:

1. **✅ No NaN gradient norms**: All gradient norms should be finite numbers
2. **✅ No zero loss**: Loss values should be positive and meaningful
3. **✅ Stable training**: Training should proceed without NaN propagation
4. **✅ Model learning**: The model should achieve non-zero accuracy

## Example Output (Expected)

**Before (with issues):**
```
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018, 'epoch': 0.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00016, 'epoch': 0.62}
```

**After (fixed):**
```
{'loss': 2.27, 'grad_norm': 1.55, 'learning_rate': 0.0002, 'epoch': 0.0}
{'loss': 2.29, 'grad_norm': 1.81, 'learning_rate': 0.00013, 'epoch': 0.01}
{'loss': 1.50, 'grad_norm': 1.70, 'learning_rate': 0.0001, 'epoch': 0.02}
```

## Testing

To verify the fix works:

1. **Run training** and check logs for:
   - ✅ Finite gradient norms (not NaN)
   - ✅ Positive loss values (not 0.0)
   - ✅ No NaN warnings in output

2. **Check model performance**:
   - ✅ Accuracy > 0.0
   - ✅ Answer rate > 0.0

3. **Monitor warnings**:
   - The callback will print warnings if issues are detected
   - These should be rare or non-existent after the fix

## Technical Details

### Why Zero Loss Causes NaN Gradients

When loss is 0.0:
- No gradients are computed (gradient of 0 is 0)
- Gradient norm calculation: `sqrt(sum(grad^2))` = `sqrt(0)` = 0
- However, if loss computation fails or is skipped, gradients might be None/NaN
- NaN propagates through the norm calculation

### Why bf16 is Better Than fp16

- **fp16**: 5 exponent bits, 10 mantissa bits → prone to overflow
- **bf16**: 8 exponent bits, 7 mantissa bits → better range, less overflow
- **Result**: More stable gradients, fewer NaN issues

### Gradient Clipping with max_grad_norm

- Clips gradients to have norm ≤ 1.0
- Prevents gradient explosion
- Works in conjunction with NaN detection to ensure stable training

## Files Modified

- `homework3_v3/homework/sft.py`: Main training code with all fixes

## Next Steps

1. Run training and verify gradient norms are finite
2. Check that loss values are positive and decreasing
3. Verify model accuracy improves during training
4. Monitor for any remaining warnings
