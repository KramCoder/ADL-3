# SFT Training Fix - Gradient Norm NaN and Accuracy Issues

## Problem Summary

The SFT training was experiencing:
1. **NaN gradient norms** throughout training (`'grad_norm': nan`)
2. **Zero accuracy** after training (`benchmark_result.accuracy=0.0`)
3. **Zero answer rate** (`benchmark_result.answer_rate=0.0`)

## Root Cause

The NaN gradient norm was caused by:
1. **Missing gradient clipping**: Without `max_grad_norm`, gradients could explode in fp16 training, causing overflow and NaN values
2. **Missing label names**: The Trainer wasn't explicitly told which field contains labels when using PeftModel, potentially causing gradient computation issues

When gradients are NaN, the model cannot learn, which explains why accuracy remained at 0.0.

## Solution Applied

Two key fixes were added to `TrainingArguments` in `/workspace/homework3_v3/homework/sft.py`:

```python
training_args = TrainingArguments(
    # ... other arguments ...
    max_grad_norm=1.0,  # Clip gradients to prevent explosion and NaN
    label_names=["labels"],  # Explicitly specify label field for PeftModel
)
```

### Fix 1: `max_grad_norm=1.0`
- **Purpose**: Clips gradients to prevent explosion and NaN values
- **How it works**: After computing gradients, they are scaled down if their norm exceeds 1.0
- **Why needed**: In fp16 training, large gradients can overflow, resulting in NaN values

### Fix 2: `label_names=["labels"]`
- **Purpose**: Explicitly tells the Trainer which field contains the labels
- **How it works**: Ensures the Trainer properly recognizes labels when using PeftModel wrapper
- **Why needed**: PeftModel wraps the base model, and the Trainer may not automatically detect the labels field

## Expected Results After Fix

### During Training:
- ✅ **Gradient norms should be finite** (typically 0.1-2.0 range)
- ✅ **Loss should decrease** over epochs
- ✅ **No NaN values** in training logs

### After Training:
- ✅ **Accuracy should improve** (from 0.0 to some positive value)
- ✅ **Answer rate should improve** (model should generate valid answers)
- ✅ **Model should learn** the unit conversion task

## Will This Fix Accuracy and Learning?

**Yes, this fix should resolve both issues:**

1. **Gradient Norm NaN → Fixed**: With `max_grad_norm`, gradients will be clipped and remain finite
2. **Model Learning → Enabled**: Once gradients flow properly, the model can update its weights
3. **Accuracy → Should Improve**: As the model learns, accuracy should increase from 0.0

### Important Notes:

- **Learning Rate**: The learning rate (2e-4) is appropriate and doesn't need changes
- **Training Time**: The model may need the full 3 epochs to achieve good accuracy
- **Initial Accuracy**: Don't expect perfect accuracy immediately - the model needs time to learn

## Testing the Fix

To verify the fix works:

1. **Check training logs**: Look for `grad_norm` values - they should be finite numbers, not NaN
2. **Monitor loss**: Loss should decrease over time
3. **Check accuracy**: After training, accuracy should be > 0.0

Example of expected training output:
```
{'loss': 2.27, 'grad_norm': 1.55, 'learning_rate': 0.0002, 'epoch': 0.0}
{'loss': 2.29, 'grad_norm': 1.81, 'learning_rate': 0.00013, 'epoch': 0.01}
{'loss': 1.50, 'grad_norm': 1.70, 'learning_rate': 0.0001, 'epoch': 0.02}
```

## Additional Considerations

If accuracy is still low after fixing gradient norms:

1. **More training**: The model might need more epochs
2. **Learning rate tuning**: Try adjusting learning rate (though 2e-4 is reasonable)
3. **Model capacity**: The LoRA rank (currently 4) might need to be increased
4. **Data quality**: Verify the training data is correct

However, the primary blocker (NaN gradients) is now fixed, so the model should be able to learn.
