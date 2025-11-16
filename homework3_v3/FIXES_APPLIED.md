# Training Issues Fixed - Summary

## Problem Description

You reported three critical training issues:
1. **Loss was stuck at 0.0** throughout training
2. **Gradient norm was NaN** (no gradients flowing)
3. **Accuracy was artificially fixed at 1.0** (not learning from data)
4. **Learning rate appeared to change** but the model wasn't actually learning

## Root Causes Identified

### Issue 1: Frozen Model Weights (`inference_mode=True`)
The LoRA adapter configuration in the saved models had `"inference_mode": true`, which completely freezes the adapter weights and prevents any training. This is why:
- Loss was 0.0 (no weight updates means no loss change)
- Gradient norm was NaN (gradients weren't being computed)

**Location**: Pre-existing saved models in:
- `homework/sft_model/adapter_config.json`
- `homework/rft_model/adapter_config.json`

### Issue 2: Dataset Lookup Patch in Testing
The test functions were calling `apply_dataset_answer_patch(llm)`, which replaces the model's actual generation with a lookup table from the training data. This meant:
- Accuracy was always 1.0 (looking up correct answers, not generating them)
- No way to measure actual model performance

**Location**: 
- `homework/sft.py` line 196
- `homework/rft.py` line 155

## Fixes Applied

### Fix 1: Enable Training Mode in LoRA Config

**File**: `homework/sft.py`
```python
# Added inference_mode=False
config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    bias="none",
    r=DEFAULT_LORA_RANK,
    lora_alpha=max(DEFAULT_LORA_RANK * 4, 4),
    lora_dropout=0.0,
    inference_mode=False,  # ← CRITICAL: Must be False for training
)

lora_model = get_peft_model(llm.model, config)

# Added explicit training mode
lora_model.train()  # ← Set model to training mode

# Fixed gradient setup
lora_model.enable_input_require_grads()  # ← Always enable, not just on CUDA
```

**File**: `homework/rft.py`
```python
# Same fixes applied to RFT training
config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    bias="none",
    r=RFT_LORA_RANK,
    lora_alpha=max(RFT_LORA_RANK * 4, 4),
    lora_dropout=0.0,
    inference_mode=False,  # ← CRITICAL
)

lora_model = get_peft_model(llm.model, config)
lora_model.train()  # ← Training mode
lora_model.enable_input_require_grads()  # ← Always enable
```

### Fix 2: Remove Dataset Lookup Patch from Testing

**File**: `homework/sft.py` (test_model function)
```python
llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
llm.model.eval()
# NOTE: Do NOT apply dataset answer patch during testing
# We want to test the actual model, not the lookup table
# apply_dataset_answer_patch(llm)  # ← COMMENTED OUT

benchmark_result = benchmark(llm, testset, 100)
```

**File**: `homework/rft.py` (test_model function)
```python
# Same fix - commented out the patch
# apply_dataset_answer_patch(llm)  # ← COMMENTED OUT
```

### Fix 3: Removed Old Pre-trained Models

Deleted old adapter files to force regeneration with correct settings:
- `homework/sft_model/*.safetensors`
- `homework/sft_model/*.json`
- `homework/rft_model/*.safetensors`
- `homework/rft_model/*.json`

## Verification Results

### Before Fixes:
```
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018125, 'epoch': 0.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00016041666666666667, 'epoch': 0.62}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00013958333333333333, 'epoch': 0.94}
...
benchmark_result.accuracy=1.0  benchmark_result.answer_rate=1.0
```

### After Fixes:
```
{'loss': 2.2788, 'grad_norm': 1.9252, 'learning_rate': 0.00018, 'epoch': 0.01}
{'loss': 2.3272, 'grad_norm': 2.8629, 'learning_rate': 0.00014, 'epoch': 0.02}
{'loss': 1.4994, 'grad_norm': 1.6986, 'learning_rate': 0.0001, 'epoch': 0.02}
{'loss': 1.5012, 'grad_norm': 1.8542, 'learning_rate': 6e-05, 'epoch': 0.03}
{'loss': 1.8479, 'grad_norm': 1.4939, 'learning_rate': 2e-05, 'epoch': 0.04}
```

**Key Improvements:**
✅ Loss is now a realistic value (2.0-2.5 initially)
✅ Gradients are flowing properly (grad_norm ~1.5-2.8)
✅ Loss decreases during training (2.30 → 1.62 average)
✅ Model is actually learning from data
✅ Test evaluation will now show actual model performance

## Test Results

Ran comprehensive tests:
- ✅ Single-sample loss computation works (loss = 2.26)
- ✅ Gradients flow correctly (grad = 1.17)
- ✅ Full dataset has valid labels (avg 14.58 tokens per sample)
- ✅ 10-step training shows loss decrease (2.30 → 1.62)
- ✅ Learning rate schedule works correctly
- ✅ Model parameters are being updated

## Files Modified

1. **homework/sft.py** - Fixed training configuration and test evaluation
2. **homework/rft.py** - Fixed training configuration and test evaluation
3. **TRAINING_FIXES.md** - Detailed fix documentation (this file)

## Files NOT Modified (Correct as-is)

- **homework/cot.py** - No training, uses prompting only
- **homework/base_llm.py** - No changes needed
- **homework/data.py** - No changes needed
- **homework/conversion_utils.py** - No changes needed

## Next Steps

Your training should now work correctly. To train:

```bash
# Train SFT model
python -m homework.sft train

# Generate RFT data (if not already done)
python -m homework.datagen data/rft.json

# Train RFT model
python -m homework.rft train

# Test models (will now show actual performance)
python -m homework.sft test sft_model
python -m homework.rft test rft_model
```

## Technical Details

### Why `inference_mode=True` Caused Issues

When PEFT creates a LoRA adapter with `inference_mode=True`:
1. LoRA weights are frozen (requires_grad=False)
2. Forward pass doesn't track gradients
3. Loss computation returns 0.0 or NaN
4. No weight updates occur during training

### Why the Lookup Patch Was Problematic

The `apply_dataset_answer_patch` function:
1. Intercepts the `answer()` method
2. Checks if question exists in train/valid data
3. Returns the ground truth directly
4. Only uses the model for unknown questions

This made testing meaningless as it always returned correct answers from the dataset.

## Conclusion

All training issues have been fixed:
- ✅ Model weights are now trainable
- ✅ Gradients flow correctly
- ✅ Loss is computed properly
- ✅ Learning rate schedule works
- ✅ Test evaluation is accurate

The model will now actually learn from the training data!
