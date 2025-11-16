# Training Fixes Summary

## Issues Found and Fixed

### 1. **Loss was 0.0 and grad_norm was NaN**

**Root Cause**: The LoRA adapter configuration had `inference_mode: true` set in the saved configuration, which freezes all adapter weights and prevents training.

**Fix Applied**:
- Added `inference_mode=False` to `LoraConfig` in both `sft.py` and `rft.py`
- Added explicit `lora_model.train()` call before training
- Ensured `enable_input_require_grads()` is always called (not just on CUDA)

**Files Modified**:
- `homework/sft.py` (lines 145-151)
- `homework/rft.py` (lines 73-87)

### 2. **Accuracy was artificially fixed at 1.0**

**Root Cause**: The `test_model` functions were calling `apply_dataset_answer_patch(llm)`, which replaces the model's answer generation with a lookup table from the training data.

**Fix Applied**:
- Commented out `apply_dataset_answer_patch(llm)` in test functions
- Now the test functions actually evaluate the trained model, not a lookup table

**Files Modified**:
- `homework/sft.py` (line 196)
- `homework/rft.py` (line 155)

### 3. **Data was not learning**

**Root Cause**: Combination of issues #1 and #2 above. The model wasn't actually training (frozen weights), and the evaluation wasn't measuring the model's performance.

**Fix Applied**: Same as fixes #1 and #2 above.

## Verification

After applying the fixes, training now shows:
- ✅ Loss starts at ~2.27 (realistic value)
- ✅ Gradients flow properly (grad_norm ~1.5-2.8)
- ✅ Loss decreases during training (from 2.30 → 1.62 in 10 steps)
- ✅ Learning rate decreases as expected (cosine schedule)
- ✅ Model is actually learning from the data

## Before vs After

### Before:
```
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018125, 'epoch': 0.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00016041666666666667, 'epoch': 0.62}
...
benchmark_result.accuracy=1.0  benchmark_result.answer_rate=1.0
```

### After:
```
{'loss': 2.2788, 'grad_norm': 1.9252, 'learning_rate': 0.00018, 'epoch': 0.01}
{'loss': 2.3272, 'grad_norm': 2.8629, 'learning_rate': 0.00014, 'epoch': 0.02}
{'loss': 1.4994, 'grad_norm': 1.6986, 'learning_rate': 0.0001, 'epoch': 0.02}
...
(Actual model performance, not lookup table)
```

## Key Changes in Code

### sft.py (train_model function)
```python
config = LoraConfig(
    task_type="CAUSAL_LM",
    target_modules="all-linear",
    bias="none",
    r=DEFAULT_LORA_RANK,
    lora_alpha=max(DEFAULT_LORA_RANK * 4, 4),
    lora_dropout=0.0,
    inference_mode=False,  # CRITICAL: Must be False for training
)

lora_model = get_peft_model(llm.model, config)

# Set model to training mode
lora_model.train()

# Enable input require grads for gradient checkpointing
lora_model.enable_input_require_grads()
```

### sft.py (test_model function)
```python
llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
llm.model.eval()
# NOTE: Do NOT apply dataset answer patch during testing
# We want to test the actual model, not the lookup table
# apply_dataset_answer_patch(llm)

benchmark_result = benchmark(llm, testset, 100)
```

## Next Steps

The training infrastructure is now working correctly. You can now:

1. Train the SFT model: `python -m homework.sft train`
2. Train the RFT model: `python -m homework.rft train` (after generating RFT data)
3. Test models without artificial accuracy inflation

The old pre-trained model files have been removed to ensure fresh training with the corrected configuration.
