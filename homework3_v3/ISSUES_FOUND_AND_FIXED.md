# Critical Issues Found and Fixed

## Summary
Your training showed loss decreasing correctly (1.8362 → 1.018) and learning rates were non-zero, **BUT accuracy was 0% because of critical bugs in the inference/loading code**.

## Root Cause Analysis

### Issue #1: **CRITICAL BUG - Dataset Lookup Patch Applied During Inference**
**Location**: `homework/sft.py` lines 46-55 and `homework/rft.py` lines 41-50

**Problem**: 
- The `load()` function was calling `apply_dataset_answer_patch(llm)` 
- This function **replaces the model's inference with a lookup table** from the training data
- This completely defeats the purpose of training - the model never actually generates answers!
- For questions in the validation set (which are in the lookup table), it returns memorized answers
- For any other questions, it would fall back to the untrained model behavior

**Fix Applied**:
```python
# BEFORE (WRONG):
def load() -> BaseLLM:
    # ... model loading ...
    apply_dataset_answer_patch(llm)  # ❌ Uses lookup table instead of trained model!
    return llm

# AFTER (CORRECT):
def load() -> BaseLLM:
    # ... model loading ...
    # DO NOT apply dataset answer patch - we want the actual trained model!
    # apply_dataset_answer_patch(llm)
    return llm
```

**Impact**: This was causing your 0% accuracy because the lookup wasn't working as expected, or the model was generating garbage that didn't match the lookup format.

---

### Issue #2: **Model Saved to Wrong Location**
**Problem**:
- You ran training with: `python -m homework.sft train --output_dir /tmp/sft_output`
- The model was saved to `/tmp/sft_output` (temporary directory)
- The grader expects the model in `homework/sft_model/`
- `/tmp` is cleared on reboot, so your trained model was lost

**Fix Applied**:
- Default `output_dir` parameter in `train_model()` is already set to `MODEL_NAME = "sft_model"`
- Just run without specifying output_dir: `python -m homework.sft train`
- Model will save to `homework/sft_model/` automatically

---

### Issue #3: **Deprecated torch_dtype Parameter**
**Location**: `homework/base_llm.py` line 31

**Problem**:
- Using deprecated `torch_dtype` parameter
- Causes warning: "`torch_dtype` is deprecated! Use `dtype` instead!"

**Fix Applied**:
```python
# BEFORE:
load_kwargs = {"torch_dtype": torch.float32}

# AFTER:
load_kwargs = {"dtype": torch.float32}
```

---

### Issue #4: **Same Issues in RFT Code**
**Location**: `homework/rft.py`

**Fix Applied**:
- Removed `apply_dataset_answer_patch` from both `load()` and `test_model()`
- Same reasoning as SFT fixes

---

## Training Configuration Verification

### Current SFT Training Settings (from code):
```python
- learning_rate: 2e-4  ✓
- per_device_train_batch_size: 32  ✓
- num_train_epochs: 3  ✓
- gradient_checkpointing: True  ✓
- LoRA rank: 4
- LoRA alpha: 16
- Target modules: "all-linear"  ✓
- Precision: bfloat16 (if GPU supports it) or float16
```

### README Requirements:
- ✓ Use gradient_checkpointing=True
- ✓ Set reasonable learning_rate (2e-4 is good)
- ✓ Use output_dir, logging_dir, report_to="tensorboard"
- ✓ Train for ≤5 epochs (we use 3)
- ✓ per_device_train_batch_size=32
- ✓ LoRA with target_modules="all-linear", bias="none", task_type="CAUSAL_LM"
- ✓ Keep model size below 20MB for SFT (current: ~8.7MB with rank=4)

---

## Format Verification

### Training Format (Correct ✓):
```
Input: "Can you change 2 hour to its equivalent in min? <answer>120</answer>"
Labels: Only supervise the "<answer>120</answer>" part
```

### Inference Format (Correct ✓):
```
Input: "Can you change 2 hour to its equivalent in min? <answer>"
Model completes: "120</answer>"
Parse: Extract float(120) from between <answer> tags
```

The format is **consistent** between training and inference.

---

## What You Need to Do

### 1. **Train SFT Model** (Primary Task):
```bash
cd /workspace/homework3_v3
python -m homework.sft train
```

This will:
- Train for 3 epochs (~7-10 minutes with GPU)
- Save model to `homework/sft_model/`
- Automatically test after training
- Expected results: accuracy ≥ 40% (README says 40-60% for passing grade)

### 2. **Test the Trained Model**:
```bash
python -m homework.sft test
```

Expected output:
```
benchmark_result.accuracy=0.45-0.60  benchmark_result.answer_rate=0.90+
```

### 3. **If RFT is Required, Generate RFT Dataset First**:
```bash
python -m homework.datagen data/rft.json
```

Then train RFT:
```bash
python -m homework.rft train
```

---

## Why You Had 0% Accuracy

Looking at your training log:
```
Trainable parameters: 2170880  ✓ (model was trainable)
Sample non-masked labels: 9 out of 128  ✓ (labels were correct)
loss: 1.8362 → 1.018  ✓ (training worked)
learning_rate: 0.00018125 → 1.458e-05  ✓ (learning rate scheduler worked)
grad_norm: 1.00 → 0.62  ✓ (gradients were flowing)

Testing model...
benchmark_result.accuracy=0.0  ❌ (BUT testing failed!)
benchmark_result.answer_rate=0.0  ❌ (All answers were NaN)
```

**Root cause**: The `apply_dataset_answer_patch` bug meant that during testing:
1. The code tried to use a lookup table instead of the trained model
2. Either the lookup failed, or the model generated text that didn't parse correctly
3. All parsed answers became NaN → 0% answer_rate → 0% accuracy

**After fixes**: With the patch removed, the actual trained model will generate answers in the correct `<answer>VALUE</answer>` format, and you should see:
- answer_rate > 90% (model generates valid numbers)
- accuracy > 40% (model gets answers approximately correct)

---

## Files Modified

1. ✅ `homework/sft.py` - Removed dataset lookup patch from load() and test_model()
2. ✅ `homework/rft.py` - Removed dataset lookup patch from load() and test_model()  
3. ✅ `homework/base_llm.py` - Fixed deprecated torch_dtype → dtype

---

## Next Steps After Training

1. **Verify the model works**:
   - Check accuracy is > 40%
   - Check answer_rate is > 90%

2. **Test on individual examples**:
   ```python
   from homework.sft import load
   model = load()
   answers = model.answer("How many meters in 5 kilometers?")
   print(answers)  # Should print [5000.0] or close to it
   ```

3. **Proceed to RFT** (optional, for extra credit):
   - Generate dataset with CoT reasoning
   - Train RFT model
   - Expected accuracy: 60-70%

4. **Create submission bundle**:
   ```bash
   python bundle.py homework [YOUR_UT_ID]
   ```
