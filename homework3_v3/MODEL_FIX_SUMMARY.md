# Model Parameter Fix - Summary

## Problem
The grader was rejecting the submission with error:
```
ValueError: Model has 1711376384 parameters, which is greater than the maximum allowed 380000000
```

The code was using `SmolLM2-1.7B-Instruct` (1.7 billion parameters) for all operations, which exceeds the grader's 380M parameter limit.

## Root Cause
In `base_llm.py`, the default checkpoint was set to:
```python
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # 1.7B parameters - TOO LARGE!
```

This affected:
- SFT model training
- SFT model inference/testing
- All other operations except data generation

## Solution Applied

### 1. Changed Default Model in `base_llm.py`
**File:** `/workspace/homework3_v3/homework/base_llm.py`

**Changed from:**
```python
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

**Changed to:**
```python
# Use 360M model by default to meet the 380M parameter limit required by the grader
# The 1.7B model should only be used for RFT data generation
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
```

**Impact:**
- ✓ SFT training now uses 360M model (~360M parameters)
- ✓ SFT inference now uses 360M model
- ✓ All other operations use 360M model by default
- ✓ Passes grader's 380M parameter limit with ~20M margin

### 2. Updated Data Generation to Use 1.7B Model
**File:** `/workspace/homework3_v3/homework/datagen.py`

**Changed:**
```python
# Use 1.7B model for better rollouts as recommended in README
# This is ONLY for data generation - the trained model will use 360M
model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
```

**Impact:**
- ✓ RFT data generation uses 1.7B model for better quality (as README recommends)
- ✓ Generated rft.json will have higher quality reasoning chains
- ✓ This is aligned with README line 138: "Using the HuggingFaceTB/SmolLM2-1.7B-Instruct model should further help you obtain better rollouts"

## What This Means

### Model Usage Summary:
| Operation | Model Used | Parameters | Reason |
|-----------|------------|------------|--------|
| **Data Generation (datagen.py)** | SmolLM2-1.7B-Instruct | ~1.7B | Better reasoning quality for training data |
| **SFT Training** | SmolLM2-360M-Instruct | ~360M | Meets grader requirements |
| **SFT Inference** | SmolLM2-360M-Instruct | ~360M | Meets grader requirements |
| **RFT Training** | SmolLM2-360M-Instruct | ~360M | Meets grader requirements |
| **RFT Inference** | SmolLM2-360M-Instruct | ~360M | Meets grader requirements |
| **CoT** | SmolLM2-360M-Instruct | ~360M | Meets grader requirements |

### README Compliance:
✓ **Line 138**: "Using the HuggingFaceTB/SmolLM2-1.7B-Instruct model should further help you obtain better rollouts"
   - **Applied to:** Data generation only (as intended)
   
✓ **Grader requirement**: Maximum 380M parameters
   - **Applied to:** All submitted models (SFT and RFT)

## Training Impact

### Expected Behavior:
1. **RFT Data Generation:**
   - When `python -m homework.sft train` runs, if `rft.json` doesn't exist, it will be automatically generated
   - Generation will use the **1.7B model** for better quality
   - This may take ~4 hours but produces high-quality reasoning chains

2. **SFT Training:**
   - Training will use the **360M model**
   - Model will be smaller but still capable
   - Training will be faster due to smaller model size
   - Final accuracy may be slightly lower than 1.7B but should still be good

3. **Grader Validation:**
   - Model will have ~360M parameters
   - Well within the 380M limit
   - No more parameter count errors

## SFT Accuracy Considerations

Your current accuracy is 0.36 (36%). Here are factors that may help improve it:

1. **Training is already on rft.json** ✓
   - The code automatically generates and uses rft.json
   - This is the correct approach per README

2. **Model Capacity:**
   - 360M model has less capacity than 1.7B
   - But should still achieve reasonable accuracy with proper training
   - The training configuration has been optimized with:
     - 6 epochs (increased from 5)
     - Learning rate 5e-4 (increased from 2e-4)
     - Cosine learning rate schedule
     - Gradient accumulation for effective batch size of 32

3. **Data Quality:**
   - Using 1.7B for data generation ensures high-quality rft.json
   - This compensates for using smaller model in training

## Next Steps

1. **Delete old rft.json if it exists:**
   ```bash
   rm /workspace/homework3_v3/data/rft.json  # If it was generated with wrong model
   ```

2. **Delete old trained models:**
   ```bash
   rm -rf /workspace/homework3_v3/homework/sft_model/*
   ```

3. **Run training fresh:**
   ```bash
   cd /workspace/homework3_v3
   python -m homework.sft train
   ```

4. **The training will:**
   - Auto-generate rft.json using 1.7B model (if missing)
   - Train SFT model using 360M model
   - Save a model that passes the grader's parameter check

## Verification

To verify the fix worked:
```bash
python -m grader [YOUR_UT_ID].zip
```

You should no longer see:
```
ValueError: Model has 1711376384 parameters, which is greater than the maximum allowed 380000000
```

Instead, the model should load and be graded successfully.

## Files Modified
1. `/workspace/homework3_v3/homework/base_llm.py` - Changed default checkpoint to 360M model
2. `/workspace/homework3_v3/homework/datagen.py` - Explicitly use 1.7B model for data generation

## Files NOT Modified (but inherit the fix)
- `sft.py` - Uses BaseLLM() which now defaults to 360M
- `rft.py` - Uses BaseLLM() which now defaults to 360M  
- `cot.py` - Inherits from BaseLLM, uses 360M by default
- All other files use BaseLLM() and get 360M automatically
