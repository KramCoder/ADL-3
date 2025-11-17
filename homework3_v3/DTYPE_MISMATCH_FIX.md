# SFT Zero Accuracy Fix - Dtype Mismatch Issue

## Problem
After training the SFT model with `python -m homework.sft train`, the model shows:
- Training completes successfully with decreasing loss
- **BUT** testing shows: `accuracy=0.0` and `answer_rate=0.0`
- All generated answers are NaN (not valid numbers)

## Root Cause
**Dtype mismatch between training and inference:**

1. **During Training** (in `train_model()`):
   - Base model loaded with: `BaseLLM(use_fp32_for_training=True)`
   - This loads the model in **FP32** (torch.float32)
   - LoRA adapter is trained and saved with FP32 weights

2. **During Inference** (in `test_model()` and `load()`):
   - Base model was loaded with: `BaseLLM()` (default)
   - This loads the model in **FP16** (torch.float16) on CUDA
   - LoRA weights are FP32 but base model is FP16
   - **Result**: Dtype mismatch causes invalid outputs (all NaN)

## Solution
Ensure both training and inference use the same dtype configuration:

### Changes Made to `/workspace/homework3_v3/homework/sft.py`:

1. **In `load()` function (line 57)**:
   ```python
   # BEFORE:
   llm = BaseLLM()
   
   # AFTER:
   llm = BaseLLM(use_fp32_for_training=True)
   ```

2. **In `test_model()` function (line 461)**:
   ```python
   # BEFORE:
   llm = BaseLLM()
   
   # AFTER:
   llm = BaseLLM(use_fp32_for_training=True)
   ```

## Key Points
✅ **torch_dtype parameter is preserved**: The code uses `"torch_dtype"` (not just `"dtype"`) as required by the grader

✅ **No chat template used**: The format_prompt correctly returns `f"{question.strip()} <answer>"` without using chat templates

✅ **Correct training format**: 
   - Training: `"question <answer>value</answer>"`
   - Inference: `"question <answer>"` (model completes)

## Verification
After retraining your model in Colab, you should see:
- `answer_rate` > 0.0 (model generates valid numeric answers)
- `accuracy` > 0.0 (model gets some questions correct)

## Why FP32?
- **FP16 training** can cause numerical instability with gradients
- **FP32 training** with bf16/fp16 mixed precision is more stable
- The code uses bf16 for forward/backward passes but maintains FP32 base weights
- This provides the best balance of memory efficiency and numerical stability

## What to Do Next in Colab
1. Delete your existing `homework/sft_model/` directory (or trained weights)
2. Run: `python -m homework.sft train`
3. The model will now train and test correctly
4. You should see non-zero accuracy and answer_rate

## Technical Details
The `use_fp32_for_training=True` flag:
- Sets `torch_dtype=torch.float32` when loading the base model
- Ensures LoRA weights and base model weights have matching dtypes
- Prevents dtype casting issues during inference
- Works correctly with mixed precision training (bf16/fp16 for computations)
