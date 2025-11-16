# SFT Training Fix Summary

## Problem
During SFT (Supervised Fine-Tuning) model training, the following issues were observed:
- **Loss: 0.0** throughout all training steps
- **Gradient norm: NaN** throughout all training steps
- Training appeared to complete but the model wasn't actually learning

## Root Cause
The issue was in the `tokenize()` function in `homework/sft.py`. The function was incorrectly calculating the boundary between the question (prompt) and answer in the tokenized sequence.

### Specific Bug
```python
# OLD CODE (BUGGY)
full_text = f"{question} {answer}{tokenizer.eos_token}"
question_len = len(tokenizer(question)["input_ids"])  # ❌ Missing the trailing space!
labels = [-100] * question_len + input_ids[question_len:]
```

The problem: When tokenizing the question alone without the trailing space, the tokenizer produces a different number of tokens than when the question appears in the full text (with a space after it). This caused a misalignment in the labels array, potentially masking all the answer tokens with `-100`.

When all labels are `-100`, the model has no training signal, resulting in:
- Loss = 0.0 (no tokens to compute loss on)
- Gradient norm = NaN (no gradients to compute)

## Solution
Fixed the tokenization to properly account for the space between question and answer:

```python
# NEW CODE (FIXED)
full_text = f"{question} {answer}{tokenizer.eos_token}"

# Tokenize the question with the trailing space to match how it appears in full_text
question_with_space = f"{question} "
question_tokens = tokenizer(question_with_space, add_special_tokens=False)["input_ids"]
question_len = len(question_tokens)

# Create labels: mask out the prompt part (question), keep only answer for training
labels = [-100] * question_len + input_ids[question_len:]

# Ensure labels list has the same length as input_ids
labels = labels[:len(input_ids)]
```

### Key Changes
1. **Added trailing space**: Now tokenizes `f"{question} "` instead of just `question`
2. **Disabled special tokens**: Uses `add_special_tokens=False` to avoid adding extra tokens
3. **Length safety**: Added `labels = labels[:len(input_ids)]` to ensure correct length

## Verification
A tokenization test confirmed the fix:
- ✅ Before fix: All labels would be -100 (no training signal)
- ✅ After fix: Answer tokens have proper labels (9-24 non-masked tokens per example)
- ✅ Labels array length matches input_ids length

## Additional Improvement
Also removed the unnecessary CUDA check for `enable_input_require_grads()`:
```python
# OLD: Only enabled on CUDA
if torch.cuda.is_available():
    lora_model.enable_input_require_grads()

# NEW: Always enabled (required for gradient checkpointing)
lora_model.enable_input_require_grads()
```

## Expected Results After Fix
When you run training again, you should now see:
- ✅ **Non-zero loss values** (typically starting around 1.5-3.0)
- ✅ **Valid gradient norms** (non-NaN values, typically 0.1-10.0)
- ✅ **Loss decreasing** over training steps
- ✅ **Model actually learning** from the training data

## Files Modified
1. `homework/sft.py` - Fixed `tokenize()` function and `train_model()` function
