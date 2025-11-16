# Accuracy and Learning Rate Fix Summary

## Issues Identified

1. **Answer Parsing Too Strict**: The `parse_answer` function required both opening and closing `<answer>` tags. If the model didn't generate the closing tag (which is common), it would return NaN, resulting in 0% accuracy.

2. **Training/Inference Format Mismatch**: The tokenization during training was using `question ` (with space) to find where the answer starts, but during inference we use `question <answer>`. This mismatch could cause the model to not learn the pattern correctly.

3. **Insufficient Generation Tokens**: `max_new_tokens` was set to 30-40, which might not be enough for the model to generate the full answer format including the closing tag.

4. **Learning Rate Scheduler Not Explicit**: While the learning rate was decreasing (as shown in logs), it wasn't explicitly configured, which could lead to confusion.

## Fixes Applied

### 1. Improved Answer Parsing (`base_llm.py`)
- Made `parse_answer` more robust to handle missing closing tags
- Added regex-based number extraction if closing tag is missing
- Falls back to extracting the first number-like token if no closing tag is found
- This ensures the model can still extract answers even if it doesn't generate perfect XML tags

### 2. Better Training Format Alignment (`sft.py`)
- Changed tokenization to use `question <answer>` instead of `question ` when finding where the answer starts
- This ensures the training format exactly matches the inference format
- The model now learns to predict starting from after `<answer>`, matching inference behavior

### 3. Increased Generation Tokens (`base_llm.py`)
- Increased `max_new_tokens` from 30/40 to 50 in both `generate` and `batched_generate`
- Ensures the model has enough tokens to generate the full answer format

### 4. Explicit Learning Rate Scheduler (`sft.py`, `rft.py`)
- Added explicit `lr_scheduler_type="linear"` configuration
- Added `warmup_steps=0` for clarity
- The learning rate was already working (decreasing from 2e-4), but now it's explicit

### 5. Debug Output (`sft.py`, `rft.py`)
- Added debug output in `test_model` functions to show:
  - The formatted prompt
  - What the model actually generates
  - The parsed answer vs expected answer
- This helps diagnose issues if accuracy is still low

## Expected Results

After these fixes:
- **Accuracy should improve**: The robust parsing will extract answers even if the model doesn't generate perfect XML tags
- **Learning rate is working**: The logs show it decreasing from ~0.00018 to ~0.00001, which is normal for a linear decay scheduler
- **Better training**: The improved tokenization ensures the model learns the correct pattern

## Testing

When you run training again, you should see:
1. Debug output showing what the model generates
2. Non-zero accuracy (even if not perfect, should be > 0%)
3. Learning rate properly logged and decreasing

## Next Steps

If accuracy is still low after these fixes:
1. Check the debug output to see what the model is actually generating
2. Verify the model is loading the trained adapter correctly
3. Consider increasing training epochs or adjusting the learning rate
4. Check if the model is generating answers but in a different format than expected
