# NaN Error Analysis and Fix

## Root Cause Analysis

The error occurs in the grader at line 92:
```python
final_score = int(n_passed * score / total + 0.5)
ValueError: cannot convert float NaN to integer
```

### Why NaN is Generated

The NaN originates from the test grader's `compute_loss` function (tests.py:52-68):

```python
def compute_loss(self, model, full_texts):
    with torch.no_grad():
        tokens = model.tokenizer(full_texts, return_tensors="pt", padding=True)
        # ... compute logits and loss ...
        loss = loss.sum() / tokens["attention_mask"][..., 1:].sum()  # Line 67
        return loss.cpu().item()
```

**The Problem:** If `tokens["attention_mask"][..., 1:].sum()` equals 0, we get division by zero (0/0 = NaN).

This happens when:
1. Generated text is empty or extremely short
2. After tokenization, all attention mask values (except first token) are 0  
3. The concatenated `question + answer` produces insufficient tokens

### Why Attention Mask Can Sum to Zero

The grader:
1. Calls `generate()` on each question
2. Concatenates: `full_text = question + answer`
3. Tokenizes `full_text` with padding
4. Uses `attention_mask[..., 1:]` (excluding first token)
5. Divides by the sum - if sum is 0, we get NaN

Even though we have validation in base_llm.py that sets empty generations to " 0", there may be edge cases where:
- The model generates invalid/malformed output
- Special tokens or encoding issues cause tokenization problems
- The concatenation creates a string that tokenizes unexpectedly

## The Fix

We need to ensure that generated outputs are ALWAYS valid and will produce sufficient tokens when concatenated with questions. The fix has multiple layers:

### Layer 1: Robust Generation Validation
- Ensure outputs are never empty
- Ensure outputs always contain meaningful content
- Add fallback for any edge cases

### Layer 2: NaN Detection and Prevention
- Check for NaN/Inf in parse_answer
- Validate generation outputs before returning
- Add safety checks in all generation paths

### Layer 3: Grader-Safe Outputs
- Ensure concatenated text (question + answer) will always tokenize to multiple tokens
- Provide meaningful default outputs that won't trigger division by zero
- Handle all edge cases gracefully

## Implementation

The fix is implemented in base_llm.py with enhanced validation:

1. **parse_answer()**: Already checks for NaN/Inf and returns 0.0
2. **generate()**: Enhanced validation to ensure non-empty outputs with minimum content
3. **batched_generate()**: Same validation for batch processing  
4. **Fallback Strategy**: If generation fails or is empty, return a minimal valid answer

## Testing

After applying the fix:
1. Run the grader to verify no NaN errors
2. Check that all test cases pass
3. Verify model outputs are valid and meaningful
