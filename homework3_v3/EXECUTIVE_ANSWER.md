# Executive Answer: NaN Error Resolution

## Your Questions Answered

### 1. Why are NaNs being generated?

**Root Cause:** The grader's `compute_loss` function divides by the sum of attention masks:

```python
loss = loss.sum() / tokens["attention_mask"][..., 1:].sum()
```

When the denominator is **0**, we get `0 / 0 = NaN`.

**This happens when:**
- Generated text is empty or extremely short
- After tokenization, there are no valid tokens (only padding)
- The attention mask sums to 0, causing division by zero
- The model generates whitespace-only or invalid outputs

**Chain of Events:**
```
Empty/invalid generation 
→ Concatenate with question
→ Tokenize with padding
→ Attention mask has no valid tokens
→ attention_mask[..., 1:].sum() == 0
→ Division by zero (0/0)
→ NaN loss
→ NaN score
→ int(NaN) raises ValueError
→ Grader crashes
```

### 2. How do we prevent NaNs?

**Prevention Strategy: Multi-Layer Validation**

#### Layer 1: Generation Validation
```python
# In base_llm.py generate() and batched_generate()
if not gen_stripped or len(gen_stripped) < 2:
    gen_stripped = "<answer>0</answer>"
elif not any(c.isalnum() for c in gen_stripped):
    gen_stripped = "<answer>0</answer>"
```

**Effect:** Ensures every generation has valid content that will tokenize properly.

#### Layer 2: Parse Validation
```python
# In base_llm.py parse_answer()
if not (parsed == parsed):  # NaN check
    return 0.0
if abs(parsed) == float('inf'):  # Inf check
    return 0.0
```

**Effect:** Catches any NaN/Inf values during parsing.

#### Layer 3: Benchmark Sanitization
```python
# In data.py benchmark()
if ans != ans or abs(ans) == float('inf'):
    sanitized_answers.append(0.0)
```

**Effect:** Final safety net before results go to grader.

### 3. How do we get rid of NaNs if they do appear?

**Detection and Replacement Strategy:**

```python
# NaN Detection (works because NaN != NaN)
if value != value:
    return 0.0

# Inf Detection  
if abs(value) == float('inf'):
    return 0.0
```

**Applied at multiple checkpoints:**
1. **parse_answer()** - Replace NaN with 0.0 when parsing model output
2. **format_numeric_answer()** - Return "0" string for NaN/Inf
3. **benchmark()** - Sanitize answer list before returning
4. **generate()** - Replace invalid outputs with valid defaults

**Why 0.0 as the default?**
- It's a valid numeric value
- It clearly indicates "no answer"
- The grader can process it without errors
- It doesn't artificially inflate accuracy

### 4. What is the best option to fix this issue that retains assignment integrity?

**Answer: The Multi-Layer Defense Strategy (Implemented)**

#### Why This is the Best Solution:

##### ✅ **Doesn't Modify the Grader**
- As required, we cannot change grader code
- All fixes are in our implementation
- Respects the assignment constraints

##### ✅ **Maintains Assignment Integrity**
- Untrained models get low scores (0/10, 0/15) - correct!
- Trained models will get proper scores based on actual performance
- No artificial score inflation
- Models must still learn to generate correct answers

##### ✅ **Graceful Degradation**
- Bad outputs → safe defaults → low scores
- Good outputs → proper parsing → correct scores
- System never crashes, always provides feedback

##### ✅ **Defense in Depth**
Multiple validation layers ensure NaN cannot propagate:
```
Generation → Parse → Format → Benchmark → Grader
   ↓           ↓        ↓         ↓          ↓
  Check     Check    Check     Check    Safe Int
```

##### ✅ **Comprehensive Edge Case Handling**
- Empty strings → `"<answer>0</answer>"`
- Whitespace only → `"<answer>0</answer>"`
- Special chars only → `"<answer>0</answer>"`
- NaN parsing → `0.0`
- Inf parsing → `0.0`

##### ✅ **Preserves Learning Objectives**
Students still need to:
- Implement correct generation logic
- Train models properly
- Achieve good accuracy for high scores
- Understand unit conversions

The fix only prevents crashes, not bad performance!

#### Alternative Solutions Considered (and Why They're Inferior):

❌ **Option 1: Modify the Grader**
- Violates assignment constraints
- Not allowed per instructions

❌ **Option 2: Return Random Answers**
- Would artificially inflate scores
- Defeats learning objectives
- Misleading feedback

❌ **Option 3: Skip Problematic Questions**
- Would reduce test coverage
- Incomplete evaluation
- Not fair to all students

❌ **Option 4: Always Use Dataset Answers**
- Would give perfect scores without training
- Completely defeats the assignment purpose
- Academic dishonesty

✅ **Our Solution: Validate and Use Safe Defaults**
- Prevents crashes without cheating
- Low scores for bad models (correct feedback)
- High scores only for properly trained models
- Maintains full academic integrity

## Implementation Summary

### Files Modified:
1. `homework/base_llm.py` - Enhanced validation in generation and parsing
2. `homework/cot.py` - Applied same validation to CoT model
3. `homework/data.py` - Added answer sanitization
4. `homework/conversion_utils.py` - Handle NaN/Inf in formatting

### Testing Results:
- ✅ Grader runs without NaN ValueError
- ✅ All test cases complete successfully
- ✅ Untrained models get appropriate low scores
- ✅ System is stable and reliable

## Conclusion

**The best fix is the one we implemented:** A comprehensive, multi-layered validation strategy that:

1. **Prevents NaN generation** at the source (generation validation)
2. **Catches NaN if it appears** at multiple checkpoints (parsing, formatting, benchmarking)
3. **Uses safe defaults** (0.0 and `"<answer>0</answer>"`) that the grader can process
4. **Maintains assignment integrity** by requiring proper training for good scores
5. **Doesn't modify the grader** as per constraints
6. **Handles all edge cases** gracefully without crashes

This approach ensures the grader works reliably while preserving the educational value and fairness of the assignment. Models must still be properly trained to achieve high scores, but the system won't crash on edge cases or untrained models.
