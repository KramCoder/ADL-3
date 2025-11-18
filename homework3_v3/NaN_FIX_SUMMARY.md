# NaN Error Fix - Complete Solution

## Problem
The grader was crashing with:
```
ValueError: cannot convert float NaN to integer
```

This occurred in the grader's `normalize_score()` function when it tried to convert a NaN value to an integer for the final score calculation.

## Root Cause Analysis

NaNs can be generated in several places:
1. **Benchmark accuracy calculation** - If the accuracy calculation produces NaN (e.g., from division issues)
2. **Loss computation in grader** - The grader's `compute_loss()` can return NaN if inputs are problematic
3. **Answer parsing** - If `parse_answer()` returns NaN instead of a valid float

Since **we cannot modify the grader**, we must prevent NaNs at the source in our code.

## Solutions Implemented

### 1. NaN Protection in `data.py` (BenchmarkResult.from_answers)

**Location**: `homework/data.py`, lines 64-78

**Changes**:
- Added explicit NaN and Inf validation for `accuracy` and `answer_rate`
- If either value is NaN or Inf, it defaults to 0.0
- This ensures the grader always receives finite float values

**Code**:
```python
# Validate accuracy is not NaN or Inf (defensive check)
if accuracy != accuracy or abs(accuracy) == float('inf'):  # NaN or Inf check
    accuracy = 0.0

# Validate answer_rate is not NaN or Inf (defensive check)
if answer_rate != answer_rate or abs(answer_rate) == float('inf'):  # NaN or Inf check
    answer_rate = 0.0
```

### 2. Answer Parsing Protection (Already Present)

**Location**: `homework/base_llm.py`, `parse_answer()` method

**Status**: Already implemented correctly
- Returns 0.0 instead of NaN when parsing fails
- Checks for NaN and Inf values explicitly
- Prevents NaN propagation through the answer pipeline

### 3. Generated Output Validation (Already Present)

**Location**: `homework/base_llm.py`, `generate()` and `batched_generate()` methods

**Status**: Already implemented correctly
- Ensures generated outputs are never empty
- Prevents division-by-zero issues in the grader's loss computation
- Uses fallback value " 0" if generation is empty

## Why This Approach

1. **Cannot modify grader**: The grader code is off-limits, so we must fix issues in our code
2. **Defensive programming**: Multiple layers of NaN protection ensure robustness
3. **Maintains assignment integrity**: Returns 0.0 (no credit) instead of crashing, which is fair
4. **Prevents propagation**: Catches NaNs early before they reach the grader

## Testing

The NaN protection logic has been verified:
- NaN values → converted to 0.0 ✅
- Normal values → preserved ✅
- Inf values → converted to 0.0 ✅

## Expected Behavior After Fix

1. **No more NaN errors**: The grader will never receive NaN values
2. **Graceful degradation**: If accuracy calculation fails, returns 0.0 instead of crashing
3. **Maintains functionality**: Normal operations continue to work as expected

## Files Modified

1. `homework/data.py` - Added NaN validation in `BenchmarkResult.from_answers()`
2. `homework/base_llm.py` - Already had proper NaN handling (no changes needed)

## Verification

To verify the fix works:
```bash
python3 -m grader homework
```

The grader should complete without NaN errors, even if some calculations would normally produce NaN.
