# SFT Training Fix - Format Prompt Alignment

## Problem Identified

Your SFT training was completing successfully with decreasing loss, but accuracy was 0.0% because of a **training-inference format mismatch**.

### What Was Happening

**Training:**
- Model sees: `"question <answer>value</answer>"`
- Model learns to predict answer tokens when it sees this format

**Inference (Before Fix):**
- Model receives: `"question"` (raw question only)
- Model generates text but without `<answer>` tags
- `parse_answer()` returns NaN because it can't find the tags
- Result: 0.0% accuracy, 0.0% answer rate

## Fix Applied

Modified `format_prompt()` in `homework/base_llm.py`:

```python
def format_prompt(self, question: str) -> str:
    """
    For SFT training, we need to match the format seen during training:
    Training format: "question <answer>value</answer>"
    Inference format: "question <answer>" (model completes the rest)
    """
    return f"{question.strip()} <answer>"
```

### Why This Works

1. **Training sees:** `"question <answer>value</answer>"`
2. **Inference sends:** `"question <answer>"` 
3. **Model completes:** `"value</answer>"`
4. **Parser extracts:** `value` from `<answer>value</answer>`
5. **Result:** Correct numeric answers!

The model now receives the same context at inference time that it saw during training, so it knows to complete the answer in the expected format.

## How to Test

### Option 1: Re-test the existing trained model
```bash
cd /workspace/homework3_v3
python3 -m homework.sft test /tmp/sft_output
```

If you already have a trained model, this will now use the fixed `format_prompt()` and should show improved accuracy.

### Option 2: Train a new model from scratch
```bash
cd /workspace/homework3_v3
python3 -m homework.sft train --output_dir /tmp/sft_output_fixed
```

This will train with the same format, and now testing will work correctly.

### Option 3: Quick validation
```python
from homework.base_llm import BaseLLM
from homework.sft import load
from homework.data import Dataset, benchmark

# Test with a few samples
testset = Dataset("valid")
llm = load()  # Load SFT model
result = benchmark(llm, testset, 10)
print(f"Accuracy: {result.accuracy}")
print(f"Answer rate: {result.answer_rate}")
```

## Expected Results

After this fix:
- **Answer rate:** Should be close to 100% (model generates parseable answers)
- **Accuracy:** Depends on training quality, but should be > 0% (likely 20-80%)
- **Generations:** Should include `<answer>...</answer>` tags

## Additional Notes

### Why Loss Was Decreasing But Accuracy Was 0%

The loss measures how well the model predicts the next token given the training context. Since the training data includes both question and answer, the model was learning successfully. However, at inference time, without the proper prompt format, the model didn't know to produce the answer format.

Think of it like:
- **Training:** Teacher shows you: "2+2= 4"
- **Test (before fix):** "2+2" → You say "four" (correct but wrong format)
- **Test (after fix):** "2+2= " → You say "4" (correct format!)

### This Is a Common Issue

This type of format mismatch is one of the most common issues in LLM fine-tuning:
1. Training format includes structure (prompts, tags, etc.)
2. Inference format doesn't match
3. Model generates correct content but wrong format
4. Parser fails → 0% accuracy

The fix is always: **Make inference format match training format as closely as possible**.
