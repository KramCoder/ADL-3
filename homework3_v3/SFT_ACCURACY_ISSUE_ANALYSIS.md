# SFT Training Accuracy Issue - Root Cause Analysis

## Problem
Training completes successfully with decreasing loss (1.8395 → 1.021), but test accuracy is 0.0% and answer_rate is 0.0%.

## Root Cause

### Training vs Inference Format Mismatch

**During Training:**
- Input format: `question <answer>value</answer><eos>`
- Example: `"Can you change 2 hour to its equivalent in min? <answer>120.0</answer><|endoftext|>"`
- The model learns to produce `<answer>` tags because it sees them in the training data

**During Inference:**
- Input format: `question` (raw question only)
- Example: `"Can you change 2 hour to its equivalent in min?"`
- **Problem:** The model receives NO instruction to wrap its output in `<answer>` tags!

### Why This Causes 0.0 Accuracy

1. The model generates text, but likely without `<answer>` tags
2. `parse_answer()` in `BaseLLM` looks for `<answer>...</answer>` tags
3. If tags are missing, `parse_answer()` returns `float("nan")`
4. All answers become NaN → 0.0 accuracy and 0.0 answer_rate

### The Missing Link: `format_prompt()`

```python
def format_prompt(self, question: str) -> str:
    """
    Take a question and convert it into an input to SmolLM2.
    """
    return question  # <-- PROBLEM: Just returns raw question!
```

The model needs to be told during inference to:
1. Answer the math question
2. Wrap the answer in `<answer>` tags
3. Format it exactly as seen during training

## Solution

Modify `format_prompt()` to include clear instructions that match the training format:

```python
def format_prompt(self, question: str) -> str:
    """
    Format the question with instructions to produce <answer> tags.
    This ensures inference matches the training format.
    """
    return f"{question} <answer>"
```

Or with more explicit instructions:

```python
def format_prompt(self, question: str) -> str:
    """
    Format question with instruction to wrap answer in tags.
    """
    return f"{question.strip()}\n\nProvide your numeric answer in <answer></answer> tags. <answer>"
```

## Why Training Still Worked

The loss decreased because:
1. The model learned the token patterns
2. Cross-entropy loss measures next-token prediction
3. The model got better at predicting answer tokens given question+answer prefix
4. But without proper prompting, it can't reproduce this behavior from just the question

## Fix Implementation

Update `base_llm.py`:
- Modify `format_prompt()` to prepend instructions or suffix `<answer>` tag
- This makes inference context match training context
- Model will then complete with the numeric value and closing tag
