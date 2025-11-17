# SFT Model Diagnostic Guide

Your model shows `accuracy=0.0` and `answer_rate=0.0`, which indicates the model is either:
1. Not generating valid outputs (all NaN)
2. Generating outputs that don't match the expected format
3. Not loading the trained adapter correctly

## Quick Diagnostic

Run this script to gather diagnostic information:

```bash
cd /workspace/homework3_v3
python diagnose_sft_model.py
```

This will show you:
- Whether adapter files exist
- What the model is actually generating
- How inputs are being tokenized
- Format mismatches between training and inference

## What Information Would Help Diagnose?

### 1. **Model Generation Samples** (Most Important)

What is the model actually outputting? Run:

```python
from homework.sft import load
from homework.data import Dataset

llm = load()
testset = Dataset("valid")

# Check first few generations
for i in range(5):
    question = testset[i][0]
    expected = testset[i][1]
    
    # Get raw generation
    output = llm.generate(question)
    parsed = llm.parse_answer(output)
    
    print(f"Q: {question}")
    print(f"Expected: {expected}")
    print(f"Raw output: '{output}'")
    print(f"Parsed: {parsed}")
    print()
```

**What to look for:**
- Is `output` empty?
- Does it contain `<answer>` tags?
- Is `parsed` always NaN?

### 2. **Model Files Check**

Check if adapter files exist:

```bash
ls -lh homework/sft_model/
```

**What to look for:**
- `adapter_model.bin` or `adapter_model.safetensors` (should be > 0 bytes)
- `adapter_config.json` (should exist)

If these files are missing or 0 bytes, the model wasn't saved properly.

### 3. **Training Logs**

Check your training output for:
- Final loss value (should be decreasing, typically 0.5-2.0)
- Any warnings about NaN gradients
- Any warnings about all labels being masked
- Number of trainable parameters

**What to look for:**
- Loss that's too low (< 0.1) might indicate all labels were masked
- Loss that's too high (> 5.0) might indicate training didn't work
- NaN gradients mean the model didn't learn

### 4. **Tokenization Check**

Verify that labels aren't all masked:

```python
from homework.sft import tokenize, format_example
from homework.data import Dataset
from homework.base_llm import BaseLLM

llm = BaseLLM()
train_dataset = Dataset("train")

# Check a sample
formatted = format_example(*train_dataset[0])
tokens = tokenize(llm.tokenizer, formatted['question'], formatted['answer'])

non_masked = sum(1 for l in tokens['labels'] if l != -100)
print(f"Non-masked labels: {non_masked} out of {len(tokens['labels'])}")
```

**What to look for:**
- Should have > 0 non-masked labels
- If all labels are masked (-100), the model can't learn

### 5. **Format Mismatch Check**

Verify training format matches inference format:

```python
from homework.sft import format_example
from homework.base_llm import BaseLLM
from homework.data import Dataset

llm = BaseLLM()
testset = Dataset("valid")

question = testset[0][0]
answer = testset[0][1]

# Training format
train_format = format_example(question, answer)
print(f"Training: '{train_format['question']} {train_format['answer']}'")

# Inference format  
inference_format = llm.format_prompt(question)
print(f"Inference: '{inference_format}'")
```

**What to look for:**
- Training format should be: `"question <answer>value</answer>"`
- Inference format should be: `"question <answer>"`
- They should match up to the `<answer>` tag

## Common Issues and What They Mean

### Issue: `answer_rate=0.0` (all answers are NaN)

**Possible causes:**
1. Model generates empty strings
2. Model doesn't generate `<answer>` tags
3. Parser can't find the answer in the output

**Diagnostic:** Check raw model outputs (see #1 above)

### Issue: `accuracy=0.0` but `answer_rate>0.0`

**Possible causes:**
1. Model generates valid numbers but they're all wrong
2. Training didn't work (loss didn't decrease)
3. Format mismatch between training and inference

**Diagnostic:** Check training logs and format comparison

### Issue: Model files missing

**Possible causes:**
1. Training crashed before saving
2. Model saved to wrong location
3. Files were deleted

**Diagnostic:** Check if adapter files exist (see #2 above)

### Issue: All labels masked during training

**Possible causes:**
1. Tokenization bug - question tokens don't match
2. Answer is too short or gets truncated
3. Padding masks all labels

**Diagnostic:** Check tokenization (see #4 above)

## What to Share

If you want help diagnosing, share:

1. **Output of `diagnose_sft_model.py`** (most helpful!)
2. **Training logs** (especially final loss and any warnings)
3. **Sample raw generations** (what `llm.generate()` returns)
4. **Model file sizes** (output of `ls -lh homework/sft_model/`)

## Quick Fixes to Try

1. **Re-run training** - Sometimes training just needs to be run again
2. **Check model files** - Make sure adapter files exist and aren't empty
3. **Verify format** - Make sure `format_prompt()` matches training format
4. **Check tokenization** - Ensure labels aren't all masked

Run the diagnostic script first - it will identify the most likely issue!
