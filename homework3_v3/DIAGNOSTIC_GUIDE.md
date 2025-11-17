# SFT Model Diagnostic Guide

## Quick Start - Run This Now!

```bash
cd /workspace/homework3_v3
python3 diagnose_sft.py
```

**Share the ENTIRE output of this script** - it will tell us exactly what's wrong.

---

## What This Script Checks

The diagnostic script will check:

1. **Model Files** - Are the trained adapter files present?
2. **Model Loading** - Can the model load correctly?
3. **Prompt Formatting** - Is the prompt format correct for inference?
4. **Generation** - What is the model actually outputting?
5. **Tokenization** - Was the training data formatted correctly?

---

## What to Share for Diagnosis

### Option 1: Run the diagnostic (RECOMMENDED)

```bash
python3 diagnose_sft.py
```

Copy and paste the **entire output**.

### Option 2: Manual checks

If you can't run the script, please provide:

#### A. Model Files

```bash
# List what files exist in your model directory
ls -lh /content/ADL-3/homework3_v3/homework/sft_model/

# Or wherever your model was saved
ls -lh /content/ADL-3/homework3_v3/homework/homework/sft_output/
```

**Share:** The complete file listing

#### B. Model Generation Output

```python
# In Python:
from homework.sft import load
from homework.data import Dataset

# Load model
llm = load()

# Test on a few examples
testset = Dataset("valid")
for i in range(3):
    question, answer = testset[i]
    print(f"Question: {question}")
    print(f"Correct: {answer}")
    
    # See what the model generates
    raw_output = llm.generate(question)
    print(f"Model output: {raw_output!r}")
    
    # See what gets parsed
    parsed = llm.parse_answer(raw_output)
    print(f"Parsed: {parsed}")
    print("-" * 80)
```

**Share:** The output of this code

#### C. Training Log (last 20 lines)

```bash
# If you saved training output
tail -20 full_training_log.txt
```

**Share:** The last few lines showing final loss and test results

---

## Common Issues and Fixes

### Issue 1: "adapter_model.safetensors not found"

**Problem:** Model didn't save correctly or wrong directory

**Fix:** Check where training actually saved the model:
```bash
find /content -name "adapter_model.*" 2>/dev/null
```

Then test with the correct path:
```python
python3 -m homework.sft test /full/path/to/model
```

### Issue 2: Model outputs don't contain `<answer>` tags

**Problem:** Prompt formatting is wrong

**Fix:** Check `homework/base_llm.py` line ~72:
```python
def format_prompt(self, question: str) -> str:
    return f"{question.strip()} <answer>"  # Must end with <answer>
```

### Issue 3: Model outputs gibberish or random numbers

**Problem:** Model loaded untrained weights

**Diagnosis:** Check if adapter files are very small (<100KB) - they should be 2-8MB

**Fix:** Re-train the model

### Issue 4: All outputs are NaN

**Problem:** Model isn't generating the `<answer>...</answer>` format

**Fix:** 
1. Check prompt formatting (see Issue 2)
2. Verify training used the same format:
   ```python
   from homework.sft import format_example
   ex = format_example("test question", 42.0)
   print(ex)  # Should print: {'question': 'test question', 'answer': '<answer>42.0</answer>'}
   ```

---

## Understanding the Output

### Good Output Example:
```
--- Example 1 ---
Question: Can you change 2 hour to its equivalent in min?
Correct answer: 120.0
Raw output: '120.0</answer>'
Parsed value: 120.0
Valid answer: ✅
Correct: ✅
```

### Bad Output Example (untrained model):
```
--- Example 1 ---
Question: Can you change 2 hour to its equivalent in min?
Correct answer: 120.0
Raw output: 'I don't know'
Parsed value: nan
Valid answer: ❌ (NaN)
```

### Bad Output Example (wrong format):
```
--- Example 1 ---
Question: Can you change 2 hour to its equivalent in min?
Correct answer: 120.0
Raw output: 'The answer is 120.0 minutes'
Parsed value: nan
Valid answer: ❌ (NaN)
  ⚠️  Output does NOT contain '<answer>' tag
```

---

## Next Steps

1. **Run the diagnostic script** and share the output
2. Based on the output, we can identify:
   - If the model is actually trained
   - If the prompt format is correct
   - If generation parameters are correct
   - What the actual issue is

The diagnostic will give us a clear picture of what's happening!
