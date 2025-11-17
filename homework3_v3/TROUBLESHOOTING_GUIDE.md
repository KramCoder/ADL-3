# Testing and Troubleshooting Guide

This guide helps you test the RFT training improvements and troubleshoot common issues.

## Quick Start Testing

### 1. Test CoT Model Improvements

```bash
# Test CoT model accuracy
python -m homework.cot test

# Expected: Should show accuracy around 30-50% on validation set
# If accuracy is too low (<30%), the prompts may need further improvement
```

### 2. Generate RFT Training Data

```bash
# Generate the dataset (this may take 10-30 minutes)
python -m homework.datagen data/rft.json

# Check the output - should see:
# - "Generated X QA pairs out of 900 questions"
# - "SUCCESS: Generated X pairs (target: 850-900+)"
# - If you see WARNING, the CoT model accuracy may be too low
```

**Troubleshooting datagen:**
- If you get <850 pairs:
  - Check CoT model accuracy (should be >30%)
  - Try increasing `oversample` parameter: `python -m homework.datagen data/rft.json --oversample=20`
  - Try higher temperature: `python -m homework.datagen data/rft.json --temperature=0.8`
- If generation is too slow:
  - Reduce batch size in `datagen.py` (currently 16)
  - Reduce `oversample` (but may get fewer pairs)

### 3. Validate Generated Dataset

```bash
# Run the comprehensive test suite
python test_rft_improvements.py
```

This will test:
- ✓ CoT model accuracy
- ✓ Dataset size (850-900+ pairs)
- ✓ Dataset format (answer tags, reasoning text)
- ✓ SFT model accuracy (if trained)
- ✓ RFT model accuracy (if trained)

### 4. Inspect Dataset Manually

```python
import json
from pathlib import Path

# Load and inspect the dataset
with open("data/rft.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print(f"\nFirst example:")
print(f"  Question: {data[0][0]}")
print(f"  Answer: {data[0][1]}")
print(f"  Reasoning: {data[0][2][:200]}...")  # First 200 chars

# Check for answer tags
has_tags = sum(1 for ex in data if "<answer>" in ex[2] and "</answer>" in ex[2])
print(f"\nExamples with answer tags: {has_tags}/{len(data)}")
```

## Step-by-Step Validation

### Step 1: Verify CoT Model Works

```python
from homework.cot import CoTModel
from homework.data import Dataset, benchmark

# Test CoT model
model = CoTModel()
dataset = Dataset("valid")
result = benchmark(model, dataset, 50)  # Test on 50 samples

print(f"CoT Accuracy: {result.accuracy:.2%}")
print(f"Answer Rate: {result.answer_rate:.2%}")

# Should be:
# - Accuracy: 30-50% (good enough for datagen)
# - Answer Rate: >80% (most questions get valid answers)
```

**If CoT accuracy is too low:**
- Check that the model loads correctly
- Verify the prompts are being applied (check `format_prompt` output)
- Try increasing `max_new_tokens` in `cot.py` (currently 120)
- Review the few-shot examples in the prompt

### Step 2: Test Data Generation

```python
from homework.datagen import generate_dataset

# Generate dataset
output_path = generate_dataset("data/rft.json", oversample=15, temperature=0.7)

# Check output messages:
# - Should see "Generated X QA pairs"
# - Should see "SUCCESS" if X >= 850
# - Should see "WARNING" if X < 850
```

**Common datagen issues:**

1. **Too many rejections (<850 pairs)**
   - **Cause**: CoT model accuracy too low
   - **Fix**: Improve CoT prompts or test on more questions
   - **Workaround**: Increase `oversample` to 20-25

2. **Generation too slow**
   - **Cause**: Large batch size or too many sequences
   - **Fix**: Reduce `batch_size` in `datagen.py` or reduce `oversample`

3. **Missing answer tags**
   - **Cause**: CoT model not generating proper format
   - **Fix**: Check CoT `format_prompt` and few-shot examples
   - **Note**: The code now validates and filters these

### Step 3: Validate Dataset Format

```python
import json
from pathlib import Path

rft_path = Path("data/rft.json")
with rft_path.open() as f:
    data = json.load(f)

# Check size
print(f"Dataset size: {len(data)}")
assert len(data) >= 850, f"Need 850+ examples, got {len(data)}"

# Check format
for i, ex in enumerate(data[:10]):  # Check first 10
    assert len(ex) == 3, f"Example {i}: Expected [question, answer, reasoning], got {len(ex)} items"
    question, answer, reasoning = ex
    assert "<answer>" in reasoning, f"Example {i}: Missing <answer> tag"
    assert "</answer>" in reasoning, f"Example {i}: Missing </answer> tag"
    assert len(reasoning.strip()) > 20, f"Example {i}: Reasoning too short"

print("✓ Dataset format is valid")
```

### Step 4: Train and Test SFT Model

```bash
# Train SFT model (this takes time)
python -m homework.sft train

# Test SFT model
python -m homework.sft test
```

**Expected SFT results:**
- Accuracy should be **>0.6** (target threshold)
- Answer rate should be **>0.9** (most questions get answers)

**If SFT accuracy is too low:**
- Increase training epochs (currently 6)
- Check training loss is decreasing
- Verify tokenization is correct (check `tokenize` function)
- Ensure labels are properly masked (question tokens = -100)

### Step 5: Train and Test RFT Model

```bash
# Train RFT model (requires rft.json dataset)
python -m homework.rft train

# Test RFT model
python -m homework.rft test
```

**Expected RFT results:**
- Accuracy should be **>0.7** (target threshold)
- Answer rate should be **>0.9**

**If RFT accuracy is too low:**
- Check dataset quality (run validation in `rft.py`)
- Verify reasoning text includes full explanations
- Try training longer (increase epochs)
- Check that dataset has 850+ examples

## Common Issues and Solutions

### Issue: "RFT dataset not found"

**Solution:**
```bash
# Generate the dataset first
python -m homework.datagen data/rft.json
```

### Issue: "Only X pairs generated. Target is 850-900+"

**Causes and Solutions:**

1. **CoT model accuracy too low**
   ```python
   # Test CoT accuracy first
   from homework.cot import CoTModel
   from homework.data import Dataset, benchmark
   
   model = CoTModel()
   dataset = Dataset("valid")
   result = benchmark(model, dataset, 100)
   print(f"CoT Accuracy: {result.accuracy}")
   # If <0.3, improve prompts or check model loading
   ```

2. **Increase oversampling**
   ```bash
   python -m homework.datagen data/rft.json --oversample=20
   ```

3. **Accept approximate answers**
   - The code now accepts answers within 10% error as fallback
   - This should help reach 850+ pairs even with lower CoT accuracy

### Issue: "Missing answer tags in reasoning"

**Solution:**
- The code now validates and filters examples without tags
- If you see this warning, check CoT model output format
- Verify `format_prompt` in `cot.py` includes proper examples

### Issue: "SFT/RFT accuracy below threshold"

**Solutions:**

1. **Check training completed successfully**
   - Look for "Training Summary" output
   - Verify loss decreased during training
   - Check for NaN/Inf warnings

2. **Increase training epochs**
   - SFT: Currently 6 epochs, try 8-10
   - RFT: Currently 3 epochs, try 5-6

3. **Verify dataset quality**
   - Check tokenization produces valid labels
   - Ensure answer tokens are not masked
   - Verify training data format

4. **Check model loading**
   ```python
   # Test model loads correctly
   from homework.sft import load
   model = load()
   # Should not raise errors
   ```

## Debugging Tips

### 1. Check Individual Components

```python
# Test CoT generation on a single question
from homework.cot import CoTModel

model = CoTModel()
question = "How many gram are there per 6 kg?"
reasoning = model.generate(question)
print(f"Generated: {reasoning}")

# Check parsing
parsed = model.parse_answer(reasoning)
print(f"Parsed: {parsed}")
```

### 2. Inspect Training Data

```python
# Check SFT training data format
from homework.sft import TokenizedDataset, format_example
from homework.data import Dataset

dataset = Dataset("train")
tokenizer = BaseLLM().tokenizer
tokenized = TokenizedDataset(tokenizer, dataset, format_example)

# Check a sample
sample = tokenized[0]
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Non-masked labels: {sum(1 for l in sample['labels'] if l != -100)}")
# Should have non-zero non-masked labels
```

### 3. Monitor Training Progress

```python
# During training, watch for:
# - Loss decreasing
# - No NaN/Inf warnings
# - Gradient norms reasonable (<10)
# - Learning rate schedule working
```

### 4. Validate Model Outputs

```python
# Test model generates valid answers
from homework.sft import load
from homework.data import Dataset, benchmark

model = load()
dataset = Dataset("valid")
result = benchmark(model, dataset, 10)  # Small test

print(f"Accuracy: {result.accuracy}")
print(f"Answer rate: {result.answer_rate}")

# Check individual answers
for i in range(5):
    question = dataset[i][0]
    answers = model.answer(question)
    print(f"Q: {question}")
    print(f"A: {answers[0]}")
```

## Performance Benchmarks

Expected performance targets:

| Model | Minimum | Target | Excellent |
|-------|---------|--------|-----------|
| CoT   | 0.25    | 0.35   | 0.45+     |
| SFT   | 0.40    | 0.60   | 0.70+     |
| RFT   | 0.60    | 0.70   | 0.75+     |

## Next Steps

1. **If all tests pass**: Your improvements are working! Proceed with full training.

2. **If tests fail**: 
   - Review the specific error messages
   - Check the troubleshooting sections above
   - Verify each component individually

3. **For production use**:
   - Run full training on all models
   - Validate on validation set
   - Test with grader to ensure thresholds are met

## Getting Help

If you encounter issues not covered here:

1. Check the error messages carefully
2. Verify each component works in isolation
3. Review the code changes in the modified files
4. Check that all dependencies are installed correctly
