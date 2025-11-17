# Quick Test Commands Reference

## Testing Workflow

### 1. Test CoT Model (5 minutes)
```bash
python -m homework.cot test
```
**Expected**: Accuracy 30-50%, Answer rate >80%

### 2. Generate RFT Dataset (10-30 minutes)
```bash
python -m homework.datagen data/rft.json
```
**Expected**: "SUCCESS: Generated 850-900+ pairs"

### 3. Run Full Test Suite (5 minutes)
```bash
python test_rft_improvements.py
```
**Checks**: CoT accuracy, dataset quality, model accuracies

### 4. Train SFT Model (30-60 minutes)
```bash
python -m homework.sft train
python -m homework.sft test
```
**Expected**: Accuracy >0.6

### 5. Train RFT Model (30-60 minutes)
```bash
python -m homework.rft train
python -m homework.rft test
```
**Expected**: Accuracy >0.7

## Quick Diagnostics

### Check dataset size
```python
import json
with open("data/rft.json") as f:
    data = json.load(f)
print(f"Examples: {len(data)}")
```

### Check CoT accuracy
```python
from homework.cot import CoTModel
from homework.data import Dataset, benchmark
model = CoTModel()
result = benchmark(model, Dataset("valid"), 50)
print(f"Accuracy: {result.accuracy:.2%}")
```

### Check model loads
```python
from homework.sft import load
model = load()  # Should not error
```

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| <850 pairs generated | `python -m homework.datagen data/rft.json --oversample=20` |
| CoT accuracy too low | Check prompts in `cot.py`, increase `max_new_tokens` |
| SFT accuracy <0.6 | Increase epochs in `sft.py` (currently 6) |
| RFT accuracy <0.7 | Check dataset quality, increase training epochs |
| Missing answer tags | Regenerate dataset, check CoT output format |

## Validation Checklist

- [ ] CoT model accuracy >30%
- [ ] RFT dataset has 850+ examples
- [ ] All examples have `<answer>` tags
- [ ] SFT accuracy >0.6 (if trained)
- [ ] RFT accuracy >0.7 (if trained)
- [ ] No NaN/Inf errors during training
