# RFT Training Improvement Plan

## Current Status Analysis

### 1. Dataset Information
- **Training dataset**: 1,000 questions available
- **RFT dataset**: NOT YET GENERATED ❌
- **Datagen settings**: oversample=10, temperature=0.6

### 2. Accuracy Thresholds (from grader/tests.py)

| Model | Min Accuracy | Target Accuracy | Status |
|-------|-------------|-----------------|--------|
| CoT   | 0.0         | **0.4**         | To test |
| SFT   | 0.4         | **0.6**         | To test |
| RFT   | 0.6         | **0.7**         | To test |

### 3. Key Issues Identified

#### Issue 1: RFT Dataset Not Generated
The RFT training data hasn't been created yet. This is the first blocker.

#### Issue 2: CoT Model Accuracy Unknown
We need to test the CoT model accuracy to ensure it can generate quality data for RFT training.

#### Issue 3: Potential Data Rejection Rate
With 1,000 training questions and oversample=10:
- **Maximum possible**: 1,000 valid QA pairs (if CoT gets 1/10 correct per question)
- **Target**: 850-900 QA pairs
- **Rejection threshold**: If CoT accuracy < ~10%, we won't reach 850 pairs

## Action Plan

### Step 1: Test CoT Model Accuracy
```bash
cd /workspace/homework3_v3
python3 -m homework.cot test
```

**Expected outcome**: 
- If accuracy ≥ 0.4, CoT is working well
- If accuracy < 0.4, need to improve CoT before RFT

### Step 2: Generate RFT Dataset
```bash
cd /workspace/homework3_v3
python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6
```

**After generation, check**:
```bash
python3 -c "import json; data=json.load(open('data/rft.json')); print(f'Generated {len(data)} RFT examples')"
```

### Step 3: Inspect RFT Data Quality
```bash
# Check first few examples
python3 -c "import json; data=json.load(open('data/rft.json')); print('Sample:', data[0])"
```

**Verify**:
- ✅ Full reasoning text is present
- ✅ Both `<answer>` and `</answer>` tags exist
- ✅ Answer value matches expected format

### Step 4: Improve Data Generation If Needed

#### Option A: Increase Oversample (if < 850 examples)
Edit `datagen.py` or run with higher oversample:
```bash
python3 -m homework.datagen data/rft.json --oversample 15 --temperature 0.6
```

#### Option B: Adjust Temperature
- **Lower temperature** (0.4-0.5): More consistent but less diverse
- **Higher temperature** (0.7-0.8): More diverse but might be less accurate

#### Option C: Improve CoT Prompt
The CoT model uses few-shot prompting. Consider:
- Adding more examples
- Making examples more clear
- Adjusting system prompt for better instruction following

### Step 5: Train SFT Model First
Ensure SFT is well above 0.4 accuracy threshold:
```bash
cd /workspace/homework3_v3
python3 -m homework.sft train
python3 -m homework.sft test
```

**Target**: Accuracy ≥ 0.5 (safely above 0.4 minimum)

### Step 6: Train RFT Model
Once RFT dataset has 850-900 examples:
```bash
cd /workspace/homework3_v3
python3 -m homework.rft train
python3 -m homework.rft test
```

**Target**: Accuracy ≥ 0.65 (safely above 0.6 minimum)

## Optimization Strategies

### Strategy 1: Maximize Training Examples
Current datagen processes entire training set (1,000 questions).

**To get 850-900 examples**, you need:
- 85-90% success rate with oversample=10
- 42-45% success rate with oversample=20

### Strategy 2: Improve CoT Model Performance
The CoT model's accuracy directly impacts data generation success rate.

**Current CoT improvements already in place**:
1. ✅ Chat template with system prompt
2. ✅ 4-shot examples for unit conversions
3. ✅ 80 max_new_tokens for reasoning
4. ✅ Proper answer tag format

**Potential improvements**:
1. Add more diverse few-shot examples
2. Fine-tune the prompt wording
3. Test different temperatures during inference

### Strategy 3: Data Quality Check Script
Create a validation script:

```python
import json

def validate_rft_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    issues = []
    for i, (question, answer, reasoning) in enumerate(data):
        # Check answer tags
        if '<answer>' not in reasoning or '</answer>' not in reasoning:
            issues.append(f"Example {i}: Missing answer tags")
        
        # Check for reasoning content
        if len(reasoning) < 20:
            issues.append(f"Example {i}: Reasoning too short")
        
        # Check answer tag placement
        if reasoning.count('<answer>') != 1 or reasoning.count('</answer>') != 1:
            issues.append(f"Example {i}: Multiple or malformed answer tags")
    
    print(f"Total examples: {len(data)}")
    print(f"Issues found: {len(issues)}")
    if issues:
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
    
    return len(data), len(issues)

validate_rft_data('data/rft.json')
```

## Next Steps

1. ✅ **COMPLETED**: Fixed NaN to integer conversion error in base_llm.py
2. ⏳ **TODO**: Test CoT model accuracy
3. ⏳ **TODO**: Generate RFT dataset with target 850-900 examples
4. ⏳ **TODO**: Validate RFT data quality
5. ⏳ **TODO**: Train and test models to exceed threshold boundaries

## Key Metrics to Track

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| RFT examples | 0 | 850-900 | Not yet generated |
| CoT accuracy | ? | ≥0.45 | Need to test |
| SFT accuracy | ? | ≥0.50 | Need to test |
| RFT accuracy | ? | ≥0.65 | Need to test |
| Rejection rate | ? | <15% | During datagen |

## Commands Summary

```bash
# Test CoT model
python3 -m homework.cot test

# Generate RFT data
python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6

# Check RFT data count
python3 -c "import json; print(f'{len(json.load(open(\"data/rft.json\")))} examples')"

# Train SFT
python3 -m homework.sft train
python3 -m homework.sft test

# Train RFT
python3 -m homework.rft train
python3 -m homework.rft test

# Grade everything
python3 -m grader homework
```
