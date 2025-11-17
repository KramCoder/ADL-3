# 🚀 Start Here: RFT Training Improvements

## What Was Done

### 1. ✅ FIXED: NaN to Integer Conversion Error

The grader was crashing with:
```
ValueError: cannot convert float NaN to integer
```

**Fixed in `homework/base_llm.py`**:
- `parse_answer()` now validates outputs and returns 0.0 instead of NaN
- `generate()` ensures valid output with proper answer tags
- `batched_generate()` validates all generations

**Result**: Grader runs successfully! Test with:
```bash
python3 -m grader homework
```

### 2. ✅ CREATED: Comprehensive Improvement Tools

Four new scripts to help you achieve 850-900 high-quality RFT training examples:

| Script | Purpose |
|--------|---------|
| `RFT_IMPROVEMENT_PLAN.md` | Detailed improvement guide and strategy |
| `validate_rft_data.py` | Automated data quality validation |
| `test_all_models.py` | Test all models against thresholds |
| `improve_datagen.py` | Datagen strategy analyzer |
| `IMPLEMENTATION_SUMMARY.md` | Complete implementation details |

## 🎯 Your Action Items

Based on the advice you received, here's your priority checklist:

### Priority 1: Generate 850-900 RFT Training Examples

```bash
# Step 1: Check current strategy
python3 improve_datagen.py

# Step 2: Generate RFT data
python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6

# Step 3: Validate results
python3 validate_rft_data.py

# If you got < 850 examples, regenerate with higher oversample:
python3 -m homework.datagen data/rft.json --oversample 15 --temperature 0.6
python3 validate_rft_data.py
```

**Target**: Get to 850-900 examples with:
- ✅ Complete reasoning text
- ✅ Both `<answer>` and `</answer>` tags
- ✅ Correct numerical answers

### Priority 2: Check CoT Model Accuracy

```bash
python3 test_all_models.py
```

This tells you:
- Current CoT accuracy (need ≥ 0.4, target ≥ 0.45)
- Expected rejection rate during datagen
- Whether you need to adjust oversample

**If CoT accuracy < 0.35**: Use oversample=15-20  
**If CoT accuracy ≥ 0.35**: Use oversample=10-12

### Priority 3: Verify Training Data Quality

```bash
# After generating RFT data
python3 validate_rft_data.py

# Manually inspect examples
python3 -c "import json; d=json.load(open('data/rft.json')); print(json.dumps(d[0], indent=2))"
```

Look for:
- ✅ Step-by-step reasoning before answer
- ✅ Proper tag format: `<answer>value</answer>`
- ✅ No truncated or empty reasoning

### Priority 4: Stay Above Threshold Boundaries

**Targets** (with safety margins):

| Model | Minimum | Target | Safe Zone |
|-------|---------|--------|-----------|
| CoT   | 0.4     | 0.45+  | Yellow zone: 0.4-0.45 |
| SFT   | 0.6     | 0.50+  | Yellow zone: 0.6-0.65 |
| RFT   | 0.7     | 0.65+  | Yellow zone: 0.7-0.75 |

**Check with**:
```bash
python3 test_all_models.py
```

This shows:
- ✅ Current accuracy for each model
- ✅ Distance to target
- ✅ Safety margins
- ✅ Recommendations

## 📋 Complete Workflow

```bash
# ============================================================
# Phase 1: Test Current State
# ============================================================

# Test all models and see where you stand
python3 test_all_models.py

# Analyze datagen strategy
python3 improve_datagen.py

# ============================================================
# Phase 2: Generate RFT Training Data
# ============================================================

# Generate with recommended settings
python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6

# Validate quality and count
python3 validate_rft_data.py

# If needed, adjust and regenerate
# (If < 850 examples or quality issues)
python3 -m homework.datagen data/rft.json --oversample 15 --temperature 0.6
python3 validate_rft_data.py

# ============================================================
# Phase 3: Train Models
# ============================================================

# Train SFT model (prerequisite for RFT)
python3 -m homework.sft train
python3 -m homework.sft test

# Train RFT model  
python3 -m homework.rft train
python3 -m homework.rft test

# ============================================================
# Phase 4: Verify Results
# ============================================================

# Test all models
python3 test_all_models.py

# Run full grader
python3 -m grader homework

# ============================================================
# Phase 5: Bundle and Submit
# ============================================================

# Bundle your solution
python3 bundle.py homework <your_utid>

# Test the bundle
python3 -m grader <your_utid>.zip
```

## 🔧 Troubleshooting

### Issue: Not Generating Enough RFT Examples

**Symptoms**: < 850 examples after datagen

**Solutions**:
1. Increase oversample: Try 15 or 20
2. Check CoT accuracy: Should be ≥ 0.3
3. Adjust temperature: Try 0.5 for more consistency

```bash
# Try higher oversample
python3 -m homework.datagen data/rft.json --oversample 15 --temperature 0.6
```

### Issue: RFT Model Accuracy Too Low

**Symptoms**: RFT accuracy < 0.6

**Solutions**:
1. Ensure 850-900 training examples
2. Validate data quality with `validate_rft_data.py`
3. Check that reasoning is complete (not truncated)
4. Train for more epochs if needed

### Issue: SFT Model Not Improving

**Symptoms**: SFT accuracy < 0.5

**Solutions**:
1. Check that SFT training completed successfully
2. Verify training data format
3. Consider adjusting learning rate or epochs

### Issue: High Rejection Rate During Datagen

**Symptoms**: > 20% rejection rate

**Solutions**:
1. Test CoT accuracy: `python3 -m homework.cot test`
2. If CoT < 0.3: Improve CoT first or use oversample=20
3. If CoT ≥ 0.3: This is normal, use oversample=10-15

## 📊 Quick Reference

### File Locations

```
homework3_v3/
├── homework/
│   ├── base_llm.py          # ✅ FIXED: NaN handling
│   ├── cot.py                # CoT model with few-shot prompting
│   ├── sft.py                # SFT training
│   ├── rft.py                # RFT training  
│   └── datagen.py            # RFT data generation
├── data/
│   ├── train.json            # 1000 training questions
│   ├── valid.json            # Validation set
│   └── rft.json              # Generated RFT training data
├── validate_rft_data.py      # ✅ NEW: Data validation
├── test_all_models.py        # ✅ NEW: Model testing
├── improve_datagen.py        # ✅ NEW: Strategy analyzer
└── RFT_IMPROVEMENT_PLAN.md   # ✅ NEW: Detailed guide
```

### Commands Cheat Sheet

```bash
# Data Generation
python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6

# Model Testing
python3 test_all_models.py
python3 -m homework.cot test
python3 -m homework.sft test  
python3 -m homework.rft test

# Data Validation
python3 validate_rft_data.py
python3 improve_datagen.py

# Training
python3 -m homework.sft train
python3 -m homework.rft train

# Grading
python3 -m grader homework
```

## ✅ Success Checklist

Before considering yourself done, verify:

- [ ] RFT dataset has 850-900 examples
- [ ] `validate_rft_data.py` shows no critical issues
- [ ] CoT accuracy ≥ 0.45 (test with `test_all_models.py`)
- [ ] SFT accuracy ≥ 0.50 (test with `test_all_models.py`)
- [ ] RFT accuracy ≥ 0.65 (test with `test_all_models.py`)
- [ ] Grader completes without NaN errors
- [ ] Bundle created and tested
- [ ] All models above minimum thresholds

## 💡 Pro Tips

1. **Start with datagen**: You can't train RFT without data
2. **Validate early**: Check data quality before training
3. **Test incrementally**: Test after each phase
4. **Use safety margins**: Don't aim for exactly the minimum
5. **Monitor rejection rate**: High rejection = need more oversample

## 📞 Need More Help?

1. **Read the detailed guide**: `RFT_IMPROVEMENT_PLAN.md`
2. **Check implementation details**: `IMPLEMENTATION_SUMMARY.md`
3. **Run the analyzers**: They provide specific recommendations
4. **Test frequently**: Use `test_all_models.py` to track progress

---

## 🎯 TL;DR - Quickest Path to Success

```bash
# 1. Generate RFT data (most important!)
python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6

# 2. Validate it
python3 validate_rft_data.py

# 3. Train models
python3 -m homework.sft train
python3 -m homework.rft train

# 4. Test everything
python3 test_all_models.py

# 5. Grade
python3 -m grader homework
```

Good luck! 🚀
