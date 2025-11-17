# Quick Action Guide - Fix Applied

## ✓ Changes Applied

### Fixed Files:
1. **`homework/base_llm.py`** - Now uses 360M model (was 1.7B)
2. **`homework/datagen.py`** - Explicitly uses 1.7B model for data generation only

### What This Fixes:
- ✓ Model now has ~360M parameters (under 380M limit)
- ✓ Grader will no longer reject with "Model has 1711376384 parameters" error
- ✓ Data generation still uses 1.7B for better quality (as README recommends)
- ✓ All training and inference uses 360M model

## 🚀 What You Need to Do Now

### Step 1: Clean Up Old Files
```bash
cd /workspace/homework3_v3

# Remove old rft.json if it was generated with the wrong model
rm -f data/rft.json

# Remove old trained model
rm -rf homework/sft_model/*
```

### Step 2: Run Fresh Training
```bash
# This will:
# 1. Auto-generate rft.json using 1.7B model (~4 hours, 96% success rate)
# 2. Train SFT model using 360M model (~1 hour)
# 3. Test the model
python -m homework.sft train
```

### Step 3: Create Submission
```bash
# Create submission bundle
python3 bundle.py homework [YOUR_UT_ID]

# Should see: "Submission created: homework3_v3/[YOUR_UT_ID].zip [SIZE] MB"
# Size should be under 50MB
```

### Step 4: Verify Locally
```bash
# Test the submission locally
python3 -m grader [YOUR_UT_ID].zip
```

**Expected output:**
- Should NOT see: "Model has 1711376384 parameters" error
- Should see grading results with accuracy score

## 📊 About SFT Accuracy

Your current accuracy: **0.36 (36%)**

### Why accuracy might be lower with 360M model:
- Smaller model = less capacity than 1.7B
- But still capable of good performance with proper training
- The training has been optimized:
  - 6 epochs (not 5)
  - Higher learning rate (5e-4 instead of 2e-4)
  - Cosine LR schedule with warmup
  - Gradient accumulation

### To potentially improve accuracy:
1. **Ensure high-quality rft.json:**
   - Delete old rft.json
   - Regenerate with 1.7B model (now configured correctly)
   - Should achieve 90%+ success rate

2. **Training tips:**
   - Let it train for all 6 epochs
   - Monitor loss - should decrease to ~0.3-0.4
   - Final accuracy target: 0.4-0.6 (40-60%) is good for this task

3. **If accuracy is still low:**
   - Check that rft.json has 850-950 examples
   - Verify examples have proper <answer></answer> tags
   - Consider increasing epochs in sft.py (line 381) from 6 to 8-10

## 🔍 Troubleshooting

### If training fails:
```bash
# Check if rft.json exists and is valid
ls -lh data/rft.json
python3 -c "import json; data=json.load(open('data/rft.json')); print(f'{len(data)} examples')"
```

### If grader still fails:
```bash
# Check model parameters
python3 -c "
from homework.base_llm import BaseLLM
llm = BaseLLM()
params = sum(p.numel() for p in llm.model.parameters())
print(f'Model parameters: {params:,}')
print(f'Under limit: {params < 380000000}')
"
```

### If submission is too large:
```bash
# Remove any checkpoint directories
rm -rf homework/sft_model/checkpoint-*
rm -rf homework/rft_model/checkpoint-*

# Keep only the final adapter files
ls -lh homework/sft_model/
# Should only see: adapter_config.json, adapter_model.safetensors (or .bin)
```

## 📋 Expected Timeline

1. **RFT Data Generation:** ~4 hours (one-time)
   - Generates 850-950 training examples
   - Uses 1.7B model for quality
   - 90%+ success rate expected

2. **SFT Training:** ~1 hour
   - 6 epochs on rft.json
   - Uses 360M model
   - Creates LoRA adapter (~20MB)

3. **Total first run:** ~5 hours
   - Subsequent runs are faster (rft.json cached)

## ✅ Success Criteria

When everything works:
- [ ] rft.json exists with 850-950 examples
- [ ] SFT model trains without errors
- [ ] Final loss ~0.3-0.4
- [ ] Accuracy 0.3-0.6 (30-60%)
- [ ] Submission .zip < 50MB
- [ ] Grader loads model without parameter error
- [ ] Grader runs tests and returns score

## 📝 Key Takeaway

**The fix separates concerns:**
- **Data generation** = Use powerful 1.7B model for quality
- **Training/Inference** = Use compact 360M model for grader compliance

This is the intended design per the README and gives you the best of both worlds.
