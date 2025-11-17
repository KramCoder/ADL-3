# RFT Implementation Usage Guide

## Summary of Changes

The SFT training pipeline has been successfully modified to implement RFT (Rejection Sampling Fine-Tuning) as specified in the homework requirements:

### ✅ Changes Completed

1. **Modified `homework/cot.py`**:
   - CoTModel now uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` by default instead of the 360M model
   - This provides better reasoning capabilities for generating RFT training data

2. **Modified `homework/sft.py`**:
   - Updated `format_example()` to handle RFT data format (question + reasoning + answer)
   - Modified `train_model()` to load and train on RFT data from `data/rft.json`
   - Added validation checks for RFT data quality
   - Maintains backward compatibility with original train data if RFT data is not available

3. **Existing `homework/datagen.py`**:
   - Already implements RFT data generation using CoTModel
   - Generates 10-20 completions per question with temperature > 0
   - Selects completions with correct answers
   - Saves to `data/rft.json`

## How to Use

### Step 1: Generate RFT Training Data

Run the data generation script to create the RFT dataset:

```bash
python -m homework.datagen data/rft.json
```

**What this does:**
- Uses CoTModel with the 1.7B model to generate diverse completions
- For each training question, generates 10-15 different completions
- Selects the completions with correct answers
- Saves results to `data/rft.json`
- Target: 850-900+ question/reasoning pairs with 90%+ success rate

**Expected output format in `data/rft.json`:**
```json
[
  [
    "How many gram are there per 6 kg?",
    6000.0,
    "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
  ],
  [
    "Convert 5 quart to pint?",
    10.0,
    "1 quart = 2 pint. So 5 * 2 = <answer>10</answer>"
  ],
  ...
]
```

**Parameters (in `datagen.py`):**
- `oversample`: Number of generations per question (default: 15)
- `temperature`: Sampling temperature for diversity (default: 0.7)

### Step 2: Train SFT Model on RFT Data

Train the model using the generated RFT data:

```bash
python -m homework.sft train
```

**What this does:**
- Automatically detects and loads `data/rft.json` if available
- Trains the model on question + reasoning pairs (not just answers)
- Uses LoRA adapters to keep model size small
- Saves trained adapter to `homework/sft_model/`

**Training details:**
- Model learns both the reasoning process AND the answer format
- Training format: `"question" -> "reasoning text with <answer>value</answer>"`
- Uses the base 360M model with LoRA adapters for efficient training
- Adapter size stays within submission limits

**If RFT data is not found:**
- Falls back to regular train dataset automatically
- Trains with simple answer format: `"question" -> "<answer>value</answer>"`
- Prints message: "RFT data not found. Using regular train dataset."

### Step 3: Test the Trained Model

Evaluate the model on the validation set:

```bash
python -m homework.sft test
```

**Expected improvements with RFT:**
- Better reasoning: Model shows its work before providing answers
- Higher accuracy: Training on correct reasoning chains improves generalization
- Better answer rate: Model produces valid answers more consistently

## Data Format Details

### RFT Data Format (data/rft.json)
Each entry is a list with 3 elements:
1. **Question** (string): The unit conversion question
2. **Answer** (float): The correct numerical answer
3. **Reasoning** (string): Chain-of-thought reasoning including answer tags

Example:
```json
[
  "How many gram are there per 6 kg?",
  6000.0,
  "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
]
```

### Training Format
The model is trained to complete:
- **Input**: `"How many gram are there per 6 kg? "`
- **Target**: `"1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"`

## Verification

A test script is provided to verify the implementation logic:

```bash
python3 test_rft_integration.py
```

This tests:
- ✓ RFT format handling with reasoning
- ✓ Simple format fallback without reasoning
- ✓ RFT dataset structure validation
- ✓ CoTModel checkpoint configuration

## Key Benefits

1. **Better Reasoning**: The model learns to show step-by-step work
2. **Higher Accuracy**: Training on correct reasoning improves generalization
3. **Combines Strengths**: Merges Chain-of-Thought reasoning with SFT
4. **Offline RL**: Uses rejection sampling without complex online training

## Troubleshooting

### Problem: RFT data generation has low success rate (< 80%)
**Solution:**
- The 1.7B model should achieve 90%+ success rate
- Check that `homework/cot.py` is using the 1.7B model
- Verify the format_prompt in `cot.py` provides good examples
- Consider increasing `oversample` parameter in datagen.py

### Problem: RFT dataset has fewer than 850 examples
**Solution:**
- Increase the `oversample` parameter in datagen.py
- Try running with higher temperature (0.8-0.9)
- Use the retry logic in datagen.py (already implemented)

### Problem: Training fails to find RFT data
**Solution:**
- Ensure you've run: `python -m homework.datagen data/rft.json`
- Check that `data/rft.json` exists in the correct location
- Training will automatically fall back to regular data if RFT data is missing

### Problem: Model size exceeds limits
**Solution:**
- The SFT model uses LoRA rank 16 (should be under 20MB)
- For RFT, you can increase to rank 32 (up to 50MB allowed)
- Adjust `DEFAULT_LORA_RANK` in sft.py if needed

## File Checklist

After implementation, these files should exist:
- ✓ `homework/cot.py` - Modified to use 1.7B model
- ✓ `homework/sft.py` - Modified to handle RFT data
- ✓ `homework/datagen.py` - Already implements RFT generation
- ✓ `data/rft.json` - Generated by datagen.py (create in Step 1)
- ✓ `homework/sft_model/` - Trained adapter (created in Step 2)

## Summary

The implementation is complete and ready to use. Follow the 3 steps above to:
1. Generate RFT training data using the 1.7B model
2. Train the SFT model on the RFT data
3. Test and evaluate the trained model

The changes maintain backward compatibility, so the original training workflow still works if RFT data is not generated.
