# RFT Instructions Compliance Analysis

## Summary

This document analyzes whether the code follows the RFT (Rejection Sampling Fine-Tuning) implementation instructions.

## Instructions Checklist

### ✅ **Correctly Implemented**

1. **`generate_dataset` function in `datagen.py`**:
   - ✅ Uses `CoTModel.batched_generate` with `num_return_sequences > 1` (default: 15)
   - ✅ Uses `temperature > 0` (default: 0.7)
   - ✅ Produces 10-20 different completions (default: 15, which is in range)
   - ✅ Selects the one with the correct answer
   - ✅ Ignores data points if none are correct (with fallback mechanism)
   - ✅ Stores output in JSON format matching the sample: `["question", answer_value, "reasoning with <answer>...</answer>"]`
   - ✅ Uses `CoTModel` which provides chain-of-thought reasoning

2. **RFT Training**:
   - ✅ Separate `rft.py` module trains on RFT dataset
   - ✅ Loads data from `data/rft.json`
   - ✅ Uses `format_rft_example` to format question + reasoning pairs
   - ✅ Trains on the reasoning text (which includes `<answer>...</answer>` tags)

### ❌ **Issues Found**

1. **Model Mismatch** (CRITICAL):
   - **Instruction**: "Using the HuggingFaceTB/SmolLM2-1.7B-Instruct model should further help you obtain better rollouts."
   - **Current**: `base_llm.py` line 19 uses `HuggingFaceTB/SmolLM2-360M-Instruct`
   - **Impact**: The smaller 360M model may produce lower quality rollouts compared to the 1.7B model specified in instructions
   - **Location**: `homework/base_llm.py:19`

2. **SFT Code Modification** (AMBIGUOUS):
   - **Instruction**: "Modify your SFT code to train on this new data of question + reasoning pairs."
   - **Current**: 
     - `sft.py` still trains only on regular training dataset (`Dataset("train")`)
     - Separate `rft.py` exists that trains on RFT data
   - **Interpretation**: This could be correct (separate RFT training script) or incorrect (SFT should be modified to optionally use RFT data)
   - **Note**: The instructions might mean creating a separate RFT training script, which exists

3. **Output Path** (MINOR):
   - **Instruction**: "Store the output in a json file in data/rft.json"
   - **Current**: Function takes `output_json` as parameter (flexible but not hardcoded)
   - **Impact**: Low - the function can be called with `data/rft.json` as argument

## Detailed Analysis

### `datagen.py` Implementation

The `generate_dataset` function correctly:
- Generates 15 completions per question (within 10-20 range)
- Uses temperature 0.7 for diversity
- Validates answers and selects correct ones
- Has retry logic with higher temperature if no correct answer found
- Falls back to best answer within 10% error if needed
- Stores data in the correct format

**Code Location**: `homework/datagen.py:18-134`

### Model Configuration

The base model is configured in `base_llm.py`:
```python
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Line 19
```

**Should be**:
```python
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

### RFT Training

The `rft.py` module correctly:
- Loads RFT dataset from `data/rft.json`
- Formats examples as question + reasoning pairs
- Trains on the full reasoning text (including answer tags)

**Code Location**: `homework/rft.py:82-185`

## Recommendations

1. **Update model checkpoint** to `HuggingFaceTB/SmolLM2-1.7B-Instruct` in `base_llm.py`
2. **Clarify SFT modification**: If the instruction means SFT should optionally train on RFT data, modify `sft.py` to support this. If it means a separate RFT training script, the current implementation is correct.
3. **Consider hardcoding output path** to `data/rft.json` in `generate_dataset` or update documentation to clarify the parameter usage

## Conclusion

The implementation follows most instructions correctly, but uses a different (smaller) model than specified. The RFT training logic is correctly implemented in a separate module.
