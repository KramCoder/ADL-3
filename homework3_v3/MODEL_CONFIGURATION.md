# Model Configuration Changes

## Summary
Updated the codebase to use the **360M parameter model** for all tasks except data generation, which uses the **1.7B parameter model** for better quality rollouts.

## Changes Made

### 1. `homework/base_llm.py`
- **Changed default checkpoint**: From `SmolLM2-1.7B-Instruct` → `SmolLM2-360M-Instruct`
- **Impact**: All models now default to 360M parameters
- **Reason**: Grader enforces a maximum of 380M parameters

### 2. `homework/datagen.py`
- **Added explicit checkpoint**: `CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")`
- **Impact**: Data generation uses the larger 1.7B model for better quality
- **Reason**: README section 4 (line 138) recommends using 1.7B for better rollouts

## Model Usage by Component

| Component | Model | Parameters | Purpose |
|-----------|-------|------------|---------|
| Base LLM | SmolLM2-360M-Instruct | ~360M | Default for all tasks |
| CoT Model | SmolLM2-360M-Instruct | ~360M | In-context learning |
| SFT Training | SmolLM2-360M-Instruct | ~360M | Supervised fine-tuning |
| SFT Inference | SmolLM2-360M-Instruct | ~360M | Model evaluation |
| RFT Training | SmolLM2-360M-Instruct | ~360M | RFT fine-tuning |
| **Data Generation** | **SmolLM2-1.7B-Instruct** | **~1.7B** | **RFT dataset generation only** |

## Grader Constraints
- **Maximum parameters**: 380,000,000 (380M)
- **360M model**: ~360M parameters ✓ PASSES
- **1.7B model**: ~1.7B parameters ✗ FAILS (but not used for grading)

## README Compliance
✓ Follows README section 4 guidance: "Using the `HuggingFaceTB/SmolLM2-1.7B-Instruct` model should further help you obtain better rollouts."

## How to Verify
Run the verification script:
```bash
python3 verify_model_size.py
```

## Next Steps for Better SFT Accuracy
Current accuracy is 0.36. To improve:

1. **Regenerate RFT dataset with 1.7B model**:
   ```bash
   rm data/rft.json
   python3 -m homework.sft train
   ```
   The training script will automatically generate a new RFT dataset using the 1.7B model.

2. **Training parameters are already optimized**:
   - 6 epochs
   - Learning rate: 5e-4
   - Batch size: 16 with gradient accumulation
   - Cosine learning rate schedule
   - LoRA rank 16 with alpha 32

3. **Quality RFT dataset**: The 1.7B model should produce higher quality reasoning chains, leading to better SFT performance.

## Expected Outcome
- ✓ Grader accepts model (< 380M parameters)
- ✓ SFT model trains on high-quality RFT data
- ✓ Improved accuracy from better reasoning examples
