# Grader Issues Analysis

## Summary

The grader is failing on two tests:
1. **Non-batched generation**: 0/10 points
2. **RFT Model**: 0/25 points

## Issue 1: Non-batched Generation (0/10)

### What the grader checks:
- Calls `model.generate(questions[i])` for each question individually (32 questions)
- Computes cross-entropy loss on `questions[i] + answers[i]`
- Expects loss between **6.2 and 8.0** (lower is better)
- If loss > 8.0, score = 0

### Why it's failing:
The loss is > 8.0, meaning the generated answers have high perplexity/loss.

### Root cause:
The `generate()` method doesn't reset the tokenizer's `padding_side` attribute. If `batched_generate()` is called first (which sets `padding_side = "left"`), this can affect the behavior of `generate()` even though it shouldn't matter for single prompts.

### Fix applied:
- Reset `padding_side` to `'right'` at the start of `generate()`
- Restore original value at the end

## Issue 2: RFT Model (0/25)

### What the grader checks:
- Calls `load_rft()` to load the RFT model
- Runs `benchmark(model, dataset, 100)` on validation set
- Expects accuracy between **0.6 and 0.7**
- Currently getting **0.0 accuracy**

### Why it's failing:
The trained RFT model files are **missing from the bundle**.

### What's missing:
The bundle should include these files in `homework/rft_model/`:
- `adapter_model.bin` or `adapter_model.safetensors` (the trained LoRA weights)
- `adapter_config.json` (LoRA configuration)

### Root cause:
When `load_rft()` is called and these files don't exist, the `_ensure_adapter()` function creates a **new untrained adapter** with random weights. This untrained model produces random outputs, resulting in 0.0 accuracy.

### What needs to be done:
1. **Train the RFT model** (if not already trained):
   ```bash
   python -m homework.rft train
   ```

2. **Verify model files exist**:
   ```bash
   ls homework/rft_model/
   # Should show: adapter_model.bin (or .safetensors), adapter_config.json
   ```

3. **Rebundle** - The bundle.py script should automatically include these files (they're not in the BLACKLIST).

### Note:
The bundle.py currently excludes:
- `README.md` files (`.md` extension)
- But adapter model files should be included

Check if the model was actually trained and if the files exist before bundling.
