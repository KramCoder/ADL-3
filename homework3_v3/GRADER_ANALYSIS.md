# Grader Analysis: Why Tests Are Failing

## Summary

Based on the grader output and code analysis, here are the issues:

### 1. Non-batched Generation (0/10 points) - FAILING

**What the grader checks:**
- Creates a `BaseLLM()` instance
- Calls `model.generate(question)` for each of 32 validation questions
- Computes cross-entropy loss on `question + answer` (string concatenation)
- Loss bounds: 6.2 (full score) to 8.0 (zero score)
- **Current status: Loss >= 8.0, so score = 0/10**

**The issue:**
The `generate()` method in `base_llm.py` uses `format_prompt()` which adds a space: `f"{question.strip()} "`. However, when the grader computes loss, it does simple string concatenation: `questions[i] + answers[i]` without the space formatting.

This format mismatch might cause:
1. Different tokenization between generation and loss computation
2. Higher loss because the model sees a different format than during generation

**What's needed:**
The loss computation expects the model to have low perplexity on `question + answer` text. The current implementation might be generating answers that don't match the expected format for loss computation.

### 2. Batched Generation (15/15 points) - PASSING ✓

This is working correctly!

### 3. RFT Model (0/25 points) - FAILING

**What the grader checks:**
- Calls `load_rft()` which should return a trained `BaseLLM` instance
- Runs `benchmark(model, dataset, 100)` which uses `model.answer(*questions)`
- Accuracy bounds: 0.6 (zero score) to 0.7 (full score)
- **Current status: Accuracy = 0.0, so score = 0/25**

**The issue:**
The `rft_model/` directory only contains a `README.md` file. The model adapter files are missing:
- `adapter_model.bin` or `adapter_model.safetensors` (the trained LoRA weights)
- `adapter_config.json` (the LoRA configuration)

When `load_rft()` is called, it calls `_ensure_adapter()` which creates an **empty/uninitialized adapter** if the files don't exist. This untrained adapter produces random/wrong answers, resulting in 0.0 accuracy.

**What's needed:**
The RFT model needs to be trained and the adapter files need to be included in the bundle:
1. Train the model: `python -m homework.rft train`
2. Ensure the adapter files are in `homework/rft_model/`:
   - `adapter_model.bin` or `adapter_model.safetensors`
   - `adapter_config.json`
3. Make sure these files are included when bundling (not excluded by BLACKLIST in `bundle.py`)

## Key Differences Between What Works and What Doesn't

### Batched Generation (PASSING)
- Uses `batched_generate()` which properly formats prompts
- The grader also uses `batched_generate()` so the format matches

### Non-batched Generation (FAILING)
- Uses `generate()` which formats prompts with `format_prompt()`
- But loss computation uses raw string concatenation `question + answer`
- Format mismatch causes high loss

### RFT Model (FAILING)
- Model files are missing from the bundle
- Empty adapter produces 0.0 accuracy

## Recommendations

1. **For non-batched generation:**
   - Check if the loss computation format matches the generation format
   - The `generate()` method should return answers that, when concatenated with the question, produce text with low perplexity
   - Consider if the space added by `format_prompt()` affects loss computation

2. **For RFT model:**
   - Train the RFT model: `python -m homework.rft train`
   - Verify adapter files exist in `homework/rft_model/`
   - Check that `bundle.py` doesn't exclude these files
   - Test loading: `python -m homework.rft test` should show accuracy > 0.6
