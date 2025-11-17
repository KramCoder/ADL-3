# SFT Model Diagnostic Guide

## Your Issue
After completing SFT training, you're getting:
- `benchmark_result.accuracy=0.0`
- `benchmark_result.answer_rate=0.0`

## What Information to Collect

To diagnose why the model has 0% accuracy and 0% answer rate, you need to understand what the model is actually generating. Here's what to check:

### 1. **Run the Diagnostic Script**

I've created a diagnostic script that will show you exactly what's happening:

```bash
cd /workspace/homework3_v3
python diagnose_sft_model.py
```

This script will show you:
- Raw model outputs (what the model actually generates)
- Whether outputs contain `<answer>` tags
- What `parse_answer()` extracts from the outputs
- Whether parsed answers are NaN
- Model configuration and state

### 2. **Key Diagnostic Information**

The diagnostic script will help identify:

#### A. **Empty Outputs**
- If the model generates empty strings, it means the model isn't producing any text
- **Symptom**: Raw output length = 0
- **Possible causes**: Model not trained, generation parameters wrong, model frozen

#### B. **Missing `<answer>` Tags**
- The model needs to generate text in the format: `<answer>value</answer>`
- **Symptom**: Output doesn't contain `<answer>` tag
- **Possible causes**: 
  - Training format mismatch (model wasn't trained to generate this format)
  - Prompt format doesn't match training format
  - Model generates different format

#### C. **Parsing Failures**
- Even if `<answer>` tags exist, parsing might fail if the format is wrong
- **Symptom**: Parsed answer is NaN
- **Possible causes**:
  - Malformed tags: `<answer>value` (missing closing tag)
  - Non-numeric content: `<answer>text</answer>`
  - Empty tags: `<answer></answer>`

#### D. **Model Not Loaded Correctly**
- The trained LoRA adapter might not be loading
- **Symptom**: No LoRA adapter detected, or model behaves like untrained base model
- **Possible causes**:
  - Adapter files missing or corrupted
  - Wrong model path
  - Adapter not properly saved during training

### 3. **Manual Checks You Can Do**

If you want to manually inspect, here's what to check:

#### Check 1: Model Files
```bash
ls -la /content/ADL-3/homework3_v3/homework/sft_model/
```

You should see:
- `adapter_config.json` (LoRA configuration)
- `adapter_model.bin` OR `adapter_model.safetensors` (trained weights)

#### Check 2: Test Single Generation
```python
from homework.sft import load
llm = load()

# Test on one question
question = "Convert 5 feet to meters"
formatted = llm.format_prompt(question)
print(f"Formatted prompt: {formatted}")

raw_output = llm.generate(question)
print(f"Raw output: {raw_output}")

parsed = llm.parse_answer(raw_output)
print(f"Parsed answer: {parsed}")
```

#### Check 3: Check Training Logs
Look at your training logs to see:
- Did training complete successfully?
- What was the final loss?
- Were there any warnings about NaN gradients or masked labels?

### 4. **Common Issues and Solutions**

Based on the diagnostic output, here are common issues:

#### Issue: Empty Outputs
**Solution**: 
- Check if model is in eval mode: `llm.model.eval()`
- Check generation parameters (max_new_tokens, etc.)
- Verify model actually trained (check training logs)

#### Issue: Missing `<answer>` Tags
**Solution**:
- Check if training format matches inference format
- Training uses: `"question <answer>value</answer>"`
- Inference prompt should be: `"question <answer>"` (model completes the rest)
- Verify `format_prompt()` returns the correct format

#### Issue: Parsing Returns NaN
**Solution**:
- Check if output format matches expected format
- Verify the value inside `<answer>` tags is numeric
- Check if closing tag `</answer>` is present

#### Issue: Model Not Learning
**Solution**:
- Check training logs for NaN gradients
- Verify labels are not all masked
- Check learning rate and training steps
- Verify LoRA adapter is actually being trained

### 5. **What to Share for Further Diagnosis**

If you need help after running the diagnostic, share:

1. **Output from diagnostic script** (`diagnose_sft_model.py`)
2. **Sample raw outputs** from the model (first 3-5 examples)
3. **Training log excerpt** showing final loss and any warnings
4. **Model directory contents** (`ls -la` of sft_model directory)
5. **Any error messages** during training or testing

## Quick Test

Run this to quickly see what's happening:

```bash
cd /workspace/homework3_v3
python diagnose_sft_model.py
```

The output will tell you exactly what's wrong!
