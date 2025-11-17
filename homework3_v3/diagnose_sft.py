#!/usr/bin/env python3
"""
Comprehensive diagnostic script for SFT model issues.
This script will help identify why the model is getting 0.0 accuracy.
"""

import sys
import json
from pathlib import Path
import torch

# Add homework to path
sys.path.insert(0, str(Path(__file__).parent))

from homework.sft import load, _resolve_path, MODEL_NAME
from homework.data import Dataset
from homework.base_llm import BaseLLM

def check_model_files(model_path: Path):
    """Check what files exist in the model directory."""
    print("=" * 80)
    print("1. MODEL FILES CHECK")
    print("=" * 80)
    
    if not model_path.exists():
        print(f"❌ ERROR: Model directory does not exist: {model_path}")
        return False
    
    print(f"✅ Model directory exists: {model_path}")
    print("\nFiles in model directory:")
    
    # List all files
    all_files = list(model_path.glob("*"))
    if not all_files:
        print("  ❌ Directory is EMPTY!")
        return False
    
    for file in sorted(all_files):
        size = file.stat().st_size if file.is_file() else "DIR"
        print(f"  - {file.name:40s} {size}")
    
    # Check for adapter files
    adapter_bin = model_path / "adapter_model.bin"
    adapter_safetensors = model_path / "adapter_model.safetensors"
    adapter_config = model_path / "adapter_config.json"
    
    print("\nAdapter file status:")
    has_adapter = False
    if adapter_bin.exists():
        print(f"  ✅ adapter_model.bin exists ({adapter_bin.stat().st_size:,} bytes)")
        has_adapter = True
    else:
        print("  ❌ adapter_model.bin NOT found")
    
    if adapter_safetensors.exists():
        print(f"  ✅ adapter_model.safetensors exists ({adapter_safetensors.stat().st_size:,} bytes)")
        has_adapter = True
    else:
        print("  ❌ adapter_model.safetensors NOT found")
    
    if adapter_config.exists():
        print(f"  ✅ adapter_config.json exists ({adapter_config.stat().st_size:,} bytes)")
        # Print config
        with open(adapter_config) as f:
            config = json.load(f)
        print(f"     LoRA rank: {config.get('r', 'N/A')}")
        print(f"     LoRA alpha: {config.get('lora_alpha', 'N/A')}")
    else:
        print("  ❌ adapter_config.json NOT found")
    
    return has_adapter


def test_model_loading(model_path: Path):
    """Test if the model loads correctly."""
    print("\n" + "=" * 80)
    print("2. MODEL LOADING CHECK")
    print("=" * 80)
    
    try:
        print("Loading SFT model...")
        llm = load()
        print("✅ Model loaded successfully")
        
        # Check if model is in eval mode
        if llm.model.training:
            print("  ⚠️  WARNING: Model is in training mode (should be eval)")
        else:
            print("  ✅ Model is in eval mode")
        
        # Check device
        print(f"  Device: {llm.device}")
        
        # Check trainable parameters
        trainable = sum(p.numel() for p in llm.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in llm.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,}")
        
        return llm
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prompt_formatting(llm):
    """Test how prompts are formatted."""
    print("\n" + "=" * 80)
    print("3. PROMPT FORMATTING CHECK")
    print("=" * 80)
    
    test_question = "Can you change 2 hour to its equivalent in min?"
    
    print(f"Original question: {test_question!r}")
    formatted = llm.format_prompt(test_question)
    print(f"Formatted prompt:  {formatted!r}")
    
    # Check if it ends with <answer>
    if formatted.endswith("<answer>"):
        print("✅ Prompt correctly ends with '<answer>' tag")
    else:
        print("❌ WARNING: Prompt does NOT end with '<answer>' tag")
        print("   This will cause the model to not generate properly formatted answers!")
    
    return formatted


def test_generation_detailed(llm, num_examples=5):
    """Test actual model generation with detailed output."""
    print("\n" + "=" * 80)
    print("4. GENERATION TEST (Detailed)")
    print("=" * 80)
    
    # Load test dataset
    testset = Dataset("valid")
    print(f"Loaded {len(testset)} test examples\n")
    
    results = []
    
    for i in range(min(num_examples, len(testset))):
        question, correct_answer = testset[i]
        
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {question}")
        print(f"Correct answer: {correct_answer}")
        
        # Format the prompt
        formatted_prompt = llm.format_prompt(question)
        print(f"Formatted prompt: {formatted_prompt!r}")
        
        # Generate
        try:
            raw_output = llm.generate(question)
            print(f"Raw output: {raw_output!r}")
            
            # Parse
            parsed = llm.parse_answer(raw_output)
            print(f"Parsed value: {parsed}")
            
            # Check if correct
            is_valid = parsed == parsed  # NaN check
            if is_valid:
                is_correct = abs(parsed - correct_answer) < 0.05 * abs(correct_answer)
                print(f"Valid answer: ✅")
                print(f"Correct: {'✅' if is_correct else '❌'}")
                results.append(("valid" if is_valid else "invalid", is_correct if is_valid else False))
            else:
                print(f"Valid answer: ❌ (NaN)")
                results.append(("invalid", False))
                
                # Try to diagnose why parsing failed
                if "<answer>" not in raw_output:
                    print("  ⚠️  Output does NOT contain '<answer>' tag")
                if "</answer>" not in raw_output:
                    print("  ⚠️  Output does NOT contain '</answer>' tag")
        
        except Exception as e:
            print(f"❌ ERROR during generation: {e}")
            results.append(("error", False))
    
    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    valid_count = sum(1 for status, _ in results if status == "valid")
    correct_count = sum(1 for _, correct in results if correct)
    print(f"Valid answers: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"Correct answers: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")
    
    return results


def test_tokenization_sample():
    """Test if tokenization is working correctly during training."""
    print("\n" + "=" * 80)
    print("5. TOKENIZATION CHECK")
    print("=" * 80)
    
    try:
        from homework.sft import tokenize, format_example
        
        # Load base model for tokenizer
        llm = BaseLLM()
        tokenizer = llm.tokenizer
        
        # Test example
        question = "Can you change 2 hour to its equivalent in min?"
        answer = 120.0
        
        # Format
        formatted = format_example(question, answer)
        print(f"Question: {formatted['question']}")
        print(f"Answer: {formatted['answer']}")
        
        # Tokenize
        tokenized = tokenize(tokenizer, **formatted)
        
        input_ids = tokenized["input_ids"]
        labels = tokenized["labels"]
        attention_mask = tokenized["attention_mask"]
        
        print(f"\nInput IDs length: {len(input_ids)}")
        print(f"Labels length: {len(labels)}")
        print(f"Attention mask length: {len(attention_mask)}")
        
        # Count non-masked labels
        non_masked = sum(1 for l in labels if l != -100)
        print(f"Non-masked labels: {non_masked} ({100*non_masked/len(labels):.1f}%)")
        
        if non_masked == 0:
            print("❌ ERROR: ALL labels are masked! Training cannot work.")
            return False
        elif non_masked < 5:
            print("⚠️  WARNING: Very few non-masked labels. Training might be inefficient.")
        else:
            print("✅ Tokenization looks good")
        
        # Decode to show what's being trained
        print("\n--- Decoded sequences ---")
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Full input: {full_text[:200]}...")
        
        # Show which tokens are being trained on
        trained_tokens = [tid for tid, label in zip(input_ids, labels) if label != -100]
        if trained_tokens:
            trained_text = tokenizer.decode(trained_tokens, skip_special_tokens=False)
            print(f"Training on: {trained_text}")
        
        return True
    
    except Exception as e:
        print(f"❌ ERROR during tokenization check: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic checks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SFT MODEL DIAGNOSTIC TOOL" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Get model path
    model_path = _resolve_path(MODEL_NAME)
    print(f"Expected model location: {model_path}")
    print()
    
    # Run checks
    has_model = check_model_files(model_path)
    
    if not has_model:
        print("\n" + "!" * 80)
        print("CRITICAL: No trained model found!")
        print("Please run training first: python -m homework.sft train")
        print("!" * 80)
        return
    
    # Test tokenization (doesn't require trained model)
    test_tokenization_sample()
    
    # Load and test model
    llm = test_model_loading(model_path)
    if llm is None:
        print("\n" + "!" * 80)
        print("CRITICAL: Could not load model!")
        print("!" * 80)
        return
    
    # Test prompt formatting
    test_prompt_formatting(llm)
    
    # Test generation
    results = test_generation_detailed(llm, num_examples=5)
    
    # Final diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    if all(status == "valid" and correct for status, correct in results):
        print("✅ Everything looks good! Model is working correctly.")
    elif all(status == "valid" for status, _ in results):
        print("⚠️  Model generates valid answers but they're incorrect.")
        print("   → Model may need more training or different hyperparameters")
    elif any(status == "valid" for status, _ in results):
        print("⚠️  Model sometimes generates valid answers, sometimes not.")
        print("   → Check if prompt formatting is consistent")
    else:
        print("❌ Model is NOT generating valid answers!")
        print("\nPossible causes:")
        print("1. Model is not trained (all weights are random)")
        print("2. Wrong adapter loaded (untrained adapter)")
        print("3. Prompt format doesn't match training format")
        print("4. Model generation parameters are incorrect")
        print("\nPlease share the output above with your instructor/TA.")


if __name__ == "__main__":
    main()
