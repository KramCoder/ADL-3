#!/usr/bin/env python3
"""
Diagnostic script to help identify why SFT model has 0.0 accuracy and 0.0 answer_rate.

This script will gather:
1. Model generation samples (what the model actually outputs)
2. Tokenization details (how inputs are being processed)
3. Model loading status (whether adapter files exist)
4. Training format vs inference format comparison
5. Sample predictions with expected answers
"""

import json
from pathlib import Path
import sys

# Add homework directory to path
sys.path.insert(0, str(Path(__file__).parent))

from homework.sft import load
from homework.data import Dataset
from homework.base_llm import BaseLLM


def check_model_files():
    """Check if adapter files exist"""
    print("=" * 80)
    print("1. CHECKING MODEL FILES")
    print("=" * 80)
    
    model_path = Path(__file__).parent / "homework" / "sft_model"
    print(f"Model path: {model_path}")
    print(f"Path exists: {model_path.exists()}")
    
    if model_path.exists():
        files = list(model_path.iterdir())
        print(f"Files in model directory ({len(files)}):")
        for f in files:
            size = f.stat().st_size if f.is_file() else 0
            print(f"  - {f.name} ({size:,} bytes)")
        
        # Check for adapter files
        adapter_files = [
            "adapter_model.bin",
            "adapter_model.safetensors", 
            "adapter_config.json"
        ]
        print("\nAdapter files check:")
        for fname in adapter_files:
            exists = (model_path / fname).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {fname}")
    else:
        print("  ✗ Model directory does not exist!")
    
    print()


def check_tokenization():
    """Check how inputs are being tokenized"""
    print("=" * 80)
    print("2. CHECKING TOKENIZATION")
    print("=" * 80)
    
    try:
        llm = BaseLLM()
        testset = Dataset("valid")
        
        # Get a sample question
        sample_question = testset[0][0]
        sample_answer = testset[0][1]
        
        print(f"Sample question: {sample_question}")
        print(f"Expected answer: {sample_answer}")
        print()
        
        # Check training format
        from homework.sft import format_example
        formatted = format_example(sample_question, sample_answer)
        print(f"Training format:")
        print(f"  Question: '{formatted['question']}'")
        print(f"  Answer: '{formatted['answer']}'")
        print()
        
        # Check inference format
        formatted_prompt = llm.format_prompt(sample_question)
        print(f"Inference format (format_prompt):")
        print(f"  '{formatted_prompt}'")
        print()
        
        # Tokenize both
        from homework.sft import tokenize
        train_tokens = tokenize(llm.tokenizer, formatted['question'], formatted['answer'])
        print(f"Training tokenization:")
        print(f"  Input length: {len(train_tokens['input_ids'])}")
        print(f"  Non-masked labels: {sum(1 for l in train_tokens['labels'] if l != -100)}")
        print(f"  First 20 input_ids: {train_tokens['input_ids'][:20]}")
        print(f"  First 20 labels: {train_tokens['labels'][:20]}")
        print()
        
        # Tokenize inference prompt
        inference_tokens = llm.tokenizer(formatted_prompt, return_tensors="pt")
        print(f"Inference tokenization:")
        print(f"  Input length: {inference_tokens['input_ids'].shape[1]}")
        print(f"  First 20 input_ids: {inference_tokens['input_ids'][0][:20].tolist()}")
        print()
        
    except Exception as e:
        print(f"  ✗ Error checking tokenization: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def check_model_generation():
    """Check what the model actually generates"""
    print("=" * 80)
    print("3. CHECKING MODEL GENERATION")
    print("=" * 80)
    
    try:
        print("Loading SFT model...")
        llm = load()
        print("  ✓ Model loaded successfully")
        print()
        
        testset = Dataset("valid")
        
        # Test on first 5 samples
        print("Testing on first 5 validation samples:")
        print()
        
        for i in range(min(5, len(testset))):
            question = testset[i][0]
            expected_answer = testset[i][1]
            
            print(f"Sample {i+1}:")
            print(f"  Question: {question}")
            print(f"  Expected answer: {expected_answer}")
            
            # Get raw generation
            formatted_prompt = llm.format_prompt(question)
            print(f"  Formatted prompt: '{formatted_prompt}'")
            
            try:
                raw_output = llm.generate(question)
                print(f"  Raw generation: '{raw_output}'")
                
                # Parse answer
                parsed = llm.parse_answer(raw_output)
                print(f"  Parsed answer: {parsed}")
                
                if parsed != parsed:  # NaN check
                    print(f"  ✗ Parsed as NaN!")
                else:
                    from homework.data import is_answer_valid
                    is_correct = is_answer_valid(parsed, expected_answer)
                    print(f"  Correct: {is_correct}")
                
            except Exception as e:
                print(f"  ✗ Generation error: {e}")
                import traceback
                traceback.print_exc()
            
            print()
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Trying to load base model instead...")
        try:
            llm = BaseLLM()
            print("  ✓ Base model loaded (untrained)")
            print("  This suggests the adapter files might be missing or corrupted.")
        except Exception as e2:
            print(f"  ✗ Error loading base model: {e2}")
    
    print()


def check_batch_generation():
    """Check batched generation behavior"""
    print("=" * 80)
    print("4. CHECKING BATCHED GENERATION")
    print("=" * 80)
    
    try:
        llm = load()
        testset = Dataset("valid")
        
        # Test on 10 samples
        questions = [testset[i][0] for i in range(min(10, len(testset)))]
        expected_answers = [testset[i][1] for i in range(min(10, len(testset)))]
        
        print(f"Testing batched generation on {len(questions)} samples...")
        
        try:
            answers = llm.answer(*questions)
            print(f"  ✓ Generated {len(answers)} answers")
            print()
            
            print("Results:")
            nan_count = 0
            correct_count = 0
            
            for i, (question, expected, answer) in enumerate(zip(questions, expected_answers, answers)):
                is_nan = answer != answer  # NaN check
                if is_nan:
                    nan_count += 1
                    status = "NaN"
                else:
                    from homework.data import is_answer_valid
                    is_correct = is_answer_valid(answer, expected)
                    if is_correct:
                        correct_count += 1
                        status = "✓"
                    else:
                        status = "✗"
                
                if i < 5:  # Show first 5 in detail
                    print(f"  {status} Q{i+1}: expected={expected:.6f}, got={answer:.6f if not is_nan else 'NaN'}")
            
            print(f"\nSummary:")
            print(f"  NaN answers: {nan_count}/{len(answers)} ({nan_count/len(answers)*100:.1f}%)")
            print(f"  Correct answers: {correct_count}/{len(answers)} ({correct_count/len(answers)*100:.1f}%)")
            
        except Exception as e:
            print(f"  ✗ Batched generation error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def check_training_data_format():
    """Check training data format"""
    print("=" * 80)
    print("5. CHECKING TRAINING DATA FORMAT")
    print("=" * 80)
    
    try:
        train_dataset = Dataset("train")
        print(f"Training dataset size: {len(train_dataset)}")
        print()
        
        # Show first 3 examples
        print("First 3 training examples:")
        for i in range(min(3, len(train_dataset))):
            question, answer = train_dataset[i]
            print(f"  {i+1}. Q: {question}")
            print(f"     A: {answer}")
            
            from homework.sft import format_example
            formatted = format_example(question, answer)
            print(f"     Formatted: '{formatted['question']} {formatted['answer']}'")
            print()
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """Run all diagnostic checks"""
    print("\n" + "=" * 80)
    print("SFT MODEL DIAGNOSTIC REPORT")
    print("=" * 80)
    print()
    
    check_model_files()
    check_training_data_format()
    check_tokenization()
    check_model_generation()
    check_batch_generation()
    
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print()
    print("Please share the output above to help diagnose the issue.")
    print("Key things to look for:")
    print("  1. Are adapter files present? (adapter_model.bin or adapter_model.safetensors)")
    print("  2. Are generations empty or malformed?")
    print("  3. Are all parsed answers NaN?")
    print("  4. Does training format match inference format?")


if __name__ == "__main__":
    main()
