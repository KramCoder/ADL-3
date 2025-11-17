#!/usr/bin/env python3
"""
Diagnostic script to identify why SFT model has 0.0 accuracy and answer_rate.

This script will:
1. Load the trained SFT model
2. Test it on a few sample questions
3. Show the raw model outputs
4. Show what parse_answer() extracts
5. Show the formatted prompts
6. Check if the model is generating anything at all
"""

from homework.sft import load
from homework.data import Dataset
from homework.base_llm import BaseLLM


def diagnose_model():
    print("=" * 80)
    print("SFT MODEL DIAGNOSTIC SCRIPT")
    print("=" * 80)
    
    # Load the model
    print("\n1. Loading SFT model...")
    try:
        llm = load()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get a few test samples
    print("\n2. Loading test dataset...")
    testset = Dataset("valid")
    print(f"   ✅ Loaded {len(testset)} test samples")
    
    # Test on first 5 samples
    num_samples = min(5, len(testset))
    print(f"\n3. Testing on {num_samples} sample questions...")
    print("=" * 80)
    
    for i in range(num_samples):
        question, correct_answer = testset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {question}")
        print(f"Correct Answer: {correct_answer}")
        
        # Show the formatted prompt
        formatted_prompt = llm.format_prompt(question)
        print(f"Formatted Prompt: '{formatted_prompt}'")
        
        # Generate raw output
        print("\nGenerating response...")
        try:
            raw_output = llm.generate(question)
            print(f"Raw Model Output: '{raw_output}'")
            print(f"Raw Output Length: {len(raw_output)} characters")
            
            # Check if output is empty
            if not raw_output or raw_output.strip() == "":
                print("   ⚠️  WARNING: Model generated empty output!")
            else:
                print("   ✅ Model generated non-empty output")
            
            # Check if output contains <answer> tag
            if "<answer>" in raw_output:
                print("   ✅ Output contains <answer> tag")
            else:
                print("   ❌ Output does NOT contain <answer> tag")
                print("   This is likely why parse_answer() fails!")
            
            # Try to parse the answer
            parsed_answer = llm.parse_answer(raw_output)
            print(f"Parsed Answer: {parsed_answer}")
            
            if parsed_answer != parsed_answer:  # NaN check
                print("   ❌ Parsed answer is NaN (parsing failed)")
            else:
                print(f"   ✅ Parsed answer is valid: {parsed_answer}")
                
                # Check if it's correct
                from homework.data import is_answer_valid
                is_correct = is_answer_valid(parsed_answer, correct_answer)
                print(f"   {'✅' if is_correct else '❌'} Answer is {'CORRECT' if is_correct else 'INCORRECT'}")
        
        except Exception as e:
            print(f"   ❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()
    
    # Test batched generation
    print("\n" + "=" * 80)
    print("4. Testing batched generation...")
    print("=" * 80)
    
    test_questions = [testset[i][0] for i in range(min(3, len(testset)))]
    print(f"Testing on {len(test_questions)} questions in batch...")
    
    try:
        batched_outputs = llm.batched_generate(test_questions)
        print(f"   ✅ Batched generation completed")
        print(f"   Number of outputs: {len(batched_outputs)}")
        
        for i, (question, output) in enumerate(zip(test_questions, batched_outputs)):
            print(f"\n   Batch Sample {i+1}:")
            print(f"   Question: {question[:50]}...")
            print(f"   Output: '{output[:100]}...' if len(output) > 100 else '{output}'")
            
            parsed = llm.parse_answer(output)
            print(f"   Parsed: {parsed}")
    
    except Exception as e:
        print(f"   ❌ Error during batched generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the answer() method
    print("\n" + "=" * 80)
    print("5. Testing answer() method (used by benchmark)...")
    print("=" * 80)
    
    test_questions = [testset[i][0] for i in range(min(3, len(testset)))]
    try:
        answers = llm.answer(*test_questions)
        print(f"   ✅ answer() method completed")
        print(f"   Number of answers: {len(answers)}")
        
        for i, (question, answer, correct) in enumerate(zip(test_questions, answers, [testset[i][1] for i in range(len(test_questions))])):
            print(f"\n   Answer Sample {i+1}:")
            print(f"   Question: {question[:50]}...")
            print(f"   Model Answer: {answer}")
            print(f"   Correct Answer: {correct}")
            
            if answer != answer:  # NaN check
                print("   ❌ Answer is NaN")
            else:
                from homework.data import is_answer_valid
                is_correct = is_answer_valid(answer, correct)
                print(f"   {'✅' if is_correct else '❌'} Answer is {'CORRECT' if is_correct else 'INCORRECT'}")
    
    except Exception as e:
        print(f"   ❌ Error during answer() method: {e}")
        import traceback
        traceback.print_exc()
    
    # Check model state
    print("\n" + "=" * 80)
    print("6. Model Configuration Check")
    print("=" * 80)
    
    print(f"   Device: {llm.device}")
    print(f"   Model type: {type(llm.model)}")
    print(f"   Model training mode: {llm.model.training}")
    
    # Check if LoRA adapter is loaded
    if hasattr(llm.model, 'peft_config'):
        print(f"   ✅ LoRA adapter detected")
        print(f"   LoRA config: {llm.model.peft_config}")
    else:
        print(f"   ⚠️  No LoRA adapter detected - model might not be trained!")
    
    # Check tokenizer
    print(f"\n   Tokenizer pad_token: {llm.tokenizer.pad_token}")
    print(f"   Tokenizer eos_token: {llm.tokenizer.eos_token}")
    print(f"   Tokenizer pad_token_id: {llm.tokenizer.pad_token_id}")
    print(f"   Tokenizer eos_token_id: {llm.tokenizer.eos_token_id}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nKey things to check:")
    print("1. Are raw model outputs empty?")
    print("2. Do outputs contain <answer> tags?")
    print("3. Are parsed answers NaN?")
    print("4. Is the LoRA adapter actually loaded?")
    print("5. Is the model in eval mode? (should be True)")


if __name__ == "__main__":
    diagnose_model()
