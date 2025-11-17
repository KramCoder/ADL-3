#!/usr/bin/env python3
"""Diagnostic script to investigate SFT model accuracy issues."""

import sys
from pathlib import Path

# Add the homework directory to path
sys.path.insert(0, str(Path(__file__).parent))

from homework.sft import load
from homework.data import Dataset, benchmark, is_answer_valid

def diagnose_model():
    """Run diagnostics on the SFT model."""
    print("Loading SFT model...")
    model = load()
    
    print("\n" + "="*60)
    print("Testing on validation set (first 20 samples)")
    print("="*60)
    
    testset = Dataset("valid")
    questions = [testset[i][0] for i in range(min(20, len(testset)))]
    correct_answers = [testset[i][1] for i in range(min(20, len(testset)))]
    
    print(f"\nTesting {len(questions)} questions...")
    
    # Get raw generations
    generations = model.batched_generate(questions)
    
    # Parse answers
    parsed_answers = [model.parse_answer(g) for g in generations]
    
    # Analyze results
    correct_count = 0
    parse_failures = 0
    nan_answers = 0
    
    print("\nDetailed Results:")
    print("-" * 60)
    
    for i, (question, correct_answer, generation, parsed_answer) in enumerate(
        zip(questions, correct_answers, generations, parsed_answers)
    ):
        is_valid = is_answer_valid(parsed_answer, correct_answer)
        if is_valid:
            correct_count += 1
        
        # Check for parsing issues
        if parsed_answer == 0.0 and correct_answer != 0.0:
            if "<answer>" not in generation or "</answer>" not in generation:
                parse_failures += 1
            elif parsed_answer != parsed_answer:  # NaN check
                nan_answers += 1
        
        # Print first 10 examples
        if i < 10:
            print(f"\nQuestion {i+1}: {question}")
            print(f"  Correct answer: {correct_answer}")
            print(f"  Generated: {generation[:200]}...")  # First 200 chars
            print(f"  Parsed answer: {parsed_answer}")
            print(f"  Valid: {is_valid}")
            if not is_valid and parsed_answer != 0.0:
                diff = abs(parsed_answer - correct_answer)
                rel_diff = diff / abs(correct_answer) if correct_answer != 0 else diff
                print(f"  Difference: {diff} (relative: {rel_diff:.2%})")
    
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Total questions tested: {len(questions)}")
    print(f"Correct answers: {correct_count} ({correct_count/len(questions)*100:.1f}%)")
    print(f"Parse failures (returned 0.0): {parse_failures}")
    print(f"NaN answers: {nan_answers}")
    
    # Check generation format
    print("\n" + "="*60)
    print("Generation Format Analysis:")
    print("="*60)
    has_closing_tag = sum(1 for g in generations if "</answer>" in g)
    has_opening_tag = sum(1 for g in generations if "<answer>" in g)
    print(f"Generations with <answer> tag: {has_opening_tag}/{len(generations)}")
    print(f"Generations with </answer> tag: {has_closing_tag}/{len(generations)}")
    
    # Check answer format
    print("\n" + "="*60)
    print("Answer Format Examples:")
    print("="*60)
    for i, (gen, parsed) in enumerate(zip(generations[:5], parsed_answers[:5])):
        print(f"\nExample {i+1}:")
        print(f"  Full generation: {gen}")
        print(f"  Parsed: {parsed}")
    
    # Run full benchmark
    print("\n" + "="*60)
    print("Full Benchmark (100 samples):")
    print("="*60)
    benchmark_result = benchmark(model, testset, 100)
    print(f"Accuracy: {benchmark_result.accuracy:.4f}")
    print(f"Answer rate: {benchmark_result.answer_rate:.4f}")
    
    # Show some incorrect examples
    print("\n" + "="*60)
    print("Sample Incorrect Answers:")
    print("="*60)
    incorrect_samples = [s for s in benchmark_result.samples if not s.is_correct][:5]
    for sample in incorrect_samples:
        print(f"\nQuestion: {sample.question}")
        print(f"  Expected: {sample.correct_answer}")
        print(f"  Got: {sample.answer}")
        if sample.answer == 0.0:
            print(f"  Issue: Parse failure (likely missing </answer> tag)")

if __name__ == "__main__":
    diagnose_model()
