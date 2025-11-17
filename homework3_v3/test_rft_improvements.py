#!/usr/bin/env python3
"""
Test script for RFT training improvements.
This script validates:
1. CoT model accuracy improvements
2. Data generation quality (850-900+ pairs)
3. RFT training data format
4. Model accuracy thresholds
"""

import json
from pathlib import Path
from homework.data import Dataset, benchmark, is_answer_valid
from homework.cot import CoTModel
from homework.sft import load as load_sft, test_model as test_sft
from homework.rft import load as load_rft, test_model as test_rft


def test_cot_accuracy():
    """Test CoT model accuracy to ensure it's good enough for datagen."""
    print("\n" + "="*60)
    print("TEST 1: CoT Model Accuracy")
    print("="*60)
    
    model = CoTModel()
    dataset = Dataset("valid")
    
    # Test on a sample of questions
    test_size = min(50, len(dataset))
    questions = [dataset[i][0] for i in range(test_size)]
    correct_answers = [dataset[i][1] for i in range(test_size)]
    
    print(f"Testing CoT model on {test_size} validation questions...")
    
    correct_count = 0
    valid_count = 0
    
    for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
        try:
            # Generate answer
            reasoning = model.generate(question)
            parsed = model.parse_answer(reasoning)
            
            # Check if valid (not NaN)
            if parsed == parsed:  # NaN check
                valid_count += 1
                if is_answer_valid(parsed, correct_answer):
                    correct_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{test_size} questions...")
        except Exception as e:
            print(f"  Error on question {i+1}: {e}")
    
    accuracy = correct_count / test_size if test_size > 0 else 0
    valid_rate = valid_count / test_size if test_size > 0 else 0
    
    print(f"\nResults:")
    print(f"  Valid answers: {valid_count}/{test_size} ({valid_rate*100:.1f}%)")
    print(f"  Correct answers: {correct_count}/{test_size} ({accuracy*100:.1f}%)")
    
    # CoT should have at least 30-40% accuracy for good datagen
    if accuracy >= 0.30:
        print(f"  ✓ PASS: CoT accuracy ({accuracy*100:.1f}%) is sufficient for datagen")
        return True
    else:
        print(f"  ✗ FAIL: CoT accuracy ({accuracy*100:.1f}%) is too low. Consider improving prompts.")
        return False


def test_datagen_output():
    """Test that datagen produces 850-900+ QA pairs with proper format."""
    print("\n" + "="*60)
    print("TEST 2: Data Generation Output")
    print("="*60)
    
    rft_data_path = Path(__file__).parent / "data" / "rft.json"
    
    if not rft_data_path.exists():
        print(f"  ✗ FAIL: RFT dataset not found at {rft_data_path}")
        print(f"  Run: python -m homework.datagen data/rft.json")
        return False
    
    with rft_data_path.open() as f:
        rft_data = json.load(f)
    
    print(f"Loaded {len(rft_data)} examples from {rft_data_path}")
    
    # Check count
    if len(rft_data) < 850:
        print(f"  ✗ FAIL: Only {len(rft_data)} examples. Need 850-900+")
        return False
    elif len(rft_data) >= 900:
        print(f"  ✓ PASS: {len(rft_data)} examples (exceeds 900 target)")
    else:
        print(f"  ✓ PASS: {len(rft_data)} examples (meets 850+ target)")
    
    # Check format
    invalid_count = 0
    missing_tags = 0
    missing_reasoning = 0
    
    for i, example in enumerate(rft_data):
        if len(example) < 3:
            print(f"  ✗ Example {i}: Invalid format (expected [question, answer, reasoning])")
            invalid_count += 1
            continue
        
        question, answer, reasoning = example[0], example[1], example[2]
        
        # Check for answer tags
        if "<answer>" not in reasoning or "</answer>" not in reasoning:
            missing_tags += 1
            if missing_tags <= 3:  # Show first few examples
                print(f"  ✗ Example {i}: Missing answer tags in reasoning")
        
        # Check reasoning is not empty
        if not reasoning.strip():
            missing_reasoning += 1
            if missing_reasoning <= 3:
                print(f"  ✗ Example {i}: Empty reasoning text")
        
        # Verify answer can be parsed
        try:
            from homework.cot import CoTModel
            model = CoTModel()
            parsed = model.parse_answer(reasoning)
            if parsed != parsed:  # NaN check
                if invalid_count < 3:
                    print(f"  ✗ Example {i}: Parsed answer is NaN")
                invalid_count += 1
        except Exception as e:
            if invalid_count < 3:
                print(f"  ✗ Example {i}: Error parsing answer: {e}")
            invalid_count += 1
    
    if invalid_count == 0 and missing_tags == 0 and missing_reasoning == 0:
        print(f"  ✓ PASS: All examples have proper format with answer tags")
        return True
    else:
        print(f"  ✗ FAIL: Found {invalid_count} invalid examples, {missing_tags} missing tags, {missing_reasoning} empty reasoning")
        return False


def test_sft_accuracy():
    """Test SFT model accuracy (should exceed 0.6 threshold)."""
    print("\n" + "="*60)
    print("TEST 3: SFT Model Accuracy")
    print("="*60)
    
    try:
        model = load_sft()
        dataset = Dataset("valid")
        result = benchmark(model, dataset, 100)
        
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Answer rate: {result.answer_rate:.4f}")
        
        # SFT threshold is 0.4-0.6, we want to exceed 0.6
        if result.accuracy >= 0.6:
            print(f"  ✓ PASS: SFT accuracy ({result.accuracy:.4f}) exceeds threshold (0.6)")
            return True
        elif result.accuracy >= 0.5:
            print(f"  ⚠ WARNING: SFT accuracy ({result.accuracy:.4f}) is above minimum (0.4) but below target (0.6)")
            print(f"  Consider training longer or adjusting hyperparameters")
            return True  # Still pass, but warn
        else:
            print(f"  ✗ FAIL: SFT accuracy ({result.accuracy:.4f}) is below minimum threshold (0.4)")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: Error loading/testing SFT model: {e}")
        print(f"  Make sure SFT model is trained: python -m homework.sft train")
        return False


def test_rft_accuracy():
    """Test RFT model accuracy (should exceed 0.7 threshold)."""
    print("\n" + "="*60)
    print("TEST 4: RFT Model Accuracy")
    print("="*60)
    
    try:
        model = load_rft()
        dataset = Dataset("valid")
        result = benchmark(model, dataset, 100)
        
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Answer rate: {result.answer_rate:.4f}")
        
        # RFT threshold is 0.6-0.7, we want to exceed 0.7
        if result.accuracy >= 0.7:
            print(f"  ✓ PASS: RFT accuracy ({result.accuracy:.4f}) exceeds threshold (0.7)")
            return True
        elif result.accuracy >= 0.65:
            print(f"  ⚠ WARNING: RFT accuracy ({result.accuracy:.4f}) is above minimum (0.6) but below target (0.7)")
            print(f"  Consider improving datagen or training longer")
            return True  # Still pass, but warn
        else:
            print(f"  ✗ FAIL: RFT accuracy ({result.accuracy:.4f}) is below minimum threshold (0.6)")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: Error loading/testing RFT model: {e}")
        print(f"  Make sure RFT model is trained: python -m homework.rft train")
        return False


def test_cot_reasoning_quality():
    """Test that CoT model generates proper reasoning with answer tags."""
    print("\n" + "="*60)
    print("TEST 5: CoT Reasoning Quality")
    print("="*60)
    
    model = CoTModel()
    test_questions = [
        "How many gram are there per 6 kg?",
        "Convert 5 quart to pint?",
        "How many MB is 2 G?",
    ]
    
    all_good = True
    for question in test_questions:
        try:
            reasoning = model.generate(question)
            print(f"\n  Question: {question}")
            print(f"  Reasoning: {reasoning[:200]}...")  # Show first 200 chars
            
            # Check for answer tags
            if "<answer>" not in reasoning:
                print(f"  ✗ Missing <answer> tag")
                all_good = False
            if "</answer>" not in reasoning:
                print(f"  ✗ Missing </answer> tag")
                all_good = False
            
            # Check reasoning is not just the answer
            if len(reasoning.split("<answer>")[0].strip()) < 10:
                print(f"  ✗ Reasoning too short (should have explanation before answer)")
                all_good = False
            
            # Try to parse
            parsed = model.parse_answer(reasoning)
            if parsed != parsed:  # NaN check
                print(f"  ✗ Parsed answer is NaN")
                all_good = False
            else:
                print(f"  ✓ Parsed answer: {parsed}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_good = False
    
    if all_good:
        print(f"\n  ✓ PASS: CoT generates proper reasoning with answer tags")
    else:
        print(f"\n  ✗ FAIL: CoT reasoning quality issues found")
    
    return all_good


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RFT Training Improvements - Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: CoT accuracy
    results['cot_accuracy'] = test_cot_accuracy()
    
    # Test 2: Datagen output
    results['datagen'] = test_datagen_output()
    
    # Test 3: CoT reasoning quality
    results['cot_reasoning'] = test_cot_reasoning_quality()
    
    # Test 4: SFT accuracy (optional - only if model exists)
    results['sft_accuracy'] = test_sft_accuracy()
    
    # Test 5: RFT accuracy (optional - only if model exists)
    results['rft_accuracy'] = test_rft_accuracy()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Your improvements are working correctly.")
    else:
        print("\n  ⚠ Some tests failed. Review the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
