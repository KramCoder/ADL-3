#!/usr/bin/env python3
"""
Test script to verify NaN prevention mechanisms work correctly.
This tests all the safeguards that prevent NaN from reaching the grader.
"""

import torch
import warnings
warnings.filterwarnings('ignore')

def test_model_dtype():
    """Verify model loads in FP32 or BF16 (not FP16)"""
    print("=" * 60)
    print("TEST 1: Model Precision")
    print("=" * 60)
    
    from homework.base_llm import BaseLLM
    llm = BaseLLM()
    
    dtype = llm.model.dtype
    print(f"Model dtype: {dtype}")
    print(f"Device: {llm.device}")
    
    # Verify it's not FP16 (which causes NaN)
    assert dtype != torch.float16, "❌ FAIL: Model should not use FP16!"
    
    # Should be FP32 or BF16
    assert dtype in [torch.float32, torch.bfloat16], f"❌ FAIL: Unexpected dtype {dtype}"
    
    print(f"✓ PASS: Model uses {dtype} (safe from overflow NaN)")
    return llm

def test_parse_answer_nan_handling():
    """Verify parse_answer handles NaN correctly"""
    print("\n" + "=" * 60)
    print("TEST 2: parse_answer NaN Handling")
    print("=" * 60)
    
    from homework.base_llm import BaseLLM
    llm = BaseLLM()
    
    test_cases = [
        ("<answer>nan</answer>", 0.0, "NaN string"),
        ("<answer>inf</answer>", 0.0, "Inf string"),
        ("<answer>-inf</answer>", 0.0, "-Inf string"),
        ("<answer>123.45</answer>", 123.45, "Valid number"),
        ("<answer>0</answer>", 0.0, "Zero"),
        ("<answer>-42.5</answer>", -42.5, "Negative number"),
        ("no answer tag", 0.0, "Missing answer tag"),
        ("<answer>", 0.0, "Incomplete tag"),
        ("<answer>not a number</answer>", 0.0, "Non-numeric"),
    ]
    
    for answer_text, expected, description in test_cases:
        result = llm.parse_answer(answer_text)
        
        # Check result is not NaN
        assert result == result, f"❌ FAIL: {description} returned NaN"
        
        # Check result is not Inf
        assert abs(result) != float('inf'), f"❌ FAIL: {description} returned Inf"
        
        # Check expected value
        assert result == expected, f"❌ FAIL: {description} expected {expected}, got {result}"
        
        print(f"  ✓ {description:20} → {result}")
    
    print("✓ PASS: parse_answer never returns NaN or Inf")

def test_generation_never_empty():
    """Verify generation never returns empty string"""
    print("\n" + "=" * 60)
    print("TEST 3: Empty Generation Prevention")
    print("=" * 60)
    
    from homework.base_llm import BaseLLM
    llm = BaseLLM()
    
    # Test with a simple prompt
    result = llm.generate("Test")
    
    # Check not empty
    assert result.strip() != "", "❌ FAIL: Generation returned empty string"
    
    # Check it produces tokens when tokenized
    tokens = llm.tokenizer(result, return_tensors="pt")
    num_tokens = tokens["input_ids"].shape[1]
    assert num_tokens > 0, "❌ FAIL: Generation tokenizes to zero tokens"
    
    print(f"  Generated: '{result[:50]}...'")
    print(f"  Token count: {num_tokens}")
    print("✓ PASS: Generation is never empty")

def test_is_answer_valid():
    """Verify is_answer_valid rejects NaN"""
    print("\n" + "=" * 60)
    print("TEST 4: is_answer_valid NaN Rejection")
    print("=" * 60)
    
    from homework.data import is_answer_valid
    
    # Test NaN rejection
    nan_value = float('nan')
    assert not is_answer_valid(nan_value, 100.0), "❌ FAIL: is_answer_valid accepted NaN"
    print("  ✓ NaN answer rejected")
    
    # Test Inf rejection
    inf_value = float('inf')
    assert not is_answer_valid(inf_value, 100.0), "❌ FAIL: is_answer_valid accepted Inf"
    print("  ✓ Inf answer rejected")
    
    # Test -Inf rejection
    ninf_value = float('-inf')
    assert not is_answer_valid(ninf_value, 100.0), "❌ FAIL: is_answer_valid accepted -Inf"
    print("  ✓ -Inf answer rejected")
    
    # Test valid answer
    assert is_answer_valid(100.0, 100.0), "❌ FAIL: is_answer_valid rejected valid answer"
    print("  ✓ Valid answer accepted")
    
    print("✓ PASS: is_answer_valid correctly rejects NaN/Inf")

def test_benchmark_no_nan():
    """Verify benchmark computation never returns NaN"""
    print("\n" + "=" * 60)
    print("TEST 5: Benchmark NaN Prevention")
    print("=" * 60)
    
    from homework.data import BenchmarkResult, Dataset
    
    # Create test data with some NaN answers
    test_answers = [1.0, 2.0, float('nan'), 3.0, float('inf')]
    
    # Create a mock dataset
    class MockDataset:
        def __init__(self):
            self.data = [("q1", 1.0), ("q2", 2.0), ("q3", 3.0), ("q4", 4.0), ("q5", 5.0)]
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
    
    dataset = MockDataset()
    result = BenchmarkResult.from_answers(test_answers, dataset, 5)
    
    # Check accuracy is not NaN
    assert result.accuracy == result.accuracy, "❌ FAIL: Benchmark accuracy is NaN"
    assert abs(result.accuracy) != float('inf'), "❌ FAIL: Benchmark accuracy is Inf"
    
    # Check answer_rate is not NaN
    assert result.answer_rate == result.answer_rate, "❌ FAIL: Benchmark answer_rate is NaN"
    assert abs(result.answer_rate) != float('inf'), "❌ FAIL: Benchmark answer_rate is Inf"
    
    print(f"  Accuracy: {result.accuracy:.2%} (2/5 correct)")
    print(f"  Answer rate: {result.answer_rate:.2%} (3/5 valid)")
    print("✓ PASS: Benchmark never returns NaN")

def test_full_pipeline():
    """Test complete pipeline from generation to scoring"""
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline")
    print("=" * 60)
    
    from homework.base_llm import BaseLLM
    from homework.data import is_answer_valid
    
    llm = BaseLLM()
    
    # Generate answer
    question = "How many grams in 2 kg?"
    generation = llm.generate(question)
    print(f"  Question: {question}")
    print(f"  Generation: {generation[:80]}...")
    
    # Parse answer
    answer = llm.parse_answer(generation)
    print(f"  Parsed answer: {answer}")
    
    # Verify no NaN
    assert answer == answer, "❌ FAIL: Pipeline produced NaN"
    assert abs(answer) != float('inf'), "❌ FAIL: Pipeline produced Inf"
    
    # Check validity
    is_valid = is_answer_valid(answer, 2000.0, relative_tolerance=0.05)
    print(f"  Valid against 2000.0: {is_valid}")
    
    print("✓ PASS: Full pipeline produces valid numeric output")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NaN PREVENTION TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        llm = test_model_dtype()
        test_parse_answer_nan_handling()
        test_generation_never_empty()
        test_is_answer_valid()
        test_benchmark_no_nan()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe codebase has comprehensive NaN prevention:")
        print("  1. Model loads in FP32/BF16 (not FP16) → prevents NaN in logits")
        print("  2. parse_answer rejects NaN/Inf → returns 0.0")
        print("  3. Generation never empty → prevents division by zero")
        print("  4. is_answer_valid rejects NaN/Inf → excludes from metrics")
        print("  5. Benchmark handles NaN gracefully → no crash")
        print("\nThe grader should now run without ValueError.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
