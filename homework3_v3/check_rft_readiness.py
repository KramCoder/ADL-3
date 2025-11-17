#!/usr/bin/env python3
"""
Comprehensive check for RFT training readiness based on the advice:
1. Ensure 850-900+ training QA pairs can be generated
2. Check CoT model accuracy to minimize rejections
3. Verify training data has full reasoning text with answer tags
4. Check accuracies are well above threshold boundaries
"""

import json
from pathlib import Path


def check_cot_accuracy():
    """Check CoT model accuracy."""
    print("=" * 70)
    print("1. CHECKING COT MODEL ACCURACY")
    print("=" * 70)
    
    from homework.cot import CoTModel
    from homework.data import Dataset, benchmark
    
    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    
    print(f"CoT Accuracy: {benchmark_result.accuracy:.3f}")
    print(f"CoT Answer Rate: {benchmark_result.answer_rate:.3f}")
    print(f"Threshold for full points: 0.40")
    
    if benchmark_result.accuracy >= 0.40:
        print("✓ CoT accuracy is at or above threshold")
    else:
        print(f"✗ CoT accuracy is {0.40 - benchmark_result.accuracy:.3f} below threshold")
        print("  Suggestion: Improve CoT model prompting/examples to reduce rejections")
    
    return benchmark_result.accuracy


def check_datagen_capability(oversample=10):
    """Check how many training examples can be generated with current CoT model."""
    print("\n" + "=" * 70)
    print(f"2. ESTIMATING RFT DATASET SIZE (oversample={oversample})")
    print("=" * 70)
    
    from homework.cot import CoTModel
    from homework.data import Dataset, is_answer_valid
    from tqdm import tqdm
    
    dataset = Dataset("train")
    model = CoTModel()
    
    # Sample a subset to estimate
    sample_size = 100
    successful_generations = 0
    
    print(f"Testing on {sample_size} samples to estimate success rate...")
    
    for idx in tqdm(range(min(sample_size, len(dataset)))):
        question, correct_answer = dataset[idx][:2]
        
        # Generate multiple attempts
        generations = model.batched_generate(
            [question],
            num_return_sequences=oversample,
            temperature=0.6
        )[0]
        
        # Check if any generation has correct answer
        found_correct = False
        for reasoning in generations:
            parsed_answer = model.parse_answer(reasoning)
            if is_answer_valid(parsed_answer, correct_answer):
                found_correct = True
                break
        
        if found_correct:
            successful_generations += 1
    
    success_rate = successful_generations / sample_size
    total_questions = len(dataset)
    estimated_rft_size = int(total_questions * success_rate)
    
    print(f"\nSuccess Rate: {success_rate:.1%}")
    print(f"Total Training Questions: {total_questions}")
    print(f"Estimated RFT Dataset Size: {estimated_rft_size}")
    print(f"Target: 850-900+ examples")
    
    if estimated_rft_size >= 850:
        print(f"✓ Expected to generate enough training examples ({estimated_rft_size} >= 850)")
    else:
        shortfall = 850 - estimated_rft_size
        print(f"✗ May fall short by {shortfall} examples")
        print(f"  Suggestions:")
        print(f"    - Improve CoT accuracy (current success rate: {success_rate:.1%})")
        print(f"    - Increase oversample parameter (current: {oversample})")
        
        # Calculate required oversample
        if success_rate > 0:
            required_success_rate = 850 / total_questions
            if required_success_rate <= 1.0:
                print(f"    - Need {required_success_rate:.1%} success rate for 850 examples")
    
    return estimated_rft_size


def check_existing_rft_data():
    """Check existing RFT dataset if it exists."""
    print("\n" + "=" * 70)
    print("3. CHECKING EXISTING RFT DATASET")
    print("=" * 70)
    
    rft_path = Path(__file__).parent / "data" / "rft.json"
    
    if not rft_path.exists():
        print("✗ RFT dataset not found at data/rft.json")
        print("  Run: python -m homework.datagen data/rft.json")
        return None
    
    with rft_path.open() as f:
        rft_data = json.load(f)
    
    print(f"✓ RFT dataset found with {len(rft_data)} examples")
    
    # Check data format
    print("\nChecking data format (first 3 examples):")
    for i, example in enumerate(rft_data[:3]):
        question, answer, reasoning = example
        print(f"\nExample {i+1}:")
        print(f"  Question: {question[:60]}...")
        print(f"  Correct Answer: {answer}")
        print(f"  Reasoning length: {len(reasoning)} chars")
        
        # Check if reasoning has answer tags
        has_answer_tags = "<answer>" in reasoning and "</answer>" in reasoning
        print(f"  Has <answer> tags: {'✓' if has_answer_tags else '✗'}")
        
        if not has_answer_tags:
            print(f"  ✗ WARNING: Reasoning missing answer tags!")
            print(f"  Reasoning: {reasoning[:100]}...")
    
    # Check overall quality
    missing_tags = 0
    for example in rft_data:
        reasoning = example[2]
        if "<answer>" not in reasoning or "</answer>" not in reasoning:
            missing_tags += 1
    
    if missing_tags > 0:
        print(f"\n✗ WARNING: {missing_tags}/{len(rft_data)} examples missing answer tags!")
    else:
        print(f"\n✓ All examples have proper answer tags")
    
    if len(rft_data) >= 850:
        print(f"✓ Dataset size ({len(rft_data)}) meets target (850+)")
    else:
        print(f"✗ Dataset size ({len(rft_data)}) below target (850+)")
        print(f"  Shortfall: {850 - len(rft_data)} examples")
    
    return len(rft_data)


def check_model_accuracies():
    """Check accuracies of all models against thresholds."""
    print("\n" + "=" * 70)
    print("4. CHECKING MODEL ACCURACIES VS THRESHOLDS")
    print("=" * 70)
    
    from homework.data import Dataset, benchmark
    
    testset = Dataset("valid")
    
    # Define thresholds (from grader/tests.py)
    thresholds = {
        "CoT": (0.0, 0.4),
        "SFT": (0.4, 0.6),
        "RFT": (0.6, 0.7),
    }
    
    results = {}
    
    # Check CoT
    print("\nCoT Model:")
    try:
        from homework.cot import CoTModel
        model = CoTModel()
        result = benchmark(model, testset, 100)
        results["CoT"] = result.accuracy
        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  Threshold: {thresholds['CoT'][1]:.2f} for full points")
        margin = result.accuracy - thresholds["CoT"][1]
        if margin >= 0:
            print(f"  ✓ Above threshold by {margin:.3f}")
        else:
            print(f"  ✗ Below threshold by {-margin:.3f}")
    except Exception as e:
        print(f"  ✗ Error loading CoT model: {e}")
    
    # Check SFT
    print("\nSFT Model:")
    try:
        from homework import load_sft
        model = load_sft()
        result = benchmark(model, testset, 100)
        results["SFT"] = result.accuracy
        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  Threshold: {thresholds['SFT'][1]:.2f} for full points")
        margin = result.accuracy - thresholds["SFT"][1]
        if margin >= 0:
            print(f"  ✓ Above threshold by {margin:.3f}")
            if margin < 0.05:
                print(f"  ⚠ WARNING: Margin is small ({margin:.3f}), aim for > 0.05 safety buffer")
        else:
            print(f"  ✗ Below threshold by {-margin:.3f}")
            print(f"  CRITICAL: SFT needs to be well above threshold for good RFT training!")
    except Exception as e:
        print(f"  ✗ Error loading SFT model: {e}")
        print(f"  (This is expected if SFT model hasn't been trained yet)")
    
    # Check RFT
    print("\nRFT Model:")
    try:
        from homework import load_rft
        model = load_rft()
        result = benchmark(model, testset, 100)
        results["RFT"] = result.accuracy
        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  Threshold: {thresholds['RFT'][1]:.2f} for full points")
        margin = result.accuracy - thresholds["RFT"][1]
        if margin >= 0:
            print(f"  ✓ Above threshold by {margin:.3f}")
        else:
            print(f"  ✗ Below threshold by {-margin:.3f}")
    except Exception as e:
        print(f"  ✗ Error loading RFT model: {e}")
        print(f"  (This is expected if RFT model hasn't been trained yet)")
    
    return results


def generate_recommendations(cot_acc, estimated_size):
    """Generate actionable recommendations."""
    print("\n" + "=" * 70)
    print("5. RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    # Check CoT accuracy
    if cot_acc < 0.45:
        recommendations.append(
            "1. IMPROVE COT MODEL:\n"
            f"   Current: {cot_acc:.3f}, Target: 0.45+\n"
            "   - Review and improve few-shot examples in cot.py\n"
            "   - Ensure examples cover diverse conversion types\n"
            "   - Test with different prompting strategies"
        )
    
    # Check dataset size
    if estimated_size < 850:
        recommendations.append(
            f"2. INCREASE RFT DATASET SIZE:\n"
            f"   Current estimate: {estimated_size}, Target: 850-900+\n"
            "   - Increase oversample parameter in datagen (try 15-20)\n"
            "   - Improve CoT accuracy to reduce rejections\n"
            "   - Consider multiple generation passes"
        )
    
    if not recommendations:
        recommendations.append(
            "✓ System appears ready for RFT training!\n"
            "  Next steps:\n"
            "  1. Generate RFT dataset: python -m homework.datagen data/rft.json --oversample 10\n"
            "  2. Train RFT model: python -m homework.rft train\n"
            "  3. Verify RFT accuracy is > 0.70"
        )
    
    for rec in recommendations:
        print(f"\n{rec}")


def main():
    print("RFT TRAINING READINESS CHECK")
    print("=" * 70)
    
    # Run all checks
    cot_acc = check_cot_accuracy()
    estimated_size = check_datagen_capability(oversample=10)
    actual_size = check_existing_rft_data()
    accuracies = check_model_accuracies()
    
    # Generate recommendations
    generate_recommendations(cot_acc, estimated_size if actual_size is None else actual_size)
    
    print("\n" + "=" * 70)
    print("Check complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
