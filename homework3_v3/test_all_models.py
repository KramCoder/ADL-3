#!/usr/bin/env python3
"""
Script to test all models and compare against thresholds.
"""
import sys
from pathlib import Path

# Add homework to path
sys.path.insert(0, str(Path(__file__).parent))

from homework.data import Dataset, benchmark
from homework.cot import CoTModel
from homework.sft import load as load_sft
from homework.rft import load as load_rft


# Thresholds from grader/tests.py
THRESHOLDS = {
    "cot": {"min": 0.0, "target": 0.4, "name": "CoT Model"},
    "sft": {"min": 0.4, "target": 0.6, "name": "SFT Model"},
    "rft": {"min": 0.6, "target": 0.7, "name": "RFT Model"},
}


def test_model(model, model_key, dataset, num_questions=100):
    """Test a model and compare to thresholds."""
    print(f"\n{'='*60}")
    print(f"Testing {THRESHOLDS[model_key]['name']}")
    print(f"{'='*60}")
    
    try:
        result = benchmark(model, dataset, num_questions)
        accuracy = result.accuracy
        answer_rate = result.answer_rate
        
        min_acc = THRESHOLDS[model_key]["min"]
        target_acc = THRESHOLDS[model_key]["target"]
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Answer Rate: {answer_rate:.3f}")
        print()
        
        # Determine status
        if accuracy >= target_acc:
            status = "✅ EXCELLENT - Above target!"
            safety_margin = accuracy - target_acc
            print(f"  {status}")
            print(f"  Safety margin: +{safety_margin:.3f} above target")
        elif accuracy >= (min_acc + target_acc) / 2:
            status = "⚠️  ACCEPTABLE - Between min and target"
            gap = target_acc - accuracy
            print(f"  {status}")
            print(f"  Need: +{gap:.3f} to reach target")
        elif accuracy >= min_acc:
            status = "⚠️  MINIMAL - At minimum threshold"
            gap = target_acc - accuracy
            print(f"  {status}")
            print(f"  Need: +{gap:.3f} to reach target")
        else:
            status = "❌ BELOW MINIMUM"
            gap = target_acc - accuracy
            print(f"  {status}")
            print(f"  Need: +{gap:.3f} to reach target")
        
        return accuracy, answer_rate, status
    
    except Exception as e:
        print(f"  ❌ Error testing model: {e}")
        return 0.0, 0.0, "ERROR"


def main():
    print("🧪 Testing All Models")
    print()
    
    dataset = Dataset("valid")
    results = {}
    
    # Test CoT
    print("Loading CoT model...")
    try:
        cot_model = CoTModel()
        acc, ans_rate, status = test_model(cot_model, "cot", dataset)
        results["cot"] = {"accuracy": acc, "answer_rate": ans_rate, "status": status}
    except Exception as e:
        print(f"❌ Failed to load CoT model: {e}")
        results["cot"] = {"accuracy": 0.0, "answer_rate": 0.0, "status": "FAILED"}
    
    # Test SFT
    print("\nLoading SFT model...")
    try:
        sft_model = load_sft()
        acc, ans_rate, status = test_model(sft_model, "sft", dataset)
        results["sft"] = {"accuracy": acc, "answer_rate": ans_rate, "status": status}
    except Exception as e:
        print(f"❌ Failed to load SFT model: {e}")
        results["sft"] = {"accuracy": 0.0, "answer_rate": 0.0, "status": "FAILED"}
    
    # Test RFT
    print("\nLoading RFT model...")
    try:
        rft_model = load_rft()
        acc, ans_rate, status = test_model(rft_model, "rft", dataset)
        results["rft"] = {"accuracy": acc, "answer_rate": ans_rate, "status": status}
    except Exception as e:
        print(f"❌ Failed to load RFT model: {e}")
        results["rft"] = {"accuracy": 0.0, "answer_rate": 0.0, "status": "FAILED"}
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Accuracy':<12} {'Target':<12} {'Status':<20}")
    print("-" * 60)
    
    for key in ["cot", "sft", "rft"]:
        acc = results[key]["accuracy"]
        target = THRESHOLDS[key]["target"]
        status_short = "✅" if "EXCELLENT" in results[key]["status"] else \
                      "⚠️ " if "ACCEPTABLE" in results[key]["status"] or "MINIMAL" in results[key]["status"] else \
                      "❌"
        print(f"{key.upper():<10} {acc:<12.3f} {target:<12.1f} {status_short} {results[key]['status'][:15]}")
    
    print()
    print("Recommendations:")
    print("-" * 60)
    
    # CoT recommendations
    if results["cot"]["accuracy"] < 0.35:
        print("❗ CoT accuracy is low. This will cause high rejection during datagen.")
        print("   Consider:")
        print("   - Improving CoT prompt/examples")
        print("   - Using higher oversample (15-20) for datagen")
    elif results["cot"]["accuracy"] < 0.4:
        print("⚠️  CoT is close to minimum. Use oversample=10-15 for datagen")
    else:
        print("✅ CoT looks good for data generation!")
    
    # SFT recommendations
    if results["sft"]["accuracy"] < 0.5:
        print("⚠️  SFT needs improvement. Consider:")
        print("   - Checking training data quality")
        print("   - Training for more epochs")
        print("   - Adjusting learning rate")
    else:
        print("✅ SFT looks good!")
    
    # RFT recommendations  
    if results["rft"]["accuracy"] < 0.65:
        print("⚠️  RFT needs improvement. Ensure:")
        print("   - RFT dataset has 850-900 examples")
        print("   - Training data includes full reasoning")
        print("   - Training for sufficient epochs")
    else:
        print("✅ RFT looks good!")
    
    print()


if __name__ == "__main__":
    main()
