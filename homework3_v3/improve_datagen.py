#!/usr/bin/env python3
"""
Script to help optimize RFT data generation.
Provides recommendations based on current state.
"""
import json
import sys
from pathlib import Path


def analyze_datagen_strategy():
    """Analyze current state and recommend datagen strategy."""
    print("🔍 RFT Data Generation Strategy Analyzer")
    print("=" * 60)
    
    # Check if RFT data exists
    rft_path = Path("data/rft.json")
    train_path = Path("data/train.json")
    
    with open(train_path) as f:
        train_data = json.load(f)
    total_questions = len(train_data)
    
    print(f"Training questions available: {total_questions}")
    
    if rft_path.exists():
        with open(rft_path) as f:
            rft_data = json.load(f)
        current_examples = len(rft_data)
        print(f"Current RFT examples: {current_examples}")
        
        success_rate = current_examples / total_questions * 100
        print(f"Current success rate: {success_rate:.1f}%")
    else:
        print("RFT dataset: Not yet generated")
        current_examples = 0
        success_rate = 0
    
    print()
    print("-" * 60)
    print("📊 Target Analysis")
    print("-" * 60)
    
    targets = [850, 900, 950]
    
    print(f"\n{'Target':<10} {'Success Rate':<15} {'Oversample':<15} {'Status'}")
    print("-" * 60)
    
    for target in targets:
        required_rate = target / total_questions * 100
        
        # Calculate recommended oversample
        # If CoT has ~50% accuracy, oversample=2 gives ~50% success
        # If CoT has ~10% accuracy, oversample=10 gives ~65% success
        # If CoT has ~5% accuracy, oversample=20 gives ~64% success
        
        if success_rate > 0:
            # Estimate based on current success rate
            if success_rate >= required_rate:
                recommended_oversample = "Current OK"
                status = "✅"
            else:
                # Rough estimate: need to generate more samples
                multiplier = required_rate / success_rate
                recommended_oversample = f"~{int(10 * multiplier)}"
                status = "⚠️ "
        else:
            # No data yet, provide general recommendations
            if required_rate >= 90:
                recommended_oversample = "5-10"
                status = "🎯"
            elif required_rate >= 85:
                recommended_oversample = "10-12"
                status = "🎯"
            else:
                recommended_oversample = "12-15"
                status = "🎯"
        
        print(f"{target:<10} {required_rate:>6.1f}%       {recommended_oversample:<15} {status}")
    
    print()
    print("-" * 60)
    print("💡 Recommendations")
    print("-" * 60)
    
    if current_examples == 0:
        print("\n1️⃣  First Time Generation:")
        print("   Start with moderate oversample to see success rate:")
        print("   $ python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6")
        print()
        print("   Then run validation:")
        print("   $ python3 validate_rft_data.py")
        
    elif current_examples < 850:
        shortfall = 850 - current_examples
        print(f"\n⚠️  Need {shortfall} more examples to reach 850 target")
        print()
        
        # Estimate needed oversample increase
        if success_rate > 0:
            # Current: success_rate% with some oversample
            # Need: 85% success rate
            # Estimate current oversample from success rate
            # (rough heuristic: success_rate ≈ 1 - (1 - cot_acc)^oversample)
            
            ratio = 85 / success_rate if success_rate > 0 else 2
            if ratio > 1.5:
                print("   Options:")
                print(f"   A) Increase oversample significantly (try 15-20)")
                print("   B) Improve CoT model accuracy first")
                print()
                print("   Try option A first:")
                print("   $ python3 -m homework.datagen data/rft.json --oversample 15 --temperature 0.6")
            else:
                print("   Try slightly higher oversample:")
                print("   $ python3 -m homework.datagen data/rft.json --oversample 12 --temperature 0.6")
        
    elif current_examples < 900:
        print(f"\n✅ Good! You have {current_examples} examples (target: 850-900)")
        print("   Consider generating a bit more for safety:")
        print("   $ python3 -m homework.datagen data/rft.json --oversample 12 --temperature 0.6")
        
    else:
        print(f"\n✅ EXCELLENT! You have {current_examples} examples")
        print("   This is above the recommended 850-900 range.")
        print("   You're ready for training!")
    
    print()
    print("-" * 60)
    print("🎯 Temperature Recommendations")
    print("-" * 60)
    print("Temperature affects diversity vs. consistency:")
    print()
    print("  Low (0.3-0.5):  More consistent, higher success rate")
    print("                  Use if you're getting enough examples")
    print()
    print("  Medium (0.6):   Balanced (current setting)")
    print("                  Good default choice")
    print()
    print("  High (0.7-0.8): More diverse reasoning paths")
    print("                  Use if you need more variety")
    print()
    
    print("-" * 60)
    print("📝 Quality Checks")
    print("-" * 60)
    print("After generation, always validate:")
    print()
    print("1. Check data count and quality:")
    print("   $ python3 validate_rft_data.py")
    print()
    print("2. Manually inspect a few examples:")
    print("   $ python3 -c \"import json; d=json.load(open('data/rft.json')); print(d[0])\"")
    print()
    print("3. Ensure reasoning includes:")
    print("   ✓ Step-by-step calculation")
    print("   ✓ Both <answer> and </answer> tags")
    print("   ✓ Correct numerical answer")
    print()


if __name__ == "__main__":
    analyze_datagen_strategy()
