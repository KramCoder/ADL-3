#!/usr/bin/env python3
"""
Script to validate RFT training data quality.
Checks for:
1. Presence of answer tags
2. Reasoning length
3. Answer tag format
4. Overall dataset size
"""
import json
import sys
from pathlib import Path


def validate_rft_data(filepath: str = "data/rft.json"):
    """Validate RFT dataset quality."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ Error: {filepath} not found!")
        print("Generate RFT data with:")
        print("  python3 -m homework.datagen data/rft.json --oversample 10 --temperature 0.6")
        return 0, 0
    
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"📊 RFT Dataset Validation Report")
    print(f"=" * 60)
    print(f"Total examples: {len(data)}")
    print()
    
    # Analyze target
    if len(data) >= 900:
        status = "✅ EXCELLENT"
    elif len(data) >= 850:
        status = "✅ GOOD"
    elif len(data) >= 700:
        status = "⚠️  ACCEPTABLE (but could be better)"
    else:
        status = "❌ TOO FEW (need 850-900)"
    print(f"Dataset size: {status}")
    print()
    
    issues = []
    warnings = []
    
    for i, item in enumerate(data):
        if len(item) != 3:
            issues.append(f"Example {i}: Wrong format (expected [question, answer, reasoning])")
            continue
            
        question, answer, reasoning = item
        
        # Check answer tags
        if '<answer>' not in reasoning:
            issues.append(f"Example {i}: Missing <answer> tag")
        if '</answer>' not in reasoning:
            issues.append(f"Example {i}: Missing </answer> tag")
        
        # Check for reasoning content before answer tag
        if '<answer>' in reasoning:
            text_before_answer = reasoning.split('<answer>')[0]
            if len(text_before_answer.strip()) < 10:
                warnings.append(f"Example {i}: Very short reasoning before answer")
        
        # Check for reasoning length
        if len(reasoning) < 20:
            issues.append(f"Example {i}: Reasoning too short ({len(reasoning)} chars)")
        
        # Check answer tag count
        if reasoning.count('<answer>') != 1:
            issues.append(f"Example {i}: Wrong number of <answer> tags ({reasoning.count('<answer>')})")
        if reasoning.count('</answer>') != 1:
            issues.append(f"Example {i}: Wrong number of </answer> tags ({reasoning.count('</answer>')})")
        
        # Check that question is not empty
        if not question or len(question.strip()) < 5:
            issues.append(f"Example {i}: Question is empty or too short")
    
    print(f"Critical Issues: {len(issues)}")
    if issues:
        print("First 10 critical issues:")
        for issue in issues[:10]:
            print(f"  ❌ {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("  ✅ No critical issues found!")
    print()
    
    print(f"Warnings: {len(warnings)}")
    if warnings:
        print("First 10 warnings:")
        for warning in warnings[:10]:
            print(f"  ⚠️  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    else:
        print("  ✅ No warnings!")
    print()
    
    # Show sample examples
    print("Sample Examples:")
    print("-" * 60)
    for i in range(min(3, len(data))):
        question, answer, reasoning = data[i]
        print(f"\nExample {i+1}:")
        print(f"  Question: {question[:60]}...")
        print(f"  Answer: {answer}")
        print(f"  Reasoning: {reasoning[:100]}...")
    print()
    
    # Calculate rejection rate
    total_possible = 1000  # Assuming 1000 training questions
    generated = len(data)
    rejection_rate = (total_possible - generated) / total_possible * 100
    
    print(f"📈 Statistics:")
    print(f"  Generated: {generated}/{total_possible} ({generated/total_possible*100:.1f}%)")
    print(f"  Rejection rate: {rejection_rate:.1f}%")
    
    if rejection_rate > 15:
        print(f"  ⚠️  High rejection rate! Consider:")
        print(f"     - Increasing oversample (try 15-20)")
        print(f"     - Improving CoT model accuracy")
        print(f"     - Adjusting temperature")
    else:
        print(f"  ✅ Good acceptance rate!")
    print()
    
    # Overall status
    print("=" * 60)
    if len(data) >= 850 and len(issues) == 0:
        print("✅ READY FOR TRAINING!")
    elif len(data) >= 850:
        print("⚠️  Sufficient examples but has quality issues")
    elif len(issues) == 0:
        print("⚠️  Good quality but need more examples")
    else:
        print("❌ Need to regenerate data")
    print("=" * 60)
    
    return len(data), len(issues)


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/rft.json"
    validate_rft_data(filepath)
