"""
Quick diagnosis - copy and paste this into your Colab/Jupyter notebook
or run it directly: python3 quick_diagnosis.py
"""

# If running in Colab, make sure you're in the right directory first:
# %cd /content/ADL-3/homework3_v3

from homework.sft import load
from homework.data import Dataset
import sys

print("=" * 80)
print("QUICK SFT DIAGNOSIS")
print("=" * 80)

# Try to load model
try:
    print("\n1. Loading model...")
    llm = load()
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    print("\nPlease check:")
    print("  - Did training complete?")
    print("  - Is the model in homework/sft_model/?")
    sys.exit(1)

# Check prompt formatting
print("\n2. Checking prompt format...")
test_q = "Can you change 2 hour to its equivalent in min?"
formatted = llm.format_prompt(test_q)
print(f"   Original: {test_q!r}")
print(f"   Formatted: {formatted!r}")
if formatted.endswith("<answer>"):
    print("   ✅ Format looks correct")
else:
    print("   ❌ Format is WRONG - missing <answer> tag at end")

# Test generation
print("\n3. Testing generation on 3 examples...")
testset = Dataset("valid")

for i in range(3):
    question, correct_answer = testset[i]
    print(f"\n--- Example {i+1} ---")
    print(f"Question: {question[:70]}...")
    print(f"Expected: {correct_answer}")
    
    # Generate
    raw_output = llm.generate(question)
    print(f"Model output: {raw_output!r}")
    
    # Parse
    parsed = llm.parse_answer(raw_output)
    
    if parsed == parsed:  # Not NaN
        error_pct = abs(parsed - correct_answer) / abs(correct_answer) * 100 if correct_answer != 0 else 0
        status = "✅ CORRECT" if error_pct < 5 else f"❌ WRONG (off by {error_pct:.1f}%)"
        print(f"Parsed: {parsed} - {status}")
    else:
        print(f"Parsed: NaN - ❌ INVALID")
        if "<answer>" not in raw_output:
            print("  → Model didn't generate <answer> tag")
        elif "</answer>" not in raw_output:
            print("  → Model didn't generate </answer> tag")

print("\n" + "=" * 80)
print("SHARE THIS OUTPUT FOR DIAGNOSIS")
print("=" * 80)
