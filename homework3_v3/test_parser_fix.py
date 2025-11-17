"""Quick test to verify the parser fix works"""

from homework.base_llm import BaseLLM

# Create a dummy instance just to use the parser
llm = BaseLLM()

# Test cases from your actual output
test_cases = [
    ("168</answer>", 168.0, "Completion format (your case)"),
    ("6.6742857142857</answer>", 6.6742857142857, "Completion format with decimals"),
    ("2000000</answer>", 2000000.0, "Completion format large number"),
    ("<answer>123.45</answer>", 123.45, "Full format"),
    ("123.45", 123.45, "Plain number"),
]

print("Testing parser fix...")
print("=" * 80)

all_passed = True
for raw_output, expected, description in test_cases:
    parsed = llm.parse_answer(raw_output)
    passed = parsed == expected
    status = "✅" if passed else "❌"
    
    print(f"{status} {description}")
    print(f"   Input: {raw_output!r}")
    print(f"   Expected: {expected}")
    print(f"   Got: {parsed}")
    
    if not passed:
        all_passed = False
    print()

if all_passed:
    print("=" * 80)
    print("✅ ALL TESTS PASSED! The fix works!")
    print("=" * 80)
else:
    print("=" * 80)
    print("❌ Some tests failed")
    print("=" * 80)
