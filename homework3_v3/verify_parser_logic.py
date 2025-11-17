"""Verify the parser logic without importing torch"""

def parse_answer(answer: str) -> float:
    """The fixed parser function"""
    try:
        # Try full format first (with opening tag)
        if "<answer>" in answer:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        # Handle completion format (no opening tag, just value</answer>)
        elif "</answer>" in answer:
            return float(answer.split("</answer>")[0].strip())
        else:
            # No tags at all, try to parse as plain number
            return float(answer.strip())
    except (IndexError, ValueError):
        return float("nan")

# Test with your actual outputs
test_cases = [
    ("168</answer>", 168.0),
    ("6.6742857142857</answer>", 6.6742857142857),
    ("2000000</answer>", 2000000.0),
    ("<answer>123.45</answer>", 123.45),
    ("123.45", 123.45),
]

print("Testing parser logic...")
print("=" * 70)

all_passed = True
for raw_output, expected in test_cases:
    parsed = parse_answer(raw_output)
    passed = parsed == expected
    status = "✅" if passed else "❌"
    
    print(f"{status} Input: {raw_output!r:40s} → Parsed: {parsed}")
    if not passed:
        print(f"   ❌ Expected {expected}, got {parsed}")
        all_passed = False

print("=" * 70)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("\nYour model should now work correctly!")
    print("Run: python3 -m homework.sft test")
else:
    print("❌ Some tests failed")
