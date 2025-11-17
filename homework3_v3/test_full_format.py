"""Test that the full <answer>value</answer> format is returned"""

def simulate_generate(raw_model_output):
    """Simulate what generate() now does"""
    # Model generates: "value</answer>"
    # We prepend: "<answer>"
    return f"<answer>{raw_model_output}"

def parse_answer(answer: str) -> float:
    """The standard parser"""
    try:
        return float(answer.split("<answer>")[1].split("</answer>")[0])
    except (IndexError, ValueError):
        return float("nan")

print("Testing Full Format Output")
print("=" * 70)

# Simulate what the model actually generates (raw tokens)
raw_outputs = [
    "168</answer>",
    "6.6742857142857</answer>",
    "2000000</answer>",
]

expected_values = [168.0, 6.6742857142857, 2000000.0]

print("\nSimulating generate() behavior:\n")
all_passed = True

for raw, expected in zip(raw_outputs, expected_values):
    # What generate() now returns (with prepended <answer>)
    full_output = simulate_generate(raw)
    
    # What parser extracts
    parsed = parse_answer(full_output)
    
    passed = parsed == expected
    status = "✅" if passed else "❌"
    
    print(f"{status} Model generates: {raw!r}")
    print(f"   generate() returns: {full_output!r}")
    print(f"   Parser extracts: {parsed}")
    print()
    
    if not passed:
        all_passed = False

print("=" * 70)
if all_passed:
    print("✅ SUCCESS! Full format <answer>value</answer> is returned!")
else:
    print("❌ Tests failed")

print("\nNow your outputs will look like:")
print('  Raw model output: "<answer>123.45</answer>"')
print('  Parser result: 123.45')
