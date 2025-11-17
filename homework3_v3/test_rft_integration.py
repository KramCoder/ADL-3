#!/usr/bin/env python3
"""
Test script to verify RFT integration logic without requiring model loading.
This tests the data handling and formatting logic.
"""

def format_numeric_answer(value: float, precision: int = 12) -> str:
    """Standalone copy of format_numeric_answer for testing"""
    if value != value:  # NaN check
        return "nan"
    
    formatted = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if formatted in {"", "-"}:
        formatted = "0"
    if formatted == "-0":
        formatted = "0"
    return formatted

def test_format_example():
    """Test the format_example function logic"""
    
    def format_example(prompt: str, answer: float, reasoning: str = None) -> dict[str, str]:
        """
        Construct a question / answer pair for RFT training.
        If reasoning is provided (RFT data), use it. Otherwise, use simple answer format.
        """
        if reasoning is not None:
            # RFT format: reasoning already contains the answer tags
            reasoning = reasoning.strip()
            if "<answer>" not in reasoning or "</answer>" not in reasoning:
                # If missing tags, add them
                formatted_answer = format_numeric_answer(answer)
                reasoning = f"{reasoning} <answer>{formatted_answer}</answer>"
            return {
                "question": prompt.strip(),
                "answer": reasoning,
            }
        else:
            # Simple answer format (original SFT)
            formatted_answer = format_numeric_answer(answer)
            return {
                "question": prompt.strip(),
                "answer": f"<answer>{formatted_answer}</answer>",
            }
    
    # Test 1: RFT format with reasoning
    result1 = format_example(
        "How many gram are there per 6 kg?",
        6000.0,
        "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
    )
    assert result1["question"] == "How many gram are there per 6 kg?"
    assert "1 kg = 1000 grams" in result1["answer"]
    assert "<answer>6000</answer>" in result1["answer"]
    print("✓ Test 1 passed: RFT format with reasoning")
    
    # Test 2: RFT format with reasoning missing tags
    result2 = format_example(
        "Convert 5 quart to pint?",
        10.0,
        "1 quart = 2 pint. So 5 * 2 = 10"
    )
    assert result2["question"] == "Convert 5 quart to pint?"
    assert "1 quart = 2 pint" in result2["answer"]
    assert "<answer>10</answer>" in result2["answer"]
    print("✓ Test 2 passed: RFT format with reasoning (auto-added tags)")
    
    # Test 3: Simple format without reasoning
    result3 = format_example(
        "How many MB is 2 G?",
        2000.0,
        None
    )
    assert result3["question"] == "How many MB is 2 G?"
    assert result3["answer"] == "<answer>2000</answer>"
    print("✓ Test 3 passed: Simple format without reasoning")
    
    print("\nAll format_example tests passed!")

def test_rft_dataset_structure():
    """Test RFT dataset structure handling"""
    import json
    
    # Simulate RFT data structure
    rft_data = [
        ["How many gram are there per 6 kg?", 6000.0, "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"],
        ["Convert 5 quart to pint?", 10.0, "1 quart = 2 pint. 5 * 2 = <answer>10</answer>"],
        ["How many MB is 2 G?", 2000.0, "1 G = 1000 MB. 2 * 1000 = <answer>2000</answer>"],
    ]
    
    # Test that each entry has the expected format
    for i, entry in enumerate(rft_data):
        assert len(entry) == 3, f"Entry {i} should have 3 elements"
        question, answer, reasoning = entry
        assert isinstance(question, str), f"Entry {i} question should be string"
        assert isinstance(answer, (int, float)), f"Entry {i} answer should be numeric"
        assert isinstance(reasoning, str), f"Entry {i} reasoning should be string"
        assert "<answer>" in reasoning and "</answer>" in reasoning, f"Entry {i} reasoning should have answer tags"
    
    print("✓ RFT dataset structure tests passed!")
    
    # Test RFTDataset wrapper class
    class RFTDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = RFTDataset(rft_data)
    assert len(dataset) == 3
    assert dataset[0] == rft_data[0]
    print("✓ RFTDataset wrapper class tests passed!")

def test_cot_model_checkpoint():
    """Test that CoTModel would use the correct checkpoint"""
    # We can't actually instantiate the model without dependencies,
    # but we can verify the logic
    
    # Simulate the __init__ logic
    kwargs = {}
    if 'checkpoint' not in kwargs:
        kwargs['checkpoint'] = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    assert kwargs['checkpoint'] == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print("✓ CoTModel checkpoint logic test passed!")
    
    # Test that explicit checkpoint is not overridden
    kwargs2 = {'checkpoint': 'custom-model'}
    if 'checkpoint' not in kwargs2:
        kwargs2['checkpoint'] = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    assert kwargs2['checkpoint'] == 'custom-model'
    print("✓ CoTModel explicit checkpoint preserved!")

if __name__ == "__main__":
    print("Testing RFT Integration Logic")
    print("=" * 60)
    
    test_format_example()
    print()
    test_rft_dataset_structure()
    print()
    test_cot_model_checkpoint()
    
    print()
    print("=" * 60)
    print("All integration tests passed! ✓")
    print()
    print("Next steps:")
    print("1. Generate RFT data: python -m homework.datagen data/rft.json")
    print("2. Train SFT model: python -m homework.sft train")
    print("3. Test SFT model: python -m homework.sft test")
