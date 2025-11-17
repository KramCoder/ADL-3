#!/usr/bin/env python3
"""
Test script to verify the dtype fix resolves the zero accuracy issue.
This script checks that the model loads correctly and can generate valid answers.
"""

import sys
from pathlib import Path

# Add homework directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_loading():
    """Test that model loads with correct dtype configuration"""
    print("=" * 60)
    print("Testing Model Loading and Inference")
    print("=" * 60)
    
    try:
        from homework.sft import load
        from homework.data import Dataset
        
        print("\n1. Loading SFT model...")
        llm = load()
        print(f"   ✓ Model loaded successfully")
        print(f"   - Device: {llm.device}")
        print(f"   - Model dtype: {llm.model.dtype}")
        
        print("\n2. Testing single question...")
        test_question = "What is 2 + 2?"
        result = llm.generate(test_question)
        print(f"   - Question: {test_question}")
        print(f"   - Generated: {result}")
        
        print("\n3. Testing answer parsing...")
        parsed = llm.parse_answer(f"<answer>{result}</answer>")
        print(f"   - Parsed answer: {parsed}")
        print(f"   - Is valid (not NaN): {parsed == parsed}")
        
        print("\n4. Testing on validation dataset (first 5 samples)...")
        testset = Dataset("valid")
        questions = [testset[i][0] for i in range(min(5, len(testset)))]
        answers = llm.answer(*questions)
        
        valid_count = sum(1 for a in answers if a == a)  # Count non-NaN
        print(f"   - Total questions: {len(questions)}")
        print(f"   - Valid answers (not NaN): {valid_count}/{len(questions)}")
        print(f"   - Answer rate: {valid_count/len(questions)*100:.1f}%")
        
        if valid_count > 0:
            print("\n✓ SUCCESS: Model generates valid answers!")
            print("  The dtype fix appears to be working correctly.")
        else:
            print("\n✗ FAILURE: All answers are still NaN")
            print("  Additional debugging may be needed.")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
