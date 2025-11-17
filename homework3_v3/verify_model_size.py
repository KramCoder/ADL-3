"""
Verification script to check model parameter counts.
This ensures the models meet the grader's parameter constraints.
"""

from transformers import AutoModelForCausalLM

def check_model_size(model_name: str, max_params: int = 380_000_000):
    """Check if a model meets the parameter constraint."""
    print(f"\nChecking: {model_name}")
    print("-" * 60)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Maximum allowed: {max_params:,}")
        print(f"Passes constraint: {total_params < max_params}")
        
        if total_params < max_params:
            print("✓ Model size is acceptable")
        else:
            print("✗ Model size exceeds limit!")
        
        return total_params < max_params
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL SIZE VERIFICATION")
    print("=" * 60)
    
    # Check 360M model (used for training and grading)
    check_model_size("HuggingFaceTB/SmolLM2-360M-Instruct")
    
    # Check 1.7B model (only for data generation)
    print("\n" + "=" * 60)
    print("NOTE: 1.7B model is ONLY used for data generation")
    print("=" * 60)
    check_model_size("HuggingFaceTB/SmolLM2-1.7B-Instruct")
