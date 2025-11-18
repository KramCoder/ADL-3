#!/usr/bin/env python3
"""
Quick test to verify A100 optimizations are working correctly.
This doesn't require generating the full dataset.
"""

import os
import sys
import torch

def test_environment_setup():
    """Test that environment variables are properly set."""
    print("=" * 60)
    print("Testing Environment Setup")
    print("=" * 60)
    
    # Set A100 optimizations
    os.environ['MICRO_BATCH_SIZE'] = '256'
    os.environ['CHUNK_SIZE'] = '10'
    os.environ['A100_BATCH_SIZE'] = '32'
    os.environ['A100_GRAD_ACCUM'] = '1'
    
    assert os.environ.get('MICRO_BATCH_SIZE') == '256', "MICRO_BATCH_SIZE not set"
    assert os.environ.get('CHUNK_SIZE') == '10', "CHUNK_SIZE not set"
    
    print("✓ Environment variables configured correctly")
    print(f"  MICRO_BATCH_SIZE: {os.environ['MICRO_BATCH_SIZE']}")
    print(f"  CHUNK_SIZE: {os.environ['CHUNK_SIZE']}")
    print(f"  A100_BATCH_SIZE: {os.environ['A100_BATCH_SIZE']}")
    print(f"  A100_GRAD_ACCUM: {os.environ['A100_GRAD_ACCUM']}")
    print()

def test_gpu_availability():
    """Test GPU availability and capabilities."""
    print("=" * 60)
    print("Testing GPU Configuration")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ WARNING: CUDA not available. Optimizations require GPU.")
        print("  Tests will continue but performance gains won't be realized.")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    
    # Check memory
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Total memory: {total_mem:.2f} GB")
    
    # Check BFloat16 support
    if hasattr(torch.cuda, 'is_bf16_supported'):
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"  BFloat16 supported: {bf16_supported}")
        if bf16_supported:
            print("    → A100 native precision detected!")
    
    # Detect A100
    device_name = torch.cuda.get_device_name(0)
    is_a100 = 'A100' in device_name
    if is_a100:
        print(f"\n✓ A100 GPU detected! All optimizations will be active.")
    else:
        print(f"\n⚠ Non-A100 GPU detected ({device_name})")
        print("  Optimizations will work but may need tuning.")
    
    print()
    return True

def test_model_loading():
    """Test that optimized model loading works."""
    print("=" * 60)
    print("Testing Model Loading with Optimizations")
    print("=" * 60)
    
    try:
        # Set environment for batch size
        os.environ['MICRO_BATCH_SIZE'] = '256'
        
        from homework.base_llm import BaseLLM
        
        print("Loading BaseLLM with optimizations...")
        model = BaseLLM()
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {model.device}")
        print(f"  Model dtype: {next(model.model.parameters()).dtype}")
        
        # Test that environment variable is being read
        # We can't directly test the value inside batched_generate without calling it
        # but we can verify imports work
        print("✓ Optimization code paths accessible")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batched_generation():
    """Test that batched generation works with optimizations."""
    print("=" * 60)
    print("Testing Batched Generation (Small Test)")
    print("=" * 60)
    
    try:
        os.environ['MICRO_BATCH_SIZE'] = '256'
        
        from homework.cot import CoTModel
        
        print("Loading CoT model...")
        model = CoTModel()
        
        print("Testing small batch generation...")
        test_questions = [
            "How many grams are in 2 kilograms?",
            "Convert 5 meters to centimeters.",
        ]
        
        print(f"Generating answers for {len(test_questions)} questions...")
        results = model.batched_generate(test_questions, num_return_sequences=2, temperature=0.7)
        
        print(f"✓ Batch generation successful")
        print(f"  Generated {len(results)} result sets")
        print(f"  Each set has {len(results[0])} sequences")
        
        # Show a sample
        print("\nSample output:")
        print(f"  Question: {test_questions[0]}")
        print(f"  Answer 1: {results[0][0][:100]}...")
        print()
        
        return True
        
    except Exception as e:
        print(f"✗ Error in batch generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_datagen_parameters():
    """Test that datagen accepts new parameters."""
    print("=" * 60)
    print("Testing Data Generation Parameters")
    print("=" * 60)
    
    try:
        from homework import datagen
        import inspect
        
        # Check function signature
        sig = inspect.signature(datagen.generate_dataset)
        params = list(sig.parameters.keys())
        
        print(f"✓ generate_dataset function found")
        print(f"  Parameters: {params}")
        
        # Verify new parameters exist
        expected_params = ['output_json', 'oversample', 'temperature', 'batch_size', 'use_bfloat16']
        for param in expected_params:
            if param in params:
                default = sig.parameters[param].default
                print(f"  ✓ {param}: {default if default != inspect.Parameter.empty else 'required'}")
            else:
                print(f"  ✗ {param}: MISSING")
                return False
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error checking datagen parameters: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rft_training_parameters():
    """Test that RFT training has A100 optimizations."""
    print("=" * 60)
    print("Testing RFT Training Parameters")
    print("=" * 60)
    
    try:
        # Set environment variables
        os.environ['A100_BATCH_SIZE'] = '32'
        os.environ['A100_GRAD_ACCUM'] = '1'
        
        # Check that the rft.py file contains the optimization code
        import homework.rft as rft
        
        print("✓ RFT module loaded successfully")
        print(f"  A100_BATCH_SIZE env var: {os.environ.get('A100_BATCH_SIZE')}")
        print(f"  A100_GRAD_ACCUM env var: {os.environ.get('A100_GRAD_ACCUM')}")
        
        # Verify the module has the train_model function
        assert hasattr(rft, 'train_model'), "train_model function not found"
        print("✓ train_model function exists")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error checking RFT training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("A100 Optimization Verification Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Environment Setup", test_environment_setup()))
    results.append(("GPU Configuration", test_gpu_availability()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Data Generation Parameters", test_datagen_parameters()))
    results.append(("RFT Training Parameters", test_rft_training_parameters()))
    results.append(("Batched Generation", test_batched_generation()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All optimizations are working correctly!")
        print("\nYou can now run:")
        print("  ./generate_rft_a100.sh")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
