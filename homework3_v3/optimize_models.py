#!/usr/bin/env python3
"""
Helper script to optimize model files for submission.
This script can quantize safetensors files to reduce their size.
"""
import argparse
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")


def quantize_model(input_path: Path, output_path: Path = None, dtype: str = "float16"):
    """
    Quantize a safetensors model file to reduce its size.
    
    Args:
        input_path: Path to input safetensors file
        output_path: Path to output file (defaults to overwriting input)
        dtype: Target dtype ("float16" or "bfloat16")
    """
    if not SAFETENSORS_AVAILABLE:
        print("Error: safetensors library not available")
        return False
    
    if output_path is None:
        output_path = input_path
    
    if dtype == "float16":
        target_dtype = torch.float16
    elif dtype == "bfloat16":
        target_dtype = torch.bfloat16
    else:
        print(f"Error: Unsupported dtype {dtype}")
        return False
    
    print(f"Loading model from {input_path}...")
    tensors = {}
    
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Only quantize float32 tensors
            if tensor.dtype == torch.float32:
                tensors[key] = tensor.to(target_dtype)
                print(f"  Quantized {key}: {tensor.shape} {tensor.dtype} -> {target_dtype}")
            else:
                tensors[key] = tensor
                print(f"  Kept {key}: {tensor.shape} {tensor.dtype}")
    
    print(f"Saving quantized model to {output_path}...")
    save_file(tensors, output_path)
    
    # Report size reduction
    original_size = input_path.stat().st_size / 1024 / 1024
    new_size = output_path.stat().st_size / 1024 / 1024
    reduction = (1 - new_size / original_size) * 100
    
    print(f"Size: {original_size:.2f} MB -> {new_size:.2f} MB ({reduction:.1f}% reduction)")
    return True


def check_model_size(model_path: Path):
    """Check the size of a model file and its contents."""
    if not model_path.exists():
        print(f"Error: {model_path} does not exist")
        return
    
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"{model_path}: {size_mb:.2f} MB")
    
    if SAFETENSORS_AVAILABLE and model_path.suffix == ".safetensors":
        try:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                total_params = 0
                float32_params = 0
                float16_params = 0
                
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    params = tensor.numel()
                    total_params += params
                    
                    if tensor.dtype == torch.float32:
                        float32_params += params
                    elif tensor.dtype == torch.float16:
                        float16_params += params
                
                print(f"  Total parameters: {total_params:,}")
                print(f"  Float32 parameters: {float32_params:,} ({float32_params/total_params*100:.1f}%)")
                print(f"  Float16 parameters: {float16_params:,} ({float16_params/total_params*100:.1f}%)")
                
                if float32_params > 0:
                    potential_savings = float32_params * 4 / 1024 / 1024  # 4 bytes per float32
                    print(f"  Potential size reduction (if quantized to float16): ~{potential_savings:.2f} MB")
        except Exception as e:
            print(f"  Could not analyze file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Optimize model files for submission")
    parser.add_argument("model_path", type=Path, help="Path to model file or directory")
    parser.add_argument("--quantize", action="store_true", help="Quantize model to float16")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                       help="Target dtype for quantization")
    parser.add_argument("--check", action="store_true", help="Check model size without modifying")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    
    if model_path.is_dir():
        # Find safetensors files in directory
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            print(f"No safetensors files found in {model_path}")
            return
        
        for sf in safetensors_files:
            if args.check:
                check_model_size(sf)
            elif args.quantize:
                quantize_model(sf, dtype=args.dtype)
    else:
        if args.check:
            check_model_size(model_path)
        elif args.quantize:
            quantize_model(model_path, dtype=args.dtype)
        else:
            print("Specify --check or --quantize")
            sys.exit(1)


if __name__ == "__main__":
    main()
