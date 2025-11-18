#!/usr/bin/env python3
"""
Quantize adapter model safetensors files to FP16 to reduce file size.
This maintains model integrity while roughly halving the file size.
"""
import argparse
from pathlib import Path
import torch

try:
    import safetensors
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not available, trying alternative method...")


def quantize_adapter(model_path: Path) -> None:
    """Quantize adapter_model.safetensors to FP16."""
    adapter_file = model_path / "adapter_model.safetensors"
    
    if not adapter_file.exists():
        print(f"Warning: {adapter_file} does not exist, skipping...")
        return
    
    print(f"Loading {adapter_file}...")
    
    # Get original size
    original_size = adapter_file.stat().st_size
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    
    # Load the safetensors file
    if HAS_SAFETENSORS:
        tensors = safetensors.torch.load_file(str(adapter_file))
    else:
        # Fallback: try using torch directly (may not work for safetensors format)
        try:
            # Try loading as regular torch file first
            tensors = torch.load(str(adapter_file), map_location='cpu')
            if not isinstance(tensors, dict):
                raise ValueError("File is not a dictionary of tensors")
        except Exception as e:
            print(f"Error: Could not load {adapter_file}. Please install safetensors:")
            print("  pip install safetensors")
            return
    
    # Quantize all tensors to FP16
    quantized_tensors = {}
    for key, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype in (torch.float32, torch.float64):
                quantized_tensors[key] = tensor.half()  # Convert to FP16
            else:
                # Keep non-float tensors as-is (e.g., int tensors)
                quantized_tensors[key] = tensor
        else:
            # Keep non-tensor values as-is
            quantized_tensors[key] = tensor
    
    # Create backup
    backup_file = model_path / "adapter_model.safetensors.backup"
    if not backup_file.exists():
        print(f"Creating backup: {backup_file}")
        import shutil
        shutil.copy2(adapter_file, backup_file)
    else:
        print(f"Backup already exists: {backup_file}")
    
    # Save quantized version
    print(f"Saving quantized version to {adapter_file}...")
    if HAS_SAFETENSORS:
        safetensors.torch.save_file(quantized_tensors, str(adapter_file))
    else:
        # Fallback: save as regular torch file (not ideal, but works)
        torch.save(quantized_tensors, str(adapter_file))
        print("Warning: Saved as regular torch file (not safetensors format)")
    
    # Get new size
    new_size = adapter_file.stat().st_size
    reduction = (1 - new_size / original_size) * 100
    print(f"New size: {new_size / 1024 / 1024:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize adapter model safetensors to FP16"
    )
    parser.add_argument(
        "model_paths",
        nargs="+",
        help="Paths to model directories containing adapter_model.safetensors"
    )
    
    args = parser.parse_args()
    
    for model_path_str in args.model_paths:
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"Warning: {model_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {model_path}...")
        quantize_adapter(model_path)
        print()


if __name__ == "__main__":
    main()
