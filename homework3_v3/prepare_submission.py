#!/usr/bin/env python3
"""
Prepare submission by quantizing adapter models and creating the bundle.
This script automates the process of reducing submission size.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare submission by quantizing adapters and bundling"
    )
    parser.add_argument("homework", help="Path to homework directory")
    parser.add_argument("utid", help="UTID for the submission")
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip quantization step (use if already quantized)"
    )
    
    args = parser.parse_args()
    
    homework_path = Path(args.homework)
    sft_model_path = homework_path / "sft_model"
    rft_model_path = homework_path / "rft_model"
    
    # Step 1: Quantize adapter models
    if not args.skip_quantize:
        print("Step 1: Quantizing adapter models to FP16...")
        print("=" * 60)
        
        model_paths = []
        if sft_model_path.exists():
            model_paths.append(str(sft_model_path))
        if rft_model_path.exists():
            model_paths.append(str(rft_model_path))
        
        if model_paths:
            result = subprocess.run(
                [sys.executable, "quantize_adapters.py"] + model_paths,
                cwd=Path(__file__).parent
            )
            if result.returncode != 0:
                print("Error: Quantization failed!")
                sys.exit(1)
        else:
            print("No model directories found, skipping quantization...")
    else:
        print("Skipping quantization step...")
    
    # Step 2: Create bundle
    print("\nStep 2: Creating submission bundle...")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "bundle.py", args.homework, args.utid],
        cwd=Path(__file__).parent
    )
    
    if result.returncode != 0:
        print("Error: Bundle creation failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Submission preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
