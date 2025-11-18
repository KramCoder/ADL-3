#!/usr/bin/env python3
"""
Minimal bundle script that only includes the exact files needed for grading.
This ensures no unnecessary files are included.
"""
import argparse
import zipfile
from pathlib import Path

# Exact list of files/directories that must be included
REQUIRED_FILES = [
    "rft.py",
    "datagen.py",
    "cot.py",
    "sft_model",
    "data.py",
    "rft_model",
    "__init__.py",
    "base_llm.py",
    "conversion_utils.py",
    "sft.py",
]

# Files within model directories that should be included
MODEL_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
]

MAXSIZE_MB = 50


def bundle_minimal(homework_dir: str, utid: str):
    """
    Create a minimal bundle with only the required files.
    Usage: python3 bundle_minimal.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    files_to_include = []
    
    # Collect all required files
    for item in REQUIRED_FILES:
        item_path = homework_dir / item
        
        if not item_path.exists():
            print(f"Warning: {item} not found, skipping...")
            continue
        
        if item_path.is_file():
            files_to_include.append(item_path)
        elif item_path.is_dir():
            # For directories, include all files matching MODEL_FILES patterns
            if item in ["sft_model", "rft_model"]:
                # Include all files in model directories
                for f in item_path.rglob("*"):
                    if f.is_file():
                        # Only include safetensors and config files, exclude README
                        if f.suffix in [".safetensors", ".json"] or f.name in MODEL_FILES:
                            if "README" not in f.name:
                                files_to_include.append(f)
            else:
                # For other directories, include all Python files
                for f in item_path.rglob("*.py"):
                    files_to_include.append(f)
    
    # Print what will be included
    print("Files to be included:")
    for f in sorted(files_to_include):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.relative_to(homework_dir)} ({size_mb:.2f} MB)")
    
    total_uncompressed = sum(f.stat().st_size for f in files_to_include) / 1024 / 1024
    print(f"\nTotal uncompressed size: {total_uncompressed:.2f} MB")
    
    # Try to use BZIP2 compression if available (better for binary files)
    compression = zipfile.ZIP_DEFLATED
    try:
        if hasattr(zipfile, 'ZIP_BZIP2'):
            compression = zipfile.ZIP_BZIP2
    except:
        pass
    
    # Create zip with maximum compression
    with zipfile.ZipFile(output_path, "w", compression=compression, compresslevel=9) as zf:
        for f in files_to_include:
            # Preserve directory structure relative to homework_dir
            arcname = homework_dir.stem / f.relative_to(homework_dir)
            zf.write(f, arcname)
    
    output_size_mb = output_path.stat().st_size / 1024 / 1024
    
    # Calculate compression ratio
    with zipfile.ZipFile(output_path, "r") as zf:
        compressed_size = sum(f.compress_size for f in zf.filelist) / 1024 / 1024
        compression_ratio = (1 - compressed_size / total_uncompressed) * 100 if total_uncompressed > 0 else 0
    
    if output_size_mb > MAXSIZE_MB:
        print(f"\n⚠️  WARNING: Zip file ({output_size_mb:.2f} MB) exceeds {MAXSIZE_MB} MB limit!")
        print(f"Uncompressed: {total_uncompressed:.2f} MB")
        print(f"Compression: {compression_ratio:.1f}%")
        print("\nTo reduce size:")
        print("1. Quantize model weights: python3 optimize_models.py homework/sft_model --quantize")
        print("2. Quantize RFT model: python3 optimize_models.py homework/rft_model --quantize")
        print("3. Check if models can use lower LoRA rank")
    else:
        print(f"\n✅ Submission created: {output_path.resolve()!s}")
        print(f"   Size: {output_size_mb:.2f} MB (under {MAXSIZE_MB} MB limit)")
        print(f"   Compression: {compression_ratio:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create minimal submission bundle")
    parser.add_argument("homework", help="Path to homework directory")
    parser.add_argument("utid", help="Your UTID")
    
    args = parser.parse_args()
    bundle_minimal(args.homework, args.utid)
