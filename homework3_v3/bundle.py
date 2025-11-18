import argparse
import zipfile
from pathlib import Path

BLACKLIST = [
    "__pycache__", ".pyc", ".ipynb", "grader", "bundle.py", "submission.zip",
    "README.md", ".md",  # Exclude all markdown files
    "checkpoint-",  # Exclude checkpoint directories
    "events.out.tfevents",  # Exclude TensorBoard event files
    "optimizer.pt", "scheduler.pt", "rng_state.pth",  # Exclude optimizer/scheduler states
    "training_args.bin", "trainer_state.json",  # Exclude training state
    "sft_output",  # Exclude intermediate training output directory
]
MAXSIZE_MB = 50  # Updated to match user's requirement


def bundle(homework_dir: str, utid: str):
    """
    Usage: python3 bundle.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    # Get the files from the homework directory
    files = []

    for f in homework_dir.rglob("*"):
        # Skip directories
        if f.is_dir():
            continue
        # Apply blacklist
        if all(b not in str(f) for b in BLACKLIST):
            files.append(f)

    print("\n".join(str(f.relative_to(homework_dir)) for f in files))

    # Try to use BZIP2 compression if available (better for binary files)
    # Fall back to DEFLATED if not available
    compression = zipfile.ZIP_DEFLATED
    try:
        # Check if BZIP2 is available (requires Python 3.3+ and bzip2 library)
        if hasattr(zipfile, 'ZIP_BZIP2'):
            # Test if bzip2 is actually available
            import bz2
            compression = zipfile.ZIP_BZIP2
    except (ImportError, AttributeError):
        compression = zipfile.ZIP_DEFLATED

    # Zip all files, keeping the directory structure
    # Use compresslevel=9 for maximum compression (Python 3.7+)
    with zipfile.ZipFile(output_path, "w", compression=compression, compresslevel=9) as zf:
        for f in files:
            zf.write(f, homework_dir.stem / f.relative_to(homework_dir))

    output_size_mb = output_path.stat().st_size / 1024 / 1024
    uncompressed_size = sum(f.stat().st_size for f in files) / 1024 / 1024
    
    # Calculate compression ratio
    with zipfile.ZipFile(output_path, "r") as zf:
        compressed_size = sum(f.compress_size for f in zf.filelist) / 1024 / 1024
        compression_ratio = (1 - compressed_size / uncompressed_size) * 100 if uncompressed_size > 0 else 0

    if output_size_mb > MAXSIZE_MB:
        print(f"Warning: The created zip file ({output_size_mb:.2f} MB) is larger than {MAXSIZE_MB} MB!")
        print(f"Uncompressed size: {uncompressed_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print("\nTo reduce size, consider:")
        print("1. Quantizing model weights (e.g., using float16 instead of float32)")
        print("2. Reducing LoRA rank if possible")
        print("3. Ensuring only necessary adapter files are included")

    print(f"Submission created: {output_path.resolve()!s} {output_size_mb:.2f} MB")
    print(f"Uncompressed: {uncompressed_size:.2f} MB, Compression: {compression_ratio:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("homework")
    parser.add_argument("utid")

    args = parser.parse_args()

    bundle(args.homework, args.utid)
