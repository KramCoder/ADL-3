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
    ".backup",  # Exclude backup files
]
MAXSIZE_MB = 50


def bundle(homework_dir: str, utid: str):
    """
    Usage: python3 bundle.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    # Get the files from the homework directory
    files = []

    for f in homework_dir.rglob("*"):
        if all(b not in str(f) for b in BLACKLIST):
            files.append(f)

    print("\n".join(str(f.relative_to(homework_dir)) for f in files))

    # Zip all files, keeping the directory structure
    # Use ZIP_DEFLATED with compresslevel=9 for maximum compression
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in files:
            if f.is_file():  # Only add files, not directories
                zf.write(f, homework_dir.stem / f.relative_to(homework_dir))

    output_size_mb = output_path.stat().st_size / 1024 / 1024

    if output_size_mb > MAXSIZE_MB:
        print("Warning: The created zip file is larger than expected!")

    print(f"Submission created: {output_path.resolve()!s} {output_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("homework")
    parser.add_argument("utid")

    args = parser.parse_args()

    bundle(args.homework, args.utid)
