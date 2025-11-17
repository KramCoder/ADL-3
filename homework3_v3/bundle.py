import argparse
import zipfile
from pathlib import Path

BLACKLIST = [
    "__pycache__", 
    ".pyc", 
    ".ipynb", 
    "grader", 
    "bundle.py", 
    "submission.zip", 
    "README.md",
    # Training artifacts that should not be included
    "checkpoint-",  # Checkpoint directories
    ".tfevents.",  # Tensorboard event files
    "optimizer.pt",  # Optimizer states
    "scheduler.pt",  # Scheduler states
    "rng_state.pth",  # RNG states
    "training_args.bin",  # Training arguments
    "trainer_state.json",  # Trainer state
    "events.out.tfevents",  # Tensorboard events (alternative pattern)
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
        # Skip if any blacklist pattern matches the file/directory name or path
        if any(b in str(f) for b in BLACKLIST):
            continue
        # Only include files, not directories
        if f.is_file():
            files.append(f)

    print("\n".join(str(f.relative_to(homework_dir)) for f in files))

    # Zip all files, keeping the directory structure
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
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
