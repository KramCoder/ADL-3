import argparse
import zipfile
from pathlib import Path

BLACKLIST = ["__pycache__", ".pyc", ".ipynb", "grader", "bundle.py", "submission.zip", "README.md"]
SKIP_DIR_PREFIXES = ("checkpoint-", "runs")
SKIP_FILE_PREFIXES = ("events.out.tfevents",)
HEAVY_FILE_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "pytorch_model.bin",
    "rng_state.pth",
}
MAXSIZE_MB = 40


def bundle(homework_dir: str, utid: str):
    """
    Usage: python3 bundle.py homework <utid>
    """
    homework_dir = Path(homework_dir).resolve()
    output_path = Path(__file__).parent / f"{utid}.zip"

    # Get the files from the homework directory
    files = []

    for f in homework_dir.rglob("*"):
        if _should_include(homework_dir, f):
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


def _should_include(root: Path, candidate: Path) -> bool:
    rel_path = candidate.relative_to(root)
    rel_str = str(rel_path)

    if any(b in rel_str for b in BLACKLIST):
        return False

    name = candidate.name
    if candidate.is_dir():
        if any(name.startswith(prefix) for prefix in SKIP_DIR_PREFIXES):
            return False
        return True

    if name in HEAVY_FILE_NAMES:
        return False
    if any(name.startswith(prefix) for prefix in SKIP_FILE_PREFIXES):
        return False
    # Keep LoRA adapters but drop other large .pt/.bin artifacts
    if candidate.suffix in {".pt", ".bin"} and "adapter_model" not in name:
        return False

    return True
