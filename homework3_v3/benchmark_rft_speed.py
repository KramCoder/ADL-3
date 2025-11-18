#!/usr/bin/env python3
"""
Benchmark RFT training speed with different configurations.

This script helps you:
1. Measure baseline training time
2. Test different optimization profiles
3. Find the best configuration for your GPU
4. Verify accuracy is maintained

Usage:
    python benchmark_rft_speed.py                    # Quick benchmark
    python benchmark_rft_speed.py --full             # Full benchmark (all profiles)
    python benchmark_rft_speed.py --profile balanced # Test specific profile
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return None
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    compute_cap = torch.cuda.get_device_capability(0)
    
    return {
        "name": gpu_name,
        "memory_gb": gpu_memory,
        "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
    }


def check_dataset_exists():
    """Check if RFT dataset exists."""
    rft_data = Path(__file__).parent / "data" / "rft.json"
    if not rft_data.exists():
        print("❌ RFT dataset not found!")
        print("\nPlease generate it first:")
        print("    python -m homework.datagen data/rft.json")
        return False
    
    with rft_data.open() as f:
        data = json.load(f)
    
    print(f"✓ RFT dataset found: {len(data)} examples")
    
    if len(data) < 850:
        print(f"⚠ WARNING: Only {len(data)} examples. Recommended: 850-900+")
        print("    Consider regenerating with more data")
    
    return True


def run_training(profile: str = None, epochs: int = 1, custom_args: dict = None):
    """Run RFT training and measure time.
    
    Args:
        profile: Training profile name or None for baseline
        epochs: Number of epochs to train
        custom_args: Custom training arguments
    
    Returns:
        dict with timing and accuracy results
    """
    if profile is None:
        # Baseline training
        cmd = ["python", "-m", "homework.rft", "train"]
        label = "Baseline"
    else:
        # Optimized training
        cmd = ["python", "-m", "homework.rft_fast", "train", f"--profile={profile}"]
        label = profile.capitalize()
    
    # Add custom args if provided
    if custom_args:
        for key, value in custom_args.items():
            cmd.append(f"--{key}={value}")
    
    # Always set epochs
    if "--num_train_epochs" not in " ".join(cmd):
        cmd.append(f"--num_train_epochs={epochs}")
    
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Check if successful
        if result.returncode != 0:
            print(f"❌ Training failed!")
            print(f"Error: {result.stderr[-500:]}")  # Last 500 chars
            return None
        
        # Extract accuracy from output
        accuracy = None
        answer_rate = None
        for line in result.stdout.split('\n'):
            if "accuracy=" in line.lower():
                try:
                    # Parse "accuracy=0.7500  answer_rate=0.9800"
                    parts = line.split()
                    for part in parts:
                        if "accuracy=" in part:
                            accuracy = float(part.split("=")[1])
                        if "answer_rate=" in part:
                            answer_rate = float(part.split("=")[1])
                except:
                    pass
        
        return {
            "label": label,
            "time": elapsed_time,
            "time_per_epoch": elapsed_time / epochs,
            "accuracy": accuracy,
            "answer_rate": answer_rate,
            "success": True,
        }
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"❌ Training timed out after {elapsed_time:.0f}s")
        return {
            "label": label,
            "time": elapsed_time,
            "success": False,
            "error": "timeout",
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Error: {e}")
        return {
            "label": label,
            "time": elapsed_time,
            "success": False,
            "error": str(e),
        }


def print_results(results: list[dict], baseline: dict = None):
    """Print benchmark results in a nice table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Profile':<20} {'Time':>12} {'Per Epoch':>12} {'Accuracy':>10} {'Speedup':>10}")
    print("-" * 80)
    
    # Results
    for result in results:
        if not result or not result.get("success"):
            status = "FAILED" if result else "SKIPPED"
            print(f"{result.get('label', 'Unknown'):<20} {status:>12}")
            continue
        
        time_str = f"{result['time']:.1f}s"
        per_epoch_str = f"{result['time_per_epoch']:.1f}s"
        
        # Accuracy
        if result.get("accuracy") is not None:
            acc_str = f"{result['accuracy']:.3f}"
        else:
            acc_str = "N/A"
        
        # Speedup
        if baseline and baseline.get("success"):
            speedup = baseline["time"] / result["time"]
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{result['label']:<20} {time_str:>12} {per_epoch_str:>12} {acc_str:>10} {speedup_str:>10}")
    
    print("-" * 80)
    
    # Summary
    if baseline and baseline.get("success"):
        best = min([r for r in results if r and r.get("success")], key=lambda x: x["time"])
        speedup = baseline["time"] / best["time"]
        time_saved = baseline["time"] - best["time"]
        
        print(f"\n📊 Summary:")
        print(f"   Baseline time: {baseline['time']:.1f}s ({baseline['time']/60:.1f} min)")
        print(f"   Best time: {best['time']:.1f}s ({best['time']/60:.1f} min) - {best['label']}")
        print(f"   Speedup: {speedup:.2f}x faster")
        print(f"   Time saved: {time_saved:.1f}s ({time_saved/60:.1f} min)")
        
        # Accuracy check
        if best.get("accuracy") and baseline.get("accuracy"):
            acc_diff = best["accuracy"] - baseline["accuracy"]
            if acc_diff >= 0:
                print(f"   Accuracy: ✓ Maintained ({acc_diff:+.3f})")
            else:
                print(f"   Accuracy: ⚠ Decreased ({acc_diff:+.3f})")
    
    print("="*80 + "\n")


def quick_benchmark():
    """Quick benchmark: test baseline vs balanced profile."""
    print("\n🚀 Quick RFT Training Benchmark")
    print("="*80)
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\n💻 GPU: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['memory_gb']:.1f} GB")
        print(f"   Compute: {gpu_info['compute_capability']}")
    else:
        print("\n💻 Device: CPU (GPU not available)")
    
    # Check dataset
    if not check_dataset_exists():
        return
    
    results = []
    
    # Run baseline (1 epoch only for speed)
    print("\n📌 Step 1/2: Testing baseline configuration...")
    baseline = run_training(profile=None, epochs=1)
    if baseline and baseline.get("success"):
        results.append(baseline)
        print(f"✓ Baseline: {baseline['time']:.1f}s")
    else:
        print("⚠ Baseline training failed, skipping comparison")
        baseline = None
    
    # Run balanced profile
    print("\n📌 Step 2/2: Testing optimized configuration...")
    balanced = run_training(profile="balanced", epochs=1)
    if balanced and balanced.get("success"):
        results.append(balanced)
        print(f"✓ Optimized: {balanced['time']:.1f}s")
    
    # Print results
    print_results(results, baseline)
    
    # Recommendation
    if balanced and balanced.get("success") and baseline and baseline.get("success"):
        speedup = baseline["time"] / balanced["time"]
        if speedup > 1.5:
            print("✅ RECOMMENDATION: Use optimized training!")
            print(f"   Command: python -m homework.rft_fast train --profile=balanced")
            print(f"   Expected speedup for 3 epochs: ~{speedup:.1f}x faster")
        else:
            print("ℹ️  Note: Speedup is modest. Baseline may be sufficient.")


def full_benchmark():
    """Full benchmark: test all profiles."""
    print("\n🚀 Full RFT Training Benchmark")
    print("="*80)
    print("⚠️  This will take 15-30 minutes to complete")
    print("="*80)
    
    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\n💻 GPU: {gpu_info['name']}")
        print(f"   Memory: {gpu_info['memory_gb']:.1f} GB")
        print(f"   Compute: {gpu_info['compute_capability']}")
    else:
        print("\n💻 Device: CPU")
    
    # Check dataset
    if not check_dataset_exists():
        return
    
    results = []
    profiles = ["conservative", "balanced", "aggressive"]
    
    # Test baseline
    print("\n📌 Testing baseline configuration...")
    baseline = run_training(profile=None, epochs=1)
    if baseline:
        results.append(baseline)
    
    # Test each profile
    for i, profile in enumerate(profiles, 1):
        print(f"\n📌 Testing {profile} profile ({i}/{len(profiles)})...")
        result = run_training(profile=profile, epochs=1)
        if result:
            results.append(result)
    
    # Print results
    print_results(results, baseline)
    
    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with output_file.open("w") as f:
        json.dump({
            "gpu": gpu_info,
            "results": results,
            "timestamp": time.time(),
        }, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RFT training speed")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (all profiles)")
    parser.add_argument("--profile", type=str, help="Test specific profile")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    
    args = parser.parse_args()
    
    if args.profile:
        # Test specific profile
        print(f"\n🚀 Testing {args.profile} profile")
        result = run_training(profile=args.profile, epochs=args.epochs)
        if result:
            print_results([result])
    elif args.full:
        # Full benchmark
        full_benchmark()
    else:
        # Quick benchmark
        quick_benchmark()


if __name__ == "__main__":
    main()
