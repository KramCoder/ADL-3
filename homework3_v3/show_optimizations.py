#!/usr/bin/env python3
"""
Show RFT training optimizations in a clear comparison format.

This helps you understand what each optimization does and how they combine
for maximum speedup.
"""

def print_comparison():
    """Print a side-by-side comparison of baseline vs optimized."""
    
    print("\n" + "="*80)
    print("RFT TRAINING CONFIGURATION COMPARISON")
    print("="*80)
    
    comparisons = [
        {
            "setting": "Mixed Precision",
            "baseline": "FP32 (full precision)",
            "optimized": "FP16 (half precision)",
            "impact": "2-3x speedup, -40% memory",
            "note": "Main speedup source",
        },
        {
            "setting": "Gradient Checkpointing",
            "baseline": "Enabled",
            "optimized": "Disabled",
            "impact": "+20-30% speed, +30% memory",
            "note": "Trade memory for speed",
        },
        {
            "setting": "Data Loading",
            "baseline": "Single worker (sequential)",
            "optimized": "4 workers + prefetch",
            "impact": "1.5-2x speedup",
            "note": "Parallel data loading",
        },
        {
            "setting": "Batch Size",
            "baseline": "32 per device",
            "optimized": "16 per device",
            "impact": "Better memory usage",
            "note": "Allows more optimizations",
        },
        {
            "setting": "Gradient Accumulation",
            "baseline": "None (effective batch = 32)",
            "optimized": "4 steps (effective batch = 64)",
            "impact": "Larger effective batch",
            "note": "Better gradients, may converge faster",
        },
        {
            "setting": "Optimizer",
            "baseline": "AdamW (standard)",
            "optimized": "AdamW Fused",
            "impact": "+10-15% speed",
            "note": "Optimized CUDA kernels",
        },
        {
            "setting": "LR Schedule",
            "baseline": "Constant",
            "optimized": "Cosine + warmup",
            "impact": "Better convergence",
            "note": "May need fewer epochs",
        },
        {
            "setting": "Logging",
            "baseline": "Every 10 steps",
            "optimized": "Every 50 steps",
            "impact": "+2-5% speed",
            "note": "Less overhead",
        },
        {
            "setting": "Compilation",
            "baseline": "Not used",
            "optimized": "torch.compile (optional)",
            "impact": "+20-50% speed",
            "note": "PyTorch 2.0+ only",
        },
        {
            "setting": "TF32 (Ampere GPUs)",
            "baseline": "Not used",
            "optimized": "Auto-enabled",
            "impact": "+20-50% speed",
            "note": "A100/4090/30xx series only",
        },
    ]
    
    # Print header
    print(f"\n{'Setting':<25} {'Baseline':<30} {'Optimized':<30}")
    print("-" * 85)
    
    # Print comparisons
    for comp in comparisons:
        print(f"{comp['setting']:<25} {comp['baseline']:<30} {comp['optimized']:<30}")
        print(f"{'':25} Impact: {comp['impact']}")
        print(f"{'':25} Note: {comp['note']}")
        print()
    
    print("="*80)
    print("COMBINED SPEEDUP ESTIMATE: 3-5x faster")
    print("="*80)


def print_profiles():
    """Print the three training profiles."""
    
    print("\n" + "="*80)
    print("PRE-CONFIGURED TRAINING PROFILES")
    print("="*80)
    
    profiles = {
        "Conservative": {
            "speed": "2-3x faster",
            "memory": "Low (8GB GPU ok)",
            "batch_size": "16",
            "grad_accum": "4 (effective: 64)",
            "grad_checkpoint": "Enabled",
            "fp16": "Yes",
            "workers": "2",
            "compile": "No",
            "use_case": "Safe choice, won't OOM",
        },
        "Balanced": {
            "speed": "3-5x faster",
            "memory": "Medium (12GB+ GPU)",
            "batch_size": "16",
            "grad_accum": "4 (effective: 64)",
            "grad_checkpoint": "Disabled",
            "fp16": "Yes",
            "workers": "4",
            "compile": "Yes",
            "use_case": "Recommended for most users",
        },
        "Aggressive": {
            "speed": "5-6x faster",
            "memory": "High (16GB+ GPU)",
            "batch_size": "32",
            "grad_accum": "2 (effective: 64)",
            "grad_checkpoint": "Disabled",
            "fp16": "Yes",
            "workers": "6",
            "compile": "Yes",
            "use_case": "Maximum speed, large GPU needed",
        },
    }
    
    for name, config in profiles.items():
        print(f"\n📋 {name.upper()} Profile")
        print("-" * 80)
        print(f"  Speedup: {config['speed']}")
        print(f"  Memory: {config['memory']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Gradient accumulation: {config['grad_accum']}")
        print(f"  Gradient checkpointing: {config['grad_checkpoint']}")
        print(f"  FP16: {config['fp16']}")
        print(f"  Data workers: {config['workers']}")
        print(f"  Torch compile: {config['compile']}")
        print(f"  Use case: {config['use_case']}")
        print(f"\n  Command: python -m homework.rft_fast train --profile={name.lower()}")
    
    print("\n" + "="*80)


def print_gpu_recommendations():
    """Print GPU-specific recommendations."""
    
    print("\n" + "="*80)
    print("GPU-SPECIFIC RECOMMENDATIONS")
    print("="*80)
    
    gpus = [
        {
            "name": "NVIDIA A100 / H100",
            "vram": "40-80GB",
            "profile": "Aggressive",
            "extra": "--bf16=True --fp16=False",
            "notes": "Use BF16 for best performance. Can use very large batches.",
        },
        {
            "name": "NVIDIA RTX 4090 / 4080",
            "vram": "16-24GB",
            "profile": "Aggressive",
            "extra": "--bf16=True --fp16=False",
            "notes": "Ampere architecture, use BF16. TF32 auto-enabled.",
        },
        {
            "name": "NVIDIA RTX 3090 / 3080",
            "vram": "10-24GB",
            "profile": "Balanced",
            "extra": "",
            "notes": "FP16 works great. Good balance of speed and memory.",
        },
        {
            "name": "NVIDIA V100",
            "vram": "16-32GB",
            "profile": "Balanced",
            "extra": "--gradient_checkpointing=True",
            "notes": "Volta architecture. May need gradient checkpointing.",
        },
        {
            "name": "NVIDIA RTX 2080 Ti / 2070",
            "vram": "8-11GB",
            "profile": "Conservative",
            "extra": "",
            "notes": "Limited VRAM. Use conservative profile to avoid OOM.",
        },
        {
            "name": "NVIDIA GTX 1080 Ti / 1070",
            "vram": "8-11GB",
            "profile": "Conservative",
            "extra": "--gradient_checkpointing=True",
            "notes": "Older architecture. Limited FP16 support.",
        },
    ]
    
    for gpu in gpus:
        print(f"\n💻 {gpu['name']} ({gpu['vram']} VRAM)")
        print("-" * 80)
        print(f"  Recommended profile: {gpu['profile']}")
        if gpu['extra']:
            print(f"  Additional args: {gpu['extra']}")
        print(f"  Notes: {gpu['notes']}")
        print(f"\n  Command:")
        cmd = f"  python -m homework.rft_fast train --profile={gpu['profile'].lower()}"
        if gpu['extra']:
            cmd += f" {gpu['extra']}"
        print(f"  {cmd}")
    
    print("\n" + "="*80)


def print_timing_estimate():
    """Print expected training time estimates."""
    
    print("\n" + "="*80)
    print("TRAINING TIME ESTIMATES")
    print("="*80)
    print("\nFor 3 epochs on ~900 examples:")
    print("-" * 80)
    
    estimates = [
        ("Baseline (current)", "12-15 min", "1x", "Low risk"),
        ("Conservative", "5-7 min", "2-3x faster", "No risk"),
        ("Balanced", "3-5 min", "3-5x faster", "Low risk"),
        ("Aggressive", "2-3 min", "5-6x faster", "OOM risk on small GPUs"),
    ]
    
    print(f"\n{'Configuration':<25} {'Time':<15} {'Speedup':<15} {'Risk':<20}")
    print("-" * 80)
    
    for config, time, speedup, risk in estimates:
        print(f"{config:<25} {time:<15} {speedup:<15} {risk:<20}")
    
    print("\n" + "="*80)
    print("\nNote: Times are estimates for modern GPUs (RTX 30xx/40xx, A100)")
    print("Actual times depend on GPU, CPU, storage speed, and system load.")
    print("\nTo measure actual speedup on your system:")
    print("  python benchmark_rft_speed.py")
    print("="*80)


def print_quick_commands():
    """Print quick reference commands."""
    
    print("\n" + "="*80)
    print("QUICK REFERENCE COMMANDS")
    print("="*80)
    
    commands = [
        {
            "task": "Train with balanced profile (recommended)",
            "command": "python -m homework.rft_fast train --profile=balanced",
        },
        {
            "task": "Train with conservative profile (safe)",
            "command": "python -m homework.rft_fast train --profile=conservative",
        },
        {
            "task": "Train with aggressive profile (fast)",
            "command": "python -m homework.rft_fast train --profile=aggressive",
        },
        {
            "task": "Test trained model",
            "command": "python -m homework.rft_fast test",
        },
        {
            "task": "Benchmark speedup",
            "command": "python benchmark_rft_speed.py",
        },
        {
            "task": "Custom batch size",
            "command": "python -m homework.rft_fast train --per_device_train_batch_size=8",
        },
        {
            "task": "Train for 2 epochs (faster)",
            "command": "python -m homework.rft_fast train --num_train_epochs=2",
        },
        {
            "task": "Use BF16 (A100/4090)",
            "command": "python -m homework.rft_fast train --bf16=True --fp16=False",
        },
        {
            "task": "Monitor GPU usage",
            "command": "watch -n 1 nvidia-smi",
        },
        {
            "task": "View tensorboard logs",
            "command": "tensorboard --logdir=rft_model",
        },
    ]
    
    for item in commands:
        print(f"\n📌 {item['task']}")
        print(f"   {item['command']}")
    
    print("\n" + "="*80)


def main():
    """Print all information."""
    import sys
    
    if len(sys.argv) > 1:
        section = sys.argv[1].lower()
        
        if section == "comparison":
            print_comparison()
        elif section == "profiles":
            print_profiles()
        elif section == "gpu":
            print_gpu_recommendations()
        elif section == "timing":
            print_timing_estimate()
        elif section == "commands":
            print_quick_commands()
        else:
            print(f"Unknown section: {section}")
            print("\nAvailable sections:")
            print("  comparison  - Show baseline vs optimized comparison")
            print("  profiles    - Show training profiles")
            print("  gpu         - Show GPU recommendations")
            print("  timing      - Show timing estimates")
            print("  commands    - Show quick reference commands")
            print("\nOr run without arguments to show everything.")
    else:
        # Show everything
        print_comparison()
        print_profiles()
        print_gpu_recommendations()
        print_timing_estimate()
        print_quick_commands()
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Choose your profile based on GPU:")
        print("   - Small GPU (8GB): Conservative")
        print("   - Medium GPU (12GB+): Balanced (recommended)")
        print("   - Large GPU (16GB+): Aggressive")
        print("\n2. Run training:")
        print("   python -m homework.rft_fast train --profile=balanced")
        print("\n3. Benchmark to measure actual speedup:")
        print("   python benchmark_rft_speed.py")
        print("\n4. Verify accuracy:")
        print("   python -m homework.rft_fast test")
        print("\n" + "="*80)
        print("\n💡 TIP: Start with 'balanced' profile - it works for most users!")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
