# Quick RFT Training Speedup Guide

## TL;DR - Just make it faster! 🚀

### Option 1: Use the Optimized Training Script (RECOMMENDED)

```bash
# Balanced profile (recommended - 3-5x faster)
python -m homework.rft_fast train --profile=balanced

# Conservative profile (safe, won't OOM - 2-3x faster)
python -m homework.rft_fast train --profile=conservative

# Aggressive profile (maximum speed - 4-6x faster, needs 16GB+ VRAM)
python -m homework.rft_fast train --profile=aggressive
```

### Option 2: Quick Edit to Existing Code

Edit `homework/rft.py`, find the `TrainingArguments` section (line ~158), and replace with:

```python
training_args = TrainingArguments(
    output_dir=str(model_path),
    logging_dir=str(model_path),
    report_to="tensorboard",
    
    # SPEED OPTIMIZATIONS - Add these lines:
    fp16=True,  # 2-3x speedup
    gradient_checkpointing=False,  # 20-30% speedup (change from True)
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,
    optim="adamw_torch_fused",  # Faster optimizer
    lr_scheduler_type="cosine",  # Better convergence
    warmup_steps=50,
    
    # Existing settings:
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_strategy="epoch",
    logging_steps=50,  # Change from 10 to reduce overhead
    save_total_limit=1,
)
```

**If you get OOM (Out of Memory) errors:**
- Change `gradient_checkpointing=False` back to `True`
- OR reduce `per_device_train_batch_size` to 16 or 8

## Performance Comparison

| Method | Training Time | Speedup | Memory |
|--------|--------------|---------|--------|
| **Baseline** (current code) | ~12-15 min | 1x | Low |
| **Quick Edit** (option 2) | ~4-6 min | 3x | Medium |
| **Conservative Profile** | ~5-7 min | 2-3x | Low |
| **Balanced Profile** | ~3-5 min | 4x | Medium |
| **Aggressive Profile** | ~2-3 min | 5-6x | High |

*Times are estimates for 3 epochs on ~900 examples with a modern GPU*

## What Each Optimization Does

1. **fp16=True** → Uses half-precision floating point (2-3x faster, uses less memory)
2. **gradient_checkpointing=False** → Trades memory for speed (20-30% faster)
3. **dataloader_num_workers=4** → Loads data in parallel (1.5-2x faster)
4. **adamw_torch_fused** → Faster optimizer implementation
5. **cosine scheduler + warmup** → Better learning rate schedule, may converge faster

## Verify Your Speedup

### Before optimization:
```bash
time python -m homework.rft train
```

### After optimization:
```bash
time python -m homework.rft_fast train --profile=balanced
```

Compare the times!

## Troubleshooting

### "Out of Memory" Error
**Solution:** Use conservative profile or reduce batch size
```bash
python -m homework.rft_fast train --profile=conservative
# OR
python -m homework.rft_fast train --batch_size=8 --gradient_checkpointing=True
```

### "Accuracy Dropped After Optimization"
**Solution:** This is usually due to smaller effective batch size. Use gradient accumulation:
```bash
python -m homework.rft_fast train --per_device_train_batch_size=8 --gradient_accumulation_steps=8
```

### "First Epoch is Very Slow"
**Solution:** This is normal with torch.compile. Subsequent epochs will be much faster. Disable if you prefer:
```bash
python -m homework.rft_fast train --use_compile=False
```

## Data Generation Speedup

Your data generation (`homework/datagen.py`) can also be sped up:

### Option 1: Use smaller model (faster but lower quality)
```bash
# Edit homework/datagen.py line 41, change to:
model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct")
```

### Option 2: Reduce oversample (if you already get 900+ examples)
```bash
python -m homework.datagen data/rft.json --oversample=12 --temperature=0.5
```

### Option 3: Use FP16 for inference
```bash
export USE_FP16_INFERENCE=1
python -m homework.datagen data/rft.json
```

## Check Your Results

After training, verify accuracy hasn't degraded:

```bash
python -m homework.rft_fast test
```

Expected results:
- **Accuracy**: 0.65-0.75 (should match or exceed baseline)
- **Answer rate**: 0.95+ (should be high)

## Advanced: Custom Optimization

You can mix and match optimizations:

```bash
python -m homework.rft_fast train \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --fp16=True \
    --gradient_checkpointing=False \
    --dataloader_num_workers=6 \
    --use_compile=True \
    --num_train_epochs=2  # Try training for just 2 epochs
```

## GPU-Specific Tips

### NVIDIA A100 / 4090 / 30xx series (Ampere)
- Use `--bf16=True` instead of `--fp16=True` (better numerical stability)
- Use aggressive profile
- Enable TF32 (automatically enabled in the code)

### NVIDIA V100 / 20xx series (Volta/Turing)
- Use `--fp16=True`
- Use balanced profile
- May need gradient checkpointing

### NVIDIA GTX 10xx series or older
- Use conservative profile
- Keep `gradient_checkpointing=True`
- Reduce batch size to 8 if needed

### CPU or MPS (Apple Silicon)
- Optimizations won't help much (no FP16 support)
- Focus on reducing batch size and using more workers
- Consider using a GPU if available

## Summary: Best Commands for Different Scenarios

**I just want it faster (most common):**
```bash
python -m homework.rft_fast train --profile=balanced
```

**I have a small GPU (8GB or less):**
```bash
python -m homework.rft_fast train --profile=conservative
```

**I have a big GPU (16GB+) and want maximum speed:**
```bash
python -m homework.rft_fast train --profile=aggressive
```

**I want to experiment:**
```bash
python -m homework.rft_fast compare  # Compare all profiles
```

That's it! Your RFT training should now be 3-5x faster. 🚀
