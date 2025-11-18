# RFT Training Performance Optimization Guide

## Current Performance Baseline

Your current RFT training configuration (from `homework/rft.py`):
- **Batch size**: 32
- **Epochs**: 3
- **Gradient checkpointing**: Enabled (saves memory, but slows training ~20-30%)
- **Mixed precision**: Not enabled
- **DataLoader workers**: Not configured
- **Gradient accumulation**: Not used
- **Model compilation**: Not enabled

## Key Optimizations (Ordered by Impact)

### 1. **Enable Mixed Precision Training (2-3x speedup)** ⭐ HIGHEST IMPACT

Mixed precision uses FP16 for most operations while keeping FP32 for critical parts.

**Add to TrainingArguments:**
```python
fp16=True,  # or bf16=True if your GPU supports it (A100, 4090, etc.)
fp16_opt_level="O2",  # More aggressive mixed precision
```

**Expected speedup**: 2-3x faster training
**Memory impact**: Reduces memory usage by ~40%

### 2. **Disable Gradient Checkpointing (20-30% speedup)** ⭐ HIGH IMPACT

Gradient checkpointing trades compute for memory. Since you have a relatively small batch size (32) and small model (360M params), you likely don't need it.

**Change:**
```python
gradient_checkpointing=False,  # Change from True
```

**Expected speedup**: 20-30% faster
**Memory impact**: Increases memory usage by ~30-40%
**Note**: Only do this if you have enough VRAM. If you get OOM errors, keep it enabled.

### 3. **Optimize DataLoader (1.5-2x speedup)** ⭐ HIGH IMPACT

Parallel data loading prevents data loading from being a bottleneck.

**Add to TrainingArguments:**
```python
dataloader_num_workers=4,  # Use 2-8 workers
dataloader_pin_memory=True,  # Faster GPU transfer
dataloader_prefetch_factor=2,  # Prefetch batches
```

**Expected speedup**: 1.5-2x faster (especially with slower storage)

### 4. **Enable Gradient Accumulation (Quality + Speed)** ⭐ MEDIUM IMPACT

Simulate larger batch sizes without using more memory.

**Add to TrainingArguments:**
```python
per_device_train_batch_size=16,  # Reduce from 32
gradient_accumulation_steps=4,  # Effective batch size = 16*4 = 64
```

**Benefits**:
- Larger effective batch size can improve convergence
- May need fewer epochs to reach same performance
- Slightly slower per-epoch but better quality

### 5. **Torch Compile (1.2-1.5x speedup)** ⭐ MEDIUM IMPACT

PyTorch 2.0+ compilation optimizes model execution.

**Add after creating the model:**
```python
if hasattr(torch, 'compile'):
    lora_model = torch.compile(lora_model, mode='reduce-overhead')
```

**Expected speedup**: 1.2-1.5x faster (after warmup)
**Note**: First epoch will be slower due to compilation overhead

### 6. **Optimize Save Strategy**

**Change:**
```python
save_strategy="epoch",  # Keep this
save_total_limit=1,  # Keep only last checkpoint
save_only_model=True,  # Don't save optimizer states (faster saves)
```

**Benefits**: Faster checkpoint saves, less disk I/O

### 7. **Reduce Logging Overhead**

**Change:**
```python
logging_steps=50,  # Increase from 10 (less frequent logging)
logging_first_step=False,
```

**Expected speedup**: Minor (~2-5%) but reduces clutter

### 8. **Optimize Learning Rate (Potential to Reduce Epochs)**

Consider using a learning rate scheduler for faster convergence:

**Add to TrainingArguments:**
```python
lr_scheduler_type="cosine",  # Better than constant LR
warmup_steps=50,  # Warm up learning rate
```

**Benefits**: May converge in 2 epochs instead of 3

### 9. **Batch Size Tuning**

If you have enough VRAM, increase batch size:

**Option A (More memory):**
```python
per_device_train_batch_size=64,  # Double current size
gradient_checkpointing=False,
```

**Option B (Less memory, better gradients):**
```python
per_device_train_batch_size=8,  # Smaller batches
gradient_accumulation_steps=8,  # Effective batch size = 64
```

### 10. **Enable TF32 (A100/4090 GPUs only)**

For Ampere GPUs (A100, 4090, etc.), enable TF32:

**Add before training:**
```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**Expected speedup**: 1.2-1.5x on supported GPUs

## Optimized Configuration Example

Here's a fully optimized training configuration:

```python
# Enable TF32 for Ampere GPUs
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Training arguments
training_args = TrainingArguments(
    output_dir=str(model_path),
    logging_dir=str(model_path),
    report_to="tensorboard",
    
    # Speed optimizations
    fp16=True,  # 2-3x speedup
    gradient_checkpointing=False,  # 20-30% speedup (disable if enough VRAM)
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,  # Faster GPU transfer
    dataloader_prefetch_factor=2,  # Prefetch batches
    
    # Batch size and accumulation
    per_device_train_batch_size=16,  # Adjust based on VRAM
    gradient_accumulation_steps=4,  # Effective batch size = 64
    
    # Training schedule
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",  # Better convergence
    warmup_steps=50,
    
    # Logging and saving
    logging_steps=50,  # Less frequent logging
    logging_first_step=False,
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,  # Faster saves
    
    # Optimization
    optim="adamw_torch_fused",  # Fused optimizer (faster)
    gradient_clip_norm=1.0,  # Gradient clipping for stability
)

# Optional: Compile model (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    lora_model = torch.compile(lora_model, mode='reduce-overhead')
```

## Quick Start: Conservative Optimizations

If you want to be safe and avoid OOM errors, start with these:

```python
training_args = TrainingArguments(
    # ... existing args ...
    
    # Safe optimizations (no OOM risk)
    fp16=True,  # 2-3x speedup, reduces memory
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,
    logging_steps=50,  # Less logging overhead
    optim="adamw_torch_fused",  # Fused optimizer
    
    # Keep gradient checkpointing if worried about memory
    gradient_checkpointing=True,
)
```

## Expected Overall Speedup

With all optimizations (conservative estimate):
- **Current training time**: ~10-15 minutes per epoch
- **With optimizations**: ~3-5 minutes per epoch
- **Overall speedup**: 3-5x faster

## Testing Your Optimizations

To verify optimizations work:

1. **Time baseline:**
   ```bash
   time python -m homework.rft train
   ```

2. **Apply optimizations one by one** and measure impact

3. **Monitor VRAM usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Check final accuracy is maintained:**
   ```bash
   python -m homework.rft test
   ```

## Troubleshooting

**If you get OOM (Out of Memory) errors:**
1. Re-enable `gradient_checkpointing=True`
2. Reduce `per_device_train_batch_size` to 8 or 16
3. Increase `gradient_accumulation_steps` to compensate
4. Reduce `dataloader_num_workers` to 2

**If accuracy drops:**
1. Keep `gradient_accumulation_steps` high (8+) for stable gradients
2. Use `lr_scheduler_type="cosine"` with `warmup_steps=50`
3. Consider training for 4 epochs instead of 3

**If first epoch is very slow:**
- This is normal with `torch.compile` - subsequent epochs will be much faster
- You can disable compile if the overhead isn't worth it for short training

## Data Generation Optimization

Your data generation is already well optimized (see `homework/datagen.py`):
- ✅ CUDA cache clearing after each batch
- ✅ Memory cleanup with `del` statements
- ✅ Micro-batching in `batched_generate`

Consider these additional optimizations:
- Use lower `temperature=0.5` for faster convergence (less diversity needed)
- Reduce `oversample=12` instead of 15 if you're getting 900+ examples already
- Use base 360M model instead of 1.7B for faster generation (trade quality for speed)

## Next Steps

1. **Create an optimized training script** (see `homework/rft_fast.py`)
2. **Run experiments** comparing baseline vs optimized
3. **Monitor accuracy** to ensure optimizations don't hurt quality
4. **Adjust based on your GPU** (different GPUs have different bottlenecks)
