# RFT Training Speed Optimizations

## Overview
This document describes the optimizations applied to speed up RFT (Rejection Fine-Tuning) training.

## Optimizations Applied

### 1. Mixed Precision Training (1.5-2x speedup)
- **bf16 (bfloat16)**: Used on Ampere+ GPUs (A100, RTX 30xx+, etc.)
  - Provides ~1.5-2x speedup over FP32
  - More stable than FP16 (wider dynamic range)
- **fp16 (float16)**: Fallback for older GPUs
  - Still provides speedup but may have numerical stability issues
- **FP32**: Used on CPU/MPS devices

### 2. Disabled Checkpoint Saving
- Changed `save_strategy="epoch"` to `save_strategy="no"`
- Set `save_total_limit=0`
- **Benefit**: Saves time during training by avoiding I/O overhead
- Final model is still saved at the end

### 3. Disabled TensorBoard Logging
- Changed `report_to="tensorboard"` to `report_to="none"`
- **Benefit**: Reduces I/O overhead and disk space usage

### 4. Optimized Dataloader Settings
- `dataloader_pin_memory=True`: Faster data transfer to GPU
- `dataloader_num_workers=0`: Avoids multiprocessing overhead
  - Can increase to 2-4 if data loading becomes a bottleneck

### 5. Gradient Clipping
- `max_grad_norm=1.0`: Prevents gradient explosion
- **Benefit**: Allows for more stable training with potentially higher learning rates

### 6. Learning Rate Scheduler
- `lr_scheduler_type="cosine"`: Cosine decay schedule
- `warmup_ratio=0.1`: 10% warmup steps
- **Benefit**: Better convergence, potentially fewer epochs needed

### 7. Weight Decay
- `weight_decay=0.01`: L2 regularization
- **Benefit**: Better generalization, can help with convergence

### 8. Gradient Accumulation Support
- `gradient_accumulation_steps=1` (can be increased)
- **Benefit**: Allows larger effective batch size without OOM
- **Tip**: Increase to 2-4 if you want larger effective batch size but hit memory limits

## Expected Speedup

The combined optimizations should provide:
- **Mixed precision (bf16)**: 1.5-2x speedup
- **Disabled checkpointing**: ~5-10% speedup (depends on dataset size)
- **Optimized dataloader**: ~5-10% speedup
- **Overall**: **~1.7-2.2x faster training** on modern GPUs

## Further Optimizations You Can Try

### 1. Increase Batch Size
If you have GPU memory available:
```python
per_device_train_batch_size=64,  # Increase from 32
```
Larger batches = fewer steps per epoch = faster training

### 2. Increase Gradient Accumulation
If you want larger effective batch size but hit memory limits:
```python
per_device_train_batch_size=16,
gradient_accumulation_steps=4,  # Effective batch size = 16 * 4 = 64
```

### 3. Increase Dataloader Workers
If data loading is a bottleneck:
```python
dataloader_num_workers=2,  # or 4
```
**Warning**: Can cause issues on some systems, test first

### 4. Reduce Number of Epochs
If the model converges faster with better LR scheduling:
```python
num_train_epochs=2,  # Instead of 3
```
Monitor validation loss to find optimal number of epochs

### 5. Use Compile (PyTorch 2.0+)
For additional speedup on PyTorch 2.0+:
```python
lora_model = torch.compile(lora_model)  # Before training
```
**Note**: May have compatibility issues with some transformers versions

## Monitoring Training Speed

To measure training speed, check the logs:
- Look for "samples/sec" or "tokens/sec" in training logs
- Compare before/after optimizations
- Monitor GPU utilization with `nvidia-smi`

## Memory Considerations

- **Gradient checkpointing**: Already enabled, saves memory
- **Mixed precision**: Reduces memory usage by ~50%
- **Batch size**: Main memory bottleneck, adjust based on GPU memory

## Testing

To test the optimizations:
```bash
cd /workspace/homework3_v3
python -m homework.rft train
```

The training should complete faster while maintaining or improving model quality.
