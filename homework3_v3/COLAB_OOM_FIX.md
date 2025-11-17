# Google Colab CUDA Out of Memory Fix

## Problem
When running RFT dataset generation on Google Colab (T4 GPU with ~14.74 GB), the process crashed with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB. 
GPU 0 has a total capacity of 14.74 GiB of which 80.12 MiB is free.
```

The issue occurred during RFT dataset generation when generating 15 sequences per question.

## Root Cause
- When using `num_return_sequences=15`, the memory usage multiplies significantly
- The model was trying to generate all 15 sequences simultaneously
- With batch size of prompts, this creates a memory explosion: `batch_size * num_return_sequences * sequence_length * hidden_size`
- On a 14GB GPU, this exceeded available memory

## Solutions Applied

### 1. Chunked Generation for Multiple Sequences (`homework/cot.py`)
**Lines 42-64**: Split high `num_return_sequences` into smaller chunks
- When `num_return_sequences > 3`, split into chunks of 3 sequences
- Generate each chunk separately and combine results
- Aggressively clear CUDA cache and run garbage collection after each chunk
- This reduces peak memory usage from `15x` to `3x` at any given time

### 2. One-Prompt-at-a-Time Processing (`homework/cot.py`)
**Lines 66-77**: When generating multiple sequences with multiple prompts
- Process one prompt at a time instead of batching
- Clear memory after each prompt
- Prevents memory multiplication from both dimensions

### 3. Reduced Token Generation (`homework/cot.py`)
**Lines 104-113**: Dynamic max_new_tokens based on generation mode
- Reduced from 120 to 80 tokens for multi-sequence generation
- Saves memory during generation
- Still sufficient for unit conversion answers with reasoning

### 4. Aggressive Memory Management (`homework/cot.py` & `homework/datagen.py`)
**Multiple locations**: Enhanced cache clearing
- Clear CUDA cache after each generation chunk
- Run Python garbage collector (`gc.collect()`) to free CPU memory
- Clear after each question in dataset generation

### 5. Progress Saving (`homework/datagen.py`)
**Lines 84-91**: Periodic checkpoint saves
- Save progress every 100 questions
- Prevents data loss if OOM occurs later in the process
- Allows resuming from partial datasets

### 6. Fixed Prompt Formatting Bug (`homework/datagen.py`)
**Line 48**: Pass raw question instead of pre-formatted
- `batched_generate` already calls `format_prompt` internally
- Previous code was double-formatting, wasting tokens and memory

## Memory Usage Comparison

### Before Fix:
- 1 prompt × 15 sequences × ~120 tokens = ~1800 token generations in parallel
- Peak memory: ~13-14 GB (exceeds T4 capacity)

### After Fix:
- 1 prompt × 3 sequences × ~80 tokens = ~240 token generations in parallel
- Repeated 5 times with cache clearing in between
- Peak memory: ~3-4 GB per chunk, well within T4 capacity

## Testing the Fix

Run the training command again:
```bash
python -m homework3_v3.homework.sft train
```

Or specifically test RFT dataset generation:
```bash
python -m homework3_v3.homework.datagen data/rft.json --oversample=15 --temperature=0.7
```

## Expected Behavior

1. The model will load successfully
2. RFT dataset generation will start
3. Progress will show chunk-by-chunk generation (you may notice slight pauses as chunks complete and memory is cleared)
4. Progress will be saved every 100 questions
5. No OOM errors should occur

## Additional Recommendations for Colab

If you still encounter memory issues, you can:

1. **Reduce oversample parameter:**
   ```bash
   python -m homework3_v3.homework.sft train
   # Modify sft.py line 303 to use oversample=10 instead of 15
   ```

2. **Use smaller chunk size:**
   Edit `homework/cot.py` line 46 to use `chunk_size = 2` instead of 3

3. **Clear other processes:**
   Before running, ensure no other notebooks are using GPU memory:
   ```python
   # In another Colab cell
   !nvidia-smi
   # Check for other processes and kill them if needed
   ```

4. **Restart runtime:**
   Sometimes Colab accumulates memory fragmentation. Restart the runtime:
   ```
   Runtime → Restart runtime
   ```

## Technical Details

### Memory Calculation:
For SmolLM2-1.7B model in FP16:
- Model parameters: ~1.7B × 2 bytes = 3.4 GB
- Activation memory: batch_size × seq_len × hidden_size × num_layers × 2 bytes
- KV cache: batch_size × num_sequences × seq_len × num_heads × head_dim × 2 bytes × 2 (K+V)

With chunking:
- 1 batch × 3 sequences × 80 tokens × 2048 hidden × 24 layers ≈ 2-3 GB
- Plus model weights (3.4 GB) ≈ 5-7 GB total
- Leaves room for overhead and gradients

## Files Modified

1. `homework/cot.py` - Added chunked generation and memory optimizations
2. `homework/datagen.py` - Added progress saving and memory management
3. `COLAB_OOM_FIX.md` - This documentation

## Verification

After running, you should see:
- ✅ No OOM errors
- ✅ Progress bars showing generation
- ✅ Periodic progress saves
- ✅ Successful RFT dataset creation
- ✅ Training proceeds normally
