# RFT Dataset Generation Fix

## Issues Found and Fixed

### 1. **Chunking Logic Bug**
   - **Problem**: When `oversample=15`, the code wasn't properly chunking generations, potentially causing memory issues or hangs
   - **Fix**: Changed the threshold from `> 15` to `>= 10` so that 15 sequences are properly chunked into groups of 3
   - **Location**: `homework/cot.py` line 51

### 2. **Missing Progress Indicators**
   - **Problem**: No feedback during model loading or first generation, making it unclear if the process is working
   - **Fix**: Added progress messages for:
     - Model loading
     - Model warmup
     - Generation start
   - **Location**: `homework/datagen.py` lines 43-76

### 3. **No Error Handling**
   - **Problem**: If generation failed for a question, the entire process would crash
   - **Fix**: Added try-catch around generation with error logging
   - **Location**: `homework/datagen.py` lines 82-92

### 4. **Model Warmup Missing**
   - **Problem**: First generation is often very slow (model compilation/warmup)
   - **Fix**: Added explicit warmup generation before processing questions
   - **Location**: `homework/datagen.py` lines 55-69

## What to Expect Now

When you run RFT training, you should see:

1. **Model Loading** (30-60 seconds on first run, faster if cached):
   ```
   Loading CoT model (1.7B) for RFT dataset generation...
   This may take a minute on first run (downloading model if needed)...
   Model loaded successfully!
   ```

2. **Model Warmup** (10-30 seconds):
   ```
   Warming up model (this may take 10-30 seconds)...
   Model warmup complete!
   ```

3. **Generation Progress**:
   ```
   Generating RFT dataset with 15 sequences per question...
   Processing 1000 questions...
   Generating RFT dataset: 1% 10/1000 [00:45<1:12:34, 4.5s/it]
   ```

## If It Still Hangs

If the generation still appears to hang, check:

1. **GPU Memory**: The 1.7B model needs ~4-6GB GPU memory
   ```bash
   nvidia-smi  # Check GPU memory usage
   ```

2. **Model Download**: On first run, the model downloads (~3.5GB)
   - Check network connection
   - Check disk space in HuggingFace cache

3. **First Generation**: The first question can take 30-60 seconds
   - This is normal due to model compilation
   - The warmup should help, but first real generation may still be slow

4. **Check Logs**: Look for any error messages that might indicate the issue

## Performance Notes

- **Expected time**: ~1-2 hours for 1000 questions with 15 sequences each
- **Memory usage**: ~6-8GB GPU memory during generation
- **Each question**: Takes ~4-8 seconds (15 sequences × 3 chunks × ~0.5-1s per chunk)

## Testing the Fix

To test if the fix works:

```bash
# Generate a small test dataset first
python -m homework.datagen data/rft_test.json --oversample=5

# If that works, generate the full dataset
python -m homework.datagen data/rft.json --oversample=15
```

The test with `oversample=5` should complete much faster and verify the generation pipeline works.
