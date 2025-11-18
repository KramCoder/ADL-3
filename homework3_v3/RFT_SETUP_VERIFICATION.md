# RFT Model Training Setup - Complete Verification

## Summary of Changes

### ✅ Fixed Model Saving Strategy
**Changed in `homework/rft.py`:**

1. **Training Arguments:**
   - Changed `save_strategy="epoch"` → `save_strategy="no"`
   - Changed `save_total_limit=1` → `save_total_limit=0`
   - This prevents checkpoint subdirectories and ensures clean model saving

2. **Model Saving:**
   - Changed `trainer.save_model()` → `trainer.save_model(str(model_path))`
   - Added explicit path argument to ensure saving to correct location
   - Added verification code to check that adapter files exist after saving

3. **File Verification:**
   - Added check for `adapter_model.bin` or `adapter_model.safetensors`
   - Added check for `adapter_config.json`
   - Raises clear error if files are missing

## Path Resolution Verification

### Model Path Resolution
- `MODEL_NAME = "rft_model"` (defined in `rft.py`)
- `_resolve_path(MODEL_NAME)` resolves relative paths to `Path(__file__).parent / candidate`
- Since `rft.py` is in `homework/`, the path resolves to `homework/rft_model/`
- Both `train_model()` and `load()` use the same `_resolve_path()` function
- **✅ Paths are consistent between training and loading**

### Expected File Structure After Training
```
homework/
  rft_model/
    adapter_model.bin          # or adapter_model.safetensors
    adapter_config.json
    README.md                  # (excluded from bundle, but that's fine)
```

## Bundle Inclusion Verification

### Bundle Script Analysis
The `bundle.py` BLACKLIST excludes:
- `README.md` and `.md` files
- `checkpoint-` directories
- Training state files (`optimizer.pt`, `scheduler.pt`, `trainer_state.json`)

### ✅ Adapter Files Will Be Included
- `adapter_model.bin` - NOT in BLACKLIST
- `adapter_model.safetensors` - NOT in BLACKLIST  
- `adapter_config.json` - NOT in BLACKLIST

**Conclusion:** Adapter files will be included in the bundle as long as they exist in `homework/rft_model/`.

## Training Workflow

### Step 1: Generate RFT Dataset (if needed)
```bash
python -m homework.datagen data/rft.json --oversample=15 --temperature=0.7
```

### Step 2: Train RFT Model
```bash
python -m homework.rft train
```

**Expected output:**
- Training progress logs
- Final loss metrics
- "Model saved successfully. Found files: ['adapter_model.bin', 'adapter_config.json']" (or .safetensors)
- Test results showing accuracy > 0.6

### Step 3: Verify Model Files
```bash
ls -la homework/rft_model/
# Should show:
# - adapter_model.bin (or adapter_model.safetensors)
# - adapter_config.json
# - README.md (optional, will be excluded from bundle)
```

### Step 4: Test Model
```bash
python -m homework.rft test
# Should show: benchmark_result.accuracy=0.6X  benchmark_result.answer_rate=0.XX
```

### Step 5: Create Bundle
```bash
python3 bundle.py homework <UTID>
```

### Step 6: Verify Bundle Contents
```bash
unzip -l <UTID>.zip | grep rft_model
# Should show:
# homework/rft_model/adapter_model.bin (or .safetensors)
# homework/rft_model/adapter_config.json
```

## Grader Expectations

The grader expects:
1. `load_rft()` function that returns a `BaseLLM` instance
2. Model files in `homework/rft_model/` directory
3. Model accuracy between 0.6 (minimum) and 0.7 (full score) on validation set

### Current Status
- ✅ `load_rft()` function exists and is exported
- ✅ Path resolution is correct
- ✅ Model saving is now fixed to save to correct location
- ⚠️ Model needs to be trained (files don't exist yet)
- ⚠️ Model needs to achieve accuracy > 0.6

## Potential Issues and Solutions

### Issue 1: Model Files Not Found
**Symptom:** `load_rft()` creates empty adapter, accuracy = 0.0

**Solution:**
- Train the model: `python -m homework.rft train`
- Verify files exist: `ls homework/rft_model/`
- Check that training completed successfully

### Issue 2: Model Accuracy Too Low
**Symptom:** Accuracy < 0.6 after training

**Solutions:**
- Check RFT dataset quality: `python -m homework.rft train` (will show dataset stats)
- Ensure dataset has 850+ examples
- Verify dataset format: each example should have `[question, answer, reasoning]` with `<answer>` tags
- Consider adjusting training hyperparameters (learning rate, epochs, etc.)

### Issue 3: Model Files Not in Bundle
**Symptom:** Bundle doesn't include adapter files

**Solution:**
- Verify files exist before bundling: `ls homework/rft_model/`
- Check bundle output: `unzip -l <UTID>.zip | grep rft_model`
- Ensure files are not in a checkpoint subdirectory (should be directly in `rft_model/`)

## Next Steps

1. **Train the model** (if not already trained)
2. **Verify model files exist** in `homework/rft_model/`
3. **Test the model** to ensure accuracy > 0.6
4. **Create bundle** and verify adapter files are included
5. **Run grader** to verify RFT model gets points
