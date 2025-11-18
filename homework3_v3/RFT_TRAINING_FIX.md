# RFT Model Training Setup - Verification and Fixes

## Issues Found and Fixed

### 1. Model Saving Strategy
**Problem:** The RFT training code was using `save_strategy="epoch"` which saves checkpoints in subdirectories (e.g., `rft_model/checkpoint-XXX/`). While `trainer.save_model()` should save to the main directory, this can cause confusion and potential path issues.

**Fix:** Changed to match the SFT training approach:
- Set `save_strategy="no"` to disable checkpoint saving during training
- Set `save_total_limit=0` to ensure no checkpoints are kept
- Explicitly call `trainer.save_model(str(model_path))` with the path as a string argument
- Added verification to ensure model files are actually saved

### 2. Model File Verification
**Problem:** No verification that model files were actually saved after training.

**Fix:** Added verification code that checks for adapter files:
- `adapter_model.bin` or `adapter_model.safetensors` (LoRA weights)
- `adapter_config.json` (LoRA configuration)

If files are missing, training will raise an error with a clear message.

## Path Resolution

The model path resolution works as follows:
- `MODEL_NAME = "rft_model"` (relative path)
- `_resolve_path(MODEL_NAME)` resolves to `homework/rft_model/` (relative to `homework/sft.py` or `homework/rft.py`)
- Both `train_model()` and `load()` use the same `_resolve_path()` function, ensuring consistency

## Training Workflow

1. **Training:** `python -m homework.rft train`
   - Loads RFT dataset from `data/rft.json`
   - Trains LoRA adapter for 3 epochs
   - Saves model to `homework/rft_model/`

2. **Testing:** `python -m homework.rft test`
   - Loads model from `homework/rft_model/`
   - Runs benchmark on validation set
   - Should show accuracy > 0.6

3. **Loading (for grader):** `load_rft()` function
   - Resolves path to `homework/rft_model/`
   - Loads adapter files
   - Returns `BaseLLM` instance with trained adapter

## Bundle Inclusion

The `bundle.py` script should include the adapter files:
- `adapter_model.bin` or `adapter_model.safetensors` - NOT in BLACKLIST ✓
- `adapter_config.json` - NOT in BLACKLIST ✓
- `README.md` - IN BLACKLIST (excluded, which is fine)

The adapter files will be included in the bundle as long as they exist in `homework/rft_model/`.

## Next Steps

1. **Train the RFT model:**
   ```bash
   python -m homework.rft train
   ```

2. **Verify model files exist:**
   ```bash
   ls -la homework/rft_model/
   # Should show: adapter_model.bin (or .safetensors) and adapter_config.json
   ```

3. **Test the model:**
   ```bash
   python -m homework.rft test
   # Should show accuracy > 0.6
   ```

4. **Create bundle:**
   ```bash
   python3 bundle.py homework <UTID>
   ```

5. **Verify bundle includes model files:**
   ```bash
   unzip -l <UTID>.zip | grep rft_model
   # Should show adapter files
   ```
