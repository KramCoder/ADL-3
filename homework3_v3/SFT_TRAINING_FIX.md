# SFT Training Fix - Gradient and Loss Issue

## Problem

The SFT training was experiencing:
- Loss remaining at 0.0 throughout training
- Gradient norm showing as NaN
- Final accuracy of 0.0
- Warning: "No label_names provided for model class `PeftModelForCausalLM`"

## Root Cause

The issue was that the HuggingFace Trainer wasn't properly recognizing the `labels` field in the dataset when using a PeftModel (LoRA adapter). This caused:
1. The loss calculation to fail (resulting in 0.0 loss)
2. No gradients being computed (resulting in NaN grad_norm)
3. The model not learning anything

## Solution

Two key changes were made to `homework/sft.py`:

### 1. Explicitly Set Label Names in TrainingArguments

Added `label_names=["labels"]` to the TrainingArguments to explicitly tell the Trainer which field contains the labels:

```python
training_args = TrainingArguments(
    output_dir=str(model_path),
    logging_dir=str(model_path),
    report_to="tensorboard",
    gradient_checkpointing=True,
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    label_names=["labels"],  # ← THIS IS THE KEY FIX
)
```

### 2. Use Explicit Data Collator

Added `default_data_collator` explicitly to the Trainer to ensure proper batching without overwriting the custom labels:

```python
from transformers import Trainer, TrainingArguments, default_data_collator

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator,  # ← Explicitly use default collator
)
```

## Verification

After the fix, training now shows:
- ✅ Non-zero loss values (e.g., 2.27, 2.29, 2.49)
- ✅ Valid gradient norms (e.g., 1.55, 1.81, 1.62)
- ✅ Proper learning progression

## Technical Details

The `tokenize()` function in `sft.py` creates custom labels where:
- Question tokens are masked with -100 (not used for loss calculation)
- Answer tokens retain their original token IDs (used for loss calculation)
- Padding tokens are also masked with -100

The Trainer needs to know:
1. Which field contains the labels (`label_names=["labels"]`)
2. How to collate the data into batches (using `default_data_collator`)

Without these explicit settings, PeftModel's wrapper around the base model prevented the Trainer from automatically detecting the labels, resulting in no loss calculation and no gradient updates.
