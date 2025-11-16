# Testing the SFT Training Fix

## What Was Fixed

The SFT training was showing:
- ❌ Loss: 0.0
- ❌ Gradient norm: NaN
- ❌ Accuracy: 0.0

The fix ensures:
- ✅ Non-zero loss values
- ✅ Valid gradient norms
- ✅ Model actually learns

## How to Test

### Quick Test (3 steps only)

Run this command to do a quick 3-step training test:

```bash
cd /workspace/homework3_v3
python3 -c "
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, default_data_collator
from homework.base_llm import BaseLLM
from homework.sft import format_example, TokenizedDataset
from homework.data import Dataset

llm = BaseLLM()
config = LoraConfig(
    task_type='CAUSAL_LM',
    target_modules='all-linear',
    bias='none',
    r=4,
    lora_alpha=16,
    lora_dropout=0.0,
    inference_mode=False,
)

lora_model = get_peft_model(llm.model, config)
lora_model.train()
lora_model.enable_input_require_grads()

train_dataset = Dataset('train')
tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)

training_args = TrainingArguments(
    output_dir='/tmp/test_sft',
    gradient_checkpointing=True,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    max_steps=3,
    logging_steps=1,
    save_strategy='no',
    label_names=['labels'],
    report_to='none',
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator,
)

result = trainer.train()
print(f'\\n✓ Final loss: {result.training_loss:.4f}')
print('✓ Training is working!' if result.training_loss > 0 else '✗ Training still broken!')
"
```

Expected output should show:
- Loss values around 2.2-2.5
- Gradient norms around 1.5-1.8
- No NaN values

### Full Training

To run the full SFT training:

```bash
cd /workspace/homework3_v3
python3 -m homework.sft train
```

This will:
1. Train for 3 epochs
2. Save checkpoints after each epoch
3. Test the model on the validation set
4. Show accuracy > 0.0 (if successful)

## What to Look For

### Before the Fix
```
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00018125, 'epoch': 0.31}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.00016042, 'epoch': 0.62}
```

### After the Fix
```
{'loss': 2.2723, 'grad_norm': 1.5494, 'learning_rate': 0.0002, 'epoch': 0.0}
{'loss': 2.2868, 'grad_norm': 1.8132, 'learning_rate': 0.00013333, 'epoch': 0.01}
```

The key indicators that it's working:
- ✅ `loss` > 0 (typically 2.0-3.0 initially)
- ✅ `grad_norm` is a finite number (typically 1.0-2.0)
- ✅ Loss should decrease over time (though may fluctuate)

## Changes Made

See `SFT_TRAINING_FIX.md` for technical details about the fix.

The changes were minimal but critical:
1. Added `label_names=["labels"]` to TrainingArguments
2. Added `data_collator=default_data_collator` to Trainer
