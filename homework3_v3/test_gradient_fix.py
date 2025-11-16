"""Test script to verify the gradient NaN fix."""

import torch
from homework.sft import TokenizedDataset, format_example, DEFAULT_LORA_RANK
from homework.data import Dataset
from homework.base_llm import BaseLLM
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, default_data_collator


def test_single_forward_backward():
    """Test a single forward and backward pass."""
    print("=" * 80)
    print("TEST 1: Single Forward/Backward Pass")
    print("=" * 80)
    
    # Load model in FP32
    llm = BaseLLM(use_fp32_for_training=True)
    print(f"Model dtype: {next(llm.model.parameters()).dtype}")
    print(f"Expected: torch.float32")
    
    # Create LoRA model
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 4,
        lora_dropout=0.0,
    )
    
    lora_model = get_peft_model(llm.model, config)
    lora_model.enable_input_require_grads()
    lora_model.train()
    
    # Get a sample
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    sample = tokenized_dataset[0]
    
    # Create batch
    input_ids = torch.tensor([sample["input_ids"]]).to(llm.device)
    attention_mask = torch.tensor([sample["attention_mask"]]).to(llm.device)
    labels = torch.tensor([sample["labels"]]).to(llm.device)
    
    print(f"\nBatch shape: {input_ids.shape}")
    print(f"Non-masked labels: {(labels != -100).sum().item()}")
    
    # Forward pass
    outputs = lora_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    loss = outputs.loss
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss is NaN: {torch.isnan(loss).item()}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = []
    nan_count = 0
    inf_count = 0
    
    for name, param in lora_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            
            if torch.isnan(param.grad).any():
                nan_count += 1
                print(f"  ⚠️  NaN gradient in: {name}")
                
            if torch.isinf(param.grad).any():
                inf_count += 1
                print(f"  ⚠️  Inf gradient in: {name}")
    
    print(f"\nGradient Statistics:")
    print(f"  Total parameters with gradients: {len(grad_norms)}")
    print(f"  Parameters with NaN gradients: {nan_count}")
    print(f"  Parameters with Inf gradients: {inf_count}")
    
    if grad_norms:
        total_norm = torch.sqrt(torch.tensor([g**2 for g in grad_norms]).sum())
        print(f"  Total gradient norm: {total_norm:.6f}")
        print(f"  Total gradient norm is NaN: {torch.isnan(total_norm).item()}")
        print(f"  Total gradient norm is finite: {torch.isfinite(total_norm).item()}")
    
    # Result
    if nan_count == 0 and inf_count == 0 and torch.isfinite(loss):
        print("\n✅ TEST PASSED: No NaN or Inf gradients, loss is finite")
        return True
    else:
        print("\n❌ TEST FAILED: Found NaN/Inf gradients or non-finite loss")
        return False


def test_mini_training():
    """Test a few training steps."""
    print("\n" + "=" * 80)
    print("TEST 2: Mini Training (10 steps)")
    print("=" * 80)
    
    # Load model in FP32
    llm = BaseLLM(use_fp32_for_training=True)
    
    # Create LoRA model
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 4,
        lora_dropout=0.0,
    )
    
    lora_model = get_peft_model(llm.model, config)
    lora_model.enable_input_require_grads()
    lora_model.train()
    
    # Prepare dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Training arguments (same as in sft.py but fewer steps)
    training_args = TrainingArguments(
        output_dir="/tmp/test_training",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=1,
        per_device_train_batch_size=8,  # Smaller batch for faster test
        max_steps=10,  # Only 10 steps
        logging_steps=2,
        save_strategy="no",  # Don't save
        remove_unused_columns=False,
        fp16=False,  # Disabled to prevent NaN
        bf16=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        label_names=["labels"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )
    
    # Train
    print("\nStarting mini-training...")
    result = trainer.train()
    
    # Check results
    train_loss = result.training_loss
    print(f"\nFinal training loss: {train_loss:.4f}")
    print(f"Loss is finite: {torch.isfinite(torch.tensor(train_loss)).item()}")
    
    # Get history
    history = trainer.state.log_history
    
    # Check for NaN in logs
    nan_found = False
    for entry in history:
        if 'grad_norm' in entry:
            grad_norm = entry['grad_norm']
            if grad_norm != grad_norm:  # NaN check
                print(f"  ⚠️  Found NaN grad_norm at step {entry.get('step', 'unknown')}")
                nan_found = True
    
    if not nan_found and torch.isfinite(torch.tensor(train_loss)):
        print("\n✅ TEST PASSED: No NaN gradients, loss is finite")
        return True
    else:
        print("\n❌ TEST FAILED: Found NaN gradients or non-finite loss")
        return False


def main():
    print("Testing Gradient NaN Fix\n")
    
    test1_passed = test_single_forward_backward()
    test2_passed = test_mini_training()
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Test 1 (Single Forward/Backward): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Mini Training): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The gradient NaN issue is fixed.")
        print("You can now run full training with: python -m homework.sft train")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. The issue persists.")
        return 1


if __name__ == "__main__":
    exit(main())
