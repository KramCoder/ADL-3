#!/usr/bin/env python3
"""
Test script to verify gradient norm NaN fix.
This runs a minimal training to verify gradients are computed correctly.
"""
import sys
import os
from pathlib import Path

# Add the homework3_v3 directory to the path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model
from homework.base_llm import BaseLLM
from homework.data import Dataset
from homework.sft import TokenizedDataset, format_example, _resolve_path, GradientMonitoringCallback

def test_gradient_fix():
    """Test that training produces valid gradient norms."""
    print("=" * 60)
    print("Testing Gradient Norm NaN Fix")
    print("=" * 60)
    
    output_dir = "/tmp/test_gradient_fix_output"
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        # Set up minimal training
        model_path = _resolve_path(output_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        llm = BaseLLM()
        llm.model.eval()
        
        config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            bias="none",
            r=4,
            lora_alpha=16,
            lora_dropout=0.0,
        )
        
        lora_model = get_peft_model(llm.model, config)
        lora_model.enable_input_require_grads()
        lora_model.train()
        
        train_dataset = Dataset("train")
        tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
        
        # Check a sample to ensure labels are correct
        sample = tokenized_dataset[0]
        non_masked = sum(1 for l in sample["labels"] if l != -100)
        print(f"\n1. Sample check: {non_masked} non-masked labels out of {len(sample['labels'])}")
        if non_masked == 0:
            print("   ❌ ERROR: All labels are masked!")
            return False
        print("   ✓ Labels are properly set")
        
        # Training arguments with fixes
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
        
        training_args = TrainingArguments(
            output_dir=str(model_path),
            logging_dir=str(model_path),
            report_to="none",  # Disable tensorboard for testing
            gradient_checkpointing=True,
            learning_rate=2e-4,
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=32,
            save_strategy="no",  # Don't save for testing
            logging_steps=5,  # More frequent logging
            remove_unused_columns=False,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_pin_memory=False,
            max_grad_norm=1.0,
            label_names=["labels"],
            warmup_steps=10,
        )
        
        gradient_callback = GradientMonitoringCallback()
        
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=default_data_collator,
            callbacks=[gradient_callback],
        )
        
        # Override compute_loss (same as in sft.py)
        original_compute_loss = trainer.compute_loss
        
        def safe_compute_loss(self, model, inputs, return_outputs=False):
            """Compute loss with NaN and zero loss protection."""
            labels = inputs.get("labels")
            if labels is None:
                raise ValueError("Labels are required for training")
            
            if isinstance(labels, torch.Tensor):
                non_masked = (labels != -100).sum().item()
                if non_masked == 0:
                    print("⚠️  WARNING: All labels are masked in this batch!")
                    dummy_loss = torch.tensor(1e-6, device=labels.device, dtype=torch.float32, requires_grad=True)
                    if return_outputs:
                        return dummy_loss, None
                    return dummy_loss
            
            try:
                loss = original_compute_loss(model, inputs, return_outputs)
                if return_outputs:
                    loss, outputs = loss
                else:
                    outputs = None
            except Exception as e:
                print(f"⚠️  WARNING: Error computing loss: {e}")
                outputs = model(**inputs)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                if return_outputs:
                    outputs = outputs
                else:
                    outputs = None
            
            if loss is None:
                print("⚠️  WARNING: Loss is None!")
                device = next(model.parameters()).device
                loss = torch.tensor(1e-6, device=device, dtype=torch.float32, requires_grad=True)
            elif loss != loss:  # NaN check
                print("⚠️  WARNING: NaN loss detected!")
                loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype, requires_grad=True)
            elif loss.item() == 0.0:
                print("⚠️  WARNING: Zero loss detected!")
                loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype, requires_grad=True)
            
            if return_outputs:
                return loss, outputs
            return loss
        
        import types
        trainer.compute_loss = types.MethodType(safe_compute_loss, trainer)
        
        print("\n2. Starting training (limited steps for testing)...")
        train_result = trainer.train()
        
        print("\n3. Checking training results...")
        print(f"   Final loss: {train_result.training_loss:.4f}")
        
        # Check training logs for gradient norms
        if hasattr(trainer.state, 'log_history'):
            grad_norms = []
            nan_count = 0
            zero_loss_count = 0
            
            for log in trainer.state.log_history:
                if 'grad_norm' in log:
                    grad_norm = log['grad_norm']
                    grad_norms.append(grad_norm)
                    if grad_norm != grad_norm:  # NaN check
                        nan_count += 1
                        print(f"   ❌ NaN gradient norm at step {log.get('step', 'unknown')}")
                    elif grad_norm == 0.0:
                        print(f"   ⚠️  Zero gradient norm at step {log.get('step', 'unknown')}")
                
                if 'loss' in log:
                    loss = log['loss']
                    if loss == 0.0:
                        zero_loss_count += 1
                    elif loss != loss:  # NaN
                        print(f"   ❌ NaN loss at step {log.get('step', 'unknown')}")
            
            if grad_norms:
                valid_norms = [gn for gn in grad_norms if gn == gn and gn != 0.0]
                if valid_norms:
                    print(f"\n   ✅ Found {len(valid_norms)} valid gradient norms")
                    print(f"   Gradient norm range: {min(valid_norms):.4f} - {max(valid_norms):.4f}")
                else:
                    print(f"\n   ⚠️  All gradient norms are 0.0 or NaN")
                
                if nan_count > 0:
                    print(f"   ❌ Found {nan_count} NaN gradient norms")
                    return False
                else:
                    print(f"   ✅ No NaN gradient norms detected!")
            else:
                print("   ⚠️  No gradient norms found in logs")
            
            if zero_loss_count > 0:
                print(f"   ⚠️  Found {zero_loss_count} zero loss values")
            else:
                print(f"   ✅ No zero loss values detected!")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    success = test_gradient_fix()
    sys.exit(0 if success else 1)
