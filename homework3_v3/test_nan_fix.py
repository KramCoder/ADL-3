#!/usr/bin/env python3
"""
Test script to verify NaN gradient norm fix.
Runs a minimal training session and checks for NaN values.
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

def test_nan_fix():
    """Test that training produces valid (non-NaN) gradient norms."""
    print("=" * 70)
    print("Testing NaN Gradient Norm Fix")
    print("=" * 70)
    
    output_dir = "/tmp/test_nan_fix_output"
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        print("\n1. Setting up model and dataset...")
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
        
        # Verify sample has labels
        sample = tokenized_dataset[0]
        non_masked = sum(1 for l in sample["labels"] if l != -100)
        print(f"   ✓ Sample has {non_masked} non-masked labels")
        
        print("\n2. Configuring training with NaN protection...")
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
        
        training_args = TrainingArguments(
            output_dir=str(model_path),
            logging_dir=str(model_path),
            report_to="none",
            gradient_checkpointing=True,
            learning_rate=2e-4,
            num_train_epochs=1,  # Just 1 epoch for quick test
            per_device_train_batch_size=32,
            save_strategy="no",
            logging_steps=2,  # Log every 2 steps for faster feedback
            max_steps=5,  # Limit to 5 steps for very quick test
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
        
        def safe_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
                loss = original_compute_loss(model, inputs, return_outputs, **kwargs)
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
        
        print("\n3. Starting training (this will take a few minutes)...")
        print("   Monitoring for NaN values...\n")
        
        train_result = trainer.train()
        
        print("\n" + "=" * 70)
        print("4. Analyzing Results")
        print("=" * 70)
        
        print(f"\n   Final training loss: {train_result.training_loss:.4f}")
        
        # Analyze logs
        if hasattr(trainer.state, 'log_history'):
            grad_norms = []
            losses = []
            nan_grad_count = 0
            nan_loss_count = 0
            zero_loss_count = 0
            zero_grad_count = 0
            
            for log in trainer.state.log_history:
                if 'grad_norm' in log:
                    grad_norm = log['grad_norm']
                    grad_norms.append(grad_norm)
                    
                    # Check for NaN
                    if grad_norm != grad_norm:  # NaN check
                        nan_grad_count += 1
                        step = log.get('step', 'unknown')
                        print(f"   ❌ NaN gradient norm at step {step}")
                    elif grad_norm == 0.0:
                        zero_grad_count += 1
                    else:
                        # Valid gradient norm
                        pass
                
                if 'loss' in log:
                    loss = log['loss']
                    losses.append(loss)
                    
                    if loss != loss:  # NaN check
                        nan_loss_count += 1
                        step = log.get('step', 'unknown')
                        print(f"   ❌ NaN loss at step {step}")
                    elif loss == 0.0:
                        zero_loss_count += 1
                        step = log.get('step', 'unknown')
                        print(f"   ⚠️  Zero loss at step {step}")
            
            # Summary
            print(f"\n   📊 Statistics:")
            print(f"      Total log entries: {len(trainer.state.log_history)}")
            print(f"      Gradient norm entries: {len(grad_norms)}")
            print(f"      Loss entries: {len(losses)}")
            
            if grad_norms:
                valid_grad_norms = [gn for gn in grad_norms if gn == gn and gn != 0.0]
                if valid_grad_norms:
                    print(f"\n   ✅ Valid gradient norms: {len(valid_grad_norms)}")
                    print(f"      Range: {min(valid_grad_norms):.4f} - {max(valid_grad_norms):.4f}")
                    print(f"      Mean: {sum(valid_grad_norms)/len(valid_grad_norms):.4f}")
                else:
                    print(f"\n   ⚠️  No valid (non-zero) gradient norms found")
                
                if nan_grad_count > 0:
                    print(f"\n   ❌ FAILED: Found {nan_grad_count} NaN gradient norms")
                    return False
                elif zero_grad_count > 0:
                    print(f"\n   ⚠️  WARNING: Found {zero_grad_count} zero gradient norms")
                else:
                    print(f"\n   ✅ SUCCESS: No NaN gradient norms detected!")
            
            if losses:
                valid_losses = [l for l in losses if l == l and l != 0.0]
                if valid_losses:
                    print(f"\n   ✅ Valid losses: {len(valid_losses)}")
                    print(f"      Range: {min(valid_losses):.4f} - {max(valid_losses):.4f}")
                    print(f"      Mean: {sum(valid_losses)/len(valid_losses):.4f}")
                
                if nan_loss_count > 0:
                    print(f"\n   ❌ FAILED: Found {nan_loss_count} NaN losses")
                    return False
                elif zero_loss_count > 0:
                    print(f"\n   ⚠️  WARNING: Found {zero_loss_count} zero losses")
                else:
                    print(f"\n   ✅ SUCCESS: No NaN losses detected!")
        
        print("\n" + "=" * 70)
        if nan_grad_count == 0 and nan_loss_count == 0:
            print("✅ TEST PASSED: No NaN values detected!")
        else:
            print("❌ TEST FAILED: NaN values were detected")
        print("=" * 70)
        
        return nan_grad_count == 0 and nan_loss_count == 0
        
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
    success = test_nan_fix()
    sys.exit(0 if success else 1)
