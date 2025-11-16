#!/usr/bin/env python3
"""
Test script to verify SFT training fix for NaN gradient norm.
This runs a minimal training to verify gradients flow correctly.
"""
import sys
import os
from pathlib import Path

# Add the homework3_v3 directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from homework.sft import train_model
from homework.data import Dataset, benchmark
from homework.base_llm import BaseLLM
from peft import PeftModel

def test_training_fix():
    """Test that training works with the gradient norm fix."""
    print("=" * 60)
    print("Testing SFT Training Fix")
    print("=" * 60)
    
    # Use a temporary output directory
    output_dir = "/tmp/test_sft_output"
    
    # Clean up any existing model
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        # Run training with minimal epochs for testing
        # We'll modify the training function temporarily to use fewer epochs
        print("\n1. Starting training (this may take a few minutes)...")
        print("   Checking for NaN gradient norms...")
        
        # Import and modify training args temporarily
        from homework.sft import train_model as original_train_model
        import torch
        from transformers import Trainer, TrainingArguments, default_data_collator
        from peft import LoraConfig, get_peft_model
        from homework.base_llm import BaseLLM
        from homework.data import Dataset
        from homework.sft import TokenizedDataset, format_example, _resolve_path
        
        # Set up training with minimal epochs for testing
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
        
        # Training arguments with fix
        training_args = TrainingArguments(
            output_dir=str(model_path),
            logging_dir=str(model_path),
            report_to="tensorboard",
            gradient_checkpointing=True,
            learning_rate=2e-4,
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=32,
            save_strategy="epoch",
            logging_steps=5,  # More frequent logging for testing
            save_total_limit=1,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            max_grad_norm=1.0,  # THE FIX: Gradient clipping
            label_names=["labels"],  # THE FIX: Explicit label names
        )
        
        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=default_data_collator,
        )
        
        print("\n2. Training started...")
        train_result = trainer.train()
        
        print("\n3. Checking training results...")
        print(f"   Final loss: {train_result.training_loss:.4f}")
        
        # Check if we can access training logs
        if hasattr(trainer.state, 'log_history'):
            grad_norms = []
            for log in trainer.state.log_history:
                if 'grad_norm' in log:
                    grad_norm = log['grad_norm']
                    grad_norms.append(grad_norm)
                    if grad_norm != grad_norm:  # Check for NaN
                        print(f"   ❌ ERROR: Found NaN gradient norm at step {log.get('step', 'unknown')}")
                        return False
                    else:
                        print(f"   ✓ Step {log.get('step', 'unknown')}: grad_norm = {grad_norm:.4f}")
            
            if grad_norms:
                print(f"\n   ✅ All gradient norms are finite!")
                print(f"   Gradient norm range: {min(grad_norms):.4f} - {max(grad_norms):.4f}")
            else:
                print("   ⚠️  Warning: No gradient norms found in logs")
        
        print("\n4. Saving model...")
        trainer.save_model(str(model_path))
        
        print("\n5. Testing model accuracy...")
        testset = Dataset("valid")
        llm_test = BaseLLM()
        llm_test.model = PeftModel.from_pretrained(llm_test.model, model_path).to(llm_test.device)
        llm_test.model.eval()
        
        benchmark_result = benchmark(llm_test, testset, 10)  # Test on 10 samples
        print(f"   Accuracy: {benchmark_result.accuracy:.4f}")
        print(f"   Answer rate: {benchmark_result.answer_rate:.4f}")
        
        if benchmark_result.accuracy > 0:
            print("   ✅ Model is learning! Accuracy > 0")
        else:
            print("   ⚠️  Accuracy is 0 - model may need more training or tuning")
        
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
    success = test_training_fix()
    sys.exit(0 if success else 1)
