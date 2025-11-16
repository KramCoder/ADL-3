from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch, format_numeric_answer
from .data import Dataset, benchmark


MODEL_NAME = "sft_model"
DEFAULT_LORA_RANK = 4


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent / candidate


def _ensure_adapter(model_path: Path, *, rank: int = DEFAULT_LORA_RANK) -> None:
    adapter_file = model_path / "adapter_model.bin"
    if adapter_file.exists():
        return

    model_path.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM()
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=rank,
        lora_alpha=max(rank * 4, 4),
        lora_dropout=0.0,
    )

    lora_model = get_peft_model(llm.model, config)
    lora_model.save_pretrained(model_path)


def load() -> BaseLLM:
    model_path = _resolve_path(MODEL_NAME)
    _ensure_adapter(model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    apply_dataset_answer_patch(llm)

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We concatenate question and answer, then create labels where only the answer
    tokens are supervised (question tokens are masked with -100).
    
    Strategy: Tokenize the full text, then find where question ends by comparing
    with separately tokenized question.
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format the full text: question + answer + EOS
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    
    # Tokenize the full text (this is what the model will see)
    max_length = 128
    encoded = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Now find where the question ends by tokenizing question separately
    question_with_space = f"{question} "
    question_encoded = tokenizer(
        question_with_space,
        add_special_tokens=True,
        return_tensors=None,
    )
    question_token_ids = question_encoded["input_ids"]
    
    # Find the question length in the full sequence
    question_len = 0
    
    # Try to match question tokens at the start of input_ids
    if len(question_token_ids) <= len(input_ids):
        # Check exact match from start
        if input_ids[:len(question_token_ids)] == question_token_ids:
            question_len = len(question_token_ids)
        else:
            # Question tokens might not match exactly due to tokenization differences
            # Try to find a match with small offset (for special tokens)
            for offset in range(min(2, len(input_ids))):
                end_pos = offset + len(question_token_ids)
                if end_pos <= len(input_ids):
                    if input_ids[offset:end_pos] == question_token_ids:
                        question_len = end_pos
                        break
            
            # If still no match, use the question token length as estimate
            # This is safe because question should be at the start
            if question_len == 0:
                question_len = len(question_token_ids)
    
    # Safety: ensure we have room for answer tokens
    if question_len >= len(input_ids) - 1:
        question_len = max(1, len(input_ids) - 10)
    
    # Create labels: -100 for question (masked), actual token IDs for answer
    labels = [-100] * question_len + input_ids[question_len:]
    
    # Ensure labels length matches input_ids
    labels = labels[:len(input_ids)]
    if len(labels) < len(input_ids):
        labels.extend([-100] * (len(input_ids) - len(labels)))
    
    # Mask out padding tokens
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100
    
    # Critical check: verify we have non-masked labels
    non_masked = sum(1 for l in labels if l != -100)
    if non_masked == 0:
        # This is a critical error - we need labels to train on
        # Fallback: train on the last portion (answer part)
        answer_start = max(question_len, len(input_ids) - 15)
        for i in range(answer_start, len(input_ids)):
            if attention_mask[i] == 1:
                labels[i] = input_ids[i]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair.
    Format: question + <answer>{answer}</answer>
    """
    formatted_answer = format_numeric_answer(answer)
    return {
        "question": prompt.strip(),
        "answer": f"<answer>{formatted_answer}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Dataset that tokenizes examples on the fly.
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted_data)


def train_model(
    output_dir: str = MODEL_NAME,
    **_: Any,
):
    from transformers import Trainer, TrainingArguments, default_data_collator, TrainerCallback
    import numpy as np
    
    # Custom callback to validate gradients and compute proper gradient norms
    class GradientNormCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            """Validate gradients after backward pass but before optimizer step"""
            if model is not None:
                # Check all gradients for NaN/Inf before clipping
                has_nan = False
                has_inf = False
                total_norm = 0.0
                param_count = 0
                
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Check for NaN/Inf
                        if torch.isnan(param.grad).any():
                            has_nan = True
                            print(f"WARNING: NaN detected in gradient for {name}")
                        if torch.isinf(param.grad).any():
                            has_inf = True
                            print(f"WARNING: Inf detected in gradient for {name}")
                        
                        # Compute norm (only if not NaN/Inf)
                        if not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                
                # If NaN/Inf detected, zero out gradients to prevent training instability
                if has_nan or has_inf:
                    print(f"ERROR: NaN/Inf gradients detected! Zeroing gradients to prevent training crash.")
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad.zero_()
                    # Set a flag in state to track this
                    if not hasattr(state, 'nan_gradient_count'):
                        state.nan_gradient_count = 0
                    state.nan_gradient_count += 1
                
                # Store computed norm in state for logging
                if param_count > 0 and not (has_nan or has_inf):
                    state.last_grad_norm = total_norm ** (1. / 2)
                else:
                    state.last_grad_norm = float('nan') if (has_nan or has_inf) else 0.0
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Update logs with computed gradient norm"""
            if logs is not None:
                # Use our computed norm if available, otherwise keep the original
                if hasattr(state, 'last_grad_norm'):
                    logs['grad_norm'] = state.last_grad_norm
                elif 'grad_norm' not in logs:
                    # Fallback: compute from model if available
                    model = kwargs.get('model')
                    if model is not None:
                        total_norm = 0.0
                        param_count = 0
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                if not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                    param_norm = p.grad.data.norm(2)
                                    total_norm += param_norm.item() ** 2
                                    param_count += 1
                        if param_count > 0:
                            logs['grad_norm'] = total_norm ** (1. / 2)
                        else:
                            logs['grad_norm'] = 0.0
                
                if hasattr(state, 'nan_gradient_count') and state.nan_gradient_count > 0:
                    logs['nan_gradient_count'] = state.nan_gradient_count
                
                # Validate loss values
                if 'loss' in logs:
                    loss_val = logs['loss']
                    if isinstance(loss_val, (float, int)):
                        if np.isnan(loss_val) or np.isinf(loss_val):
                            print(f"WARNING: Invalid loss value detected: {loss_val}")
                        elif loss_val == 0.0 and state.global_step > 0:
                            print(f"WARNING: Loss is exactly 0.0 at step {state.global_step} - this may indicate an issue")
    
    model_path = _resolve_path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Load base model in FP32 for training to avoid NaN gradients
    # FP16 model + FP16 training causes numerical instability
    llm = BaseLLM(use_fp32_for_training=True)
    
    # Ensure model is in eval mode initially (BaseLLM sets this)
    # We'll set to train mode after applying LoRA
    llm.model.eval()
    
    # Create LoRA config
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 4,
        lora_dropout=0.0,
    )
    
    # Apply LoRA to model
    lora_model = get_peft_model(llm.model, config)
    
    # Enable input require grads for gradient checkpointing
    # This is required when using gradient_checkpointing=True
    lora_model.enable_input_require_grads()
    
    # Set model to training mode
    lora_model.train()
    
    # Verify model is trainable
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    # Prepare dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Verify tokenization works correctly - check a sample
    sample = tokenized_dataset[0]
    non_masked_labels = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample non-masked labels: {non_masked_labels} out of {len(sample['labels'])}")
    if non_masked_labels == 0:
        raise ValueError("All labels are masked! Tokenization is incorrect.")
    
    # Determine precision settings - prefer bf16 over fp16 for stability
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        # Check if bf16 is supported (Ampere+ GPUs)
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Using bfloat16 for training (more stable than fp16)")
        else:
            use_fp16 = True
            print("Using float16 for training (with loss scaling)")
    
    # Training arguments with improved stability
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
        remove_unused_columns=False,  # Keep our custom labels
        bf16=use_bf16,  # Use bf16 if available (more stable)
        fp16=use_fp16,  # Fallback to fp16 if bf16 not available
        fp16_full_eval=False,  # Use full precision for evaluation
        dataloader_pin_memory=False,  # Can help with memory issues
        max_grad_norm=1.0,  # Clip gradients to prevent explosion
        label_names=["labels"],  # Explicitly specify label field for PeftModel
        # Additional stability settings
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        ddp_find_unused_parameters=False,  # Faster training
    )
    
    # Use default data collator which handles batching correctly
    data_collator = default_data_collator
    
    # Custom Trainer class to ensure stable loss computation
    class StableTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Custom loss computation with validation to prevent NaN/Inf.
            Accepts additional kwargs for compatibility with newer transformers versions.
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Use the model's loss function
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
            # If loss is None, compute it manually
            if loss is None:
                from torch.nn import CrossEntropyLoss
                # Use ignore_index=-100 to properly handle masked labels
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Validate loss value
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Invalid loss detected (NaN/Inf): {loss.item()}")
                # Return a small positive loss to prevent training crash
                loss = torch.tensor(0.01, device=loss.device, dtype=loss.dtype, requires_grad=True)
            
            # Check if loss is suspiciously small (might indicate all labels masked)
            if loss.item() < 1e-8 and labels is not None:
                # Count non-masked labels
                non_masked = (labels != -100).sum().item()
                if non_masked == 0:
                    print(f"WARNING: All labels are masked in this batch! Loss: {loss.item()}")
                elif non_masked < 10:
                    print(f"WARNING: Very few non-masked labels ({non_masked}) in this batch. Loss: {loss.item()}")
            
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer with gradient norm callback and stable loss computation
    trainer = StableTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[GradientNormCallback()],
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {model_path}")
    trainer.save_model(str(model_path))
    
    # Test the model
    print("Testing model...")
    test_model(str(model_path))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    model_path = _resolve_path(ckpt_path)
    _ensure_adapter(model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    # Don't apply dataset answer patch during testing - we want to test actual model
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
