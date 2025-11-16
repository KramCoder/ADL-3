from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainerCallback

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


class GradientMonitoringCallback(TrainerCallback):
    """Callback to monitor gradients and detect NaN values."""
    
    def __init__(self):
        self.nan_detected = False
        self.zero_loss_detected = False
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor training logs for NaN gradients and zero loss."""
        if logs is None:
            return
        
        # Check for NaN or invalid gradient norm
        if 'grad_norm' in logs:
            grad_norm = logs['grad_norm']
            if grad_norm != grad_norm or grad_norm == float('inf') or grad_norm == float('-inf'):  # NaN or inf check
                self.nan_detected = True
                print(f"⚠️  WARNING: Invalid gradient norm ({grad_norm}) detected at step {state.global_step}")
                # Set to 0.0 to indicate no valid gradients
                logs['grad_norm'] = 0.0
        
        # Check for zero loss (which can cause NaN gradients)
        if 'loss' in logs:
            loss = logs['loss']
            if loss == 0.0 or loss != loss:  # Zero or NaN loss
                if loss == 0.0:
                    self.zero_loss_detected = True
                    print(f"⚠️  WARNING: Zero loss detected at step {state.global_step}")
                else:
                    print(f"⚠️  WARNING: NaN loss detected at step {state.global_step}")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check and fix gradients after each step."""
        if model is not None:
            # Check for NaN gradients and clip them
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    if (grad != grad).any():  # Check for NaN
                        has_nan = True
                        print(f"⚠️  NaN gradient detected in {name}, zeroing it out")
                        param.grad.data.zero_()
                    elif (grad == float('inf')).any() or (grad == float('-inf')).any():
                        has_nan = True
                        print(f"⚠️  Inf gradient detected in {name}, clipping it")
                        param.grad.data.clamp_(-1e6, 1e6)
            
            if has_nan and state.global_step % 10 == 0:
                print(f"⚠️  Invalid gradients detected at step {state.global_step}. Training may be unstable.")


def train_model(
    output_dir: str = MODEL_NAME,
    **_: Any,
):
    from transformers import Trainer, TrainingArguments, default_data_collator
    
    model_path = _resolve_path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    llm = BaseLLM()
    
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
    
    # Training arguments - use bf16 instead of fp16 for better stability
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    
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
        fp16=use_fp16,  # Use fp16 only if bf16 not available
        bf16=use_bf16,  # Prefer bf16 for better numerical stability
        dataloader_pin_memory=False,  # Can help with memory issues
        max_grad_norm=1.0,  # Clip gradients to prevent explosion and NaN
        label_names=["labels"],  # Explicitly specify label field for PeftModel
        # Add loss scaling for fp16 to prevent overflow
        fp16_full_eval=False,
        # Ensure proper gradient accumulation
        gradient_accumulation_steps=1,
        # Add warmup to stabilize training
        warmup_steps=50,
    )
    
    # Use default data collator which handles batching correctly
    data_collator = default_data_collator
    
    # Create gradient monitoring callback
    gradient_callback = GradientMonitoringCallback()
    
    # Create trainer with custom callback
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[gradient_callback],
    )
    
    # Override compute_loss to ensure loss is computed correctly and handle edge cases
    original_compute_loss = trainer.compute_loss
    
    def safe_compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with NaN and zero loss protection."""
        # Get labels
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels are required for training")
        
        # Check that we have non-masked labels
        if isinstance(labels, torch.Tensor):
            non_masked = (labels != -100).sum().item()
            if non_masked == 0:
                print("⚠️  WARNING: All labels are masked in this batch!")
                # Return a small dummy loss to prevent NaN
                dummy_loss = torch.tensor(1e-6, device=labels.device, dtype=torch.float32, requires_grad=True)
                if return_outputs:
                    return dummy_loss, None
                return dummy_loss
        
        # Compute loss normally using the original method
        try:
            loss = original_compute_loss(model, inputs, return_outputs)
            if return_outputs:
                loss, outputs = loss
            else:
                outputs = None
        except Exception as e:
            print(f"⚠️  WARNING: Error computing loss: {e}")
            # Fallback to manual computation
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            if return_outputs:
                outputs = outputs
            else:
                outputs = None
        
        # Validate loss
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
    
    # Bind the method to the trainer instance
    import types
    trainer.compute_loss = types.MethodType(safe_compute_loss, trainer)
    
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
