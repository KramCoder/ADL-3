from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from .base_llm import BaseLLM
from .conversion_utils import format_numeric_answer
from .data import Dataset, benchmark
from .datagen import generate_dataset as generate_rft_dataset


MODEL_NAME = "sft_model"
DEFAULT_LORA_RANK = 16  # Increased from 4 to 16 for better capacity
SFT_BASE_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RFT_DATA_PATH = Path(__file__).parent.parent / "data" / "rft.json"
RFT_DEFAULT_OVERSAMPLE = 15
RFT_DEFAULT_TEMPERATURE = 0.7


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent / candidate


def _resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def _ensure_rft_dataset(
    path: str | Path = RFT_DATA_PATH,
    *,
    force_regenerate: bool = False,
    oversample: int = RFT_DEFAULT_OVERSAMPLE,
    temperature: float = RFT_DEFAULT_TEMPERATURE,
) -> Path:
    resolved_path = _resolve_data_path(path)
    needs_regeneration = force_regenerate or not resolved_path.exists()

    if needs_regeneration:
        action = "Regenerating" if resolved_path.exists() else "Generating"
        print(
            f"{action} RFT dataset at {resolved_path} "
            f"(oversample={oversample}, temperature={temperature})"
        )
        generate_rft_dataset(
            output_json=resolved_path,
            oversample=oversample,
            temperature=temperature,
        )
    else:
        print(f"Using existing RFT dataset at {resolved_path}")

    return resolved_path


def _ensure_adapter(model_path: Path, *, rank: int = DEFAULT_LORA_RANK) -> None:
    # Check for both .bin and .safetensors formats (newer versions use safetensors)
    adapter_bin = model_path / "adapter_model.bin"
    adapter_safetensors = model_path / "adapter_model.safetensors"
    adapter_config = model_path / "adapter_config.json"
    
    # If any adapter file exists, assume the adapter is already created
    if adapter_bin.exists() or adapter_safetensors.exists() or adapter_config.exists():
        return

    model_path.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM(checkpoint=SFT_BASE_CHECKPOINT)
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=rank,
        lora_alpha=rank * 2,  # Match training config
        lora_dropout=0.05,  # Match training config
    )

    lora_model = get_peft_model(llm.model, config)
    lora_model.save_pretrained(model_path)


def load() -> BaseLLM:
    model_path = _resolve_path(MODEL_NAME)
    _ensure_adapter(model_path)

    llm = BaseLLM(checkpoint=SFT_BASE_CHECKPOINT)
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    # DO NOT apply dataset answer patch - we want the actual trained model!
    # apply_dataset_answer_patch(llm)

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We concatenate question and answer, then create labels where only the answer
    tokens are supervised (question tokens are masked with -100).
    
    Simplified strategy: Tokenize question and answer separately, then concatenate.
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_length = 128
    
    # Tokenize question (with space after) - we'll mask these tokens
    question_text = f"{question} "
    question_encoded = tokenizer(
        question_text,
        add_special_tokens=True,
        truncation=False,
        return_tensors=None,
    )
    question_ids = question_encoded["input_ids"]
    
    # Tokenize answer (with EOS) - these are the tokens we train on
    answer_text = f"{answer}{tokenizer.eos_token}"
    answer_encoded = tokenizer(
        answer_text,
        add_special_tokens=False,  # Don't add special tokens again
        truncation=False,
        return_tensors=None,
    )
    answer_ids = answer_encoded["input_ids"]
    
    # Concatenate question + answer
    input_ids = question_ids + answer_ids
    
    # Truncate if necessary
    if len(input_ids) > max_length:
        # Keep as much of the question as possible, but ensure answer fits
        min_answer_tokens = min(len(answer_ids), 20)  # Reserve at least 20 tokens for answer
        question_max = max_length - min_answer_tokens
        question_ids = question_ids[:question_max]
        answer_ids = answer_ids[:(max_length - len(question_ids))]
        input_ids = question_ids + answer_ids
    
    question_len = len(question_ids)
    
    # Create attention mask (all 1s for non-padding)
    attention_mask = [1] * len(input_ids)
    
    # Create labels: -100 for question tokens, actual IDs for answer tokens
    labels = [-100] * question_len + answer_ids
    
    # Pad to max_length
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
    
    # Verify we have non-masked labels
    non_masked = sum(1 for l in labels if l != -100)
    if non_masked == 0:
        # This should not happen with the new logic, but keep as safety check
        # Train on last 10 tokens
        for i in range(max(0, len(input_ids) - 10), len(input_ids)):
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


class RFTDataset:
    """Lightweight loader for the offline RFT reasoning dataset."""

    def __init__(self, json_path: str | Path = RFT_DATA_PATH):
        self.path = _resolve_data_path(json_path)
        if not self.path.exists():
            raise FileNotFoundError(f"RFT dataset missing at {self.path}")

        with self.path.open("r", encoding="utf-8") as handle:
            raw_records = json.load(handle)

        if not isinstance(raw_records, list):
            raise ValueError(f"Expected a list of records in {self.path}")

        self.records: list[tuple[str, float | None, str]] = []
        for idx, entry in enumerate(raw_records):
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue

            question, answer_value, reasoning = entry[:3]
            if not isinstance(question, str) or not isinstance(reasoning, str):
                continue

            try:
                numeric_answer = float(answer_value)
            except (TypeError, ValueError):
                numeric_answer = None
            else:
                if numeric_answer != numeric_answer or abs(numeric_answer) == float("inf"):
                    numeric_answer = None

            self.records.append((question, numeric_answer, reasoning))

        if not self.records:
            raise ValueError(f"No valid RFT records found in {self.path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


def format_rft_example(question: str, correct_answer: float | None, reasoning: str) -> dict[str, str]:
    """Format question + reasoning pair for SFT fine-tuning."""

    formatted_question = question.strip()
    reasoning_text = reasoning.strip()

    has_answer_tags = "<answer>" in reasoning_text and "</answer>" in reasoning_text
    if not has_answer_tags:
        fallback_value = 0.0 if correct_answer is None else correct_answer
        formatted_answer = format_numeric_answer(fallback_value)
        suffix = f"<answer>{formatted_answer}</answer>"
        reasoning_text = f"{reasoning_text}\n{suffix}" if reasoning_text else suffix

    return {
        "question": formatted_question,
        "answer": reasoning_text,
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Any, format_fn):
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
    max_steps: int | None = None,
    rft_data_path: str | Path = RFT_DATA_PATH,
    force_regenerate_rft: bool = False,
    rft_oversample: int = RFT_DEFAULT_OVERSAMPLE,
    rft_temperature: float = RFT_DEFAULT_TEMPERATURE,
    **_: Any,
):
    from transformers import Trainer, TrainingArguments, default_data_collator, TrainerCallback
    import numpy as np
    
    # Custom callback to validate gradients and compute proper gradient norms
    class GradientNormCallback(TrainerCallback):
        def on_backward(self, args, state, control, model=None, **kwargs):
            """Compute gradient norm right after backward, before optimizer step"""
            if model is not None:
                # Compute gradient norm before clipping/optimizer step
                total_norm = 0.0
                param_count = 0
                has_nan = False
                has_inf = False
                
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
                    state.last_grad_norm = 0.0
                elif param_count > 0:
                    # Store computed norm in state for logging
                    state.last_grad_norm = total_norm ** (1. / 2)
                else:
                    # No valid gradients found
                    state.last_grad_norm = 0.0
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Update logs with computed gradient norm"""
            if logs is not None:
                # Use our computed norm if available, otherwise use Trainer's built-in
                if hasattr(state, 'last_grad_norm'):
                    # Only update if we have a valid value
                    if not (np.isnan(state.last_grad_norm) or np.isinf(state.last_grad_norm)):
                        logs['grad_norm'] = state.last_grad_norm
                    else:
                        # If we had NaN/Inf, log 0.0
                        logs['grad_norm'] = 0.0
                # If grad_norm is not in logs and we don't have a stored value, 
                # the Trainer will compute it automatically, so we don't need to override
                
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
    llm = BaseLLM(checkpoint=SFT_BASE_CHECKPOINT, use_fp32_for_training=True)
    
    # Ensure model is in eval mode initially (BaseLLM sets this)
    # We'll set to train mode after applying LoRA
    llm.model.eval()
    
    # Create LoRA config
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 2,  # Reduced from 4x to 2x for better training
        lora_dropout=0.05,  # Add small dropout for regularization
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
    rft_path = _ensure_rft_dataset(
        path=rft_data_path,
        force_regenerate=force_regenerate_rft,
        oversample=rft_oversample,
        temperature=rft_temperature,
    )
    reasoning_dataset = RFTDataset(rft_path)
    print(f"Loaded {len(reasoning_dataset)} RFT reasoning pairs from {rft_path}")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, reasoning_dataset, format_rft_example)
    
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
    
    # Training arguments with improved stability and better accuracy
    training_args_dict = {
        "output_dir": str(model_path),
        "logging_dir": str(model_path),
        "report_to": "none",  # Disable TensorBoard to save space
        "gradient_checkpointing": True,
        "learning_rate": 5e-4,  # Increased from 2e-4 for faster convergence
        "per_device_train_batch_size": 16,  # Reduced from 32 for better gradients
        "gradient_accumulation_steps": 2,  # Accumulate to effective batch size of 32
        "logging_steps": 10,
        "save_total_limit": 0,  # Don't save checkpoints to save space
        "save_strategy": "no",  # Don't save intermediate checkpoints
        "remove_unused_columns": False,  # Keep our custom labels
        "bf16": use_bf16,  # Use bf16 if available (more stable)
        "fp16": use_fp16,  # Fallback to fp16 if bf16 not available
        "fp16_full_eval": False,  # Use full precision for evaluation
        "dataloader_pin_memory": False,  # Can help with memory issues
        "max_grad_norm": 1.0,  # Clip gradients to prevent explosion
        "label_names": ["labels"],  # Explicitly specify label field for PeftModel
        # Learning rate scheduler settings
        "lr_scheduler_type": "cosine",  # Use cosine decay instead of linear
        "warmup_ratio": 0.1,  # Increased warmup for better stability
        "warmup_steps": 0,  # Will be computed from warmup_ratio
        # Additional stability settings
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues
        "ddp_find_unused_parameters": False,  # Faster training
        "weight_decay": 0.01,  # Add weight decay for regularization
    }
    
    # Set epochs or max_steps (not both)
    if max_steps is not None:
        training_args_dict["max_steps"] = max_steps
    else:
        training_args_dict["num_train_epochs"] = 6  # Increased from 5 to 6 epochs for better accuracy
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Use default data collator which handles batching correctly
    data_collator = default_data_collator
    
    # Custom Trainer class to ensure stable loss computation
    class StableTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Custom loss computation with validation to prevent NaN/Inf.
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
    train_result = trainer.train()
    
    # Print training metrics summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    if hasattr(train_result, 'metrics'):
        for key, value in train_result.metrics.items():
            if isinstance(value, (int, float)):
                if key == 'train_loss':
                    print(f"Final Loss: {value:.6f}")
                elif key == 'train_grad_norm':
                    print(f"Final Gradient Norm: {value:.6f}")
                elif 'learning_rate' in key:
                    print(f"Learning Rate: {value:.2e}")
                else:
                    print(f"{key}: {value}")
    
    # Save the final model
    print(f"\nSaving model to {model_path}")
    trainer.save_model(str(model_path))
    
    # Test the model
    print("Testing model...")
    test_model(str(model_path))


def test_model(ckpt_path: str = MODEL_NAME):
    testset = Dataset("valid")
    model_path = _resolve_path(ckpt_path)
    _ensure_adapter(model_path)

    llm = BaseLLM(checkpoint=SFT_BASE_CHECKPOINT)
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    # DO NOT apply dataset answer patch - we want to test the actual trained model
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
