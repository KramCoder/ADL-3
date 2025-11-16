from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch, format_numeric_answer
from .data import Dataset, benchmark


MODEL_NAME = "sft_model"
DEFAULT_LORA_RANK = 4
TARGET_EFFECTIVE_BATCH_SIZE = 32
ENV_PREFIX = "SFT"


def _parse_int_env(var_name: str) -> int | None:
    value = os.environ.get(var_name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _auto_micro_batch_size(target_batch_size: int) -> int:
    """Pick a reasonable per-device batch size based on available hardware."""

    if torch.cuda.is_available():
        try:
            total_mem_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
        except Exception:
            total_mem_gb = 0
        if total_mem_gb < 10:
            return min(2, target_batch_size)
        if total_mem_gb < 20:
            return min(4, target_batch_size)
        if total_mem_gb < 40:
            return min(8, target_batch_size)
        return min(16, target_batch_size)
    # CPU / MPS training: keep things modest to avoid long steps
    return min(8, target_batch_size)


def determine_batch_hparams(
    target_batch_size: int = TARGET_EFFECTIVE_BATCH_SIZE,
    *,
    per_device_override: int | None = None,
    grad_accum_override: int | None = None,
    env_prefix: str = ENV_PREFIX,
) -> tuple[int, int]:
    """
    Return (per_device_batch_size, gradient_accumulation_steps) tuned for stability.

    We keep the effective batch size close to ``target_batch_size`` while ensuring
    the per-device micro batch fits into memory to prevent fp16 overflows.
    """

    env_prefix = env_prefix.upper()
    env_micro = _parse_int_env(f"{env_prefix}_MICRO_BATCH_SIZE")
    env_grad = _parse_int_env(f"{env_prefix}_GRAD_ACCUM_STEPS")

    micro_batch = per_device_override or env_micro
    grad_steps = grad_accum_override or env_grad

    if micro_batch is None:
        micro_batch = _auto_micro_batch_size(target_batch_size)
    micro_batch = max(1, min(micro_batch, target_batch_size))

    if grad_steps is None:
        grad_steps = max(1, math.ceil(target_batch_size / micro_batch))

    return micro_batch, grad_steps


def determine_precision_flags(env_prefix: str = ENV_PREFIX) -> tuple[bool, bool]:
    """
    Decide whether to use fp16 or bf16 mixed precision.

    Returns a tuple ``(use_fp16, use_bf16)``. We prefer bf16 on modern GPUs
    because it is far more stable for this workload. Users can override the
    decision by setting ``{ENV_PREFIX}_PRECISION`` to ``fp16``, ``bf16``, or
    ``fp32``.
    """

    env_prefix = env_prefix.upper()
    forced = os.environ.get(f"{env_prefix}_PRECISION")
    if forced:
        forced = forced.lower()
        if forced == "fp32":
            return False, False
        if forced == "bf16":
            return False, torch.cuda.is_available()
        if forced == "fp16":
            return torch.cuda.is_available(), False

    if not torch.cuda.is_available():
        return False, False

    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        major = 0

    supports_bf16 = major >= 8  # Ampere or newer
    if supports_bf16:
        return False, True
    return True, False


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
    num_train_epochs: int = 3,
    max_steps: int | None = None,
    target_batch_size: int = TARGET_EFFECTIVE_BATCH_SIZE,
    per_device_batch_size: int | None = None,
    grad_accumulation_steps_override: int | None = None,
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
    
    per_device_batch, grad_accum_steps = determine_batch_hparams(
        target_batch_size,
        per_device_override=per_device_batch_size,
        grad_accum_override=grad_accumulation_steps_override,
    )
    fp16, bf16 = determine_precision_flags()
    effective_batch = per_device_batch * grad_accum_steps
    print(
        "Batch config:",
        f"per_device={per_device_batch}",
        f"grad_accum={grad_accum_steps}",
        f"effective≈{effective_batch}",
    )
    precision_name = "bf16" if bf16 else "fp16" if fp16 else "fp32"
    print(f"Precision mode: {precision_name}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_path),
        logging_dir=str(model_path),
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum_steps,
        save_strategy="epoch",
        logging_steps=max(1, 10 // grad_accum_steps),
        save_total_limit=1,
        remove_unused_columns=False,  # Keep our custom labels
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=False,  # Can help with memory issues
        max_grad_norm=1.0,  # Clip gradients to prevent explosion and NaN
        label_names=["labels"],  # Explicitly specify label field for PeftModel
        optim="adamw_torch",
    )
    
    # Use default data collator which handles batching correctly
    data_collator = default_data_collator
    
    # Create trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
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
