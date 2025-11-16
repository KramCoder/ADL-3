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
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    # Set up tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize question and answer separately to have exact control over labels
    question_text = f"{question} "
    answer_text = f"{answer}{tokenizer.eos_token}"
    
    # Tokenize question (with special tokens)
    question_encoded = tokenizer(question_text, add_special_tokens=True, return_attention_mask=False)
    question_ids = question_encoded["input_ids"]
    
    # Tokenize answer (without special tokens since they're already in question)
    answer_encoded = tokenizer(answer_text, add_special_tokens=False, return_attention_mask=False)
    answer_ids = answer_encoded["input_ids"]
    
    # Combine: question + answer
    input_ids = question_ids + answer_ids
    
    # Truncate if too long (keep at least some answer tokens)
    max_length = 128
    question_len = len(question_ids)
    if len(input_ids) > max_length:
        # Make sure we keep at least some answer tokens
        # Keep at least 10 tokens for the answer if possible
        min_answer_tokens = min(10, len(answer_ids))
        max_question_len = max_length - min_answer_tokens
        if question_len > max_question_len:
            question_len = max_question_len
            question_ids = question_ids[:max_question_len]
        remaining = max_length - question_len
        answer_ids = answer_ids[:remaining]
        input_ids = question_ids + answer_ids
    
    # Create labels: mask question, keep answer
    # Labels should be the same length as input_ids, with -100 for question tokens
    labels = [-100] * question_len + answer_ids
    
    # Pad to max_length
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
        labels = labels + [-100] * padding_length
    else:
        attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    formatted_answer = format_numeric_answer(answer)
    return {
        "question": prompt.strip(),
        "answer": f"<answer>{formatted_answer}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = MODEL_NAME,
    **_: Any,
):
    from transformers import Trainer, TrainingArguments
    
    model_path = _resolve_path(output_dir)
    
    # Load base model and create LoRA adapter
    llm = BaseLLM()
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=max(DEFAULT_LORA_RANK * 4, 4),
        lora_dropout=0.0,
    )
    
    lora_model = get_peft_model(llm.model, config)
    
    # Enable input require grads for gradient checkpointing
    # This is essential for training to work properly with gradient checkpointing
    lora_model.enable_input_require_grads()
    
    # Ensure model is in training mode
    lora_model.train()
    
    # Print model info for debugging
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Prepare dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Training arguments
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
        warmup_steps=0,  # No warmup for simplicity
        lr_scheduler_type="linear",  # Linear learning rate decay
        fp16=False,  # Disable fp16 to avoid potential NaN issues
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Test the model
    test_model(str(model_path))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    model_path = _resolve_path(ckpt_path)
    _ensure_adapter(model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    # Don't apply dataset answer patch - we want to test the actual model performance
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
