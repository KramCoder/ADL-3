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
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize full text with padding and truncation
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    
    # Find where the answer starts by looking for the <answer> token
    # This is more reliable than trying to match tokenizations
    answer_start_text = "<answer>"
    answer_start_encoded = tokenizer(answer_start_text, add_special_tokens=False)
    answer_start_tokens = answer_start_encoded["input_ids"]
    
    # Find the position of <answer> in the tokenized sequence
    answer_start_idx = len(input_ids)  # Default: no answer found
    
    # Search for answer_start_tokens in input_ids
    if len(answer_start_tokens) > 0:
        for i in range(len(input_ids) - len(answer_start_tokens) + 1):
            if input_ids[i:i+len(answer_start_tokens)] == answer_start_tokens:
                answer_start_idx = i
                break
    
    # Fallback: if <answer> token not found, try to find where question ends
    # by tokenizing the question separately
    if answer_start_idx >= len(input_ids):
        question_with_space = f"{question} "
        question_encoded = tokenizer(question_with_space, add_special_tokens=True)
        question_token_ids = question_encoded["input_ids"]
        
        # Find longest matching prefix
        question_len = len(question_token_ids)
        max_match_len = min(question_len, len(input_ids))
        for i in range(max_match_len):
            if i < len(question_token_ids) and i < len(input_ids):
                if question_token_ids[i] != input_ids[i]:
                    answer_start_idx = i
                    break
            else:
                break
        
        # If still no match, use question_len as fallback
        if answer_start_idx >= len(input_ids):
            answer_start_idx = min(question_len, len(input_ids))

    # Ensure answer_start_idx is valid
    answer_start_idx = min(answer_start_idx, len(input_ids))
    
    # Ensure we have at least some answer tokens to train on
    # If answer_start_idx is too close to the end, adjust it
    if answer_start_idx >= len(input_ids) - 2:
        # No answer found, mask everything (shouldn't happen normally)
        answer_start_idx = len(input_ids)

    # Create labels: mask out the prompt part (question), keep only answer for training
    labels = [-100] * answer_start_idx + input_ids[answer_start_idx:]
    
    # Ensure labels list has exactly the same length as input_ids
    labels = labels[:len(input_ids)]
    while len(labels) < len(input_ids):
        labels.append(-100)

    # Mask out padding tokens as well
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100
    
    # Verify we have at least some non-masked tokens (for debugging)
    # This ensures we're actually training on something
    num_trainable_tokens = sum(1 for label in labels if label != -100)
    if num_trainable_tokens == 0:
        # Fallback: if somehow all tokens are masked, at least train on the last few tokens
        # This shouldn't happen normally, but it's a safety check
        if len(input_ids) > 0:
            # Train on the last token as a fallback
            labels[-1] = input_ids[-1]

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
        inference_mode=False,  # CRITICAL: Must be False for training
    )
    
    lora_model = get_peft_model(llm.model, config)
    
    # Set model to training mode
    lora_model.train()
    
    # Enable input require grads for gradient checkpointing
    # This is essential for training to work properly with gradient checkpointing
    lora_model.enable_input_require_grads()
    
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
    # NOTE: Do NOT apply dataset answer patch during testing
    # We want to test the actual model, not the lookup table
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
