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
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the full text (question + space + answer + eos)
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128, return_tensors=None)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    
    # Find where the question ends by matching tokens
    # Tokenize question with space to match the format in full_text
    question_with_space = f"{question} "
    
    # Try with special tokens first (to match full tokenization)
    # Use same max_length and truncation to handle long questions
    question_tokens_with_special = tokenizer(
        question_with_space, 
        add_special_tokens=True, 
        max_length=128,
        truncation=True,
        return_tensors=None
    )["input_ids"]
    question_tokens_no_special = tokenizer(
        question_with_space, 
        add_special_tokens=False,
        max_length=128,
        truncation=True,
        return_tensors=None
    )["input_ids"]
    
    # Find where question tokens end in the full sequence
    answer_start_idx = len(input_ids)  # Default: no answer (all masked)
    
    # First try matching with special tokens (most common case)
    if len(question_tokens_with_special) <= len(input_ids):
        # Check if tokens match from the start
        if all(
            i < len(input_ids) and question_tokens_with_special[i] == input_ids[i]
            for i in range(len(question_tokens_with_special))
        ):
            answer_start_idx = len(question_tokens_with_special)
    
    # If that didn't work, try matching without special tokens
    # (accounting for potential BOS token at position 0)
    if answer_start_idx == len(input_ids) and len(question_tokens_no_special) <= len(input_ids):
        # Try matching from position 0
        if len(question_tokens_no_special) <= len(input_ids):
            if all(
                i < len(input_ids) and question_tokens_no_special[i] == input_ids[i]
                for i in range(len(question_tokens_no_special))
            ):
                answer_start_idx = len(question_tokens_no_special)
            # Try matching from position 1 (if there's a BOS token)
            elif len(question_tokens_no_special) < len(input_ids):
                if all(
                    i < len(input_ids) and question_tokens_no_special[i] == input_ids[i + 1]
                    for i in range(len(question_tokens_no_special))
                ):
                    answer_start_idx = len(question_tokens_no_special) + 1
    
    # Fallback: if no match found and question seems short, use length-based heuristic
    # This handles edge cases where tokenization might differ slightly
    if answer_start_idx == len(input_ids):
        # Estimate question length (rough heuristic)
        # Most questions should be shorter than the sequence
        estimated_q_len = min(len(question_tokens_no_special), len(input_ids) - 10)
        if estimated_q_len > 0 and estimated_q_len < len(input_ids):
            # Use estimated length as fallback (better than masking everything)
            answer_start_idx = estimated_q_len
    
    # Create labels: mask out the question part, keep the answer part
    labels = [-100] * len(input_ids)
    
    # Set labels for answer tokens (everything after question)
    for i in range(answer_start_idx, len(input_ids)):
        if attention_mask[i] == 1:  # Only set labels for non-padded tokens
            labels[i] = input_ids[i]
        else:
            labels[i] = -100
    
    # Ensure all padded tokens are masked
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


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
    if torch.cuda.is_available():
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
        label_names=["labels"],  # Explicitly specify label column name
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
    apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
