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
    Tokenize a question/answer pair for causal LM training.
    We mask out the question tokens (set labels to -100) and only compute loss on the answer.
    """
    # Format: question + space + answer + EOS
    # According to README: simply ask the model to complete a question with <answer>{answer}</answer>
    full_text = f"{question} {answer}{tokenizer.eos_token}"
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the full text
    encoded = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors=None,
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Find where the answer starts by looking for "<answer>" tag
    # Tokenize the "<answer>" tag to find its token sequence
    answer_tag = "<answer>"
    tag_encoded = tokenizer(answer_tag, add_special_tokens=False, return_tensors=None)
    tag_token_ids = tag_encoded["input_ids"]
    
    # Create labels: -100 for question tokens, actual token ids for answer tokens
    labels = [-100] * len(input_ids)
    
    # Find where "<answer>" appears in the input_ids
    answer_start_idx = None
    if len(tag_token_ids) > 0:
        for i in range(len(input_ids) - len(tag_token_ids) + 1):
            if input_ids[i:i + len(tag_token_ids)] == tag_token_ids:
                # Found the answer tag, start labeling from here
                answer_start_idx = i
                break
    
    # If we found the answer tag, set labels from that point onward
    if answer_start_idx is not None:
        for i in range(answer_start_idx, len(input_ids)):
            if attention_mask[i] == 1:  # Only set labels for non-padding tokens
                labels[i] = input_ids[i]
    else:
        # Fallback: tokenize question separately to estimate where answer starts
        question_text = f"{question} "
        question_encoded = tokenizer(
            question_text,
            add_special_tokens=True,
            return_tensors=None,
        )
        question_token_ids = question_encoded["input_ids"]
        
        # Remove EOS if present
        if question_token_ids and question_token_ids[-1] == tokenizer.eos_token_id:
            question_token_ids = question_token_ids[:-1]
        
        # Try to find question tokens in input_ids
        for start_idx in range(min(3, len(input_ids))):
            end_idx = start_idx + len(question_token_ids)
            if end_idx <= len(input_ids):
                if input_ids[start_idx:end_idx] == question_token_ids:
                    # Set labels from end of question
                    for i in range(end_idx, len(input_ids)):
                        if attention_mask[i] == 1:
                            labels[i] = input_ids[i]
                    break
        else:
            # Last resort: use question length estimate
            answer_start = min(len(question_token_ids), len(input_ids) - 1)
            for i in range(answer_start, len(input_ids)):
                if attention_mask[i] == 1:
                    labels[i] = input_ids[i]
    
    # Ensure we have at least some non-masked labels for training
    non_masked_count = sum(1 for l in labels if l != -100)
    if non_masked_count == 0:
        # Emergency fallback: train on last few tokens
        for i in range(max(0, len(input_ids) - 10), len(input_ids)):
            if attention_mask[i] == 1:
                labels[i] = input_ids[i]
    
    encoded["labels"] = labels
    return encoded


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair.
    According to README: simply ask the model to complete a question with <answer>{answer}</answer>
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
        # Get question and answer from dataset
        question, answer = self.data[idx]
        # Format into question/answer dict
        formatted = self.format_fn(question, answer)
        # Tokenize and return
        return tokenize(self.tokenizer, formatted["question"], formatted["answer"])


def train_model(
    output_dir: str = MODEL_NAME,
    **_: Any,
):
    from transformers import Trainer, TrainingArguments
    
    model_path = _resolve_path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    llm = BaseLLM()
    
    # Create LoRA config as per README
    # - target_modules="all-linear"
    # - bias="none" and task_type="CAUSAL_LM"
    # - r rank such that overall model size stays below 20MB
    # - lora_alpha about 4-5 times the rank
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 4,
        lora_dropout=0.0,
    )
    
    # Convert to LoRA model
    lora_model = get_peft_model(llm.model, config)
    
    # Enable input require grads for gradient checkpointing (per README)
    lora_model.enable_input_require_grads()
    
    # Set to training mode
    lora_model.train()
    
    # Prepare dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Training arguments as per README
    # - gradient_checkpointing=True to save GPU memory
    # - reasonable learning_rate
    # - output_dir, logging_dir, report_to="tensorboard"
    # - per_device_train_batch_size=32
    # - num_train_epochs (shouldn't need more than 5)
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
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Use default data collator which preserves our custom labels
    from transformers import default_data_collator
    
    # Create trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save the final model to the specified directory
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
    # Do NOT apply dataset answer patch during testing - we want to test the actual model
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
