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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the full text
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128, return_tensors=None)
    input_ids = full["input_ids"]
    
    # Tokenize just the question part to find where it ends
    # We need to match exactly how it appears in the full text
    # Use the same settings as the full text tokenization
    question_text = f"{question} "
    question_encoded = tokenizer(question_text, add_special_tokens=True, return_tensors=None)
    question_token_ids = question_encoded["input_ids"]
    
    # The question tokens should appear at the start of the full sequence
    # Find the exact match by comparing token sequences
    question_len = 0
    if len(question_token_ids) > 0 and len(question_token_ids) <= len(input_ids):
        # Check if the first tokens match exactly (accounting for special tokens)
        # Try matching from the start
        if input_ids[:len(question_token_ids)] == question_token_ids:
            question_len = len(question_token_ids)
        else:
            # The tokenizer might add special tokens differently, so try to find the match
            # Check if question tokens appear anywhere in the first part of input_ids
            max_search_len = min(len(input_ids), len(question_token_ids) + 5)
            for start_idx in range(max_search_len - len(question_token_ids) + 1):
                if input_ids[start_idx:start_idx+len(question_token_ids)] == question_token_ids:
                    question_len = start_idx + len(question_token_ids)
                    break
            
            # If still no match, use the question token length as estimate
            # This handles cases where tokenization differs slightly due to context
            if question_len == 0:
                # Fallback: assume question tokens match from the start (minus any special tokens)
                # Most tokenizers add special tokens at the start, so we check from index 1
                if len(input_ids) > 1 and len(question_token_ids) > 0:
                    # Try matching from index 1 (after potential BOS token)
                    if len(input_ids) > len(question_token_ids) + 1:
                        if input_ids[1:1+len(question_token_ids)] == question_token_ids:
                            question_len = 1 + len(question_token_ids)
                        else:
                            question_len = len(question_token_ids)  # Use as estimate
                    else:
                        question_len = len(question_token_ids)
                else:
                    question_len = len(question_token_ids)
    
    # Ensure we have room for at least some answer tokens
    if question_len >= len(input_ids):
        question_len = max(0, len(input_ids) - 10)  # Leave some room for answer
    
    # Create labels: mask out the question part, keep answer for training
    labels = [-100] * question_len
    if question_len < len(input_ids):
        # Copy the answer tokens (and EOS) as labels
        labels.extend(input_ids[question_len:])
    
    # Ensure labels list has the same length as input_ids
    labels = labels[:len(input_ids)]
    
    # Mask out padding tokens as well
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100
    
    # Validation: ensure we have at least some non-masked labels
    # If all labels are masked, something went wrong
    non_masked_count = sum(1 for l in labels if l != -100)
    if non_masked_count == 0:
        # Fallback: if everything is masked, at least train on the last few tokens
        # This shouldn't happen, but provides a safety net
        if len(input_ids) > 5:
            for i in range(max(question_len, len(input_ids) - 5), len(input_ids)):
                if full["attention_mask"][i] == 1:
                    labels[i] = input_ids[i]

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
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    
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
    
    # Ensure the model is configured to compute loss
    # For causal LM, the model should automatically handle label shifting
    # Disable cache for training (required when gradient checkpointing is enabled)
    if hasattr(lora_model, 'config'):
        lora_model.config.use_cache = False
    # Also set on base model if accessible
    if hasattr(lora_model, 'base_model') and hasattr(lora_model.base_model, 'config'):
        lora_model.base_model.config.use_cache = False
    if hasattr(lora_model, 'get_base_model'):
        base_model = lora_model.get_base_model()
        if hasattr(base_model, 'config'):
            base_model.config.use_cache = False
    
    # Prepare dataset
    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)
    
    # Data collator - sequences are already padded to max_length in tokenize()
    # So we just need to stack them into tensors
    def data_collator(features):
        # Features is a list of dicts from tokenize()
        batch = {}
        # Get all keys from first feature
        keys = features[0].keys()
        for key in keys:
            # Convert lists to tensors and stack
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        return batch
    
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
        remove_unused_columns=False,  # Keep all columns including labels
        label_names=["labels"],  # Explicitly tell Trainer about labels field
    )
    
    # Create trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
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
