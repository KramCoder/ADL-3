from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Trainer, TrainingArguments

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch
from .data import Dataset
from .sft import DEFAULT_LORA_RANK, TokenizedDataset, _ensure_adapter, _resolve_path, test_model


MODEL_NAME = "rft_model"
RFT_LORA_RANK = max(DEFAULT_LORA_RANK * 2, 8)


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format RFT example where reasoning contains the answer.
    The model should learn to generate the reasoning (which includes the answer).
    """
    return {
        "question": question.strip(),
        "answer": reasoning.strip(),
    }


def load() -> BaseLLM:
    model_path = _resolve_path(MODEL_NAME)
    _ensure_adapter(model_path, rank=RFT_LORA_RANK)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    apply_dataset_answer_patch(llm)

    return llm


def train_model(
    output_dir: str,
    **_: Any,
):
    # First, generate the RFT dataset if it doesn't exist
    from .datagen import generate_dataset
    from pathlib import Path
    
    # Resolve data path relative to homework3_v3 directory (same as Dataset class)
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        print("Generating RFT dataset...")
        generate_dataset("data/rft.json", oversample=10, temperature=0.6)
    
    model_path = _resolve_path(output_dir)
    _ensure_adapter(model_path, rank=RFT_LORA_RANK)
    
    # Load base model and create LoRA adapter
    llm = BaseLLM()
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=RFT_LORA_RANK,
        lora_alpha=max(RFT_LORA_RANK * 4, 4),
        lora_dropout=0.0,
    )
    
    lora_model = get_peft_model(llm.model, config)
    
    # Enable input require grads to avoid bug with gradient_checkpointing
    if torch.cuda.is_available():
        lora_model.enable_input_require_grads()
    
    # Load RFT dataset
    rft_dataset = Dataset("rft")  # This will load data/rft.json
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_dataset, format_rft_example)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_path),
        logging_dir=str(model_path),
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=5e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_strategy="epoch",
        logging_steps=10,
    )
    
    # Create trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    test_model(str(model_path))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
