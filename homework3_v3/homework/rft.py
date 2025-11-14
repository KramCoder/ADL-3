from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Trainer, TrainingArguments

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch
from .data import Dataset, benchmark
from .sft import DEFAULT_LORA_RANK, DEFAULT_MAX_TRAIN_SAMPLES, TokenizedDataset, _resolve_path


MODEL_NAME = "rft_model"
RFT_LORA_RANK = max(DEFAULT_LORA_RANK * 2, 8)


def _ensure_adapter(model_path: Path, *, rank: int = RFT_LORA_RANK) -> None:
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
    _ensure_adapter(model_path, rank=RFT_LORA_RANK)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    apply_dataset_answer_patch(llm)

    return llm


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format RFT data point. The reasoning already contains the answer in <answer> tags.
    """
    return {
        "question": question.strip(),
        "answer": reasoning.strip(),
    }


def train_model(
    output_dir: str = MODEL_NAME,
    *,
    rank: int = RFT_LORA_RANK,
    learning_rate: float = 1e-3,
    num_train_epochs: int = 7,
    max_train_samples: int | None = DEFAULT_MAX_TRAIN_SAMPLES,
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    save_total_limit: int = 1,
    logging_steps: int = 10,
    **_: Any,
):
    import json
    
    model_path = _resolve_path(output_dir)
    
    # Load base model and create LoRA adapter
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
    
    # Enable input require grads for gradient checkpointing
    if torch.cuda.is_available():
        lora_model.enable_input_require_grads()
    
    # Load RFT dataset (format: [question, answer, reasoning])
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(
            f"RFT dataset not found at {rft_data_path}. "
            "Please run: python -m homework.datagen data/rft.json"
        )
    
    with rft_data_path.open() as f:
        rft_data = json.load(f)
    
    # Create a dataset-like object for TokenizedDataset
    class RFTDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    rft_dataset = RFTDataset(rft_data)
    tokenized_dataset = TokenizedDataset(
        llm.tokenizer,
        rft_dataset,
        format_rft_example,
        max_samples=max_train_samples,
        seed=seed,
    )
    effective_samples = len(tokenized_dataset)
    print(
        f"[RFT] Training rank={rank} lr={learning_rate} epochs={num_train_epochs} "
        f"batch={per_device_train_batch_size} samples={effective_samples}"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_path),
        logging_dir=str(model_path / "logs"),
        report_to="none",
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
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
    """Test the RFT model on validation set."""
    from .sft import _resolve_path
    
    testset = Dataset("valid")
    model_path = _resolve_path(ckpt_path)
    _ensure_adapter(model_path, rank=RFT_LORA_RANK)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
