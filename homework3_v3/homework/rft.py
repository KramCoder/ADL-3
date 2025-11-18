from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Trainer, TrainingArguments

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch
from .data import Dataset, benchmark
from .sft import DEFAULT_LORA_RANK, TokenizedDataset, _resolve_path, tokenize


MODEL_NAME = "rft_model"
RFT_LORA_RANK = 16  # Balanced size: SFT=15MB + RFT=30MB = 45MB uncompressed (~32-35MB compressed)


def _ensure_adapter(model_path: Path, *, rank: int = RFT_LORA_RANK) -> None:
    adapter_bin = model_path / "adapter_model.bin"
    adapter_safetensors = model_path / "adapter_model.safetensors"
    adapter_config = model_path / "adapter_config.json"
    
    if adapter_bin.exists() or adapter_safetensors.exists() or adapter_config.exists():
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

    return llm


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format RFT data point. The reasoning already contains the answer in <answer> tags.
    
    We ensure the reasoning text includes:
    1. Full reasoning/explanation
    2. Both <answer> and </answer> tags
    3. The correct numerical value inside the tags
    """
    reasoning = reasoning.strip()
    if "<answer>" not in reasoning or "</answer>" not in reasoning:
        from .conversion_utils import format_numeric_answer
        formatted_answer = format_numeric_answer(answer)
        reasoning = f"{reasoning} <answer>{formatted_answer}</answer>"
    
    return {
        "question": question.strip(),
        "answer": reasoning,
    }


def train_model(
    output_dir: str = MODEL_NAME,
    **_: Any,
):
    import json
    from transformers import default_data_collator
    
    model_path = _resolve_path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    llm = BaseLLM(use_fp32_for_training=True)
    
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=RFT_LORA_RANK,
        lora_alpha=max(RFT_LORA_RANK * 4, 4),
        lora_dropout=0.0,
        inference_mode=False,
    )
    
    lora_model = get_peft_model(llm.model, config)
    
    lora_model.enable_input_require_grads()
    
    lora_model.train()
    
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(
            f"RFT dataset not found at {rft_data_path}. "
            "Please run: python -m homework.datagen data/rft.json"
        )
    
    with rft_data_path.open() as f:
        rft_data = json.load(f)
    
    if not rft_data or len(rft_data) == 0:
        raise ValueError(
            f"RFT dataset is empty at {rft_data_path}. "
            f"This can happen if data generation failed to produce any valid QA pairs. "
            f"Please run: python -m homework.datagen data/rft.json --oversample=15 --temperature=0.7"
        )
    
    print(f"Loaded {len(rft_data)} RFT training examples")
    if len(rft_data) < 850:
        print(f"WARNING: Only {len(rft_data)} examples. Target is 850-900+ for better generalization.")
    
    invalid_count = 0
    for i, example in enumerate(rft_data):
        if len(example) < 3:
            print(f"WARNING: Example {i} has invalid format: {example}")
            invalid_count += 1
            continue
        question, answer, reasoning = example[0], example[1], example[2]
        if "<answer>" not in reasoning or "</answer>" not in reasoning:
            print(f"WARNING: Example {i} missing answer tags in reasoning")
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"WARNING: {invalid_count} examples have format issues. Consider regenerating dataset.")
    else:
        print("All examples have proper format with answer tags.")
    
    class RFTDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    rft_dataset = RFTDataset(rft_data)
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_dataset, format_rft_example)
    
    sample = tokenized_dataset[0]
    non_masked_labels = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample non-masked labels: {non_masked_labels} out of {len(sample['labels'])}")
    if non_masked_labels == 0:
        raise ValueError("All labels are masked! Tokenization is incorrect.")
    
    use_bf16 = False
    if torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Using bfloat16 for training (faster than FP32, more stable than FP16)")
        else:
            print("Using FP32 for training (bf16 not available, fp16 disabled for stability)")
    
    training_args = TrainingArguments(
        output_dir=str(model_path),
        logging_dir=str(model_path),
        report_to="none",
        gradient_checkpointing=True,
        learning_rate=5e-4,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        bf16=use_bf16,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        label_names=["labels"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        remove_unused_columns=False,
    )
    
    data_collator = default_data_collator
    
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    train_result = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    if hasattr(train_result, 'metrics'):
        for key, value in train_result.metrics.items():
            if isinstance(value, (int, float)):
                if key == 'train_loss':
                    print(f"Final Loss: {value:.6f}")
                elif 'learning_rate' in key:
                    print(f"Learning Rate: {value:.2e}")
                else:
                    print(f"{key}: {value}")
    
    print(f"\nSaving model to {model_path}")
    trainer.save_model()
    
    print("Testing model...")
    test_model(str(model_path))


def test_model(ckpt_path: str = MODEL_NAME):
    """Test the RFT model on validation set."""
    from .sft import _resolve_path
    
    testset = Dataset("valid")
    model_path = _resolve_path(ckpt_path)
    _ensure_adapter(model_path, rank=RFT_LORA_RANK)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
