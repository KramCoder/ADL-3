from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Trainer, TrainingArguments, default_data_collator

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch
from .data import Dataset, benchmark
from .sft import DEFAULT_LORA_RANK, TokenizedDataset, _resolve_path, tokenize


MODEL_NAME = "rft_model"
RFT_LORA_RANK = max(DEFAULT_LORA_RANK * 2, 16)


def _ensure_adapter(model_path: Path, *, rank: int = RFT_LORA_RANK) -> None:
    # Check for both .bin and .safetensors formats (newer versions use safetensors)
    adapter_bin = model_path / "adapter_model.bin"
    adapter_safetensors = model_path / "adapter_model.safetensors"
    adapter_config = model_path / "adapter_config.json"
    
    # If any adapter file exists, assume the adapter is already created
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
    # DO NOT apply dataset answer patch - we want the actual trained model!
    # apply_dataset_answer_patch(llm)

    return llm


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    Format RFT data point. The reasoning already contains the answer in <answer> tags.
    
    We ensure the reasoning text includes:
    1. Full reasoning/explanation
    2. Both <answer> and </answer> tags
    3. The correct numerical value inside the tags
    """
    # Verify reasoning has answer tags
    reasoning = reasoning.strip()
    if "<answer>" not in reasoning or "</answer>" not in reasoning:
        # If missing tags, add them (shouldn't happen with proper datagen)
        from .conversion_utils import format_numeric_answer
        formatted_answer = format_numeric_answer(answer)
        reasoning = f"{reasoning} <answer>{formatted_answer}</answer>"
    
    return {
        "question": question.strip(),
        "answer": reasoning,  # Full reasoning text with answer tags
    }


def train_model(
    output_dir: str = MODEL_NAME,
    **_: Any,
):
    import json
    
    model_path = _resolve_path(output_dir)
    
    # Load base model in FP32 for training stability
    # Mixed precision will be handled by TrainingArguments
    llm = BaseLLM(use_fp32_for_training=True)
    
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=RFT_LORA_RANK,
        lora_alpha=max(RFT_LORA_RANK * 4, 4),
        lora_dropout=0.0,
        inference_mode=False,  # CRITICAL: Must be False for training
    )
    
    lora_model = get_peft_model(llm.model, config)
    
    # Set model to training mode
    lora_model.train()
    
    # Enable input require grads for gradient checkpointing
    lora_model.enable_input_require_grads()
    
    # Try to compile model for additional speedup (PyTorch 2.0+)
    # This can provide 10-30% speedup on modern GPUs
    # Note: torch.compile may not work with all PeftModel configurations
    # If it fails, training will continue without compilation
    try:
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            # Check PyTorch version (compile requires 2.0+)
            if torch.__version__ >= "2.0":
                print("Compiling model with torch.compile for faster training...")
                # Use 'default' mode for better compatibility with PeftModel
                lora_model = torch.compile(lora_model, mode="default")
                print("Model compilation successful!")
    except Exception as e:
        print(f"Model compilation not available or failed: {e}")
        print("Continuing without compilation (this is normal for some configurations)...")
    
    # Load RFT dataset (format: [question, answer, reasoning])
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(
            f"RFT dataset not found at {rft_data_path}. "
            "Please run: python -m homework.datagen data/rft.json"
        )
    
    with rft_data_path.open() as f:
        rft_data = json.load(f)
    
    # Validate dataset quality
    print(f"Loaded {len(rft_data)} RFT training examples")
    if len(rft_data) < 850:
        print(f"WARNING: Only {len(rft_data)} examples. Target is 850-900+ for better generalization.")
    
    # Verify all examples have proper format with answer tags
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
    
    # Create a dataset-like object for TokenizedDataset
    class RFTDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    rft_dataset = RFTDataset(rft_data)
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_dataset, format_rft_example)
    
    # Determine precision settings - prefer bf16 for stability and speed
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        # Check if bf16 is supported (Ampere+ GPUs: A100, RTX 30xx, etc.)
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Using bfloat16 for training (2x speedup, more stable than fp16)")
        else:
            # Fallback to fp16 if bf16 not available
            use_fp16 = True
            print("Using float16 for training (2x speedup, bf16 not available)")
    else:
        print("Using FP32 for training (CPU/MPS - no mixed precision)")
    
    # Optimized training arguments for speed
    training_args = TrainingArguments(
        output_dir=str(model_path),
        logging_dir=str(model_path),
        report_to="tensorboard",
        # Mixed precision training (2x speedup)
        bf16=use_bf16,
        fp16=use_fp16,
        # Gradient settings
        gradient_checkpointing=True,  # Trade compute for memory
        gradient_accumulation_steps=2,  # Effective batch size = 32 * 2 = 64
        max_grad_norm=1.0,  # Gradient clipping for stability
        # Batch size - reduced per device to allow gradient accumulation
        per_device_train_batch_size=16,  # Effective batch size: 16 * 2 = 32
        # Learning rate and optimization
        learning_rate=2e-4,
        lr_scheduler_type="cosine",  # Better convergence than linear
        warmup_ratio=0.1,  # 10% warmup steps
        weight_decay=0.01,  # L2 regularization
        # Training duration
        num_train_epochs=3,
        # DataLoader optimizations
        dataloader_num_workers=4,  # Parallel data loading (adjust based on CPU cores)
        dataloader_pin_memory=True,  # Faster GPU transfer
        # Logging and saving
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        # Other optimizations
        remove_unused_columns=False,  # Keep our custom labels
        ddp_find_unused_parameters=False,  # Faster distributed training if used
        label_names=["labels"],  # Explicitly specify label field
    )
    
    # Create trainer with optimized data collator
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,  # Efficient batching
    )
    
    # Train
    print("Starting optimized RFT training...")
    print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"DataLoader workers: {training_args.dataloader_num_workers}")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Test the model
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
    # DO NOT apply dataset answer patch - we want to test the actual trained model!
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
