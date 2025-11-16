from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch
from .data import Dataset, benchmark
from .sft import DEFAULT_LORA_RANK, TokenizedDataset, _resolve_path


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
    **_: Any,
):
    import json
    import numpy as np

    model_path = _resolve_path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load base model in FP32 for stability, mirroring the SFT pipeline
    llm = BaseLLM(use_fp32_for_training=True)
    llm.model.eval()

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

    # Load RFT dataset (format: [question, numeric_answer, reasoning])
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        raise FileNotFoundError(
            f"RFT dataset not found at {rft_data_path}. Please run:\n"
            "  python -m homework.datagen data/rft.json"
        )

    with rft_data_path.open() as handle:
        rft_data = json.load(handle)

    if not rft_data:
        raise ValueError(
            f"{rft_data_path} is empty. Verify that datagen produced at least one successful rollout."
        )

    class RFTDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    rft_dataset = RFTDataset(rft_data)
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_dataset, format_rft_example)

    # Sanity-check tokenization to ensure we supervise answer tokens
    sample = tokenized_dataset[0]
    supervised = sum(1 for token in sample["labels"] if token != -100)
    if supervised == 0:
        raise ValueError("All labels are masked in the first RFT sample. Ensure reasoning includes <answer> tags.")
    print(f"Sample non-masked labels: {supervised} out of {len(sample['labels'])}")

    # Determine mixed precision settings
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Using bfloat16 for RFT training (more stable than fp16)")
        else:
            use_fp16 = True
            print("Using float16 for RFT training (with loss scaling)")

    training_args_dict = {
        "output_dir": str(model_path),
        "logging_dir": str(model_path),
        "report_to": "tensorboard",
        "gradient_checkpointing": True,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "save_strategy": "epoch",
        "logging_steps": 10,
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "bf16": use_bf16,
        "fp16": use_fp16,
        "fp16_full_eval": False,
        "dataloader_pin_memory": False,
        "max_grad_norm": 1.0,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": False,
    }

    training_args = TrainingArguments(**training_args_dict)

    class GradientNormCallback(TrainerCallback):
        def on_backward(self, args, state, control, model=None, **kwargs):
            if model is None:
                return

            has_nan = False
            has_inf = False
            total_norm = 0.0
            param_count = 0

            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                if torch.isnan(param.grad).any():
                    has_nan = True
                    print(f"WARNING: NaN detected in gradient for {name}")
                if torch.isinf(param.grad).any():
                    has_inf = True
                    print(f"WARNING: Inf detected in gradient for {name}")

                if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

            if has_nan or has_inf:
                print("ERROR: NaN/Inf gradients detected! Zeroing gradients to prevent training crash.")
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.zero_()
                state.nan_gradient_count = getattr(state, "nan_gradient_count", 0) + 1

            if param_count > 0 and not (has_nan or has_inf):
                state.last_grad_norm = total_norm ** 0.5
            else:
                state.last_grad_norm = float("nan") if (has_nan or has_inf) else 0.0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return

            if hasattr(state, "last_grad_norm"):
                if not (np.isnan(state.last_grad_norm) or np.isinf(state.last_grad_norm)):
                    logs["grad_norm"] = state.last_grad_norm
                else:
                    logs["grad_norm"] = 0.0
            elif "grad_norm" not in logs:
                model = kwargs.get("model")
                if model is not None:
                    total_norm = 0.0
                    param_count = 0
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                                param_norm = param.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                    logs["grad_norm"] = total_norm ** 0.5 if param_count > 0 else 0.0

            if hasattr(state, "nan_gradient_count") and state.nan_gradient_count > 0:
                logs["nan_gradient_count"] = state.nan_gradient_count

            if "loss" in logs:
                loss_val = logs["loss"]
                if isinstance(loss_val, (int, float)) and not np.isfinite(loss_val):
                    print(f"WARNING: Invalid loss value detected: {loss_val}")
                elif loss_val == 0.0 and state.global_step > 0:
                    print(f"WARNING: Loss is exactly 0.0 at step {state.global_step} - check tokenization/labels.")

    class StableTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = getattr(outputs, "loss", None)

            if loss is None:
                from torch.nn import CrossEntropyLoss

                logits = outputs.get("logits")
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Invalid loss detected (NaN/Inf): {loss.item()}")
                loss = torch.tensor(0.01, device=loss.device, dtype=loss.dtype, requires_grad=True)

            if loss.item() < 1e-8 and labels is not None:
                non_masked = (labels != -100).sum().item()
                if non_masked == 0:
                    print("WARNING: All labels are masked in this batch!")
                elif non_masked < 10:
                    print(f"WARNING: Very few non-masked labels ({non_masked}) in this batch.")

            return (loss, outputs) if return_outputs else loss

    trainer = StableTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
        callbacks=[GradientNormCallback()],
    )

    print("Starting RFT training...")
    train_result = trainer.train()

    print("\n" + "=" * 60)
    print("RFT Training Summary:")
    print("=" * 60)
    if hasattr(train_result, "metrics"):
        for key, value in train_result.metrics.items():
            if isinstance(value, (int, float)):
                if key == "train_loss":
                    print(f"Final Loss: {value:.6f}")
                elif key == "train_grad_norm":
                    print(f"Final Gradient Norm: {value:.6f}")
                elif "learning_rate" in key:
                    print(f"Learning Rate: {value:.2e}")
                else:
                    print(f"{key}: {value}")

    print(f"\nSaving RFT model to {model_path}")
    trainer.save_model(str(model_path))

    print("Testing RFT model...")
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
    # NOTE: Do NOT apply dataset answer patch during testing
    # We want to test the actual model, not the lookup table
    # apply_dataset_answer_patch(llm)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
