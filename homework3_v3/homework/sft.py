from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch, format_numeric_answer
from .data import Dataset, benchmark


MODEL_NAME = "sft_model"
DEFAULT_LORA_RANK = 16


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent / candidate


def _ensure_adapter(model_path: Path, *, rank: int = DEFAULT_LORA_RANK) -> None:
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
        lora_alpha=rank * 2,
        lora_dropout=0.05,
    )

    lora_model = get_peft_model(llm.model, config)
    lora_model.save_pretrained(model_path)


def load() -> BaseLLM:
    model_path = _resolve_path(MODEL_NAME)
    _ensure_adapter(model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We concatenate question and answer, then create labels where only the answer
    tokens are supervised (question tokens are masked with -100).
    
    Simplified strategy: Tokenize question and answer separately, then concatenate.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_length = 128
    
    question_text = f"{question} "
    question_encoded = tokenizer(
        question_text,
        add_special_tokens=True,
        truncation=False,
        return_tensors=None,
    )
    question_ids = question_encoded["input_ids"]
    
    answer_text = f"{answer}{tokenizer.eos_token}"
    answer_encoded = tokenizer(
        answer_text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
    )
    answer_ids = answer_encoded["input_ids"]
    
    input_ids = question_ids + answer_ids
    
    if len(input_ids) > max_length:
        min_answer_tokens = min(len(answer_ids), 20)
        question_max = max_length - min_answer_tokens
        question_ids = question_ids[:question_max]
        answer_ids = answer_ids[:(max_length - len(question_ids))]
        input_ids = question_ids + answer_ids
    
    question_len = len(question_ids)
    
    attention_mask = [1] * len(input_ids)
    
    labels = [-100] * question_len + answer_ids
    
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
    
    non_masked = sum(1 for l in labels if l != -100)
    if non_masked == 0:
        for i in range(max(0, len(input_ids) - 10), len(input_ids)):
            if attention_mask[i] == 1:
                labels[i] = input_ids[i]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair.
    Format: question + <answer>{answer}</answer>
    """
    formatted_answer = format_numeric_answer(answer)
    return {
        "question": prompt.strip(),
        "answer": f"<answer>{formatted_answer}</answer>",
    }


def format_example_rft(question: str, correct_answer: float, reasoning: str) -> dict[str, str]:
    """
    Construct a question / reasoning pair for RFT training.
    Format: question + reasoning (where reasoning already contains <answer> tags)
    """
    return {
        "question": question.strip(),
        "answer": reasoning.strip(),
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
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted_data)


def train_model(
    output_dir: str = MODEL_NAME,
    max_steps: int | None = None,
    **_: Any,
):
    from transformers import Trainer, TrainingArguments, default_data_collator, TrainerCallback
    import numpy as np
    
    class GradientNormCallback(TrainerCallback):
        def on_backward(self, args, state, control, model=None, **kwargs):
            """Compute gradient norm right after backward, before optimizer step"""
            if model is not None:
                total_norm = 0.0
                param_count = 0
                has_nan = False
                has_inf = False
                
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if torch.isnan(param.grad).any():
                            has_nan = True
                            print(f"WARNING: NaN detected in gradient for {name}")
                        if torch.isinf(param.grad).any():
                            has_inf = True
                            print(f"WARNING: Inf detected in gradient for {name}")
                        
                        if not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                
                if has_nan or has_inf:
                    print(f"ERROR: NaN/Inf gradients detected! Zeroing gradients to prevent training crash.")
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad.zero_()
                    state.last_grad_norm = 0.0
                elif param_count > 0:
                    state.last_grad_norm = total_norm ** (1. / 2)
                else:
                    state.last_grad_norm = 0.0
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Update logs with computed gradient norm"""
            if logs is not None:
                if hasattr(state, 'last_grad_norm'):
                    if not (np.isnan(state.last_grad_norm) or np.isinf(state.last_grad_norm)):
                        logs['grad_norm'] = state.last_grad_norm
                    else:
                        logs['grad_norm'] = 0.0
                
                if 'loss' in logs:
                    loss_val = logs['loss']
                    if isinstance(loss_val, (float, int)):
                        if np.isnan(loss_val) or np.isinf(loss_val):
                            print(f"WARNING: Invalid loss value detected: {loss_val}")
                        elif loss_val == 0.0 and state.global_step > 0:
                            print(f"WARNING: Loss is exactly 0.0 at step {state.global_step} - this may indicate an issue")
    
    model_path = _resolve_path(output_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    llm = BaseLLM(use_fp32_for_training=True)
    
    llm.model.eval()
    
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_RANK * 2,
        lora_dropout=0.05,
    )
    
    lora_model = get_peft_model(llm.model, config)
    
    lora_model.enable_input_require_grads()
    
    lora_model.train()
    
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    import json
    from pathlib import Path
    from .datagen import generate_dataset
    
    data_dir = Path(__file__).parent.parent / "data"
    rft_data_path = data_dir / "rft.json"
    
    if not rft_data_path.exists():
        print(f"RFT data file not found at {rft_data_path}.")
        print("Automatically generating RFT dataset...")
        relative_path = "data/rft.json"
        generated_path = generate_dataset(relative_path, oversample=15, temperature=0.7)
        resolved_generated = Path(generated_path).resolve()
        resolved_expected = rft_data_path.resolve()
        if resolved_generated != resolved_expected:
            print(f"Warning: Generated path {resolved_generated} differs from expected {resolved_expected}")
        print(f"RFT dataset generated successfully at {rft_data_path}")
    
    with rft_data_path.open() as f:
        rft_data = json.load(f)
    
    if not rft_data or len(rft_data) == 0:
        raise ValueError(
            f"RFT dataset is empty at {rft_data_path}. "
            f"This can happen if data generation failed to produce any valid QA pairs. "
            f"Please run: python -m homework.datagen {relative_path} --oversample=15 --temperature=0.7"
        )
    
    class RFTDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = RFTDataset(rft_data)
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example_rft)
    
    sample = tokenized_dataset[0]
    non_masked_labels = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample non-masked labels: {non_masked_labels} out of {len(sample['labels'])}")
    if non_masked_labels == 0:
        raise ValueError("All labels are masked! Tokenization is incorrect.")
    
    use_bf16 = False
    if torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Using bfloat16 for training (more stable than fp16)")
        else:
            print("Using FP32 for training (bf16 not available, fp16 disabled for stability)")
    
    training_args_dict = {
        "output_dir": str(model_path),
        "logging_dir": str(model_path),
        "report_to": "none",
        "gradient_checkpointing": True,
        "learning_rate": 5e-4,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "logging_steps": 10,
        "save_total_limit": 0,
        "save_strategy": "no",
        "remove_unused_columns": False,
        "bf16": use_bf16,
        "dataloader_pin_memory": False,
        "max_grad_norm": 1.0,
        "label_names": ["labels"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "warmup_steps": 0,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": False,
        "weight_decay": 0.01,
    }
    
    if max_steps is not None:
        training_args_dict["max_steps"] = max_steps
    else:
        training_args_dict["num_train_epochs"] = 10
    
    training_args = TrainingArguments(**training_args_dict)
    
    data_collator = default_data_collator
    
    class StableTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Custom loss computation with validation to prevent NaN/Inf.
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
            if loss is None:
                from torch.nn import CrossEntropyLoss
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
                    print(f"WARNING: All labels are masked in this batch! Loss: {loss.item()}")
                elif non_masked < 10:
                    print(f"WARNING: Very few non-masked labels ({non_masked}) in this batch. Loss: {loss.item()}")
            
            return (loss, outputs) if return_outputs else loss
    
    trainer = StableTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[GradientNormCallback()],
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
                elif key == 'train_grad_norm':
                    print(f"Final Gradient Norm: {value:.6f}")
                elif 'learning_rate' in key:
                    print(f"Learning Rate: {value:.2e}")
                else:
                    print(f"{key}: {value}")
    
    print(f"\nSaving model to {model_path}")
    trainer.save_model(str(model_path))
    
    print("Testing model...")
    test_model(str(model_path))


def test_model(ckpt_path: str = MODEL_NAME):
    testset = Dataset("valid")
    model_path = _resolve_path(ckpt_path)
    _ensure_adapter(model_path)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
