#!/usr/bin/env python3
"""
Regression test for the SFT gradient-norm fix.

The script runs a small LoRA fine-tune, asserts that every logged grad_norm is
finite, and sanity-checks that we can decode numeric answers afterwards.
"""

from __future__ import annotations

import math
import re
import shutil
import sys
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Trainer, TrainingArguments, default_data_collator

from homework.base_llm import BaseLLM
from homework.data import Dataset
from homework.sft import TokenizedDataset, determine_batch_hparams, determine_precision_flags, format_example


TEST_OUTPUT_DIR = Path("/tmp/test_sft_output")
TEST_TARGET_BATCH = 8
TEST_MAX_STEPS = 8


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _collect_grad_norms(trainer: Trainer) -> list[float]:
    grad_norms: list[float] = []
    for log in trainer.state.log_history:
        grad = log.get("grad_norm")
        if grad is None:
            continue
        if not math.isfinite(grad):
            raise AssertionError(f"Detected non-finite grad_norm: {grad} (log entry: {log})")
        grad_norms.append(float(grad))
    if not grad_norms:
        raise AssertionError("Trainer did not log any grad_norm values.")
    return grad_norms


def _sanity_check_generation(model_path: Path) -> None:
    dataset = Dataset("train")
    questions = [dataset[i][0] for i in range(3)]
    prompts = [f"{question} <answer>" for question in questions]

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    generations = llm.batched_generate(prompts)
    parsed: list[float] = []
    print("Sample generations:")
    for question, prompt, generation in zip(questions, prompts, generations, strict=True):
        print(f"  Q: {question}")
        print(f"     Prompt tail: '{prompt.split(question)[-1].strip()}'")
        print(f"     Raw: {generation}")
        match = re.search(r"-?\d+(?:\.\d+)?", generation)
        parsed.append(float(match.group()) if match else math.nan)

    if not any(math.isfinite(value) for value in parsed):
        raise AssertionError("Model failed to emit numeric text during sanity check.")


def test_training_fix() -> bool:
    print("=" * 60)
    print("Testing SFT training pipeline")
    print("=" * 60)

    _prepare_output_dir(TEST_OUTPUT_DIR)

    llm = BaseLLM()
    llm.model.eval()

    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    lora_model = get_peft_model(llm.model, config)
    lora_model.enable_input_require_grads()
    lora_model.train()

    train_dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)

    per_device_batch, grad_accum = determine_batch_hparams(target_batch_size=TEST_TARGET_BATCH)
    fp16, bf16 = determine_precision_flags()

    training_args = TrainingArguments(
        output_dir=str(TEST_OUTPUT_DIR),
        logging_dir=str(TEST_OUTPUT_DIR / "logs"),
        report_to="none",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=TEST_MAX_STEPS,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        save_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
        fp16=fp16,
        bf16=bf16,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        label_names=["labels"],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )

    print("Running short training run…")
    result = trainer.train()
    print(f"Final loss: {result.training_loss:.4f}")

    grad_norms = _collect_grad_norms(trainer)
    print(f"Logged grad_norm range: {min(grad_norms):.4f} – {max(grad_norms):.4f}")

    trainer.save_model(str(TEST_OUTPUT_DIR))

    print("Sanity-checking generation…")
    _sanity_check_generation(TEST_OUTPUT_DIR)

    print("✓ Training pipeline healthy.")
    return True


if __name__ == "__main__":
    success = test_training_fix()
    sys.exit(0 if success else 1)
