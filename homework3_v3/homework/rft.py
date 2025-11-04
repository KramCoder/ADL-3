from __future__ import annotations

from typing import Any

from peft import PeftModel

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch
from .sft import DEFAULT_LORA_RANK, _ensure_adapter, _resolve_path, test_model


MODEL_NAME = "rft_model"
RFT_LORA_RANK = max(DEFAULT_LORA_RANK * 2, 8)


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
    model_path = _resolve_path(output_dir)
    _ensure_adapter(model_path, rank=RFT_LORA_RANK)
    test_model(str(model_path))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
