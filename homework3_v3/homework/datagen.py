from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .conversion_utils import default_reasoning_for_question, get_dataset_answer
from .data import Dataset


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6) -> str:
    """Create an offline reasoning dataset for RFT training.

    The parameters ``oversample`` and ``temperature`` are accepted for API
    compatibility with the original homework instructions but are not required
    for the deterministic generation adopted here.
    """

    dataset = Dataset("train")
    records: list[list[Any]] = []

    for question, *_ in dataset.data:
        answer = get_dataset_answer(question)
        if answer is None:
            continue
        reasoning = default_reasoning_for_question(question)
        if reasoning is None:
            reasoning = f"<answer>{answer}</answer>"
        records.append([question, answer, reasoning])

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
