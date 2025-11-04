from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cot import CoTModel
from .conversion_utils import get_dataset_answer
from .data import Dataset


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6) -> str:
    """Create an offline reasoning dataset for RFT training.
    
    Uses CoTModel to generate multiple completions per question, then selects
    the one with the correct answer (rejection sampling).
    """
    from .data import is_answer_valid

    dataset = Dataset("train")
    model = CoTModel()
    records: list[list[Any]] = []

    # Process questions in batches for efficiency
    batch_size = 32
    num_return_sequences = min(oversample, 20)  # Use oversample, but cap at 20
    
    for i in range(0, len(dataset), batch_size):
        batch_questions = [dataset.data[j][0] for j in range(i, min(i + batch_size, len(dataset)))]
        batch_ground_truth = [dataset.data[j][1] for j in range(i, min(i + batch_size, len(dataset)))]
        
        # Format prompts using CoTModel's format_prompt (applies chat template)
        formatted_prompts = [model.format_prompt(q) for q in batch_questions]
        
        # Generate multiple completions for each question
        # batched_generate returns list[list[str]] when num_return_sequences is provided
        all_completions = model.batched_generate(
            formatted_prompts,
            num_return_sequences=num_return_sequences,
            temperature=temperature
        )
        
        # Process each question and its completions
        for question, ground_truth, completions in zip(batch_questions, batch_ground_truth, all_completions):
            # Find the first completion with the correct answer
            found_correct = False
            for completion in completions:
                parsed_answer = model.parse_answer(completion)
                
                # Check if answer is valid (not NaN and matches ground truth)
                if parsed_answer == parsed_answer and is_answer_valid(parsed_answer, ground_truth):
                    # Found a correct answer - add to dataset
                    records.append([question, float(ground_truth), completion])
                    found_correct = True
                    break
            
            # If none of the samples were correct, skip this data point (as per instructions)

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
