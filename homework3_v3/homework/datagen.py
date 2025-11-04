from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate 10-20 different completions from CoTModel, then select the one with
    the correct answer. If none of the answers are correct, ignore that data point.
    """
    from tqdm import tqdm

    dataset = Dataset("train")
    model = CoTModel()
    records: list[list[Any]] = []

    # Process questions in batches for efficiency
    batch_size = 16
    questions_batch = []
    prompts_batch = []
    indices_batch = []
    
    for idx, (question, correct_answer, *_) in enumerate(tqdm(dataset.data, desc="Generating RFT dataset")):
        questions_batch.append(question)
        prompts_batch.append(model.format_prompt(question))
        indices_batch.append((idx, correct_answer))
        
        # Process batch when it's full or at the end
        if len(prompts_batch) >= batch_size or idx == len(dataset.data) - 1:
            # Generate multiple sequences per question
            generations = model.batched_generate(
                prompts_batch,
                num_return_sequences=oversample,
                temperature=temperature
            )
            
            # Process each question's generations
            for (orig_idx, correct_answer), question_text, generations_list in zip(indices_batch, questions_batch, generations):
                # Find the first generation with a correct answer
                found_correct = False
                for reasoning in generations_list:
                    parsed_answer = model.parse_answer(reasoning)
                    if not (parsed_answer != parsed_answer):  # Check for NaN
                        if is_answer_valid(parsed_answer, correct_answer):
                            records.append([question_text, correct_answer, reasoning])
                            found_correct = True
                            break
                
                # If no correct answer found, skip this data point
                if not found_correct:
                    continue
            
            # Reset batch
            questions_batch = []
            prompts_batch = []
            indices_batch = []

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
