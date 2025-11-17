from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def generate_dataset(output_json: str, oversample: int = 15, temperature: float = 0.7) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate 10-20 different completions from CoTModel, then select the one with
    the correct answer. If none of the answers are correct, ignore that data point.
    
    Args:
        output_json: Path to output JSON file
        oversample: Number of generations per question (10-20 as specified)
        temperature: Sampling temperature (> 0 for diversity)
    """
    from tqdm import tqdm

    # Convert to proper types (Fire may pass these as strings from command line)
    oversample = int(oversample)
    temperature = float(temperature)

    dataset = Dataset("train")
    model = CoTModel()
    records: list[list[Any]] = []
    rejected_count = 0

    # Process questions one at a time to ensure proper handling and minimize memory usage
    for idx, (question, correct_answer, *_) in enumerate(tqdm(dataset.data, desc="Generating RFT dataset")):
        # Generate multiple sequences (10-20) per question
        # Process one question at a time to minimize memory usage
        generations = model.batched_generate(
            [model.format_prompt(question)],
            num_return_sequences=oversample,
            temperature=temperature
        )
        
        # generations is a list of lists, so get the first (and only) item
        generations_list = generations[0]
        
        # Find the first generation with a correct answer
        found_correct = False
        
        for reasoning in generations_list:
            # Verify reasoning contains answer tags
            if "<answer>" not in reasoning or "</answer>" not in reasoning:
                continue
            
            parsed_answer = model.parse_answer(reasoning)
            # Check for NaN
            if parsed_answer != parsed_answer:
                continue
            
            # Check if answer is valid
            if is_answer_valid(parsed_answer, correct_answer):
                records.append([question, correct_answer, reasoning])
                found_correct = True
                break
        
        # Clear generations list to free memory
        del generations, generations_list
        
        # If no correct answer found, ignore this data point (as per instructions)
        if not found_correct:
            rejected_count += 1
        
        # Aggressively clear CUDA cache and run garbage collection after each question
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete
        gc.collect()  # Force Python garbage collection

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"\nGenerated {len(records)} QA pairs out of {len(dataset)} questions")
    print(f"Rejected {rejected_count} questions (no valid answer found)")
    print(f"Success rate: {len(records)/len(dataset)*100:.1f}%")

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
