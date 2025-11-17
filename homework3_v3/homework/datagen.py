from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .cot import CoTModel
from .data import Dataset, is_answer_valid
from .base_llm import CHECKPOINT_1_7B


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
    # Use 1.7B model for data generation as specified in README
    # "Using the HuggingFaceTB/SmolLM2-1.7B-Instruct model should further help you obtain better rollouts."
    model = CoTModel(checkpoint=CHECKPOINT_1_7B)
    records: list[list[Any]] = []
    rejected_count = 0

    # Process questions one at a time to ensure proper handling
    for idx, (question, correct_answer, *_) in enumerate(tqdm(dataset.data, desc="Generating RFT dataset")):
        # Generate multiple sequences (10-20) per question
        # The batched_generate method will handle memory optimization internally
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
        
        # If no correct answer found, ignore this data point (as per instructions)
        if not found_correct:
            rejected_count += 1
        
        # Clear CUDA cache after each question to prevent OOM
        # Also delete the generations list to free Python memory
        del generations, generations_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure cleanup completes

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
