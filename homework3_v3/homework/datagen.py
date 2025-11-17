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


def generate_dataset(output_json: str, num_completions: int = 15, temperature: float = 0.7) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate 10-20 different completions from CoTModel, then select the one with
    the correct answer. If none of the answers are correct, ignore that data point.
    
    Args:
        output_json: Path to output JSON file (should be data/rft.json)
        num_completions: Number of generations per question (10-20, default 15)
        temperature: Sampling temperature (should be > 0 for diversity)
    """
    from tqdm import tqdm

    dataset = Dataset("train")
    model = CoTModel()
    records: list[list[Any]] = []
    rejected_count = 0

    # Process questions one at a time to ensure we get diverse outputs
    for idx, (question, correct_answer, *_) in enumerate(tqdm(dataset.data, desc="Generating RFT dataset")):
        prompt = model.format_prompt(question)
        
        # Generate 10-20 different completions with temperature > 0 for diversity
        # Use num_return_sequences to get multiple diverse outputs
        generations = model.batched_generate(
            [prompt],
            num_return_sequences=num_completions,
            temperature=temperature
        )
        
        # generations is a list of lists: [[gen1, gen2, ..., genN]]
        generations_list = generations[0]
        
        # Find the first generation with a correct answer
        found_correct = False
        for reasoning in generations_list:
            # Verify reasoning contains answer tags
            if "<answer>" not in reasoning or "</answer>" not in reasoning:
                continue
            
            parsed_answer = model.parse_answer(reasoning)
            if parsed_answer != parsed_answer:  # Check for NaN
                continue
            
            if is_answer_valid(parsed_answer, correct_answer):
                # Store as [question, correct_answer, reasoning]
                records.append([question, correct_answer, reasoning])
                found_correct = True
                break
        
        # If no correct answer found, ignore this data point (as per instructions)
        if not found_correct:
            rejected_count += 1

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"\nGenerated {len(records)} QA pairs out of {len(dataset)} questions")
    print(f"Rejected {rejected_count} questions (no valid answer found)")
    print(f"Success rate: {len(records)/len(dataset)*100:.1f}%")
    
    # According to instructions, we should have 90+% success rate
    success_rate = len(records)/len(dataset)*100 if len(dataset) > 0 else 0
    if success_rate >= 90:
        print(f"SUCCESS: Success rate is {success_rate:.1f}% (target: 90%+)")
    else:
        print(f"WARNING: Success rate is {success_rate:.1f}% (target: 90%+)")
        print("Consider increasing num_completions or temperature for better diversity.")

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
