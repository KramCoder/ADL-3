from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cot import CoTModel
from .data import Dataset, is_answer_valid

DEFAULT_OUTPUT_PATH = Path("data/rft.json")
RFT_MODEL_CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MIN_GENERATIONS = 10
MAX_GENERATIONS = 20


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def generate_dataset(
    output_json: str | Path = DEFAULT_OUTPUT_PATH,
    oversample: int = 15,
    temperature: float = 0.7,
) -> str:
    """Create an offline reasoning dataset for RFT training.

    For every question in the supervised dataset we sample `oversample` chain-of-thought
    completions from a stronger CoT model (SmolLM2-1.7B). Only the completions that contain the
    correct numeric answer inside the <answer></answer> tags are kept; the rest are discarded.

    Args:
        output_json: Path to output JSON file (defaults to data/rft.json)
        oversample: Number of generations per question (must be between 10 and 20)
        temperature: Sampling temperature (> 0 for diverse reasoning)
    """
    from tqdm import tqdm

    if not (MIN_GENERATIONS <= oversample <= MAX_GENERATIONS):
        raise ValueError(
            f"oversample must be between {MIN_GENERATIONS} and {MAX_GENERATIONS} inclusive; got {oversample}"
        )

    dataset = Dataset("train")
    model = CoTModel(checkpoint=RFT_MODEL_CHECKPOINT)
    records: list[list[Any]] = []
    rejected_count = 0

    # Process questions in batches for efficiency
    batch_size = 16
    questions_batch = []
    prompts_batch = []
    indices_batch = []
    
    for idx, (question, correct_answer, *_) in enumerate(tqdm(dataset.data, desc="Generating RFT dataset")):
        questions_batch.append(question)
        prompts_batch.append(model.format_prompt(question))
        indices_batch.append(correct_answer)
        
        # Process batch when it's full or at the end
        if len(prompts_batch) >= batch_size or idx == len(dataset.data) - 1:
            # Generate multiple sequences per question
            generations = model.batched_generate(
                prompts_batch,
                num_return_sequences=oversample,
                temperature=temperature,
            )
            
            # Process each question's generations
            for correct_answer, question_text, generations_list in zip(
                indices_batch, questions_batch, generations
            ):
                found_correct = False
                
                for reasoning in generations_list:
                    if "<answer>" not in reasoning or "</answer>" not in reasoning:
                        continue
                    
                    parsed_answer = model.parse_answer(reasoning)
                    if parsed_answer != parsed_answer:  # NaN check
                        continue
                    
                    if is_answer_valid(parsed_answer, correct_answer):
                        records.append([question_text, correct_answer, reasoning])
                        found_correct = True
                        break
                
                if not found_correct:
                    rejected_count += 1
            
            # Reset batch
            questions_batch = []
            prompts_batch = []
            indices_batch = []

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"\nGenerated {len(records)} QA pairs out of {len(dataset)} questions")
    print(f"Rejected {rejected_count} questions (no valid answer found)")
    print(f"Success rate: {len(records)/len(dataset)*100:.1f}%")
    
    # Warn if we don't have enough pairs
    if len(records) < 850:
        print(f"WARNING: Only {len(records)} pairs generated. Target is 850-900+ pairs.")
        print("Consider increasing oversample or improving CoT model accuracy.")
    elif len(records) >= 850:
        print(f"SUCCESS: Generated {len(records)} pairs (target: 850-900+)")

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
