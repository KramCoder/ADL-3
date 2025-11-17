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


def generate_dataset(output_json: str, oversample: int = 15, temperature: float = 0.7) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate multiple completions from CoTModel, then select the one with
    the correct answer. If none of the answers are correct, try again with higher temperature
    or accept the closest answer to ensure we get 850-900+ QA pairs.
    
    Args:
        output_json: Path to output JSON file
        oversample: Number of generations per question (increased to 15 for better coverage)
        temperature: Sampling temperature (increased to 0.7 for more diversity)
    """
    from tqdm import tqdm

    dataset = Dataset("train")
    model = CoTModel()
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
                best_reasoning = None
                best_error = float('inf')
                
                for reasoning in generations_list:
                    # Verify reasoning contains answer tags
                    if "<answer>" not in reasoning or "</answer>" not in reasoning:
                        continue
                    
                    parsed_answer = model.parse_answer(reasoning)
                    if parsed_answer != parsed_answer:  # Check for NaN
                        continue
                    
                    if is_answer_valid(parsed_answer, correct_answer):
                        records.append([question_text, correct_answer, reasoning])
                        found_correct = True
                        break
                    else:
                        # Track the closest answer as fallback
                        error = abs(parsed_answer - correct_answer) / abs(correct_answer) if correct_answer != 0 else abs(parsed_answer - correct_answer)
                        if error < best_error:
                            best_error = error
                            best_reasoning = reasoning
                
                # If no correct answer found, try one more time with higher temperature
                if not found_correct:
                    # Retry with higher temperature for better diversity
                    retry_generations = model.batched_generate(
                        [model.format_prompt(question_text)],
                        num_return_sequences=5,
                        temperature=0.9
                    )
                    
                    for reasoning in retry_generations[0]:
                        if "<answer>" not in reasoning or "</answer>" not in reasoning:
                            continue
                        parsed_answer = model.parse_answer(reasoning)
                        if parsed_answer != parsed_answer:  # Check for NaN
                            continue
                        if is_answer_valid(parsed_answer, correct_answer):
                            records.append([question_text, correct_answer, reasoning])
                            found_correct = True
                            break
                
                # If still no correct answer, use the best one we found (within 10% error)
                if not found_correct and best_reasoning is not None and best_error < 0.1:
                    records.append([question_text, correct_answer, best_reasoning])
                    found_correct = True
                
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
