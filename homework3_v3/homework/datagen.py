from __future__ import annotations

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
    import sys

    oversample = int(oversample)
    temperature = float(temperature)

    dataset = Dataset("train")
    
    print("Loading CoT model (1.7B) for RFT dataset generation...")
    print("This may take a minute on first run (downloading model if needed)...")
    sys.stdout.flush()
    
    try:
        model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
        print("Model loaded successfully!")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        raise
    
    print("Warming up model (this may take 10-30 seconds)...")
    sys.stdout.flush()
    try:
        _ = model.batched_generate(
            ["How many grams are in 1 kg?"],
            num_return_sequences=1,
            temperature=temperature
        )
        print("Model warmup complete!")
        sys.stdout.flush()
    except Exception as e:
        print(f"WARNING: Model warmup failed: {e}")
        print("Continuing anyway...")
        sys.stdout.flush()
    
    records: list[list[Any]] = []
    rejected_count = 0

    print(f"\nGenerating RFT dataset with {oversample} sequences per question...")
    print(f"Processing {len(dataset)} questions...")
    sys.stdout.flush()

    batch_size = 16 if torch.cuda.is_available() else 1
    
    for batch_idx in tqdm(range(0, len(dataset.data), batch_size), desc="Generating RFT dataset"):
        batch_data = dataset.data[batch_idx:batch_idx + batch_size]
        
        if not batch_data:
            continue
        
        batch_questions = []
        batch_correct_answers = []
        batch_data_mapping = []
        
        for item in batch_data:
            try:
                if len(item) < 2:
                    print(f"WARNING: Skipping malformed data entry at index {batch_idx}: {item}")
                    continue
                question, correct_answer = item[0], item[1]
                batch_questions.append(model.format_prompt(question))
                batch_correct_answers.append(correct_answer)
                batch_data_mapping.append(item)
            except (IndexError, TypeError, ValueError) as e:
                print(f"WARNING: Error processing data entry at index {batch_idx}: {e}")
                continue
        
        if not batch_questions:
            rejected_count += len(batch_data)
            continue
        
        try:
            all_generations = model.batched_generate(
                batch_questions,
                num_return_sequences=oversample,
                temperature=temperature
            )
        except Exception as e:
            print(f"\nERROR generating sequences for batch starting at index {batch_idx}: {e}")
            rejected_count += len(batch_data)
            continue
        
        if not isinstance(all_generations, list):
            print(f"\nERROR: batched_generate returned unexpected type at batch {batch_idx}: {type(all_generations)}")
            rejected_count += len(batch_data)
            continue
        
        if len(all_generations) != len(batch_questions):
            print(f"\nERROR: Mismatch between batch size ({len(batch_questions)}) and generations ({len(all_generations)}) at batch {batch_idx}")
            rejected_count += len(batch_data)
            continue
        
        for question_idx in range(len(batch_questions)):
            try:
                generations_list = all_generations[question_idx]
            except (IndexError, TypeError) as e:
                print(f"WARNING: Error accessing generations for question {question_idx} in batch {batch_idx}: {e}")
                rejected_count += 1
                continue
            
            if not isinstance(generations_list, (list, tuple)):
                print(f"WARNING: Unexpected generation type for question {question_idx} in batch {batch_idx}: {type(generations_list)}")
                rejected_count += 1
                continue
            
            if question_idx >= len(batch_data_mapping):
                print(f"WARNING: Question index {question_idx} exceeds batch_data_mapping length")
                rejected_count += 1
                continue
            
            question, correct_answer = batch_data_mapping[question_idx][0], batch_data_mapping[question_idx][1]
            
            found_correct = False
            
            for reasoning in generations_list:
                if not isinstance(reasoning, str):
                    continue
                
                if "<answer>" not in reasoning or "</answer>" not in reasoning:
                    continue
                
                parsed_answer = model.parse_answer(reasoning)
                if parsed_answer != parsed_answer:
                    continue
                
                if is_answer_valid(parsed_answer, correct_answer):
                    records.append([question, correct_answer, reasoning])
                    found_correct = True
                    break
            
            if not found_correct:
                rejected_count += 1
        
        del all_generations
        if torch.cuda.is_available() and batch_idx % (batch_size * 4) == 0:
            torch.cuda.empty_cache()

    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"\nGenerated {len(records)} QA pairs out of {len(dataset)} questions")
    print(f"Rejected {rejected_count} questions (no valid answer found)")
    if len(dataset) > 0:
        print(f"Success rate: {len(records)/len(dataset)*100:.1f}%")
    else:
        print("Success rate: N/A (no questions in dataset)")

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
