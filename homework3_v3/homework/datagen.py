from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cot import CoTModel
from .data import Dataset, benchmark, is_answer_valid


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).parent.parent / candidate


def _validate_reasoning_has_answer_tags(reasoning: str) -> bool:
    """Validate that reasoning contains both opening and closing answer tags."""
    has_open_tag = "<answer>" in reasoning
    has_close_tag = "</answer>" in reasoning
    return has_open_tag and has_close_tag


def _check_cot_accuracy(model: CoTModel, min_accuracy: float = 0.35) -> float:
    """Check CoT model accuracy and warn if below threshold.
    
    Args:
        model: The CoT model to test
        min_accuracy: Minimum accuracy threshold (default 0.35, grader needs >0.4)
    
    Returns:
        The actual accuracy achieved
    """
    from .data import Dataset
    testset = Dataset("valid")
    benchmark_result = benchmark(model, testset, 100)
    accuracy = benchmark_result.accuracy
    
    print(f"\n{'='*60}")
    print(f"CoT Model Accuracy Check: {accuracy:.4f}")
    print(f"{'='*60}")
    if accuracy < min_accuracy:
        print(f"WARNING: CoT accuracy ({accuracy:.4f}) is below recommended threshold ({min_accuracy:.4f})")
        print("This may result in many rejections during datagen.")
        print("Consider improving the CoT model before generating RFT dataset.")
    else:
        print(f"✓ CoT accuracy ({accuracy:.4f}) is above threshold ({min_accuracy:.4f})")
    print()
    
    return accuracy


def generate_dataset(
    output_json: str, 
    oversample: int = 15, 
    temperature: float = 0.7,
    target_pairs: int = 850,
    check_cot_accuracy: bool = True,
    min_cot_accuracy: float = 0.35
) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate multiple completions from CoTModel, then select the one with
    the correct answer. Ensures we generate at least target_pairs (default 850-900).
    
    Args:
        output_json: Path to output JSON file
        oversample: Number of sequences to generate per question (default 15)
        temperature: Sampling temperature (default 0.7 for diversity)
        target_pairs: Target number of QA pairs to generate (default 850)
        check_cot_accuracy: Whether to check CoT accuracy before generation (default True)
        min_cot_accuracy: Minimum CoT accuracy threshold (default 0.35)
    
    Returns:
        Path to the generated dataset file
    """
    from tqdm import tqdm

    dataset = Dataset("train")
    model = CoTModel()
    
    # Check CoT accuracy first if requested
    if check_cot_accuracy:
        _check_cot_accuracy(model, min_cot_accuracy)
    
    records: list[list[Any]] = []
    rejected_count = 0
    invalid_reasoning_count = 0

    # Process questions in batches for efficiency
    batch_size = 16
    questions_batch = []
    prompts_batch = []
    indices_batch = []
    
    print(f"\nGenerating RFT dataset with target of {target_pairs} QA pairs...")
    print(f"Using oversample={oversample}, temperature={temperature}\n")
    
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
                # Find the first generation with a correct answer and valid reasoning
                found_correct = False
                for reasoning in generations_list:
                    # Validate reasoning has answer tags
                    if not _validate_reasoning_has_answer_tags(reasoning):
                        invalid_reasoning_count += 1
                        continue
                    
                    parsed_answer = model.parse_answer(reasoning)
                    if not (parsed_answer != parsed_answer):  # Check for NaN
                        if is_answer_valid(parsed_answer, correct_answer):
                            records.append([question_text, correct_answer, reasoning])
                            found_correct = True
                            break
                
                # If no correct answer found, skip this data point
                if not found_correct:
                    rejected_count += 1
            
            # Reset batch
            questions_batch = []
            prompts_batch = []
            indices_batch = []
        
        # Early exit if we've reached target
        if len(records) >= target_pairs:
            print(f"\n✓ Reached target of {target_pairs} QA pairs!")
            break

    # Report results
    print(f"\n{'='*60}")
    print(f"Dataset Generation Summary:")
    print(f"{'='*60}")
    print(f"Generated QA pairs: {len(records)}")
    print(f"Target: {target_pairs}")
    print(f"Rejected questions: {rejected_count}")
    print(f"Invalid reasoning (missing tags): {invalid_reasoning_count}")
    print(f"Success rate: {len(records)/(len(records)+rejected_count)*100:.1f}%" if (len(records)+rejected_count) > 0 else "N/A")
    
    if len(records) < target_pairs:
        print(f"\nWARNING: Only generated {len(records)} pairs, below target of {target_pairs}")
        print("Consider:")
        print("  1. Increasing oversample (currently {})".format(oversample))
        print("  2. Improving CoT model accuracy")
        print("  3. Adjusting temperature (currently {})".format(temperature))
    else:
        print(f"\n✓ Successfully generated {len(records)} QA pairs (target: {target_pairs})")
    
    # Validate all records have proper format
    valid_records = []
    for record in records:
        if len(record) >= 3:
            question, answer, reasoning = record[0], record[1], record[2]
            if _validate_reasoning_has_answer_tags(reasoning):
                valid_records.append(record)
    
    if len(valid_records) < len(records):
        print(f"\nWARNING: {len(records) - len(valid_records)} records missing answer tags, filtering them out")
        records = valid_records
    
    output_path = _resolve_output_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"Final dataset size: {len(records)} QA pairs")
    print(f"{'='*60}\n")

    return str(output_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
