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

    # Convert to proper types (Fire may pass these as strings from command line)
    oversample = int(oversample)
    temperature = float(temperature)

    dataset = Dataset("train")
    
    # Use 1.7B model specifically for data generation as per README instructions
    # The README states: "Using the HuggingFaceTB/SmolLM2-1.7B-Instruct model should 
    # further help you obtain better rollouts."
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
    
    # Warm up the model with a dummy generation to ensure it's ready
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
    
    # Detect A100 GPU and optimize batch size accordingly
    is_a100 = False
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        is_a100 = "a100" in gpu_name
        if is_a100:
            print(f"Detected A100 GPU: {torch.cuda.get_device_name(0)}")
            print("Using optimized batch processing for A100 (80GB memory)")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Batch size: Process multiple questions at once for A100
    # A100 can handle much larger batches than smaller GPUs
    if is_a100:
        question_batch_size = 8  # Process 8 questions at a time on A100
    else:
        question_batch_size = 1  # Conservative for other GPUs
    
    sys.stdout.flush()

    # Process questions in batches for A100, one at a time for other GPUs
    dataset_list = list(dataset.data)
    for batch_start in tqdm(range(0, len(dataset_list), question_batch_size), desc="Generating RFT dataset"):
        batch_end = min(batch_start + question_batch_size, len(dataset_list))
        batch_questions = []
        batch_answers = []
        batch_indices = []
        
        # Prepare batch
        for idx in range(batch_start, batch_end):
            question, correct_answer, *_ = dataset_list[idx]
            batch_questions.append(model.format_prompt(question))
            batch_answers.append(correct_answer)
            batch_indices.append(idx)
        
        # Generate multiple sequences for all questions in batch
        try:
            # batched_generate returns list[list[str]] when num_return_sequences is set
            generations_batch = model.batched_generate(
                batch_questions,
                num_return_sequences=oversample,
                temperature=temperature
            )
        except Exception as e:
            print(f"\nERROR generating sequences for batch starting at {batch_start}: {e}")
            rejected_count += len(batch_questions)
            continue
        
        # Process each question's generations
        for batch_idx in range(len(batch_questions)):
            # Extract original question from dataset
            idx = batch_indices[batch_idx]
            original_question = dataset_list[idx][0]
            correct_answer = batch_answers[batch_idx]
            generations_list = generations_batch[batch_idx]
            
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
                    records.append([original_question, correct_answer, reasoning])
                    found_correct = True
                    break
            
            # If no correct answer found, ignore this data point (as per instructions)
            if not found_correct:
                rejected_count += 1
        
        # Clear CUDA cache after each batch (less frequent on A100)
        del generations_batch
        if torch.cuda.is_available() and not is_a100:
            # Only clear cache frequently on non-A100 GPUs
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
