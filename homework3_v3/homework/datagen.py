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


def generate_dataset(output_json: str, oversample: int = 15, temperature: float = 0.7, batch_size: int = 8) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate 10-20 different completions from CoTModel, then select the one with
    the correct answer. If none of the answers are correct, ignore that data point.
    
    Args:
        output_json: Path to output JSON file
        oversample: Number of generations per question (10-20 as specified)
        temperature: Sampling temperature (> 0 for diversity)
        batch_size: Number of questions to process in parallel (optimized for A100)
    """
    from tqdm import tqdm
    import sys

    # Convert to proper types (Fire may pass these as strings from command line)
    oversample = int(oversample)
    temperature = float(temperature)
    batch_size = int(batch_size)

    dataset = Dataset("train")
    
    # Detect A100 GPU and optimize batch size accordingly
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Detected GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        if "A100" in gpu_name:
            # A100 can handle much larger batches
            if batch_size == 8:  # Only override if using default
                batch_size = 16  # Process 16 questions at a time on A100
            print(f"Using A100-optimized batch size: {batch_size}")
        sys.stdout.flush()
    
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
    print(f"Processing {len(dataset)} questions in batches of {batch_size}...")
    sys.stdout.flush()

    # Process questions in batches for much faster generation on A100
    dataset_list = list(dataset.data)
    for batch_idx in tqdm(range(0, len(dataset_list), batch_size), desc="Processing batches"):
        batch_data = dataset_list[batch_idx:batch_idx + batch_size]
        batch_questions = []
        batch_answers = []
        batch_indices = []
        
        # Prepare batch
        for idx, (question, correct_answer, *_) in enumerate(batch_data):
            batch_questions.append(model.format_prompt(question))
            batch_answers.append(correct_answer)
            batch_indices.append(batch_idx + idx)
        
        # Generate multiple sequences for all questions in batch
        try:
            # Generate all sequences for the batch at once
            all_generations = model.batched_generate(
                batch_questions,
                num_return_sequences=oversample,
                temperature=temperature
            )
        except Exception as e:
            print(f"\nERROR generating sequences for batch starting at index {batch_idx}: {e}")
            # Fallback: process one at a time if batch fails
            for local_idx, (question, correct_answer, *_) in enumerate(batch_data):
                try:
                    generations = model.batched_generate(
                        [model.format_prompt(question)],
                        num_return_sequences=oversample,
                        temperature=temperature
                    )
                    generations_list = generations[0]
                    
                    found_correct = False
                    for reasoning in generations_list:
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
                except Exception as e2:
                    print(f"ERROR for question {batch_indices[local_idx]}: {e2}")
                    rejected_count += 1
            continue
        
        # Process results for each question in the batch
        for local_idx, (question, correct_answer, *_) in enumerate(batch_data):
            # all_generations is a list of lists: [batch_size][num_return_sequences]
            generations_list = all_generations[local_idx]
            
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
        
        # Clear CUDA cache after each batch (less frequent than per-question)
        del all_generations
        if torch.cuda.is_available() and (batch_idx + batch_size) % (batch_size * 4) == 0:
            # Only clear cache every 4 batches to reduce overhead
            torch.cuda.empty_cache()

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
