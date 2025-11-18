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


def generate_dataset(
    output_json: str, 
    oversample: int = 30, 
    temperature: float = 0.7,
    batch_size: int = 4,
    use_bfloat16: bool = True
) -> str:
    """Create an offline reasoning dataset for RFT training.

    Generate 10-20+ different completions from CoTModel, then select the one with
    the correct answer. If none of the answers are correct, ignore that data point.
    
    Args:
        output_json: Path to output JSON file
        oversample: Number of generations per question (increased to 30 for A100)
        temperature: Sampling temperature (> 0 for diversity)
        batch_size: Process multiple questions in parallel (A100 optimization)
        use_bfloat16: Use bfloat16 for faster inference on A100 (default: True)
    """
    from tqdm import tqdm
    import sys

    # Convert to proper types (Fire may pass these as strings from command line)
    oversample = int(oversample)
    temperature = float(temperature)
    batch_size = int(batch_size)
    use_bfloat16 = str(use_bfloat16).lower() in ('true', '1', 'yes')

    dataset = Dataset("train")
    
    # Use 1.7B model specifically for data generation as per README instructions
    # The README states: "Using the HuggingFaceTB/SmolLM2-1.7B-Instruct model should 
    # further help you obtain better rollouts."
    print("Loading CoT model (1.7B) for RFT dataset generation...")
    print("A100 GPU optimizations enabled: bfloat16, larger batches, parallel processing")
    print(f"Configuration: oversample={oversample}, batch_size={batch_size}, use_bfloat16={use_bfloat16}")
    print("This may take a minute on first run (downloading model if needed)...")
    sys.stdout.flush()
    
    try:
        # A100 optimization: Load model in bfloat16 for faster inference
        # Set environment variable to increase batch sizes
        import os
        os.environ['MICRO_BATCH_SIZE'] = '256'  # A100 can handle much larger batches
        os.environ['CHUNK_SIZE'] = '10'  # Larger chunks for sequence generation
        
        model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
        
        # A100 optimization: Convert model to bfloat16 for faster inference
        if use_bfloat16 and torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported'):
            if torch.cuda.is_bf16_supported():
                model.model = model.model.to(torch.bfloat16)
                print("Model converted to bfloat16 for faster inference on A100")
        
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
    print(f"Expected total generations: {len(dataset) * oversample}")
    sys.stdout.flush()

    # A100 optimization: Process multiple questions in parallel batches
    # This dramatically speeds up generation by utilizing GPU parallelism
    questions_data = list(dataset.data)
    total_batches = (len(questions_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(questions_data), batch_size), 
                          desc="Generating RFT dataset (batched)",
                          total=total_batches):
        batch_data = questions_data[batch_idx:batch_idx + batch_size]
        batch_questions = [data[0] for data in batch_data]
        batch_answers = [data[1] for data in batch_data]
        batch_prompts = [model.format_prompt(q) for q in batch_questions]
        # A100 optimization: Generate sequences for entire batch at once
        # This is much faster than processing one question at a time
        try:
            batch_generations = model.batched_generate(
                batch_prompts,
                num_return_sequences=oversample,
                temperature=temperature
            )
        except Exception as e:
            print(f"\nERROR generating sequences for batch {batch_idx}: {e}")
            print(f"Batch size: {len(batch_questions)}")
            rejected_count += len(batch_questions)
            continue
        
        # Process each question's generations in the batch
        for question, correct_answer, generations_list in zip(batch_questions, batch_answers, batch_generations):
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
        
        # A100 optimization: Only clear CUDA cache every N batches instead of every question
        # This reduces overhead while still preventing memory buildup
        del batch_generations
        if batch_idx % 10 == 0 and torch.cuda.is_available():
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
