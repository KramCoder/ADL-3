# RFT Data Generation Logic Analysis

## Overview
This document explains how the Rejection Sampling Fine-Tuning (RFT) dataset generation works and how it integrates with the SFT training pipeline.

## RFT Dataset Generation Logic (`datagen.py`)

### Purpose
Generate a high-quality dataset of question/reasoning/answer tuples by using a pre-trained LLM with in-context learning (CoTModel) to create diverse completions and selecting only correct ones.

### Implementation Details

#### 1. **Input Parameters**
```python
def generate_dataset(
    output_json: str,           # Path to save rft.json
    oversample: int = 15,       # Number of generations per question (10-20 range)
    temperature: float = 0.7    # Sampling temperature for diversity
)
```

#### 2. **Process Flow**

**Step 1: Load Training Data**
- Uses `Dataset("train")` to load all training questions
- Each entry has format: `[question, correct_answer, ...]`

**Step 2: Generate Multiple Diverse Completions**
```python
generations = model.batched_generate(
    [model.format_prompt(question)],
    num_return_sequences=oversample,  # Generate 10-20 diverse outputs
    temperature=temperature            # > 0 for diversity
)
```

**Key Features:**
- Uses `CoTModel` which has in-context learning examples
- `num_return_sequences=15` generates 15 diverse outputs per question
- `temperature=0.7` provides good balance between diversity and quality
- Batched generation is efficient (processes one question at a time to ensure proper handling)

**Step 3: Select First Correct Completion**
For each question, the code:
1. Iterates through all generated completions
2. Validates each completion:
   - Must contain `<answer>` and `</answer>` tags
   - Parsed answer must not be NaN
   - Answer must match correct answer (using `is_answer_valid`)
3. Selects the **first** correct completion
4. If no correct answer found, **skips that data point** (as per requirements)

**Step 4: Save Dataset**
- Saves to `data/rft.json` in format:
  ```json
  [
    ["question", correct_answer, "reasoning with <answer>X</answer>"],
    ...
  ]
  ```

#### 3. **Quality Metrics**
The function prints:
- Number of QA pairs generated
- Number of questions rejected (no valid answer)
- Success rate (should be 90+%)

### Example Output Format
```json
[
  [
    "How many gram are there per 6 kg?",
    6000.0,
    "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
  ],
  ...
]
```

## Integration with SFT Training (`sft.py`)

### Automatic Dataset Generation

The `train_model` function now includes automatic dataset generation:

```python
# Auto-generate RFT dataset if it doesn't exist
if not rft_data_path.exists():
    print("RFT dataset not found. Generating it now...")
    
    from .datagen import generate_dataset
    
    # Generate the dataset with default parameters
    generated_path = generate_dataset(
        output_json="data/rft.json",
        oversample=15,        # 10-20 diverse completions
        temperature=0.7       # Good diversity
    )
    
    print(f"RFT dataset generated successfully at {generated_path}")
    print("Proceeding with SFT training...")
```

### Workflow When Running `python -m homework.sft train`

1. **Check for RFT Dataset**
   - Looks for `data/rft.json`
   
2. **Auto-Generate if Missing**
   - If not found, automatically calls `generate_dataset`
   - Uses CoTModel with in-context learning
   - Generates 15 diverse completions per question
   - Selects correct completions only
   - Saves to `data/rft.json`
   
3. **Load Dataset**
   - Loads the generated (or existing) `rft.json`
   - Validates format

4. **Train on RFT Data**
   - Uses `format_example_rft` to format data
   - Trains on question + reasoning pairs
   - Reasoning includes full CoT and answer in `<answer>` tags

## Key Advantages of This Approach

1. **Fully Automatic**: No manual data generation step required
2. **High Quality**: Only correct reasoning chains are included
3. **Diverse**: Temperature sampling creates varied reasoning patterns
4. **Efficient**: Batched generation for speed
5. **Robust**: Validates all completions before including them

## Expected Results

- **Success Rate**: 90+% (9 out of 10 questions get at least one correct completion)
- **Dataset Size**: 850-900+ examples (from ~1000 training questions)
- **Training Improvement**: RFT model should outperform basic SFT due to reasoning chains

## Usage

Simply run:
```bash
python -m homework.sft train
```

The system will:
1. Check for `data/rft.json`
2. Generate it if missing (takes a few minutes)
3. Train the SFT model on the generated dataset
4. Test the model on validation set

No manual intervention needed!
