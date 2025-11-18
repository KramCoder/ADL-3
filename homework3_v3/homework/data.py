import json
from dataclasses import dataclass
from pathlib import Path

from .base_llm import BaseLLM

DATA_DIR = Path(__file__).parent.parent / "data"


class Dataset:
    def __init__(self, split: str):
        with (DATA_DIR / f"{split}.json").open() as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def is_answer_valid(answer: float, correct_answer: float, relative_tolerance: float = 0.05) -> bool:
    # Handle NaN and Inf values - they are never valid answers
    # Check if answer is NaN (NaN != NaN is True)
    if answer != answer or abs(answer) == float('inf'):
        return False
    # Check if correct_answer is NaN or Inf (shouldn't happen, but be defensive)
    if correct_answer != correct_answer or abs(correct_answer) == float('inf'):
        return False
    return abs(round(answer, 3) - round(correct_answer, 3)) < relative_tolerance * abs(round(correct_answer, 3))


@dataclass
class BenchmarkResult:
    @dataclass
    class Sample:
        question: str
        answer: float
        correct_answer: float
        is_correct: bool

    accuracy: float
    answer_rate: float  # How often was the answer not NaN
    samples: list[Sample]

    @classmethod
    def from_answers(cls, answers: list[float], dataset: Dataset, max_question: int) -> "BenchmarkResult":
        samples = [
            cls.Sample(
                question=item[0], answer=answer, correct_answer=item[1], is_correct=is_answer_valid(answer, item[1])
            )
            for item, answer in zip(dataset, answers[:max_question])
        ]
        n = min(len(dataset), max_question)
        # Prevent division by zero - return 0.0 for accuracy and answer_rate if n is 0
        # This ensures the grader never receives NaN values
        if n == 0:
            return cls(
                accuracy=0.0,
                answer_rate=0.0,
                samples=samples,
            )
        
        # Calculate accuracy and answer_rate with NaN protection
        # The grader cannot process NaN values, so we must ensure these are always finite
        correct_count = sum(sample.is_correct for sample in samples)
        accuracy = correct_count / n if n > 0 else 0.0
        
        # Validate accuracy is not NaN or Inf (defensive check)
        if accuracy != accuracy or abs(accuracy) == float('inf'):  # NaN or Inf check
            accuracy = 0.0
        
        valid_answer_count = sum(sample.answer == sample.answer for sample in samples)  # Count non-NaN answers
        answer_rate = valid_answer_count / n if n > 0 else 0.0
        
        # Validate answer_rate is not NaN or Inf (defensive check)
        if answer_rate != answer_rate or abs(answer_rate) == float('inf'):  # NaN or Inf check
            answer_rate = 0.0
        
        return cls(
            accuracy=accuracy,
            answer_rate=answer_rate,
            samples=samples,
        )


def benchmark(func: BaseLLM, dataset: Dataset, max_question: int) -> BenchmarkResult:
    idx = range(min(len(dataset), max_question))
    questions = [dataset[i][0] for i in idx]
    answers = func.answer(*questions)
    return BenchmarkResult.from_answers(answers, dataset, max_question)


if __name__ == "__main__":
    print(Dataset("train")[0])
