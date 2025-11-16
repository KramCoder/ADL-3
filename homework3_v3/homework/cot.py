import os
import warnings

# Suppress CUDA warnings by setting environment variables before importing torch
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=1")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*computation placer.*")

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch


class CoTModel(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Removed apply_dataset_answer_patch to actually test the LLM

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        question = question.strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You solve unit conversions concisely. "
                    "State the conversion factor, calculate, then give the answer in <answer></answer> tags."
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g, so 6 * 1000 = 6000. <answer>6000</answer>",
            },
            {
                "role": "user",
                "content": question,  # Add the actual question here
            },
        ]

        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


def main():
    """Main entry point to avoid RuntimeWarning when running as module"""
    from fire import Fire
    Fire({"test": test_model, "load": load})


if __name__ == "__main__":
    main()
