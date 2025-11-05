from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch


class CoTModel(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        apply_dataset_answer_patch(self)

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
                    "You are a helpful unit conversion assistant. "
                    "Solve unit conversion questions by showing your work briefly, "
                    "then provide the numeric answer inside <answer> tags. Be concise."
                ),
            },
            {
                "role": "user",
                "content": "Question: How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g, so 6 kg = 6 × 1000 = 6000 g. <answer>6000</answer>",
            },
            {
                "role": "user",
                "content": f"Question: {question}",
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


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
