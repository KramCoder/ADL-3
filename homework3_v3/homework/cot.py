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
                    "You are a precise but friendly unit-conversion tutor."
                    " For each question: identify the unit relationship, show the key calculation"
                    " in one short sentence, and finish with the numeric result wrapped in"
                    " <answer> tags. Keep the whole reply concise."
                ),
            },
            {
                "role": "user",
                "content": "Question: Convert 3 miles to feet.",
            },
            {
                "role": "assistant",
                "content": (
                    "1 mile = 5280 feet, so 3 * 5280 = 15840. <answer>15840</answer>"
                ),
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
