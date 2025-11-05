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
                    "You are a helpful unit conversion assistant. You solve unit conversion problems step by step. "
                    "First, identify the conversion factor, then perform the calculation, and finally provide the "
                    "numeric answer inside <answer></answer> tags. Be concise and precise."
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "Use 1 kg = 1000 g. So 6 kg = 6 * 1000 = 6000 g. <answer>6000</answer>",
            },
            {
                "role": "user",
                "content": "What is the conversion of 2 hour to seconds?",
            },
            {
                "role": "assistant",
                "content": "Use 1 hour = 3600 seconds. So 2 hours = 2 * 3600 = 7200 seconds. <answer>7200</answer>",
            },
            {
                "role": "user",
                "content": "Convert 5 km to meter?",
            },
            {
                "role": "assistant",
                "content": "Use 1 km = 1000 m. So 5 km = 5 * 1000 = 5000 m. <answer>5000</answer>",
            },
            {
                "role": "user",
                "content": question,
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
