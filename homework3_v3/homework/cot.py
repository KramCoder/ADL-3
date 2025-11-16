import os
import sys
import warnings

# Suppress CUDA/TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*computation placer.*")
# Suppress RuntimeWarning about module import order
warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch


class CoTModel(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Removed apply_dataset_answer_patch to actually test the LLM

    def generate(self, prompt: str) -> str:
        """Override to allow more tokens for CoT reasoning"""
        formatted_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        import torch
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=80,  # Increased for CoT reasoning
                min_new_tokens=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=False,
                use_cache=True,
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """Override to allow more tokens for CoT reasoning"""
        from tqdm import tqdm
        import torch
        
        # Preventing OOM
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs = {
            "max_new_tokens": 100,  # Increased for CoT reasoning
            "min_new_tokens": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }

        if temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
            })
        else:
            generation_kwargs["do_sample"] = False
            
        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        if num_return_sequences is None:
            generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return generations
        else:
            batch_size = len(prompts)
            generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            reshaped_generations = []
            for i in range(batch_size):
                start_idx = i * num_return_sequences
                end_idx = start_idx + num_return_sequences
                reshaped_generations.append(generations[start_idx:end_idx])
            return reshaped_generations

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
                    "You are an expert at unit conversions. "
                    "For each question, identify the conversion factor between the units, "
                    "perform the calculation step by step, and provide your final answer "
                    "inside <answer></answer> tags. Be precise with your calculations and "
                    "ensure the answer is a number (not text)."
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g. To convert 6 kg to grams: 6 * 1000 = 6000. <answer>6000</answer>",
            },
            {
                "role": "user",
                "content": "What is the measurement of 3 kg when converted into pound?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 2.2046226218487757 pounds. To convert 3 kg to pounds: 3 * 2.2046226218487757 = 6.613867865546327. <answer>6.613867865546327</answer>",
            },
            {
                "role": "user",
                "content": "How many MB is 2 G?",
            },
            {
                "role": "assistant",
                "content": "1 G (gigabyte) = 1000 MB. To convert 2 G to MB: 2 * 1000 = 2000. <answer>2000</answer>",
            },
            {
                "role": "user",
                "content": "How does 4 years measure up in terms of week?",
            },
            {
                "role": "assistant",
                "content": "1 year = 365.24219878125 days, and 1 week = 7 days. So 1 year = 365.24219878125 / 7 = 52.17745696875 weeks. To convert 4 years to weeks: 4 * 52.17745696875 = 208.709827875. <answer>208.709827875</answer>",
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


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
