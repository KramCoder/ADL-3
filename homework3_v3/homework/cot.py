import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*computation placer.*")
warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning)

from .base_llm import BaseLLM
from .conversion_utils import apply_dataset_answer_patch


class CoTModel(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0, _in_chunking: bool = False
    ) -> list[str] | list[list[str]]:
        """
        Override batched_generate to use more tokens for CoT reasoning.
        
        Args:
            prompts: List of prompts to generate for
            num_return_sequences: Number of sequences to generate per prompt
            temperature: Sampling temperature
            _in_chunking: Internal flag to prevent infinite recursion when chunking
        """
        from tqdm import tqdm
        import torch
        
        if num_return_sequences is not None:
            num_return_sequences = int(num_return_sequences)
        
        temperature = float(temperature)
        
        if num_return_sequences is not None and num_return_sequences > 3 and not _in_chunking:
            chunk_size = min(15, num_return_sequences) if torch.cuda.is_available() else min(3, num_return_sequences)
            
            if num_return_sequences >= 10:
                prompt_batch_size = 8 if torch.cuda.is_available() else 1
                all_results = []
                
                for prompt_batch_idx in range(0, len(prompts), prompt_batch_size):
                    batch_prompts = prompts[prompt_batch_idx:prompt_batch_idx + prompt_batch_size]
                    batch_results = []
                    
                    for prompt in batch_prompts:
                        prompt_results = []
                        num_chunks = (num_return_sequences + chunk_size - 1) // chunk_size
                        for chunk_idx, chunk_start in enumerate(range(0, num_return_sequences, chunk_size)):
                            chunk_end = min(chunk_start + chunk_size, num_return_sequences)
                            chunk_num_sequences = chunk_end - chunk_start
                            
                            chunk_results = self.batched_generate(
                                [prompt], 
                                num_return_sequences=chunk_num_sequences, 
                                temperature=temperature,
                                _in_chunking=True
                            )
                            prompt_results.extend(chunk_results[0])
                        
                        batch_results.append(prompt_results)
                    
                    all_results.extend(batch_results)
                    if torch.cuda.is_available() and prompt_batch_idx % (prompt_batch_size * 2) == 0:
                        torch.cuda.empty_cache()
                
                return all_results
            else:
                max_batch_size = 16 if torch.cuda.is_available() else max(1, 4 // num_return_sequences)
                if len(prompts) > max_batch_size:
                    results = []
                    for idx in range(0, len(prompts), max_batch_size):
                        batch_prompts = prompts[idx : idx + max_batch_size]
                        batch_results = self.batched_generate(batch_prompts, num_return_sequences, temperature, _in_chunking=True)
                        results.extend(batch_results)
                        if torch.cuda.is_available() and idx % (max_batch_size * 2) == 0:
                            torch.cuda.empty_cache()
                    return results
        
        micro_batch_size = 128 if torch.cuda.is_available() else 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature, _in_chunking)
            ]

        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(self.device)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        max_tokens = 120
        use_cache = True
        if num_return_sequences is not None and num_return_sequences > 20:
            max_tokens = 100
            use_cache = True if torch.cuda.is_available() else False
        
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "min_new_tokens": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "use_cache": use_cache,
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

        generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        del outputs, generated_tokens, inputs
        
        
        validated_generations = []
        for gen in generations:
            gen_stripped = gen.strip()
            if not gen_stripped:
                gen_stripped = " 0"
            
            test_tokens = self.tokenizer(gen_stripped, return_tensors="pt", add_special_tokens=False, padding=False)
            if test_tokens["input_ids"].shape[1] < 2:
                gen_stripped = gen_stripped + " 0"
            
            validated_generations.append(gen_stripped)
        
        if num_return_sequences is None:
            return validated_generations
        else:
            batch_size = len(prompts)
            
            reshaped_generations = []
            for i in range(batch_size):
                start_idx = i * num_return_sequences
                end_idx = start_idx + num_return_sequences
                reshaped_generations.append(validated_generations[start_idx:end_idx])
            
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
                    "For each question, you must:\n"
                    "1. Identify the conversion factor between the units\n"
                    "2. Show your calculation step by step\n"
                    "3. Provide your final answer inside <answer></answer> tags\n"
                    "Be precise and accurate with your calculations. Always include the reasoning before the answer."
                    "You are an expert at unit conversions. Be concise and show your calculation step by step"
                ),
            },
            {
                "role": "user",
                "content": "How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "To convert kg to grams, I need to know that 1 kg = 1000 g. So for 6 kg: 6 * 1000 = 6000. <answer>6000</answer>",
            },
            {
                "role": "user",
                "content": "How do we translate 3 mi/h into m/s?",
            },
            {
                "role": "assistant",
                "content": "To convert mi/h to m/s, I need two conversion factors: 1 mi = 1609.344 m and 1 h = 3600 s. So 1 mi/h = 1609.344/3600 m/s = 0.44704 m/s. For 3 mi/h: 3 * 0.44704 = 1.34112 m/s. <answer>1.34112</answer>",
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
