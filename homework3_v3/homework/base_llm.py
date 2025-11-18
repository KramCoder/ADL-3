import os
import re
import warnings
from typing import overload

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*computation placer.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint, use_fp32_for_training=False, use_fp32_for_inference=None):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        if use_fp32_for_training:
            load_kwargs = {"torch_dtype": torch.float32}
        elif use_fp32_for_inference is not None:
            load_kwargs = {"torch_dtype": torch.float32 if use_fp32_for_inference else (torch.float16 if device == "cuda" else torch.float32)}
        else:
            import os
            use_fp16 = os.environ.get("USE_FP16_INFERENCE", "0").lower() in ("1", "true", "yes")
            if use_fp16 and device == "cuda":
                load_kwargs = {"torch_dtype": torch.float16}
            elif device == "cuda" and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                load_kwargs = {"torch_dtype": torch.bfloat16}
            else:
                load_kwargs = {"torch_dtype": torch.float32}
        
        if device == "cuda" and load_kwargs.get("torch_dtype") != torch.float32:
            load_kwargs["attn_implementation"] = "sdpa"
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs).to(device)
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs).to(device)
        
        self.model.eval()
        self.device = device
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        
        with torch.inference_mode():
            dummy_input = self.tokenizer("test", return_tensors="pt").to(device)
            _ = self.model.generate(**dummy_input, max_new_tokens=1, do_sample=False)

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        
        For SFT/RFT training, we need to match the format seen during training:
        Training format: "question " → model generates "reasoning <answer>value</answer>"
        Inference format: "question " (model generates reasoning + answer)
        """
        return f"{question.strip()} "

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            if "<answer>" not in answer:
                return 0.0
            
            after_tag = answer.split("<answer>")[1]
            
            if "</answer>" in after_tag:
                value_str = after_tag.split("</answer>")[0]
            else:
                match = re.search(r'-?\d+\.?\d*', after_tag)
                if match:
                    value_str = match.group(0)
                else:
                    return 0.0
            
            parsed = float(value_str)
            if not (parsed == parsed):
                return 0.0
            if abs(parsed) == float('inf'):
                return 0.0
            return parsed
        except (IndexError, ValueError, AttributeError):
            return 0.0

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        formatted_prompt = self.format_prompt(prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                min_new_tokens=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=False,
                use_cache=True,
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_length:]
        
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        decoded_stripped = decoded.strip()
        if not decoded_stripped:
            decoded_stripped = " 0"
        
        test_tokens = self.tokenizer(decoded_stripped, return_tensors="pt", add_special_tokens=False, padding=False)
        if test_tokens["input_ids"].shape[1] < 2:
            decoded_stripped = decoded_stripped + " 0"
        
        return decoded_stripped

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        micro_batch_size = 128 if torch.cuda.is_available() else 32
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
            "max_new_tokens": 100,
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

        generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        validated_generations = []
        for gen in generations:
            gen_stripped = gen.strip()
            if not gen_stripped:
                gen_stripped = " 0"
            
            test_tokens = self.tokenizer(gen_stripped, return_tensors="pt", add_special_tokens=False, padding=False)
            if test_tokens["input_ids"].shape[1] < 2:
                gen_stripped = gen_stripped + " 0"
            
            validated_generations.append(gen_stripped)
        
        generations = validated_generations
        
        if num_return_sequences is None:
            return generations
        else:
            batch_size = len(prompts)
            
            reshaped_generations = []
            for i in range(batch_size):
                start_idx = i * num_return_sequences
                end_idx = start_idx + num_return_sequences
                reshaped_generations.append(generations[start_idx:end_idx])
            
            return reshaped_generations

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        generations = self.batched_generate(list(questions))
        return [self.parse_answer(g) for g in generations]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
