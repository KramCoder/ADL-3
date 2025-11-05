from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Load model with optimizations
        load_kwargs = {"dtype": torch.float16 if device == "cuda" else torch.float32}
        if device == "cuda":
            # Use memory-efficient attention if available
            load_kwargs["attn_implementation"] = "sdpa"  # Scaled Dot Product Attention
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs).to(device)
        except Exception:
            # Fallback if sdpa not available
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs).to(device)
        
        self.model.eval()  # Set model to evaluation mode for better performance
        self.device = device
        
        # Set up pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Additional optimizations for faster inference
        if device == "cuda":
            # Enable cudnn benchmarking for faster inference
            torch.backends.cudnn.benchmark = True
        
        # Warm up the model with a dummy generation to ensure consistent performance
        with torch.inference_mode():
            dummy_input = self.tokenizer("test", return_tensors="pt").to(device)
            _ = self.model.generate(**dummy_input, max_new_tokens=1, do_sample=False)

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        # Use format_prompt to handle prompt formatting
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize the formatted prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Set up pad token
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        # Generate with inference mode for maximum speed
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,  # Short answers for unit conversion (e.g., "<answer>6000</answer>")
                min_new_tokens=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=False,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding for speed
            )
        
        # Decode only the generated tokens (exclude input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_length:]
        
        # Decode and return
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        # Use format_prompt to handle prompt formatting for each prompt
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        
        # Set padding side to left for proper alignment during generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        # Tokenize all formatted prompts with padding
        inputs = self.tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(self.device)

        # Set up generation parameters
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs = {
            "max_new_tokens": 10,  # Short answers for unit conversion (e.g., "<answer>6000</answer>")
            "min_new_tokens": 1,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "use_cache": True,  # Enable KV cache for faster generation
            "num_beams": 1,  # Greedy decoding for speed
        }

        # Handle sampling vs greedy decoding
        if temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
            })
        else:
            generation_kwargs["do_sample"] = False
            
        # Handle multiple return sequences
        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences
        
        # Generate responses with inference mode for maximum speed
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )

        # Decode only the generated tokens (exclude input tokens). All prompts have been
        # left padded to the same length, so slicing with the maximum input length works
        # for each sequence individually.
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        # Decode the generated tokens
        if num_return_sequences is None:
            # Single generation per prompt
            generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return generations
        else:
            # Multiple generations per prompt - reshape the output
            batch_size = len(prompts)
            generations = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Reshape to [batch_size, num_return_sequences]
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
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
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
