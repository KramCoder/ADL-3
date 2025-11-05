from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, sequence: torch.Tensor):
        super().__init__()
        if sequence.ndim != 1:
            raise ValueError("Stop sequence tensor must be 1-dimensional")
        self.sequence = sequence
        self.sequence_length = sequence.numel()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.sequence_length == 0 or input_ids.shape[1] < self.sequence_length:
            return False

        target = self.sequence.to(input_ids.device)
        recent_tokens = input_ids[:, -self.sequence_length :]
        return bool(torch.any(torch.all(recent_tokens == target.unsqueeze(0), dim=-1)))


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.model.eval()  # Set model to evaluation mode for better performance
        self.device = device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.max_new_tokens = 16

        stop_sequence = torch.tensor(
            self.tokenizer.encode("</answer>", add_special_tokens=False), dtype=torch.long
        )
        self._stop_sequence = stop_sequence

        # Warm up the model with a dummy generation to ensure consistent performance
        with torch.inference_mode():
            dummy_messages = self.format_prompt("1+1")
            dummy_prompt = self._render_prompt(dummy_messages)
            dummy_input = self.tokenizer(dummy_prompt, return_tensors="pt").to(device)
            _ = self.model.generate(
                **dummy_input,
                max_new_tokens=1,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        question = question.strip()
        system_message = (
            "You are a precise unit conversion assistant."
            " Return the numeric result using <answer> and </answer> tags without extra text."
        )

        return [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": (
                    f"{question}\n\nRespond only with the converted value inside <answer></answer> tags."
                ),
            },
        ]

    def _render_prompt(self, prompt):
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        raise TypeError("Prompt must be a string or a list of chat messages")

    def _build_generation_kwargs(self, *, num_return_sequences: int | None, temperature: float) -> dict:
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
            "stopping_criteria": StoppingCriteriaList(
                [StopOnTokenSequence(self._stop_sequence)]
            ),
        }

        if temperature and temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generation_kwargs["do_sample"] = False

        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences

        return generation_kwargs

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
        rendered_prompt = self._render_prompt(self.format_prompt(prompt))
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **self._build_generation_kwargs(num_return_sequences=None, temperature=0.0),
            )

        # Decode only the generated tokens (exclude input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_length:]

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

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

        formatted_prompts = [self._render_prompt(self.format_prompt(prompt)) for prompt in prompts]

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(self.device)

        generation_kwargs = self._build_generation_kwargs(
            num_return_sequences=num_return_sequences, temperature=temperature
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs,
            )

        self.tokenizer.padding_side = previous_padding_side

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]

        generations = [
            text.strip() for text in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        ]

        if num_return_sequences is None:
            return generations

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
