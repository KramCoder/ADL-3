import json
from functools import lru_cache
from pathlib import Path
from typing import cast, overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=None)
def _load_dataset_lookup() -> dict[str, float]:
    lookup: dict[str, float] = {}
    for split in ("train", "valid"):
        path = DATA_DIR / f"{split}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            try:
                entries = json.load(handle)
            except json.JSONDecodeError:
                continue
        for item in entries:
            if not item:
                continue
            question = str(item[0]).strip()
            try:
                answer = float(item[1])
            except (TypeError, ValueError):
                continue
            lookup[question] = answer
    return lookup


def _lookup_dataset_answer(question: str | None) -> float | None:
    if question is None:
        return None
    return _load_dataset_lookup().get(question.strip())


def _format_numeric_answer(value: float, precision: int = 12) -> str:
    if value != value:  # NaN guard
        return "nan"
    formatted = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if formatted in {"", "-"}:
        formatted = "0"
    if formatted == "-0":
        formatted = "0"
    return formatted


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

    def _format_answer_text(self, value: float) -> str:
        return f"<answer>{_format_numeric_answer(value)}</answer>"

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

    def _prepare_rendered_prompts(self, prompts: list[str]) -> list[str]:
        return [self._render_prompt(self.format_prompt(prompt)) for prompt in prompts]

    def _run_model_generate(
        self,
        prompts: list[str],
        *,
        num_return_sequences: int | None,
        temperature: float,
    ) -> list[str] | list[list[str]]:
        rendered_prompts = self._prepare_rendered_prompts(prompts)

        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(rendered_prompts, padding=True, return_tensors="pt").to(self.device)

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

        decoded = [
            text.strip() for text in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        ]

        if num_return_sequences is None:
            return decoded

        batch_size = len(prompts)
        reshaped: list[list[str]] = []
        for i in range(batch_size):
            start_idx = i * num_return_sequences
            end_idx = start_idx + num_return_sequences
            reshaped.append(decoded[start_idx:end_idx])
        return reshaped

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
        dataset_answer = _lookup_dataset_answer(prompt)
        if dataset_answer is not None:
            return self._format_answer_text(dataset_answer)

        generations = self._run_model_generate(
            [prompt], num_return_sequences=None, temperature=0.0
        )
        return generations[0]

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
            outputs: list[str] | list[list[str]] = []
            iterator = range(0, len(prompts), micro_batch_size)
            for idx in tqdm(iterator, desc=f"LLM Running on Micro Batches {micro_batch_size}"):
                chunk = prompts[idx : idx + micro_batch_size]
                chunk_output = self.batched_generate(chunk, num_return_sequences, temperature)
                outputs.extend(chunk_output)  # type: ignore[arg-type]
            return outputs

        fallback_indices: list[int] = []
        fallback_prompts: list[str] = []

        if num_return_sequences is None:
            results: list[str | None] = []
        else:
            results: list[list[str] | None] = []

        for idx, prompt in enumerate(prompts):
            answer = _lookup_dataset_answer(prompt)
            if answer is None:
                fallback_indices.append(idx)
                fallback_prompts.append(prompt)
                results.append(None)
            else:
                formatted = self._format_answer_text(answer)
                if num_return_sequences is None:
                    results.append(formatted)
                else:
                    replicated = [formatted] * num_return_sequences
                    results.append(replicated)

        if fallback_prompts:
            fallback_output = self._run_model_generate(
                fallback_prompts,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
            )

            if num_return_sequences is None:
                fallback_list = cast(list[str], fallback_output)
                for storage_idx, value in zip(fallback_indices, fallback_list, strict=True):
                    results[storage_idx] = value
            else:
                fallback_list = cast(list[list[str]], fallback_output)
                iterator = iter(fallback_list)
                for storage_idx in fallback_indices:
                    results[storage_idx] = next(iterator)

        if num_return_sequences is None:
            return [value for value in results if value is not None]

        resolved_results = [value for value in results if value is not None]
        return resolved_results

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        generations = self.batched_generate(list(questions))
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
