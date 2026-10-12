"""Training sequences built the way vLLM serves the model.

At serving time vLLM renders `apply_chat_template(prompt, add_generation_prompt=True)`
and the model generates the label, then the template's end-of-turn token (a stop
token). Training on a render of the whole conversation is not equivalent for
every template: Gemma 4 with thinking off puts an empty thought channel
(`<|channel>thought\\n<channel|>`) in the generation prompt but not in a finished
assistant turn. So each sequence here is

    generation prompt  +  label tokens  +  end-of-turn token
    (no loss)             (loss)           (loss: the model learns to stop)

Imported inside the training container; depends only on transformers.
"""

from __future__ import annotations

_MARKER = "⁣dmi-label⁣"


def stop_token_ids(model_id: str, tokenizer) -> set[int]:
    """Token ids that end generation: generation_config.eos_token_id plus the tokenizer's EOS."""
    from transformers import GenerationConfig

    ids: set[int] = set()
    try:
        eos = GenerationConfig.from_pretrained(model_id).eos_token_id
    except OSError:
        eos = None
    if eos is not None:
        ids.update(eos if isinstance(eos, list) else [eos])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def end_of_turn_ids(tokenizer, stop_ids: set[int]) -> list[int]:
    """Tokens the template places after an assistant message, up to and including the first stop token."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": _MARKER}], tokenize=False
    )
    suffix = rendered.split(_MARKER, 1)[1]
    ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    for i, token in enumerate(ids):
        if token in stop_ids:
            return ids[: i + 1]
    raise ValueError(f"No stop token in the template's assistant-turn suffix {suffix!r}")


class SampleTokenizer:
    def __init__(self, tokenizer, model_id: str):
        self.tokenizer = tokenizer
        self.end_ids = end_of_turn_ids(tokenizer, stop_token_ids(model_id, tokenizer))

    def prompt_ids(self, prompt: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        # The rendered template already contains BOS/special tokens, as in vLLM's chat path
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def completion_ids(self, label_text: str) -> list[int]:
        return self.tokenizer(label_text, add_special_tokens=False)["input_ids"] + self.end_ids

    def __call__(self, prompt: list[dict], label_text: str) -> dict[str, list[int]]:
        prompt_ids = self.prompt_ids(prompt)
        completion_ids = self.completion_ids(label_text)
        return {
            "input_ids": prompt_ids + completion_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
        }
