"""Training sequences must equal what vLLM renders at serving time, plus label + stop token."""

import pytest

transformers = pytest.importorskip("transformers")

from dmi.train.tokenization import SampleTokenizer  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

# Gemma 4's shape: a finished assistant turn renders without the empty thought
# channel that the generation prompt adds when thinking is off.
GEMMA_LIKE_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think></think>' }}{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        tok = transformers.AutoTokenizer.from_pretrained(TINY)
    except OSError as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")
    tok.chat_template = GEMMA_LIKE_TEMPLATE
    return tok


def test_sequence_is_generation_prompt_then_label_then_stop(tokenizer):
    st = SampleTokenizer(tokenizer, TINY)
    prompt = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    out = st(prompt, '{"a":1}')

    serving_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    text = tokenizer.decode(out["input_ids"])
    assert text == serving_prompt + '{"a":1}<|im_end|>'

    loss_ids = [t for t, m in zip(out["input_ids"], out["completion_mask"]) if m]
    assert tokenizer.decode(loss_ids) == '{"a":1}<|im_end|>'
    assert out["completion_mask"] == sorted(out["completion_mask"])  # prompt masked, then one loss span
    assert st.end_ids == [tokenizer.convert_tokens_to_ids("<|im_end|>")]
