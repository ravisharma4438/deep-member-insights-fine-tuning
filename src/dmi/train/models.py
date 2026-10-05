"""Fine-tuning targets. Adding a model = adding an entry here.

Imported both by the Studio-side launcher and inside the training container,
so keep it dependency-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    # Samples longer than this (prompt + label, in this model's tokens) are dropped,
    # never truncated: truncation would cut the JSON label.
    max_seq_len: int
    # LoRA. target_modules=None uses PEFT's per-architecture default.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] | None = "all-linear"
    load_in_4bit: bool = False
    # Optimization (effective batch = gpus * per_device * grad_accum)
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    # SageMaker
    instance_type: str = "ml.p4d.24xlarge"
    gpus_per_instance: int = 8
    volume_size_gb: int = 300
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="gemma-4-12b",
            hf_id="google/gemma-4-12B-it",
            max_seq_len=12288,
            # Gemma4UnifiedForConditionalGeneration (model_type gemma4_unified): PEFT has no
            # default targets for it, and "all-linear" would also hit the vision/audio
            # embedders. Every attention + MLP projection of the 48 text layers.
            lora_target_modules=(
                r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
            ),
            notes=(
                "11.96B dense, Apache-2.0, not gated. Needs transformers>=5.10 (gemma4_unified; "
                ">=5.5.2 for the KV-sharing training fix) and vLLM>=0.23 to serve. bf16 weights "
                "~24 GB. Chat template adds an empty thought channel to the generation prompt "
                "when thinking is off; dmi.train.tokenization trains on that exact prompt."
            ),
        ),
        ModelSpec(
            key="llama-3.1-8b",
            hf_id="meta-llama/Llama-3.1-8B-Instruct",
            max_seq_len=12288,
            notes="Previous production fine-tune base; kept as a like-for-like baseline.",
        ),
    )
}


def get(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise SystemExit(f"Unknown model {key!r}. Known: {', '.join(MODELS)}") from None
