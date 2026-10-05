"""LoRA SFT entry point. Runs inside the SageMaker training container under torchrun.

Launched by dmi.train.launch; hyperparameters arrive as `--key value` flags.
Inputs:  /opt/ml/input/data/training/{train,val}.jsonl  (dmi.data.build_dataset)
Outputs: /opt/ml/model/adapter/   LoRA adapter
         /opt/ml/model/merged/    base + adapter merged, ready for vLLM
         /opt/ml/model/run_info.json

Loss is on the label only (TRL prompt-completion format), and the completion
ends with the chat template's end-of-turn token, so the model learns to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# SageMaker runs this file from the uploaded source dir (src/); make `dmi` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dmi.train import models  # noqa: E402

SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
SM_TRAIN_CHANNEL = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
# Weight files the merge writes itself; everything else (tokenizer, chat template,
# processor configs) is copied from the base model so vLLM loads it like the original.
_WEIGHT_FILES = ("*.safetensors", "*.safetensors.index.json", "*.bin", "*.pt", "*.pth", "*.gguf")


def _bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True)
    p.add_argument("--data_dir", type=Path, default=SM_TRAIN_CHANNEL)
    p.add_argument("--output_dir", type=Path, default=SM_MODEL_DIR)
    p.add_argument("--max_seq_len", type=int)
    p.add_argument("--num_train_epochs", type=float)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--lora_r", type=int)
    p.add_argument("--max_train_samples", type=int, help="cap for smoke tests")
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--merge", type=_bool, default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = models.get(args.model_key)
    max_seq_len = args.max_seq_len or spec.max_seq_len

    import torch
    from accelerate import PartialState
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    state = PartialState()
    log = (lambda *a: print(*a, flush=True)) if state.is_main_process else (lambda *a: None)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

    data_files = {"train": str(args.data_dir / "train.jsonl"), "val": str(args.data_dir / "val.jsonl")}
    raw = load_dataset("json", data_files=data_files)
    keep = ["prompt", "completion"]
    raw = raw.remove_columns([c for c in raw["train"].column_names if c not in keep])

    def n_tokens(example: dict) -> dict:
        text = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)
        return {"n_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"])}

    # Rank 0 tokenizes and caches; other ranks reuse the cache.
    with state.main_process_first():
        measured = raw.map(n_tokens, num_proc=min(16, os.cpu_count() or 1), desc="Measuring lengths")
        fitting = measured.filter(lambda n: n <= max_seq_len, input_columns="n_tokens")
    dropped = {split: len(measured[split]) - len(fitting[split]) for split in measured}
    log(f"max_seq_len={max_seq_len}: kept {dict((s, len(fitting[s])) for s in fitting)}, dropped {dropped}")

    train_ds = fitting["train"].remove_columns("n_tokens").shuffle(seed=args.seed)
    eval_ds = fitting["val"].remove_columns("n_tokens").shuffle(seed=args.seed)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    peft_config = LoraConfig(
        r=args.lora_r or spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=spec.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if spec.load_in_4bit
        else None
    )

    work_dir = Path("/tmp/sft")
    config = SFTConfig(
        output_dir=str(work_dir),
        max_length=max_seq_len,
        packing=False,
        num_train_epochs=args.num_train_epochs or spec.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=spec.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps or spec.gradient_accumulation_steps,
        learning_rate=args.learning_rate or spec.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        logging_steps=10,
        eval_strategy="steps" if len(eval_ds) else "no",
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
        dataset_num_proc=min(16, os.cpu_count() or 1),
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=spec.hf_id,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if len(eval_ds) else {}

    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    base_cls = type(trainer.model.get_base_model())

    if state.is_main_process:
        tokenizer.save_pretrained(adapter_dir)
        run_info = {
            "model": spec.to_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "max_seq_len": max_seq_len,
            "samples": {"train": len(train_ds), "eval": len(eval_ds), "dropped_too_long": dropped},
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "base_class": base_cls.__name__,
        }
        manifest = args.data_dir / "manifest.json"
        if manifest.exists():
            run_info["dataset"] = json.loads(manifest.read_text())
        (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))

    del trainer
    torch.cuda.empty_cache()
    state.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    if args.merge and state.is_main_process:
        merge(spec.hf_id, base_cls, adapter_dir, args.output_dir / "merged")


def merge(base_id: str, base_cls, adapter_dir: Path, merged_dir: Path) -> None:
    """Merge the adapter into bf16 base weights (vLLM can't load Gemma 4 LoRAs at runtime)."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    print(f"Merging adapter into {base_id} ({base_cls.__name__}) on CPU...", flush=True)
    base = base_cls.from_pretrained(base_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")

    base_files = Path(snapshot_download(base_id, ignore_patterns=list(_WEIGHT_FILES)))
    for src in base_files.rglob("*"):
        dest = merged_dir / src.relative_to(base_files)
        if src.is_file() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    print(f"Merged model written to {merged_dir}", flush=True)


if __name__ == "__main__":
    main()
