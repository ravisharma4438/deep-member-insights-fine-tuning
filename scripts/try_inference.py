"""Quick check that a trained model produces usable screens. Not the eval (see dmi.eval).

    modal run scripts/try_inference.py --job <training-job-name>
    modal run scripts/try_inference.py --job <training-job-name> --n 5 --show

Describes the SageMaker job (with your AWS credentials) to find its model and
dataset, then on a Modal GPU loads merged/ if the job merged, else the base
model + adapter/ (a smoke run), and generates for a few test-split samples
with plain transformers. Decoding is greedy and unconstrained (no JSON schema),
so a model that hasn't learned the format shows up as invalid output. Prints
each output scored against its label; saves them to outputs/try_inference/.

The Modal side reads S3 through the `aws-secret` Modal secret, like
serving/load_model.py. Model files are cached in the Modal volume
`dmi-try-cache`, so reruns on the same job skip the download.
GPU: $DMI_TRY_GPU (default H100).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import modal

AWS_SECRET = os.environ.get("DMI_MODAL_AWS_SECRET", "aws-secret")
GPU = os.environ.get("DMI_TRY_GPU", "H100")

app = modal.App("dmi-try-inference")
cache = modal.Volume.from_name("dmi-try-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # Same library versions as training (src/dmi/train/requirements.txt)
    .uv_pip_install("torch==2.10.0", "transformers==5.18.0", "peft==0.21.2", "accelerate==1.15.0", "boto3")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("dmi")
)


def _fetch(model_s3: str) -> Path:
    """Download run_info.json and merged/ (or adapter/ if the job didn't merge) into the cache volume."""
    from concurrent.futures import ThreadPoolExecutor

    import boto3

    bucket, _, prefix = model_s3.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    dest = Path("/cache/models") / bucket / prefix
    if (dest / ".complete").exists():
        print(f"Using cached copy of {model_s3}", flush=True)
        return dest

    s3 = boto3.client("s3")
    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if not obj["Key"].endswith("/")
    ]
    rel = [key[len(prefix):] for key in keys]
    subdir = next((d for d in ("merged/", "adapter/") if any(r.startswith(d) for r in rel)), None)
    if subdir is None:
        raise SystemExit(f"No merged/ or adapter/ under {model_s3}; pass --model-s3 <.../output/model/>.")
    wanted = [key for key, r in zip(keys, rel) if r.startswith(subdir) or r == "run_info.json"]
    print(f"Downloading {len(wanted)} files ({subdir}) from {model_s3}...", flush=True)

    def download(key: str) -> None:
        target = dest / key[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, wanted))
    (dest / ".complete").touch()
    cache.commit()
    return dest


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name(AWS_SECRET)],
    timeout=3600,
)
def generate(model_s3: str, prompts: list[list[dict]], max_new_tokens: int) -> list[dict]:
    import torch
    import transformers
    from peft import PeftModel

    from dmi.train.tokenization import SampleTokenizer, stop_token_ids

    local = _fetch(model_s3)
    run_info_path = local / "run_info.json"
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    model_cls = getattr(transformers, run_info.get("base_class") or "AutoModelForCausalLM")

    if (local / "merged" / "config.json").exists():
        source = str(local / "merged")
        print(f"Loading merged model with {model_cls.__name__}...", flush=True)
        model = model_cls.from_pretrained(source, dtype=torch.bfloat16, device_map="cuda")
    else:
        adapter_dir = local / "adapter"
        source = json.loads((adapter_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
        print(f"No merged/ (smoke run?): loading {source} + adapter with {model_cls.__name__}...", flush=True)
        base = model_cls.from_pretrained(source, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        cache.commit()  # keep the base model download for next time
    model.eval()

    # Prompt tokens exactly as in training / vLLM serving: the generation prompt
    tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    sample_tokenizer = SampleTokenizer(tokenizer, source)
    stop_ids = sorted(stop_token_ids(source, tokenizer))

    results = []
    for i, prompt in enumerate(prompts, 1):
        input_ids = torch.tensor([sample_tokenizer.prompt_ids(prompt)], device=model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        seconds = time.perf_counter() - start
        new_ids = output[0, input_ids.shape[1]:].tolist()
        stopped = bool(new_ids) and new_ids[-1] in stop_ids
        results.append({
            "text": tokenizer.decode(new_ids[:-1] if stopped else new_ids),
            "prompt_tokens": input_ids.shape[1],
            "new_tokens": len(new_ids),
            "stopped": stopped,
            "seconds": round(seconds, 1),
        })
        print(f"  [{i}/{len(prompts)}] {input_ids.shape[1]} prompt tokens -> "
              f"{len(new_ids)} tokens in {seconds:.0f}s", flush=True)
    return results


def _describe(job: str) -> tuple[str, str]:
    """The job's model artifacts prefix and the name of the dataset it trained on."""
    import boto3

    desc = boto3.client("sagemaker").describe_training_job(TrainingJobName=job)
    if desc["TrainingJobStatus"] != "Completed":
        print(f"Note: {job} is {desc['TrainingJobStatus']}")
    artifacts = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")
    if artifacts.endswith(".tar.gz"):  # uncompressed output lands in the sibling model/ prefix
        artifacts = artifacts.rsplit("/", 1)[0] + "/model/"
    channel = next(c for c in desc["InputDataConfig"] if c["ChannelName"] == "training")
    dataset = channel["DataSource"]["S3DataSource"]["S3Uri"].rstrip("/").rsplit("/", 1)[-1]
    return artifacts, dataset


@app.local_entrypoint()
def main(
    job: str = "",
    model_s3: str = "",
    dataset: str = "",
    n: int = 3,
    seed: int = 0,
    max_new_tokens: int = 4096,
    show: bool = False,
) -> None:
    from dmi import settings
    from dmi.eval import metrics
    from dmi.eval.predict import load_split
    from dmi.screening.labels import validation_error

    if job:
        described_model, described_dataset = _describe(job)
        model_s3, dataset = model_s3 or described_model, dataset or described_dataset
    if not model_s3 or not dataset:
        raise SystemExit("Pass --job <training-job-name>, or --model-s3 <.../output/model/> and --dataset <name>.")

    examples = load_split(dataset, "test")
    picked = random.Random(seed).sample(examples, min(n, len(examples)))
    print(f"Model:   {model_s3}\nSamples: {len(picked)} from {dataset}/test.jsonl (seed {seed}), GPU {GPU}")
    results = generate.remote(model_s3, [e["prompt"] for e in picked], max_new_tokens)

    rows = []
    for i, (example, result) in enumerate(zip(picked, results), 1):
        label = json.loads(example["completion"][0]["content"])
        prediction = metrics.parse_prediction(result["text"])
        row = metrics.score(prediction, label)
        fields = [v for k, v in row.items() if k.startswith(("acc_", "num_", "jac_"))]
        agreement = sum(fields) / len(fields)
        overall = prediction.get("scoreOverall") if prediction else None
        pred_score = overall.get("score") if isinstance(overall, dict) else None
        label_score = label["scoreOverall"]["score"]
        rows.append({"row": row, "stopped": result["stopped"], "agreement": agreement})

        print(f"\n[{i}] {example['id'][:12]} ({example.get('provider')})  "
              f"{result['prompt_tokens']} prompt tokens -> {result['new_tokens']} tokens, "
              f"{result['seconds']}s ({result['new_tokens'] / max(result['seconds'], 1e-9):.1f} tok/s)")
        print(f"    stopped: {'yes' if result['stopped'] else 'NO (hit max tokens)'}  "
              f"json: {'ok' if prediction is not None else 'INVALID'}  "
              f"schema: {'ok' if row['schema_valid'] else 'INVALID'}  "
              f"tier: {row['pred_tier']} (label {row['label_tier']})  "
              f"scoreOverall: {pred_score} (label {label_score})  "
              f"field agreement: {agreement:.0%}")
        if prediction is not None and not row["schema_valid"]:
            print(f"    schema error: {validation_error(prediction)}")
        if show or prediction is None:
            body = json.dumps(prediction, indent=2, ensure_ascii=False) if prediction else result["text"][:3000]
            print("    " + body.replace("\n", "\n    "))

    count = lambda key: sum(1 for r in rows if r["row"][key])  # noqa: E731
    print(f"\n{len(rows)} samples: stopped {sum(r['stopped'] for r in rows)}, valid JSON {count('valid_json')}, "
          f"schema-valid {count('schema_valid')}, tier match {count('tier')}, "
          f"mean field agreement {sum(r['agreement'] for r in rows) / max(len(rows), 1):.0%}")

    out = settings.REPO_ROOT / "outputs" / "try_inference" / f"{job or 'run'}-{int(time.time())}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for example, result in zip(picked, results):
            f.write(json.dumps({"id": example["id"], "provider": example.get("provider"), **result,
                                "label": example["completion"][0]["content"]}, ensure_ascii=False) + "\n")
    print(f"Outputs saved to {out}")
