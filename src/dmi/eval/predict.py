"""Run a dataset split through an OpenAI-compatible endpoint (e.g. the Modal vLLM server).

    python -m dmi.eval.predict --dataset screens-20251205-20261005 --run gemma-4-12b-v1
    python -m dmi.eval.predict --dataset <name> --run <run> --limit 50 --concurrency 16

Requests mirror the platform's generateObject call: the stored system + user
messages, temperature 0.1, and response_format json_schema with ScreenSchema.
Writes outputs/evals/<run>/predictions.jsonl (one line per sample, including
failures) for dmi.eval.score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dmi import settings
from dmi.screening.labels import screen_schema

TEMPERATURE = 0.1  # platform AI_TEMPERATURES.deterministic
MAX_TOKENS = 4096
RETRIES = 4
BACKOFF_SCALE = 1.0  # seconds multiplier; tests shrink it


def load_split(dataset: str, split: str) -> list[dict]:
    """From data/datasets/<name>/ if present locally, else from S3."""
    local = settings.REPO_ROOT / "data" / "datasets" / dataset / f"{split}.jsonl"
    if not local.exists():
        import boto3

        uri = settings.s3_uri("datasets", dataset, f"{split}.jsonl")
        bucket, _, key = uri.removeprefix("s3://").partition("/")
        local.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(local))
    with local.open() as f:
        return [json.loads(line) for line in f if line.strip()]


async def predict_one(client, model: str, example: dict, semaphore: asyncio.Semaphore) -> dict:
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    record = {
        "id": example["id"],
        "provider": example.get("provider"),
        "label": example["completion"][0]["content"],
    }
    async with semaphore:
        start = time.perf_counter()
        for attempt in range(RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=example["prompt"],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "response", "schema": screen_schema()},
                    },
                )
                choice = response.choices[0]
                record.update(
                    prediction=choice.message.content,
                    finish_reason=choice.finish_reason,
                    served_model=response.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                    completion_tokens=response.usage.completion_tokens if response.usage else None,
                    attempts=attempt + 1,
                    latency_s=round(time.perf_counter() - start, 3),
                )
                return record
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                retryable = not isinstance(e, APIStatusError) or e.status_code in (408, 429) or e.status_code >= 500
                if not retryable or attempt == RETRIES - 1:
                    record.update(prediction=None, error=f"{type(e).__name__}: {e}"[:500], attempts=attempt + 1)
                    return record
                await asyncio.sleep(2 ** (attempt + 2) * BACKOFF_SCALE)
    return record


async def run(args: argparse.Namespace) -> Path:
    from openai import AsyncOpenAI

    examples = load_split(args.dataset, args.split)
    if args.limit:
        examples = examples[: args.limit]

    client = AsyncOpenAI(
        base_url=args.base_url or settings.require("DMI_INFERENCE_URL"),
        api_key=args.api_key or settings.require("DMI_INFERENCE_API_KEY"),
        timeout=args.timeout,
        max_retries=0,  # retries handled above, with longer backoff for cold starts
    )

    out_dir = settings.REPO_ROOT / "outputs" / "evals" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"

    semaphore = asyncio.Semaphore(args.concurrency)
    print(f"Predicting {len(examples)} {args.split} samples with model {args.model!r} -> {out_path}")
    start = time.perf_counter()
    done = 0
    with out_path.open("w") as out:
        for coro in asyncio.as_completed([predict_one(client, args.model, ex, semaphore) for ex in examples]):
            record = await coro
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            done += 1
            if done % 25 == 0 or done == len(examples):
                print(f"  {done}/{len(examples)} ({time.perf_counter() - start:.0f}s)", flush=True)

    (out_dir / "predict_config.json").write_text(json.dumps({
        "dataset": args.dataset, "split": args.split, "limit": args.limit, "model": args.model,
        "base_url": args.base_url or settings.get("DMI_INFERENCE_URL"),
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "wall_clock_s": round(time.perf_counter() - start, 1),
    }, indent=2) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True, help="name for outputs/evals/<run>/")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="dmi-screen", help="served model name")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600, help="per-request seconds (cold starts are slow)")
    parser.add_argument("--base-url", help="default: $DMI_INFERENCE_URL")
    parser.add_argument("--api-key", help="default: $DMI_INFERENCE_API_KEY")
    args = parser.parse_args()
    asyncio.run(run(args))
    print(f"Done. Score with:  python -m dmi.eval.score --run {args.run}")


if __name__ == "__main__":
    main()
