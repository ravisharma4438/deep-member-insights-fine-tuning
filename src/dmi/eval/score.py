"""Score predictions from dmi.eval.predict against their labels.

    python -m dmi.eval.score --run gemma-4-12b-v1
    python -m dmi.eval.score --run gemma-4-12b-v1 --judge     # + LLM judge on free-text fields

Writes outputs/evals/<run>/scores.csv (per sample) and summary.json, and prints
the summary overall and by scrape provider, plus a tier confusion matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter, defaultdict
from statistics import mean

from dmi import settings
from dmi.eval import metrics

TIERS = ("H+", "H", "M", "L", "L-")
HEADLINE = (
    "valid_json", "schema_valid", "tier", "num_scoreOverall.score", "acc_role", "acc_seniority",
    "acc_archetype.type", "jac_skills", "jac_industries", "jac_riskFlags",
)

JUDGE_PROMPT = """You are grading one field of an AI-generated LinkedIn profile analysis against a reference.
Field: {field}
Reference: {reference}
Candidate: {candidate}

Score how well the candidate matches the reference in meaning and factual content (wording may differ):
1 = wrong or unrelated, 2 = mostly wrong, 3 = main idea right but misses or adds key details,
4 = close with minor differences, 5 = equivalent.
Reply with only the integer."""


def _mean(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(mean(vals), 4) if vals else None


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    return round(vals[min(len(vals) - 1, int(q * len(vals)))], 2) if vals else None


async def judge(rows: list[dict], records: list[dict], concurrency: int) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.require("OPENAI_API_KEY"))
    model = settings.get("DMI_JUDGE_MODEL", "gpt-5.4-mini")
    semaphore = asyncio.Semaphore(concurrency)

    async def grade(row: dict, field: str, reference: str, candidate: str) -> None:
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        field=field, reference=reference, candidate=candidate)}],
                )
                row[f"judge_{field}"] = (int(response.choices[0].message.content.strip()) - 1) / 4
            except Exception as e:  # a failed grade leaves the cell empty rather than 0
                print(f"  judge failed on {row['id']}/{field}: {e}")

    tasks = []
    for row, record in zip(rows, records):
        label = json.loads(record["label"])
        pred = metrics.parse_prediction(record.get("prediction")) or {}
        for field in metrics.FREE_TEXT:
            if pred.get(field) and label.get(field):
                tasks.append(grade(row, field, label[field], pred[field]))
            else:
                row[f"judge_{field}"] = 0.0 if label.get(field) else None
    print(f"Judging {len(tasks)} free-text fields with {model}...")
    await asyncio.gather(*tasks)


def summarize(rows: list[dict]) -> dict:
    columns = [c for c in rows[0] if c not in ("id", "provider", "label_tier", "pred_tier")]
    return {"n": len(rows), **{c: _mean([r.get(c) for r in rows]) for c in columns}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--judge", action="store_true", help="LLM-judge free-text fields (needs OPENAI_API_KEY)")
    parser.add_argument("--judge-concurrency", type=int, default=16)
    args = parser.parse_args()

    run_dir = settings.REPO_ROOT / "outputs" / "evals" / args.run
    with (run_dir / "predictions.jsonl").open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for record in records:
        row = {"id": record["id"], "provider": record.get("provider")}
        row.update(metrics.score(metrics.parse_prediction(record.get("prediction")), json.loads(record["label"])))
        rows.append(row)
    if args.judge:
        asyncio.run(judge(rows, records, args.judge_concurrency))

    by_provider = defaultdict(list)
    for row in rows:
        by_provider[row["provider"] or "unknown"].append(row)

    confusion = Counter((r["label_tier"], r["pred_tier"]) for r in rows)
    summary = {
        "overall": summarize(rows),
        "by_provider": {p: summarize(rs) for p, rs in sorted(by_provider.items())},
        "tier_confusion": {f"{t}->{p}": n for (t, p), n in sorted(confusion.items(), key=str)},
        "ops": {
            "errors": sum(1 for r in records if r.get("error")),
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in records)),
            "latency_p50_s": _percentile([r.get("latency_s") for r in records], 0.5),
            "latency_p95_s": _percentile([r.get("latency_s") for r in records], 0.95),
            "mean_completion_tokens": _mean([r.get("completion_tokens") for r in records]),
            "served_models": dict(Counter(r.get("served_model") for r in records)),
        },
    }

    with (run_dir / "scores.csv").open("w", newline="") as f:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    providers = list(summary["by_provider"])
    print(f"\n{args.run}: {len(rows)} samples")
    print(f"  {'metric':28s} {'overall':>8s}" + "".join(f" {p[:15]:>15s}" for p in providers))
    keys = list(HEADLINE) + [k for k in summary["overall"] if k not in HEADLINE and k != "n"]
    for key in ["n", *keys]:
        values = [summary["overall"].get(key)] + [summary["by_provider"][p].get(key) for p in providers]
        print(f"  {key:28s}" + "".join(
            f" {'' if v is None else (f'{v:.3f}' if isinstance(v, float) else v):>{8 if i == 0 else 15}}"
            for i, v in enumerate(values)))

    print("\nTier confusion (rows = label, cols = prediction)")
    cols = [*TIERS, None]
    print("  " + " " * 6 + "".join(f"{str(c):>6s}" for c in cols))
    for t in TIERS:
        print(f"  {t:6s}" + "".join(f"{confusion.get((t, c), 0):6d}" for c in cols))
    print(f"\nops: {json.dumps(summary['ops'])}")
    print(f"Wrote {run_dir / 'scores.csv'} and summary.json")


if __name__ == "__main__":
    main()
