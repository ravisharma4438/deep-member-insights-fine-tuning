"""Compare eval runs side by side, on the samples every run scored.

    python -m dmi.eval.compare gemma-4-12b-v1 grok-rerun gemma-4-12b-base
    python -m dmi.eval.compare gemma-4-12b-v1 grok-rerun --provider reverse_contact

Reads outputs/evals/<run>/scores.csv (dmi.eval.score) for each run and averages
each metric over the sample ids all runs share, so runs predicted on different
--limit subsets are still compared like for like. Higher is better except
abs_err_* (absolute error). Ops stats (latency, failures) come from each run's
summary.json and cover all of that run's samples.

Reading it: `grok-rerun` (the production model re-screening the same profiles)
is the ceiling. Its agreement with the stored labels measures how consistent
the teacher is with itself, so a fine-tuned model close to it on a metric has
learned about as much of that field as the labels can teach.
"""

from __future__ import annotations

import argparse
import csv
import json
from statistics import mean

from dmi import settings
from dmi.eval.score import HEADLINE

NON_METRICS = {"id", "provider", "label_tier", "pred_tier"}
OPS = ("errors", "latency_p50_s", "latency_p95_s", "mean_completion_tokens")


def _run_dir(run: str):
    return settings.REPO_ROOT / "outputs" / "evals" / run


def load_scores(run: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    path = _run_dir(run) / "scores.csv"
    if not path.exists():
        raise SystemExit(f"No {path}. Run: python -m dmi.eval.score --run {run}")
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = {row["id"]: row for row in reader}
    return list(reader.fieldnames or []), rows


def _mean(values: list[str | None]) -> float | None:
    numbers = [float(v) for v in values if v not in (None, "")]
    return mean(numbers) if numbers else None


def compare(runs: list[str], provider: str | None = None) -> dict:
    """Mean of every metric per run over the shared sample ids."""
    loaded = {run: load_scores(run) for run in runs}
    common = set.intersection(*(set(rows) for _, rows in loaded.values()))
    if provider:
        first = loaded[runs[0]][1]
        common = {i for i in common if first[i]["provider"] == provider}

    columns: list[str] = []
    for fields, _ in loaded.values():
        columns += [c for c in fields if c not in NON_METRICS and c not in columns]
    ordered = [c for c in HEADLINE if c in columns] + [c for c in columns if c not in HEADLINE]
    return {
        "n_common": len(common),
        "n_per_run": {run: len(rows) for run, (_, rows) in loaded.items()},
        "metrics": {
            metric: {run: _mean([loaded[run][1][i].get(metric) for i in common]) for run in runs}
            for metric in ordered
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run names under outputs/evals/")
    parser.add_argument("--provider", help="only samples from this scrape provider (scrapin, reverse_contact)")
    args = parser.parse_args()

    result = compare(args.runs, args.provider)
    if not result["n_common"]:
        raise SystemExit("These runs share no scored samples (predict them with the same --limit and --seed).")

    width = max(10, *(len(run) for run in args.runs))
    per_run = ", ".join(f"{run} {n}" for run, n in result["n_per_run"].items())
    scope = f", provider {args.provider}" if args.provider else ""
    print(f"{result['n_common']} samples scored by every run{scope} (per run: {per_run})\n")
    print(f"  {'metric':28s}" + "".join(f" {run:>{width}s}" for run in args.runs))
    for metric, values in result["metrics"].items():
        cells = ("" if v is None else f"{v:.3f}" for v in values.values())
        print(f"  {metric:28s}" + "".join(f" {cell:>{width}s}" for cell in cells))

    print("\n  ops (all samples of each run)")
    ops = {}
    for run in args.runs:
        summary_path = _run_dir(run) / "summary.json"
        ops[run] = json.loads(summary_path.read_text()).get("ops", {}) if summary_path.exists() else {}
    for key in OPS:
        cells = ("" if ops[run].get(key) is None else str(ops[run][key]) for run in args.runs)
        print(f"  {key:28s}" + "".join(f" {cell:>{width}s}" for cell in cells))


if __name__ == "__main__":
    main()
