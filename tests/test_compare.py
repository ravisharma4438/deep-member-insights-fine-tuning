import csv

import pytest

from dmi import settings
from dmi.eval import compare


def _write_scores(root, run: str, rows: list[dict]) -> None:
    run_dir = root / "outputs" / "evals" / run
    run_dir.mkdir(parents=True)
    with (run_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    _write_scores(tmp_path, "tuned", [
        {"id": "a", "provider": "scrapin", "tier": "1.0", "acc_role": "1.0"},
        {"id": "b", "provider": "reverse_contact", "tier": "0.0", "acc_role": "1.0"},
        {"id": "c", "provider": "scrapin", "tier": "0.0", "acc_role": "0.0"},  # only in this run
    ])
    _write_scores(tmp_path, "grok", [
        {"id": "a", "provider": "scrapin", "tier": "1.0", "acc_role": "", "judge_profileSummary": "0.5"},
        {"id": "b", "provider": "reverse_contact", "tier": "1.0", "acc_role": "0.0", "judge_profileSummary": ""},
    ])


def test_compares_on_shared_samples_only(runs):
    result = compare.compare(["tuned", "grok"])
    assert result["n_common"] == 2
    assert result["n_per_run"] == {"tuned": 3, "grok": 2}
    metrics = result["metrics"]
    assert list(metrics)[:2] == ["tier", "acc_role"]  # headline metrics first
    assert metrics["tier"] == {"tuned": 0.5, "grok": 1.0}  # sample c excluded
    assert metrics["acc_role"] == {"tuned": 1.0, "grok": 0.0}  # empty cells skipped
    assert metrics["judge_profileSummary"] == {"tuned": None, "grok": 0.5}


def test_provider_filter(runs):
    result = compare.compare(["tuned", "grok"], provider="reverse_contact")
    assert result["n_common"] == 1
    assert result["metrics"]["tier"] == {"tuned": 0.0, "grok": 1.0}
