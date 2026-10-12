import copy

from dmi.eval import metrics
from dmi.screening import labels
from tests.test_labels import _valid_label


def _label() -> dict:
    return labels.normalize(_valid_label())


def test_identical_prediction_scores_perfectly():
    label = _label()
    row = metrics.score(copy.deepcopy(label), label)
    scored = {k: v for k, v in row.items() if isinstance(v, float) and not k.startswith("abs_err_")}
    assert all(v == 1.0 for v in scored.values()), {k: v for k, v in scored.items() if v != 1.0}
    assert row["label_tier"] == row["pred_tier"] == "M"


def test_unparseable_prediction_scores_zero():
    assert metrics.parse_prediction("{not json") is None
    assert metrics.parse_prediction('["a list"]') is None
    row = metrics.score(None, _label())
    assert row["valid_json"] == row["schema_valid"] == row["tier"] == 0.0
    assert row["acc_role"] == 0.0 and row["num_yearsExperience"] == 0.0


def test_numeric_tolerance_and_tier_boundary():
    label = _label()
    pred = copy.deepcopy(label)
    pred["yearsExperience"] = label["yearsExperience"] + 2  # within tolerance 2
    pred["estimatedAge"] = label["estimatedAge"] + 4  # outside tolerance 3
    pred["scoreOverall"]["score"] = 8  # 6 -> 8: within score tolerance? no (tol 1), and tier M -> H
    row = metrics.score(pred, label)
    assert row["num_yearsExperience"] == 1.0
    assert row["num_estimatedAge"] == 0.0 and row["abs_err_estimatedAge"] == 4
    assert row["num_scoreOverall.score"] == 0.0
    assert (row["label_tier"], row["pred_tier"], row["tier"]) == ("M", "H", 0.0)


def test_nulls_and_text_normalization():
    label = _label()
    label["estimatedAge"] = None
    pred = copy.deepcopy(label)
    pred["currentTitle"] = "  ml   engineer "
    row = metrics.score(pred, label)
    assert row["num_estimatedAge"] == 1.0 and row["abs_err_estimatedAge"] is None
    assert row["acc_currentTitle"] == 1.0

    pred["estimatedAge"] = 30
    assert metrics.score(pred, label)["num_estimatedAge"] == 0.0


def test_set_overlaps():
    label = _label()
    pred = copy.deepcopy(label)
    pred["skills"] = ["python", "Rust"]  # {python} shared of {python, pytorch, rust}
    pred["riskFlags"] = []
    row = metrics.score(pred, label)
    assert abs(row["jac_skills"] - 1 / 3) < 1e-9
    assert row["jac_riskFlags"] == 0.0
