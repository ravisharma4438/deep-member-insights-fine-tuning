"""Field-level comparison of a predicted screen against its label (Grok's production output).

Pure functions, no I/O. Every metric is in [0, 1] (higher is better) except the
`abs_err_*` columns, which are absolute errors for the numeric fields.
"""

from __future__ import annotations

import json
from typing import Any

from dmi.screening.labels import tier_for_rating, validation_error

# Enum / boolean fields compared exactly. Dotted paths reach into {type, justification} objects.
CATEGORICAL = (
    "gender", "role", "seniority", "highestEducation", "fieldOfStudy",
    "archetype.type", "careerTrajectory.type", "collabMode.type",
    "isFounder", "founderCompanyStage", "isInvestor", "isThoughtLeader",
)

# Extracted strings compared after case/whitespace normalization.
EXTRACTED_TEXT = ("fullName", "location", "currentCompany", "currentTitle")

# Numeric fields: a prediction within the tolerance counts as correct.
NUMERIC_TOLERANCE = {
    "estimatedAge": 3,
    "yearsExperience": 2,
    "yearsAiExperience": 2,
    "growthSlope": 0.2,
    "jobSeekingProb90d": 0.2,
    "partnershipsPotential": 1,
    "mentorshipLikelihood": 1,
    "quirkiness": 1,
    "missionAlignment.score": 1,
    "communityEngagement.score": 1,
    "scoreExperience": 1,
    "scoreEducation": 1,
    "scoreInfluence": 1,
    "scoreAiRelevance": 1,
    "scoreOverall.score": 1,
}

# Score maps ({name: 0-10}); compared on which keys are present.
KEYED_SCORES = ("industries", "topicInterests", "causeAffinities")

# Free text, judged by an LLM when enabled.
FREE_TEXT = ("profileSummary", "intellectualStance", "engagementHook")

_MISSING = object()


def _path(obj: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return _MISSING
        obj = obj[key]
    return obj


def _norm_text(value: Any) -> Any:
    return " ".join(value.lower().split()) if isinstance(value, str) else value


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is _MISSING or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_prediction(text: str | None) -> dict | None:
    """The model's output as an object, or None if it isn't a JSON object."""
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score(prediction: dict | None, label: dict) -> dict[str, float | str | None]:
    """All metrics for one sample. An unparseable prediction scores 0 on every field."""
    row: dict[str, float | str | None] = {
        "valid_json": float(prediction is not None),
        "schema_valid": float(prediction is not None and validation_error(prediction) is None),
    }
    pred = prediction or {}

    label_score = _num(_path(label, "scoreOverall.score"))
    pred_score = _num(_path(pred, "scoreOverall.score"))
    label_tier = tier_for_rating(label_score) if label_score is not None else None
    pred_tier = tier_for_rating(pred_score) if pred_score is not None else None
    row["label_tier"] = label_tier
    row["pred_tier"] = pred_tier
    row["tier"] = float(label_tier is not None and pred_tier == label_tier)

    for field in CATEGORICAL:
        row[f"acc_{field}"] = float(_path(pred, field) == _path(label, field))

    for field in EXTRACTED_TEXT:
        row[f"acc_{field}"] = float(_norm_text(_path(pred, field)) == _norm_text(_path(label, field)))

    for field, tolerance in NUMERIC_TOLERANCE.items():
        truth, guess = _path(label, field), _path(pred, field)
        t, g = _num(truth), _num(guess)
        if t is None and g is None:
            row[f"num_{field}"] = float(truth is None and guess is None)  # both null counts as agreement
            row[f"abs_err_{field}"] = None
        elif t is None or g is None:
            row[f"num_{field}"] = 0.0
            row[f"abs_err_{field}"] = None
        else:
            row[f"num_{field}"] = float(abs(t - g) <= tolerance + 1e-9)
            row[f"abs_err_{field}"] = abs(t - g)

    skills = lambda d: {_norm_text(s) for s in d.get("skills") or [] if isinstance(s, str)}  # noqa: E731
    row["jac_skills"] = _jaccard(skills(pred), skills(label))
    for field in KEYED_SCORES:
        keys = lambda d: set(d.get(field) or {}) if isinstance(d.get(field), dict) else set()  # noqa: E731
        row[f"jac_{field}"] = _jaccard(keys(pred), keys(label))
    flags = lambda d: {f.get("flag") for f in d.get("riskFlags") or [] if isinstance(f, dict)}  # noqa: E731
    row["jac_riskFlags"] = _jaccard(flags(pred), flags(label))
    return row
