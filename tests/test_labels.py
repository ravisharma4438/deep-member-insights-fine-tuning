import json

import pytest

from dmi.screening import labels


def _valid_label() -> dict:
    """A schema-valid screen, built in reverse schema order to exercise reordering."""
    label = {
        "scoreOverall": {"justification": "Solid core member.", "score": 6},
        "scoreAiRelevance": 7, "scoreInfluence": 3, "scoreEducation": 5, "scoreExperience": 6,
        "riskFlags": [{"reason": "Two roles under a year.", "flag": "Job Hopping"}],
        "collabMode": {"justification": "Uses 'we'.", "type": "Team Player"},
        "quirkiness": 2, "jobSeekingProb90d": 0.2, "mentorshipLikelihood": 5, "partnershipsPotential": 3,
        "isThoughtLeader": False, "isInvestor": False, "founderCompanyStage": None, "isFounder": False,
        "growthSlope": 0.3,
        "careerTrajectory": {"justification": "Steady.", "type": "Deepening Specialization"},
        "archetype": {"justification": "Ships products.", "type": "The Builder"},
        "communityEngagement": {"justification": "Some meetups.", "score": 5},
        "missionAlignment": {"justification": "Mentions open source.", "score": 4},
        "causeAffinities": {},
        "topicInterests": {},
        "skills": ["Python", "PyTorch"],
        "fieldOfStudy": "Computer Science & AI", "highestEducation": "Masters", "seniority": "Mid-Level",
        "yearsAiExperience": 3, "yearsExperience": 5,
        "industries": {},
        "engagementHook": "Invite to a builders' demo night.",
        "intellectualStance": "Practical ML beats clever ML.",
        "profileSummary": "ML engineer shipping recommender systems.",
        "role": "Data & Research", "currentTitle": "ML Engineer", "currentCompany": "Acme",
        "location": "Austin, Texas, United States", "estimatedAge": 29, "gender": "Female",
        "fullName": "Jane Doe",
        "tier": "M",
    }
    schema_props = labels.screen_schema()["properties"]
    # Fill keyed score maps with a real key from the schema so they aren't trivially empty
    label["industries"] = {next(iter(schema_props["industries"]["properties"])): 6}
    label["topicInterests"] = {next(iter(schema_props["topicInterests"]["properties"])): 4}
    return label


def test_normalize_drops_tier_and_follows_schema_order():
    normalized = labels.normalize(_valid_label())
    assert "tier" not in normalized
    assert list(normalized) == list(labels.screen_schema()["properties"])
    assert list(normalized["scoreOverall"]) == ["score", "justification"]
    assert list(normalized["archetype"]) == ["type", "justification"]
    assert list(normalized["riskFlags"][0]) == ["flag", "reason"]


def test_normalized_label_is_schema_valid_and_compact():
    normalized = labels.normalize(_valid_label())
    assert labels.validation_error(normalized) is None
    text = labels.serialize(normalized)
    assert ": " not in text and ", " not in text.replace("Austin, Texas, United States", "")
    assert json.loads(text) == normalized


@pytest.mark.parametrize(
    "mutate, expected_path",
    [
        (lambda d: d.update(gender="Unknown"), "gender"),
        (lambda d: d.update(extraField=1), "<root>"),
        (lambda d: d["scoreOverall"].update(score=11), "scoreOverall/score"),
        (lambda d: d.pop("fullName"), "<root>"),
    ],
)
def test_validation_error_reports_path(mutate, expected_path):
    label = labels.normalize(_valid_label())
    mutate(label)
    error = labels.validation_error(label)
    assert error is not None and error.startswith(expected_path)


@pytest.mark.parametrize(
    "rating, tier",
    [(10, "H+"), (9, "H+"), (8, "H"), (7, "M"), (6, "M"), (5, "L"), (3, "L"), (2, "L-"), (1, "L-")],
)
def test_tier_for_rating_matches_platform(rating, tier):
    assert labels.tier_for_rating(rating) == tier
