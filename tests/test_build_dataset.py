import json
from collections import Counter
from datetime import datetime, timezone

from dmi.data import build_dataset
from dmi.screening import labels
from tests.test_labels import _valid_label


def test_split_is_stable_and_roughly_proportional():
    emails = [f"member{i}@example.com" for i in range(20_000)]
    splits = [build_dataset.split_for(e, 0.05, 0.05) for e in emails]
    assert splits == [build_dataset.split_for(e.upper(), 0.05, 0.05) for e in emails]  # case-insensitive
    counts = Counter(splits)
    assert 0.04 < counts["test"] / len(emails) < 0.06
    assert 0.04 < counts["val"] / len(emails) < 0.06


def test_example_matches_production_request_and_hides_email():
    screened_at = datetime(2026, 2, 3, 23, 30, tzinfo=timezone.utc)
    linkedin_data = {"person": {"firstName": "Jane", "location": {"city": "Austin", "country": "US"}}}
    label = labels.normalize(_valid_label())

    example = build_dataset.build_example("Jane@Example.com", screened_at, "scrapin", label, linkedin_data)

    assert "jane" not in json.dumps(example["id"]).lower()
    assert example["id"] == build_dataset.sample_id("jane@example.com")
    system, user = example["prompt"]
    assert system["role"] == "system" and "current date is 2026-02-03" in system["content"]
    assert user == {"role": "user", "content": '{"firstName":"Jane","location":"Austin, US","positions":[]}'}
    (assistant,) = example["completion"]
    assert assistant["role"] == "assistant"
    assert json.loads(assistant["content"]) == label and "tier" not in assistant["content"]
    assert example["era"] == "grok41_gateway"
