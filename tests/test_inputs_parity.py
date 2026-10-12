"""Byte-for-byte parity between dmi.screening.inputs and the platform's TypeScript.

Extracts `relevantData`, `enrichExperience` and `formatScrapinLocation` from a
platform checkout, runs them under Node (native type stripping), and compares
against the Python port on edge cases plus seeded random profiles.

Skips when Node or the platform checkout (PLATFORM_REPO, default ../platform)
is unavailable.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dmi.screening.inputs import user_message

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATFORM = Path(os.environ.get("PLATFORM_REPO", REPO_ROOT.parent / "platform"))
SCREEN_MEMBER_TS = PLATFORM / "src/lib/screen/screen-member.ts"
SCRAPIN_TS = PLATFORM / "src/lib/types/scrapin.ts"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not SCREEN_MEMBER_TS.exists(),
    reason="needs node and a platform checkout",
)


def _extract(pattern: str, path: Path) -> str:
    match = re.search(pattern, path.read_text(), re.DOTALL)
    assert match, f"pattern not found in {path}: {pattern}"
    return match.group(1)


def _platform_user_messages(fixtures_json: str, tmp_path: Path) -> list[str]:
    relevant_data = _extract(r"const relevantData = (\{.*?\n  \});", SCREEN_MEMBER_TS)
    enrich = _extract(r"(function enrichExperience\(.*?\n\})", SCREEN_MEMBER_TS)
    fmt_location = _extract(r"export (function formatScrapinLocation\(.*?\n\})", SCRAPIN_TS)

    harness = tmp_path / "harness.mts"
    harness.write_text(
        'import { readFileSync } from "node:fs";\n'
        f"{fmt_location}\n{enrich}\n"
        "function build(linkedinData: any): string {\n"
        f"  const relevantData = {relevant_data};\n"
        "  return JSON.stringify(relevantData);\n"
        "}\n"
        "const fixtures = JSON.parse(readFileSync(process.argv[2], 'utf8'));\n"
        "process.stdout.write(JSON.stringify(fixtures.map(build)));\n"
    )
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(fixtures_json)

    out = subprocess.run(
        ["node", str(harness), str(fixtures)], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


# --- fixtures -----------------------------------------------------------------

LONG = "x" * 520
ASTRAL_AT_CUT = "a" * 499 + "😀" + "tail"  # UTF-16 slice(0, 500) splits the emoji

HANDWRITTEN = [
    # company merged by matching linkedInId; location object; long + astral descriptions
    {
        "person": {
            "firstName": "Ada", "lastName": "Lovelace 👩‍💻", "pronoun": "she/her",
            "location": {"city": "London", "state": "", "country": "UK", "countryCode": "GB"},
            "followerCount": 5.0, "headline": "Engine \"whisperer\"\n ", "openToWork": False,
            "premium": True, "summary": "Ünïcödé\t\u0001",
            "positions": {"positionsCount": 2, "positionHistory": [
                {"title": "CTO", "companyName": "Old", "linkedInId": "42",
                 "description": ASTRAL_AT_CUT, "startEndDate": {"start": {"year": 2020}}},
                {"title": "Eng", "companyName": "Other", "linkedInId": "7", "description": LONG},
            ]},
            "schools": {"educationsCount": 1, "educationHistory": [
                {"schoolName": "Cambridge", "degreeName": None, "fieldOfStudy": "Math"}]},
            "skills": ["Python", "Analytical Engines"],
            "certifications": {"certificationsCount": 0, "certificationHistory": []},
            "interests": None,
        },
        "company": {"linkedInId": "42", "name": "Engines Inc", "description": LONG,
                    "employeeCount": 12, "fundingData": {"lastRound": "Seed"}},
    },
    # sparse: no company, null location, positions null, schools missing
    {"person": {"firstName": "Bo", "location": None, "positions": None}},
    # empty-string location -> null; educationHistory null -> key dropped
    {"person": {"firstName": "Cy", "location": "", "schools": {"educationHistory": None}}},
    # undefined === undefined: company without linkedInId merges into positions without one
    {"person": {"positions": {"positionHistory": [{"title": "Founder"}]}},
     "company": {"name": "NoId Co", "tagline": "t"}},
    # null !== undefined: company.linkedInId null does not merge position missing linkedInId
    {"person": {"positions": {"positionHistory": [{"title": "PM", "companyName": "X"}]}},
     "company": {"linkedInId": None, "name": "NullId Co"}},
    # empty company object is truthy in JS
    {"person": {"positions": {"positionHistory": [{"title": "Dev"}]}}, "company": {}},
    # location object with nothing usable -> null; empty position list
    {"person": {"location": {"countryCode": "US"}, "positions": {"positionHistory": []}}},
]


def _random_value(rng: random.Random, depth: int = 0):
    choices = ["str", "int", "float", "bool", "null", "emoji"]
    if depth < 2:
        choices += ["list", "dict"]
    kind = rng.choice(choices)
    if kind == "str":
        return "".join(rng.choice("abc é\n\"\\/") for _ in range(rng.randint(0, 12)))
    if kind == "emoji":
        return "🙂" * rng.randint(1, 3) + "x" * rng.choice([0, 497, 498, 499, 500])
    if kind == "int":
        return rng.randint(-5, 10_000)
    if kind == "float":
        return rng.choice([1.5, 2.0, 0.0, 1e3])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {k: _random_value(rng, depth + 1) for k in rng.sample(["a", "b", "c"], rng.randint(0, 3))}


def _maybe(rng: random.Random, obj: dict, key: str, value) -> None:
    """Leave the key undefined, set it null, or set it."""
    roll = rng.random()
    if roll < 0.2:
        return
    obj[key] = None if roll < 0.3 else value


def _random_profile(rng: random.Random) -> dict:
    ids = ["1", "2", None]
    person: dict = {}
    for key in ["firstName", "lastName", "pronoun", "headline", "summary", "followerCount",
                "openToWork", "premium", "skills", "interests"]:
        _maybe(rng, person, key, _random_value(rng))
    _maybe(rng, person, "location", rng.choice([
        "Paris, France", "", {"city": "Lyon", "country": "FR"}, {"state": "CA"}, {}]))

    positions = []
    for _ in range(rng.randint(0, 3)):
        exp: dict = {}
        for key in ["title", "companyName", "contractType", "startEndDate", "companyLocation"]:
            _maybe(rng, exp, key, _random_value(rng))
        _maybe(rng, exp, "description", rng.choice([LONG, ASTRAL_AT_CUT, "short", ""]))
        if rng.random() < 0.7:
            exp["linkedInId"] = rng.choice(ids)
        positions.append(exp)
    _maybe(rng, person, "positions", {"positionsCount": len(positions), "positionHistory": positions})

    edus = [{k: _random_value(rng) for k in rng.sample(
        ["schoolName", "degreeName", "fieldOfStudy", "description", "startEndDate"], 3)}
        for _ in range(rng.randint(0, 2))]
    _maybe(rng, person, "schools", {"educationsCount": len(edus), "educationHistory": edus})
    _maybe(rng, person, "certifications", {"certificationsCount": 1,
                                           "certificationHistory": [_random_value(rng)]})
    _maybe(rng, person, "volunteeringExperiences", {
        "volunteeringExperiencesCount": 1, "volunteeringExperienceHistory": [_random_value(rng)]})

    profile: dict = {"person": person}
    if rng.random() < 0.7:
        company: dict = {}
        for key in ["name", "tagline", "followerCount", "employeeCount", "employeeCountRange",
                    "industry", "foundedOn", "fundingData"]:
            _maybe(rng, company, key, _random_value(rng))
        _maybe(rng, company, "description", rng.choice([LONG, ASTRAL_AT_CUT, "d"]))
        if rng.random() < 0.7:
            company["linkedInId"] = rng.choice(ids)
        profile["company"] = company
    return profile


def test_user_message_matches_platform(tmp_path: Path) -> None:
    rng = random.Random(1234)
    profiles = HANDWRITTEN + [_random_profile(rng) for _ in range(300)]
    # ensure_ascii keeps lone surrogates as \u escapes so both parsers see the same strings
    fixtures_json = json.dumps(profiles, ensure_ascii=True)

    expected = _platform_user_messages(fixtures_json, tmp_path)
    actual = [user_message(p) for p in json.loads(fixtures_json)]

    mismatches = [i for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    assert not mismatches, (
        f"{len(mismatches)} mismatches, first #{mismatches[0]}:\n"
        f"platform: {expected[mismatches[0]][:400]}\npython:   {actual[mismatches[0]][:400]}"
    )
