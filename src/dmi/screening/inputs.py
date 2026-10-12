"""The model's user message: a port of `relevantData` in the platform's `screenMember()`.

Training inputs must be byte-identical to what production sends, so this
mirrors the JS behaviour of platform/src/lib/screen/screen-member.ts:

- keys whose value is `undefined` (absent in the JSON) are dropped; `null` is kept
- `a?.b` short-circuits on null as well as undefined
- enrichExperience's `exp.linkedInId === company.linkedInId` is true when both
  are undefined
- `str.slice(0, 500)` counts UTF-16 code units, not code points
- `JSON.stringify` output: no whitespace, non-ASCII unescaped, lone
  surrogates escaped
"""

from __future__ import annotations

import json
import re
from typing import Any


class _Undefined:
    """Stand-in for JS `undefined` (distinct from `None`, which is JSON null)."""

    def __repr__(self) -> str:
        return "undefined"


UNDEFINED: Any = _Undefined()

DESCRIPTION_MAX_UTF16 = 500


def _get(obj: Any, *path: str) -> Any:
    """`obj?.a?.b`"""
    for key in path:
        if not isinstance(obj, dict):
            return UNDEFINED
        obj = obj.get(key, UNDEFINED)
    return obj


def _defined(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if v is not UNDEFINED}


def _truthy(value: Any) -> bool:
    if value is UNDEFINED or value is None:
        return False
    if isinstance(value, (dict, list)):
        return True  # JS objects and arrays are truthy even when empty
    return bool(value)


def _slice_utf16(text: str, n: int) -> str:
    encoded = text.encode("utf-16-le", "surrogatepass")
    if len(encoded) <= 2 * n:
        return text
    return encoded[: 2 * n].decode("utf-16-le", "surrogatepass")


def _truncated_description(value: Any) -> str:
    return _slice_utf16(value, DESCRIPTION_MAX_UTF16) if isinstance(value, str) and value else ""


def format_scrapin_location(location: Any) -> str | None:
    """Port of `formatScrapinLocation` in platform/src/lib/types/scrapin.ts."""
    if not _truthy(location):
        return None
    if isinstance(location, str):
        return location
    if isinstance(location, dict):
        parts = [_get(location, k) for k in ("city", "state", "country")]
        return ", ".join(str(p) for p in parts if _truthy(p)) or None
    return None


def _enrich_experience(position_history: Any, company: Any) -> list[dict[str, Any]]:
    if not _truthy(position_history):
        return []

    enriched = []
    for exp in position_history:
        if _truthy(company) and _get(exp, "linkedInId") == _get(company, "linkedInId"):
            enriched.append(_defined({
                "title": _get(exp, "title"),
                "companyName": _get(company, "name"),
                "contractType": _get(exp, "contractType"),
                "startEndDate": _get(exp, "startEndDate"),
                "description": _truncated_description(_get(exp, "description")),
                "companyLocation": _get(exp, "companyLocation"),
                "companyTagline": _get(company, "tagline"),
                "companyDescription": _truncated_description(_get(company, "description")),
                "followerCount": _get(company, "followerCount"),
                "employeeCount": _get(company, "employeeCount"),
                "employeeCountRange": _get(company, "employeeCountRange"),
                "industry": _get(company, "industry"),
                "foundedOn": _get(company, "foundedOn"),
                "fundingData": _get(company, "fundingData"),
            }))
        else:
            enriched.append(_defined({
                "title": _get(exp, "title"),
                "companyName": _get(exp, "companyName"),
                "contractType": _get(exp, "contractType"),
                "startEndDate": _get(exp, "startEndDate"),
                "description": _truncated_description(_get(exp, "description")),
                "companyLocation": _get(exp, "companyLocation"),
            }))
    return enriched


def relevant_data(linkedin_data: dict[str, Any]) -> dict[str, Any]:
    """`relevantData` for a stored `LinkedinData` (or Scrapin response) object."""
    person = _get(linkedin_data, "person")
    education_history = _get(person, "schools", "educationHistory")

    return _defined({
        "firstName": _get(person, "firstName"),
        "lastName": _get(person, "lastName"),
        "pronoun": _get(person, "pronoun"),
        "location": format_scrapin_location(_get(person, "location")),
        "followerCount": _get(person, "followerCount"),
        "headline": _get(person, "headline"),
        "openToWork": _get(person, "openToWork"),
        "premium": _get(person, "premium"),
        "summary": _get(person, "summary"),
        "positionsCount": _get(person, "positions", "positionsCount"),
        "positions": _enrich_experience(
            _get(person, "positions", "positionHistory"),
            _get(linkedin_data, "company"),
        ),
        "educationsCount": _get(person, "schools", "educationsCount"),
        "educations": (
            [
                _defined({
                    "schoolName": _get(edu, "schoolName"),
                    "degreeName": _get(edu, "degreeName"),
                    "fieldOfStudy": _get(edu, "fieldOfStudy"),
                    "description": _get(edu, "description"),
                    "startEndDate": _get(edu, "startEndDate"),
                })
                for edu in education_history
            ]
            if isinstance(education_history, list)
            else UNDEFINED
        ),
        "skills": _get(person, "skills"),
        "certificationsCount": _get(person, "certifications", "certificationsCount"),
        "certifications": _get(person, "certifications", "certificationHistory"),
        "volunteeringExperiencesCount": _get(
            person, "volunteeringExperiences", "volunteeringExperiencesCount"
        ),
        "volunteeringExperiences": _get(
            person, "volunteeringExperiences", "volunteeringExperienceHistory"
        ),
        "interests": _get(person, "interests"),
    })


_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _js_numbers(value: Any) -> Any:
    """JSON.stringify prints integral floats without a decimal point."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _js_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_js_numbers(v) for v in value]
    return value


def js_json_dumps(value: Any) -> str:
    """`JSON.stringify(value)`."""
    text = json.dumps(_js_numbers(value), ensure_ascii=False, separators=(",", ":"))
    return _LONE_SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


def user_message(linkedin_data: dict[str, Any]) -> str:
    """The user message production sends: `JSON.stringify(relevantData)`."""
    return js_json_dumps(relevant_data(linkedin_data))
