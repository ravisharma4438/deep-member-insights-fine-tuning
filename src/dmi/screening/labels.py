"""The model's target output: a stored screen response, normalized to the platform schema.

Stored responses differ from what the model should learn to emit in two ways:

- they carry `tier`, which the platform computes from scoreOverall after the
  call (tierForRating) and is not in ScreenSchema
- Postgres jsonb doesn't preserve key order (it sorts keys by length), while
  constrained decoding emits properties in schema order; training targets are
  reordered to match so the model learns the order it will be held to
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from dmi.screening.inputs import js_json_dumps

DERIVED_KEYS = ("tier",)


@cache
def screen_schema() -> dict[str, Any]:
    """ScreenSchema as JSON Schema, exactly as the AI SDK sends it (vendored)."""
    return json.loads((files("dmi.screening") / "resources" / "screen_schema.json").read_text())


@cache
def _validator():
    from jsonschema import Draft7Validator

    return Draft7Validator(screen_schema())


def _reorder(value: Any, schema: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        props = schema.get("properties")
        if props is None:
            branch = next((b for b in schema.get("anyOf", []) if "properties" in b), None)
            props = branch["properties"] if branch else {}
        ordered = {k: _reorder(value[k], props[k]) for k in props if k in value}
        ordered.update({k: v for k, v in value.items() if k not in props})  # validation rejects these
        return ordered
    if isinstance(value, list):
        items = schema.get("items", {})
        return [_reorder(v, items) for v in value]
    return value


def normalize(response: dict[str, Any]) -> dict[str, Any]:
    """Stored response -> target object: derived keys dropped, keys in schema order."""
    stripped = {k: v for k, v in response.items() if k not in DERIVED_KEYS}
    return _reorder(stripped, screen_schema())


def validation_error(label: dict[str, Any]) -> str | None:
    """First schema violation as a short string, or None if the label is valid."""
    error = next(iter(_validator().iter_errors(label)), None)
    if error is None:
        return None
    path = "/".join(str(p) for p in error.absolute_path) or "<root>"
    return f"{path}: {error.validator}"


def serialize(label: dict[str, Any]) -> str:
    """Compact JSON, matching JSON.stringify (what the platform would store)."""
    return js_json_dumps(label)


def tier_for_rating(rating: int | float) -> str:
    """Port of tierForRating in platform/src/lib/screen/tier-for-rating.ts."""
    if rating >= 9:
        return "H+"
    if rating >= 8:
        return "H"
    if rating >= 6:
        return "M"
    if rating >= 3:
        return "L"
    return "L-"
