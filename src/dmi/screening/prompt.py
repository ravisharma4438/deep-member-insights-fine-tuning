"""Screening system prompt, vendored from the platform (scripts/sync_from_platform.py)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from importlib.resources import files

_RESOURCES = files("dmi.screening") / "resources"
_TEMPLATE = (_RESOURCES / "system_prompt.txt").read_text()
PROMPT_ID: str = json.loads((_RESOURCES / "platform_sync.json").read_text())["prompt_id"]


def system_prompt(as_of: date | datetime | str) -> str:
    """The prompt as production renders it on `as_of` (it embeds the current date).

    For training rows pass the screen's date, so the label's age/tenure math
    lines up with the date the teacher model was told.
    """
    if isinstance(as_of, datetime):
        # Production uses toISOString(), i.e. the UTC calendar date.
        if as_of.tzinfo is not None:
            as_of = as_of.astimezone(timezone.utc)
        as_of = as_of.date()
    if isinstance(as_of, date):
        as_of = as_of.isoformat()
    return _TEMPLATE.replace("${date}", as_of)
