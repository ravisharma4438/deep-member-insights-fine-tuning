"""Which model and prompt produced a stored `screen`, by date.

The platform's `members.screen` JSON doesn't record the model, so labels are
attributed by `screen.date` against the production deploy history below.
Timestamps are the merge into the platform `production` branch (UTC), taken
from platform git history. Update this table whenever the platform changes the
screening model, prompt, or input format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Covers deploy propagation and in-flight jobs still running the previous build.
DEPLOY_BUFFER = timedelta(hours=1)


@dataclass(frozen=True)
class LabelEra:
    key: str
    start: datetime
    model: str
    change: str


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


LABEL_ERAS: tuple[LabelEra, ...] = (
    LabelEra("gemini", _utc("2025-06-30T04:05:36"), "google/gemini-2.5-pro",
             "promptId 3 screening system introduced"),
    LabelEra("grok4", _utc("2025-09-24T13:12:12"), "xai/grok-4-fast",
             "switched to Grok"),
    LabelEra("grok41_xai", _utc("2025-12-05T05:16:18"), "xai/grok-4.1-fast-reasoning",
             "current model; called via xAI SDK, default temperature"),
    LabelEra("grok41_openrouter", _utc("2025-12-19T17:26:55"), "x-ai/grok-4.1-fast (OpenRouter)",
             "same model via OpenRouter; temperature 0.1"),
    LabelEra("grok41_gateway", _utc("2026-01-26T10:48:31"), "xai/grok-4.1-fast-reasoning (AI Gateway)",
             "same model via Vercel AI Gateway"),
    LabelEra("mission_v2", _utc("2026-02-15T22:56:57"), "xai/grok-4.1-fast-reasoning (AI Gateway)",
             "missionAlignment prompt text changed to the Q1 2026 mission"),
    LabelEra("location_fmt", _utc("2026-03-16T16:43:49"), "xai/grok-4.1-fast-reasoning (AI Gateway)",
             "input location formatted as a string (matches current serving input)"),
)

# First moment every production screen came from the current model.
CURRENT_MODEL_SINCE: datetime = LABEL_ERAS[2].start + DEPLOY_BUFFER

# Screens written after the platform's Modal integration record `screen.model`:
# the gateway model id, or "modal/<served model>" for the fine-tuned model.
# Training labels must come from the gateway model only (never the fine-tune's
# own outputs); screens without the field predate it and are attributed by date.
LABEL_MODEL = "xai/grok-4.1-fast-reasoning"

# Input-side shift: profiles scraped from here on come from Reverse Contact v2,
# adapted to the Scrapin shape (no `company` object, no volunteering/interests).
# fetchLinkedinData stamps cached data with its original scrape time, so
# `linkedin_data.date >= REVERSE_CONTACT_SINCE` identifies Reverse Contact input.
REVERSE_CONTACT_SINCE: datetime = _utc("2026-05-25T06:02:26")


def era_for(screen_date: datetime) -> LabelEra | None:
    """Era a screen belongs to, or None if it predates promptId 3."""
    match = None
    for era in LABEL_ERAS:
        if screen_date >= era.start:
            match = era
    return match
