"""SQL shared by the sample counter and the dataset build, so they agree on what's usable."""

from __future__ import annotations

from datetime import datetime, timedelta

from dmi.screening.eras import LABEL_ERAS, LABEL_MODEL, REVERSE_CONTACT_SINCE
from dmi.screening.prompt import PROMPT_ID

# Input re-scraped this long after the screen no longer matches what the label saw.
RESCRAPE_TOLERANCE = timedelta(hours=1)

_ISO = "'^[0-9]{4}-[0-9]{2}-[0-9]{2}T'"

# promptId-3 member screens with a real response. Postgres inlines single-use
# CTEs, so the jsonb columns cost nothing in aggregate queries.
SCREENS_CTE = f"""
WITH s AS (
  SELECT
    email, screen, linkedin_data,
    CASE WHEN screen->>'date' ~ {_ISO} THEN (screen->>'date')::timestamptz END AS screened_at,
    CASE WHEN linkedin_data->>'date' ~ {_ISO} THEN (linkedin_data->>'date')::timestamptz END AS scraped_at,
    COALESCE(jsonb_typeof(linkedin_data->'person') = 'object', false) AS has_input,
    COALESCE(screen->'usage'->>'reasoningTokens', '0') NOT IN ('0', '') AS has_reasoning_tokens
  FROM members
  WHERE jsonb_typeof(screen->'response') = 'object'
    AND screen->>'promptId' = %(prompt_id)s
    -- labels only from the gateway model; excludes the fine-tuned model's own outputs
    AND (screen->>'model' IS NULL OR screen->>'model' = %(label_model)s)
)
"""

# A sample is usable when its stored input is the one the label was generated from.
USABLE = "has_input AND (scraped_at IS NULL OR scraped_at <= screened_at + %(rescrape_tol)s)"

IN_WINDOW = "screened_at >= %(since)s AND screened_at < %(until)s"

PROVIDER = "CASE WHEN scraped_at >= %(rc_since)s THEN 'reverse_contact' ELSE 'scrapin' END"

USABLE_ROWS = (
    SCREENS_CTE
    + f"""
    SELECT email, screened_at, scraped_at, {PROVIDER} AS provider,
           screen->'response' AS response, linkedin_data
    FROM s
    WHERE {IN_WINDOW} AND {USABLE}
    ORDER BY screened_at
    """
)


def params(since: datetime, until: datetime) -> dict:
    return {
        "prompt_id": PROMPT_ID,
        "label_model": LABEL_MODEL,
        "since": since,
        "until": until,
        "rc_since": REVERSE_CONTACT_SINCE,
        "rescrape_tol": RESCRAPE_TOLERANCE,
        **{f"era_{i}": era.start for i, era in enumerate(LABEL_ERAS)},
    }
