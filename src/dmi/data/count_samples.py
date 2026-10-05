"""How many training samples would the current-model label window give?

Read-only. From the repo root, in a Studio terminal or a notebook cell (prefix with !):

    python -m dmi.data.count_samples
    python -m dmi.data.count_samples --tokenizer <hf-model-id> --max-seq-len 8192
    python -m dmi.data.count_samples --since 2026-03-16T17:43:49Z   # stricter window

Without --tokenizer the counts come from SQL aggregates (seconds). With it,
rows in the window are streamed and the real training messages (system prompt,
user message, label) are tokenized with that model's chat template.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from dmi.db import connect
from dmi.screening.eras import (
    CURRENT_MODEL_SINCE,
    LABEL_ERAS,
    REVERSE_CONTACT_SINCE,
)
from dmi.screening.prompt import PROMPT_ID

# Input re-scraped this long after the screen no longer matches what the label saw.
RESCRAPE_TOLERANCE = timedelta(hours=1)
TOKEN_THRESHOLDS = (4096, 6144, 8192, 9400, 10240, 12288, 16384)

_ISO = "'^[0-9]{4}-[0-9]{2}-[0-9]{2}T'"

# Postgres inlines single-use CTEs, so the jsonb columns cost nothing in the aggregates.
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
)
"""

# A sample is usable when its stored input is the one the label was generated from.
USABLE = "has_input AND (scraped_at IS NULL OR scraped_at <= screened_at + %(rescrape_tol)s)"


def _parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _table(headers: list[str], rows: list[list]) -> None:
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for n, row in enumerate(cells):
        print("  " + "  ".join(c.rjust(w) if n and i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))))
        if n == 0:
            print("  " + "  ".join("-" * w for w in widths))


def _fmt_day(ts: datetime | None) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d") if ts else ""


def report_eras(cur, params: dict) -> None:
    era_case = "CASE " + " ".join(
        f"WHEN screened_at >= %(era_{i})s THEN {i}" for i in reversed(range(len(LABEL_ERAS)))
    ) + " ELSE -1 END"
    cur.execute(
        SCREENS_CTE
        + f"""
        SELECT {era_case} AS era, count(*),
               count(*) FILTER (WHERE {USABLE}),
               count(*) FILTER (WHERE has_reasoning_tokens),
               min(screened_at), max(screened_at)
        FROM s GROUP BY 1 ORDER BY 1
        """,
        params,
    )
    rows = []
    for era, total, usable, reasoning, first, last in cur.fetchall():
        name = LABEL_ERAS[era].key if era >= 0 else "(older or bad date)"
        model = LABEL_ERAS[era].model if era >= 0 else ""
        rows.append([name, model, total, usable, reasoning, _fmt_day(first), _fmt_day(last)])
    print("\nAll promptId-3 screens by production era (members table)")
    _table(["era", "model", "screens", "usable", "w/ reasoningTokens", "first", "last"], rows)
    print("  usable = has linkedin_data that wasn't re-scraped after the screen")
    print("  w/ reasoningTokens is only a fingerprint: some providers don't report it")


def report_window(cur, params: dict) -> int:
    window = "screened_at >= %(since)s AND screened_at < %(until)s"
    cur.execute(
        SCREENS_CTE
        + f"""
        SELECT count(*),
               count(*) FILTER (WHERE NOT has_input),
               count(*) FILTER (WHERE has_input AND NOT ({USABLE})),
               count(*) FILTER (WHERE {USABLE}),
               count(*) FILTER (WHERE {USABLE} AND scraped_at >= %(rc_since)s)
        FROM s WHERE {window}
        """,
        params,
    )
    total, no_input, rescraped, usable, reverse_contact = cur.fetchone()

    print(f"\nLabel window: {params['since']:%Y-%m-%d %H:%M} UTC -> {params['until']:%Y-%m-%d %H:%M} UTC")
    _table(
        ["", "members"],
        [
            ["screens in window", total],
            ["- no linkedin_data", no_input],
            ["- re-scraped after screen", rescraped],
            ["= usable samples", usable],
            ["  of which Scrapin input", usable - reverse_contact],
            ["  of which Reverse Contact input", reverse_contact],
        ],
    )

    cur.execute(
        SCREENS_CTE
        + f"""
        SELECT date_trunc('month', screened_at AT TIME ZONE 'UTC') AS month, count(*) FILTER (WHERE {USABLE})
        FROM s WHERE {window} GROUP BY 1 ORDER BY 1
        """,
        params,
    )
    print("\nUsable samples by month")
    _table(["month", "usable"], [[m.strftime("%Y-%m"), n] for m, n in cur.fetchall()])
    return usable


def report_position_keys(cur, params: dict) -> None:
    """Keys present on the first position entry, per scrape provider.

    The input builder reads title, companyName, contractType, startEndDate,
    description, companyLocation and linkedInId. Missing keys mean the model
    sees less than intended.
    """
    cur.execute(
        SCREENS_CTE
        + f""",
        p AS (
          SELECT
            CASE WHEN scraped_at >= %(rc_since)s THEN 'reverse_contact' ELSE 'scrapin' END AS provider,
            linkedin_data->'person'->'positions'->'positionHistory'->0 AS first_position
          FROM s
          WHERE screened_at >= %(since)s AND screened_at < %(until)s
            AND scraped_at IS NOT NULL AND {USABLE}
            AND jsonb_typeof(linkedin_data->'person'->'positions'->'positionHistory'->0) = 'object'
        ),
        totals AS (SELECT provider, count(*) AS n FROM p GROUP BY 1)
        SELECT p.provider, k, count(*), totals.n
        FROM p CROSS JOIN LATERAL jsonb_object_keys(first_position) AS k
        JOIN totals USING (provider)
        GROUP BY 1, 2, 4 ORDER BY 1, 3 DESC, 2
        """,
        params,
    )
    rows = [[prov, key, n, f"{100 * n / total:.0f}%"] for prov, key, n, total in cur.fetchall()]
    print("\nKeys on the first position entry, by scrape provider (window only)")
    if rows:
        _table(["provider", "key", "rows", "share"], rows)
    else:
        print("  (no rows)")
    print("  input builder reads: title, companyName, contractType, startEndDate, description,")
    print("  companyLocation, linkedInId")


def report_contacts(cur, params: dict) -> None:
    """Screens stored on linkedin_data rows (CRM contacts) for people not in members."""
    cur.execute(
        f"""
        SELECT count(*)
        FROM linkedin_data l
        WHERE jsonb_typeof(l.screen->'response') = 'object'
          AND l.screen->>'promptId' = %(prompt_id)s
          AND l.screen->>'date' ~ {_ISO}
          AND (l.screen->>'date')::timestamptz >= %(since)s
          AND (l.screen->>'date')::timestamptz < %(until)s
          AND jsonb_typeof(l.response->'person') = 'object'
          AND NOT EXISTS (
            SELECT 1 FROM members m
            WHERE m.email = l.email AND jsonb_typeof(m.screen->'response') = 'object'
          )
        """,
        params,
    )
    print(f"\nExtra samples from contacts (linkedin_data table, not in members): {cur.fetchone()[0]}")
    print("  not included above; their stored input may have been re-scraped after the screen")


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def report_token_lengths(conn, params: dict, tokenizer_id: str, max_seq_len: int | None, limit: int | None) -> None:
    from jinja2.exceptions import TemplateError
    from transformers import AutoTokenizer

    from dmi.screening.inputs import js_json_dumps, user_message
    from dmi.screening.prompt import system_prompt

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def n_tokens(messages: list[dict], generation_prompt: bool = False) -> int:
        render = lambda msgs: tokenizer.apply_chat_template(  # noqa: E731
            msgs, tokenize=False, add_generation_prompt=generation_prompt
        )
        try:
            text = render(messages)
        except TemplateError:
            # Templates without a system role: fold it into the first user turn
            merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
            text = render(merged + messages[2:])
        # The rendered template already carries BOS and other special tokens
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    sql = (
        SCREENS_CTE
        + f"""
        SELECT screened_at, screen->'response', linkedin_data FROM s
        WHERE screened_at >= %(since)s AND screened_at < %(until)s AND {USABLE}
        ORDER BY md5(email)
        """
        + (" LIMIT %(limit)s" if limit else "")
    )
    totals, prompts, labels = [], [], []
    with conn.cursor(name="token_lengths") as cur:
        cur.itersize = 500
        cur.execute(sql, {**params, "limit": limit})
        for i, (screened_at, response, linkedin_data) in enumerate(cur, 1):
            label = {k: v for k, v in response.items() if k != "tier"}
            prompt_msgs = [
                {"role": "system", "content": system_prompt(screened_at)},
                {"role": "user", "content": user_message(linkedin_data)},
            ]
            full = n_tokens(prompt_msgs + [{"role": "assistant", "content": js_json_dumps(label)}])
            prompt_only = n_tokens(prompt_msgs, generation_prompt=True)
            totals.append(full)
            prompts.append(prompt_only)
            labels.append(full - prompt_only)
            if i % 2000 == 0:
                print(f"  tokenized {i} rows...", flush=True)

    if not totals:
        print("\nNo usable rows to tokenize.")
        return

    print(f"\nToken lengths with {tokenizer_id} chat template ({len(totals)} rows)")
    _table(
        ["", "p50", "p90", "p95", "p99", "max"],
        [
            [name, *(_percentile(sorted(v), q) for q in (0.5, 0.9, 0.95, 0.99)), max(v)]
            for name, v in (("full sequence", totals), ("prompt", prompts), ("label", labels))
        ],
    )
    thresholds = sorted(set(TOKEN_THRESHOLDS) | ({max_seq_len} if max_seq_len else set()))
    print("\nSamples that fit (full sequence <= max_seq_len)")
    _table(
        ["max_seq_len", "samples", "share"],
        [[t, sum(n <= t for n in totals), f"{100 * sum(n <= t for n in totals) / len(totals):.1f}%"]
         for t in thresholds],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", type=_parse_ts, default=CURRENT_MODEL_SINCE,
                        help="label window start, ISO UTC (default: current Grok model went live + 1h)")
    parser.add_argument("--until", type=_parse_ts, default=datetime.now(timezone.utc),
                        help="label window end, ISO UTC (default: now)")
    parser.add_argument("--tokenizer", help="HF model id; also report exact token lengths")
    parser.add_argument("--max-seq-len", type=int, help="extra threshold to report with --tokenizer")
    parser.add_argument("--limit", type=int, help="tokenize a deterministic sample of N rows")
    parser.add_argument("--include-contacts", action="store_true",
                        help="also count screens stored for CRM contacts")
    args = parser.parse_args()

    params = {
        "prompt_id": PROMPT_ID,
        "since": args.since,
        "until": args.until,
        "rc_since": REVERSE_CONTACT_SINCE,
        "rescrape_tol": RESCRAPE_TOLERANCE,
        **{f"era_{i}": era.start for i, era in enumerate(LABEL_ERAS)},
    }

    with connect() as conn:
        with conn.cursor() as cur:
            report_eras(cur, params)
            report_window(cur, params)
            report_position_keys(cur, params)
            if args.include_contacts:
                report_contacts(cur, params)
        if args.tokenizer:
            report_token_lengths(conn, params, args.tokenizer, args.max_seq_len, args.limit)


if __name__ == "__main__":
    main()
