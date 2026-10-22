"""How many stale members Reverse Contact already has fresh in its cache.

    python scripts/rc_cache_check.py --sample 2000   # estimate, ~1 h at the default 30/min
    python scripts/rc_cache_check.py                 # every member (~22 h)

Takes members with a real screen, a /in/ URL, tier not Unknown/Blacklist,
and stored LinkedIn data dated more than --older-than-days ago (members with
no dated LinkedIn data are left out: they are mostly profiles that couldn't be
scraped), and asks Reverse Contact's free check endpoint
(POST /v2/fetch/persons/check) when it last fetched each profile. On a
subscription, a fetch is free when that copy is under 90 days old and costs
one credit otherwise, so the share of fresh copies is the share of free
re-scrapes.

Free (no credits used), but RC requires at least 2 credits in the workspace.

RC's rate limits are per workspace and shared by every endpoint, so this
competes with the platform's own scraping. It paces itself (--per-minute,
default 30), pauses whenever RC reports fewer than --keep-free requests left
in the current minute, and stops once the workspace's daily allowance drops
below --daily-keep-free-pct percent.
Reads the platform DB read-only. Needs DATABASE_URL and REVERSE_CONTACT_API_KEY
in .env. Progress is appended to outputs/rc_cache_check/<run>.csv, and rerunning
with the same --run skips profiles already checked, so a full run can be
stopped and resumed. The report is printed and saved to
outputs/rc_cache_check/<run>-summary.txt.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmi import settings  # noqa: E402

CHECK_URL = "https://api.reversecontact.com/v2/fetch/persons/check"
FREE_CACHE_DAYS = 90

COHORT_SQL = """
SELECT linkedin, scraped_at FROM (
  SELECT linkedin,
         CASE WHEN linkedin_data->>'date' ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
              THEN (linkedin_data->>'date')::timestamptz END AS scraped_at
  FROM members
  WHERE linkedin ILIKE '%%/in/%%'
    AND jsonb_typeof(screen->'response') = 'object'
    AND (tier IS NULL OR tier NOT IN ('Unknown', 'Blacklist'))
) m
WHERE scraped_at < now() - make_interval(days => %(days)s)
"""

FIELDS = ["url", "status", "rc_last_update", "rc_age_days", "our_scraped_at", "rc_newer_than_ours"]


def sanitize_linkedin_url(url: str) -> str | None:
    """Port of the platform's sanitizeLinkedInUrl; None where it would throw."""
    s = unquote(url.lower().strip())
    if len(s) >= 4 and all(c.isalnum() or c in "_-" for c in s):
        return f"https://linkedin.com/in/{s}"
    if s.startswith("/in/"):
        s = f"https://linkedin.com{s}"
    else:
        for prefix in ("https://www.", "http://www.", "https://", "http://"):
            if s.startswith(prefix):
                s = "https://" + s[len(prefix):]
                break
        if not s.startswith("https://"):
            s = f"https://{s}"
    s = s.split("#")[0]
    if "linkedin.com/" not in s or "/company/" in s:
        return None
    s = s.replace("/mwlite/", "/")
    if "/in/" not in s:
        s = s.replace("linkedin.com/", "linkedin.com/in/", 1)
    s = s.split("?")[0]
    profile = s.split("/in/", 1)[1] if "/in/" in s else ""
    if not profile or s.endswith(("/in/me", "/in/me/")):
        return None
    return s.rstrip("/") if s.endswith("/") else s


def load_cohort(older_than_days: int) -> dict[str, datetime]:
    from dmi.db import connect

    with connect() as conn, conn.cursor() as cur:
        cur.execute(COHORT_SQL, {"days": older_than_days})
        rows = cur.fetchall()
    cohort: dict[str, datetime] = {}
    for raw_url, scraped_at in rows:
        url = sanitize_linkedin_url(raw_url or "")
        if url:
            cohort.setdefault(url, scraped_at)
    print(f"{len(rows):,} members with LinkedIn data over {older_than_days} days old; "
          f"{len(cohort):,} distinct valid profile URLs")
    return cohort


def check_profile(url: str, api_key: str) -> dict:
    """POST the check endpoint, retrying rate limits and transient errors."""
    body = json.dumps({"url": url}).encode()
    for attempt in range(5):
        request = urllib.request.Request(
            CHECK_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return {"http": 200, "body": json.load(response)}
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(f"Reverse Contact rejected the API key ({e.code}). Check REVERSE_CONTACT_API_KEY.")
            if e.code == 402:
                raise SystemExit("Reverse Contact says NO_CREDITS: the free check needs at least 2 credits in the workspace.")
            if e.code == 429 or e.code >= 500:
                try:
                    retry_after = json.load(e)["error"]["details"]["retryAfter"]
                except (KeyError, TypeError, ValueError):
                    retry_after = e.headers.get("Retry-After")
                time.sleep(float(retry_after or 5 * (attempt + 1)) + 1)
                continue
            try:
                return {"http": e.code, "body": json.load(e)}
            except (json.JSONDecodeError, ValueError):
                return {"http": e.code, "body": None}
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            time.sleep(5 * (attempt + 1))
    return {"http": None, "body": None}


def rate_limits(quotas: object, path: str = "quotas") -> list[tuple[str, dict]]:
    """Every {limit, left, ...} object in RC's `quotas`, with its path."""
    if not isinstance(quotas, dict):
        return []
    if "left" in quotas and "limit" in quotas:
        return [(path, quotas)]
    return [found for key, value in quotas.items() for found in rate_limits(value, f"{path}.{key}")]


def throttle(body: dict | None, keep_free: int, daily_keep_free_pct: float) -> str | None:
    """Wait out a nearly used-up minute; return a reason to stop if the daily allowance is low."""
    for path, limit in rate_limits((body or {}).get("quotas")):
        name = path.lower()
        left, total = int(limit.get("left") or 0), int(limit.get("limit") or 0)
        if "minute" in name and left <= keep_free:
            try:
                reset = datetime.fromisoformat(str(limit["nextReset"]).replace("Z", "+00:00"))
                wait = (reset - datetime.now(timezone.utc)).total_seconds() + 1
            except (KeyError, ValueError):
                wait = 60
            time.sleep(min(max(wait, 1), 61))
        elif ("day" in name or "daily" in name) and total and left < total * daily_keep_free_pct / 100:
            return f"only {left:,} of {total:,} daily requests left ({path}); stopping to leave them for the platform"
    return None


def classify(result: dict, our_scraped_at: datetime | None, now: datetime) -> dict:
    data = (result["body"] or {}).get("data") or {}
    row = {"status": "error", "rc_last_update": "", "rc_age_days": "", "rc_newer_than_ours": ""}
    if result["http"] == 404 or (result["http"] == 200 and data.get("exists") is False):
        row["status"] = "not_in_rc"
    elif result["http"] == 200 and data.get("exists"):
        last_update = datetime.fromisoformat(str(data["lastUpdate"]).replace("Z", "+00:00"))
        age = now - last_update
        row.update(
            status="fresh" if age < timedelta(days=FREE_CACHE_DAYS) else "stale",
            rc_last_update=last_update.isoformat(),
            rc_age_days=age.days,
            rc_newer_than_ours=("" if our_scraped_at is None else int(last_update > our_scraped_at)),
        )
    return row


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(rows: list[dict], cohort: dict[str, datetime]) -> list[str]:
    """The report, as lines."""
    counts = Counter(r["status"] for r in rows)
    checked = sum(counts[s] for s in ("fresh", "stale", "not_in_rc"))
    if not checked:
        return ["Nothing checked yet."]
    cohort_size = len(cohort)
    sampled = checked < cohort_size
    lines = [f"Checked {checked:,} of {cohort_size:,} profiles"
             + (f" ({counts['error']} errors, not counted)" if counts["error"] else "")]
    labels = {
        "fresh": f"RC copy under {FREE_CACHE_DAYS} days old (free re-scrape)",
        "stale": f"RC copy {FREE_CACHE_DAYS}+ days old (1 credit)",
        "not_in_rc": "not in RC's database (1 credit, or free if the profile is gone)",
    }
    for status, label in labels.items():
        share = counts[status] / checked
        interval = ""
        if sampled:
            low, high = wilson(counts[status], checked)
            interval = f"  (95% range {low:.0%}–{high:.0%})"
        lines.append(f"  {label:62s} {counts[status]:>7,}  {share:6.1%}{interval}")

    paid_share = (counts["stale"] + counts["not_in_rc"]) / checked
    paid = paid_share * cohort_size
    lines.append("")
    lines.append(f"Estimated paid re-scrapes for all {cohort_size:,}: ~{paid:,.0f} credits "
                 f"(~${paid * 0.009:,.0f}–{paid * 0.02:,.0f} at $0.009–0.02/credit)")

    ages = sorted(int(r["rc_age_days"]) for r in rows if r["rc_age_days"] not in ("", None))
    if ages:
        lines.append(f"Age of RC's copies: median {ages[len(ages) // 2]} days")
    newer = [r["rc_newer_than_ours"] for r in rows if r["rc_newer_than_ours"] in (0, 1, "0", "1")]
    if newer:
        lines.append(f"RC's copy is newer than our stored data for "
                     f"{sum(int(v) for v in newer) / len(newer):.0%} of profiles")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=0, help="check a random sample of this many (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--older-than-days", type=int, default=365, help="cohort: LinkedIn data older than this")
    parser.add_argument("--run", help="results file name (default: all-<days>d or sample<N>-<days>d)")
    parser.add_argument("--per-minute", type=float, default=30, help="request pace (RC's workspace limit is ~60)")
    parser.add_argument("--keep-free", type=int, default=20,
                        help="pause when RC reports this few requests left in the current minute")
    parser.add_argument("--daily-keep-free-pct", type=float, default=25,
                        help="stop when less than this %% of the workspace's daily requests is left")
    args = parser.parse_args()

    api_key = settings.require("REVERSE_CONTACT_API_KEY")
    cohort = load_cohort(args.older_than_days)
    # Random order even for a full run, so a run stopped partway is still a fair sample
    urls = sorted(cohort)
    random.Random(args.seed).shuffle(urls)
    if args.sample:
        urls = urls[: args.sample]

    run = args.run or f"{f'sample{args.sample}' if args.sample else 'all'}-{args.older_than_days}d"
    out = settings.REPO_ROOT / "outputs" / "rc_cache_check" / f"{run}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: list[dict] = []
    if out.exists():
        with out.open() as f:
            done = [r for r in csv.DictReader(f) if r["status"] != "error"]
    checked = {r["url"] for r in done}
    todo = [u for u in urls if u not in checked]
    interval = 60 / args.per_minute
    print(f"{len(checked):,} already checked in {out.name}; {len(todo):,} to go "
          f"(~{len(todo) * interval / 60:.0f} min at {args.per_minute:g}/min)")

    now = datetime.now(timezone.utc)
    new_file = not out.exists()
    try:
        with out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            for i, url in enumerate(todo, 1):
                started = time.monotonic()
                our_scraped_at = cohort[url]
                result = check_profile(url, api_key)
                if i == 1:
                    limits = rate_limits((result["body"] or {}).get("quotas"))
                    print("RC limits: " + ("; ".join(f"{p} {l.get('left')}/{l.get('limit')} left" for p, l in limits)
                                          if limits else "not reported; pacing only"), flush=True)
                row = {"url": url, "our_scraped_at": our_scraped_at.isoformat() if our_scraped_at else "",
                       **classify(result, our_scraped_at, now)}
                writer.writerow(row)
                done.append(row)
                if i % 100 == 0:
                    f.flush()
                    print(f"  {i:,}/{len(todo):,}  {dict(Counter(r['status'] for r in done))}", flush=True)
                stop_reason = throttle(result["body"], args.keep_free, args.daily_keep_free_pct)
                if stop_reason:
                    print(f"\nStopped: {stop_reason}. Rerun later to resume.", flush=True)
                    break
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nStopped; rerun the same command to resume.")
    report = [
        f"Reverse Contact cache check, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"Cohort: members whose stored LinkedIn data is over {args.older_than_days} days old "
        f"({len(cohort):,} distinct profiles); run '{run}', seed {args.seed}",
        "",
        *summarize(done, cohort),
        "",
        f"Per-profile results: {out}",
    ]
    summary_path = out.with_name(f"{run}-summary.txt")
    summary_path.write_text("\n".join(report) + "\n")
    print("\n" + "\n".join(report))
    print(f"Report saved to {summary_path}")


if __name__ == "__main__":
    main()
