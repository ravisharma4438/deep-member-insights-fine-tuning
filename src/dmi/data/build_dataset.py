"""Build the train/val/test dataset from production screens and upload it to S3.

Read-only against the database. From the repo root (Studio terminal or `!` cell):

    python -m dmi.data.build_dataset                       # current-model window, upload to S3
    python -m dmi.data.build_dataset --no-upload           # local files only
    python -m dmi.data.build_dataset --since 2026-03-16T17:43:49Z --name strict-window

Each row is a TRL conversational prompt-completion example, built exactly as the
platform builds the request:

    {"id", "screened_at", "provider", "era",
     "prompt": [system (prompt dated to the screen), user (JSON.stringify(relevantData))],
     "completion": [assistant (label: schema-ordered, tier dropped)]}

Rows are model-agnostic: length filtering for a model's max_seq_len happens at
training time. Emails never leave the database; ids are salted hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dmi import settings
from dmi.data.queries import USABLE_ROWS, params as query_params
from dmi.db import connect
from dmi.screening import labels
from dmi.screening.eras import CURRENT_MODEL_SINCE, era_for
from dmi.screening.inputs import user_message
from dmi.screening.prompt import system_prompt

SPLITS = ("train", "val", "test")
_SPLIT_SALT = "dmi-split-v1"
_ID_SALT = "dmi-id-v1"


def _parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def sample_id(email: str) -> str:
    return hashlib.sha256(f"{_ID_SALT}:{email.lower()}".encode()).hexdigest()[:20]


def split_for(email: str, val_frac: float, test_frac: float) -> str:
    """Stable per member: rebuilding with more data never moves someone between splits."""
    bucket = int(hashlib.sha256(f"{_SPLIT_SALT}:{email.lower()}".encode()).hexdigest(), 16) % 10_000
    if bucket < test_frac * 10_000:
        return "test"
    if bucket < (test_frac + val_frac) * 10_000:
        return "val"
    return "train"


def build_example(email: str, screened_at: datetime, provider: str, label: dict, linkedin_data: dict) -> dict:
    """`label` is a normalized response (labels.normalize)."""
    era = era_for(screened_at)
    return {
        "id": sample_id(email),
        "screened_at": screened_at.astimezone(timezone.utc).isoformat(),
        "provider": provider,
        "era": era.key if era else None,
        "prompt": [
            {"role": "system", "content": system_prompt(screened_at)},
            {"role": "user", "content": user_message(linkedin_data)},
        ],
        "completion": [{"role": "assistant", "content": labels.serialize(label)}],
    }


def upload(local_dir: Path, name: str) -> str:
    import boto3

    dest = settings.s3_uri("datasets", name)
    bucket, _, prefix = dest.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    for path in sorted(local_dir.iterdir()):
        s3.upload_file(str(path), bucket, f"{prefix}/{path.name}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", type=_parse_ts, default=CURRENT_MODEL_SINCE)
    parser.add_argument("--until", type=_parse_ts, default=None, help="default: now")
    parser.add_argument("--name", help="dataset name (default: screens-<since>-<until>)")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=settings.REPO_ROOT / "data" / "datasets")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    until = args.until or datetime.now(timezone.utc)
    name = args.name or f"screens-{args.since:%Y%m%d}-{until:%Y%m%d}"
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    skipped: Counter = Counter()
    files = {split: (out_dir / f"{split}.jsonl").open("w") for split in SPLITS}
    try:
        with connect() as conn, conn.cursor(name="build_dataset") as cur:
            cur.itersize = 500
            cur.execute(USABLE_ROWS, query_params(args.since, until))
            for n, (email, screened_at, _scraped_at, provider, response, linkedin_data) in enumerate(cur, 1):
                label = labels.normalize(response)
                error = labels.validation_error(label)
                if error:
                    skipped[f"label fails schema ({error})"] += 1
                    continue
                split = split_for(email, args.val_frac, args.test_frac)
                example = build_example(email, screened_at, provider, label, linkedin_data)
                files[split].write(json.dumps(example, ensure_ascii=False) + "\n")
                counts[(split, provider)] += 1
                if n % 2000 == 0:
                    print(f"  processed {n} rows...", flush=True)
    finally:
        for f in files.values():
            f.close()

    sync_meta = json.loads(
        (Path(labels.__file__).parent / "resources" / "platform_sync.json").read_text()
    )
    manifest = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_window": {"since": args.since.isoformat(), "until": until.isoformat()},
        "split_fracs": {"val": args.val_frac, "test": args.test_frac},
        "counts": {
            split: {
                "total": sum(v for (s, _), v in counts.items() if s == split),
                **{p: v for (s, p), v in counts.items() if s == split},
            }
            for split in SPLITS
        },
        "skipped": dict(skipped),
        "platform": sync_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nDataset {name} -> {out_dir}")
    for split in SPLITS:
        c = manifest["counts"][split]
        detail = ", ".join(f"{k} {v}" for k, v in c.items() if k != "total")
        print(f"  {split:5s} {c['total']:6d}  ({detail})")
    for reason, n in skipped.most_common():
        print(f"  skipped {n}: {reason}")

    if not args.no_upload:
        print(f"Uploaded to {upload(out_dir, name)}")


if __name__ == "__main__":
    main()
