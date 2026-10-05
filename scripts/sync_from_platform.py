"""Vendor the screening prompt from a platform checkout.

The platform repo is the source of truth for the screening prompt. This script
copies it into src/dmi/screening/resources/ so training data is built with the
exact text production sends.

    python scripts/sync_from_platform.py --platform ../platform          # update
    python scripts/sync_from_platform.py --platform ../platform --check  # CI / pre-train drift check
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = REPO_ROOT / "src" / "dmi" / "screening" / "resources"
PROMPT_FILE = RESOURCES / "system_prompt.txt"
META_FILE = RESOURCES / "platform_sync.json"

SCREEN_MEMBER_TS = Path("src/lib/screen/screen-member.ts")
DATE_PLACEHOLDER = "${date}"


def extract(platform: Path) -> tuple[str, dict]:
    source = (platform / SCREEN_MEMBER_TS).read_text()

    fn = re.search(
        r"function getScreeningPrompt\(\): string \{.*?return `(.*?)`;\s*\n\}",
        source,
        re.DOTALL,
    )
    if not fn:
        sys.exit(f"Could not find getScreeningPrompt() in {SCREEN_MEMBER_TS}")
    template = fn.group(1)

    interpolations = re.findall(r"\$\{[^}]*\}", template)
    if interpolations != [DATE_PLACEHOLDER]:
        sys.exit(
            "Prompt interpolations changed (expected only ${date}), found: "
            f"{interpolations}. Update dmi.screening.prompt before syncing."
        )

    prompt_id = re.search(r'MEMBER_SCREEN_PROMPT_ID = "([^"]+)"', source)
    if not prompt_id:
        sys.exit("Could not find MEMBER_SCREEN_PROMPT_ID")

    commit = subprocess.run(
        ["git", "-C", str(platform), "log", "-1", "--format=%H", "--", str(SCREEN_MEMBER_TS)],
        capture_output=True,
        text=True,
    ).stdout.strip()

    meta = {
        "prompt_id": prompt_id.group(1),
        "source": str(SCREEN_MEMBER_TS),
        "source_commit": commit or None,
    }
    return template, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", type=Path, default=REPO_ROOT.parent / "platform")
    parser.add_argument("--check", action="store_true", help="exit 1 if the vendored copy is stale")
    args = parser.parse_args()

    template, meta = extract(args.platform)

    if args.check:
        current = PROMPT_FILE.read_text() if PROMPT_FILE.exists() else None
        if current != template:
            sys.exit("Vendored prompt differs from platform. Run without --check to update.")
        print(f"Prompt in sync with platform (promptId {meta['prompt_id']}).")
        return

    RESOURCES.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(template)
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {PROMPT_FILE.relative_to(REPO_ROOT)} (promptId {meta['prompt_id']}, commit {meta['source_commit']})")


if __name__ == "__main__":
    main()
