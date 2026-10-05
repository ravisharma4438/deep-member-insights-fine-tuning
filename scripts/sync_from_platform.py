"""Vendor the screening prompt and output schema from a platform checkout.

The platform repo is the source of truth for both. This script copies them into
src/dmi/screening/resources/ so training data and evals use exactly what
production sends:

- system_prompt.txt: the template returned by getScreeningPrompt()
- screen_schema.json: ScreenSchema converted to JSON Schema by the AI SDK's own
  asSchema(), i.e. the schema generateObject sends as response_format

    python scripts/sync_from_platform.py --platform ../platform          # update
    python scripts/sync_from_platform.py --platform ../platform --check  # drift check

Exporting the schema needs the platform's node_modules (`yarn install` there).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = REPO_ROOT / "src" / "dmi" / "screening" / "resources"
PROMPT_FILE = RESOURCES / "system_prompt.txt"
SCHEMA_FILE = RESOURCES / "screen_schema.json"
META_FILE = RESOURCES / "platform_sync.json"

SCREEN_MEMBER_TS = Path("src/lib/screen/screen-member.ts")
SCREEN_DATA_TS = Path("src/lib/screen/screen-data.ts")
DATE_PLACEHOLDER = "${date}"

EXPORT_SCRIPT = """
import { asSchema } from "ai";
import { ScreenSchema } from "@/lib/screen/screen-data";
const schema = await asSchema(ScreenSchema).jsonSchema;
process.stdout.write(JSON.stringify(schema));
"""


def extract_prompt(platform: Path) -> tuple[str, str]:
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
    if "\\" in template:
        sys.exit("Prompt contains escape sequences; extend extract_prompt to unescape them.")

    prompt_id = re.search(r'MEMBER_SCREEN_PROMPT_ID = "([^"]+)"', source)
    if not prompt_id:
        sys.exit("Could not find MEMBER_SCREEN_PROMPT_ID")
    return template, prompt_id.group(1)


def export_schema(platform: Path) -> dict:
    vite_node = platform / "node_modules" / ".bin" / "vite-node"
    if not vite_node.exists():
        sys.exit(f"{vite_node} not found; run `yarn install` in {platform} first.")

    # Runs inside the platform so `ai`, zod, and the @/ alias resolve to its versions.
    # node_modules/.cache is gitignored, so the platform working tree stays clean.
    cache = platform / "node_modules" / ".cache"
    cache.mkdir(exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="dmi-", dir=cache))
    try:
        script = workdir / "export-screen-schema.ts"
        script.write_text(EXPORT_SCRIPT)
        result = subprocess.run(
            [str(vite_node), "--config", "vitest.config.ts", str(script)],
            cwd=platform,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if result.returncode != 0:
        sys.exit(f"Schema export failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout)


def last_commit(platform: Path, *paths: Path) -> str | None:
    out = subprocess.run(
        ["git", "-C", str(platform), "log", "-1", "--format=%H", "--", *map(str, paths)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", type=Path, default=REPO_ROOT.parent / "platform")
    parser.add_argument("--check", action="store_true", help="exit 1 if the vendored copies are stale")
    args = parser.parse_args()

    template, prompt_id = extract_prompt(args.platform)
    schema_text = json.dumps(export_schema(args.platform), indent=2) + "\n"

    if args.check:
        stale = [
            path.name
            for path, fresh in ((PROMPT_FILE, template), (SCHEMA_FILE, schema_text))
            if not path.exists() or path.read_text() != fresh
        ]
        if stale:
            sys.exit(f"Out of sync with platform: {', '.join(stale)}. Run without --check to update.")
        print(f"Prompt and schema in sync with platform (promptId {prompt_id}).")
        return

    meta = {
        "prompt_id": prompt_id,
        "prompt_source": str(SCREEN_MEMBER_TS),
        "prompt_commit": last_commit(args.platform, SCREEN_MEMBER_TS),
        "schema_source": str(SCREEN_DATA_TS),
        "schema_commit": last_commit(args.platform, SCREEN_DATA_TS, Path("src/lib/constants/member-criteria.ts")),
    }
    RESOURCES.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(template)
    SCHEMA_FILE.write_text(schema_text)
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Synced prompt (promptId {prompt_id}) and schema from {args.platform}")


if __name__ == "__main__":
    main()
