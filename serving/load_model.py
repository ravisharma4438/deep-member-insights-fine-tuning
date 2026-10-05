"""Copy a trained (merged) model from S3 into the Modal volume the server reads from.

    modal run serving/load_model.py \
        --s3-uri s3://<bucket>/dmi/training-jobs/<job>/output/model/merged \
        --name gemma-4-12b-<date>

Uses the Modal secret named by $DMI_MODAL_AWS_SECRET (default `aws-secret`),
which must hold AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION with
read access to the bucket.
"""

from __future__ import annotations

import os

import modal

VOLUME_NAME = "dmi-models"
AWS_SECRET = os.environ.get("DMI_MODAL_AWS_SECRET", "aws-secret")

app = modal.App("dmi-load-model")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("boto3")


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name(AWS_SECRET)],
    timeout=3600,
    cpu=4,
    memory=8192,
)
def copy_from_s3(s3_uri: str, name: str) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import boto3

    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    dest = Path("/models") / name
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"/models/{name} already exists; pick a new name (names are versions).")

    s3 = boto3.client("s3")
    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if not obj["Key"].endswith("/")
    ]
    if not keys:
        raise SystemExit(f"No objects under {s3_uri}")

    def fetch(key: str) -> str:
        target = dest / key[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        return str(target.relative_to(dest))

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = sorted(pool.map(fetch, keys))
    if not (dest / "config.json").exists():
        raise SystemExit(f"No config.json under {s3_uri}; point --s3-uri at the merged/ directory.")

    volume.commit()
    return files


@app.local_entrypoint()
def main(s3_uri: str, name: str) -> None:
    files = copy_from_s3.remote(s3_uri, name)
    print(f"Copied {len(files)} files to volume {VOLUME_NAME}:/{name}")
    print(f"Serve it with:  DMI_SERVE_MODEL={name} modal deploy serving/vllm_server.py")
