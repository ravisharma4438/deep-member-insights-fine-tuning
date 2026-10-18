"""Copy a model into the Modal volume the server reads from.

    # A trained (merged) model from S3
    modal run serving/load_model.py \
        --s3-uri s3://<bucket>/dmi/training-jobs/<job>/output/model/merged \
        --name gemma-4-12b-<date>

    # An untuned base model from Hugging Face, e.g. as an eval baseline
    modal run serving/load_model.py --hf-id google/gemma-4-12B-it --name gemma-4-12b-base

S3 copies use the Modal secret named by $DMI_MODAL_AWS_SECRET (default
`aws-secret`), which must hold AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
AWS_REGION with read access to the bucket.
"""

from __future__ import annotations

import os

import modal

VOLUME_NAME = "dmi-models"
AWS_SECRET = os.environ.get("DMI_MODAL_AWS_SECRET", "aws-secret")

app = modal.App("dmi-load-model")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("boto3", "huggingface_hub")


def _empty_destination(name: str):
    from pathlib import Path

    dest = Path("/models") / name
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"/models/{name} already exists; pick a new name (names are versions).")
    return dest


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

    import boto3

    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    dest = _empty_destination(name)

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


@app.function(image=image, volumes={"/models": volume}, timeout=3600, cpu=4, memory=8192)
def download_from_hf(hf_id: str, name: str) -> list[str]:
    from huggingface_hub import snapshot_download

    dest = _empty_destination(name)
    snapshot_download(hf_id, local_dir=str(dest))
    files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts)
    volume.commit()
    return files


@app.local_entrypoint()
def main(name: str, s3_uri: str = "", hf_id: str = "") -> None:
    if bool(s3_uri) == bool(hf_id):
        raise SystemExit("Pass exactly one of --s3-uri (trained model) or --hf-id (base model).")
    files = copy_from_s3.remote(s3_uri, name) if s3_uri else download_from_hf.remote(hf_id, name)
    print(f"Copied {len(files)} files to volume {VOLUME_NAME}:/{name}")
    # A base model gets its own app so it doesn't replace the production deployment
    app_env = "DMI_SERVE_APP=dmi-screening-base " if hf_id else ""
    print(f"Serve it with:  {app_env}DMI_SERVE_MODEL={name} modal deploy serving/vllm_server.py")
