"""OpenAI-compatible vLLM server for the fine-tuned screening model, on Modal.

    DMI_SERVE_MODEL=gemma-4-12b-20261010 modal deploy serving/vllm_server.py

The model must already be in the `dmi-models` volume (serving/load_model.py).
Clients call POST <url>/v1/chat/completions with
response_format={"type": "json_schema", ...}, which is what the platform's
AI SDK generateObject sends.

Auth is Modal proxy auth (on by default for servers). Create a token with
`modal workspace proxy-tokens` and use "<token-id>.<token-secret>" as the
OpenAI API key: clients then send `Authorization: Bearer <id>.<secret>`.

The model answers to two names: its versioned name, which responses report back
(so callers can record which model produced an output), and the stable alias
`dmi-screen`, which callers request.

Deploy-time settings (environment variables):
    DMI_SERVE_MODEL               required, directory name in the dmi-models volume
    DMI_SERVE_GPU                 default L40S (48 GB; fits a 12B model in bf16)
    DMI_SERVE_QUANTIZATION        e.g. fp8 to quantize at load time (needed for a 12B model on an L4)
    DMI_SERVE_MAX_LEN             default 16384 tokens (prompt + output)
    DMI_SERVE_MIN_CONTAINERS      default 0 (scale to zero)
    DMI_SERVE_MAX_CONTAINERS      default 2
    DMI_SERVE_TARGET_CONCURRENCY  default 64 requests per container before scaling out
    DMI_SERVE_SCALEDOWN_SECONDS   default 600
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

VLLM_VERSION = "0.30.0"
ALIAS = "dmi-screen"
PORT = 8000

SETTINGS = {
    "DMI_SERVE_MODEL": os.environ.get("DMI_SERVE_MODEL", ""),
    "DMI_SERVE_GPU": os.environ.get("DMI_SERVE_GPU", "L40S"),
    "DMI_SERVE_QUANTIZATION": os.environ.get("DMI_SERVE_QUANTIZATION", ""),
    "DMI_SERVE_MAX_LEN": os.environ.get("DMI_SERVE_MAX_LEN", "16384"),
    "DMI_SERVE_MIN_CONTAINERS": os.environ.get("DMI_SERVE_MIN_CONTAINERS", "0"),
    "DMI_SERVE_MAX_CONTAINERS": os.environ.get("DMI_SERVE_MAX_CONTAINERS", "2"),
    "DMI_SERVE_TARGET_CONCURRENCY": os.environ.get("DMI_SERVE_TARGET_CONCURRENCY", "64"),
    "DMI_SERVE_SCALEDOWN_SECONDS": os.environ.get("DMI_SERVE_SCALEDOWN_SECONDS", "600"),
}
if not SETTINGS["DMI_SERVE_MODEL"]:
    raise SystemExit("Set DMI_SERVE_MODEL to a model directory in the dmi-models volume.")
MODEL = SETTINGS["DMI_SERVE_MODEL"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"VLLM_LOG_STATS_INTERVAL": "10"})
)
models_volume = modal.Volume.from_name("dmi-models")
vllm_cache = modal.Volume.from_name("dmi-vllm-cache", create_if_missing=True)

app = modal.App("dmi-screening")


@app.server(
    image=image,
    gpu=SETTINGS["DMI_SERVE_GPU"],
    volumes={"/models": models_volume, "/root/.cache/vllm": vllm_cache},
    env=SETTINGS,  # the container re-imports this module and reads the same settings
    port=PORT,
    startup_timeout=15 * 60,
    min_containers=int(SETTINGS["DMI_SERVE_MIN_CONTAINERS"]),
    max_containers=int(SETTINGS["DMI_SERVE_MAX_CONTAINERS"]),
    target_concurrency=float(SETTINGS["DMI_SERVE_TARGET_CONCURRENCY"]),
    scaledown_window=int(SETTINGS["DMI_SERVE_SCALEDOWN_SECONDS"]),
)
class Server:
    @modal.enter()
    def start(self) -> None:
        model_dir = f"/models/{MODEL}"
        with open(f"{model_dir}/config.json") as f:
            config = json.load(f)

        cmd = [
            "vllm", "serve", model_dir,
            "--served-model-name", MODEL, ALIAS,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--max-model-len", SETTINGS["DMI_SERVE_MAX_LEN"],
            "--gpu-memory-utilization", "0.92",
            # Every request shares the ~3.7k-token system prompt (up to its date line)
            "--enable-prefix-caching",
            "--uvicorn-log-level=info",
        ]
        if SETTINGS["DMI_SERVE_QUANTIZATION"]:
            cmd += ["--quantization", SETTINGS["DMI_SERVE_QUANTIZATION"]]
        if "vision_config" in config or "audio_config" in config:
            # Text-only workload: skip multimodal memory profiling
            cmd += ["--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0})]

        print("Starting:", " ".join(cmd), flush=True)
        subprocess.Popen(cmd)
