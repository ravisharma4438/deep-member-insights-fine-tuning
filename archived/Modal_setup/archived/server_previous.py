import modal
import os
import json
from typing import Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 1. Image Setup
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.15.1", "fastapi[standard]", "outlines", "transformers")
)

app = modal.App("nonprofit-llm")

model_volume = modal.Volume.from_name("my-model-vol")
MODEL_DIR = "/data/model_files"

auth_scheme = HTTPBearer()


def _normalize_schema(schema: Optional[Union[str, dict]]) -> Optional[dict]:
    """
    Accept schema as either:
      - dict (already parsed)
      - str (JSON string)
      - None
    Return a dict or None.
    """
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, str):
        schema = schema.strip()
        if not schema:
            return None
        try:
            return json.loads(schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema string: {e}") from e
    raise ValueError(f"Unsupported schema type: {type(schema)}")


@app.cls(
    gpu="L4",
    image=vllm_image,
    volumes={"/data": model_volume},
    scaledown_window=60,
    timeout=600,
)
class Model:
    @modal.enter()
    def load_model(self):
        import vllm
        from vllm import LLM
        from transformers import AutoTokenizer

        print("vLLM version:", vllm.__version__)
        print(f"Loading model and tokenizer from {MODEL_DIR}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        self.llm = LLM(
            model=MODEL_DIR,
            max_model_len=11500,
        )

    @modal.method()
    def generate(
        self,
        messages: list,
        schema: Optional[Union[str, dict]] = None,
        temp: float = 0.1,
        top_p: float = 0.1,
        max_tokens: int = 2500,
    ):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        # Apply Chat Template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        schema_dict = _normalize_schema(schema)

        # IMPORTANT:
        # vLLM 0.15.1 constrained decoding uses StructuredOutputsParams via SamplingParams(structured_outputs=...)
        if schema_dict is not None:
            structured = StructuredOutputsParams(json=schema_dict)
            sampling_params = SamplingParams(
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
                structured_outputs=structured,
            )
        else:
            sampling_params = SamplingParams(
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


@app.function(
    image=vllm_image,
    secrets=[modal.Secret.from_name("my-custom-secret")],
)
@modal.fastapi_endpoint(method="POST", label="inference")
def api(
    item: dict,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    # Auth Check
    expected_key = os.environ.get("MY_API_KEY")
    if expected_key is None:
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing API key.")

    if token.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate required fields
    if "messages" not in item:
        raise HTTPException(status_code=400, detail="Missing required field: messages")

    model = Model()

    schema = item.get("schema", None)

    try:
        return model.generate.remote(
            messages=item["messages"],
            schema=schema,
            temp=item.get("temp", 0.1),
            top_p=item.get("top_p", 0.1),
            max_tokens=item.get("max_tokens", 2500),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
