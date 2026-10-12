import modal
import os
import json
import uuid
from typing import Optional, Union, List, Dict, Any 

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 1. Image Setup
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install("vllm==0.15.1", "fastapi[standard]", "outlines", "transformers")
)

app = modal.App("nonprofit-llm")

model_volume = modal.Volume.from_name("my-model-vol")
MODEL_DIR = "/data/model_files_fp8"

auth_scheme = HTTPBearer()

def _normalize_schema(schema: Optional[Union[str, dict]]) -> Optional[dict]:
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
    timeout=1200,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()
    def load_model(self):
        import vllm
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from transformers import AutoTokenizer

        print("vLLM version:", vllm.__version__)
        print(f"Loading AsyncLLMEngine from {MODEL_DIR}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        # UPDATE 2: Add the optimization flags
        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            max_model_len=11500,
            
            # Bumped up from 10. Start at 40. If you hit OOMs, dial back to 30.
            # If it runs smoothly, you can try pushing to 50 or 60.
            max_num_seqs=40,              
            
            gpu_memory_utilization=0.95,
            
            # New Optimizations:
            kv_cache_dtype="fp8",         # Compresses the KV cache context
            enable_chunked_prefill=True,  # Prevents OOMs on massive inputs
            enable_prefix_caching=True,   # Caches your system prompts
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    # FIX 2: Make generate an async function so Modal can cancel cleanly
    @modal.method()
    async def generate(
        self,
        messages,       # No type hint
        schema = None,  # <--- Drop the type hint here too!
        temp: float = 0.1,
        top_p: float = 0.1,
        max_tokens: int = 2500,
    ):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        schema_dict = _normalize_schema(schema)

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

        # FIX 3: AsyncLLMEngine requires a unique request_id and returns an async generator
        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text

@app.function(
    image=vllm_image,
    secrets=[modal.Secret.from_name("my-custom-secret")],
    timeout=1200,
)
# FIX 4: Add concurrency to the API proxy so Modal doesn't boot 50 web servers
@modal.concurrent(max_inputs=100) 
@modal.fastapi_endpoint(method="POST", label="inference")
async def api(
    item: dict,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    expected_key = os.environ.get("MY_API_KEY")
    if expected_key is None:
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing API key.")

    if token.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if "messages" not in item:
        raise HTTPException(status_code=400, detail="Missing required field: messages")

    model = Model()
    schema = item.get("schema", None)

    try:
        # FIX 5: Use .remote.aio() to await the Modal class method asynchronously
        return await model.generate.remote.aio(
            messages=item["messages"],
            schema=schema,
            temp=item.get("temp", 0.1),
            top_p=item.get("top_p", 0.1),
            max_tokens=item.get("max_tokens", 2500),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))