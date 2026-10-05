import modal

quantize_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install("llmcompressor", "transformers", "accelerate", "torch")
)

app = modal.App("quantize-fp8")
model_volume = modal.Volume.from_name("my-model-vol")

@app.function(
    image=quantize_image,
    volumes={"/data": model_volume},
    gpu="L4", 
    timeout=3600,
)
def quantize_model():
    # UPDATE: We now import standard AutoModelForCausalLM and the oneshot function
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    SOURCE_DIR = "/data/model_files"
    DEST_DIR = "/data/model_files_fp8"

    print(f"Loading unquantized model from {SOURCE_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_DIR)
    
    # UPDATE: Load directly using standard Hugging Face class
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_DIR, 
        device_map="auto", 
        torch_dtype="auto"
    )

    print("Configuring FP8 Dynamic Quantization...")
    recipe = QuantizationModifier(
        targets="Linear", 
        scheme="FP8_DYNAMIC", 
        ignore=["lm_head"]
    )
    
    print("Applying quantization (this might take a few minutes)...")
    # UPDATE: Apply the recipe using oneshot()
    oneshot(model=model, recipe=recipe)

    print(f"Saving FP8 model to {DEST_DIR}...")
    model.save_pretrained(DEST_DIR)
    tokenizer.save_pretrained(DEST_DIR)
    
    model_volume.commit()
    print("✅ Quantization complete and volume committed!")