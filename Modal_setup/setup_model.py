import modal
import os
import subprocess

model_volume = modal.Volume.from_name("my-model-vol", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("boto3")

app = modal.App("model-setup")

@app.function(
    image=image,
    volumes={"/data": model_volume}, 
    secrets=[modal.Secret.from_name("aws-secret")], 
    timeout=3600,
    # requesting more CPU helps extracting tar files faster, reducing preemption risk
    cpu=4, 
    memory=4096 
)
def download_and_extract():
    import boto3
    
    # --- CONFIG ---
    BUCKET_NAME = "sagemaker-us-east-2-777347522666"
    MODEL_KEY = "deep-member-insights-clean-2025-12-21-20-37-11-633/output/model/model.tar.gz"
    local_tar = "/data/model.tar.gz"
    extract_dir = "/data/model_files"
    
    # --- STEP 1: DOWNLOAD (with skip logic) ---
    # Check if we already have the file (from a previous interrupted run)
    if os.path.exists(local_tar):
        print("resume: 💾 Found existing model.tar.gz, checking size...")
        # Optional: You could check file size here against S3 if you wanted to be super safe
        # but for now, we assume if it exists, the previous run's 'commit' succeeded.
        print("resume: ⏭️ Skipping download.")
    else:
        print(f"🚀 Starting download from {BUCKET_NAME}...")
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.download_file(BUCKET_NAME, MODEL_KEY, local_tar)
        print("✅ Download complete.")
        
        # CRITICAL FIX: Commit immediately after download.
        # If extraction crashes, we won't have to download again.
        print("💾 Committing volume to save download progress...")
        model_volume.commit() 

    # --- STEP 2: EXTRACT ---
    print("📦 Extracting tarball (this takes a while)...")
    
    # Clean previous failed extraction attempts
    if os.path.exists(extract_dir):
        import shutil
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract
    subprocess.run(["tar", "-xvf", local_tar, "-C", extract_dir], check=True)
    
    # Cleanup
    print("🧹 Removing tar file to free space...")
    os.remove(local_tar)
    
    # Final Commit
    print("💾 Final commit...")
    model_volume.commit()
    
    print(f"🎉 Success! Model stored in Modal Volume at '{extract_dir}'")
    print("Files found:", os.listdir(extract_dir))