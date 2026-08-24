# ===========================================================================
# Fill the GCS bucket with everything the gateway needs.
# RUN THIS IN A COLAB NOTEBOOK on a **GPU runtime** (Runtime -> Change runtime
# type -> T4 GPU) — the merge step in cell 4 needs a GPU.
#
# MULTI-ACCOUNT SETUP (NO DRIVE MOUNT)
# ====================================
# This script uses TWO accounts. No Drive mounting needed!
#
#   1. GCP + Kaggle Account (cells 1a, 1b)
#      → The Google account that owns your GCP project (farmervision-prod)
#      → Same account as your Kaggle credentials
#      → Manages the bucket, downloads from Kaggle datasets
#
#   2. Drive Account (for getting shareable links only)
#      → A DIFFERENT Google account with your RAG artifacts
#      → You'll provide shareable Drive links (no login needed in Colab)
#      → Use gdown to download directly without mounting
#
# Sources (matches scripts/run_e2e_eval.py):
#   * RAG snapshot + manifest -> from your DRIVE ACCOUNT (via shareable link)
#   * Generator LoRA adapter  -> from your DRIVE ACCOUNT (via shareable link)
#   * IEG + Vision models     -> downloaded via KAGGLE ACCOUNT (kagglehub)
#
# Paste each cell (1a, 1b, 2, 3, 4, 5, 6) into its own Colab cell.
#
# ===========================================================================
# TROUBLESHOOTING: "Bucket does not exist" or permission errors?
# ===========================================================================
# This means you signed in with the WRONG Google account in cell 1a.
# 
# Solution:
#   1. Runtime → Restart runtime (to clear the wrong login)
#   2. Re-run cell 1a with the CORRECT account (the one that owns farmervision-prod)
#   3. Re-run cell 2 and the rest
#
# How to check which account owns your GCP project:
#   - Go to: https://console.cloud.google.com/iam-admin/iam
#   - Look at "Project Owner" role — that's the account to use
# ===========================================================================

# --- cell 1a: authenticate to GCP (your GCP/Kaggle account) -----------------
# This will open a browser popup to sign in with the Google account linked to your
# GCP project. This is also your Kaggle account.
# 
# ⚠️  IMPORTANT: Make sure you sign in with the CORRECT account — the one that
# owns or has access to your farmervision-prod GCP project. If you use the wrong
# account, the bucket creation will fail later.
from google.colab import auth
print("Step 1a: Signing in to GCP...")
print("  A browser popup will appear. Sign in with the Google account that OWNS your GCP project.")
print("  ⚠️  DO NOT use a different Google account — it must have access to: farmervision-prod")
auth.authenticate_user()
print("  ✓ GCP authenticated")
print("  Note: If you used the wrong account, restart the kernel and try again.")


# --- cell 1b: authenticate to Kaggle (same account as GCP) -------------------
# Kaggle uses API credentials (username + token), not Google login.
import kagglehub
print("\nStep 1b: Signing in to Kaggle...")
print("  Paste your Kaggle username and API token when prompted below.")
print("  (Get these from https://www.kaggle.com/settings/account)")
kagglehub.login()
print("  ✓ Kaggle authenticated")

# --- cell 2: settings (edit these) -----------------------------------------
# EDIT THESE to match your setup:
PROJECT  = "your-gcp-project-id"                       # your GCP project ID
BUCKET   = "gs://your-globally-unique-bucket-name"    # must match .env BUCKET
REGION   = "us-central1"                             # GCS region (e.g. us-central1, asia-south1)
HF_TOKEN = "hf_replace_with_your_huggingface_token"   # gated Gemma token

# ===== DRIVE SHAREABLE LINKS =====
# Get these from your OTHER Google account (where RAG artifacts are stored).
# For each folder/file, right-click -> Share -> "Anyone with the link" -> copy link.
# Replace the fake links below with your actual shareable links.
#
# RAG Snapshot folder link: the folder containing manifest.json and snapshot files
DRIVE_RAG_FOLDER_LINK = "https://drive.google.com/drive/folders/YOUR-RAG-FOLDER-ID"
# LoRA Adapter folder link: the folder containing adapter_config.json, adapter_model.bin, etc.
DRIVE_ADAPTER_FOLDER_LINK = "https://drive.google.com/drive/folders/YOUR-ADAPTER-FOLDER-ID"

print("=== Account & link verification ===")
print(f"GCP Project: {PROJECT}")
print(f"GCS Bucket: {BUCKET}")
print(f"RAG folder link: {DRIVE_RAG_FOLDER_LINK}")
print(f"Adapter folder link: {DRIVE_ADAPTER_FOLDER_LINK}")
print("\n⚠️  Replace the folder links above with your actual shareable Drive links.")
print("    Links must be 'Anyone with link can view/download' — not private.")

# Validate links
if "YOUR-DRIVE-FOLDER-ID" in DRIVE_RAG_FOLDER_LINK or "YOUR-ADAPTER-FOLDER-ID" in DRIVE_ADAPTER_FOLDER_LINK:
    print("\n❌ ERROR: Placeholder links found! Edit cell 2 with your actual Drive links and re-run.")
    raise ValueError("Missing Drive links — edit DRIVE_RAG_FOLDER_LINK and DRIVE_ADAPTER_FOLDER_LINK")

# Create the bucket and set the GCP project
print("\nVerifying GCP project access...")
get_ipython().system(f"gcloud config set project {PROJECT}")

# Try to verify project access by listing buckets
print(f"Checking access to project: {PROJECT}")
result = get_ipython().system(f"gcloud projects describe {PROJECT} --format='value(projectId)' 2>&1")

# Create the bucket with error checking
print(f"\nCreating/verifying bucket: {BUCKET}")
create_result = get_ipython().system(f"gcloud storage buckets create {BUCKET} --location={REGION} 2>&1")

# Check if bucket exists or was created
check_result = get_ipython().system(f"gcloud storage buckets describe {BUCKET} 2>&1")

if "already exists" in str(check_result) or "Location" in str(check_result) or "storage_class" in str(check_result):
    print(f"✓ Bucket exists and is accessible: {BUCKET}")
elif "does not exist" in str(check_result) or "404" in str(check_result):
    print(f"\n❌ ERROR: Cannot access bucket {BUCKET}")
    print(f"   Reasons:")
    print(f"   1. You may be signed in with the WRONG Google account (not the one that owns {PROJECT})")
    print(f"   2. The bucket doesn't exist and couldn't be created due to permissions")
    print(f"   3. Your GCP account doesn't have storage.buckets.create permission")
    print(f"\n   Solution:")
    print(f"   - Restart the kernel (Runtime → Restart runtime)")
    print(f"   - Re-run cell 1a and sign in with the correct account (the one that owns {PROJECT})")
    print(f"   - Then re-run cell 2")
    raise RuntimeError(f"Cannot access bucket {BUCKET} — check your GCP account")
else:
    print(f"✓ GCP project configured: {PROJECT}")
    print(f"✓ Bucket ready: {BUCKET}")

# Install gdown for downloading from Drive links
get_ipython().system("pip install -q gdown")
print("✓ gdown installed for downloading from Drive links")

# --- cell 3: download RAG snapshot + manifest from Drive link ----------------
# Uses gdown to download from your shareable Drive folder link.
# This avoids mounting your other Drive account — just need the shareable link!
import os
import subprocess
from pathlib import Path

print("Downloading RAG artifacts from Drive (using gdown)...")
print("(First time may take a few minutes depending on snapshot size.)\n")

# Extract folder ID from Drive link
# Link format: https://drive.google.com/drive/folders/FOLDER_ID
def extract_drive_folder_id(drive_link):
    """Extract folder ID from a Google Drive shareable link."""
    if "/folders/" in drive_link:
        return drive_link.split("/folders/")[1].split("?")[0]
    else:
        raise ValueError(f"Invalid Drive link format: {drive_link}")

try:
    rag_folder_id = extract_drive_folder_id(DRIVE_RAG_FOLDER_LINK)
    print(f"Downloading RAG folder (ID: {rag_folder_id})...")
    
    # gdown can download entire folders
    download_cmd = f'gdown --folder -O /content/rag_download "{DRIVE_RAG_FOLDER_LINK}" -q'
    result = os.system(download_cmd)
    
    if result == 0:
        rag_dir = Path("/content/rag_download")
        files = list(rag_dir.rglob("*"))
        if not (rag_dir / "manifest.json").is_file():
            raise FileNotFoundError("RAG folder must contain manifest.json at its top level")
        print(f"✓ Downloaded {len(files)} files from RAG folder")
        print(f"  Contents:")
        for f in sorted(list(rag_dir.glob("*")))[:10]:  # show first 10
            print(f"    - {f.name}")
        
        # Upload to bucket
        print(f"\nUploading to {BUCKET}/qdrant/...")
        get_ipython().system(f'gcloud storage cp --recursive "/content/rag_download"/* {BUCKET}/qdrant/')
        print(f"✓ Uploaded RAG artifacts to {BUCKET}/qdrant/")
    else:
        print("❌ gdown download failed — check your Drive link is shareable and public.")
        
except Exception as e:
    print(f"❌ ERROR downloading RAG: {e}")
    print("   Check your DRIVE_RAG_FOLDER_LINK in cell 2 is correct and shareable.")

# --- cell 4: download adapter from Drive link + MERGE into base Gemma --------
# ⚠️  REQUIRES GPU RUNTIME (Runtime -> Change runtime type -> T4 GPU)
#
# 1. Downloads your LoRA adapter from your shareable Drive link (using gdown).
# 2. Merges the adapter directly into base Gemma-3-4B weights.
# 3. Uploads the merged model to the bucket for fast inference.
#
# The adapter should be in your other Drive account at the link you provided in cell 2.
import os
import torch
from pathlib import Path
from transformers import (AutoConfig, AutoTokenizer,
                          AutoModelForCausalLM, AutoModelForImageTextToText)
from peft import PeftModel

BASE    = "google/gemma-3-4b-it"
ADAPTER_DOWNLOAD = "/content/adapter_download"
MERGED  = "/content/merged_gemma"

# Ensure dependencies are compatible
print("Installing compatible dependencies for adapter merging...")
get_ipython().system("pip install -q --upgrade peft torchao")
print("✓ Dependencies updated")

# Download adapter from Drive link
print("Downloading LoRA adapter from Drive link...")
try:
    def extract_drive_folder_id(drive_link):
        if "/folders/" in drive_link:
            return drive_link.split("/folders/")[1].split("?")[0]
        else:
            raise ValueError(f"Invalid Drive link format: {drive_link}")
    
    adapter_folder_id = extract_drive_folder_id(DRIVE_ADAPTER_FOLDER_LINK)
    print(f"Downloading adapter folder (ID: {adapter_folder_id})...")
    
    download_cmd = f'gdown --folder -O "{ADAPTER_DOWNLOAD}" "{DRIVE_ADAPTER_FOLDER_LINK}" -q'
    result = os.system(download_cmd)
    
    if result != 0:
        print("❌ gdown download failed — check your DRIVE_ADAPTER_FOLDER_LINK is correct and shareable.")
        raise RuntimeError("Adapter download failed")
    
    adapter_path = Path(ADAPTER_DOWNLOAD)
    adapter_files = list(adapter_path.rglob("*"))
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError("Adapter folder must contain adapter_config.json at its top level")
    print(f"✓ Downloaded adapter ({len(adapter_files)} files)")
    print(f"  Contents: {', '.join([f.name for f in list(adapter_path.glob('*'))[:5]])}")
    
except Exception as e:
    print(f"❌ ERROR downloading adapter: {e}")
    raise

print(f"\n✓ Adapter ready at {ADAPTER_DOWNLOAD}")
print(f"Loading base model {BASE}...")

# Load config to check if multimodal
cfg = AutoConfig.from_pretrained(BASE, token=HF_TOKEN)
multimodal = hasattr(cfg, "vision_config") or "text_config" in (getattr(cfg, "sub_configs", None) or {})
Cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM

print(f"Model type: {'multimodal' if multimodal else 'text-only'}")
print("Loading base model (this may take a minute)...")
base   = Cls.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)

print("Merging adapter into base weights (this will take a few minutes)...")
try:
    merged = PeftModel.from_pretrained(base, ADAPTER_DOWNLOAD).merge_and_unload()  # <-- fold adapter in
except ImportError as e:
    if "torchao" in str(e) or "PEFT" in str(e):
        print(f"❌ ERROR: {e}")
        print("   Installing updated dependencies...")
        get_ipython().system("pip install -q --upgrade peft torchao transformers")
        print("   Please re-run this cell (it may need a kernel restart).")
        raise
    else:
        raise

print(f"Saving merged model to {MERGED}...")
merged.save_pretrained(MERGED, safe_serialization=True)

# Copy tokenizer
tok_src = ADAPTER_DOWNLOAD if os.path.exists(os.path.join(ADAPTER_DOWNLOAD, "tokenizer_config.json")) else BASE
AutoTokenizer.from_pretrained(tok_src, token=HF_TOKEN).save_pretrained(MERGED)

print(f"✓ Merged model saved to {MERGED}")
print(f"Uploading to {BUCKET}/generator/merged/ (this may take a minute)...")
get_ipython().system(f'gcloud storage cp -r "{MERGED}"/* {BUCKET}/generator/merged/')
print(f"✓ Merged model uploaded to {BUCKET}/generator/merged/")

# --- cell 5: IEG + Vision — download from Kaggle (your Kaggle account) --------
# Downloads pre-trained checkpoints from public Kaggle datasets using your Kaggle
# credentials (authenticated in cell 1b), then uploads to GCS.
#
# No multi-account switching needed here — uses the Kaggle login from cell 1b.
#
# Datasets:
#   - IEG: "aneeqasiddiqui377/v4-output" (intent/entity/guardrail model)
#   - Vision: "iitm21f1003346/vits16-crop-disease" (ViT leaf disease classifier)
from pathlib import Path

print("Downloading IEG + Vision models from Kaggle...")
print("(First time may take a few minutes; Kaggle caches downloads.)\n")

# IEG: dataset "aneeqasiddiqui377/v4-output"; prefer the ieg_adamw checkpoint.
print("Downloading IEG dataset from Kaggle...")
ieg_dir = Path(kagglehub.dataset_download("aneeqasiddiqui377/v4-output"))
_pts    = list(ieg_dir.rglob("*.pt"))
ieg_ckpt   = next((p for p in _pts if "ieg_adamw" in p.name), _pts[0] if _pts else None)
ieg_labels = next(iter(ieg_dir.rglob("label_maps.json")), None)

if ieg_ckpt:
    print(f"✓ IEG checkpoint: {ieg_ckpt.name}")
    get_ipython().system(f'gcloud storage cp "{ieg_ckpt}" {BUCKET}/ieg/intent_entity_guardrail_model.pt')
    print(f"  Uploaded to {BUCKET}/ieg/intent_entity_guardrail_model.pt")
else:
    print("❌ IEG checkpoint not found in the dataset!")

if ieg_labels:
    print(f"✓ IEG labels: {ieg_labels.name}")
    get_ipython().system(f'gcloud storage cp "{ieg_labels}" {BUCKET}/ieg/label_maps.json')
    print(f"  Uploaded to {BUCKET}/ieg/label_maps.json")
else:
    print("⚠️  IEG labels not found (optional)")

# Vision: dataset "iitm21f1003346/vits16-crop-disease".
print("\nDownloading Vision dataset from Kaggle...")
vit_dir  = Path(kagglehub.dataset_download("iitm21f1003346/vits16-crop-disease"))
_vpts    = list(vit_dir.rglob("*.pt")) + list(vit_dir.rglob("*.pth"))
vit_ckpt = next((p for p in _vpts if "p3_full_best" in p.name), _vpts[0] if _vpts else None)
vit_lbls = next(iter(vit_dir.rglob("label_to_idx.json")), None)

if vit_ckpt:
    print(f"✓ ViT checkpoint: {vit_ckpt.name}")
    get_ipython().system(f'gcloud storage cp "{vit_ckpt}" {BUCKET}/vision/p3_full_best.pt')
    print(f"  Uploaded to {BUCKET}/vision/p3_full_best.pt")
else:
    print("❌ ViT checkpoint not found in the dataset!")

if vit_lbls:
    print(f"✓ ViT labels: {vit_lbls.name}")
    get_ipython().system(f'gcloud storage cp "{vit_lbls}" {BUCKET}/vision/label_to_idx.json')
    print(f"  Uploaded to {BUCKET}/vision/label_to_idx.json")
else:
    print("⚠️  label_to_idx.json not found — the gateway's vision.py needs it.")
    print("   Options:")
    print("   1. Check if the Kaggle dataset has a label file under a different name.")
    print("   2. Manually add your label map to the bucket: gcloud storage cp your_labels.json {BUCKET}/vision/label_to_idx.json")

# --- cell 6: verify the final structure in the bucket ----------------------
# List everything in the bucket to confirm all models are in the right place.
print("\n=== VERIFYING BUCKET CONTENTS ===\n")
get_ipython().system(f"gcloud storage ls -r {BUCKET}")

print("\n=== EXPECTED STRUCTURE ===")
print(f"""
{BUCKET}/
  qdrant/
    manifest.json              <- RAG config (from your Drive account via shareable link)
    snapshot_*.json            <- Vector DB snapshot (from your Drive account via shareable link)
  
  generator/
    merged/
      config.json              <- Merged model config
      model.safetensors        <- Merged weights (adapter baked in)
      tokenizer.json
      tokenizer_config.json
  
  ieg/
    intent_entity_guardrail_model.pt  <- From Kaggle (downloaded in cell 5)
    label_maps.json                   <- From Kaggle (downloaded in cell 5)
  
  vision/
    p3_full_best.pt            <- From Kaggle (downloaded in cell 5)
    label_to_idx.json          <- From Kaggle (downloaded in cell 5)
""")

print("\n=== HOW TO GET SHAREABLE DRIVE LINKS ===")
print("""
If you need to re-run cells 3 or 4, make sure you have shareable Drive links:

1. In your OTHER Google account, go to Google Drive.
2. Find the folder with your RAG snapshot (from notebook 08b).
   Right-click → Share → Set to "Anyone with the link can view"
   Copy the link: https://drive.google.com/drive/folders/FOLDER_ID
   Paste into cell 2: DRIVE_RAG_FOLDER_LINK

3. Find the folder with your LoRA adapter (from distillation training).
   Right-click → Share → Set to "Anyone with the link can view"
   Copy the link: https://drive.google.com/drive/folders/FOLDER_ID
   Paste into cell 2: DRIVE_ADAPTER_FOLDER_LINK

4. Re-run cells 2, 3, 4 in order if needed.
""")

print("\n=== NEXT STEPS ===")
print(f"✓ All artifacts uploaded to: {BUCKET}")
print(f"1. Go back to Cloud Shell and run:  bash 03_create_vm.sh")
print(f"2. The VM will pull models from the bucket")
print(f"3. If any folder is missing, check your Drive links and re-run the cell")