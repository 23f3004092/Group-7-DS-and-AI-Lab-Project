"""
Complete Qdrant Setup and Verification
Merged script - Downloads, Sets up, and Tests Qdrant
"""

import os
import sys
import json
import time
import shutil
import subprocess
import zipfile
import requests
import gdown
from pathlib import Path

# Force UTF-8 output so ✓ / ❌ / ⚠️ characters don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Auto-detect Kaggle kernel environment
# On Kaggle: KAGGLE_KERNEL_RUN_TYPE is set, and /kaggle/working exists.
ON_KAGGLE = os.path.isdir("/kaggle/working") or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

if ON_KAGGLE:
    WORK_ROOT      = "/kaggle/working/qdrant_setup"
    QDRANT_BIN     = os.path.join(WORK_ROOT, "qdrant")          # Linux binary (no .exe)
else:
    WORK_ROOT      = os.path.join(os.path.expanduser("~"), "qdrant_rag_setup")
    QDRANT_BIN     = os.path.join(WORK_ROOT, "qdrant.exe")      # Windows binary

ART_DIR        = os.path.join(WORK_ROOT, "rag_production_bge_m3")
QDRANT_STORAGE = os.path.join(WORK_ROOT, "qdrant_storage")
QDRANT_URL     = "http://localhost:6333"
COLLECTION_NAME = "agri_knowledge"

# Kaggle dataset slug that contains the new RAG artifacts
KAGGLE_DATASET       = "lokeshvns/up-agri-kcc-rag-artifacts"
# Dataset slug basename used by Kaggle when it mounts datasets as /kaggle/input/<slug-name>/
KAGGLE_INPUT_NAME    = KAGGLE_DATASET.split("/")[-1]           # "up-agri-kcc-rag-artifacts"
KAGGLE_INPUT_ROOT    = os.path.join("/kaggle/input", KAGGLE_INPUT_NAME)

# Relative paths inside the Kaggle dataset
KAGGLE_SNAPSHOT_REL  = os.path.join("rag_out", "agri_knowledge.snapshot")
KAGGLE_MANIFEST_REL  = os.path.join("rag_out", "manifest.json")
KAGGLE_QDRANT_REL    = "qdrant"   # Linux binary shipped inside the dataset

# Google Drive IDs (fallback)
SNAPSHOT_ID = "1FhTHMfyOzLGfOq6VLh_V6tnTNGe-ro1N"
MANIFEST_ID = "1JnbcSbVzqcOEeZLU_-kuNb5uB6ZzTePL"

# ============================================================================
# STEP 1: DOWNLOAD FILES
# ============================================================================

def _try_kaggle_input(dest_dir):
    """Priority-0 source: read snapshot/manifest directly from the Kaggle
    input dataset mount (/kaggle/input/<dataset>/...).

    This is the fastest path when the dataset is attached as a Kaggle input
    because the files are already on-disk — zero network traffic required.
    Returns (snapshot_path, manifest_path) or (None, None).
    """
    if not os.path.isdir("/kaggle/input"):
        return None, None

    import glob
    snapshot_candidates = glob.glob("/kaggle/input/**/agri_knowledge.snapshot", recursive=True)
    
    if not snapshot_candidates:
        print("  Kaggle input mounted at /kaggle/input but snapshot not found.")
        return None, None

    src_snap = snapshot_candidates[0]
    snap_dir = os.path.dirname(src_snap)
    src_mani = os.path.join(snap_dir, "manifest.json")

    os.makedirs(dest_dir, exist_ok=True)
    dst_snap = os.path.join(dest_dir, "agri_knowledge.snapshot")
    dst_mani = os.path.join(dest_dir, "manifest.json")

    # Snapshot: copy only if not already present (idempotent)
    if not (os.path.exists(dst_snap) and os.path.getsize(dst_snap) > 1_000_000_000):
        print(f"  Copying snapshot from Kaggle input ({os.path.getsize(src_snap)/1e9:.2f} GB) ...")
        shutil.copy2(src_snap, dst_snap)
        print(f"  ✓ Snapshot ready at {dst_snap}")
    else:
        print(f"  Snapshot already present: {os.path.getsize(dst_snap)/1e9:.2f} GB")

    if os.path.exists(src_mani) and not os.path.exists(dst_mani):
        shutil.copy2(src_mani, dst_mani)
        print(f"  ✓ Manifest copied")

    return dst_snap, dst_mani


def _try_kagglehub(dest_dir):
    """Try to download snapshot + manifest from Kaggle Hub.

    Returns (snapshot_path, manifest_path) on success, or (None, None) on failure.
    """
    try:
        import kagglehub  # optional dependency
    except ImportError:
        print("  kagglehub not installed — skipping Kaggle source.")
        print("  Install with: pip install kagglehub")
        return None, None

    try:
        print(f"  Downloading from Kaggle: {KAGGLE_DATASET} ...")
        dataset_root = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"  Kaggle dataset root: {dataset_root}")

        src_snap = os.path.join(dataset_root, KAGGLE_SNAPSHOT_REL)
        src_mani = os.path.join(dataset_root, KAGGLE_MANIFEST_REL)

        if not os.path.exists(src_snap):
            print(f"  WARNING: snapshot not found at expected path: {src_snap}")
            return None, None

        # Copy into our ART_DIR so the rest of the script stays path-agnostic
        os.makedirs(dest_dir, exist_ok=True)
        dst_snap = os.path.join(dest_dir, "agri_knowledge.snapshot")
        dst_mani = os.path.join(dest_dir, "manifest.json")

        if not (os.path.exists(dst_snap) and os.path.getsize(dst_snap) > 1_000_000_000):
            print(f"  Copying snapshot ({os.path.getsize(src_snap)/1e9:.2f} GB) ...")
            shutil.copy2(src_snap, dst_snap)
            print(f"  ✓ Snapshot copied to {dst_snap}")
        else:
            print(f"  Snapshot already present: {os.path.getsize(dst_snap)/1e9:.2f} GB")

        if os.path.exists(src_mani) and not os.path.exists(dst_mani):
            shutil.copy2(src_mani, dst_mani)
            print(f"  ✓ Manifest copied to {dst_mani}")

        return dst_snap, dst_mani

    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        return None, None


def download_with_retry(file_id, output, max_retries=5, delay=60):
    """Download with retry on quota errors"""
    
    for attempt in range(1, max_retries + 1):
        print(f"\nAttempt {attempt}/{max_retries}...")
        
        try:
            gdown.download(id=file_id, output=output, quiet=False)
            
            # Verify download
            if os.path.exists(output) and os.path.getsize(output) > 100_000_000:
                print(f"✓ Download successful! Size: {os.path.getsize(output)/1e9:.2f} GB")
                return True
            else:
                print("Download incomplete, retrying...")
                
        except Exception as e:
            print(f"Error: {e}")
            
            if "quota" in str(e).lower() or "too many" in str(e).lower():
                print(f"Quota exceeded. Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"Error: {e}")
                time.sleep(10)
    
    return False

def download_files():
    """Download manifest and snapshot.

    Source priority:
      1. Kaggle Hub  (kagglehub.dataset_download — fast, no quota limits)
      2. Google Drive (gdown fallback — requires SNAPSHOT_ID / MANIFEST_ID)
    """
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOADING FILES")
    print("=" * 70)
    
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"\nOutput directory: {ART_DIR}")

    snapshot_path = os.path.join(ART_DIR, "agri_knowledge.snapshot")
    manifest_path = os.path.join(ART_DIR, "manifest.json")

    # ---- Source 0: Kaggle input mount (already on-disk, zero download) ---
    print("\n[Source 0/3] Kaggle input dataset mount ...")
    snap, mani = _try_kaggle_input(ART_DIR)
    if snap and os.path.exists(snap) and os.path.getsize(snap) > 1_000_000_000:
        print(f"\n✓ Kaggle input OK — snapshot: {os.path.getsize(snap)/1e9:.2f} GB")
        return snap, mani or manifest_path

    # ---- Source 1: Kaggle Hub (download via API) -------------------------
    print("\n[Source 1/3] Kaggle Hub download ...")
    snap, mani = _try_kagglehub(ART_DIR)
    if snap and os.path.exists(snap) and os.path.getsize(snap) > 1_000_000_000:
        print(f"\n✓ Kaggle Hub OK — snapshot: {os.path.getsize(snap)/1e9:.2f} GB")
        return snap, mani or manifest_path

    # ---- Source 2: Google Drive fallback --------------------------------
    print("\n[Source 2/3] Google Drive fallback ...")
    
    # Download manifest
    print("\n1. Downloading manifest...")
    if not os.path.exists(manifest_path):
        try:
            gdown.download(id=MANIFEST_ID, output=manifest_path, quiet=False)
        except Exception as e:
            print(f"Warning: Could not download manifest: {e}")
    else:
        print(f"  Manifest already exists")
    
    # Download snapshot
    print("\n2. Downloading snapshot (3.8 GB)...")
    
    if os.path.exists(snapshot_path) and os.path.getsize(snapshot_path) > 1_000_000_000:
        print(f"  Snapshot already exists: {os.path.getsize(snapshot_path)/1e9:.2f} GB")
    else:
        success = download_with_retry(SNAPSHOT_ID, snapshot_path)
        if not success:
            print("\n❌ Failed to download snapshot.")
            print("\nPlease try one of these options:")
            print("1. Install kagglehub and configure Kaggle API credentials")
            print("2. Copy the file to your own Google Drive and update the file ID")
            print("3. Run the download again later when quota resets")
            print(f"4. Download manually from: https://drive.google.com/file/d/{SNAPSHOT_ID}/view")
            sys.exit(1)
    
    return snapshot_path, manifest_path


# ============================================================================
# STEP 2: EXTRACT AND VERIFY
# ============================================================================

def load_manifest(manifest_path):
    """Load and display manifest"""
    if not os.path.exists(manifest_path):
        print("  Manifest not found")
        return None
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    print("\n  Manifest loaded:")
    for key in ["collection", "embed_model", "embed_dim", "max_seq_length", "n_chunks", "built_utc"]:
        if key in manifest:
            print(f"    {key}: {manifest[key]}")
    
    return manifest

# ============================================================================
# STEP 3: START QDRANT
# ============================================================================

def qdrant_alive(timeout=1):
    try:
        return requests.get(f"{QDRANT_URL}/readyz", timeout=timeout).ok
    except:
        return False

def download_qdrant():
    """Download Qdrant binary for Windows if not present"""
    if os.path.exists(QDRANT_BIN):
        return True
    
    print("\n  Downloading Qdrant for Windows...")
    
    qdrant_url = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip"
    zip_path = os.path.join(WORK_ROOT, "qdrant.zip")
    
    try:
        response = requests.get(qdrant_url, stream=True, timeout=600)
        if response.status_code != 200:
            qdrant_url = "https://github.com/qdrant/qdrant/releases/download/v1.19.0/qdrant-x86_64-pc-windows-msvc.zip"
            response = requests.get(qdrant_url, stream=True, timeout=600)
        
        if response.status_code != 200:
            print(f"  Failed to download Qdrant (status: {response.status_code})")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r    [{'=' * progress}{' ' * (50 - progress)}] {downloaded/1024/1024:.1f}MB")
                        sys.stdout.flush()
        print()
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(WORK_ROOT)
        os.remove(zip_path)
        
        for root, dirs, files in os.walk(WORK_ROOT):
            for f in files:
                if f == "qdrant.exe":
                    src = os.path.join(root, f)
                    if src != QDRANT_BIN:
                        shutil.move(src, QDRANT_BIN)
                        print("  Qdrant downloaded and extracted")
                    return True
        
        return False
    except Exception as e:
        print(f"  Error downloading Qdrant: {e}")
        return False

def _ensure_qdrant_bin_linux():
    """On Kaggle/Linux: copy the qdrant binary from the input dataset
    (read-only) into WORK_ROOT (writable), then make it executable.
    Falls back to downloading from GitHub if the dataset binary is missing.
    """
    import platform, tarfile
    if os.path.exists(QDRANT_BIN):
        os.chmod(QDRANT_BIN, 0o755)   # ensure +x even across session restores
        return True

    # Try the binary shipped inside the Kaggle input dataset first
    import glob
    qdrant_candidates = [p for p in glob.glob("/kaggle/input/**/qdrant", recursive=True) if os.path.isfile(p)]
    if qdrant_candidates:
        src_bin = qdrant_candidates[0]
        os.makedirs(WORK_ROOT, exist_ok=True)
        shutil.copy2(src_bin, QDRANT_BIN)
        os.chmod(QDRANT_BIN, 0o755)
        try:
            result = subprocess.run([QDRANT_BIN, "--version"], capture_output=True, timeout=10)
            if result.returncode == 0:
                print(f"  ✓ Qdrant binary from dataset: {result.stdout.decode().strip()}")
                return True
            print(f"  Dataset binary failed ({result.stderr.decode()[:200]}), will download.")
        except Exception as e:
            print(f"  Dataset binary failed ({e}), will download.")
        if os.path.exists(QDRANT_BIN):
            os.remove(QDRANT_BIN)

    # Download the musl static binary from GitHub
    print("  Downloading Qdrant Linux (musl) binary from GitHub ...")
    rel = requests.get(
        "https://api.github.com/repos/qdrant/qdrant/releases/latest", timeout=30
    ).json()
    asset = None
    for suffix in ("x86_64-unknown-linux-musl.tar.gz", "x86_64-unknown-linux-gnu.tar.gz"):
        asset = next((a for a in rel["assets"] if a["name"].endswith(suffix)), None)
        if asset:
            break
    if asset is None:
        print("  Could not find Linux x86_64 Qdrant asset on GitHub.")
        return False
    tar_path = QDRANT_BIN + ".tar.gz"
    os.makedirs(WORK_ROOT, exist_ok=True)
    with open(tar_path, "wb") as f:
        f.write(requests.get(asset["browser_download_url"], timeout=600).content)
    import tarfile as _tarfile
    with _tarfile.open(tar_path) as t:
        try:
            t.extractall(WORK_ROOT, filter="data")
        except TypeError:
            t.extractall(WORK_ROOT)
    os.remove(tar_path)
    extracted = os.path.join(WORK_ROOT, "qdrant")
    if extracted != QDRANT_BIN:
        shutil.move(extracted, QDRANT_BIN)
    os.chmod(QDRANT_BIN, 0o755)
    result = subprocess.run([QDRANT_BIN, "--version"], capture_output=True, timeout=10)
    if result.returncode != 0:
        print(f"  Downloaded binary failed: {result.stderr.decode()[:300]}")
        return False
    print(f"  ✓ Qdrant from GitHub: {result.stdout.decode().strip()}")
    return True


def start_qdrant():
    """Start Qdrant server (Windows or Linux/Kaggle)."""
    import platform
    print("\n" + "=" * 70)
    print("STEP 2: STARTING QDRANT")
    print("=" * 70)
    print(f"  Environment: {'Kaggle / Linux' if ON_KAGGLE else 'Local / Windows'}")
    
    if qdrant_alive():
        print("✓ Qdrant already running on :6333")
        return True

    os.makedirs(WORK_ROOT, exist_ok=True)

    if platform.system() == "Linux":
        if not _ensure_qdrant_bin_linux():
            return False
    else:
        if not os.path.exists(QDRANT_BIN):
            if not download_qdrant():
                return False

    os.makedirs(QDRANT_STORAGE, exist_ok=True)
    
    # Check if port is in use
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 6333))
    sock.close()
    
    if result == 0:
        print("  Port 6333 is in use. Trying to connect...")
        if qdrant_alive():
            print("✓ Qdrant is running")
            return True
    
    print("  Starting Qdrant...")
    log_path = os.path.join(WORK_ROOT, "qdrant.log")
    
    env = dict(os.environ, 
               QDRANT__STORAGE__STORAGE_PATH=QDRANT_STORAGE,
               QDRANT__TELEMETRY_DISABLED="true")
    
    try:
        kwargs = {
            "env": env,
            "cwd": WORK_ROOT,
            "stdout": open(log_path, "w"),
            "stderr": subprocess.STDOUT,
        }
        if platform.system() == "Linux":
            kwargs["preexec_fn"] = os.setpgrp
        else:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            
        subprocess.Popen([QDRANT_BIN], **kwargs)
        
        print("  Waiting for Qdrant to be ready...")
        for i in range(120):
            if qdrant_alive():
                print("✓ Qdrant is ready!")
                return True
            time.sleep(1)
            if i % 10 == 0:
                print(f"    Waiting... ({i}s)")
        
        print("  Timeout waiting for Qdrant")
        return False
    except Exception as e:
        print(f"  Error starting Qdrant: {e}")
        return False

# ============================================================================
# STEP 4: RESTORE SNAPSHOT
# ============================================================================

def restore_snapshot(snapshot_path):
    """Restore snapshot to Qdrant"""
    print("\n" + "=" * 70)
    print("STEP 3: RESTORING SNAPSHOT")
    print("=" * 70)
    
    if not os.path.exists(snapshot_path):
        print(f"❌ Snapshot not found: {snapshot_path}")
        return False
    
    file_size = os.path.getsize(snapshot_path) / 1e9
    print(f"Snapshot: {os.path.basename(snapshot_path)} ({file_size:.2f} GB)")
    
    if not qdrant_alive():
        print("❌ Qdrant is not running")
        return False
    
    # Check if collection already exists
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=10)
        if response.status_code == 200:
            collections = response.json()
            if collections.get('result', {}).get('collections'):
                for c in collections['result']['collections']:
                    if c.get('name') == COLLECTION_NAME:
                        print(f"✓ Collection '{COLLECTION_NAME}' already exists")
                        return True
    except:
        pass
    
    # Upload snapshot
    print(f"  Uploading {file_size:.2f} GB snapshot...")
    print("  This will take several minutes...")
    
    start_time = time.time()
    
    try:
        with open(snapshot_path, 'rb') as f:
            response = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/upload?priority=snapshot",
                files={"snapshot": (os.path.basename(snapshot_path), f)},
                timeout=7200
            )
        
        elapsed = (time.time() - start_time) / 60
        
        if response.status_code == 200:
            print(f"✓ Snapshot restored in {elapsed:.1f} minutes")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Upload timed out (may still be processing)")
        return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    return True

# ============================================================================
# STEP 5: VERIFY
# ============================================================================

def verify_qdrant():
    """Verify the Qdrant setup"""
    print("\n" + "=" * 70)
    print("STEP 4: VERIFYING QDRANT")
    print("=" * 70)
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        
        # Check collection info
        info = client.get_collection(COLLECTION_NAME)
        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"Points: {info.points_count:,}")
        print(f"Status: {info.status}")
        print(f"Indexed Vectors: {info.indexed_vectors_count:,}")
        
        # Sample a point to verify data
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if points:
            print("\n✅ Sample point:")
            print(f"  ID: {points[0].id}")
            payload = points[0].payload
            if payload:
                text = payload.get('text', 'N/A')
                print(f"  Text: {text[:200]}..." if len(text) > 200 else f"  Text: {text}")
                print(f"  Source: {payload.get('source_type', 'N/A')}")
                print(f"  Crop: {payload.get('crop', 'N/A')}")
        else:
            print("\n⚠️ No points found in collection")
        
        print("\n✅ Qdrant is ready for use!")
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("QDRANT COMPLETE SETUP")
    print("=" * 70)
    
    # STEP 1: Download files (if needed)
    snapshot_path, manifest_path = download_files()
    
    # Load manifest
    manifest = load_manifest(manifest_path)
    
    # STEP 2: Start Qdrant
    if not start_qdrant():
        print("❌ Failed to start Qdrant")
        sys.exit(1)
    
    # STEP 3: Restore snapshot
    if not restore_snapshot(snapshot_path):
        print("⚠️ Snapshot restoration may have issues")
    
    # STEP 4: Verify
    success = verify_qdrant()
    
    # Final summary
    print("\n" + "=" * 70)
    if success:
        print("✅ SETUP COMPLETE! QDRANT IS READY")
    else:
        print("⚠️ SETUP COMPLETE BUT VERIFICATION FAILED")
    print("=" * 70)
    
    print(f"\n📍 Qdrant URL: {QDRANT_URL}")
    print(f"📍 Dashboard: {QDRANT_URL}/dashboard")
    print(f"📍 Collection: {COLLECTION_NAME}")
    print(f"📍 Storage: {QDRANT_STORAGE}")
    
    if ON_KAGGLE:
        print("\nTo stop Qdrant: pkill qdrant")
        print(f"To restart: nohup {QDRANT_BIN} > {WORK_ROOT}/qdrant.log 2>&1 &")
    else:
        print("\nTo stop Qdrant: taskkill /F /IM qdrant.exe")
        print("To restart: cd ~/qdrant_rag_setup && .\\qdrant.exe")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)