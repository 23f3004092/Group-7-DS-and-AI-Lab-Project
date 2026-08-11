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

# ============================================================================
# CONFIGURATION
# ============================================================================

WORK_ROOT = os.path.join(os.path.expanduser("~"), "qdrant_rag_setup")
ART_DIR = os.path.join(WORK_ROOT, "rag_production_bge_m3")
QDRANT_STORAGE = os.path.join(WORK_ROOT, "qdrant_storage")
QDRANT_BIN = os.path.join(WORK_ROOT, "qdrant.exe")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "agri_knowledge"

# Google Drive IDs
SNAPSHOT_ID = "1FhTHMfyOzLGfOq6VLh_V6tnTNGe-ro1N"
MANIFEST_ID = "1JnbcSbVzqcOEeZLU_-kuNb5uB6ZzTePL"

# ============================================================================
# STEP 1: DOWNLOAD FILES
# ============================================================================

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
    """Download manifest and snapshot from Google Drive"""
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOADING FILES")
    print("=" * 70)
    
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"\nOutput directory: {ART_DIR}")
    
    # Download manifest
    print("\n1. Downloading manifest...")
    manifest_path = os.path.join(ART_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        try:
            gdown.download(id=MANIFEST_ID, output=manifest_path, quiet=False)
        except Exception as e:
            print(f"Warning: Could not download manifest: {e}")
    else:
        print(f"  Manifest already exists")
    
    # Download snapshot
    print("\n2. Downloading snapshot (3.8 GB)...")
    snapshot_path = os.path.join(ART_DIR, "agri_knowledge.snapshot")
    
    if os.path.exists(snapshot_path) and os.path.getsize(snapshot_path) > 1_000_000_000:
        print(f"  Snapshot already exists: {os.path.getsize(snapshot_path)/1e9:.2f} GB")
    else:
        success = download_with_retry(SNAPSHOT_ID, snapshot_path)
        if not success:
            print("\n❌ Failed to download snapshot.")
            print("\nPlease try one of these options:")
            print("1. Copy the file to your own Google Drive and update the file ID")
            print("2. Run the download again later when quota resets")
            print("3. Download manually from: https://drive.google.com/file/d/1FhTHMfyOzLGfOq6VLh_V6tnTNGe-ro1N/view")
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

def start_qdrant():
    """Start Qdrant server"""
    print("\n" + "=" * 70)
    print("STEP 2: STARTING QDRANT")
    print("=" * 70)
    
    if qdrant_alive():
        print("✓ Qdrant already running on :6333")
        return True
    
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
        subprocess.Popen([QDRANT_BIN], env=env, cwd=WORK_ROOT,
                        stdout=open(log_path, "w"), 
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
        
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