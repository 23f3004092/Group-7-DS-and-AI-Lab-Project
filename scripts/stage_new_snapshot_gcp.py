#!/usr/bin/env python3
"""stage_new_snapshot_gcp.py
===========================
Stage the NEW routed-schema Qdrant snapshot (payload key "source" = kcc_qa /
ppqs_advisories / up_acp / schemes, plus per-chunk query_type) into an
artifacts folder that docs/internal/do_not_open/requiredforgcp/restore_qdrant.py
and 02_upload_artifacts.sh expect:

    <out>/manifest.json
    <out>/agri_knowledge.snapshot

Source priority (mirrors scripts/setup_vectordb.py):
  1. local file already on disk (--from)
  2. Kaggle Hub dataset lokeshvns/up-agri-kcc-rag-artifacts
     (rag_out/agri_knowledge.snapshot + rag_out/manifest.json)
  3. Google Drive fallback IDs from setup_vectordb.py

Usage:
    python3 scripts/stage_new_snapshot_gcp.py --out ./staging/qdrant
    python3 scripts/stage_new_snapshot_gcp.py --out ./staging/qdrant \
        --from ~/qdrant_rag_setup/rag_production_bge_m3

Then upload + restore:
    gsutil cp ./staging/qdrant/* gs://$BUCKET/artifacts/qdrant/
    # ... on the VM:
    python3 restore_qdrant.py --artifacts /opt/farmervision/artifacts/qdrant
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

KAGGLE_DATASET = "lokeshvns/up-agri-kcc-rag-artifacts"
SNAPSHOT_REL   = os.path.join("rag_out", "agri_knowledge.snapshot")
MANIFEST_REL   = os.path.join("rag_out", "manifest.json")

# Google Drive fallbacks (same IDs as scripts/setup_vectordb.py)
GDRIVE_SNAPSHOT_ID = "1FhTHMfyOzLGfOq6VLh_V6tnTNGe-ro1N"
GDRIVE_MANIFEST_ID = "1JnbcSbVzqcOEeZLU_-kuNb5uB6ZzTePL"

MIN_SNAPSHOT_BYTES = 1_000_000_000


def stage_from_local(src_dir: Path, out: Path):
    snap = src_dir / "agri_knowledge.snapshot"
    mani = src_dir / "manifest.json"
    if not snap.exists():
        return False
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snap, out / snap.name)
    if mani.exists():
        shutil.copy2(mani, out / mani.name)
    print(f"[stage] local OK — snapshot {snap.stat().st_size/1e9:.2f} GB")
    return True


def stage_from_kaggle(out: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[stage] kagglehub not installed — pip install kagglehub")
        return False
    try:
        root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as e:
        print(f"[stage] kagglehub download failed: {e}")
        return False
    snap, mani = root / SNAPSHOT_REL, root / MANIFEST_REL
    if not snap.exists():
        print(f"[stage] snapshot missing in dataset: {snap}")
        return False
    out.mkdir(parents=True, exist_ok=True)
    print(f"[stage] copying snapshot ({snap.stat().st_size/1e9:.2f} GB) …")
    shutil.copy2(snap, out / "agri_knowledge.snapshot")
    if mani.exists():
        shutil.copy2(mani, out / "manifest.json")
    return True


def stage_from_gdrive(out: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("[stage] gdown not installed — pip install gdown")
        return False
    out.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(id=GDRIVE_MANIFEST_ID,
                       output=str(out / "manifest.json"), quiet=False)
        gdown.download(id=GDRIVE_SNAPSHOT_ID,
                       output=str(out / "agri_knowledge.snapshot"), quiet=False)
    except Exception as e:
        print(f"[stage] gdown failed: {e}")
        return False
    return (out / "agri_knowledge.snapshot").exists()


def verify_routed_schema(artifacts: Path):
    """Best-effort sanity check: manifest should describe the new index."""
    mp = artifacts / "manifest.json"
    if not mp.exists():
        print("[verify] WARNING: no manifest.json staged")
        return
    import json
    m = json.load(open(mp, encoding="utf-8"))
    print("[verify] manifest keys:", sorted(m.keys()))
    for k in ("collection", "embed_model", "embed_dim", "tiers"):
        if k not in m:
            print(f"[verify] WARNING: manifest missing '{k}' — gateway config.py needs it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging output dir for artifacts")
    ap.add_argument("--from", dest="src", default=None,
                    help="use a snapshot already on disk (dir containing "
                         "agri_knowledge.snapshot + manifest.json)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    ok = False
    if args.src:
        ok = stage_from_local(Path(args.src).expanduser(), out)
    if not ok:
        ok = stage_from_kaggle(out)
    if not ok:
        ok = stage_from_gdrive(out)
    if not ok:
        print("[stage] FAILED: all sources exhausted")
        return 1

    snap = out / "agri_knowledge.snapshot"
    size = snap.stat().st_size
    if size < MIN_SNAPSHOT_BYTES:
        print(f"[verify] ERROR: snapshot suspiciously small ({size:,} bytes)")
        return 1
    print(f"[verify] snapshot OK ({size/1e9:.2f} GB)")
    verify_routed_schema(out)

    print(f"""
NEXT STEPS
----------
1. Also stage the NEW IEG checkpoint (from Kaggle dataset
   lokeshtiwariiitm/ieg-model) and the yield LightGBM model:
     python3 scripts/stage_models_gcp.py
   It validates label_maps.json contains "model_name" (must be
   l3cube-pune/hing-mbert-mixed — gateway ieg.py reads it).

2. Upload to GCS and restore on the VM:
     gsutil cp {out}/* gs://$BUCKET/artifacts/qdrant/
     ssh <vm> 'python3 /opt/farmervision/restore_qdrant.py \\
         --artifacts /opt/farmervision/artifacts/qdrant'

3. Restart the gateway; retrieval.load() auto-detects the ROUTED schema
   and enables intent routing + confidence-gated qtype filters.

4. A/B-tune the gate without code changes:
     echo "CONF_GATE_LOOSEN=0.60" >> runtime.env && docker compose up -d
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
