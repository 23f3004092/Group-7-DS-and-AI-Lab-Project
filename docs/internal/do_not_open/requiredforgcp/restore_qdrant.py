#!/usr/bin/env python3
"""Restore the FarmerVision vector DB into a running Qdrant server.

Reads manifest.json (which names the snapshot file + collection), uploads the
snapshot through Qdrant's HTTP recovery endpoint -- the SAME method the build
notebook used -- then verifies the point count so you know it actually worked.

Idempotent: if the collection already exists with points, it does nothing.

Usage (run on the VM after Qdrant is up):
    python3 restore_qdrant.py --artifacts /opt/farmervision/artifacts/qdrant
"""
import argparse
import json
import os
import sys

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True,
                    help="folder containing manifest.json + the .snapshot file")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    manifest_path = os.path.join(args.artifacts, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: no manifest.json in {args.artifacts}", file=sys.stderr)
        return 1

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    collection = manifest["collection"]
    snap_name = manifest["snapshot"]
    snap_path = os.path.join(args.artifacts, snap_name)

    if not os.path.isfile(snap_path):
        print(f"ERROR: snapshot '{snap_name}' not found in {args.artifacts}", file=sys.stderr)
        return 1

    # Already restored? Then leave it alone (lets you re-run 04_vm_setup.sh safely).
    info = requests.get(f"{args.qdrant_url}/collections/{collection}", timeout=30)
    if info.status_code == 200:
        pts = info.json()["result"].get("points_count") or 0
        if pts > 0:
            print(f"OK: collection '{collection}' already has {pts:,} points — skipping restore.")
            return 0

    size_gb = os.path.getsize(snap_path) / 1e9
    print(f">> Uploading '{snap_name}' ({size_gb:.2f} GB) into '{collection}' — this is slow ...")
    with open(snap_path, "rb") as fh:
        r = requests.post(
            f"{args.qdrant_url}/collections/{collection}/snapshots/upload?priority=snapshot",
            files={"snapshot": (snap_name, fh)},
            timeout=7200,
        )
    r.raise_for_status()

    # Verify
    info = requests.get(f"{args.qdrant_url}/collections/{collection}", timeout=60).json()["result"]
    pts = info.get("points_count") or 0
    print(f">> Restored '{collection}': {pts:,} points, status={info.get('status')}")
    if pts == 0:
        print("ERROR: 0 points after restore — do NOT trust this. Check the snapshot file.",
              file=sys.stderr)
        return 1
    print("OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
