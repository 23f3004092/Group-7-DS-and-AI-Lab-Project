#!/usr/bin/env python3
"""
download_data.py — Automated dataset downloader for AgriAssist (Milestone 2)

Downloads and organises all raw datasets into data/raw/:
  - PlantVillage (Kaggle)   → data/raw/plantvillage/
  - PlantDoc (GitHub)        → data/raw/plantdoc/
  - KCC Q&A Logs (Kaggle)   → data/raw/kcc/
  - Yield Data (manual)     → data/raw/yield/  (prints instructions)

Requirements:
  pip install kaggle requests tqdm

Kaggle datasets require a valid ~/.kaggle/kaggle.json API token.
See: https://www.kaggle.com/docs/api

Usage:
  python scripts/download_data.py --all
  python scripts/download_data.py --plantvillage --plantdoc
  python scripts/download_data.py --kcc
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Kaggle dataset slugs
PLANTVILLAGE_SLUG = "abdallahalidev/plantvillage-dataset"
PLANTVILLAGE_ALT_SLUG = "vipoooool/new-plant-diseases-dataset"
KCC_SLUG = "rajanand/kisan-call-center-data"  # Common Kaggle mirror

# PlantDoc GitHub release
PLANTDOC_GITHUB_URL = (
    "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_kaggle_cli() -> bool:
    """Check if the Kaggle CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        # Check for credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print(
                "⚠️  Kaggle CLI found but ~/.kaggle/kaggle.json is missing.\n"
                "   → Create an API token at https://www.kaggle.com/settings\n"
                "   → Save it to ~/.kaggle/kaggle.json"
            )
            return False
        return True
    except FileNotFoundError:
        return False


def download_kaggle_dataset(slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset."""
    ensure_dir(dest)
    print(f"📦 Downloading {slug} → {dest}")
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                slug,
                "-p",
                str(dest),
                "--unzip",
            ],
            check=True,
        )
        print(f"✅ {slug} downloaded successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {slug}: {e}")
        print(f"   Try manually: kaggle datasets download -d {slug}")
        return False


def download_url(url: str, dest_path: Path) -> bool:
    """Download a file from a URL using requests with a progress bar."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("⚠️  Install requests and tqdm: pip install requests tqdm")
        return False

    ensure_dir(dest_path.parent)
    print(f"📦 Downloading {url}")
    print(f"   → {dest_path}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✅ Downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------


def download_plantvillage():
    """Download PlantVillage dataset from Kaggle."""
    dest = RAW_DIR / "plantvillage"

    # Check if already downloaded
    if any(dest.iterdir()) and not all(
        f.name == ".gitkeep" for f in dest.iterdir()
    ):
        print(f"⏭️  PlantVillage already exists at {dest}, skipping.")
        return True

    if not check_kaggle_cli():
        print(
            "\n📋 Manual download instructions for PlantVillage:\n"
            f"   1. Go to: https://www.kaggle.com/datasets/{PLANTVILLAGE_SLUG}\n"
            f"   2. Download and extract to: {dest}\n"
            f"   Alt: https://www.kaggle.com/datasets/{PLANTVILLAGE_ALT_SLUG}\n"
        )
        return False

    return download_kaggle_dataset(PLANTVILLAGE_SLUG, dest)


def download_plantdoc():
    """Download PlantDoc dataset from GitHub."""
    dest = RAW_DIR / "plantdoc"
    zip_path = dest / "plantdoc-master.zip"

    # Check if already downloaded
    if any(dest.iterdir()) and not all(
        f.name == ".gitkeep" for f in dest.iterdir()
    ):
        print(f"⏭️  PlantDoc already exists at {dest}, skipping.")
        return True

    ensure_dir(dest)

    # Try GitHub download
    success = download_url(PLANTDOC_GITHUB_URL, zip_path)

    if success and zip_path.exists():
        print(f"📂 Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest)
            zip_path.unlink()  # Remove zip after extraction
            print("✅ PlantDoc extracted successfully.")
            return True
        except zipfile.BadZipFile:
            print("❌ Corrupt zip file. Try manual download.")
            zip_path.unlink()
            return False
    else:
        print(
            "\n📋 Manual download instructions for PlantDoc:\n"
            "   1. Go to: https://github.com/pratikkayal/PlantDoc-Dataset\n"
            f"   2. Download ZIP and extract to: {dest}\n"
            "   Alt: https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset\n"
        )
        return False


def download_kcc():
    """Download KCC dataset from Kaggle."""
    dest = RAW_DIR / "kcc"

    # Check if already downloaded
    if any(dest.iterdir()) and not all(
        f.name == ".gitkeep" for f in dest.iterdir()
    ):
        print(f"⏭️  KCC dataset already exists at {dest}, skipping.")
        return True

    if not check_kaggle_cli():
        print(
            "\n📋 Manual download instructions for KCC dataset:\n"
            "   Option 1 (Kaggle): Search 'Kisan Call Centre' on kaggle.com\n"
            f"   Option 2 (data.gov.in): Search 'Kisan Call Centre' at https://data.gov.in\n"
            "   Option 3 (AIKosh): https://www.indiaai.gov.in — 'KCC Transcripts'\n"
            f"   Extract to: {dest}\n"
        )
        return False

    return download_kaggle_dataset(KCC_SLUG, dest)


def download_yield():
    """Provide instructions for downloading yield data (requires manual portal access)."""
    dest = RAW_DIR / "yield"
    ensure_dir(dest)

    print(
        "\n" + "=" * 70 + "\n"
        "📊 YIELD DATASET — Manual Download Required\n"
        "=" * 70 + "\n"
        "\n"
        "District-level yield data must be downloaded manually from\n"
        "government portals (they use interactive query builders).\n"
        "\n"
        "Recommended sources (in order of preference):\n"
        "\n"
        "  1. UPAg (Unified Portal for Agricultural Statistics)\n"
        "     → https://upag.gov.in/\n"
        "     → Navigate: Reports → Area, Production, Yield\n"
        "     → Filter: State = Uttar Pradesh, Crop = Rice / Wheat\n"
        "     → Download CSV\n"
        "\n"
        "  2. DES (Directorate of Economics & Statistics)\n"
        "     → https://aps.dac.gov.in/APY/Public_Report1.aspx\n"
        "     → Filter by State, District, Crop, Year range\n"
        "\n"
        "  3. ICRISAT District-Level Database\n"
        "     → http://data.icrisat.org/dld/src/crops.html\n"
        "     → Includes yield, rainfall, and fertilizer data\n"
        "     → Best option for combined features\n"
        "\n"
        "  4. data.gov.in\n"
        "     → https://data.gov.in\n"
        "     → Search: 'crop production statistics'\n"
        "\n"
        f"Save downloaded files to: {dest}\n"
        "\n" + "=" * 70
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for AgriAssist Milestone 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/download_data.py --all\n"
            "  python scripts/download_data.py --plantvillage --plantdoc\n"
            "  python scripts/download_data.py --kcc\n"
        ),
    )
    parser.add_argument(
        "--plantvillage", action="store_true", help="Download PlantVillage dataset"
    )
    parser.add_argument(
        "--plantdoc", action="store_true", help="Download PlantDoc dataset"
    )
    parser.add_argument(
        "--kcc", action="store_true", help="Download KCC Q&A dataset"
    )
    parser.add_argument(
        "--yield-data",
        action="store_true",
        dest="yield_data",
        help="Show yield data download instructions",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download all datasets"
    )

    args = parser.parse_args()

    # If no flags, show help
    if not any([args.plantvillage, args.plantdoc, args.kcc, args.yield_data, args.all]):
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print("  AgriAssist — Dataset Downloader (Milestone 2)")
    print("=" * 70)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data dir : {RAW_DIR}")
    print("=" * 70 + "\n")

    results = {}

    if args.all or args.plantvillage:
        results["PlantVillage"] = download_plantvillage()
        print()

    if args.all or args.plantdoc:
        results["PlantDoc"] = download_plantdoc()
        print()

    if args.all or args.kcc:
        results["KCC"] = download_kcc()
        print()

    if args.all or args.yield_data:
        results["Yield"] = download_yield()
        print()

    # Summary
    print("\n" + "=" * 70)
    print("  Download Summary")
    print("=" * 70)
    for name, success in results.items():
        status = "✅ Ready" if success else "⚠️  Needs manual download"
        print(f"  {name:20s} {status}")
    print("=" * 70)

    # Return non-zero if any download failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
