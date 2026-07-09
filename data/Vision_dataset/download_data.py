#!/usr/bin/env python3
"""
download_data.py — Automated dataset downloader for AgriAssist (Milestone 2)

Downloads and organises all raw datasets into data/raw/:

  VISION (Strategy A — Primary):
    - Rice Leaf Diseases (Kaggle)       → data/raw/rice_diseases/
    - Wheat Plant Diseases (Kaggle)     → data/raw/wheat_diseases/

  VISION (Strategy D — Expansion, optional):
    - Rice Leaf Disease Images (Kaggle) → data/raw/rice_diseases_extra/
    - PlantVillage (Kaggle)             → data/raw/plantvillage/

  VISION (Field benchmark):
    - PlantDoc (Kaggle)                 → data/raw/plantdoc/

  NLP / RAG:
    - KCC Q&A Logs (Kaggle)             → data/raw/kcc/

  YIELD:
    - District-level yield data         → data/raw/yield/  (manual)

Requirements:
  pip install kaggle requests tqdm

Kaggle datasets require a valid ~/.kaggle/kaggle.json API token.
See: https://www.kaggle.com/docs/api

Usage:
  python scripts/download_data.py --all          # Strategy A + KCC + yield
  python scripts/download_data.py --rice --wheat  # Vision only (Strategy A)
  python scripts/download_data.py --expand        # Strategy D extras
  python scripts/download_data.py --kcc           # KCC only
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# --- Strategy A: Primary Rice & Wheat datasets ---
RICE_SLUG = "vbookshelf/rice-leaf-diseases"
# Classes: Brown Spot, Blast (Leaf & Neck), Bacterial Leaf Blight, Healthy
# ~38 MB, ~28K downloads, widely cited

WHEAT_SLUG = "kushagra3204/wheat-plant-diseases"
# Classes: Brown Rust, Yellow Rust, Stem Rust, Septoria, Blast, Powdery Mildew, etc.
# ~14K+ images, comprehensive wheat disease coverage

# --- Strategy D: Expansion datasets ---
RICE_EXTRA_SLUG = "nirmalsankalana/rice-leaf-disease-image"
# Additional rice disease images (~205 MB) for dataset integration

PLANTVILLAGE_SLUG = "abdallahalidev/plantvillage-dataset"
# 38 classes, ~54K images — optional supplementary for transfer learning

# --- Field benchmark ---
PLANTDOC_SLUG = "andresmgs/plantdec"
# PlantDoc field images (30 classes, ~74 MB) — domain gap evaluation

# --- NLP / RAG ---
KCC_SLUG = "daskoushik/farmers-call-query-data-qa"
# KCC Q&A pairs (~4.5 MB, ~179K records, 2 columns: questions, answers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_already_downloaded(dest: Path) -> bool:
    """Check if a directory has real content (not just .gitkeep)."""
    if not dest.exists():
        return False
    contents = list(dest.iterdir())
    return any(f.name != ".gitkeep" for f in contents)


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
        kaggle_dir = Path.home() / ".kaggle"
        has_token = (kaggle_dir / "kaggle.json").exists() or (kaggle_dir / "access_token").exists()
        if not has_token:
            print(
                "⚠️  Kaggle CLI found but ~/.kaggle credentials are missing.\n"
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
                "kaggle", "datasets", "download",
                "-d", slug,
                "-p", str(dest),
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


def _download_or_manual(slug: str, dest: Path, name: str) -> bool:
    """Attempt Kaggle download, or print manual instructions."""
    if _is_already_downloaded(dest):
        print(f"⏭️  {name} already exists at {dest}, skipping.")
        return True

    if not check_kaggle_cli():
        print(
            f"\n📋 Manual download instructions for {name}:\n"
            f"   1. Go to: https://www.kaggle.com/datasets/{slug}\n"
            f"   2. Download and extract to: {dest}\n"
        )
        return False

    return download_kaggle_dataset(slug, dest)


# ---------------------------------------------------------------------------
# Strategy A — Primary Rice & Wheat datasets
# ---------------------------------------------------------------------------


def download_rice():
    """Download Rice Leaf Diseases dataset (Strategy A primary)."""
    return _download_or_manual(
        RICE_SLUG,
        RAW_DIR / "rice_diseases",
        "Rice Leaf Diseases (vbookshelf)",
    )


def download_wheat():
    """Download Wheat Plant Diseases dataset (Strategy A primary)."""
    return _download_or_manual(
        WHEAT_SLUG,
        RAW_DIR / "wheat_diseases",
        "Wheat Plant Diseases (kushagra3204)",
    )


# ---------------------------------------------------------------------------
# Strategy D — Expansion datasets
# ---------------------------------------------------------------------------


def download_rice_extra():
    """Download additional Rice Leaf Disease images (Strategy D expansion)."""
    return _download_or_manual(
        RICE_EXTRA_SLUG,
        RAW_DIR / "rice_diseases_extra",
        "Rice Leaf Disease Images — extra (nirmalsankalana)",
    )


def download_plantvillage():
    """Download PlantVillage dataset (Strategy D, optional supplementary)."""
    return _download_or_manual(
        PLANTVILLAGE_SLUG,
        RAW_DIR / "plantvillage",
        "PlantVillage (supplementary)",
    )


# ---------------------------------------------------------------------------
# Field benchmark
# ---------------------------------------------------------------------------


def download_plantdoc():
    """Download PlantDoc field images (domain gap benchmark)."""
    return _download_or_manual(
        PLANTDOC_SLUG,
        RAW_DIR / "plantdoc",
        "PlantDoc (field benchmark)",
    )


# ---------------------------------------------------------------------------
# NLP / RAG
# ---------------------------------------------------------------------------


def download_kcc():
    """Download KCC Q&A dataset from Kaggle."""
    return _download_or_manual(
        KCC_SLUG,
        RAW_DIR / "kcc",
        "KCC Q&A Logs (daskoushik)",
    )


# ---------------------------------------------------------------------------
# Yield
# ---------------------------------------------------------------------------


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
            "  python scripts/download_data.py --all            # Strategy A core + KCC + yield\n"
            "  python scripts/download_data.py --rice --wheat    # Vision only (Strategy A)\n"
            "  python scripts/download_data.py --expand          # Add Strategy D extras\n"
            "  python scripts/download_data.py --kcc             # KCC only\n"
            "  python scripts/download_data.py --everything      # All datasets including expansion\n"
        ),
    )

    # Strategy A: Primary vision
    parser.add_argument("--rice", action="store_true",
                        help="Download Rice Leaf Diseases dataset (Strategy A)")
    parser.add_argument("--wheat", action="store_true",
                        help="Download Wheat Plant Diseases dataset (Strategy A)")
    parser.add_argument("--plantdoc", action="store_true",
                        help="Download PlantDoc field benchmark")

    # Strategy D: Expansion
    parser.add_argument("--expand", action="store_true",
                        help="Download Strategy D expansion datasets (Rice extra + PlantVillage)")
    parser.add_argument("--rice-extra", action="store_true", dest="rice_extra",
                        help="Download additional Rice disease images only")
    parser.add_argument("--plantvillage", action="store_true",
                        help="Download PlantVillage (supplementary, optional)")

    # NLP / Yield
    parser.add_argument("--kcc", action="store_true",
                        help="Download KCC Q&A dataset")
    parser.add_argument("--yield-data", action="store_true", dest="yield_data",
                        help="Show yield data download instructions")

    # Combo flags
    parser.add_argument("--all", action="store_true",
                        help="Download Strategy A core (Rice + Wheat + PlantDoc + KCC + yield)")
    parser.add_argument("--everything", action="store_true",
                        help="Download ALL datasets (Strategy A + D expansion)")

    args = parser.parse_args()

    # If no flags, show help
    has_any = any([
        args.rice, args.wheat, args.plantdoc, args.expand, args.rice_extra,
        args.plantvillage, args.kcc, args.yield_data, args.all, args.everything,
    ])
    if not has_any:
        parser.print_help()
        sys.exit(0)

    print("=" * 70)
    print("  AgriAssist — Dataset Downloader (Milestone 2)")
    print("=" * 70)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Raw data dir : {RAW_DIR}")
    print("=" * 70 + "\n")

    results = {}

    # --- Strategy A core ---
    if args.all or args.everything or args.rice:
        results["Rice Diseases"] = download_rice()
        print()

    if args.all or args.everything or args.wheat:
        results["Wheat Diseases"] = download_wheat()
        print()

    if args.all or args.everything or args.plantdoc:
        results["PlantDoc"] = download_plantdoc()
        print()

    # --- Strategy D expansion ---
    if args.everything or args.expand or args.rice_extra:
        results["Rice Extra"] = download_rice_extra()
        print()

    if args.everything or args.expand or args.plantvillage:
        results["PlantVillage"] = download_plantvillage()
        print()

    # --- NLP / Yield ---
    if args.all or args.everything or args.kcc:
        results["KCC"] = download_kcc()
        print()

    if args.all or args.everything or args.yield_data:
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

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
