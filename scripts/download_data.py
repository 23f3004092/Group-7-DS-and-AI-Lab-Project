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
    - KCC transcripts (data.gov.in API) → data/raw/kcc/

  YIELD:
    - District-level yield data         → data/raw/yield/  (manual)

Requirements:
  pip install kaggle requests tqdm pandas pyarrow
  (pandas + pyarrow are only needed for the KCC download)

Kaggle datasets require a valid ~/.kaggle/kaggle.json API token.
See: https://www.kaggle.com/docs/api

-------------------------------------------------------------------------
KCC dataset — how to download (data.gov.in GET API)
-------------------------------------------------------------------------
The KCC (Kisan Call Centre) query-answer transcripts are fetched directly
from the official Open Government Data platform API, resource:
  https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f

1. Get an API key (one time):
     - Register / log in at https://data.gov.in
     - Go to "My Account" → "Generate API Key"
   The sample key shown in the API docs returns at most 10 records —
   you need your own key for a full download.

2. Run the downloader, passing the API parameters as arguments:
     python download_data.py --kcc --kcc-api-key YOUR_KEY
     python download_data.py --kcc --kcc-api-key YOUR_KEY \
            --kcc-state "UTTAR PRADESH" --kcc-year 2025 --kcc-months 1-12

   Arguments (all optional except the key):
     --kcc-api-key    your personal data.gov.in API key   (required)
     --kcc-state      StateName filter (default: UTTAR PRADESH)
     --kcc-year       year filter      (default: 2025)
     --kcc-months     months to fetch: "1-12", "1,3,7", "6" (default: 1-12)
     --kcc-page-size  records per API call (default: 5000)

3. Output (under data/raw/kcc/):
     - one JSONL file per month  → RAG-ready, one record per line,
       Hindi text preserved (UTF-8)
     - one combined Parquet file → for EDA with pandas
   Months with no data on the server are skipped. The download is
   resumable: rerunning skips months whose JSONL is already complete.

Notes:
  - data.gov.in blocks the default python-requests User-Agent, so the
    script sends a browser-like one. It also retries transient 502s.
  - Record fields: KCCCallID, CreatedOn, StateName, DistrictName,
    BlockName, Sector, Category, Crop, Season, QueryType, QueryText,
    KccAns, day, month, year.
-------------------------------------------------------------------------

Usage:
  python download_data.py --all --kcc-api-key KEY   # Strategy A + KCC + yield
  python download_data.py --rice --wheat            # Vision only (Strategy A)
  python download_data.py --expand                  # Strategy D extras
  python download_data.py --kcc --kcc-api-key KEY   # KCC only
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

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

# --- NLP / RAG: KCC transcripts via data.gov.in GET API ---
KCC_RESOURCE_ID = "cef25fe2-9231-4128-8aec-2c948fedd43f"
KCC_BASE_URL = f"https://api.data.gov.in/resource/{KCC_RESOURCE_ID}"
# Kisan Call Centre transcripts: farmer queries + FTA answers.
# Filterable by StateName / year / month; paginated via offset + limit.
KCC_MAX_RETRIES = 8       # data.gov.in intermittently returns 502s
KCC_TIMEOUT_S = 120
KCC_REQUEST_DELAY_S = 0.5  # pause between calls to be polite to the server


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
# NLP / RAG — KCC transcripts from the data.gov.in GET API
# ---------------------------------------------------------------------------


def _parse_months(spec: str) -> list:
    """Parse a months spec like "1-12", "1,3,7" or "6" into a sorted list."""
    months = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            months.update(range(int(start), int(end) + 1))
        elif part:
            months.add(int(part))
    if not months or any(m < 1 or m > 12 for m in months):
        raise ValueError(f"invalid months spec: {spec!r} (use e.g. '1-12' or '1,3,7')")
    return sorted(months)


def _kcc_fetch_page(session, api_key, state, year, month, offset, limit) -> dict:
    """Fetch one page of KCC results, retrying on transient failures."""
    params = {
        "api-key": api_key,
        "format": "json",
        "offset": offset,
        "limit": limit,
        "filters[StateName]": state,
        "filters[year]": year,
        "filters[month]": month,
    }
    last_err = None
    for attempt in range(1, KCC_MAX_RETRIES + 1):
        try:
            resp = session.get(KCC_BASE_URL, params=params, timeout=KCC_TIMEOUT_S)
            resp.raise_for_status()
            payload = resp.json()
            if "records" not in payload or "total" not in payload:
                raise ValueError(f"unexpected response shape: {list(payload)[:10]}")
            return payload
        except (requests.RequestException, ValueError) as err:
            last_err = err
            if attempt < KCC_MAX_RETRIES:
                wait = min(60, 5 * 2 ** (attempt - 1))
                print(f"    attempt {attempt} failed ({err}); retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(
        f"KCC month {month} offset {offset}: giving up after {KCC_MAX_RETRIES} attempts"
    ) from last_err


def _kcc_download_month(session, api_key, state, year, month, page_size, dest) -> int:
    """Download one month to JSONL. Returns the number of records on disk."""
    state_slug = state.lower().replace(" ", "_")
    out_path = dest / f"kcc_{state_slug}_{year}_month_{month:02d}.jsonl"

    first = _kcc_fetch_page(session, api_key, state, year, month, offset=0, limit=1)
    total = int(first["total"])
    if total == 0:
        print(f"  Month {month:02d}: no data on server, skipping.")
        return 0

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing == total:
            print(f"  Month {month:02d}: already complete ({existing} records), skipping.")
            return existing
        print(f"  Month {month:02d}: found partial file ({existing}/{total}), re-downloading.")

    print(f"  Month {month:02d}: downloading {total} records...")
    tmp_path = out_path.with_suffix(".jsonl.part")
    fetched = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        offset = 0
        while offset < total:
            payload = _kcc_fetch_page(
                session, api_key, state, year, month, offset=offset, limit=page_size
            )
            records = payload["records"]
            if not records:
                break
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fetched += len(records)
            offset += len(records)
            print(f"    {fetched}/{total} records")
            time.sleep(KCC_REQUEST_DELAY_S)

    if fetched != total:
        print(f"    ⚠️  expected {total} records but got {fetched} "
              "(server total may have shifted mid-download)")
    tmp_path.replace(out_path)
    return fetched


def download_kcc(api_key, state, year, months_spec, page_size):
    """Download KCC transcripts from data.gov.in into data/raw/kcc/.

    Writes one JSONL file per month (RAG-ready) plus a combined Parquet
    file for EDA. Resumable: complete months are skipped on rerun.
    """
    dest = ensure_dir(RAW_DIR / "kcc")

    if not api_key:
        print(
            "\n📋 KCC download needs a data.gov.in API key:\n"
            "   1. Register / log in at https://data.gov.in\n"
            "   2. Go to 'My Account' → 'Generate API Key'\n"
            "   3. Rerun with: --kcc --kcc-api-key YOUR_KEY\n"
        )
        return False

    try:
        months = _parse_months(months_spec)
    except ValueError as err:
        print(f"❌ {err}")
        return False

    # data.gov.in rejects the default python-requests User-Agent
    # (502s / timeouts), so present a browser-like one.
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    print(f"📦 KCC transcripts | state={state} year={year} months={months} → {dest}")
    summary = {}
    failed = []
    for month in months:
        try:
            summary[month] = _kcc_download_month(
                session, api_key, state, year, month, page_size, dest
            )
        except RuntimeError as err:
            print(f"  ❌ {err}")
            failed.append(month)

    print("\n  Records per month:")
    for month, n in summary.items():
        print(f"    {year}-{month:02d}: {n}")
    print(f"    TOTAL: {sum(summary.values())}")

    # Combined Parquet for EDA (pandas + pyarrow only needed here).
    state_slug = state.lower().replace(" ", "_")
    month_files = [
        dest / f"kcc_{state_slug}_{year}_month_{m:02d}.jsonl"
        for m, n in summary.items() if n > 0
    ]
    if month_files:
        try:
            import pandas as pd

            df = pd.concat(
                [pd.read_json(p, lines=True) for p in month_files],
                ignore_index=True,
            )
            parquet_path = dest / f"kcc_{state_slug}_{year}_full.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"  ✅ Combined Parquet: {parquet_path} "
                  f"({len(df)} rows, {len(df.columns)} columns)")
        except ImportError:
            print("  ⚠️  pandas/pyarrow not installed — skipped the combined "
                  "Parquet (JSONL files are complete). Fix: pip install pandas pyarrow")

    if failed:
        print(f"  ⚠️  months failed after retries: {failed} — rerun to resume.")
        return False
    return True


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
            "  python download_data.py --all --kcc-api-key KEY   # Strategy A core + KCC + yield\n"
            "  python download_data.py --rice --wheat            # Vision only (Strategy A)\n"
            "  python download_data.py --expand                  # Add Strategy D extras\n"
            "  python download_data.py --kcc --kcc-api-key KEY   # KCC only (all months, UP, 2025)\n"
            "  python download_data.py --kcc --kcc-api-key KEY --kcc-state RAJASTHAN \\\n"
            "         --kcc-year 2024 --kcc-months 1,6-9         # KCC with custom filters\n"
            "  python download_data.py --everything --kcc-api-key KEY\n"
            "\n"
            "Get a data.gov.in API key: log in at https://data.gov.in →\n"
            "'My Account' → 'Generate API Key'.\n"
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
                        help="Download KCC transcripts from the data.gov.in API")
    parser.add_argument("--yield-data", action="store_true", dest="yield_data",
                        help="Show yield data download instructions")

    # KCC API parameters (used with --kcc / --all / --everything)
    kcc_group = parser.add_argument_group(
        "KCC API options",
        "Parameters for the data.gov.in KCC transcripts API. "
        "An API key is required: https://data.gov.in → My Account → Generate API Key.",
    )
    kcc_group.add_argument("--kcc-api-key", dest="kcc_api_key", default=None,
                           help="Your personal data.gov.in API key (required for --kcc)")
    kcc_group.add_argument("--kcc-state", dest="kcc_state", default="UTTAR PRADESH",
                           help='StateName filter, e.g. "UTTAR PRADESH" (default: %(default)s)')
    kcc_group.add_argument("--kcc-year", dest="kcc_year", default="2025",
                           help="Year filter (default: %(default)s)")
    kcc_group.add_argument("--kcc-months", dest="kcc_months", default="1-12",
                           help='Months to fetch: "1-12", "1,3,7" or "6" (default: %(default)s)')
    kcc_group.add_argument("--kcc-page-size", dest="kcc_page_size", type=int, default=5000,
                           help="Records per API call (default: %(default)s)")

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
        results["KCC"] = download_kcc(
            api_key=args.kcc_api_key,
            state=args.kcc_state,
            year=args.kcc_year,
            months_spec=args.kcc_months,
            page_size=args.kcc_page_size,
        )
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
