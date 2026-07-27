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
    - UP advisory/scheme PDF corpus     → data/raw/pdfs/  (--pdfs, Drive zip id in .env)

  YIELD:
    - District-level yield data         → data/raw/yield/  (manual)

Requirements:
  pip install kaggle requests tqdm pandas pyarrow
  (pandas + pyarrow are optional, used for combining KCC records to Parquet)

Kaggle datasets require a valid ~/.kaggle/kaggle.json API token.
See: https://www.kaggle.com/docs/api

Usage:
  python scripts/download_data.py --all          # Strategy A + KCC + yield
  python scripts/download_data.py --rice --wheat  # Vision only (Strategy A)
  python scripts/download_data.py --expand        # Strategy D extras
  python scripts/download_data.py --kcc           # KCC only
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def get_api_key(cli_key: str = None) -> str:
    """Get the data.gov.in API key from CLI argument, environment variable, or .env file at root."""
    if cli_key:
        return cli_key

    # Check environment variables
    for env_var in ["DATA_GOV_KEY", "KCC_API_KEY", "API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val.strip()

    # Read from .env file at root
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("DATA_GOV_KEY", "KCC_API_KEY", "API_KEY") and val:
                            return val
        except Exception as e:
            print(f"⚠️  Could not read .env file at {env_path}: {e}")
    return None


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
    """Parse a months spec like "1-12", "1,3,7" or ""/"all" (for no month filter) into a list."""
    if not spec or str(spec).strip().lower() in ("all", "none", "", "--"):
        return [None]
    months = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            months.update(range(int(start), int(end) + 1))
        elif part:
            months.add(int(part))
    if not months or any(m < 1 or m > 12 for m in months):
        raise ValueError(f"invalid months spec: {spec!r} (use e.g. '1-12' or '' for all months)")
    return sorted(months)


def _parse_years(spec: str) -> list:
    """Parse a years spec like "2020-2025", "2020,2023" or ""/"all" into a sorted list of strings."""
    if not spec or str(spec).strip().lower() in ("all", "none", "", "--"):
        return [None]
    years = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(str(y) for y in range(int(start), int(end) + 1))
        elif part:
            years.add(str(int(part)))
    return sorted(years)


def _kcc_fetch_page(session, api_key, state, year, month, offset, limit, fmt="json") -> dict:
    """Fetch one page of KCC results from data.gov.in API, retrying on transient failures."""
    params = {
        "api-key": api_key,
        "format": fmt,
        "offset": offset,
        "limit": limit,
    }
    if state and state.upper() != "ALL" and state != "--":
        params["filters[StateName]"] = state
    if year:
        params["filters[year]"] = str(year)
    if month is not None:
        params["filters[month]"] = str(month)

    last_err = None
    for attempt in range(1, KCC_MAX_RETRIES + 1):
        try:
            resp = session.get(KCC_BASE_URL, params=params, timeout=KCC_TIMEOUT_S)
            resp.raise_for_status()
            if fmt != "json":
                return resp.text
            payload = resp.json()
            if "records" not in payload or "total" not in payload:
                raise ValueError(f"unexpected response shape: {list(payload)[:10] if isinstance(payload, dict) else payload[:100]}")
            return payload
        except (requests.RequestException, ValueError) as err:
            last_err = err
            if attempt < KCC_MAX_RETRIES:
                wait = min(60, 5 * 2 ** (attempt - 1))
                print(f"    attempt {attempt} failed ({err}); retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(
        f"KCC API request failed (month={month}, offset={offset}): giving up after {KCC_MAX_RETRIES} attempts"
    ) from last_err


def _kcc_download_batch(session, api_key, state, year, month, page_size, dest, fmt="json", start_offset=0, max_limit=None) -> int:
    """Download KCC records for a year/month batch using pagination. Returns records saved."""
    state_slug = (state or "all").lower().replace(" ", "_")
    state_short = "up" if state_slug == "uttar_pradesh" else state_slug
    month_suffix = f"_month_{month:02d}" if isinstance(month, int) else ""

    if fmt not in ("json", "csv"):
        # For non-tabular formats like xml, make a single API request with offset/limit
        out_path = dest / f"kcc_{state_slug}_{year or 'all'}{month_suffix}.{fmt}"
        print(f"  Fetching {out_path.name} ({fmt.upper()})...")
        content = _kcc_fetch_page(session, api_key, state, year, month, offset=start_offset, limit=max_limit or page_size, fmt=fmt)
        out_path.write_text(content, encoding="utf-8")
        print(f"    Saved {out_path}")
        return 1

    # For json or csv, paginate reliably using json responses so all records are fetched cleanly without timing out
    jsonl_path = dest / f"kcc_{state_slug}_{year or 'all'}{month_suffix}.jsonl"
    csv_path = dest / (f"kcc_{state_short}_{year or 'all'}.csv" if not month_suffix else f"kcc_{state_slug}_{year or 'all'}{month_suffix}.csv")
    out_path = csv_path if fmt == "csv" else jsonl_path

    first = _kcc_fetch_page(session, api_key, state, year, month, offset=start_offset, limit=1, fmt="json")
    total = int(first["total"])
    if total == 0:
        print(f"  {out_path.name}: no data on server, skipping.")
        return 0

    if max_limit is not None:
        total = min(total, max_limit)

    if out_path.exists():
        if fmt == "csv":
            with out_path.open("r", encoding="utf-8") as f:
                existing = max(0, sum(1 for _ in f) - 1)
        else:
            with out_path.open("r", encoding="utf-8") as f:
                existing = sum(1 for _ in f)
        if existing == total:
            print(f"  {out_path.name}: already complete ({existing:,} records), skipping.")
            return existing
        print(f"  {out_path.name}: found partial file ({existing:,}/{total:,}), re-downloading.")

    print(f"  {out_path.name}: downloading {total:,} records (year={year or 'ALL'}, month={month or 'ALL'})...")
    tmp_path = out_path.with_suffix(f".{fmt}.part")
    fetched = 0
    header_written = False

    with tmp_path.open("w", encoding="utf-8") as f:
        offset = start_offset
        while fetched < total:
            batch_size = min(page_size, total - fetched)
            payload = _kcc_fetch_page(
                session, api_key, state, year, month, offset=offset, limit=batch_size, fmt="json"
            )
            records = payload.get("records", [])
            if not records:
                break
            if fmt == "csv":
                import csv
                if not header_written:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    header_written = True
                for rec in records:
                    writer.writerow(rec)
            else:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fetched += len(records)
            offset += len(records)
            print(f"    {fetched:,}/{total:,} records")
            time.sleep(KCC_REQUEST_DELAY_S)

    if fetched != total:
        print(f"    ⚠️  expected {total:,} records but got {fetched:,} "
              "(server total may have shifted mid-download)")
    tmp_path.replace(out_path)
    return fetched


def download_kcc(api_key=None, state="UTTAR PRADESH", year="2025", months_spec="",
                 fmt="json", offset=0, limit=None, page_size=5000):
    """Download KCC transcripts from data.gov.in into data/raw/kcc/.

    Reads API key automatically from .env file at root if not supplied.
    By default (months_spec=""), does NOT filter by month, fetching all records
    for each year directly in paginated batches.
    """
    dest = ensure_dir(RAW_DIR / "kcc")
    api_key = get_api_key(api_key)

    if not api_key:
        print(
            "\n❌ KCC download requires a data.gov.in API key.\n"
            "   1. Make sure your .env file at the project root exists and contains:\n"
            "      DATA_GOV_KEY=your_api_key_here\n"
            "   2. Or pass your key via CLI:\n"
            "      python scripts/download_data.py --kcc --kcc-api-key YOUR_KEY\n"
        )
        return False

    try:
        months = _parse_months(months_spec)
        years = _parse_years(year)
    except ValueError as err:
        print(f"❌ {err}")
        return False

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    print(f"📦 KCC transcripts | state={state} years={years} months={months_spec or 'ALL'} format={fmt} → {dest}")
    summary = {}
    failed = []
    for y in years:
        for month in months:
            key = f"{y}-{month:02d}" if isinstance(month, int) else f"{y or 'all'}"
            try:
                summary[key] = _kcc_download_batch(
                    session, api_key, state, y, month, page_size, dest,
                    fmt=fmt, start_offset=offset, max_limit=limit
                )
            except RuntimeError as err:
                print(f"  ❌ {err}")
                failed.append(key)

    print("\n  Records processed:")
    for m, n in summary.items():
        print(f"    Batch {m}: {n:,}" if isinstance(n, int) else f"    Batch {m}: {n}")
    total_recs = sum(summary.values()) if all(isinstance(v, int) for v in summary.values()) else len(summary)
    print(f"    TOTAL: {total_recs:,}")

    # Optional Parquet conversion when downloading as JSONL
    if fmt == "json":
        state_slug = (state or "all").lower().replace(" ", "_")
        try:
            import pandas as pd
            jsonl_files = sorted(dest.glob(f"kcc_{state_slug}_*.jsonl"))
            if jsonl_files:
                df = pd.concat([pd.read_json(p, lines=True) for p in jsonl_files], ignore_index=True)
                parquet_path = dest / f"kcc_{state_slug}_full.parquet"
                df.to_parquet(parquet_path, index=False)
                print(f"  ✅ Combined Parquet: {parquet_path} ({len(df):,} rows, {len(df.columns)} columns)")
        except ImportError:
            print("  ⚠️  pandas/pyarrow not installed — skipped Parquet conversion.")

    if failed:
        print(f"  ⚠️  batches failed after retries: {failed} — rerun to resume.")
        return False
    return True



# ---------------------------------------------------------------------------
# NLP / RAG — UP government advisory PDF corpus (Google Drive)
# ---------------------------------------------------------------------------

PDF_EXPECTED_FOLDERS = ["Other_docs", "PPQS_Advisories", "Schemes", "UP_ACP_PDFs"]


def _env_lookup(keys):
    """Read a value from environment variables or the .env file at project root."""
    for k in keys:
        val = os.environ.get(k)
        if val:
            return val.strip()
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() in keys and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


def download_pdfs(zip_id=None, folder_id=None):
    """Download the RAG PDF corpus (187 UP advisory/scheme PDFs) into data/raw/pdfs/.

    Preferred route: a single zip of the DS_AI_RAG_pdfs folder shared on Drive
    (PDF_ZIP_DRIVE_ID in .env) — gdown folder downloads cap at ~50 files per
    folder, and PPQS_Advisories alone has 90, so the zip route is reliable.
    """
    dest = ensure_dir(RAW_DIR / "pdfs")

    def n_pdfs():
        return sum(1 for _ in dest.glob("**/*.pdf"))

    if all((dest / d).is_dir() for d in PDF_EXPECTED_FOLDERS) and n_pdfs() > 0:
        print(f"⏭️  RAG PDFs already present at {dest} ({n_pdfs()} PDFs), skipping.")
        return True

    zip_id = zip_id or _env_lookup(["PDF_ZIP_DRIVE_ID"])
    folder_id = folder_id or _env_lookup(["PDF_FOLDER_DRIVE_ID"])

    if not zip_id and not folder_id:
        print(
            "\n📋 RAG PDF corpus — no Drive id configured. Two options:\n"
            "   A) Automated (recommended):\n"
            "      1. In Google Drive, zip the DS_AI_RAG_pdfs folder (contains\n"
            f"         {', '.join(PDF_EXPECTED_FOLDERS)})\n"
            "      2. Share the zip as 'Anyone with the link' and copy its file id\n"
            "      3. Add to .env at project root:  PDF_ZIP_DRIVE_ID=<file-id>\n"
            "      4. Re-run: python scripts/download_data.py --pdfs\n"
            "   B) Manual: download/extract the four folders yourself into\n"
            f"      {dest}\n"
        )
        return False

    def _gdown():
        # imported lazily: not needed when a downloaded zip is already on disk
        try:
            import gdown
            return gdown
        except ImportError:
            print("❌ gdown not installed. Run: pip install gdown")
            return None

    if zip_id:
        zip_path = dest / "rag_pdfs.zip"
        if zip_path.exists() and zip_path.stat().st_size > 0:
            print(f"⏭️  Reusing already-downloaded zip: {zip_path} "
                  f"({zip_path.stat().st_size/1e6:.0f} MB)")
        else:
            gdown = _gdown()
            if gdown is None:
                return False
            print(f"📦 RAG PDFs: downloading Drive zip → {zip_path}")
            gdown.download(id=zip_id, output=str(zip_path), quiet=False)
        if not zip_path.exists():
            print("❌ zip download failed — check the id and link-sharing setting.")
            return False
        # Windows MAX_PATH (260 chars): several advisory filenames are ~150 chars,
        # so extraction needs the \\?\ extended-length path prefix.
        extract_root = str(dest.resolve())
        if os.name == "nt" and not extract_root.startswith("\\\\?\\"):
            extract_root = "\\\\?\\" + extract_root
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
        zip_path.unlink()
    else:
        gdown = _gdown()
        if gdown is None:
            return False
        print("📦 RAG PDFs: downloading Drive folder via gdown "
              "(⚠️ gdown caps ~50 files per folder — the zip route is more reliable)")
        gdown.download_folder(id=folder_id, output=str(dest), quiet=False,
                              use_cookies=False, remaining_ok=True)

    # Flatten a single wrapper directory (e.g. DS_AI_RAG_pdfs/) if the archive had one
    if not all((dest / d).is_dir() for d in PDF_EXPECTED_FOLDERS):
        for sub in list(dest.iterdir()):
            if sub.is_dir() and all((sub / d).is_dir() for d in PDF_EXPECTED_FOLDERS):
                for item in list(sub.iterdir()):
                    target = dest / item.name
                    if not target.exists():
                        item.rename(target)
                sub.rmdir()
                break

    missing = [d for d in PDF_EXPECTED_FOLDERS if not (dest / d).is_dir()]
    if missing or n_pdfs() == 0:
        print(f"⚠️  Layout unexpected after download (missing: {missing}, PDFs: {n_pdfs()}).")
        print(f"   Expected {dest}/<folder>/*.pdf for {PDF_EXPECTED_FOLDERS} — see data/README.md")
        return False
    print(f"✅ RAG PDF corpus ready: {n_pdfs()} PDFs → {dest}")
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
                        help="Download KCC transcripts from the data.gov.in API")
    parser.add_argument("--pdfs", action="store_true",
                        help="Download the RAG PDF corpus from Google Drive into data/raw/pdfs/")
    parser.add_argument("--pdf-zip-id", dest="pdf_zip_id", default=None,
                        help="Drive file id of the zipped PDF corpus (defaults to PDF_ZIP_DRIVE_ID in .env)")
    parser.add_argument("--pdf-folder-id", dest="pdf_folder_id", default=None,
                        help="Drive folder id of the PDF corpus (fallback; gdown caps ~50 files/folder)")
    parser.add_argument("--yield-data", action="store_true", dest="yield_data",
                        help="Show yield data download instructions")

    # KCC API parameters (used with --kcc / --all / --everything)
    kcc_group = parser.add_argument_group(
        "KCC API options",
        "Parameters for the data.gov.in KCC transcripts API (resource: cef25fe2-9231-4128-8aec-2c948fedd43f). "
        "The API key is automatically read from the .env file at root (`DATA_GOV_KEY`) or passed via --kcc-api-key.",
    )
    kcc_group.add_argument("--kcc-api-key", dest="kcc_api_key", default=None,
                           help="Your personal data.gov.in API key (defaults to DATA_GOV_KEY in .env)")
    kcc_group.add_argument("--kcc-format", dest="kcc_format", default="json", choices=["json", "xml", "csv"],
                           help="Output format: json, xml, csv (default: %(default)s)")
    kcc_group.add_argument("--kcc-state", dest="kcc_state", default="UTTAR PRADESH",
                           help='StateName filter, e.g. "UTTAR PRADESH" or "" for all (default: %(default)s)')
    kcc_group.add_argument("--kcc-year", dest="kcc_year", default="2025",
                           help='Year filter or multi-year range, e.g. "2025" or "2020-2025" (default: %(default)s)')
    kcc_group.add_argument("--kcc-months", dest="kcc_months", default="",
                           help='Months filter: "" for all months (no month filter), or "1-12", "1,3" (default: %(default)r)')
    kcc_group.add_argument("--kcc-offset", dest="kcc_offset", type=int, default=0,
                           help="Number of records to skip for pagination (default: %(default)s)")
    kcc_group.add_argument("--kcc-limit", dest="kcc_limit", type=int, default=None,
                           help="Maximum total records to return across download (default: all available)")
    kcc_group.add_argument("--kcc-page-size", dest="kcc_page_size", type=int, default=5000,
                           help="Records per API call when paginating (default: %(default)s)")

    # Combo flags
    parser.add_argument("--all", action="store_true",
                        help="Download Strategy A core (Rice + Wheat + PlantDoc + KCC + yield)")
    parser.add_argument("--everything", action="store_true",
                        help="Download ALL datasets (Strategy A + D expansion)")

    args = parser.parse_args()

    # If no flags, show help
    has_any = any([
        args.rice, args.wheat, args.plantdoc, args.expand, args.rice_extra,
        args.plantvillage, args.kcc, args.pdfs, args.yield_data, args.all, args.everything,
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
            fmt=args.kcc_format,
            offset=args.kcc_offset,
            limit=args.kcc_limit,
            page_size=args.kcc_page_size,
        )
        print()

    if args.all or args.everything or args.pdfs:
        results["RAG PDFs"] = download_pdfs(zip_id=args.pdf_zip_id, folder_id=args.pdf_folder_id)
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
