"""
build_rag_artifacts.py
======================
Rebuilds the Qdrant RAG production artifacts (snapshot + manifest.json)
from updated KCC and/or PDF .jsonl chunk files.

The output artifacts are what setup_vectordb.py downloads from Google Drive
and restores into the local Qdrant instance.

Kaggle / Linux notes
--------------------
  * qdrant.exe is Windows-only. On Linux the script auto-downloads the
    statically-linked musl binary (no GLIBC dependency) from GitHub releases.
  * Embedding cache (--emb-cache) saves per-shard float16 .npy files so GPU
    time is not lost if the Kaggle session dies mid-build.
  * Use --kcc-max to take a crop-stratified subsample for faster iteration.

Usage
-----
  # Both corpora (most common):
  python scripts/build_rag_artifacts.py \\
      --kcc  data/processed/kcc/kcc_chunks_rag.jsonl \\
      --pdf  data/final/pdfs/pdf_chunks_final.jsonl \\
      --out  rag_production_bge_m3

  # KCC only (if only the KCC file changed):
  python scripts/build_rag_artifacts.py \\
      --kcc  data/processed/kcc/kcc_chunks_rag.jsonl \\
      --out  rag_production_bge_m3

  # PDF only:
  python scripts/build_rag_artifacts.py \\
      --pdf  data/final/pdfs/pdf_chunks_final.jsonl \\
      --out  rag_production_bge_m3

  # Kaggle: subsample + embedding cache (survives session crashes):
  python scripts/build_rag_artifacts.py \\
      --kcc  /kaggle/input/.../kcc_chunks_rag.jsonl \\
      --pdf  /kaggle/input/.../pdf_chunks_final.jsonl \\
      --out  /kaggle/working/rag_out \\
      --emb-cache /kaggle/working/emb_cache

Output directory (--out)
------------------------
  <out>/agri_knowledge.snapshot   <- upload to Drive, paste ID into setup_vectordb.py
  <out>/manifest.json             <- upload to Drive, paste ID into setup_vectordb.py
  <out>/shard_progress.json       <- resume checkpoint (delete to force a clean rebuild)
  <out>/qdrant_storage/           <- live Qdrant data directory
  <emb-cache>/emb_NNNNN.npy      <- float16 shard vectors (delete to re-embed)

Requirements
------------
  pip install qdrant-client sentence-transformers transformers torch tqdm requests

Qdrant server
-------------
The script starts Qdrant automatically:
  Windows : looks for qdrant.exe at --qdrant-bin (default: scripts/qdrant.exe)
  Linux   : auto-downloads the musl static binary from GitHub releases

If Qdrant is already running on :6333 it is reused as-is.
"""

import argparse
import gc
import gzip
import hashlib
import json
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# CONFIG  (mirrors notebook 08b cell 6)
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME   = "BAAI/bge-m3"
MODEL_TAG          = "bge_m3"
MODEL_MAX_TOKENS   = 512        # pinned for controlled comparison; bge-m3 native = 8192
EMBED_DIM_EXPECTED = 1024
QUERY_PREFIX       = ""         # bge-m3 takes NO prefixes
DOC_PREFIX         = ""
EMBED_BATCH_SIZE   = 64
CHUNK_TOKEN_BUDGET = 400        # re-split target for oversize KCC chunks
SHARD_SIZE         = 20_000
UPSERT_BATCH       = 256

COLLECTION_NAME    = "agri_knowledge"
QDRANT_URL         = "http://localhost:6333"

# Tier defaults (from last calibrated run; recorded in manifest for reference)
TIER_GROUNDED = 0.638
TIER_FALLBACK = 0.553

FUSION_WEIGHTS = {
    "policy":         {"pdf": 2.0, "kcc": 0.5},
    "field_practice": {"pdf": 0.5, "kcc": 2.0},
    "general":        {"pdf": 1.0, "kcc": 1.0},
}
TOP_K_DEFAULT = 5

# ---------------------------------------------------------------------------
# NORMALISATION HELPERS  (verbatim from notebook 08b cell 8)
# ---------------------------------------------------------------------------

def detect_language(text, sample_chars=1000):
    """Per-chunk script-ratio language tag: en / hi / mixed."""
    s = text[:sample_chars]
    dev = len(re.findall(r"[\u0900-\u097F]", s))
    lat = len(re.findall(r"[a-zA-Z]", s))
    total = max(dev + lat, 1)
    if dev / total > 0.15 and lat / total > 0.15:
        return "mixed"
    return "hi" if dev / total > 0.3 else "en"


DISTRICT_CANON = {
    "allahabad": "prayagraj",            "faizabad": "ayodhya",
    "prabuddh nagar": "shamli",          "prabudh nagar": "shamli",
    "bhim nagar": "sambhal",             "panchsheel nagar": "hapur",
    "jyotiba phule nagar": "amroha",     "jyotibaphule nagar": "amroha",
    "kanshi ram nagar": "kasganj",       "kanshiram nagar": "kasganj",
    "chhatrapati shahuji maharaj nagar": "amethi",
    "mahamaya nagar": "hathras",         "ramabai nagar": "kanpur dehat",
    "banaras": "varanasi",               "kashi": "varanasi",
    "kanpur city": "kanpur nagar",       "maharahganj": "maharajganj",
    "sant ravidas nagar": "bhadohi",
}


def canon_district(raw):
    if not raw or str(raw).lower() in ("unknown", "nan", "none", ""):
        return None
    d = re.sub(r"\s+", " ", str(raw).strip().lower())
    return DISTRICT_CANON.get(d, d)


CROP_CANON = {
    "rice": "rice",       "paddy": "rice",      "dhan": "rice",     "chawal": "rice",
    "wheat": "wheat",     "gehun": "wheat",      "gehu": "wheat",    "kanak": "wheat",
    "maize": "maize",     "makka": "maize",      "makai": "maize",   "bhutta": "maize",
    "corn": "maize",
    "sugarcane": "sugarcane", "ganna": "sugarcane", "noble cane": "sugarcane",
    "mustard": "mustard", "sarson": "mustard",   "raya": "mustard",
    "indian mustard": "mustard", "indian rapeseed and mustard": "mustard",
    "yellow sarson": "mustard",
    "urad": "urad",       "black gram": "urad",  "urd": "urad",      "urd bean": "urad",
    "gram": "gram",       "bengal gram": "gram", "chana": "gram",
    "chick pea": "gram",  "kabuli": "gram",
    "moong": "moong",     "green gram": "moong", "moong bean": "moong", "mung": "moong",
    "arhar": "arhar",     "pigeon pea": "arhar", "red gram": "arhar","tur": "arhar",
    "masur": "masur",     "lentil": "masur",
    "okra": "okra",       "bhindi": "okra",      "ladysfinger": "okra",
    "bajra": "bajra",     "pearl millet": "bajra","bulrush millet": "bajra",
    "spiked millet": "bajra",
    "jowar": "jowar",     "sorghum": "jowar",    "great millet": "jowar",
    "barley": "barley",   "jau": "barley",
    "sesame": "sesame",   "til": "sesame",       "gingelly": "sesame","sesamum": "sesame",
    "groundnut": "groundnut", "pea nut": "groundnut", "peanut": "groundnut",
    "mung phalli": "groundnut",
    "colocasia": "arvi",  "arvi": "arvi",        "arbi": "arvi",     "arum": "arvi",
    "cotton": "cotton",   "kapas": "cotton",
    "soybean": "soybean", "bhat": "soybean",
    "linseed": "linseed", "alsi": "linseed",
    "spinach": "spinach", "palak": "spinach",
    "methi": "fenugreek", "fenugreek": "fenugreek",
    "rajma": "rajma",     "french bean": "rajma",
    "sunflower": "sunflower", "suryamukhi": "sunflower",
    "finger millet": "ragi", "fingermillet": "ragi", "ragi": "ragi", "mandika": "ragi",
    "pea": "pea",         "peas": "pea",         "matar": "pea",
    "field peas": "pea",  "garden peas": "pea",
}

_CROP_PAREN = re.compile(r"^([^(]+?)\s*\((.*)\)\s*$")
_CROP_NULLS = ("unknown", "nan", "none", "", "na", "n/a", "other", "others")


def canon_crop(raw):
    """Normalise ANY crop surface form to one canonical token (idempotent)."""
    if raw is None:
        return None
    c = re.sub(r"\s+", " ", str(raw).strip().lower())
    if c in _CROP_NULLS:
        return None
    if c in CROP_CANON:
        return CROP_CANON[c]
    m = _CROP_PAREN.match(c)
    if m:
        base, inner = m.group(1).strip(), m.group(2)
        if base in CROP_CANON:
            return CROP_CANON[base]
        for alias in re.split(r"[/,]", inner):
            alias = alias.strip()
            if alias in CROP_CANON:
                return CROP_CANON[alias]
        return base
    return c


# ---------------------------------------------------------------------------
# JSONL READER
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Stream a .jsonl or .jsonl.gz file line by line."""
    p = str(path)
    if not p.endswith(".gz") and not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# CORPUS LOADERS
# ---------------------------------------------------------------------------

def load_pdf_chunks(path):
    """Load PDF chunk .jsonl -> unified payload schema."""
    unified = []
    for c in tqdm(read_jsonl(path), desc="PDF chunks"):
        m = c["metadata"]
        filename = m.get("filename", "")
        district = (
            canon_district(os.path.splitext(filename)[0])
            if m.get("source") == "up_acp" else None
        )
        unified.append({
            # --- shared fields ---
            "chunk_id":        m["chunk_id"],
            "source_type":     "pdf",
            "source":          m.get("source", ""),
            "text":            c["text"],
            "language":        m.get("detected_language") or detect_language(c["text"]),
            "year":            int(m["detected_year"]) if m.get("detected_year") else None,
            "crop":            None,
            "district":        district,
            "chunk_index":     m.get("chunk_index", 0),
            "n_chunks_in_doc": m.get("n_chunks_in_doc", 1),
            # --- pdf-only fields ---
            "filename":           filename,
            "doc_category":       m.get("doc_category", ""),
            "heading_hierarchy":  m.get("heading_hierarchy", ""),
            "page_start":         m.get("page_start"),
            "page_end":           m.get("page_end"),
            "has_table":          bool(m.get("has_table", False)),
            "extraction_method":  m.get("extraction_method", ""),
            "source_pdf_sha256":  m.get("source_pdf_sha256", ""),
        })
    print(f"  PDF chunks loaded: {len(unified):,}")
    return unified


def _kcc_chunk_id(text, meta, chunk_no, seq):
    """Deterministic uuid5 chunk id for KCC rows."""
    basis = "|".join([
        text,
        str(meta.get("crop")),
        str(meta.get("district")),
        str(meta.get("year")),
        str(meta.get("month")),
        str(seq),
        str(chunk_no),
    ])
    h = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kcc:{h}"))


def load_kcc_chunks(path, tokenizer, budget_hard):
    """Load KCC chunk .jsonl -> unified schema with token-budget re-split.

    Handles two formats automatically:

    OLD format (notebook 08b):
        {"text": "Q: ... A: ...", "metadata": {"crop": "rice", "district": "...",
         "year": .., "month": .., "season": ..}, "chunk_number": .., "total_chunks": ..}

    NEW format (e2e-dataset):
        {"id": "482", "question": "...", "answer": "...",
         "metadata": {"crop": "262", "category": "Cereals", "district": "AZAMGARH"}}

    In the new format:
      * text  = "Q: {question}\\nA: {answer}"
      * crop  = numeric ID -> cannot be canonicalised, stored as None
      * year / month / season / query_type are absent -> stored as None
    """
    sent_split = re.compile(r"(?<=[.!?।])\s+")

    def n_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    def split_to_budget(text, budget=CHUNK_TOKEN_BUDGET):
        sents = sent_split.split(text)
        parts, cur, cur_tok = [], [], 0
        for s in sents:
            t = n_tokens(s)
            if cur and cur_tok + t > budget:
                parts.append(" ".join(cur))
                cur, cur_tok = [], 0
            cur.append(s)
            cur_tok += t
        if cur:
            parts.append(" ".join(cur))
        return parts or [text]

    rows, n_resplit, n_seen = [], 0, 0
    for c in tqdm(read_jsonl(path), desc="KCC chunks"):
        n_seen += 1
        m = c.get("metadata", {})

        # ---- Detect format and extract text --------------------------------
        if "text" in c and c["text"]:
            # OLD format: text already assembled
            text = c["text"]
            crop_raw = m.get("crop")
            year  = m.get("year")
            month = m.get("month")
        elif "question" in c or "answer" in c:
            # NEW format: assemble Q+A into one retrieval chunk
            q = (c.get("question") or "").strip()
            a = (c.get("answer")   or "").strip()
            if not q and not a:
                continue
            text = f"Q: {q}\nA: {a}" if (q and a) else (q or a)
            crop_raw = m.get("crop")
            year, month = None, None
        else:
            continue   # unknown format, skip

        if len(text.strip()) < 10:
            continue

        # ---- Crop canonicalisation -----------------------------------------
        # New format uses numeric IDs ("262") that cannot be mapped to a crop name.
        if crop_raw and str(crop_raw).strip().isdigit():
            crop_canon = None
        else:
            crop_canon = canon_crop(crop_raw)

        # ---- Token-budget re-split -----------------------------------------
        pieces = [text]
        if len(text) > 900:
            if n_tokens(text) > budget_hard:
                pieces = split_to_budget(text)
                n_resplit += 1

        base = {
            "source_type":  "kcc",
            "source":       "kcc_qa",
            "language":     detect_language(text),
            "year":         int(year) if year else None,
            "crop":         crop_canon,
            "district":     canon_district(m.get("district")),
            "district_raw": m.get("district"),
            "season":       m.get("season") if m.get("season") not in (None, "unknown") else None,
            "query_type":   m.get("query_type"),
            "category":     m.get("category"),
            "month":        int(month) if month else None,
        }
        for i, piece in enumerate(pieces):
            row = dict(base)
            row["text"]            = piece
            row["chunk_index"]     = i if len(pieces) > 1 else int(c.get("chunk_number", 1)) - 1
            row["n_chunks_in_doc"] = len(pieces) if len(pieces) > 1 else int(c.get("total_chunks", 1))
            # Include record id in hash so new-format ids stay deterministic
            m_for_id = dict(m, _record_id=c.get("id", ""))
            row["chunk_id"]        = _kcc_chunk_id(piece, m_for_id, row["chunk_index"], n_seen)
            rows.append(row)

    print(f"  KCC chunks read: {n_seen:,} -> normalized rows: {len(rows):,}"
          f" (oversize re-split: {n_resplit:,})")
    return rows



# ---------------------------------------------------------------------------
# QDRANT HELPERS
# ---------------------------------------------------------------------------

def qdrant_alive(timeout=1):
    try:
        return requests.get(f"{QDRANT_URL}/readyz", timeout=timeout).ok
    except Exception:
        return False


def _download_qdrant_linux(dest_bin):
    """
    Auto-download the statically-linked musl Qdrant binary for Linux.
    Uses the musl asset (not gnu) to avoid GLIBC version mismatches —
    the same approach as notebook 08b cell 4.
    """
    print("  Fetching latest Qdrant release info ...")
    rel = requests.get(
        "https://api.github.com/repos/qdrant/qdrant/releases/latest", timeout=30
    ).json()
    # musl first (statically linked, no libc dependency)
    asset = None
    for suffix in ("x86_64-unknown-linux-musl.tar.gz",
                   "x86_64-unknown-linux-gnu.tar.gz"):
        asset = next((a for a in rel["assets"] if a["name"].endswith(suffix)), None)
        if asset:
            break
    if asset is None:
        raise RuntimeError(
            f"No linux x86_64 asset in {rel['tag_name']}: "
            + ", ".join(a["name"] for a in rel["assets"])
        )
    print(f"  Downloading Qdrant {rel['tag_name']} ({asset['name']}) ...")
    tar_path = dest_bin + ".tar.gz"
    with open(tar_path, "wb") as f:
        f.write(requests.get(asset["browser_download_url"], timeout=600).content)
    with tarfile.open(tar_path) as t:
        try:
            t.extractall(os.path.dirname(dest_bin), filter="data")   # py>=3.12
        except TypeError:
            t.extractall(os.path.dirname(dest_bin))
    os.remove(tar_path)
    # The binary is extracted as 'qdrant' in the same directory
    extracted = os.path.join(os.path.dirname(dest_bin), "qdrant")
    if extracted != dest_bin:
        shutil.move(extracted, dest_bin)
    os.chmod(dest_bin, 0o755)
    # Verify it actually runs (gnu asset fails here on old glibc)
    result = subprocess.run([dest_bin, "--version"], capture_output=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(
            f"Downloaded Qdrant binary failed to execute:\n"
            + result.stderr.decode()[-600:]
        )
    print(f"  Qdrant binary OK: {result.stdout.decode().strip()}")


def start_qdrant(qdrant_bin, storage_dir):
    """Start Qdrant binary if not already running on :6333.

    On Linux, auto-downloads the musl static binary if not present.
    On Windows, expects qdrant.exe at qdrant_bin.
    """
    if qdrant_alive():
        print("  Qdrant already running on :6333")
        return True

    if not os.path.exists(qdrant_bin):
        if platform.system() == "Linux":
            print(f"  Qdrant binary not found — downloading Linux (musl) binary ...")
            try:
                _download_qdrant_linux(qdrant_bin)
            except Exception as e:
                print(f"  ERROR: Could not download Qdrant: {e}")
                return False
        else:
            print(f"  Qdrant binary not found: {qdrant_bin}")
            print("  Download: https://github.com/qdrant/qdrant/releases/latest")
            print("  Place qdrant.exe in scripts/ or pass --qdrant-bin <path>")
            return False

    # Check port already bound by some other process
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_use = sock.connect_ex(("127.0.0.1", 6333)) == 0
    sock.close()
    if in_use:
        print("  Port 6333 already bound by a non-Qdrant process. Please free it first.")
        return False

    os.makedirs(storage_dir, exist_ok=True)
    bin_dir  = os.path.dirname(os.path.abspath(qdrant_bin))
    log_path = os.path.join(bin_dir, "qdrant.log")
    env = dict(
        os.environ,
        QDRANT__STORAGE__STORAGE_PATH=storage_dir,
        QDRANT__TELEMETRY_DISABLED="true",
    )
    
    if platform.system() == "Linux" and os.path.exists(qdrant_bin):
        os.chmod(qdrant_bin, 0o755)
        
    print(f"  Starting Qdrant (log: {log_path}) ...")
    subprocess.Popen(
        [qdrant_bin],
        env=env,
        cwd=bin_dir,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for i in range(120):
        if qdrant_alive():
            print("  Qdrant ready!")
            return True
        time.sleep(1)
        if i % 15 == 0 and i > 0:
            print(f"    Still waiting for Qdrant ... ({i}s)")
    print("  Timeout waiting for Qdrant.")
    return False


def create_collection(client, embed_dim, force_fresh):
    """Create Qdrant collection with HNSW + payload indexes, or resume."""
    from qdrant_client.models import (
        Distance, HnswConfigDiff, OptimizersConfigDiff, VectorParams,
    )
    exists = client.collection_exists(COLLECTION_NAME)
    if force_fresh and exists:
        print("  Deleting existing collection for a clean rebuild ...")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE, on_disk=True),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
            # Indexing disabled during bulk upload to avoid incremental-index OOM
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
        for field in ["source_type", "source", "language", "doc_category",
                      "crop", "district", "season", "query_type"]:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema="keyword")
        client.create_payload_index(COLLECTION_NAME, field_name="year",      field_schema="integer")
        client.create_payload_index(COLLECTION_NAME, field_name="has_table", field_schema="bool")
        print(f"  Collection '{COLLECTION_NAME}' created (HNSW indexing deferred)")
    else:
        print(f"  Collection '{COLLECTION_NAME}' exists — resuming")


def embed_and_upsert(client, unified, out_dir, emb_cache_dir=None):
    """Embed corpus shard-by-shard and upsert into Qdrant.

    emb_cache_dir : if set, each shard's vectors are saved as float16 .npy
                    so the GPU stage can be skipped on a resumed run.
                    Mirrors the EMB_CACHE_DIR pattern from notebook 08b.
    Returns embed_dim.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from qdrant_client.models import PointStruct

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus >= 2:
        target_devices = [f"cuda:{i}" for i in range(n_gpus)]
        # Scale batch size up — each process gets its own GPU (16 GB T4),
        # 128/GPU is safe for bge-m3 at 512 tokens; 64 was over-conservative.
        batch_size = 128
        print(f"  Embedding devices: {n_gpus}x GPU ({', '.join(target_devices)})")
        print(f"  Using encode_multi_process — batch_size={batch_size} per GPU")
    elif n_gpus == 1:
        target_devices = ["cuda:0"]
        batch_size = 96   # single T4 16 GB: 96 is comfortable for bge-m3 @ 512 tok
        print(f"  Embedding device: cuda:0 (single GPU) — batch_size={batch_size}")
    else:
        target_devices = ["cpu"]
        batch_size = EMBED_BATCH_SIZE
        print(f"  Embedding device: CPU (no GPU found) — batch_size={batch_size}")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=target_devices[0])
    embed_model.max_seq_length = MODEL_MAX_TOKENS
    embed_dim = embed_model.get_sentence_embedding_dimension()
    print(f"  Model: {EMBED_MODEL_NAME} | dim={embed_dim}")

    if emb_cache_dir:
        os.makedirs(emb_cache_dir, exist_ok=True)
        cached = len([f for f in os.listdir(emb_cache_dir) if f.endswith(".npy")])
        print(f"  Embedding cache: {emb_cache_dir}  ({cached} shards cached)")

    progress_path = os.path.join(out_dir, "shard_progress.json")
    n_shards = (len(unified) + SHARD_SIZE - 1) // SHARD_SIZE

    done = set()
    if os.path.exists(progress_path):
        done = set(json.load(open(progress_path)))
        print(f"  Resuming: {len(done)}/{n_shards} shards already indexed")

    done_chunks = sum(
        min((k + 1) * SHARD_SIZE, len(unified)) - k * SHARD_SIZE for k in done
    )
    pbar = tqdm(total=len(unified), initial=done_chunks, desc="Indexing", unit="chunk")

    # sentence-transformers >= 3.x merged encode_multi_process into encode().
    # Passing a list of devices distributes batches across all of them automatically.
    # No pool setup/teardown needed — eliminates the semaphore leak warning.
    use_pool = len(target_devices) > 1
    encode_devices = target_devices if use_pool else target_devices[0]

    try:
        for s in range(n_shards):
            if s in done:
                continue
            lo = s * SHARD_SIZE
            hi = min((s + 1) * SHARD_SIZE, len(unified))
            shard = unified[lo:hi]

            cache_f = os.path.join(emb_cache_dir, f"emb_{s:05d}.npy") if emb_cache_dir else None

            if cache_f and os.path.exists(cache_f):
                # Reload from cache — GPU stage already paid for this shard
                vecs = np.load(cache_f).astype(np.float32)
                if len(vecs) != len(shard):
                    raise RuntimeError(
                        f"Cache {cache_f} has {len(vecs)} rows but shard needs {len(shard)}. "
                        "Delete the cache dir and rebuild."
                    )
                pbar.set_postfix_str(f"shard {s + 1}/{n_shards} (cached)")
            else:
                pbar.set_postfix_str(f"shard {s + 1}/{n_shards} embedding")
                texts = [DOC_PREFIX + r["text"] for r in shard]
                # encode() accepts a device string or list of device strings;
                # when a list is passed sentence-transformers >= 3.x distributes
                # batches across all devices automatically (no pool needed).
                vecs = embed_model.encode(
                    texts,
                    device=encode_devices,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
                if cache_f:
                    # float16 halves storage; cosine is insensitive at this precision
                    # (vectors are L2-normalised before saving)
                    np.save(cache_f, vecs.astype(np.float16))

            pbar.set_postfix_str(f"shard {s + 1}/{n_shards} upserting")
            for b in range(0, len(shard), UPSERT_BATCH):
                pts = [
                    PointStruct(
                        id=r["chunk_id"],
                        vector=vecs[b + j].tolist(),
                        payload={k: v for k, v in r.items() if v is not None},
                    )
                    for j, r in enumerate(shard[b: b + UPSERT_BATCH])
                ]
                client.upsert(collection_name=COLLECTION_NAME, points=pts)

            done.add(s)
            json.dump(sorted(done), open(progress_path, "w"))
            pbar.update(hi - lo)
            del vecs, shard, texts
            gc.collect()
    finally:
        pbar.close()

    return embed_dim


def enable_hnsw_and_wait(client):
    """Turn on HNSW indexing and poll until GREEN."""
    from qdrant_client.models import OptimizersConfigDiff
    print("  Enabling HNSW indexing ...")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=100),
    )
    deadline = time.time() + 7200
    t0 = time.time()
    while time.time() < deadline:
        info = client.get_collection(COLLECTION_NAME)
        status = str(info.status).split(".")[-1].lower()
        if status == "green":
            print(f"  HNSW built in {(time.time() - t0) / 60:.1f} min")
            return info
        print(f"    indexing ... {info.indexed_vectors_count:,}/{info.points_count:,}"
              f"  ({(time.time() - t0) / 60:.1f} min)")
        time.sleep(30)
    print("  [WARN] HNSW still building after 2 h. Search works but is slower.")
    return client.get_collection(COLLECTION_NAME)


# ---------------------------------------------------------------------------
# ARTIFACT EXPORT
# ---------------------------------------------------------------------------

def export_artifacts(out_dir, embed_dim, n_chunks):
    """Download Qdrant snapshot and write manifest.json to out_dir."""
    print(f"\n{'=' * 70}")
    print("EXPORTING ARTIFACTS")
    print(f"{'=' * 70}")

    # Reuse existing snapshot or trigger a new one
    snaps = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots", timeout=120
    ).json()["result"]
    if not snaps:
        print("  Creating snapshot ...")
        r = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots", timeout=3600
        )
        r.raise_for_status()
        snaps = [r.json()["result"]]

    # Name embeds the timestamp; sort lexicographically to get the latest
    snap = sorted(snaps, key=lambda s: s["name"])[-1]
    snap_name = snap["name"]
    snap_size_gb = (snap.get("size") or 0) / 1e9
    print(f"  Snapshot : {snap_name}  ({snap_size_gb:.2f} GB)")

    snap_out = os.path.join(out_dir, snap_name)
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/{snap_name}"
    t0, done_bytes, next_mark = time.time(), 0, 512 << 20
    print(f"  Downloading to {snap_out} ...")
    with requests.get(url, stream=True, timeout=7200) as r:
        r.raise_for_status()
        with open(snap_out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done_bytes += len(chunk)
                if done_bytes >= next_mark:
                    print(f"    {done_bytes / 1e9:.2f} GB  ({(time.time() - t0) / 60:.1f} min)")
                    next_mark += 512 << 20
    print(f"  Snapshot saved: {done_bytes / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")

    if done_bytes < 1_000_000:
        raise RuntimeError(f"Snapshot only {done_bytes} bytes — download failed.")

    # Rename to a fixed, predictable filename
    fixed_snap = os.path.join(out_dir, "agri_knowledge.snapshot")
    if snap_out != fixed_snap:
        shutil.move(snap_out, fixed_snap)
        print("  Renamed -> agri_knowledge.snapshot")

    # Manifest: query-side contract (model, prefixes, tiers, weights)
    manifest = {
        "collection":      COLLECTION_NAME,
        "snapshot":        "agri_knowledge.snapshot",
        "embed_model":     EMBED_MODEL_NAME,
        "model_tag":       MODEL_TAG,
        "embed_dim":       embed_dim,
        "max_seq_length":  MODEL_MAX_TOKENS,
        "query_prefix":    QUERY_PREFIX,
        "doc_prefix":      DOC_PREFIX,
        "distance":        "COSINE (on L2-normalised vectors)",
        "hnsw":            {"m": 16, "ef_construct": 128},
        "n_chunks":        n_chunks,
        "tiers":           {"fallback": TIER_FALLBACK, "grounded": TIER_GROUNDED},
        "fusion_weights":  FUSION_WEIGHTS,
        "top_k_default":   TOP_K_DEFAULT,
        "built_utc":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_out = os.path.join(out_dir, "manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest  -> {manifest_out}")

    return fixed_snap, manifest_out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Qdrant RAG artifacts (snapshot + manifest.json) from .jsonl chunk files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--kcc", metavar="PATH",
                        help="KCC chunks .jsonl (or .jsonl.gz)")
    parser.add_argument("--pdf", metavar="PATH",
                        help="PDF chunks .jsonl (or .jsonl.gz)")
    parser.add_argument("--out", metavar="DIR", default="rag_production_bge_m3",
                        help="Output dir for snapshot + manifest (default: rag_production_bge_m3)")
    # On Linux the binary is auto-downloaded; on Windows default to qdrant.exe
    _default_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "qdrant" if platform.system() == "Linux" else "qdrant.exe",
    )
    parser.add_argument("--qdrant-bin", metavar="PATH", default=_default_bin,
                        help="Path to Qdrant binary (auto-downloaded on Linux if missing)")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing Qdrant collection and rebuild from scratch")
    parser.add_argument("--emb-cache", metavar="DIR", default=None,
                        help="Directory for per-shard float16 .npy embedding cache "
                             "(strongly recommended on Kaggle/Colab to survive session crashes)")
    parser.add_argument("--kcc-max", metavar="N", type=int, default=None,
                        help="Max KCC chunks via crop-stratified sampling (None = full corpus)")
    args = parser.parse_args()

    if not args.kcc and not args.pdf:
        parser.error("Provide at least one of --kcc or --pdf")

    out_dir     = os.path.abspath(args.out)
    storage_dir = os.path.join(out_dir, "qdrant_storage")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("RAG ARTIFACT BUILDER")
    print("=" * 70)
    print(f"  KCC chunks : {args.kcc or '(not provided)'}")
    print(f"  PDF chunks : {args.pdf or '(not provided)'}")
    print(f"  Output dir : {out_dir}")
    print(f"  Embedder   : {EMBED_MODEL_NAME}")
    print(f"  Fresh build: {args.fresh}")
    print(f"  Emb cache  : {args.emb_cache or '(disabled)'}")
    print(f"  KCC max    : {args.kcc_max or 'full corpus'}")

    # ------------------------------------------------------------------
    # STEP 1: Load and normalise chunks
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 1: LOADING + NORMALISING CHUNKS")
    print(f"{'=' * 70}")

    from transformers import AutoTokenizer
    print(f"  Loading tokenizer ({EMBED_MODEL_NAME}) for token-budget enforcement ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    budget_hard = MODEL_MAX_TOKENS - 2

    unified = []

    if args.pdf:
        if not os.path.exists(args.pdf) and not os.path.exists(args.pdf + ".gz"):
            print(f"  ERROR: PDF file not found: {args.pdf}")
            sys.exit(1)
        unified.extend(load_pdf_chunks(args.pdf))

    n_pdf = len(unified)

    if args.kcc:
        if not os.path.exists(args.kcc) and not os.path.exists(args.kcc + ".gz"):
            print(f"  ERROR: KCC file not found: {args.kcc}")
            sys.exit(1)
        kcc_rows = load_kcc_chunks(args.kcc, tokenizer, budget_hard)

        # Crop-stratified subsample (mirrors notebook 08b config cell KCC_MAX_CHUNKS)
        if args.kcc_max and len(kcc_rows) > args.kcc_max:
            random.seed(42)
            by_crop = defaultdict(list)
            for r in kcc_rows:
                by_crop[r["crop"] or "_none"].append(r)
            frac = args.kcc_max / len(kcc_rows)
            sampled = []
            for crop, rows in by_crop.items():
                k = max(1, round(len(rows) * frac))
                sampled.extend(random.sample(rows, min(k, len(rows))))
            kcc_rows = sampled[: args.kcc_max]
            print(f"  Stratified subsample: {len(kcc_rows):,} chunks "
                  f"across {len(by_crop)} crops (--kcc-max={args.kcc_max:,})")

        unified.extend(kcc_rows)

    n_kcc = len(unified) - n_pdf
    print(f"\n  Unified corpus: {len(unified):,} chunks  (pdf={n_pdf:,}, kcc={n_kcc:,})")

    # Uniqueness check
    ids, dups = set(), 0
    for u in unified:
        if u["chunk_id"] in ids:
            dups += 1
        else:
            ids.add(u["chunk_id"])
    if dups:
        print(f"  WARNING: {dups} duplicate chunk_ids — check your input files")
    else:
        print(f"  chunk_id uniqueness OK ({len(ids):,} unique)")
    del ids

    # ------------------------------------------------------------------
    # STEP 2: Start Qdrant
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 2: STARTING QDRANT")
    print(f"{'=' * 70}")

    if not start_qdrant(args.qdrant_bin, storage_dir):
        print("ERROR: Could not start Qdrant.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3: Collection setup
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 3: COLLECTION SETUP")
    print(f"{'=' * 70}")

    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, timeout=600)
    create_collection(client, embed_dim=EMBED_DIM_EXPECTED, force_fresh=args.fresh)

    # ------------------------------------------------------------------
    # STEP 4: Embed + Upsert
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 4: EMBEDDING + UPSERTING (sharded, resumable)")
    print(f"{'=' * 70}")

    embed_dim = embed_and_upsert(client, unified, out_dir, emb_cache_dir=args.emb_cache)

    # ------------------------------------------------------------------
    # STEP 5: Enable HNSW and wait
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("STEP 5: HNSW INDEXING")
    print(f"{'=' * 70}")

    info = enable_hnsw_and_wait(client)
    print(f"\n  points:       {info.points_count:,}")
    print(f"  indexed vecs: {info.indexed_vectors_count:,}")
    print(f"  status:       {info.status}")

    # ------------------------------------------------------------------
    # STEP 6: Export artifacts
    # ------------------------------------------------------------------
    snap_path, manifest_path = export_artifacts(out_dir, embed_dim, len(unified))

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("BUILD COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Snapshot : {snap_path}")
    print(f"  Manifest : {manifest_path}")
    print()
    print("  Next steps:")
    print("  1. Upload both files to Google Drive")
    print("  2. Get the file IDs from the share links")
    print("  3. Update SNAPSHOT_ID and MANIFEST_ID in scripts/setup_vectordb.py")
    print("  4. Run: python scripts/setup_vectordb.py")
    print()
    print("  Output files:")
    for fn in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            print(f"    {os.path.getsize(fp) / 1e6:>10,.1f} MB  {fn}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
