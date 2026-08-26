"""Central config. All the query-side numbers come from the manifest.json that
was written when the index was built, so they always match the index."""
import json
import os

# --- environment (set by docker-compose) -----------------------------------
QDRANT_URL   = os.environ.get("QDRANT_URL", "http://localhost:6333")
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "/artifacts/qdrant/manifest.json")
API_KEY      = os.environ.get("API_KEY", "")

GEN_MODEL_ID = os.environ.get("GEN_MODEL_ID", "google/gemma-3-4b-it")
# Preferred: a model with the LoRA adapter already MERGED in (fast — no adapter
# overhead at inference). Falls back to base + ADAPTER_DIR if the merged dir is absent.
MERGED_MODEL_DIR = os.environ.get("MERGED_MODEL_DIR", "/artifacts/generator/merged")
ADAPTER_DIR  = os.environ.get("ADAPTER_DIR", "/artifacts/generator/best_adapter")
IEG_DIR      = os.environ.get("IEG_DIR", "/artifacts/ieg")
VISION_CKPT  = os.environ.get("VISION_CKPT", "/artifacts/vision/p3_full_best.pt")
VISION_LABELS = os.environ.get("VISION_LABELS", "/artifacts/vision/label_to_idx.json")
HF_TOKEN     = os.environ.get("HF_TOKEN")

# NOTE: mandi prices, weather, and yield prediction are handled by the MAIN APP
# and injected into /query via `live_data`. This gateway does not serve them.

# --- load the manifest (the query-side contract) ----------------------------
with open(MANIFEST_PATH, encoding="utf-8") as f:
    MANIFEST = json.load(f)

COLLECTION      = MANIFEST["collection"]                    # "agri_knowledge"
EMBED_MODEL     = MANIFEST["embed_model"]                   # "BAAI/bge-m3"
EMBED_DIM       = MANIFEST["embed_dim"]                     # 1024
MAX_SEQ_LENGTH  = MANIFEST.get("max_seq_length", 512)
QUERY_PREFIX    = MANIFEST.get("query_prefix", "")          # "" for bge-m3
DOC_PREFIX      = MANIFEST.get("doc_prefix", "")            # "" for bge-m3
TIER_FALLBACK   = MANIFEST["tiers"]["fallback"]            # 0.553
TIER_GROUNDED   = MANIFEST["tiers"]["grounded"]            # 0.638
FUSION_WEIGHTS  = MANIFEST["fusion_weights"]               # per-intent pdf/kcc weights
TOP_K_DEFAULT   = MANIFEST.get("top_k_default", 5)

# --- experimental improvements (backported from run_e2e_eval.py) ------------
# Confidence-gated filter loosening: below this top-1 softmax confidence,
# the KCC query_type sub-filter is DROPPED (source-level filters kept).
# Rationale: holdout calibration shows top-1 accuracy collapses below ~0.6 conf,
# so a strict qtype filter there mostly excludes the right chunks. The reranker
# recovers precision, it cannot recover filtered-out chunks.
# Tunable via env var for production A/B testing (0.0 = never loosen, 1.0 = always loosen)
CONF_GATE_LOOSEN = float(os.environ.get("CONF_GATE_LOOSEN", "0.60"))

# Retrieval filtering thresholds (from run_e2e_eval.py lines 68-70)
MIN_CHUNK_SCORE = float(os.environ.get("MIN_CHUNK_SCORE", "0.50"))  # Filter out low-scoring chunks
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "50"))      # Minimum chunk length to avoid OCR noise
MAX_PER_SOURCE  = int(os.environ.get("MAX_PER_SOURCE", "3"))        # Cap chunks from same source for diversity
TOP_K_BASE      = int(os.environ.get("TOP_K_BASE", "10"))           # Base value for dynamic TOP_K scaling
RERANK_TOP_N    = int(os.environ.get("RERANK_TOP_N", "5"))          # Keep top-N after cross-encoder reranking

# Which optional modules are present on disk
HAS_IEG    = os.path.isdir(IEG_DIR) and os.path.isfile(os.path.join(IEG_DIR, "intent_entity_guardrail_model.pt"))
HAS_VISION = os.path.isfile(VISION_CKPT) and os.path.isfile(VISION_LABELS)
