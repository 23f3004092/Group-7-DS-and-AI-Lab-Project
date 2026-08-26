# %%writefile run_e2e_eval.py 
"""
run_e2e_eval.py
===============
End-to-end pipeline evaluation for FarmerVision.
Runs LOCALLY — Qdrant must be running (scripts/setup_vectordb.py).
All model artefacts are auto-downloaded via kagglehub on first run.

Three pathways:
  A: text query -> IEG guardrail -> bge-m3 Qdrant retrieval -> gemma-3-4b-it -> answer
  B: image path -> ViT-S/16 CropDiseaseModel -> disease label -> Qdrant -> gemma -> answer
  C: crop+district+area -> LightGBM (lightgbm_tuned.txt) -> yield t/ha

Usage:
  .venv/Scripts/python.exe scripts/run_e2e_eval.py
  .venv/Scripts/python.exe scripts/run_e2e_eval.py --skip-gen  # skip slow generator
  .venv/Scripts/python.exe scripts/run_e2e_eval.py --pathway A  # single pathway

Outputs:
  data/eval/e2e_results.csv
  data/eval/e2e_summary.json
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# Lazy CrossEncoder import (only loaded when needed)
_reranker = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N   = 5   # keep top-N after reranking

def _get_reranker():
    """Lazy-load the cross-encoder reranker (downloaded once, cached in _reranker)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_CSV = ROOT / "data" / "eval" / "e2e_scenarios.csv"
RESULTS_CSV   = ROOT / "data" / "eval" / "e2e_results.csv"
SUMMARY_JSON  = ROOT / "data" / "eval" / "e2e_summary.json"


def find_in_kaggle_input(*patterns):
    """Glob /kaggle/input/** for an artifact BEFORE any network download
    (kagglehub / gdown / HF hub). Returns the first match as a Path, or None.
    On Kaggle kernels attached datasets are mounted read-only at /kaggle/input,
    so this is a zero-download fast path."""
    if not os.path.isdir("/kaggle/input"):
        return None
    for pat in patterns:
        cands = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
        if cands:
            print(f"  [kaggle-input] Found {pat} -> {cands[0]}")
            return Path(cands[0])
    return None

QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "agri_knowledge"
BGE_MODEL_ID    = "BAAI/bge-m3"
IEG_MODEL_ID    = "l3cube-pune/hing-mbert-mixed"
MAX_LEN         = 64
TOP_K           = 10
TOP_K_BASE      = 10  # Base value for dynamic scaling
TIER_GROUNDED   = 0.66
TIER_FALLBACK   = 0.56
MIN_CHUNK_SCORE = 0.50  # Filter out low-scoring chunks
MIN_CHUNK_CHARS = 50    # Minimum chunk length to avoid OCR noise
MAX_PER_SOURCE  = 3     # Cap chunks from same source for diversity
# Confidence-gated filter loosening (Fix #1):
# Below this top-1 softmax confidence, the KCC query_type sub-filter is DROPPED
# (source-level filters kept). Rationale: holdout calibration shows top-1
# accuracy collapses below ~0.6 conf, so a strict qtype filter there mostly
# excludes the right chunks. The reranker recovers precision, it cannot
# recover filtered-out chunks. 0 ms added latency — conf is already computed.
CONF_GATE_LOOSEN = 0.60
# Runtime-overridable via --gate-threshold (A/B testing).
#   0.0 = never loosen (gate off)   1.0 = always loosen
GATE_THRESHOLD = CONF_GATE_LOOSEN
# Small models (IEG / bge-m3 / reranker) go to CPU by default to prevent VRAM
# overflow alongside 4-bit gemma. Set EVAL_DEVICE=auto (or "cuda") to place them
# on GPU — on a T4 with gemma-3-4b 4-bit (~3 GB) there is ample headroom; this is
# what the production GCP app does (app/retrieval.py keeps bge-m3 in fp32-on-CUDA
# to preserve calibrated similarity scores).
DEVICE = os.environ.get("EVAL_DEVICE", "cpu")
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# External judge (Qwen3-8B on vLLM/TGI) endpoint for completeness/citation judging.
# Set via --judge-url; if unset, falls back to local gemma self-judge.
JUDGE_URL = None

# Intent to source routing map for payload filtering
# Note: KCC chunks use "kcc_qa" as their source, not "kcc"
INTENT_SOURCE_MAP = {
    "disease_pest":         ["ppqs_advisories", "kcc_qa"],
    "cultivation_practice": ["up_acp", "kcc_qa"],
    "nutrition_fertilizer": ["ppqs_advisories", "kcc_qa"],
    "yield_estimation":     ["kcc_qa"],
    "post_harvest_storage": ["kcc_qa"],
    "specialty_other":      ["kcc_qa"],
    "policy":               ["schemes"],   # UP govt scheme circulars live here
    "general":              None,   # no filter → search all
}

# Reverse mapping from IEG intent back to raw KCC query_type.
# This prevents us from searching 716k KCC chunks when we only care about a specific intent.
INTENT_QTYPE_MAP = {
    "disease_pest": [
        "Plant Protection", "Weed Management", "Insect Management", "Pathogenic Disease Management"
    ],
    "nutrition_fertilizer": [
        "Nutrient Management", "Fertilizer Use and Availability", 
        "Nutrient Deficiency/Excessiveness Management", "Bio-Pesticides and Bio-Fertilizers"
    ],
    "cultivation_practice": [
        "Cultural Practices", "Varieties", "Varietal Selection", "Seeds and Planting Material",
        "Seeds", "Seed Sowing And Treatment", "Field Preparation", "Water Management",
        "Water Management, Micro Irrigation", "Irrigation Management", "Soil Testing",
        "Abiotic Stress Management"
    ],
    "post_harvest_storage": [
        "Post Harvest Preservation", "Storage", "Cold Storage"
    ],
    "specialty_other": [
        "Organic Farming", "Floriculture", "Beekeeping"
    ]
}

print(f"Device: {DEVICE}")

# ── Yield model constants (from actual training data) ─────────────────────────
YIELD_CAT_COLS   = ["crop", "state", "district", "season", "data_source", "crop_type"]
YIELD_NUM_COLS   = ["year", "area", "annual_rainfall", "fertilizer", "pesticide"]
YIELD_ALL_FEATS  = YIELD_CAT_COLS + ["year", "area", "annual_rainfall", "fertilizer", "pesticide"]

# Typical UP values for inputs we don't have from the user query
# (median from the cleaned training data for UP rows)
UP_DEFAULTS = {
    "state":           "uttar pradesh",
    "data_source":     "area_production",
    "annual_rainfall": 850.0,   # UP average mm
    "fertilizer":      1_200_000.0,  # district total kg (median UP value)
    "pesticide":       2_400.0,      # district total kg (median UP value)
    "year":            2023,
}
CROP_TO_TYPE = {"wheat": "cereals", "rice": "cereals", "paddy": "cereals"}
CROP_TO_SEASON = {"wheat": "Rabi", "rice": "Kharif", "paddy": "Kharif"}


# ── IEG model (same architecture as notebook cell 5) ─────────────────────────

class IEGModel(nn.Module):
    def __init__(self, n_intents: int, n_ner: int):
        from transformers import AutoModel
        super().__init__()
        self.backbone       = AutoModel.from_pretrained(IEG_MODEL_ID)
        h                   = self.backbone.config.hidden_size
        self.intent_head    = nn.Linear(h, n_intents)
        self.ner_head       = nn.Linear(h, n_ner)
        self.guardrail_head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 0, :]
        return self.intent_head(pooled), self.ner_head(out), self.guardrail_head(pooled)


# ── Guardrail keyword rules (from notebook cell 7) ───────────────────────────

BANNED_TERMS = [
    "monocrotophos", "endosulfan", "phorate",
    "methyl parathion", "ddt", "aldrin", "chlorpyrifos",
]
DOSAGE_PATTERNS = [
    r"\b(\d+)\s*(x|times)\b.*\b(dose|dosage|dawa|spray)\b",
    r"\b(double|triple|4x|5x|10x)\b.*\b(dose|dosage|strength|dawa)\b",
    r"\bz?jyada\b.*\bdawa\b",
    r"\b(\d{3,})\s*kg\b.*\b(per acre|prati ekad|ekad)\b",
]

def rule_flag(text: str) -> bool:
    t = str(text).lower()
    if any(term in t for term in BANNED_TERMS):
        return True
    return any(re.search(p, t) for p in DOSAGE_PATTERNS)


# ── Post-generation checks ────────────────────────────────────────────────────

def check_citation(answer: str) -> bool:
    return bool(re.search(r"source[s]?\s*[:\[]|doc\s*[:\[]|\[\d+\]", answer, re.IGNORECASE))


def check_lang_match(query_lang: str, answer: str) -> bool:
    query_lang = str(query_lang).strip().lower()
    if query_lang in ("n/a", "english", "hinglish"):
        return True  # flexible for these
    # Hindi query -> expect some Devanagari in the answer
    dev_ratio = sum(1 for c in answer if "\u0900" <= c <= "\u097F") / max(len(answer), 1)
    return dev_ratio > 0.15


def check_numeric_grounding(answer: str, chunks: list) -> bool:
    """Fixed version (M4 App. E bug corrected): normalise thousands separators."""
    def norm(s):
        return re.sub(r"[,\s]", "", s)
    answer_nums = {norm(m) for m in re.findall(r"\d[\d,\.]*", answer)}
    # Ignore small integers (list markers 1, 2, 3…)
    answer_nums = {n for n in answer_nums if len(re.sub(r"\D", "", n)) > 1}
    if not answer_nums:
        return True
    ctx = " ".join(chunks)
    ctx_nums = {norm(m) for m in re.findall(r"\d[\d,\.]*", ctx)}
    return answer_nums.issubset(ctx_nums)


def _call_judge(judge_url: str, topics: list, answer: str) -> bool:
    """Call external judge (e.g., Qwen3-8B on vLLM) for completeness.
    Returns True if verdict is 'yes'. Times out after 30s, falls back to local."""
    import requests
    payload = {"topics": topics, "answer": answer}
    try:
        r = requests.post(judge_url.rstrip("/") + "/judge", json=payload, timeout=30)
        r.raise_for_status()
        d = r.json()
        return str(d.get("verdict", "no")).strip().lower() == "yes"
    except Exception as e:
        print(f"  [judge] request failed ({e}); falling back to local model")
        return False


def check_completeness(answer: str, topics: list, gen_tok=None, gen_mdl=None, judge_url: str = None) -> bool:
    if judge_url:
        return _call_judge(judge_url, topics, answer)

    if not topics:
        return True
    
    # Fallback: local gemma self-judge (kept for backward compatibility)
    if gen_tok is None or gen_mdl is None:
        return False
    
    readable_topics = [t.replace("_", " ").title() for t in topics]
    
    prompt = (
        "You are an evaluator grading an AI's response.\n"
        f"Goal: Does the answer address all of these topics: {readable_topics}?\n"
        "Example:\n"
        "Topics: ['Disease Pest', 'Market Price']\n"
        "Answer: The leaf shows rust. You should spray fungicide. I cannot help with market prices.\n"
        "Decision: yes\n\n"
        f"Topics: {readable_topics}\n"
        f"Answer: {answer}\n"
        "Decision:"
    )
    inputs = gen_tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        out = gen_mdl.generate(**inputs, max_new_tokens=5, temperature=0.1, do_sample=False)
    res = gen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    return "yes" in res


# ── Component loaders ─────────────────────────────────────────────────────────

def load_ieg():
    """Load experimental IEG model from Kaggle kernel output."""
    from transformers import AutoTokenizer
    print("  Loading experimental IEG checkpoint from outputs/ieg_final…")
    
    ckpt_candidates = []
    label_candidates = []

    # 1. Search in Kaggle input first
    hit = find_in_kaggle_input("**/ieg_adamw*.pt", "**/ieg_model/**/*.pt")
    if hit:
        ckpt_candidates = [hit]
        label_candidates = list(hit.parent.rglob("label_maps.json"))
        print(f"  IEG dataset found in Kaggle input: {hit.parent}")
    else:
        # 2. Fallback to local outputs directory
        dataset_path = ROOT / "outputs" / "ieg_final_latest" / "outputs" / "ieg_model"
        if not dataset_path.exists():
            dataset_path = ROOT / "outputs" / "ieg_final_latest"
        
        print(f"  IEG dataset at: {dataset_path}")
        ckpt_candidates = list(dataset_path.rglob("*.pt"))
        label_candidates = list(dataset_path.rglob("label_maps.json"))

    if not ckpt_candidates:
        raise FileNotFoundError("No .pt checkpoint found for IEG model.")
    
    ckpt = ckpt_candidates[0]
    print(f"  Using checkpoint: {ckpt}")
    
    if label_candidates:
        label_maps_path = label_candidates[0]
        print(f"  Using label maps: {label_maps_path}")
        with open(label_maps_path) as f:
            label_maps = json.load(f)
    else:
        label_maps = {}

    intent_classes = label_maps.get("intent_classes", [
        "cultivation_practice", "disease_pest", "general", "non_agri",
        "nutrition_fertilizer", "post_harvest_storage", "specialty_other", "yield_estimation"
    ])
    ner_labels = label_maps.get("ner_labels",
                                ["O", "B-CROP", "I-CROP", "B-DISTRICT", "I-DISTRICT"])

    model = IEGModel(n_intents=len(intent_classes), n_ner=len(ner_labels))
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    tok      = AutoTokenizer.from_pretrained(IEG_MODEL_ID)
    id2intent = {i: c for i, c in enumerate(intent_classes)}
    print(f"  IEG loaded. {len(intent_classes)} intents: {intent_classes}")
    return model, tok, id2intent, ner_labels


def load_vit():
    """Load ViT CropDiseaseModel from Kaggle dataset."""
    print("  Loading ViT model …")
    # Kaggle-input fast path: look for the checkpoint (with its predict.py) locally first
    ckpt_hit = find_in_kaggle_input("**/p3_full_best.pt")
    if ckpt_hit:
        model_dir = ckpt_hit.parent.parent          # .../weights/p3_full_best.pt
        if not (model_dir / "predict.py").exists():
            pred = find_in_kaggle_input("**/predict.py")
            if pred is None:
                raise FileNotFoundError("ViT predict.py not found in /kaggle/input")
            model_dir = pred.parent
    else:
        import kagglehub
        print("  Loading ViT via kagglehub …")
        path = Path(kagglehub.dataset_download("iitm21f1003346/vits16-crop-disease"))
        print(f"  ViT dataset at: {path}")
        model_dir = next(path.rglob("predict.py")).parent

    sys.path.insert(0, str(model_dir))
    from predict import CropDiseaseModel

    ckpt_path = model_dir / "weights" / "p3_full_best.pt"
    print(f"  Using checkpoint: {ckpt_path}")
    
    model = CropDiseaseModel(ckpt=ckpt_path, device=DEVICE)
    print(f"  ViT loaded. Classes: {model.classes[:4]} …")
    return model


def load_retriever():
    """Connect to Qdrant and load bge-m3 embedder."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    info   = client.get_collection(COLLECTION_NAME)
    print(f"  Qdrant OK. Points: {info.points_count:,}")
    embedder = SentenceTransformer(BGE_MODEL_ID, device="cpu")
    print(f"  bge-m3 embedder loaded")
    return client, embedder


def load_generator():
    """Load gemma-3-4b-it in 4-bit. Falls back to CPU if no GPU."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    print("  Loading gemma-3-4b-it (4-bit) …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    base_mdl = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        quantization_config=bnb_config,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    from peft import PeftModel
    import os
    import gdown
    import shutil
    import inspect

    adapter_path = ROOT / "outputs" / "generation" / "best_adapter"
    if not (adapter_path / "adapter_model.safetensors").exists():
        # Kaggle-input fast path before downloading from Google Drive
        khit = find_in_kaggle_input("**/adapter_model.safetensors")
        if khit:
            print(f"  Using LoRA adapter from Kaggle input: {khit.parent}")
            mdl = PeftModel.from_pretrained(base_mdl, str(khit.parent))
            mdl.eval()
            print("  Generator loaded")
            return tok, mdl
    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"  Downloading distilled adapter to {adapter_path} …")
        adapter_path.mkdir(parents=True, exist_ok=True)
        fid = "1nAbR9-hyba_vUhPE68Idet-oEK4cNA1W"
        _sig = inspect.signature(gdown.download_folder)
        _kwargs = dict(id=fid, output=str(adapter_path), quiet=False, use_cookies=False)
        if "remaining_ok" in _sig.parameters:
            _kwargs["remaining_ok"] = True
        try:
            gdown.download_folder(**_kwargs)
            hit = None
            for root, dirs, files in os.walk(str(adapter_path)):
                if "adapter_model.safetensors" in files:
                    hit = os.path.join(root, "adapter_model.safetensors")
                    break
            if hit and os.path.dirname(hit) != str(adapter_path):
                src = os.path.dirname(hit)
                for f in os.listdir(src):
                    shutil.move(os.path.join(src, f), os.path.join(str(adapter_path), f))
        except Exception as e:
            print(f"  [folder] download failed: {type(e).__name__}: {str(e)[:160]}")

    mdl = PeftModel.from_pretrained(base_mdl, str(adapter_path))
    mdl.eval()
    print("  Generator loaded")
    return tok, mdl


YIELD_KERNEL = "tanmay23f1/yield-v2-1-7"   # Kaggle kernel whose output holds the model
YIELD_KEEP_FILES = {"lightgbm_tuned.txt"}  # everything else in the kernel output is discarded


def _fetch_yield_kernel_output():
    """Fetch LightGBM model via `kaggle kernels output` into the working dir,
    then delete everything except YIELD_KEEP_FILES.

    Destination: /kaggle/working on Kaggle kernels, outputs/yield_kernel_output locally.
    Returns Path to lightgbm_tuned.txt or None."""
    import shutil
    import subprocess

    if shutil.which("kaggle") is None:
        print("  [yield] kaggle CLI not available — skipping kernel-output fetch.")
        return None

    dest = Path("/kaggle/working/yield_kernel_output") if os.path.isdir("/kaggle/working") \
        else ROOT / "outputs" / "yield_kernel_output"
    dest.mkdir(parents=True, exist_ok=True)

    # Already fetched previously? Don't re-download.
    existing = next(dest.rglob("lightgbm_tuned.txt"), None)
    if existing:
        print(f"  [yield] Kernel output already fetched: {existing}")
        return existing

    print(f"  [yield] Fetching kernel output {YIELD_KERNEL} -> {dest} …")
    try:
        subprocess.run(
            ["kaggle", "kernels", "output", YIELD_KERNEL, "-p", str(dest)],
            check=True, capture_output=True, text=True, timeout=900,
        )
    except Exception as e:
        err = getattr(e, "stderr", "") or str(e)
        print(f"  [yield] kernel-output fetch failed: {err[:200]}")
        return None

    # Delete everything except the files we need
    removed = kept = 0
    for f in dest.rglob("*"):
        if f.is_file() and f.name not in YIELD_KEEP_FILES:
            f.unlink()
            removed += 1
        elif f.is_file():
            kept += 1
    # Prune now-empty subdirectories (deepest first)
    for d in sorted(dest.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print(f"  [yield] Kernel output pruned: kept {kept}, deleted {removed} files.")

    hit = next(dest.rglob("lightgbm_tuned.txt"), None)
    return hit


def load_yield_model():
    """Load LightGBM tuned model from local outputs directory."""
    import lightgbm as lgb
    model_path = (ROOT / "outputs" / "Yeild_output_files" / "kaggle" /
                  "working" / "saved_models" / "lightgbm_tuned.txt")
    if not model_path.exists():
        # Kaggle-input fast path
        hit = find_in_kaggle_input("**/lightgbm_tuned.txt")
        if hit:
            model_path = hit
    if not model_path.exists():
        # Last resort: pull the model out of the Kaggle kernel output
        hit = _fetch_yield_kernel_output()
        if hit:
            model_path = hit
    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model not found at {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    print(f"  LightGBM loaded. {booster.num_trees()} trees, features: {booster.feature_name()}")
    return booster


# ── Inference helpers ─────────────────────────────────────────────────────────

MAX_QUERY_CHARS = 200   # compress queries longer than this before IEG

def compress_query(text: str) -> str:
    """Truncate long conversational prompts to first sentence / first 200 chars.
    Distribution shift fix: long first_user_prompts bury the intent signal.
    """
    if len(text) <= MAX_QUERY_CHARS:
        return text
    # Try to cut at first sentence boundary
    for sep in ("?", ".", "!", "|"):
        idx = text.find(sep)
        if 30 < idx <= MAX_QUERY_CHARS:
            return text[:idx + 1].strip()
    return text[:MAX_QUERY_CHARS].strip()


def ieg_run(model, tok, id2intent, ner_labels, text: str):
    compressed = compress_query(text)   # ← query compression
    enc = tok(compressed, truncation=True, max_length=MAX_LEN,
               padding="max_length", return_tensors="pt")
    # DistilBERT doesn't use token_type_ids
    enc = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        il, nl, gl = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    # Fix 2 (Option B): Use Softmax — consistent with CrossEntropyLoss training
    probs = torch.softmax(il[0], dim=-1).cpu()
    # Top-3 soft routing: always include top-3 + any above relative threshold (15% of total mass)
    top3_idx   = probs.topk(min(3, len(probs))).indices.tolist()
    thresh_idx = [i for i, p in enumerate(probs) if p > 0.15]
    intent_idx = list(dict.fromkeys(top3_idx + thresh_idx))  # top3 first, deduped
    intents    = [id2intent[i] for i in intent_idx]

    m_flag  = int(gl.argmax(-1).item()) == 1
    r_flag  = rule_flag(text)
    blocked = m_flag or r_flag
    top_conf = float(probs.max().item())
    return intents, blocked, m_flag, r_flag, top_conf


def retrieve(client, embedder, query: str, intents: list = None, loosen_qtypes: bool = False):
    from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
    
    # Dynamic TOP_K scaling with intent count
    active_top_k = max(TOP_K_BASE, len(intents) * 5) if intents and len(intents) > 2 else TOP_K
    
    # Intent → Source routing: build payload filter
    query_filter = None
    if intents and len(intents) > 0:
        sources = set()
        qtypes = set()
        allow_all_kcc = False
        
        for intent in intents:
            srcs = INTENT_SOURCE_MAP.get(intent)
            if srcs:
                sources.update(srcs)
                if "kcc_qa" in srcs:
                    if intent in INTENT_QTYPE_MAP:
                        qtypes.update(INTENT_QTYPE_MAP[intent])
                    else:
                        allow_all_kcc = True
                        
        if sources:
            should_clauses = []
            
            non_kcc = [s for s in sources if s != "kcc_qa"]
            if non_kcc:
                should_clauses.append(
                    Filter(must=[FieldCondition(key="source", match=MatchAny(any=non_kcc))])
                )
                
            if "kcc_qa" in sources:
                if allow_all_kcc or not qtypes or loosen_qtypes:
                    should_clauses.append(
                        Filter(must=[FieldCondition(key="source", match=MatchValue(value="kcc_qa"))])
                    )
                else:
                    should_clauses.append(
                        Filter(must=[
                            FieldCondition(key="source", match=MatchValue(value="kcc_qa")),
                            FieldCondition(key="query_type", match=MatchAny(any=list(qtypes)))
                        ])
                    )
            
            if should_clauses:
                query_filter = Filter(should=should_clauses)
    
    if not intents or len(intents) <= 1:
        q_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        hits  = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=query_filter,
            limit=active_top_k,
            with_payload=True
        ).points
    else:
        # Multi-Query Retrieval: BATCHED embedding + parallel searches
        limit_per_intent = max(2, active_top_k // len(intents))
        
        # Batched encoding: encode all sub-queries at once
        sub_queries = [f"{intent.replace('_', ' ')}: {query}" for intent in intents]
        vecs = embedder.encode(sub_queries, batch_size=len(sub_queries), normalize_embeddings=True)
        
        # Parallel Qdrant queries using ThreadPoolExecutor
        def _search(vec):
            return client.query_points(
                collection_name=COLLECTION_NAME,
                query=vec.tolist(),
                query_filter=query_filter,
                limit=limit_per_intent,
                with_payload=True
            ).points
        
        with ThreadPoolExecutor(max_workers=len(vecs)) as pool:
            results = list(pool.map(_search, vecs))
        
        hits = []
        for sub_hits in results:
            hits.extend(sub_hits)
            
        # Deduplicate and sort
        seen = set()
        unique_hits = []
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                unique_hits.append(h)
        hits = sorted(unique_hits, key=lambda x: x.score, reverse=True)[:active_top_k]

    if not hits:
        return [], "abstain", 0.0
    
    # Per-chunk score floor: filter out low-scoring chunks
    hits = [h for h in hits if h.score >= MIN_CHUNK_SCORE]
    if not hits:
        return [], "abstain", 0.0
    
    # Minimum chunk length filter: remove OCR noise and empty chunks
    hits = [h for h in hits if len(h.payload.get("text", "")) >= MIN_CHUNK_CHARS]
    if not hits:
        return [], "abstain", 0.0
    
    # Source diversity cap: limit chunks from same source
    source_counts = Counter()
    filtered_hits = []
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        src = h.payload.get("source", "unknown")
        if source_counts[src] < MAX_PER_SOURCE:
            filtered_hits.append(h)
            source_counts[src] += 1
    hits = filtered_hits[:active_top_k]
    
    if not hits:
        return [], "abstain", 0.0

    # ── Cross-encoder reranking ────────────────────────────────────────────
    if len(hits) > 1:
        try:
            reranker = _get_reranker()
            pairs    = [(query, h.payload.get("text", "")[:512]) for h in hits]
            scores   = reranker.predict(pairs)
            hits     = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
            hits     = hits[:RERANK_TOP_N]
        except Exception as e:
            pass  # reranker optional — fall back to vector score order

    top = hits[0].score
    tier = "grounded" if top >= TIER_GROUNDED else ("fallback" if top >= TIER_FALLBACK else "abstain")
    return hits, tier, top


def generate(gen_tok, gen_mdl, query: str, hits: list) -> str:
    ctx = "\n\n".join(
        f"[{i+1}] {h.payload.get('text', '')[:300]}"
        for i, h in enumerate(hits)
    )
    prompt = (
        "You are a helpful agricultural advisor for Indian farmers.\n"
        "Answer using ONLY the provided context. You MUST cite your sources as [1], [2] at the end of every sentence.\n"
        "If the question contains multiple topics, structure your answer to address each topic individually.\n\n"
        "Example:\n"
        "Context:\n[1] Urea costs 266 INR. [2] Use 50kg per hectare.\n"
        "Question: What is the price and dose of urea?\n"
        "Answer: The price of urea is 266 INR per bag [1]. For the dosage, you should apply 50kg per hectare [2].\n\n"
        f"Context:\n{ctx}\n\n"
        f"Farmer's question: {query}\n\nAnswer:"
    )
    inputs = gen_tok(prompt, return_tensors="pt",
                     truncation=True, max_length=2048).to(DEVICE)
    with torch.inference_mode():
        out = gen_mdl.generate(
            **inputs, max_new_tokens=150,
            temperature=0.3, do_sample=True, top_p=0.9,
            pad_token_id=gen_tok.eos_token_id,
        )
    return gen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def yield_predict(booster, crop: str, district: str, area_ha: float) -> float:
    """
    Predict yield (t/ha) for a crop in a given UP district.
    Uses UP defaults for inputs not provided by the user query.
    """
    crop_norm   = crop.lower().strip()
    dist_norm   = district.lower().strip()
    season      = CROP_TO_SEASON.get(crop_norm, "Kharif")
    crop_type   = CROP_TO_TYPE.get(crop_norm, "cereals")

    X = pd.DataFrame([{
        "crop":            crop_norm,
        "state":           UP_DEFAULTS["state"],
        "district":        dist_norm,
        "season":          season,
        "data_source":     UP_DEFAULTS["data_source"],
        "crop_type":       crop_type,
        "year":            UP_DEFAULTS["year"],
        "area":            area_ha,          # in the training data area is in hectares
        "annual_rainfall": UP_DEFAULTS["annual_rainfall"],
        "fertilizer":      UP_DEFAULTS["fertilizer"],
        "pesticide":       UP_DEFAULTS["pesticide"],
    }])
    for c in YIELD_CAT_COLS:
        X[c] = X[c].astype("category")

    return float(booster.predict(X)[0])


# ── Pathway evaluators ────────────────────────────────────────────────────────

def eval_A(row, ieg_model, ieg_tok, id2intent, ner_labels,
           retriever, embedder, gen_tok, gen_mdl):
    query        = str(row["query"])
    lang         = str(row["language"])
    expect_block = int(row["expected_block"])
    t0 = time.time()

    # Guard
    t_ieg = time.time()
    intent, blocked, m_flag, r_flag, top_conf = ieg_run(ieg_model, ieg_tok, id2intent, ner_labels, query)
    ieg_ms = round((time.time() - t_ieg) * 1000)
    guard_correct = (int(blocked) == expect_block)
    loosen = top_conf < GATE_THRESHOLD

    if blocked:
        return {
            "guardrail_fired":     True,
            "guardrail_correct":   guard_correct,
            "intent":              intent,
            "model_flag":          int(m_flag),
            "rule_flag":           int(r_flag),
            "tier":                "blocked",
            "top_score":           None,
            "citation_ok":         None,
            "lang_match_ok":       None,
            "numeric_grounded_ok": None,
            "latency_ms":          round((time.time() - t0) * 1000),
            "answer":              "[BLOCKED by guardrail]",
        }

    # Intercept Temporal intents (policy is NOT bypassed — it retrieves scheme PDFs)
    # NOTE: any() — same semantics as eval_AB; a weather/market intent anywhere
    # in the routing set bypasses, not just the top-ranked one.
    if any(i in ["weather", "market"] for i in intent):
        return {
            "guardrail_fired":     False,
            "guardrail_correct":   guard_correct,
            "intent":              ",".join(intent),
            "model_flag":          int(m_flag),
            "rule_flag":           int(r_flag),
            "tier":                "temporal_bypass",
            "top_score":           None,
            "citation_ok":         None,
            "lang_match_ok":       None,
            "numeric_grounded_ok": None,
            "latency_ms":          round((time.time() - t0) * 1000),
            "ieg_ms":              ieg_ms,
            "retrieval_ms":        0,
            "gen_ms":              0,
            "answer":              "[Temporal/Out-of-Scope Bypass] This query relates to weather or market prices which are not stored in the knowledge base.",
        }

    # Retrieve
    t_ret = time.time()
    hits, tier, top_score = retrieve(retriever, embedder, query, intents=intent,
                                     loosen_qtypes=loosen)
    retrieval_ms = round((time.time() - t_ret) * 1000)
    if tier == "abstain":
        return {
            "guardrail_fired":     False,
            "guardrail_correct":   guard_correct,
            "intent":              intent,
            "model_flag":          int(m_flag),
            "rule_flag":           int(r_flag),
            "top_conf":            round(top_conf, 4),
            "filters_loosened":    loosen,
            "tier":                "abstain",
            "top_score":           top_score,
            "citation_ok":         None,
            "lang_match_ok":       None,
            "numeric_grounded_ok": None,
            "latency_ms":          round((time.time() - t0) * 1000),
            "answer":              "[ABSTAINED — low retrieval score]",
        }

    # Generate
    answer = "[generator skipped]"
    completeness_ok = None
    gen_ms = 0
    if gen_tok and gen_mdl and tier != "abstain":
        t_gen = time.time()
        answer = generate(gen_tok, gen_mdl, query, hits)
        gen_ms = round((time.time() - t_gen) * 1000)
        if len(intent) > 1 or row["pathway"] == "A_Multi":
            completeness_ok = check_completeness(answer, intent, gen_tok, gen_mdl, JUDGE_URL)

    chunks = [h.payload.get("text", "") for h in hits]
    return {
        "guardrail_fired":     False,
        "guardrail_correct":   guard_correct,
        "intent":              ",".join(intent),
        "model_flag":          int(m_flag),
        "rule_flag":           int(r_flag),
        "top_conf":            round(top_conf, 4),
        "filters_loosened":    loosen,
        "tier":                tier,
        "top_score":           round(top_score, 4),
        "citation_ok":         check_citation(answer),
        "lang_match_ok":       check_lang_match(lang, answer),
        "numeric_grounded_ok": check_numeric_grounding(answer, chunks),
        "completeness_ok":     completeness_ok,
        "latency_ms":          round((time.time() - t0) * 1000),
        "ieg_ms":              ieg_ms,
        "retrieval_ms":        retrieval_ms,
        "gen_ms":              gen_ms,
        "answer":              answer[:300],
    }


def eval_B(row, vit_model, retriever, embedder, gen_tok, gen_mdl):
    vis_cls   = str(row["vision_class"])
    image_col = str(row.get("image_path", ""))
    t0 = time.time()

    # 1. ViT inference
    t_vit = time.time()
    if image_col and not Path(image_col).exists():
        if os.path.isdir("/kaggle/input"):
            cands = glob.glob(f"/kaggle/input/**/{Path(image_col).name}", recursive=True)
            if cands:
                image_col = cands[0]

    if image_col and Path(image_col).exists():
        # real image from kagglehub test split
        result = vit_model.predict(image_col)
        print(f"(real img) ", end="", flush=True)
    else:
        # no image path in CSV — simulate with the expected class label
        result = {
            "label": vis_cls, "confidence": 0.85,
            "top3": [(vis_cls, 0.85)], "rejected": False, "ood_score": 0.20,
            "_simulated": True,
        }
        print(f"(simulated) ", end="", flush=True)
    vit_ms = round((time.time() - t_vit) * 1000)


    label    = result["label"]
    conf     = result["confidence"]
    rejected = result["rejected"]

    if rejected or label is None:
        return {
            "vit_label":           label,
            "vit_conf":            conf,
            "vit_rejected":        True,
            "vit_simulated":       result.get("_simulated", False),
            "tier":                "rejected",
            "citation_ok":         None,
            "numeric_grounded_ok": None,
            "latency_ms":          round((time.time() - t0) * 1000),
            "answer":              "[Image rejected by OOD detector]",
        }

    # Build disease query from label
    disease_name = label.replace("__", " ").replace("_", " ")
    query        = f"{disease_name} disease treatment and management for Indian farmers"

    t_ret = time.time()
    hits, tier, top_score = retrieve(retriever, embedder, query, intents=["disease_pest"])
    retrieval_ms = round((time.time() - t_ret) * 1000)
    
    answer = "[generator skipped]"
    gen_ms = 0
    if gen_tok and gen_mdl and tier != "abstain":
        t_gen = time.time()
        answer = generate(gen_tok, gen_mdl, query, hits)
        gen_ms = round((time.time() - t_gen) * 1000)

    chunks = [h.payload.get("text", "") for h in hits]
    return {
        "vit_label":           label,
        "vit_conf":            round(conf, 4),
        "vit_rejected":        False,
        "vit_simulated":       result.get("_simulated", False),
        "tier":                tier,
        "top_score":           round(top_score, 4),
        "citation_ok":         check_citation(answer),
        "numeric_grounded_ok": check_numeric_grounding(answer, chunks),
        "latency_ms":          round((time.time() - t0) * 1000),
        "vit_ms":              vit_ms,
        "retrieval_ms":        retrieval_ms,
        "gen_ms":              gen_ms,
        "answer":              answer[:300],
    }


def eval_C(row, booster):
    t0 = time.time()
    crop     = str(row["yield_crop"])
    district = str(row["yield_district"])
    try:
        area = float(row["yield_area_ha"])
    except (ValueError, TypeError):
        area = 1.0

    pred = yield_predict(booster, crop, district, area)
    in_range = 0.2 <= pred <= 15.0  # plausible UP yield range t/ha

    return {
        "yield_t_ha":      round(pred, 3),
        "total_yield_t":   round(pred * area, 2),
        "yield_in_range":  in_range,
        "latency_ms":      round((time.time() - t0) * 1000),
        "answer":          (
            f"Estimated yield for {crop} in {district}: "
            f"{pred:.2f} t/ha ({pred*area:.1f} t total for {area} ha)"
        ),
    }


def eval_AB(row, ieg_model, ieg_tok, id2intent, ner_labels,
            vit_model, retriever, embedder, gen_tok, gen_mdl):
    text = str(row["query"])
    vis_cls = str(row["vision_class"])
    image_col = str(row.get("image_path", ""))
    if image_col and not Path(image_col).exists():
        if os.path.isdir("/kaggle/input"):
            cands = glob.glob(f"/kaggle/input/**/{Path(image_col).name}", recursive=True)
            if cands:
                image_col = cands[0]
    t0 = time.time()

    # Run ViT inference and IEG inference in PARALLEL
    def _run_vit():
        t_vit = time.time()
        if image_col and Path(image_col).exists():
            result = vit_model.predict(image_col)
        else:
            result = {
                "label": vis_cls, "confidence": 0.85,
                "top3": [(vis_cls, 0.85)], "rejected": False, "ood_score": 0.20,
                "_simulated": True,
            }
        vit_ms = round((time.time() - t_vit) * 1000)
        return result, vit_ms
    
    def _run_ieg():
        t_ieg = time.time()
        intent, blocked, m_flag, r_flag, top_conf = ieg_run(ieg_model, ieg_tok, id2intent, ner_labels, text)
        ieg_ms = round((time.time() - t_ieg) * 1000)
        return intent, blocked, m_flag, r_flag, ieg_ms, top_conf

    with ThreadPoolExecutor(max_workers=2) as pool:
        vit_future = pool.submit(_run_vit)
        ieg_future = pool.submit(_run_ieg)

        result, vit_ms = vit_future.result()
        intent, blocked, m_flag, r_flag, ieg_ms, top_conf = ieg_future.result()
    loosen = top_conf < GATE_THRESHOLD
    
    label = result["label"]
    if result["rejected"] or label is None:
        return {
            "vit_label":           label,
            "vit_rejected":        True,
            "tier":                "rejected",
            "latency_ms":          round((time.time() - t0) * 1000),
            "answer":              "[Image rejected]",
        }

    if blocked:
        return {
            "vit_label":           label,
            "vit_rejected":        False,
            "guardrail_fired":     True,
            "intent":              ",".join(intent),
            "tier":                "blocked",
            "latency_ms":          round((time.time() - t0) * 1000),
            "answer":              "[BLOCKED by guardrail]",
        }

    # Intercept Temporal intents (policy is NOT bypassed — it retrieves scheme PDFs)
    if any(i in ["weather", "market"] for i in intent):
        return {
            "vit_label":           label,
            "vit_rejected":        False,
            "guardrail_fired":     False,
            "intent":              ",".join(intent),
            "tier":                "temporal_bypass",
            "top_score":           None,
            "citation_ok":         None,
            "numeric_grounded_ok": None,
            "completeness_ok":     None,
            "latency_ms":          round((time.time() - t0) * 1000),
            "vit_ms":              vit_ms,
            "ieg_ms":              ieg_ms,
            "retrieval_ms":        0,
            "gen_ms":              0,
            "answer":              "[Temporal/Out-of-Scope Bypass] This query relates to weather or market prices which are not stored in the knowledge base.",
        }

    # 3. Retrieve
    disease_name = label.replace("__", " ").replace("_", " ")
    combined_query = f"{disease_name}. {text}"
    t_ret = time.time()
    hits, tier, top_score = retrieve(retriever, embedder, combined_query,
                                     intents=intent + ["disease_pest"],
                                     loosen_qtypes=loosen)
    retrieval_ms = round((time.time() - t_ret) * 1000)
    
    # 4. Generate
    answer = "[generator skipped]"
    completeness_ok = None
    gen_ms = 0
    if gen_tok and gen_mdl and tier != "abstain":
        t_gen = time.time()
        answer = generate(gen_tok, gen_mdl, combined_query, hits)
        gen_ms = round((time.time() - t_gen) * 1000)
        completeness_ok = check_completeness(answer, intent + [disease_name], gen_tok, gen_mdl, JUDGE_URL)

    chunks = [h.payload.get("text", "") for h in hits]
    return {
        "vit_label":           label,
        "vit_rejected":        False,
        "guardrail_fired":     False,
        "intent":              ",".join(intent),
        "top_conf":            round(top_conf, 4),
        "filters_loosened":    loosen,
        "tier":                tier,
        "top_score":           round(top_score, 4),
        "citation_ok":         check_citation(answer),
        "numeric_grounded_ok": check_numeric_grounding(answer, chunks),
        "completeness_ok":     completeness_ok,
        "latency_ms":          round((time.time() - t0) * 1000),
        "vit_ms":              vit_ms,
        "ieg_ms":              ieg_ms,
        "retrieval_ms":        retrieval_ms,
        "gen_ms":              gen_ms,
        "answer":              answer[:300],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pathway", choices=["A","B","C","A_Multi","AB","all"], default="all",
                   help="Which pathway(s) to evaluate")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip generator (fast mode — checks guardrail, retrieval, ViT only)")
    p.add_argument("--scenarios", default=str(SCENARIOS_CSV),
                   help="Path to scenarios CSV")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit evaluation to first N rows (for quick testing)")
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="Override CONF_GATE_LOOSEN: loosen qtype filters when "
                        "top-1 softmax conf < threshold. 0=off, 1.0=always loose.")
    p.add_argument("--budget-min", type=float, default=None,
                   help="Time budget in minutes. If set, proportionally stratify-sample "
                        "rows (by gold intent_label) to fit the budget.")
    p.add_argument("--est-sec-per-row", type=float, default=None,
                   help="Estimated seconds per row for --budget-min. Defaults: "
                        "~3 skip-gen / ~30 with generator (T4).")
    p.add_argument("--judge-url", type=str, default=None,
                   help="External judge endpoint (e.g. Qwen3-8B on vLLM) for "
                        "completeness/citation evaluation. POSTs {'topics':[], 'answer':str} "
                        "expects {'verdict':'yes'|'no', 'score':float}.")
    return p.parse_args()


def stratified_sample_for_budget(df, budget_min: float, est_sec: float):
    """Proportionally sample rows (stratified by gold intent_label when available)
    so the run fits inside the time budget."""
    n_max = max(1, int(budget_min * 60 / est_sec))
    if len(df) <= n_max:
        print(f"Budget: {len(df)} rows fit within {budget_min} min (est {est_sec}s/row). No sampling.")
        return df
    label_col = "intent_label" if "intent_label" in df.columns else None
    if label_col:
        # proportional per-group quota, min 1 per class; cumcount-based selection
        # (groupby.apply+sample can drop columns on some pandas versions)
        quota = (df.groupby(label_col).size()
                   .mul(n_max / len(df)).round().clip(lower=1).astype(int))
        keep = df.groupby(label_col).cumcount() < df[label_col].map(quota)
        sampled = df[keep].sample(frac=1.0, random_state=42)   # shuffle
    else:
        sampled = df.sample(n=n_max, random_state=42)
    print(f"Budget: sampling {len(sampled)}/{len(df)} rows "
          f"({budget_min} min @ {est_sec}s/row, stratified by {label_col or 'uniform'}).")
    if label_col:
        print(f"Budget: class balance -> {sampled[label_col].value_counts().to_dict()}")
    return sampled


def resolve_scenarios_csv(path_arg: str) -> Path:
    """Resolve the scenarios CSV.

    Priority:
      1. Explicit path if it exists (local or absolute).
      2. Kaggle input mount: glob /kaggle/input/** for scenario CSVs
         (prefers e2e_scenarios*.csv, then any *scenarios*.csv).
      3. Fall back to the given path (will raise a clear error on read).
    """
    p = Path(path_arg)
    if p.exists():
        return p

    if os.path.isdir("/kaggle/input"):
        for pattern in ("kcc_eval_1_aug*.csv", "e2e_scenarios*.csv", "*scenarios*.csv", "kcc_eval_*.csv"):
            cands = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
            if cands:
                print(f"  [scenarios] Using Kaggle input: {cands[0]}")
                return Path(cands[0])
        print("  [scenarios] WARNING: /kaggle/input mounted but no scenario CSV found.")

    print(f"  [scenarios] WARNING: {p} not found and no Kaggle input fallback.")
    return p


def main():
    global GATE_THRESHOLD, JUDGE_URL
    args = parse_args()
    if args.gate_threshold is not None:
        GATE_THRESHOLD = args.gate_threshold
    if args.judge_url:
        JUDGE_URL = args.judge_url
    print("=" * 60)
    print("FarmerVision — E2E Pipeline Evaluation")
    print(f"Pathway: {args.pathway} | Skip generator: {args.skip_gen} "
          f"| Conf-gate: {GATE_THRESHOLD}")
    print("=" * 60)

    scenarios_csv = resolve_scenarios_csv(args.scenarios)
    df = pd.read_csv(scenarios_csv)
    if "multi_turn_json" in df.columns:
        df["fup"] = df["multi_turn_json"].apply(
            lambda x: next((t["content"] for t in (json.loads(x) if isinstance(x,str) else x) if t["role"]=="user"), "")
            if pd.notna(x) else ""
        )
    
    # Standardize columns
    if "query" not in df.columns:
        df["query"] = df.get("fup", df.get("QueryText", ""))
    if "language" not in df.columns:
        df["language"] = "English"
    if "expected_block" not in df.columns:
        # Default to 1 (blocked) for non_agri, else 0
        if "intent_label" in df.columns:
            df["expected_block"] = (df["intent_label"] == "non_agri").astype(int)
        elif "guardrail" in df.columns:
            df["expected_block"] = df["guardrail"].astype(int)
        else:
            df["expected_block"] = 0
    if "scenario_id" not in df.columns:
        df["scenario_id"] = [f"SCEN-{i:03d}" for i in range(len(df))]
    if "pathway" not in df.columns:
        df["pathway"] = "A"

    if args.pathway != "all" and "pathway" in df.columns:
        df = df[df["pathway"] == args.pathway].copy()
    if args.limit:
        df = df.head(args.limit)
    if args.budget_min:
        est = args.est_sec_per_row or (3.0 if args.skip_gen else 30.0)
        df = stratified_sample_for_budget(df, args.budget_min, est)
    print(f"\nScenarios: {len(df)}")

    # ── Load components ─────────────────────────────────────────────────────
    print("\n--- Loading models ---")
    STATUS = {}

    print("[IEG]")
    try:
        ieg_model, ieg_tok, id2intent, ner_labels = load_ieg()
        STATUS["ieg"] = "OK"
    except Exception as e:
        print(f"  ERROR: {e}")
        ieg_model = ieg_tok = id2intent = ner_labels = None
        STATUS["ieg"] = str(e)

    print("[ViT]")
    try:
        vit_model = load_vit()
        STATUS["vit"] = "OK"
    except Exception as e:
        print(f"  ERROR: {e}")
        vit_model = None
        STATUS["vit"] = str(e)

    print("[Qdrant + bge-m3]")
    try:
        retriever, embedder = load_retriever()
        STATUS["retriever"] = "OK"
    except Exception as e:
        print(f"  ERROR: {e}")
        retriever = embedder = None
        STATUS["retriever"] = str(e)

    gen_tok = gen_mdl = None
    if not args.skip_gen:
        print("[Generator]")
        try:
            gen_tok, gen_mdl = load_generator()
            STATUS["generator"] = "OK"
        except Exception as e:
            print(f"  ERROR: {e}")
            STATUS["generator"] = str(e)
    else:
        STATUS["generator"] = "skipped"

    print("[Yield LightGBM]")
    try:
        yield_booster = load_yield_model()
        STATUS["yield"] = "OK"
    except Exception as e:
        print(f"  ERROR: {e}")
        yield_booster = None
        STATUS["yield"] = str(e)

    print(f"\nStatus: {STATUS}\n")

    # ── Run evaluation ───────────────────────────────────────────────────────
    print("--- Running scenarios ---")
    results = []
    for i, row in df.iterrows():
        sid     = row.get("scenario_id", f"SCEN-{i:03d}")
        pathway = row.get("pathway", "A")
        print(f"  [{i+1:02d}/{len(df)}] {sid} … ", end="", flush=True)

        query_text = row.get("query", row.get("fup", row.get("QueryText", "")))
        base = {
            "scenario_id":      sid,
            "pathway":          pathway,
            "query":            str(query_text)[:80],
            "language":         row.get("language", "English"),
            "crop":             row.get("crop", "unknown"),
            "expected_block":   row.get("expected_block", row.get("intent_label", "")),
            "gold_intent_label": row.get("intent_label", ""),   # kept for post-hoc per-intent analysis
            "notes":            row.get("notes", ""),
        }

        try:
            if pathway in ["A", "A_Multi"]:
                if ieg_model is None or retriever is None:
                    raise RuntimeError("IEG or retriever not available")
                r = eval_A(row, ieg_model, ieg_tok, id2intent, ner_labels,
                           retriever, embedder, gen_tok, gen_mdl)
            elif pathway == "B":
                if vit_model is None or retriever is None:
                    raise RuntimeError("ViT or retriever not available")
                r = eval_B(row, vit_model, retriever, embedder, gen_tok, gen_mdl)
            elif pathway == "AB":
                if vit_model is None or ieg_model is None or retriever is None:
                    raise RuntimeError("ViT, IEG, or retriever not available")
                r = eval_AB(row, ieg_model, ieg_tok, id2intent, ner_labels,
                            vit_model, retriever, embedder, gen_tok, gen_mdl)
            elif pathway == "C":
                if yield_booster is None:
                    raise RuntimeError("Yield model not available")
                r = eval_C(row, yield_booster)
            else:
                r = {"error": f"unknown pathway {pathway}"}

            base.update(r)
            base["error"] = None
            indicator = "BLOCK" if r.get("guardrail_fired") else r.get("tier", "OK")
            print(f"{indicator} ({r.get('latency_ms', '?')} ms)")
        except Exception as e:
            base["error"] = str(e)
            print(f"ERROR: {e}")

        results.append(base)

    df_results = pd.DataFrame(results)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nResults saved -> {RESULTS_CSV.relative_to(ROOT)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "status":         STATUS,
        "total":          len(df_results),
        "errors":         int(df_results["error"].notna().sum()),
        "gate_threshold": GATE_THRESHOLD,
        "by_pathway":     {},
    }

    for pathway in df_results["pathway"].unique():
        sub = df_results[df_results["pathway"] == pathway]
        if len(sub) == 0:
            continue
        p = {"n": len(sub), "errors": int(sub["error"].notna().sum())}

        if pathway in ["A", "A_Multi"]:
            sub_clean = sub[sub["expected_block"] == 0]
            sub_block = sub[sub["expected_block"] == 1]
            if "guardrail_correct" in sub:
                p["guardrail_correct"]      = int(sub["guardrail_correct"].sum())
                p["guardrail_correct_pct"]  = round(sub["guardrail_correct"].mean() * 100, 1)
            if "tier" in sub:
                p["tier_distribution"] = sub["tier"].value_counts().to_dict()
            # Generation-dependent metrics are only meaningful when the
            # generator ran; "[generator skipped]" rows would poison them.
            sub_gen = sub_clean[sub_clean["answer"] != "[generator skipped]"]
            for col in ["citation_ok", "lang_match_ok", "numeric_grounded_ok", "completeness_ok"]:
                if col in sub_gen:
                    vals = sub_gen[col].dropna()
                    if len(vals):
                        p[col] = int(vals.sum())
                        p[f"{col}_pct"] = round(vals.mean() * 100, 1)
                        p[f"{col}_n"] = len(vals)

        if pathway in ["B", "AB"]:
            if "vit_simulated" in sub:
                p["simulated"] = int(sub["vit_simulated"].sum())
            for col in ["citation_ok", "numeric_grounded_ok", "completeness_ok"]:
                if col in sub:
                    vals = sub[col].dropna()
                    if len(vals):
                        p[col] = int(vals.sum())

        if pathway == "C":
            if "yield_in_range" in sub:
                p["yield_in_range"]     = int(sub["yield_in_range"].sum())
                p["yield_in_range_pct"] = round(sub["yield_in_range"].mean() * 100, 1)
            if "yield_t_ha" in sub:
                p["yield_t_ha_mean"] = round(sub["yield_t_ha"].mean(), 3)

        summary["by_pathway"][pathway] = p

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFull results -> {RESULTS_CSV.relative_to(ROOT)}")
    print(f"Summary      -> {SUMMARY_JSON.relative_to(ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
