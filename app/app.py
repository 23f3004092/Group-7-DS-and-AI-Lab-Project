"""
app/app.py
==========
FarmerVision FastAPI backend — three pathways wired to real models.

Endpoints
---------
POST /query/text    -> Pathway A: text -> IEG guardrail -> Qdrant -> gemma-3-4b-it
POST /query/image   -> Pathway B: image file -> ViT-S/16 -> Qdrant -> gemma-3-4b-it
POST /query/yield   -> Pathway C: crop+district+area -> LightGBM (lightgbm_tuned.txt)
GET  /health        -> component health check

Usage
-----
  # 1. Start Qdrant:   python scripts/setup_vectordb.py
  # 2. Start the API:  uvicorn app.app:app --reload --port 8000
  # 3. Test:
  #    curl http://localhost:8000/health
  #    curl -X POST http://localhost:8000/query/text \\
  #         -H 'Content-Type: application/json' \\
  #         -d '{"text": "wheat crop turning yellow what to do"}'

Environment variables (all optional)
-------------------------------------
QDRANT_URL          default: http://localhost:6333
SKIP_GENERATOR      set to 1 to skip gemma (fast mode for retrieval-only testing)
"""

import io
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent.parent
# Force smaller models to CPU to prevent VRAM overflow and WDDM RAM spilling
DEVICE = "cpu"

# Qdrant / retrieval config — from manifest.json
QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "agri_knowledge"
BGE_MODEL_ID    = "BAAI/bge-m3"
TOP_K           = 10
TOP_K_BASE      = 10  # Base value for dynamic scaling
TIER_GROUNDED   = 0.66
TIER_FALLBACK   = 0.56
MIN_CHUNK_SCORE = 0.50  # Filter out low-scoring chunks
MIN_CHUNK_CHARS = 50    # Minimum chunk length to avoid OCR noise
MAX_PER_SOURCE  = 3     # Cap chunks from same source for diversity

# Intent to source routing map for payload filtering
INTENT_SOURCE_MAP = {
    "disease_pest":         ["ppqs_advisories", "kcc"],
    "cultivation_practice": ["up_acp", "kcc"],
    "nutrition_fertilizer": ["ppqs_advisories", "kcc"],
    "yield_estimation":     ["kcc"],
    "general":              None,   # no filter → search all
}

# IEG config — from real label_maps.json
IEG_MODEL_ID   = "distilbert-base-multilingual-cased"
INTENT_CLASSES = [
    "cultivation_practice", "disease_pest", "general", "non_agri",
    "nutrition_fertilizer", "post_harvest_storage", "specialty_other",
]
NER_LABELS = ["O", "B-CROP", "I-CROP"]
MAX_LEN    = 64

# Yield model — LightGBM (lightgbm_tuned.txt from notebook output)
YIELD_MODEL_PATH = (ROOT / "outputs" / "Yeild_output_files" / "kaggle" /
                    "working" / "saved_models" / "lightgbm_tuned.txt")
YIELD_CAT_COLS   = ["crop", "state", "district", "season", "data_source", "crop_type"]
CROP_TO_TYPE     = {"wheat": "cereals", "rice": "cereals", "paddy": "cereals"}
CROP_TO_SEASON   = {"wheat": "Rabi",    "rice": "Kharif",  "paddy": "Kharif"}
UP_DEFAULTS      = {
    "state": "uttar pradesh",
    "data_source": "area_production",
    "annual_rainfall": 850.0,
    "fertilizer": 1_200_000.0,
    "pesticide": 2_400.0,
    "year": 2023,
}

SKIP_GENERATOR = bool(os.environ.get("SKIP_GENERATOR", ""))
USE_RERANKER = bool(os.environ.get("USE_RERANKER", ""))  # Optional cross-encoder reranking

# ── IEG model (exact architecture from notebook) ──────────────────────────────

class IEGModel(nn.Module):
    def __init__(self, n_intents: int, n_ner: int):
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


# ── Guardrail keyword rules ───────────────────────────────────────────────────

_BANNED = [
    "monocrotophos", "endosulfan", "phorate",
    "methyl parathion", "ddt", "aldrin", "chlorpyrifos",
]
_DOSAGE = [
    r"\b(\d+)\s*(x|times)\b.*\b(dose|dosage|dawa|spray)\b",
    r"\b(double|triple|4x|5x|10x)\b.*\b(dose|dosage|strength|dawa)\b",
    r"\bz?jyada\b.*\bdawa\b",
    r"\b(\d{3,})\s*kg\b.*\b(per acre|prati ekad|ekad)\b",
]

def _rule_flag(text: str) -> bool:
    t = str(text).lower()
    return any(w in t for w in _BANNED) or any(re.search(p, t) for p in _DOSAGE)


# ── Global component state ────────────────────────────────────────────────────

_state: dict = {
    "ieg_model":  None, "ieg_tok":    None, "id2intent": None,
    "retriever":  None, "embedder":   None,
    "gen_tok":    None, "gen_mdl":    None,
    "yield_mdl":  None,
    "vit":        None,
    "reranker":   None,
}
STATUS: dict = {}


def _load_ieg():
    import kagglehub
    path = Path(kagglehub.dataset_download("aneeqasiddiqui377/v4-output"))
    ckpt = path / "results" / "final_run2" / "ieg_adamw_lr3e-5_5ep.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = IEGModel(n_intents=len(INTENT_CLASSES), n_ner=len(NER_LABELS))
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(IEG_MODEL_ID)
    _state["ieg_model"] = model
    _state["ieg_tok"]   = tok
    _state["id2intent"] = {i: c for i, c in enumerate(INTENT_CLASSES)}
    STATUS["ieg"] = "OK"
    print("IEG loaded")


def _load_vit():
    import kagglehub
    path = Path(kagglehub.dataset_download("iitm21f1003346/vits16-crop-disease"))
    model_dir = next(path.rglob("predict.py")).parent
    sys.path.insert(0, str(model_dir))
    from predict import CropDiseaseModel
    _state["vit"] = CropDiseaseModel(device=DEVICE)
    STATUS["vit"] = "OK"
    print("ViT CropDiseaseModel loaded")


def _load_retriever():
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    client = QdrantClient(url=QDRANT_URL, timeout=30)
    client.get_collection(COLLECTION_NAME)
    _state["retriever"] = client
    _state["embedder"]  = SentenceTransformer(BGE_MODEL_ID, device="cpu")
    
    # Optional: Load cross-encoder reranker
    if USE_RERANKER:
        from sentence_transformers import CrossEncoder
        _state["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        STATUS["reranker"] = "OK"
        print("  Cross-encoder reranker loaded")
    else:
        STATUS["reranker"] = "disabled"
    
    STATUS["retriever"] = "OK"
    print(f"Qdrant OK ({QDRANT_URL}), bge-m3 loaded")


def _load_generator():
    if SKIP_GENERATOR:
        STATUS["generator"] = "skipped (SKIP_GENERATOR=1)"
        return
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    mdl = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it", 
        quantization_config=bnb, 
        device_map="cuda",
        attn_implementation="sdpa",
    )
    mdl.eval()
    _state["gen_tok"] = tok
    _state["gen_mdl"] = mdl
    STATUS["generator"] = "OK"
    print("Generator (gemma-3-4b-it, 4-bit) loaded")


def _load_yield():
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(YIELD_MODEL_PATH))
    _state["yield_mdl"] = booster
    STATUS["yield"] = f"OK ({booster.num_trees()} trees)"
    print(f"Yield LightGBM loaded ({booster.num_trees()} trees)")


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FarmerVision API",
    description="Multimodal agricultural advisory for Indian farmers",
    version="0.2.0",
)


@app.on_event("startup")
async def startup():
    print(f"Starting FarmerVision API on {DEVICE} …")
    for name, fn in [("IEG", _load_ieg), ("ViT", _load_vit),
                     ("Retriever", _load_retriever),
                     ("Generator", _load_generator), ("Yield", _load_yield)]:
        try:
            fn()
        except Exception as e:
            STATUS[name.lower()] = f"ERROR: {e}"
            print(f"  {name} failed: {e}")
    print(f"Startup done. STATUS={STATUS}")


# ── Pydantic request models ───────────────────────────────────────────────────

class TextQuery(BaseModel):
    text: str

class YieldQuery(BaseModel):
    crop:     str    # "wheat" | "rice"
    district: str    # UP district name in lowercase
    area_ha:  float  # farm area in hectares


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ieg_run(text: str):
    m, tok, id2i = _state["ieg_model"], _state["ieg_tok"], _state["id2intent"]
    if m is None:
        return "unknown", False
    enc = tok(text, truncation=True, max_length=MAX_LEN,
               padding="max_length", return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        il, _, gl = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    
    probs = torch.sigmoid(il[0])
    intents = [id2i[i] for i, p in enumerate(probs) if p > 0.3]
    if not intents:
        intents = [id2i[il.argmax(-1).item()]]
        
    blocked = bool(gl.argmax(-1).item() == 1) or _rule_flag(text)
    return intents, blocked


def _retrieve(query: str, intents: list = None):
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    
    client, emb = _state["retriever"], _state["embedder"]
    if client is None:
        return [], "retrieval_unavailable", 0.0
    
    # Dynamic TOP_K scaling with intent count
    active_top_k = max(TOP_K_BASE, len(intents) * 5) if intents and len(intents) > 2 else TOP_K
    
    # Intent → Source routing: build payload filter
    query_filter = None
    if intents and len(intents) > 0:
        sources = set()
        for intent in intents:
            srcs = INTENT_SOURCE_MAP.get(intent)
            if srcs:
                sources.update(srcs)
        if sources:
            query_filter = Filter(must=[
                FieldCondition(key="source", match=MatchAny(any=list(sources)))
            ])
    
    if not intents or len(intents) <= 1:
        q_vec = emb.encode(query, normalize_embeddings=True).tolist()
        hits  = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=query_filter,
            limit=active_top_k,
            with_payload=True
        ).points
    else:
        # Multi-Query Retrieval with BATCHED embedding
        limit_per_intent = max(2, active_top_k // len(intents))
        
        # Batched encoding: encode all sub-queries at once
        sub_queries = [f"{intent.replace('_', ' ')}: {query}" for intent in intents]
        vecs = emb.encode(sub_queries, batch_size=len(sub_queries), normalize_embeddings=True)
        
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
    
    # Cross-encoder reranking (optional)
    reranker = _state.get("reranker")
    if reranker and hits:
        pairs = [(query, h.payload.get("text", "")) for h in hits]
        scores = reranker.predict(pairs)
        # Reorder by cross-encoder scores
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        # Optionally filter out very low cross-encoder scores
        hits = [h for h, s in zip(hits, sorted(scores, reverse=True)) if s > 0.0]
    
    if not hits:
        return [], "abstain", 0.0
    
    score = hits[0].score
    tier  = ("grounded" if score >= TIER_GROUNDED
             else "fallback" if score >= TIER_FALLBACK else "abstain")
    return hits, tier, score


def _generate(query: str, hits: list) -> str:
    tok, mdl = _state["gen_tok"], _state["gen_mdl"]
    if tok is None:
        return "[generator not loaded — set SKIP_GENERATOR=0 and restart]"
    ctx = "\n\n".join(
        f"[{i+1}] {h.payload.get('text','')[:300]}" for i, h in enumerate(hits)
    )
    prompt = (
        "You are a helpful agricultural advisor for Indian farmers.\n"
        "Answer using ONLY the provided context. You MUST cite your sources as [1], [2] at the end of every sentence.\n"
        "If the question contains multiple topics, structure your answer to address each topic individually.\n\n"
        "Example:\n"
        "Context:\n[1] Urea costs 266 INR. [2] Use 50kg per hectare.\n"
        "Question: What is the price and dose of urea?\n"
        "Answer: The price of urea is 266 INR per bag [1]. For the dosage, you should apply 50kg per hectare [2].\n\n"
        f"Context:\n{ctx}\n\nFarmer's question: {query}\n\nAnswer:"
    )
    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=150, temperature=0.3,
                            do_sample=True, top_p=0.9,
                            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _yield_predict(crop: str, district: str, area_ha: float) -> float:
    import pandas as pd
    booster = _state["yield_mdl"]
    crop_n  = crop.lower().strip()
    X = pd.DataFrame([{
        "crop":            crop_n,
        "state":           UP_DEFAULTS["state"],
        "district":        district.lower().strip(),
        "season":          CROP_TO_SEASON.get(crop_n, "Kharif"),
        "data_source":     UP_DEFAULTS["data_source"],
        "crop_type":       CROP_TO_TYPE.get(crop_n, "cereals"),
        "year":            UP_DEFAULTS["year"],
        "area":            area_ha,
        "annual_rainfall": UP_DEFAULTS["annual_rainfall"],
        "fertilizer":      UP_DEFAULTS["fertilizer"],
        "pesticide":       UP_DEFAULTS["pesticide"],
    }])
    for c in YIELD_CAT_COLS:
        X[c] = X[c].astype("category")
    return float(booster.predict(X)[0])


def _caveat(tier: str) -> Optional[str]:
    if tier == "fallback":
        return "Advice based on general guidance. Please confirm with your local KVK officer."
    if tier == "abstain":
        return "Your query is outside the system's knowledge. Please contact a KVK officer."
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    all_ok = all("OK" in str(v) or "skipped" in str(v) for v in STATUS.values())
    return JSONResponse({
        "status":     "ok" if all_ok else "degraded",
        "device":     DEVICE,
        "components": STATUS,
    })


@app.post("/query/text")
def query_text(q: TextQuery):
    """Pathway A — text query."""
    t0   = time.time()
    text = q.text.strip()
    if not text:
        raise HTTPException(400, "text is empty")

    intents, blocked = _ieg_run(text)
    if blocked:
        return JSONResponse({
            "pathway":    "A",
            "intent":     intents,
            "blocked":    True,
            "tier":       "blocked",
            "answer":     "This query cannot be answered safely. Please contact a KVK officer.",
            "latency_ms": round((time.time() - t0) * 1000),
        })

    hits, tier, top_score = _retrieve(text, intents=intents)
    caveat = _caveat(tier)
    if tier == "abstain":
        return JSONResponse({
            "pathway":    "A",
            "intent":     intents,
            "blocked":    False,
            "tier":       "abstain",
            "top_score":  round(top_score, 4),
            "answer":     caveat,
            "latency_ms": round((time.time() - t0) * 1000),
        })

    answer = _generate(text, hits)
    if caveat:
        answer = f"{answer}\n\n⚠ {caveat}"

    sources = [{"rank": i+1, "score": round(h.score, 3),
                "crop": h.payload.get("crop",""),
                "type": h.payload.get("source_type","")}
               for i, h in enumerate(hits[:5])]

    return JSONResponse({
        "pathway":    "A",
        "intent":     intents,
        "blocked":    False,
        "tier":       tier,
        "top_score":  round(top_score, 4),
        "answer":     answer,
        "sources":    sources,
        "latency_ms": round((time.time() - t0) * 1000),
    })


@app.post("/query/image")
async def query_image(file: UploadFile = File(...)):
    """Pathway B — leaf image -> ViT -> retrieval -> generation."""
    t0  = time.time()
    vit = _state["vit"]
    if vit is None:
        raise HTTPException(503, "ViT model not loaded")

    from PIL import Image
    img_bytes = await file.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result    = vit.predict(pil_img)

    if result["rejected"] or result["label"] is None:
        return JSONResponse({
            "pathway":    "B",
            "rejected":   True,
            "ood_score":  result["ood_score"],
            "answer":     "Image rejected — please submit a clear, close-up leaf photo.",
            "latency_ms": round((time.time() - t0) * 1000),
        })

    label = result["label"]
    conf  = result["confidence"]
    top3  = result["top3"]

    query  = f"{label.replace('__',' ').replace('_',' ')} disease treatment and management"
    hits, tier, score = _retrieve(query)
    caveat = _caveat(tier)

    answer = "[generator not available]"
    if tier != "abstain":
        answer = _generate(query, hits)
        if caveat:
            answer = f"{answer}\n\n⚠ {caveat}"
    else:
        answer = caveat

    return JSONResponse({
        "pathway":    "B",
        "rejected":   False,
        "vit_label":  label,
        "confidence": round(conf, 4),
        "top3":       top3,
        "ood_score":  result["ood_score"],
        "tier":       tier,
        "top_score":  round(score, 4),
        "answer":     answer,
        "latency_ms": round((time.time() - t0) * 1000),
    })


@app.post("/query/yield")
def query_yield(q: YieldQuery):
    """Pathway C — yield estimation via LightGBM."""
    t0 = time.time()
    if _state["yield_mdl"] is None:
        raise HTTPException(503, "Yield model not loaded")
    try:
        pred_t_ha   = _yield_predict(q.crop, q.district, q.area_ha)
        total_yield = round(pred_t_ha * q.area_ha, 2)
    except Exception as e:
        raise HTTPException(500, f"Yield prediction failed: {e}")

    return JSONResponse({
        "pathway":              "C",
        "crop":                 q.crop,
        "district":             q.district,
        "area_ha":              q.area_ha,
        "predicted_yield_t_ha": round(pred_t_ha, 3),
        "total_yield_t":        total_yield,
        "answer": (
            f"Expected yield for {q.crop} on {q.area_ha} ha in {q.district}: "
            f"~{pred_t_ha:.2f} t/ha ({total_yield:.1f} tonnes total). "
            "Estimate based on historical UP district data (LightGBM, R²=0.957)."
        ),
        "latency_ms": round((time.time() - t0) * 1000),
    })


@app.post("/query/multimodal")
async def query_multimodal(text: str = Form(...), file: UploadFile = File(...)):
    """Pathway AB — text + image -> combined retrieval -> generation."""
    t0 = time.time()
    
    # 1. Prepare inputs
    vit = _state["vit"]
    if vit is None:
        raise HTTPException(503, "ViT model not loaded")

    from PIL import Image
    img_bytes = await file.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    text = text.strip()
    
    # 2. Run ViT and IEG in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        vit_future = pool.submit(vit.predict, pil_img)
        ieg_future = pool.submit(_ieg_run, text)
        
        result = vit_future.result()
        intents, blocked = ieg_future.result()

    if result["rejected"] or result["label"] is None:
        return JSONResponse({
            "pathway":    "AB",
            "rejected":   True,
            "ood_score":  result["ood_score"],
            "answer":     "Image rejected — please submit a clear, close-up leaf photo.",
            "latency_ms": round((time.time() - t0) * 1000),
        })

    label = result["label"]
    conf  = result["confidence"]
    top3  = result["top3"]
    
    if blocked:
        return JSONResponse({
            "pathway":    "AB",
            "intent":     intents,
            "blocked":    True,
            "tier":       "blocked",
            "answer":     "This query cannot be answered safely. Please contact a KVK officer.",
            "latency_ms": round((time.time() - t0) * 1000),
        })
        
    # 3. Combine and Retrieve
    combined_query = f"{label.replace('__',' ').replace('_',' ')}. {text}"
    hits, tier, top_score = _retrieve(combined_query, intents=intents + ["disease_pest"])
    caveat = _caveat(tier)

    # 4. Generate
    answer = "[generator not available]"
    if tier != "abstain":
        answer = _generate(combined_query, hits)
        if caveat:
            answer = f"{answer}\n\n⚠ {caveat}"
    else:
        answer = caveat

    sources = [{"rank": i+1, "score": round(h.score, 3),
                "crop": h.payload.get("crop",""),
                "type": h.payload.get("source_type","")}
               for i, h in enumerate(hits[:5])]

    return JSONResponse({
        "pathway":    "AB",
        "intent":     intents,
        "blocked":    False,
        "rejected":   False,
        "vit_label":  label,
        "confidence": round(conf, 4),
        "top3":       top3,
        "ood_score":  result["ood_score"],
        "tier":       tier,
        "top_score":  round(top_score, 4),
        "answer":     answer,
        "sources":    sources,
        "latency_ms": round((time.time() - t0) * 1000),
    })
