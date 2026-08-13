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
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
TIER_GROUNDED   = 0.66
TIER_FALLBACK   = 0.56

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

# CORS (allow mobile/web apps during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("CORS_ALLOW_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Simple API-key middleware. Set `API_KEY` env var to enable; if not set, middleware is a no-op."""
    api_key = os.environ.get("API_KEY", "")
    # allow if no API_KEY configured
    if not api_key:
        return await call_next(request)

    # Whitelist health and docs endpoints
    path = request.url.path
    whitelist = ["/health", "/openapi.json", "/docs", "/redoc"]
    if any(path.startswith(p) for p in whitelist) or request.method == "OPTIONS":
        return await call_next(request)

    header_key = request.headers.get("x-api-key", "")
    if header_key != api_key:
        return JSONResponse({"detail": "Unauthorized - invalid API key"}, status_code=401)

    return await call_next(request)


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
    client, emb = _state["retriever"], _state["embedder"]
    if client is None:
        return [], "retrieval_unavailable", 0.0
    
    if not intents or len(intents) <= 1:
        q_vec = emb.encode(query, normalize_embeddings=True).tolist()
        hits  = client.query_points(collection_name=COLLECTION_NAME,
                                    query=q_vec, limit=TOP_K, with_payload=True).points
    else:
        # Multi-Query Retrieval
        hits = []
        limit_per_intent = max(2, TOP_K // len(intents))
        for intent in intents:
            sub_query = f"{intent.replace('_', ' ')}: {query}"
            q_vec = emb.encode(sub_query, normalize_embeddings=True).tolist()
            sub_hits = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=limit_per_intent, with_payload=True).points
            hits.extend(sub_hits)
            
        seen = set()
        unique_hits = []
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                unique_hits.append(h)
        hits = sorted(unique_hits, key=lambda x: x.score, reverse=True)[:TOP_K]

    if not hits:
        return [], "abstain", 0.0
    score = hits[0].score
    tier  = ("grounded" if score >= TIER_GROUNDED
             else "fallback" if score >= TIER_FALLBACK else "abstain")
    return hits, tier, score


def _generate(query: str, hits: list) -> str:
    """Generate an answer using either a managed LLM adapter (if configured)
    or the in-process generator when available.
    """
    # Try managed LLM first (if configured via LLM_API_URL)
    try:
        from backend.adapters import llm_adapter
        llm_url = os.environ.get("LLM_API_URL", "")
        if llm_url:
            try:
                return llm_adapter.safe_generate(query, hits)
            except Exception as e:
                print(f"Managed LLM call failed: {e}")
    except Exception:
        # adapter unavailable — fall through to in-process generator
        pass

    # Fallback to in-process generator (if loaded)
    tok, mdl = _state.get("gen_tok"), _state.get("gen_mdl")
    if tok is None or mdl is None:
        return "[generator not loaded — set SKIP_GENERATOR=0 and restart, or configure LLM_API_URL]"

    ctx = "\n\n".join(
        f"[{i+1}] {h.payload.get('text','')[:300]}" for i, h in enumerate(hits)
    )
    prompt = (
        "You are a helpful agricultural advisor for Indian farmers.\n"
        "Answer using ONLY the provided context. You MUST cite your sources as [1], [2] at the end of every sentence.\n"
        "If the question contains multiple topics, structure your answer to address each topic individually.\n\n"
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
    
    # 1. Run ViT
    vit = _state["vit"]
    if vit is None:
        raise HTTPException(503, "ViT model not loaded")

    from PIL import Image
    img_bytes = await file.read()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result    = vit.predict(pil_img)

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
    
    # 2. Run IEG on text
    text = text.strip()
    intents, blocked = _ieg_run(text)
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
        "top_score":  round(score, 4),
        "answer":     answer,
        "sources":    sources,
        "latency_ms": round((time.time() - t0) * 1000),
    })
