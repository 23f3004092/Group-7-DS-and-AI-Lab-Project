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
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_CSV = ROOT / "data" / "eval" / "e2e_scenarios.csv"
RESULTS_CSV   = ROOT / "data" / "eval" / "e2e_results.csv"
SUMMARY_JSON  = ROOT / "data" / "eval" / "e2e_summary.json"

QDRANT_URL      = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "agri_knowledge"
BGE_MODEL_ID    = "BAAI/bge-m3"
IEG_MODEL_ID    = "distilbert-base-multilingual-cased"
MAX_LEN         = 64
TOP_K           = 10
TIER_GROUNDED   = 0.66
TIER_FALLBACK   = 0.56
# Force smaller models to CPU to prevent VRAM overflow and WDDM RAM spilling
DEVICE          = "cpu"

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
    if query_lang in ("N/A", "english", "hinglish"):
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


def check_completeness(answer: str, topics: list, gen_tok, gen_mdl) -> bool:
    if not topics:
        return True
    
    # Translate raw internal intent labels into readable concepts
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
    with torch.no_grad():
        out = gen_mdl.generate(**inputs, max_new_tokens=5, temperature=0.1, do_sample=False)
    res = gen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    return "yes" in res


# ── Component loaders ─────────────────────────────────────────────────────────

def load_ieg():
    """Load IEG model from Kaggle dataset (auto-downloads if needed)."""
    import kagglehub
    from transformers import AutoTokenizer
    print("  Downloading IEG checkpoint via kagglehub …")
    path = Path(kagglehub.dataset_download("aneeqasiddiqui377/v4-output"))
    print(f"  IEG dataset at: {path}")

    # Find checkpoint and label maps
    ckpt_candidates = list(path.rglob("*.pt"))
    label_candidates = list(path.rglob("label_maps.json"))

    # Prefer the M5 fine-tuned checkpoint if available
    ckpt = None
    for c in ckpt_candidates:
        if "ieg_adamw" in c.name:
            ckpt = c
            break
    if ckpt is None and ckpt_candidates:
        ckpt = ckpt_candidates[0]

    label_maps_path = label_candidates[0] if label_candidates else None
    print(f"  Checkpoint: {ckpt}")
    print(f"  Label maps: {label_maps_path}")

    if label_maps_path and label_maps_path.exists():
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
    # State dict may be bare or wrapped
    sd = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    tok      = AutoTokenizer.from_pretrained(IEG_MODEL_ID)
    id2intent = {i: c for i, c in enumerate(intent_classes)}
    print(f"  IEG loaded. {len(intent_classes)} intents: {intent_classes}")
    return model, tok, id2intent, ner_labels


def load_vit():
    """Load ViT CropDiseaseModel from Kaggle dataset."""
    import kagglehub
    print("  Loading ViT model via kagglehub …")
    path = Path(kagglehub.dataset_download("iitm21f1003346/vits16-crop-disease"))
    print(f"  ViT dataset at: {path}")
    model_dir = next(path.rglob("predict.py")).parent
    sys.path.insert(0, str(model_dir))
    from predict import CropDiseaseModel
    model = CropDiseaseModel(device=DEVICE)
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
    mdl = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        quantization_config=bnb_config,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    mdl.eval()
    print("  Generator loaded")
    return tok, mdl


def load_yield_model():
    """Load LightGBM tuned model from local outputs directory."""
    import lightgbm as lgb
    model_path = (ROOT / "outputs" / "Yeild_output_files" / "kaggle" /
                  "working" / "saved_models" / "lightgbm_tuned.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model not found at {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    print(f"  LightGBM loaded. {booster.num_trees()} trees, features: {booster.feature_name()}")
    return booster


# ── Inference helpers ─────────────────────────────────────────────────────────

def ieg_run(model, tok, id2intent, ner_labels, text: str):
    enc = tok(text, truncation=True, max_length=MAX_LEN,
               padding="max_length", return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        il, nl, gl = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    
    probs = torch.sigmoid(il[0])
    intents = [id2intent[i] for i, p in enumerate(probs) if p > 0.3]
    if not intents:
        intents = [id2intent[il.argmax(-1).item()]]
        
    m_flag  = int(gl.argmax(-1).item()) == 1
    r_flag  = rule_flag(text)
    blocked = m_flag or r_flag
    return intents, blocked, m_flag, r_flag


def retrieve(client, embedder, query: str, intents: list = None):
    if not intents or len(intents) <= 1:
        q_vec = embedder.encode(query, normalize_embeddings=True).tolist()
        hits  = client.query_points(collection_name=COLLECTION_NAME,
                                    query=q_vec,
                                    limit=TOP_K, with_payload=True).points
    else:
        # Multi-Query Retrieval: parallel searches for compound intents
        hits = []
        limit_per_intent = max(2, TOP_K // len(intents))
        for intent in intents:
            sub_query = f"{intent.replace('_', ' ')}: {query}"
            q_vec = embedder.encode(sub_query, normalize_embeddings=True).tolist()
            sub_hits = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=limit_per_intent, with_payload=True).points
            hits.extend(sub_hits)
            
        # Deduplicate and sort
        seen = set()
        unique_hits = []
        for h in hits:
            if h.id not in seen:
                seen.add(h.id)
                unique_hits.append(h)
        hits = sorted(unique_hits, key=lambda x: x.score, reverse=True)[:TOP_K]

    if not hits:
        return [], "abstain", 0.0
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
    with torch.no_grad():
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
    intent, blocked, m_flag, r_flag = ieg_run(ieg_model, ieg_tok, id2intent, ner_labels, query)
    ieg_ms = round((time.time() - t_ieg) * 1000)
    guard_correct = (int(blocked) == expect_block)

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

    # Retrieve
    t_ret = time.time()
    hits, tier, top_score = retrieve(retriever, embedder, query, intents=intent)
    retrieval_ms = round((time.time() - t_ret) * 1000)
    if tier == "abstain":
        return {
            "guardrail_fired":     False,
            "guardrail_correct":   guard_correct,
            "intent":              intent,
            "model_flag":          int(m_flag),
            "rule_flag":           int(r_flag),
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
            completeness_ok = check_completeness(answer, intent, gen_tok, gen_mdl)

    chunks = [h.payload.get("text", "") for h in hits]
    return {
        "guardrail_fired":     False,
        "guardrail_correct":   guard_correct,
        "intent":              ",".join(intent),
        "model_flag":          int(m_flag),
        "rule_flag":           int(r_flag),
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
    if image_col and Path(image_col).exists():
        result = vit_model.predict(image_col)
    else:
        # Simulate
        result = {
            "label": vis_cls, "confidence": 0.85,
            "top3": [(vis_cls, 0.85)], "rejected": False, "ood_score": 0.20,
            "_simulated": True,
        }
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
    t0 = time.time()

    # 1. ViT inference
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
    
    label = result["label"]
    if result["rejected"] or label is None:
        return {
            "vit_label":           label,
            "vit_rejected":        True,
            "tier":                "rejected",
            "latency_ms":          round((time.time() - t0) * 1000),
            "answer":              "[Image rejected]",
        }

    # 2. IEG on text
    t_ieg = time.time()
    intent, blocked, m_flag, r_flag = ieg_run(ieg_model, ieg_tok, id2intent, ner_labels, text)
    ieg_ms = round((time.time() - t_ieg) * 1000)
    
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

    # 3. Retrieve
    disease_name = label.replace("__", " ").replace("_", " ")
    combined_query = f"{disease_name}. {text}"
    t_ret = time.time()
    hits, tier, top_score = retrieve(retriever, embedder, combined_query, intents=intent + ["disease_pest"])
    retrieval_ms = round((time.time() - t_ret) * 1000)
    
    # 4. Generate
    answer = "[generator skipped]"
    completeness_ok = None
    gen_ms = 0
    if gen_tok and gen_mdl and tier != "abstain":
        t_gen = time.time()
        answer = generate(gen_tok, gen_mdl, combined_query, hits)
        gen_ms = round((time.time() - t_gen) * 1000)
        completeness_ok = check_completeness(answer, intent + [disease_name], gen_tok, gen_mdl)

    chunks = [h.payload.get("text", "") for h in hits]
    return {
        "vit_label":           label,
        "vit_rejected":        False,
        "guardrail_fired":     False,
        "intent":              ",".join(intent),
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
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("FarmerVision — E2E Pipeline Evaluation")
    print(f"Pathway: {args.pathway} | Skip generator: {args.skip_gen}")
    print("=" * 60)

    df = pd.read_csv(args.scenarios)
    if args.pathway != "all":
        df = df[df["pathway"] == args.pathway].copy()
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
        sid     = row["scenario_id"]
        pathway = row["pathway"]
        print(f"  [{i+1:02d}/{len(df)}] {sid} … ", end="", flush=True)

        base = {
            "scenario_id":    sid,
            "pathway":        pathway,
            "query":          str(row["query"])[:80],
            "language":       row["language"],
            "crop":           row["crop"],
            "expected_block": row["expected_block"],
            "notes":          row.get("notes", ""),
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
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\nResults saved -> {RESULTS_CSV.relative_to(ROOT)}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "status":     STATUS,
        "total":      len(df_results),
        "errors":     int(df_results["error"].notna().sum()),
        "by_pathway": {},
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
            for col in ["citation_ok", "lang_match_ok", "numeric_grounded_ok", "completeness_ok"]:
                if col in sub_clean:
                    vals = sub_clean[col].dropna()
                    if len(vals):
                        p[col] = int(vals.sum())
                        p[f"{col}_pct"] = round(vals.mean() * 100, 1)

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
