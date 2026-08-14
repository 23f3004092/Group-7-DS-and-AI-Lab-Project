# FarmerVision — Technical Documentation

*Milestone 6 · Section B · Audience: developers / evaluators · Goal: reproducibility & maintainability*

This document explains how FarmerVision works, what's inside, and how to reproduce or extend it.
For the end-user guide see [`user_guide.md`](user_guide.md); for the API contract see
[`api_doc.md`](api_doc.md); for licences see [`licenses.md`](licenses.md).

---

## 1. Environment Setup

### 1.1 Requirements

| Layer | Requirement |
|---|---|
| Python | 3.11 (3.10–3.12 supported) |
| ML training / inference | see repo `requirements.txt` (torch 2.x + cu12x, transformers ≥ 4.44, peft, bitsandbytes, sentence-transformers / fastembed, qdrant-client, timm, lightgbm, scikit-learn) |
| Product backend | FastAPI, uvicorn, SQLModel/SQLAlchemy, httpx, pydantic (`backend/requirements.txt`, `tanmay` branch) |
| Vector DB | Qdrant v1.19.0 (Docker image `qdrant/qdrant:v1.19.0`) |
| Mobile | Node 18+, Expo SDK; `expo-location`, `react-i18next`, `expo-sqlite` |
| Hardware | **Training/serving the LLM:** 1 GPU (≥16 GB; NVIDIA T4/L4 used). **Everything else (retrieval, ViT, guardrail, yield, backend):** CPU only |

### 1.2 Install — research / model code (this repo)
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1.3 Install — product backend (`tanmay` branch)
```bash
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # set GEMINI_API_KEY, MANDI_API_KEY, WEATHER_API_KEY
python run.py                 # serves on :8000 with auto-reload
```

### 1.4 Vector DB + Mobile
```bash
docker-compose up -d qdrant                 # or: docker run -p 6333:6333 qdrant/qdrant
cd mobile && npm install && npx expo start  # Expo mobile app
```

Secrets live only in a **gitignored `.env`** and are never shipped to clients — all external calls
are proxied through the backend.

---

## 2. Data Pipeline

### 2.1 Datasets

| Dataset | Purpose | Source | Size | Licence |
|---|---|---|---|---|
| KCC Query Dataset (UP, 2020–25) | RAG knowledge base + intent/entity/guardrail training | data.gov.in | 716,303 chunks; 28,772/3,597/3,597 (train/val/test) classification rows | Govt. Open Data (data.gov.in) |
| Rice + Wheat leaf-disease corpus | ViT fine-tuning (20 classes) | Kaggle / curated | 12,823 images — 10,252/1,284/1,287 | Kaggle dataset terms |
| ICAR & govt advisory PDFs | scheme/advisory RAG | icar.org.in / govt | 7,136 chunks | Govt. publications |
| Crop production / yield records | yield prediction | Govt. agri-production | 426,803 cleaned rows — 308,364/54,418/64,021 | Govt. Open Data |
| Agmarknet (data.gov.in) | live mandi prices | data.gov.in | real-time API | Govt. Open Data |
| Open-Meteo | live weather | open-meteo.com | real-time API | CC-BY 4.0 (free) |

Full licensing in [`licenses.md`](licenses.md).

### 2.2 Preprocessing & feature extraction (Milestones 2–3)

- **Text / RAG:** KCC records and PDFs cleaned, deduplicated, and chunked; each chunk embedded with
  **BGE-M3** (1024-dim, no prefixes) and indexed in Qdrant with payload metadata (source_type,
  crop, district, season, query_type, doc_category, year, has_table) for filtered retrieval.
- **Vision:** images letterbox-audited and split scene-grouped to avoid train/test contamination;
  augmentation randomises exposure/blur/crop to destroy brightness "shortcuts"; eval transform is
  CenterCrop(224) + Normalize(0.5) (the checkpoint's native stats).
- **Intent/entity/guardrail:** weak supervision from KCC `QueryType`, alias matching, and authored
  rules; 5,172 generated non-agri rows added for the off-domain guardrail.
- **Yield:** cleaned to remove rows where stored yield disagreed with production/area; features =
  `[crop, state, district, season, data_source, crop_type]` (native categorical) +
  `[year, area, annual_rainfall, fertilizer, pesticide]`; target = `yield` (t/ha, original scale).

Data lives under `data/` (samples + a `data/README.md`); full corpora are external (links above).

---

## 3. Model Architecture

FarmerVision routes each request to one or more of four models plus retrieval:

| Model | Architecture | Key hyperparameters |
|---|---|---|
| **Vision** | ViT-S/16 `vit_small_patch16_224.augreg_in21k_ft_in1k` (21.67M params) + linear head, 20 classes | AdamW + LLRD base 3e-4, batch 64, 12+20+18 epochs (3 phases), top-3-block unfreeze |
| **Intent/Entity/Guardrail** | Multilingual DistilBERT (134.7M), **one backbone, three heads** (intent classify, NER tag, guardrail) | AdamW 3e-5, batch 32, 5 epochs |
| **Retriever** | BAAI/BGE-M3 frozen (1024-d) + Qdrant HNSW (m=16, ef_construct=128), COSINE | top-k 5; fusion weights per intent; tiers grounded ≥0.638 / fallback ≥0.553 |
| **Generator** | `gemma-3-4b-it` 4-bit NF4 + distilled **LoRA (r=32, α=64)**, merged for serving | paged AdamW-8bit 2e-4 cosine, 4 epochs (kept epoch 2), greedy decode |
| **Yield** | LightGBM regressor | 202 estimators, 94 leaves, depth 12, lr 0.2537, `min_child_samples` 30 |

**RAG pipeline:** query → (guardrail + intent) → BGE-M3 embed → Qdrant top-k with per-intent
pdf/kcc fusion → relevance tier → distilled Gemma writes a cited answer in the query's language.
Architecture diagrams: [`docs/architecture/`](architecture/) (`request_flow_m3.png`,
`disease_detector_arch.png`, `three_tier.png`).

---

## 4. Training Summary (Milestone 4)

| Module | Compute | Duration | Optimizer / schedule | Key setting |
|---|---|---|---|---|
| Vision (3 phases) | 2× T4 | ~42 min | AdamW + LLRD, base 3e-4 | native (0.5) normalisation won |
| Intent/entity/guardrail | 1× T4 | 949 s | AdamW 3e-5, 5 epochs | one backbone, 3 heads |
| Distillation data prep | 2× T4 | 353.9 min (API rate-limited) | teacher-distilled set (2,868/200/28) | 26% refusal share |
| Generator HPT (27 configs) | 1× T4 | 364.9 min | QLoRA single-variable sweeps | noise floor 0.0074 |
| Final adapter | 1× T4 | 578.1 min (9.6 h) | rank 32, 4 epochs (kept ep 2) | val loss plateaued after ep 2 |
| Yield search | CPU | 165 fits, 14–70 s each | RandomizedSearchCV, 3-fold | original-scale target kept |

All on free-tier hardware (Kaggle/Colab), which shaped choices such as 4-bit generation. Full
detail: [`docs/reports/Milestone_4_Report_Updated.md`](reports/Milestone_4_Report_Updated.md).

---

## 5. Evaluation Summary (Milestone 5)

| Module | Metric | Result |
|---|---|---|
| Vision | wheat-15 macro-F1 (test) / 20-way / accuracy | **0.8631** / 0.8671 / 0.8998 (n=1,287) |
| Intent | accuracy / macro-F1 | **0.884** / 0.715 |
| NER | entity F1 | **0.958** |
| Guardrail | F1 (test) / adversarial recall (combined) | 1.000 / **1.000** (model + rules) |
| Retrieval | Precision@5 / Recall@5 (human-judged) | **0.725** / 0.498 (n=48 Q, 480 chunks) |
| Generation | numeric grounding / language match | **1.000** / 0.991 (219 real + 48 curated) |
| Yield | R² / RMSE | **0.9572** / 2.4112 (n=64,021) |
| End-to-end | pipeline error rate (83 scenarios) | 0% hard failures |

Key insights: baselines decide conclusions (a fair one-shot baseline turned most of the
distillation "gain" into a safety gain, not fluency); a perfect guardrail score on the standard
split hid 0.571 adversarial recall (fixed by model+rules); retrieval precision doubled when a human
replaced the automatic scorer. Full detail + caveats:
[`docs/reports/Milestone_5_Report_Updated.md`](reports/Milestone_5_Report_Updated.md).

---

## 6. Inference Pipeline

**Flow:** input → guardrail/intent → retrieve (contextualised for follow-ups) + optional live data
→ distilled LLM (language-forced, cited) → response. The vision path is image → ViT → disease →
retrieve treatment → grounded advice.

**Model-inference API call (deployed gateway):**
```bash
curl -s -X POST http://<HOST>:8000/query \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query":"wheat me yellow rust ki dawa","intent":"field_practice"}'
```
```json
{"tier":"grounded","answer":"Spray Mancozeb 75 WP ... \nSources: [1], [2]",
 "sources":[{"n":1,"score":0.71,"source_type":"kcc","citation":{...}}],
 "lang":"hinglish","top_score":0.71,"gen_ms":11840,"latency_ms":12010}
```

**Product-backend call (mobile app):**
```bash
curl -s -X POST http://<host>:8000/api/query/text \
  -H "Content-Type: application/json" -d '{"query":"best time to sow wheat in UP"}'
```

**Core retrieval + generation (simplified):**
```python
tier, best, hits = retrieve(query, intent)          # BGE-M3 embed -> Qdrant (fusion + tier)
if tier == "abstain": answer = localized_abstain(query)
else:                 answer = generate(query, hits, live_data, history)  # distilled Gemma, cited
```

---

## 7. Deployment Details

FarmerVision is deployed as **two surfaces**:

### 7.1 Model-inference API (research models) — GCP GPU VM
- **Platform:** GCP Compute Engine — `g2-standard-8` + 1× **NVIDIA L4** (24 GB), zone `us-east4-c`.
- **Hosting:** FastAPI gateway (Docker) on `:8000`; Qdrant (Docker) restored from a 3.8 GB snapshot
  (723,439 vectors); distilled Gemma (LoRA merged, 4-bit) on the GPU; BGE-M3 / ViT / guardrail on CPU.
- **Interact:** `POST /query`, `/classify`, `/vision`, `/diagnose`, `GET /health` — see
  [`api_doc.md`](api_doc.md) and the full internal spec `docs/internal/do_not_open/API_SPEC.md`.
- **Ops:** `stop_vm.sh` / `start_vm.sh` (cost control); ~$0.85/hr running, ~$10/mo stopped.
- **Fallback demo:** `notebooks/19_farmervision_serve_colab.ipynb` runs the same stack on a free
  Colab GPU behind a cloudflared tunnel.

### 7.2 Product backend + app (`tanmay` branch)
- **Backend:** FastAPI, five router groups (`/api/query`, `/api/mandi`, `/api/weather`,
  `/api/admin`, `/api/mcp`); Swagger at `/docs`; runs locally or on a cloud instance.
- **Mobile:** Expo React Native app (Home, Leaf Scanner, Advisor Chat, Yield, Settings).
- **Admin:** React/Vite dashboard consuming `/api/admin/*`.

### 7.3 Example — run the product backend + mobile
```bash
pip install -r backend/requirements.txt && cp .env.example .env   # add API keys
python run.py                                                     # backend :8000
cd mobile && npm install && npx expo start                       # scan QR in Expo Go
```

---

## 8. System Design Considerations

- **Modularity:** each capability is an independent service (`services/*.py`) behind a thin router;
  models can be swapped without touching callers. The research models and the product backend are
  decoupled — the backend orchestrates and can call the GCP inference API or cloud models per config.
- **RAG DB ↔ retriever interaction:** the retriever embeds the query with BGE-M3 and queries Qdrant
  in **server mode** (HNSW ANN, not O(n) local mode); per-intent fusion weights blend the PDF vs KCC
  corpora; a relevance **tier** (grounded/fallback/abstain) gates whether the LLM is invoked. The
  serving contract (tiers, weights, dims) is read from `manifest.json` so it always matches the index.
- **Scalability:** the GPU LLM is the bottleneck; the CPU components scale independently. The design
  supports horizontal scaling (stateless gateway + external Qdrant) and a managed cluster later.
- **Multi-turn:** two modes — server `session_id` memory or client-managed `history` (stateless).
- **Data flow / privacy:** secrets are backend-only; external calls are proxied; admin chat sessions
  are isolated per login.

---

## 9. Error Handling & Monitoring

- **Retrieval fallback:** if Qdrant is unavailable, the backend degrades to an in-memory local
  retriever so the advisory flow still works.
- **Mandi fallback:** `mandi_service.py` uses a 6 h cache and a static **MSP fallback** when
  data.gov.in is rate-limited (100 req/day) or returns no record.
- **Weather fallback:** Open-Meteo default (keyless, live) with a 30 min cache and a static card if
  it's unreachable; indianapi.in (IMD) wired behind `WEATHER_PROVIDER=indian`.
- **Generation safety:** numeric-grounding checks (no invented figures), guardrail refusal for
  off-domain/restricted queries, language-match forcing, and a rule-appended KVK disclaimer on the
  fallback tier.
- **Health & logs:** `GET /health` (both surfaces) reports component status (SQLite, vector DB,
  yield model, cloud AI, mandi, weather); interactions are logged to SQLite; the admin dashboard
  surfaces stats + logs. First request after a (re)start is slower while models warm up.

---

## 10. Reproducibility Checklist

| Item | Value |
|---|---|
| Seeds | vision: frozen split/transform on disk; intent/entity/guardrail: 42; retrieval/generation: 13 (bootstrap 2000); yield: 42 |
| Python | 3.11; deps pinned in `requirements.txt` / `backend/requirements.txt` |
| Checkpoints | vision `p3_full_best.pt` (epoch 16); IEG `intent_entity_guardrail_model.pt`; generator LoRA `best_adapter/` (→ merged); yield `lightgbm_*.txt` |
| Vector index | Qdrant `agri_knowledge` snapshot + `manifest.json` (embed_model, dims, prefixes, tiers, fusion_weights, top_k) |
| Serving config | BGE-M3 @ 1024-d, no prefixes, top-k 5, tiers abstain <0.553 / fallback 0.553–0.638 / grounded ≥0.638 |
| Notebooks | training/eval under `notebooks/` (`vit-train-01`, `11_kcc_intent_entity_guardrail`, `13_distill_training`, `14_distillation_hpt`, `08b_rag_vector_db_bge_m3`, `10c_rag_baseline_vs_distilled`, yield notebooks, `16_rag_evals`) |
| Deployment | `docs/internal/do_not_open/requiredforgcp/` (scripts + gateway app), `RUNBOOK_AND_API.md`, `Deployment_Report_GCP.md` |
| Decoding | greedy (temperature 0) for reproducibility |

To replicate the deployed inference API from artifacts: fill the GCS bucket
(`upload_artifacts_from_colab.py`), run `03_create_vm.sh` → `04_vm_setup.sh`, and verify `/health`
shows `points: 723439`. Full step-by-step: `docs/internal/do_not_open/RUNBOOK_AND_API.md` §5.
