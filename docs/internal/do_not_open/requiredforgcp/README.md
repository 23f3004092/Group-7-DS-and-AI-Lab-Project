# requiredforgcp — deploy files

Everything needed to run FarmerVision on one GCP GPU VM. **Read
`../GCP_DEPLOYMENT_PLAN.md` first** — it explains each step in plain language.

## Run order

| # | On your PC | On the VM |
|---|---|---|
| 0 | `cp .env.example .env` and fill it in | |
| 1 | `bash 01_gcloud_setup.sh` (then request GPU quota + set a budget in the console) | |
| 2 | `bash 02_upload_artifacts.sh` | |
| 3 | `bash 03_create_vm.sh` | |
| 4 | | `cd ~/farmervision && bash 04_vm_setup.sh` |
| 5 | test: SSH tunnel, then POST `/query` | |
| — | `bash stop_vm.sh` when done (**do this every time**) | |

## Scope

This gateway serves **vision + intent/entity/guardrail + RAG (vector DB + generation)**.
**Mandi prices, weather, and yield are handled by your main app** and injected into
`/query` as `live_data` — the LLM weaves them into the grounded answer.

## The API (what your app calls)

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | — (no auth) | GPU, point count, modules loaded |
| POST | `/classify` | `{query}` | intent + entities + guardrail + `suggested_external` |
| POST | `/query` | `{query, intent?, filters?, live_data?, skip_retrieval?}` | grounded cited answer + tier |
| POST | `/diagnose` | image file + optional `question` | disease + grounded advice |
| POST | `/vision` | image file | disease label + confidence + top-5 |

All POSTs need header `X-API-Key: <API_KEY>`.

**Typical orchestration in your app:**
1. `POST /classify` → read `suggested_external` (e.g. `["mandi_prices"]`) + intent.
2. Fetch mandi/weather/yield from **your own** APIs/model.
3. `POST /query` with `live_data: {"mandi_prices": "...", "weather": "...", "yield": "..."}`
   → the answer uses those exact values + cited RAG context.

(You can also skip step 1 and pass `live_data` directly if your app already knows what it needs.
For a pure weather/mandi question with no advisory lookup, set `skip_retrieval: true`.)

## Already faithful to your notebooks (wired, not stubbed)

- **retrieval** — exact port of `search_agri_knowledge` (tiers, fusion weights, filters
  read from `manifest.json`), FastEmbed CPU embedder.
- **generation** — Gemma-3-4B 4-bit; prefers a **merged model** (adapter folded into the
  weights → fast inference, per M5 §12/§15), falls back to base + LoRA. Unified system
  prompt handles RAG answers, photo-diagnosis advice, **and injected live_data**
  (mandi/weather/yield), language-matched + cited.
- **vision** — `Net(backbone+head)` ViT-S/16, CenterCrop(224)+Normalize(0.5), classes
  from `label_to_idx.json` (vit-train-01.ipynb).

Model sourcing (via `upload_artifacts_from_colab.py`): **IEG + Vision** come from
**kagglehub** (same datasets as `scripts/run_e2e_eval.py`); **snapshot + manifest** from
**Drive**; **generator** is the adapter **merged** into base Gemma.

## What you may still want to fill in

- **`app/ieg.py`** — paste your 3-head DistilBERT class from notebook 11 where marked.
  Until then the guardrail runs **rules-only** + keyword intent (RAG path works fully).
  The `external_hints()` keyword patterns are simple — tune them if routing misses cases.
- **`app/generation.py`** — swap `SYSTEM_PROMPT` for your exact distillation prompt if
  you want byte-identical output. `live_data` keys render with friendly labels in `_LIVE_LABELS`.
- **`app/retrieval.py`** — replace `canon_crop`/`canon_district` with the real maps if
  you use crop/district filters.

## Notes
- Qdrant and the gateway both bind to **localhost only**. Reach the API through the
  SSH tunnel (plan Step 9). Don't open ports publicly unless you must.
- The gateway uses **1 uvicorn worker** on purpose (models load once, big in RAM/VRAM).
- HF model cache is a Docker volume, so restarts don't re-download Gemma.
