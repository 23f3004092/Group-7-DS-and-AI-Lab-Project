# FarmerVision — Overview

*Milestone 6 · Deployment & Documentation · Overview (Section A)*

## Purpose

Indian farmers face urgent, time-sensitive questions about crop diseases, pests, fertiliser use,
mandi (market) prices, and government schemes. India's Kisan Call Centre (KCC) — the primary
support channel — is human-staffed, limited in hours, and cannot scale. **FarmerVision** is a
multimodal AI advisory platform that answers a farmer's typed question or leaf photo instantly,
24/7, in the farmer's own language (English / Hindi / Hinglish), grounded in a real agricultural
knowledge base with citations — and adds live mandi prices, weather, and yield estimates for the
farmer's location.

FarmerVision combines four capabilities that no existing system unifies:
- **Crop-disease detection** from a leaf image (fine-tuned ViT-S/16).
- **Intent / entity / guardrail** routing (multilingual DistilBERT, 3 heads + rules).
- **RAG advisory** — retrieval over 723K KCC + government chunks, answered by a distilled 4-bit
  Gemma-3-4B with citations and numeric-grounding safety.
- **Yield estimation** (tuned LightGBM), plus live **mandi prices** and **weather**.

## Architecture Summary

FarmerVision is a three-tier system: a mobile app + admin dashboard (frontend), a FastAPI backend
(product API + external-data integrations), and a GPU-hosted model-inference layer (the distilled
research models) backed by a Qdrant vector database.

```
   ┌──────────────┐    ┌──────────────┐
   │ Expo mobile  │    │ Admin web    │        FRONTEND
   │ app (RN)     │    │ dashboard    │
   └──────┬───────┘    └──────┬───────┘
          │  REST / JSON      │
          ▼                   ▼
   ┌────────────────────────────────────────────┐
   │  FastAPI product backend  (:8000, /api/*)   │  PRODUCT / ORCHESTRATION
   │   query · mandi · weather · admin · mcp      │
   └───┬───────────────┬───────────────┬──────────┘
       │               │               │
       ▼               ▼               ▼
  ┌──────────┐   ┌────────────┐   ┌────────────────────────┐
  │ External │   │ SQLite     │   │ Model inference (GPU)   │  ML CORE
  │ data     │   │ logs/      │   │  • RAG generation       │
  │ • mandi  │   │ feedback   │   │  • leaf diagnosis (ViT) │
  │ • weather│   └────────────┘   │  • intent/guardrail     │
  └──────────┘                    │  • yield (LightGBM)     │
                                  └───────────┬─────────────┘
                                              ▼
                                   ┌────────────────────────┐
                                   │ Qdrant vector DB        │
                                   │ agri_knowledge (723K)   │
                                   └────────────────────────┘
```

Reference diagrams: [`docs/architecture/FarmerVisionServiceArch.png`](architecture/FarmerVisionServiceArch.png),
[`docs/architecture/updated_architecture.png`](architecture/updated_architecture.png),
[`docs/architecture/request_flow_m3.png`](architecture/request_flow_m3.png).

**Data flow (per query):** input (text / image / location) → the backend classifies intent and
applies the guardrail → retrieves grounding chunks from Qdrant (with contextualization for
follow-ups) and/or fetches live data → the distilled LLM writes a cited, language-matched answer
→ response returned to the app, and the interaction logged.

## Deployed Components (what's live and where)

| Component | Tech | Where it runs |
|---|---|---|
| **Mobile app** | Expo / React Native (5 tabs) | user devices (Expo Go / build) — `mobile/` (`tanmay` branch) |
| **Admin dashboard** | React / Vite | local / web — `admin/` (`tanmay` branch) |
| **Product backend API** | FastAPI (`/api/*`) | local or a cloud instance — `backend/` (`tanmay` branch) |
| **Model-inference API** | FastAPI gateway, distilled Gemma-3-4B (GPU) + BGE-M3 + ViT-S/16 | **GCP GPU VM** (`g2-standard-8` + NVIDIA L4) — this repo, `docs/internal/do_not_open/requiredforgcp/` |
| **Vector database** | Qdrant v1.19.0, `agri_knowledge`, 723,439 vectors | Docker (co-located with the inference API) |
| **External data** | data.gov.in (mandi), Open-Meteo (weather) | third-party APIs, proxied by the backend |
| **Colab demo (fallback)** | notebook + cloudflared tunnel | Google Colab (free GPU) — `notebooks/19_farmervision_serve_colab.ipynb` |

The **research models** (Milestones 4–5) are deployed as a GPU-hosted inference API on GCP; the
**product backend** (mobile-facing) adds mandi/weather/MCP and orchestrates retrieval + generation,
with graceful local fallbacks so the app stays usable even when a dependency is down.

> Detailed reproducibility, model, and deployment documentation: [`technical_doc.md`](technical_doc.md).
> End-user instructions: [`user_guide.md`](user_guide.md). API contract: [`api_doc.md`](api_doc.md).
