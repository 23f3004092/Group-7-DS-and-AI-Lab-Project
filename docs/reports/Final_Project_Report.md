# FarmerVision — Final Project Report

**Course:** DS & AI Lab · **Group:** 7 · **Institution:** IIT Madras
**Prepared:** 13 August 2026 · **Status:** Milestones 1–6 complete



---

## Abstract

FarmerVision is a multimodal AI advisory platform that answers Indian farmers' agricultural
questions — by text (English/Hindi/Hinglish) or leaf photo — instantly and 24/7. It unifies four
capabilities no prior system combines: crop-disease detection from images (fine-tuned ViT-S/16),
intent/entity/off-domain routing (multilingual DistilBERT, three heads + rules), retrieval-augmented
generation grounded in 723K Kisan Call Centre (KCC) and government chunks (BGE-M3 + Qdrant, answered
by a distilled 4-bit Gemma-3-4B with citations), and yield estimation (tuned LightGBM), plus live
mandi prices and weather. Every module met its held-out target (vision 0.8998 accuracy; intent 0.884;
NER F1 0.958; retrieval P@5 0.725; generation numeric-grounding 1.000 / language-match 0.991; yield
R² 0.9572), the assembled pipeline passed an 83-scenario end-to-end evaluation with zero hard
failures, and the system is deployed as a public REST API (GCP GPU VM) plus an Expo mobile app.

---

## 1. Introduction

Indian farmers face urgent queries on crop diseases, pests, fertiliser use, mandi prices, and
government schemes. The Kisan Call Centre — the primary support channel — is human-staffed, limited
in hours, and cannot scale; existing automated systems (e.g. KisanQRS) cluster past answers and
explicitly filter out price queries, leaving much unmet. **FarmerVision** closes this gap with a
single unified assistant that classifies intent, detects disease from photos, retrieves from
authoritative sources, and generates grounded, plain-language answers with market and weather
context. This report documents the full arc: problem, data, models, training, evaluation, and the
Milestone-6 deployment and documentation.

## 2. Literature Review (Milestone 1)

Prior work spans (i) agricultural QA/retrieval systems built on KCC data, which are largely
keyword/cluster based and omit price and image modalities; (ii) plant-disease vision models, which
report high lab accuracy but drop sharply in the field; and (iii) retrieval-augmented generation for
domain QA. FarmerVision's contribution is the **integration** — multi-source retrieval, visual
diagnosis, and intent routing behind one grounded, multilingual assistant with explicit safety
(numeric grounding, off-domain guardrail). Full survey: `docs/reports/Milestone_1_Report.md`.

## 3. Dataset & Methodology (Milestones 2–3)

**Datasets** (details + licences in `docs/licenses.md`): KCC UP queries 2020–25 (716,303 retrieval
chunks; 28,772/3,597/3,597 classification split), a 20-class rice+wheat leaf-disease corpus
(12,823 images; 10,252/1,284/1,287), ICAR/government advisory PDFs (7,136 chunks), crop-production
records for yield (426,803 cleaned rows; 308,364/54,418/64,021), and live Agmarknet + Open-Meteo APIs.

**Methodology:** text cleaned/chunked and embedded with BGE-M3 (1024-d) into Qdrant with rich payload
metadata for filtered retrieval; images scene-grouped to prevent contamination with shortcut-
destroying augmentation; classification labels from weak supervision + authored rules plus generated
non-agri rows; yield features are native-categorical + numeric on the original-scale target. The
end-to-end architecture (input → intent/guardrail → retrieve → generate, with vision and yield
branches) was set up and validated in Milestone 3. See `docs/reports/Milestone_{2,3}_Report_Updated.md`.

## 4. Model Development & Hyperparameter Tuning (Milestone 4)

| Model | Architecture | Notable HPT outcome |
|---|---|---|
| Vision | ViT-S/16 augreg (21.67M) + head, 20 classes | native (0.5) normalisation beat ImageNet/dataset stats; top-3-block unfreeze captured nearly all gain |
| Intent/entity/guardrail | Multilingual DistilBERT (134.7M), 3 heads | AdamW 3e-5, batch 32, 5 epochs won a 6-config search |
| Generator | Gemma-3-4B 4-bit + distilled LoRA (r=32, α=64) | 27-config QLoRA sweep vs a measured 0.0074 noise floor; regularisation was decoration |
| Yield | LightGBM (202 trees, depth 12) | RandomizedSearchCV; log-target transform rejected (worsened RMSE/R²) |

Training ran entirely on free-tier GPUs (Kaggle/Colab), which motivated 4-bit generation and compact
studies. Full detail: `docs/reports/Milestone_4_Report_Updated.md`.

## 5. Evaluation & Analysis (Milestone 5)

| Module | Headline metric | Result |
|---|---|---|
| Vision | 20-way macro-F1 / accuracy | 0.8671 / **0.8998** (n=1,287) |
| Intent / NER / Guardrail | acc / entity-F1 / adversarial recall | 0.884 / 0.958 / **1.000** (model+rules) |
| Retrieval | Precision@5 / Recall@5 | **0.725** / 0.498 (human-judged) |
| Generation | numeric grounding / language match | **1.000** / 0.991 |
| Yield | R² / RMSE | **0.9572** / 2.4112 |
| End-to-end | hard failures / 83 scenarios | **0** |

Analytical highlights: a fair one-shot baseline reframed the distillation "gain" as a **safety**
gain (numeric grounding + language match), not fluency; a perfect guardrail test score masked 0.571
adversarial recall, fixed by deploying model+rules; retrieval precision doubled once a human replaced
the automatic scorer. Standing limitations (single-annotator retrieval set, lab-only vision, no
user-centred study yet) are documented in `docs/reports/Milestone_5_Report_Updated.md`.

## 6. Deployment & Documentation (Milestone 6)

**Deployment.** The distilled models are deployed as a **REST inference API** on a **GCP GPU VM**
(`g2-standard-8` + NVIDIA L4): a FastAPI gateway (Docker, `:8000`) serving `/query`, `/classify`,
`/vision`, `/diagnose`, backed by Qdrant (723,439 vectors) and the merged 4-bit Gemma on the GPU,
with BGE-M3/ViT/guardrail on CPU. The product layer — a FastAPI backend (`/api/*`), an Expo mobile
app, and an admin dashboard — adds live mandi prices (Agmarknet), weather (Open-Meteo), yield, and an
MCP server, with graceful fallbacks (local retrieval, MSP prices, static weather). A Colab notebook
provides a free GPU demo fallback. Key engineering choices: adapter merged for fast inference,
serving contract read from `manifest.json`, Qdrant HNSW server mode, forced reply-language matching,
and multi-turn support (`session_id`/`history`). The deployment journey and all fixes (free-tier GPU
upgrade, GPU stockout zone-sweep, image/Docker/PEP-668 issues, chat-template fix, firewall
permissions) are logged in `docs/internal/do_not_open/Deployment_Report_GCP.md`.

**Documentation.** A complete `docs/` set was produced: `overview.md`, `technical_doc.md`
(10-section reproducibility guide), `user_guide.md`, `api_doc.md`, and `licenses.md`, plus an
operational runbook and a standalone shareable `API_SPEC.md`.

## 7. Conclusion & Future Work

FarmerVision progressed from research notebooks to a working, documented, publicly reachable product
that answers farmers' questions across text, image, market, and weather modalities — grounded, cited,
multilingual, and safe. All six milestones are complete.

**Future work:** collect in-field leaf photos and calibrate a vision abstention threshold; run a
user-centred/expert evaluation of generated advice; wire the trained DistilBERT guardrail head
alongside the rules; add HTTPS/TLS and a static IP (or a managed cluster) for production hardening;
reduce over-refusal in the distilled generator; and expand the retrieval evaluation beyond a single
annotator. Maintenance and retraining pointers are in `docs/technical_doc.md` (§10) and the runbook.

## 8. References & Appendix

- Milestone reports: `docs/reports/Milestone_{1..5}_Report_Updated.md`.
- Deployment report + runbook + API spec: `docs/internal/do_not_open/`.
- Model/dataset citations: `docs/licenses.md`.
- Architecture diagrams: `docs/architecture/`.
- Team contributions: `docs/TEAM_CONTRIBUTIONS.md`.

*Appendix A — Metric tables, per-class vision results, confusion matrices, and hyperparameter
searches are in the respective Milestone-4/5 reports.*

---

## Team Review and Sign-Off


| # | Team Member | Approved | Date |
|:-:|-------------|:--------:|:----:|
| 1 | Mahesh | Yes | 13 Aug 2026 |
| 2 | Harliv | Yes | 13 Aug 2026 |
| 3 | Lokesh | Yes | 13 Aug 2026 |
| 4 | Aneeqa | Yes | 13 Aug 2026 |
| 5 | Tanmay | Yes | 13 Aug 2026 |

**Document:** FarmerVision — Milestone 5 Report · **Prepared:** 13 August 2026
