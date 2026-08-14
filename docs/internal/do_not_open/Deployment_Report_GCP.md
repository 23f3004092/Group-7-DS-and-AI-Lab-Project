# FarmerVision — Deployment Report (GCP)

**Phase:** Production deployment of the Milestone-5 stack
**Prepared:** 14 August 2026
**Status:** ✅ Deployed and serving a public API

---

## Executive Summary

The five FarmerVision modules validated in Milestone 5 (retrieval, generation, vision,
intent/guardrail, yield) were assembled into a single **inference gateway** and deployed to
**Google Cloud Platform** on one GPU virtual machine. The system now exposes a **public REST
API** that answers Indian farmers' crop questions — grounded, cited, language-matched
(English/Hindi/Hinglish), multi-turn — plus a leaf-disease vision path.

The core design principle: **only the LLM needs a GPU; everything else runs on CPU.** This kept
the deployment to a single modest VM and inside the GCP free credit.

| Item | Value |
|---|---|
| Platform | GCP Compute Engine, 1 VM |
| Machine | `g2-standard-8` + 1× **NVIDIA L4** (24 GB), zone `us-east4-c` |
| Serving | FastAPI gateway (Docker) on port 8000; Qdrant (Docker) for the vector DB |
| Vector DB | Qdrant v1.19.0, `agri_knowledge`, **723,439 vectors** (HNSW) |
| Generator | distilled **Gemma-3-4B** (LoRA merged in), 4-bit, on the GPU |
| Embedder / Vision / Guardrail | BGE-M3, ViT-S/16, rule-based guardrail — all on CPU |
| API | `/health`, `/classify`, `/query`, `/vision`, `/diagnose` |
| Public endpoint | `http://<IP>:8000` (firewall-opened, API-key protected) |
| Running cost | ~$0.85/hr; ~$10/mo when stopped; covered by the $300 free credit |

---

## 1. Objective & Scope

Turn the assembled Milestone-5 pipeline (previously runnable only in notebooks / the
`run_e2e_eval.py` harness) into a **persistent, externally reachable service** that a future
mobile/web application can call, without the app needing any ML or infrastructure knowledge.

In scope: hosting the vector DB, the distilled LLM, the vision model, the intent/guardrail
layer, and the retrieval/generation logic behind one API. Out of scope (by design): mandi
prices, weather, and yield prediction — these are fetched by the **client app** and injected
into the prompt via a `live_data` field.

---

## 2. Architecture

```
                        Client app  ── HTTPS-ish (HTTP + API key) ──►
                              │
                              ▼
        ┌──────────────  ONE GCP VM (g2-standard-8 + L4)  ──────────────┐
        │  Gateway (FastAPI, Docker, :8000)                             │
        │    • intent + guardrail          (CPU, rules)                 │
        │    • query embedding  BGE-M3      (CPU)                        │
        │    • generation  Gemma-3-4B 4-bit (GPU)  ◄── the only GPU user │
        │    • vision  ViT-S/16             (CPU)                        │
        │                    │                                          │
        │                    ▼                                          │
        │  Qdrant (Docker, :6333)  —  agri_knowledge, 723k vectors, HNSW │
        └───────────────────────────────────────────────────────────────┘
                 Models pulled at boot from a GCS bucket (~12 GB)
```

**Rationale for the CPU/GPU split:** Milestone-5 §13 measured that co-loading every model on one
GPU saturated VRAM. Dispatching the light models (embedder, ViT, guardrail) to CPU and reserving
the GPU for the 4-bit LLM is exactly what the eval harness used, so the deployment reproduces a
proven configuration.

---

## 3. What Was Deployed

| Component | Model / tech | Placement | Notes |
|---|---|---|---|
| Vector DB | Qdrant v1.19.0 | Docker (CPU) | 723,439 chunks restored from a 3.8 GB snapshot; HNSW server mode |
| Query embedding | BAAI/BGE-M3 (1024-d) | CPU | via sentence-transformers; tiers/weights read from `manifest.json` |
| Generation | Gemma-3-4B-it, distilled LoRA **merged**, 4-bit NF4 | GPU (L4) | merge removes per-step adapter overhead (M5 §12/§15) |
| Vision | ViT-S/16 (`p3_full_best.pt`), 20 classes | CPU | leaf-disease classification |
| Intent/guardrail | rule + keyword layer | CPU | DistilBERT head not wired; rules block non-agri / restricted substances |
| Gateway | FastAPI + Uvicorn | Docker | orchestrates the pipeline; single public surface |

**API endpoints:** `/health`, `/classify` (intent + guardrail + external-data hints), `/query`
(RAG, multi-turn, live-data injection), `/vision` (photo → disease), `/diagnose` (photo →
grounded treatment). Full contract in `API_SPEC.md`.

---

## 4. Deployment Approach & Options Considered

Several hosting strategies were evaluated before committing:

| Option | Verdict | Reason |
|---|---|---|
| Laptop (local Docker) | rejected as primary | free, but no GPU → LLM too slow; not externally reachable |
| **Single GCP GPU VM** | ✅ **chosen** | only the LLM needs a GPU; simplest persistent, public setup; reuses the §13 CPU/GPU split |
| GKE (Autopilot) | rejected | Kubernetes overhead not justified at one-of-each-model scale; same GPU-quota wall |
| Colab + tunnel | kept as **fallback** | free GPU, fast to demo, but ephemeral (dies with the runtime) |

The single VM was operated **entirely from Google Cloud Shell** (browser) — no local tooling —
which suited the team's workflow and kept everything account-side.

**Model delivery:** artifacts are staged in a **GCS bucket** and pulled by the VM at setup. The
snapshot + manifest come from Google Drive; the IEG and vision models are pulled via **kagglehub**
(the same datasets `run_e2e_eval.py` uses); and the generator is the LoRA **merged into base
Gemma** and re-quantised for fast inference.

---

## 5. Infrastructure & Automation

**GCP resources**

| Resource | Detail |
|---|---|
| Project | `project-7f232935-b8f7-4aad-881` |
| VM | `farmervision`, `g2-standard-8`, 1× NVIDIA L4, 100 GB pd-balanced, image `common-cu129-ubuntu-2404-nvidia-580` |
| Zone | `us-east4-c` (selected automatically by a stock-seeking sweep) |
| Bucket | `gs://farmervision-prod-artifacts` (`us-central1`), ~12 GB of artifacts |
| Firewall | `fv-gateway` allows `tcp:8000` |

**Automation delivered** (in `docs/internal/do_not_open/requiredforgcp/`):

- `03_create_vm.sh` — provisions the GPU VM; discovers the newest CUDA image family and
  **sweeps zones/GPU types until one has stock**.
- `04_vm_setup.sh` — pulls artifacts, starts Qdrant, restores the snapshot, builds and starts the
  gateway; ends on a health check.
- `upload_artifacts_from_colab.py` — fills the bucket (merges the adapter, pulls IEG/vision via
  kagglehub, snapshot from Drive).
- `docker-compose.yml`, `Dockerfile`, `app/` — the gateway service.
- `stop_vm.sh` / `start_vm.sh` — cost control.

**Setup flow:** upload artifacts → `03_create_vm.sh` → copy `.env` → `04_vm_setup.sh` → open
firewall. First-time setup is ~15 minutes (mostly the 12 GB pull + image build).

---

## 6. Key Engineering Decisions

1. **Adapter merged into base weights.** The distilled LoRA was merged offline and re-quantised
   to 4-bit, eliminating the per-step adapter cost flagged in M5 §12 — faster generation with the
   same behaviour.
2. **Serving contract read from `manifest.json`.** Tiers (grounded ≥ 0.638 / fallback ≥ 0.553 /
   abstain), fusion weights, dims and prefixes are loaded at runtime, so they always match the
   built index.
3. **Reproducible retrieval.** Qdrant runs in **server mode** with HNSW (never the O(n) local
   mode); the snapshot is restored via Qdrant's own API.
4. **Stateless-friendly multi-turn.** Two modes: server-side `session_id` memory, or
   client-managed `history` returned in every response.

---

## 7. Enhancements Added During Deployment

Beyond lift-and-shift, several capabilities were added to make the API app-ready:

- **Language control (bug fix).** KCC chunks are stored as Hindi `question:…answer:…` records; the
  LLM was copying the chunk's language and mistaking the chunk's embedded question for the user's.
  The prompt was rewritten to (a) detect the query language (en/hi/hinglish) and **force it at the
  very end of the prompt**, and (b) state that a chunk's question is *not* the farmer's question —
  only its answer is knowledge.
- **Multi-turn conversations** — `session_id` (server-remembered) or `history` (client-managed),
  with retrieval contextualised by recent turns so follow-ups ("its dose?", "wahi") resolve.
- **Live-data injection** — the app fetches mandi prices / weather / yield and passes them in
  `live_data`; the model treats them as authoritative. `/classify` returns `suggested_external`
  hints so the app knows what to fetch.
- **Localised abstain + KVK disclaimer**, and a fixed `Sources: [n]` answer format.

---

## 8. Challenges & Resolutions

The deployment surfaced a series of GCP/OS-level issues; each and its fix is recorded here (and in
the runbook's troubleshooting table) so a rebuild is smooth.

| # | Challenge | Resolution |
|---|---|---|
| 1 | Free-trial account **cannot use GPUs** | Upgraded to a paid account (keeps the $300 credit); requested `GPUS_ALL_REGIONS` quota |
| 2 | Deep-Learning **image family names had been retired** | Discover the newest `common-cu*` family dynamically instead of hard-coding |
| 3 | **`STOCKOUT`** for T4, then L4, across multiple zones | A zone/GPU **sweep** that tries combos until one has stock (landed L4 in `us-east4-c`) |
| 4 | `04` failed: **`.env not found`** | `scp *` skips dotfiles — copy `.env` explicitly |
| 5 | **Docker not preinstalled** on the newer image | Installed Docker + NVIDIA Container Toolkit on the VM |
| 6 | Ubuntu 24.04 **PEP-668 / no pip** blocked `requests` | Installed `python3-requests` via `apt` |
| 7 | `/query` 500: **Gemma-3 `apply_chat_template` returns a dict** | Use `return_dict=True` and pass `input_ids`/`attention_mask` explicitly |
| 8 | Firewall create **denied on the VM** | Run firewall commands from **Cloud Shell** (user account), not the VM (service account) |
| 9 | LLM copying **Hindi context language** | Prompt rewrite forcing the reply language (see §7) |

---

## 9. Cost Analysis

| State | Cost | Notes |
|---|---|---|
| Running (serving) | **~$0.85 / hour** | L4 + VM + disk |
| Stopped | **~$10 / month** | 100 GB disk only; GPU billing off |
| Bucket | ~$0.30 / month | ~12 GB artifacts |
| Static IP (optional) | ~$7 / month when stopped | only if the IP is pinned for stable integration |

All usage draws from the **$300 free credit** (90 days from signup). Operating rule: **stop the VM
when idle** — at demo/integration cadence the credit lasts well beyond the project.

---

## 10. Security Posture

- **Auth:** every mutating endpoint requires an `X-API-Key` header; requests without it are 401'd
  (verified).
- **Network:** only the gateway port (8000) is exposed; Qdrant and the models are not public.
- **Current gaps (acceptable for a prototype):** plain **HTTP (no TLS)**, and the firewall is open
  to `0.0.0.0/0` (API key is the sole guard). Both are documented with hardening steps (restrict
  source IP; add a load balancer + certificate for TLS).

---

## 11. Limitations & Future Work

- **No TLS / no custom domain** — add an HTTPS load balancer or a reverse proxy for production.
- **Single instance, no HA/autoscaling** — one VM; a restart briefly interrupts service. GKE or a
  managed group would add resilience if traffic grows.
- **Ephemeral IP** — changes on stop/start unless a static IP is reserved (documented).
- **In-memory sessions** — server-side `session_id` history resets on gateway restart; the
  client-managed `history` mode avoids this.
- **Guardrail is rules-only** — the trained DistilBERT head is not wired; the RAG path is
  unaffected, but wiring it would strengthen adversarial coverage.
- **Vision is lab-trained** — outputs are presented as suggestions, consistent with M5 §14.

---

## 12. Deliverables

All under `docs/internal/do_not_open/`:

| File | Purpose |
|---|---|
| `requiredforgcp/` | all deployment scripts + the FastAPI gateway app |
| `GCP_DEPLOYMENT_PLAN.md` | single-VM deployment plan (step by step) |
| `RUN_FROM_CLOUD_SHELL.md` | laptop-free (Cloud Shell) variant |
| `RUNBOOK_AND_API.md` | operations bible: start/stop/cost, where to change things, full API, troubleshooting |
| `API_SPEC.md` | standalone, GCP-free API reference to hand to integrators |
| `requiredforgcp/upload_artifacts_from_colab.py` | fills the GCS bucket (merge + kagglehub + Drive) |
| `notebooks/19_farmervision_serve_colab.ipynb` | Colab + tunnel fallback (free, ephemeral) |

---

## 13. Current Status

Deployed and serving. `/health` reports `gpu:true` (NVIDIA L4), `points:723439`, and all modules
loaded; `/query` returns grounded, cited, language-matched answers with working multi-turn; the
public endpoint responds over the internet with API-key auth enforced. Day-to-day operation is a
two-command loop — `start_vm.sh` / `stop_vm.sh` — with the runbook covering everything else.

**The deployment phase is complete.** Recommended near-term hardening: reserve a static IP for
stable integration, and add TLS before any non-prototype use.

---

*Prepared for the FarmerVision team. Operational detail: `RUNBOOK_AND_API.md`. Integration:
`API_SPEC.md`.*
