# FarmerVision — Milestone 3 Report
## Model Architecture & End-to-End Pipeline Design

**Version:** v15 · **Prepared:** 23 July 2026 · **Revised:** 28 July 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [End-to-End Workflow](#3-end-to-end-workflow)
4. [Model Selection](#4-model-selection)
5. [Justification of Model Choices](#5-justification-of-model-choices)
6. [Model Inputs and Outputs](#6-model-inputs-and-outputs)
7. [Training Strategy](#7-training-strategy)
8. [Model Pipeline](#8-model-pipeline)
9. [Retrieval and Knowledge Components](#9-retrieval-and-knowledge-components) 
10. [Prompt Engineering](#10-prompt-engineering)
11. [System Integration](#11-system-integration)
12. [Computational Requirements](#12-computational-requirements)
13. [Design Decisions](#13-design-decisions)
14. [Risks and Limitations](#14-risks-and-limitations)
15. [Deliverables](#15-deliverables)
16. [Summary and Next Steps](#16-summary-and-next-steps)

**Appendices:** [A — Model Configuration](#appendix-a--model-configuration) · [B — Hyperparameters](#appendix-b--hyperparameter-summary) · [C — Assumptions Requiring Validation](#appendix-c--assumptions-requiring-validation) · [D — Change Log](#appendix-d--change-log)

---

## 1. Introduction

### 1.1 Project Recap
FarmerVision is a multimodal AI advisory system for farmers, primarily targeting Hindi and
regional-dialect speakers in Uttar Pradesh. Farmers interact via text, photo, or voice for three
core needs: **crop disease diagnosis** (from a photo), **government policy/scheme lookup**, and
**yield/profitability guidance**.

### 1.2 Objectives of Milestone 3
Select and justify the model(s) powering each module, and design a complete pipeline from raw farmer
input to delivered output — including retrieval, tool-use, guardrails, and generation. The target
environment assumes reliable connectivity, with an end-to-end **latency target of 200–300ms** for
the core reasoning path. See §12.3: this target is now in tension with the first measured component.

### 1.3 Architecture and Project Goals
The domain is narrower than open-domain assistance: a finite set of crops, diseases, districts, and
schemes, with numeric data that has computable ground truth. This is what makes the latency target
achievable — most of the "intelligence" lives in fast retrieval and small purpose-built models (Tier
1/2), with a single generative call reserved for turning already-validated facts into fluent
language (Tier 3). One large LLM doing retrieval, computation, and generation end-to-end would miss
the target and add hallucination risk on numeric claims.

### 1.4 Validation Status
This is a **design-phase document, with one exception.** Model choices, pipeline structure, and the
latency target rest on published benchmarks and standard practice — **not measured on this system**.
Latency figures are architectural estimates; hyperparameters are standard starting points; §5
justifications reference known model properties rather than our own experiments.

**The retrieval stack is the exception.** The embedder selection (§5.2) and retrieval components
(§9) are built and measured over a 723,439-chunk production index; those sections state observed
behaviour, and the caveats above do not apply. That component also produced the one measured result
that contradicts a claim made elsewhere in this report (§12.3) — where a built result and a design
estimate disagree, both are reported rather than reconciled prematurely. Appendix C lists
assumptions still needing validation.

### 1.5 Capstone Scope vs. Production Extension
Naming a production-grade technology in an architecture diagram is not a claim that this project is
built at production scale. This table separates what is actually built from what is documented as
the extension path.

| Component | Capstone (built & demoed) | Production extension (documented only) |
|---|---|---|
| ML mesh topology | One Python service hosting all models, in-process calls | Split into per-model microservices |
| Inter-service transport | Plain REST/JSON | gRPC over Protobuf/HTTP2 |
| Vector DB | Qdrant, single node, **server mode** (§9.3) | Qdrant cluster, read replicas |
| LLM serving | Simple batched inference (HF `transformers`, single GPU) | vLLM (continuous batching, PagedAttention) |
| Voice (ASR/TTS) | **Not built.** Text/image is the demo path; voice is designed here but out of implementation scope | Streaming ASR/TTS, sub-second perceived latency |
| Guardrails | Multi-task head on Intent/Entity Extractor + rule-based checks | + optional independent LLM second-pass |

The rest of this report describes the full target architecture, needed to justify model choices
coherently; exclusions are called out inline as final decisions, not open options.

---

## 2. System Architecture

### 2.1 High-Level Diagram

```
[Client: React/Next.js or Flutter] ── text / image / voice
        │
[Backend Core — Python FastAPI] ── auth, orchestration, trace context
        │
   ┌────┼──────────────────┬──────────────────────────────────────┐
   ▼    ▼                  ▼                                      ▼
[Redis] [Qdrant]      [Python ML Mesh: Vision (ViT-Small) │      [External APIs:
        Policy/KCC +   Embedder (bge-m3) │ Intent/Entity +        Agmarknet Mandi,
        Yield Cache    guardrail head │ Yield GBT │                Weather]
        namespaces     Profitability │ Synthesis LLM │            circuit breaker,
                       ASR/TTS — designed, not built (§1.5)]      cache-first
        │
        ▼ (async, production extension only — §1.5)
[Kafka] → [Data Lake] → [Drift Detection + Retraining]
```

Capstone substitutes on-demand EvidentlyAI batch reports for the Kafka→retraining path, runs the ML
mesh as one deployed service, and Qdrant as a single node.

### 2.2 Data Flow
Text/image/voice → (ASR if voice) → parallel vision + intent/entity extraction → fast-path cache
check → (on miss) parallel tool fan-out (retrieval, yield cache/GBT, mandi/weather, profitability) →
guardrails pre-filter → LLM synthesis → guardrails post-check → streamed response.

### 2.3 Technology Stack
| Layer | Technology | Capstone scope |
|---|---|---|
| Frontend | React/Next.js, Flutter | Built |
| Backend | Python, FastAPI | Built, single instance |
| Transport | Plain REST/JSON | Built (gRPC is production-only) |
| ML serving | Simple batched LLM inference; ONNX Runtime for small models | Built, minus vLLM/TensorRT |
| Vector DB | Qdrant (HNSW, cosine), **server mode** | Built, single node |

---

## 3. End-to-End Workflow

### 3.1 Workflow Diagram

```
[typed text | photo | spoken] ──(spoken → ASR → Dialect Normalizer)──┐
                                                                     ▼
                                        [text + optional image, entry point]
                                                     │
            ┌────────────────────────────────────────┴──────┐  (parallel, Tier 1)
     [Vision Classifier]                        [Intent/Entity Extractor]
            └────────────────────┬──────────────────────────┘
                                 ▼
            [Fast-path key: disease_label + intent_flags]
                    │                            │
                cache HIT                    cache MISS
                    │                            ▼
                    │      [Parallel fan-out: Qdrant retrieval (§9) │
                    │       Yield Cache → GBT │ Mandi/Weather │ Profitability]
                    │                            ▼
                    │      [Guardrails Pre-Filter] → [Synthesis LLM, streamed]
                    │                            → [Guardrails Post-Check]
                    └──────────────┬─────────────┘
                                   ▼
                 [Response: streamed text (+ TTS if requested)]
```

### 3.2 Error Handling and Fallbacks
| Failure | Fallback |
|---|---|
| Vector DB unavailable | Skip retrieval, synthesize from tool outputs, flag `sources: []` |
| Mandi/Weather API down | Serve last cached value, flag `stale: true` |
| LLM timeout (>6s hard cap) | Return retrieved facts without synthesized prose |
| Guardrails down | Fail-closed — never return unvalidated advice; `escalate_to_expert` |
| Retrieval below threshold | Abstain tier (§9.9) — refuse rather than answer from a weak match |
| Network drop mid-upload | Client queues locally, retries with same idempotent trace ID |

### 3.3 Storage Components
- **Qdrant** — two logical collections: Policy/KCC (bge-m3 embeddings) and Yield Cache
  (structured-feature embeddings), kept separate (§9.8).
- **Redis** — embedding cache, search result cache, external API cache, session state, fast-path
  template store.

---

## 4. Model Selection

### 4.1 Models per Module

| Module | Model | Pretrained / Custom | Params |
|---|---|---|---|
| Vision (disease classification) | ViT-Small (fallback MobileViT-XXS) | Pretrained (ImageNet) → fine-tuned | ~22–86M |
| Text embedding (retrieval) | BAAI/bge-m3 | Pretrained, frozen (§7.2) | ~568M |
| Structured feature embedding | Custom two-tower encoder | Trained from scratch | ~5–10M |
| Intent/entity + guardrails | DistilBERT-class, three-head | Pretrained → fine-tuned | ~66M |
| Yield prediction | Gradient-boosted trees | Trained from scratch | N/A |
| Profitability estimation | Rule-based composite | Non-ML | N/A |
| Synthesis LLM | Gemma, distilled 12B → 2–4B | Teacher → distilled student | 2–4B |
| Speculative draft model | Same tokenizer family — production only | Derived from §7.5 student | 350–500M |
| ASR *(not built, §1.5)* | IndicWhisper | Pretrained → optional dialect fine-tune | ~250M–1.5B |
| Dialect normalization *(not built)* | IndicTrans2 | Pretrained, distilled variant preferred | ~1.2B |
| TTS *(not built)* | VITS/FastSpeech2-class, Indic multi-speaker | Pretrained | ~30–90M |

### 4.2 Architecture Notes
- **ViT-Small**: standard Vision Transformer, 16×16 patches, 12 encoder layers, classification head
  sized to the disease-class count.
- **BAAI/bge-m3**: XLM-RoBERTa-large (24 layers, 1024 hidden), ~568M params, **1024-dim** dense
  vectors, 8192-token window, and **no query/passage prefix** (unlike the e5 family). Selected for
  two properties (§5.2): trained specifically for *retrieval* using hard negatives, which produces
  fine-grained separation *within* a domain; and able to emit sparse/lexical vectors from the same
  checkpoint — the designated fix for the Hinglish gap in §14. *Earlier versions of this report
  credited bge-m3 with MuRIL's properties (BERT-base, 768-dim, Indic-specific pretraining); those
  were incorrect — see §7.0.4.*
- **Distilled synthesis LLM**: same transformer-decoder family as the teacher, reduced
  layers/hidden size to reach 2–4B; exact config finalized in M4.
- **GBT yield model**: not neural — gradient-boosted trees, chosen because yield prediction is
  tabular, not sequential or spatial (§5.3).

### 4.3 Integration
Vision and intent/entity extraction run in parallel; outputs combine into a fast-path cache key. On
a miss, embedder, GBT, external tools, and profitability outputs assemble into one structured
context object that becomes the LLM's input — the LLM never calls these tools itself in the
low-latency path (§10).

---

## 5. Justification of Model Choices

### 5.1 Vision: ViT-Small (vs. CNNs)
Transformers generalize better than CNNs on fine-grained texture differences (early-stage fungal vs.
bacterial lesions) given sufficient fine-tuning data, and patch attention gives some
interpretability. **Rejected:** ResNet/EfficientNet — cheaper and a strong baseline, but
historically weaker on fine-grained classification without heavy augmentation. **Trade-off:** ViT
needs more data; mitigated by ImageNet-pretrained initialization.

### 5.2 Text Embedding: BAAI/bge-m3 (vs. MuRIL, multilingual-e5)

**This is the only §5 choice backed by our own controlled experiment rather than published
properties.** The original model failed in build; the evidence below is from measured runs on our
corpus (full detail in the RAG audit, §15).

**What failed first.** The initial index used `Yunika/muril-base-sentence-transformer`, chosen on
exactly the reasoning an earlier version of this section gave — MuRIL is pretrained on Indian
languages including transliterated Hindi. The build ran clean, but retrieval was unusable: a
urea-in-wheat query returned capsicum fertiliser advice at similarity 1.000; "brown spots on rice
leaves" returned a letter's signature block. Probed directly, three unrelated agricultural sentences
scored **1.000** with each other, while "car engine repair" scored **−0.879**; on a 6-document test
where every answer was obvious, it scored **1 of 6**.

The diagnosis matters more than the failure: MuRIL separated *farming from non-farming* but not
*wheat from mango*. Retrieval needs the second, and pretraining language coverage does not supply
it.

**The comparison.** Three candidates were built into identical notebooks — same chunks, chunking,
512-token cap, and tests, only the model varying — on the same 47,136-chunk corpus:

| Measure | e5-base\* | e5-large | **bge-m3** | Why it matters |
|---|---|---|---|---|
| Correct doc ranked first | 5/5 | 5/5 | 5/5 | all three pass the *basic* test |
| Average winning margin | +0.042 | +0.041 | **+0.222** | how decisively right beats wrong |
| Worst margin (Hinglish) | +0.006 | +0.010 | **+0.122** | the hardest case |
| Unrelated-doc similarity | 0.861 | 0.844 | **0.404** | lower is better; 0.86 = everything looks alike |
| In-domain vs. off-domain gap | +0.009 | +0.018 | **+0.072** | the headroom refusal needs |
| Checks passed | 8/11 | 9/11 | **10/11** | |

\* e5-base ran on a larger 107k corpus — indicative only, not like-for-like.

**The decisive criterion was safety, not accuracy.** All three rank the right document first. What
separates them is the gap between real and junk queries: e5-large 0.838 vs. 0.820 (gap 0.018);
bge-m3 0.590 vs. 0.517 (gap 0.072). This system emits **pesticide dosages** (§14), so answering
confidently from a poor match is a safety failure, not a quality one. A 0.018 gap cannot support a
reliable answer/refuse boundary; 0.072 can. **bge-m3 is the first model tested where the abstain
tier (§9.9) works at all.**

**Model size was never the variable.** e5-large has double e5-base's parameters and gained
essentially nothing (margin +0.042 → +0.041) for ~50% more inference time. What distinguishes bge-m3
is *training objective* — retrieval training with hard negatives, i.e. documents that look right and
are wrong. Learning to reject near-misses is exactly the skill in-domain separation requires.

**Trade-off accepted:** ~568M params vs. MuRIL's ~236M and 1024-dim vs. 768-dim vectors (~33% more
storage, §12.2) — both buy the only refusal margin any tested model produced. Hinglish *technical*
vocabulary remains weak (§14); the fix is bge-m3's own sparse vectors, which neither e5 model
offers.

### 5.3 Yield Prediction: GBT (vs. neural net, vs. vector k-NN)
Yield prediction is tabular regression, where gradient-boosted trees are the established baseline —
outperforming deep nets and k-NN at this data scale and serving cheaply on CPU. **Rejected:**
embedding yield outcomes into the RAG index and answering by nearest-neighbour averaging, which is
weaker than GBT for continuous targets and would force a shared embedding space with the semantic
index, degrading both. Vector retrieval is used only as a **cache in front of** GBT (§9.8).

### 5.4 Synthesis LLM: Distilled 2–4B
A 12B model cannot generate within the 150–180ms slice available to it (decode is sequential per
token; even batched serving has a throughput floor). Distilling preserves domain behaviour rather
than starting from a generic small model. **Rejected:** an off-the-shelf 2–3B instruction-tuned
model — a legitimate fallback if distillation is constrained, but expected to underperform on domain
accuracy without its own fine-tuning, so similar cost either way. **Trade-off:** narrower general
reasoning, acceptable since the model only synthesizes over already-retrieved facts (§10).

### 5.5 Vector DB: Qdrant (vs. ChromaDB, Pinecone, Weaviate)
Native HNSW performance, self-hostable (data sovereignty on government policy data), clean
multi-collection support (needed to keep Policy/KCC and Yield Cache separate), and read replicas.
**Rejected:** Pinecone/Weaviate (managed, less control over co-location/latency, recurring cost);
ChromaDB (weaker clustering/replication at scale). Qdrant must run in **server mode** — see §9.3,
where this proved a first-order latency decision, not a deployment detail.

### 5.6 ASR/TTS: Self-hosted Indic models (vs. cloud APIs)
Streaming control is required for acceptable perceived latency; cloud APIs vary in streaming support
and add per-call cost at scale. **Trade-off:** more MLOps investment. **Capstone scope:** voice is
out of scope (§1.5) — the hardest module to get working end-to-end relative to its share of the core
reasoning story. The design is retained because it is part of the target product, not because it is
still being decided.

---

## 6. Model Inputs and Outputs

| Model | Input | Preprocessing | Output |
|---|---|---|---|
| Vision Classifier | Image (JPEG/PNG) | Resize, ImageNet normalize, client-side compression | `disease_label`, `confidence`, optional bbox |
| bge-m3 Embedder | Raw text (query or chunk) | XLM-R SentencePiece; model window 8192, build capped at **512** tokens, over-length chunks split on sentence boundaries incl. the Hindi danda `।`. **No `query:`/`passage:` prefix** — correct for e5, wrong for bge-m3, and adding one degrades every vector silently | **1024-dim** float vector (cosine) |
| Structured Feature Embedder | Crop, region, soil, season, acreage | Categorical encoding + numeric normalization | 64–128-dim vector |
| Intent/Entity Extractor | Raw query text | Tokenization (backbone family) | Intent flags + NER spans + guardrail flags |
| Yield Prediction (GBT) | Crop, region, soil, rainfall, sowing date, variety | Standard tabular preprocessing | Yield estimate (+ optional quantile bands) |
| Synthesis LLM | System instructions + retrieved chunks + tool outputs + query | SentencePiece; prefix caching for shared instructions | Token sequence, capped 60–100 tokens |
| ASR *(not built)* | Raw audio | 16kHz mono PCM, streaming chunked | Transcript + per-token confidence |
| TTS *(not built)* | Text (sentence-chunked) | Sentence segmentation for streaming | Streamed audio |

---

## 7. Training Strategy

*Hyperparameters are recommended starting points per model class; final values are expected to be
tuned in Milestone 4.*

### 7.0 Dataset-to-Model Mapping (M2 → M3)

Milestone 2 identified, cleaned, and split the datasets; this milestone selects the models. This is
the explicit join between them.

| # | Model / component | M2 dataset | Prepared size | Status |
|---|---|---|---|---|
| 1 | Vision Classifier (§7.1) | Merged rice + wheat corpus | 12,859 images, 20 classes; 80/10/10 split, 0 group leakage across 11,530 pHash groups | **Ready** |
| 2 | Vision OOD eval (§7.1, §14) | PlantDoc | 2,598 images, reserved holdout | **Ready (eval-only)** |
| 3 | Text Embedder (§7.2) | Pretrained, frozen; no training set required. Optional contrastive fine-tune would use KCC pairs | 1,459,692 Q&A pairs available | **Corpus encoded; fine-tune deprioritized** |
| 4 | Retrieval index (§9) | RAG PDF Corpus + KCC UP 2020–2025 | **Built: 723,439 chunks (7,136 PDF + 716,303 KCC)** from 184 docs and 1,459,692 records | **Built and validated** — see note 2 |
| 5 | Yield GBT, UP model (§7.4) | UP district rice & wheat APY + IMD/ICRISAT covariates | 3,886 records, 74 districts; chronological split 3,256 / 296 / 444 | **Ready** |
| 6 | Yield GBT, pan-India (§7.4) | Unified multi-crop production dataset | 440,962 records, 35 states, 124 crops | **Ready** |
| 7 | Structured Feature Embedder (§7.3) | Feature columns of datasets 5–6; triplets from yield proximity | same rows | **Derivable — triplet mining is M4 work** |
| 8 | Intent/Entity Extractor (§7.9–7.10) | **No prepared dataset.** Nearest sources: KCC `QueryType` for weak intent labels, `Crop`/`District`/`Season` for distant NER. No source for guardrail flags | — | **Gap — see §7.0.3** |
| 9 | Synthesis LLM (§7.5) | KCC Q&A (real leg) + PDF corpus (grounds synthetic leg) | 1,459,692 pairs; 184 docs | **Sources ready; distillation set not built** |
| 10 | Profitability Estimator | None — rule-based by design | — | **N/A** |
| 11 | ASR / TTS / Dialect (§7.6–7.8) | None; not built for capstone (§1.5) | — | **N/A** |

**Coverage.** Six capstone components are trainable. Three (vision, both GBTs) have data ready to
use. Three (feature embedder, distillation set, optional embedder fine-tune) have raw sources ready
but need a derivation step that is M4 work, not new data collection. One — the Intent/Entity
Extractor — has no prepared dataset and is the single genuine gap. Retrieval (row 4) is past
readiness entirely: it is built.

#### 7.0.3 The Intent/Entity Extractor Data Gap
This is the only model M2 produced no dataset for, and it needs supervision for three heads at once.

- **Intent labels** — bootstrap from KCC's `QueryType` taxonomy (Plant Protection, Nutrient
  Management, etc.), which maps closely onto the needed taxonomy, giving weak labels at corpus
  scale; a stratified sample is hand-corrected for quality.
- **Entity labels** — distant supervision from KCC's `Crop`/`District`/`Season` columns, which
  supply answer spans without manual annotation. The M2 alias dictionary (32 entries, 984 harmonized
  occurrences) is the surface-form list.
- **Guardrail flags** — no natural source; must be authored (out-of-range dosages, banned terms,
  off-domain queries), consistent with §14 red-teaming. Since §7.9 trains this head
  recall-prioritized on rare positives, the set needs deliberate positive oversampling.

#### 7.0.4 Consistency Notes Against Milestone 2

| # | M2 assumption | M3 selection | Consequence |
|---|---|---|---|
| 1 | Vision: EfficientNet-B0 | ViT-Small | Not blocking. M2's `final/` artifacts (256² letterbox → 224² crop, ImageNet stats) transfer unchanged, and 224 divides evenly into the 16×16 patch grid; the imbalance recipe carries over. One item does not: the Tungro background-shortcut diagnostic was specified as CNN Grad-CAM and must be re-implemented as attention rollout for a ViT. |
| 2 | Embedder: MuRIL | BAAI/bge-m3 | **Resolved, one item outstanding.** MuRIL was not merely superseded on paper — it was built, measured, and found unusable (§5.2). M2 sized chunks at 512 chars *because* MuRIL caps at 512 subword tokens; bge-m3 accepts 8192, so 512 is now a retrieval-precision floor, not a model limit. PDF chunking has since been run (7,136 chunks). **Outstanding:** the built index holds 716,303 KCC chunks against M2's projected 1,468,625 — roughly half. The larger window plausibly explains fewer splits, but this is unconfirmed and must be reconciled before the corpus is called complete (Appendix C). |
| 3 | Vector store: FAISS `IndexFlatIP` | Qdrant + HNSW | Chunk artifacts and the shared metadata schema port directly. Qdrant payload filtering natively covers the crop/district/season reverse indices M2 planned to hand-build, so that step drops out. |
| 4 | — | Earlier drafts described bge-m3 as ~236M params, 768-dim, BERT-base, "pretrained on code-mixed Indic text" | **Corrected in v15.** Those were MuRIL's properties, left behind by an incomplete find-and-replace: the model name was swapped, the surrounding claims were not. bge-m3 is XLM-RoBERTa-large, ~568M, 1024-dim, 8192-token, and *not* India-specific — it won on measured in-domain separation (§5.2), not Indic pretraining. Process point: a model swap must propagate to every claim that justified the original choice, or the report defends the new model with the old one's reasoning. |

### 7.1 Vision Classifier
- **Data**: merged 20-class rice+wheat corpus, 12,859 images, zero leakage. PlantVillage was
  evaluated and **rejected** in M2 (lacks the required classes; lab imagery drops to 40–50% on field
  photos). PlantDoc is held out purely to measure the lab-to-field gap and never trained on.
- **Approach**: transfer learning from ImageNet-pretrained ViT-Small; freeze patch-embedding + early
  layers, progressively unfreeze as validation plateaus. Label-smoothed cross-entropy; early
  stopping on validation **macro**-F1 (not accuracy, to avoid bias toward over-represented classes).
  Hyperparameters in Appendix B.

### 7.2 Text Embedder (BAAI/bge-m3)
- **Status: no training required, and the index is built.** bge-m3 is used frozen. The production
  index over 723,439 chunks exists and passes its checks (§9.10) — the one component already past M4
  readiness rather than pending it.
- **Why frozen works**: bge-m3 is the only candidate that meaningfully separates documents *within*
  agriculture — unrelated chunks at 0.404 median cosine against ~0.85 for e5 and effectively 1.000
  for MuRIL. That spread is what gives the relevance tiers (§9.9) headroom to distinguish answerable
  from out-of-scope — a safety requirement, since the system emits pesticide dosages (§14).
- **Fine-tuning**: optional and **deprioritized**. The frozen model clears every retrieval and
  refusal gate; M2 KCC pairs remain available if the §14 Hinglish gap proves unsolvable by enabling
  sparse vectors. Fine-tuning would invalidate the §9.9 thresholds and require re-deriving them.
- **Serving discipline (§9.11)**: prefixes, vector dimension, and thresholds are read at load time
  from the index manifest, never hand-entered. Both failure modes are silent — an e5-style `query:`
  prefix degrades every result with no error, and a threshold calibrated for a different model
  refuses every query including perfect matches.

### 7.3 Structured Feature Embedder (Yield Cache)
- **Data**: feature columns of the two yield datasets. Triplets mined by yield proximity within
  matched agro-climatic zones — the M2 zone breakdown (Western UP / Eastern UP / Bundelkhand) is the
  natural stratifier, since yields differ ~40% across zones at equal crop and season. Not yet done.
- **Approach**: metric learning from scratch (triplet loss, hard-negative mining); stop on
  validation recall@k plateau.

### 7.4 Yield Prediction (GBT)
- **Data**: two models. The deployed UP model uses the 3,886-record subset with M2's
  **chronological** split (train 1997–2018, val 2019–20, holdout 2021–23), so evaluation is
  genuinely out-of-time. A pan-India model trains on the 440,962-record unified dataset for
  out-of-UP or out-of-crop queries. Both ready.
- **Approach**: gradient boosting, regression objective (squared error, or quantile loss for
  uncertainty bands); cross-validated Optuna search over `max_depth`, `learning_rate`,
  `n_estimators`, `subsample`; early stopping on validation RMSE.

### 7.5 Synthesis LLM (12B → 2–4B Distillation)
- **Data**: the 1,459,692 cleaned KCC pairs are the real (query, expert-answer) leg — genuine
  agronomist responses to genuine farmer queries, exactly the behaviour the student must imitate.
  The 184-document PDF corpus grounds the teacher-generated synthetic leg. Both sources exist; the
  paired set does not.
- **Approach**: sequence-level knowledge distillation; student initialized from a smaller checkpoint
  in the same family, fine-tuned on (a) teacher-generated synthetic agri-advisory Q&A and (b) real
  retrieved-context + reviewed-answer pairs.
- **Loss/config**: weighted KL-divergence (student vs. teacher) + cross-entropy against ground
  truth; full fine-tuning if compute allows, LoRA/QLoRA otherwise; few epochs, to avoid catastrophic
  forgetting. Stop on validation perplexity plus a held-out agri-QA set (automated +
  human-reviewed). Appendix B for the rest.

### 7.6–7.8 Speculative Draft, ASR, TTS *(not built for capstone, §1.5)*
Speculative draft: a further-distilled checkpoint from the §7.5 lineage, or a small model sharing
the tokenizer — the next optimization once base latency is measured. ASR: fine-tune IndicWhisper on
domain vocabulary and dialect audio; freeze encoder initially; SpecAugment. TTS: pretrained,
fine-tuned only if domain terms are mispronounced.

### 7.9 Guardrail Flags (folded into the Intent/Entity Extractor)
Rather than training and serving a standalone guardrails model, guardrail-flag prediction
(dosage-bounds violation, banned terms, off-domain query) is a **third head** on the same
DistilBERT-class backbone used for intent/entity extraction — same forward pass, negligible added
serving cost, one fewer model to version and deploy. This keeps guardrails independent of the
synthesis LLM's own judgment (not the same model grading its own output). Trained with cross-entropy
and heavy class weighting (violations are rare), **recall-prioritized** over precision, since a
missed violation is worse than a false alarm. Rule-based checks (banned-term blocklist, numeric
range checks) remain alongside.

### 7.10 Intent/Entity Extractor
- **Data — the one genuine gap (§7.0.3)**: no M2 dataset covers this model. Intent labels are
  bootstrappable from `QueryType`, entity labels from `Crop`/`District`/`Season`, but guardrail-flag
  labels must be authored from scratch. This is the item most likely to delay M4 if not started
  early.
- **Approach**: fine-tune a DistilBERT-class backbone with three joint heads — multi-label intent
  (BCE), token-level NER (cross-entropy or CRF), and the §7.9 guardrail head. Early stopping on
  combined intent/NER F1 plus guardrail recall.

---

## 8. Model Pipeline

Flow: raw input → preprocessing (§6) → parallel inference (vision + intent/entity) → structured
context assembly → guardrails pre-filter → generation → guardrails post-check → formatting.

**Intermediate outputs** passed between stages: disease label + confidence; intent flags + entities;
retrieved chunks + citation IDs + relevance tier (§9.9); yield estimate; price/weather;
profitability range.

**Feature engineering for generation.** The pre-filter derives boolean flags (`stale_price`,
`dosage_out_of_range`) injected into the LLM context, turning raw retrieved facts into
pre-validated, labeled inputs. Post-processing applies the hallucination/dosage/banned-term scan,
then template slot-filling (fast path) or structured parsing (slow path). Final output is
guardrail-approved text plus structured metadata (disease label, sources, price context, guardrail
flag).

---

## 9. Retrieval and Knowledge Components

*This section describes a system that is **built and measured**, not designed. Figures are from
actual runs on the production index; the §1.4 design-phase caveat does not apply. Reproduction steps
are in the RAG audit (§15).*

### 9.1 Retrieval Pipeline
Query → bge-m3 embedding (no prefix) → Qdrant HNSW search (cosine) over PDF and KCC **searched
separately** → intent-weighted merge (§9.6) → confidence tier on the **raw** score (§9.9) →
guardrails pre-filter → LLM context.

Every chunk carries provenance: PDF results carry file and page; KCC results carry record ID with
crop, district, and year.

### 9.2 Embedding Model
BAAI/bge-m3 — frozen, 1024-dim, cosine, **no query/passage prefix** (§4.2, §6). Justification and
the comparative experiment are in §5.2.

### 9.3 Vector Database
Qdrant, two collections: **Policy/KCC** and **Yield Cache** (different embedding spaces, §9.8).

**Server mode is mandatory, not a deployment preference.** Qdrant's embedded "local mode" ignores
the HNSW index and scans every record per query. Measured: 1.6s at 47k records, 3.7s at 107k —
linear in corpus size, extrapolating to roughly **25 seconds per query** at 723k. The same data in
server mode returns in **466ms**. This is the single largest latency factor in the retrieval path,
and easy to get wrong: local mode is the simpler default and fails only on speed, never on
correctness.

### 9.4 Similarity Search
HNSW, cosine, `m=16`, `ef_construct=128`; top-K=5 default. Index construction is disabled during
bulk load and enabled once at the end, so the graph is built a single time (15 min) rather than
rebuilt after each batch.

### 9.5 Chunking and Metadata Normalization
- **As built**: 512-token cap per chunk, over-length chunks split at sentence boundaries including
  the Hindi danda (`।`). PDF and KCC are normalized into one schema — same language detection,
  crop/district spellings, and ID scheme — with every ID verified unique at load.
- **Corpus**: 723,439 chunks = 7,136 PDF + 716,303 KCC.
- Each chunk carries metadata (source type, crop, district, season, year, page).
- **Filters**: `intent`, `source_type`, `crop` (Hindi names included), `district` (historical names
  auto-converted), `season`, `only_tables`, `year_from`, `top_k`.

**Metadata normalization is load-bearing, and it silently failed.** KCC stores crops in bracketed
dual-language form — `Paddy (Dhan)`, `Maize (Makka)`, `Bengal Gram (Gram/Chick Pea)` — and **63 of
281** crop names have brackets. The original converter matched only plain names, so `crop="rice"`
matched **0 of 112,269 rice chunks**. No error, no warning: an empty result set indistinguishable
from a legitimate "nothing found."

Fixed by matching the full string, then the pre-bracket portion, then each alias inside the bracket
— verified across all 716k records and confirmed *idempotent* (the converter's output converts to
itself), which is what makes the query-time filter and the stored value agree.

After the fix: `rice` 0 → 112,269 chunks, `maize` 0 → 15,047, and the Hindi surface form `dhan` also
resolves to 112,269. The lesson generalizes: **a filter that matches nothing is indistinguishable
from a query with no results** unless coverage is asserted explicitly. Filter cardinality is now
checked, not assumed.

### 9.6 RAG Workflow and Intent-Based Source Weighting
Retrieve top-K → guardrails pre-filter validates content (dosage bounds) → filtered chunks +
citation IDs enter the LLM context → the LLM cites retrieved material rather than generating from
parametric knowledge. PDF and KCC are searched as separate pools and merged under intent-set
weights, because the two sources answer different question types well:

| Intent | PDF weight | KCC weight | Rationale |
|---|---|---|---|
| `policy` | 2.0 | 0.5 | Scheme/eligibility answers must come from official documents |
| `field_practice` | 0.5 | 2.0 | Practical how-to is better served by expert KCC answers |
| `general` | balanced | balanced | No prior either way |

This also partly compensates for the 100:1 corpus imbalance (716k KCC vs. 7k PDF), which otherwise
lets KCC chunks crowd out the correct PDF on document-grounded queries — an effect still visible in
the §14 Hinglish failure.

**Confidence is computed on the raw score, never the weighted one.** Weighting exists to reorder
candidates; applying it before the threshold check would let a multiplier manufacture confidence the
underlying match does not support — precisely the failure the abstain tier exists to prevent.

### 9.7 Re-ranking
Conditional, not default: retrieve a larger pool (top-20) only when top-1 similarity falls below a
confidence threshold, then apply a lightweight cross-encoder to select the final top-5. Adds
~10–20ms only on ambiguous queries.

### 9.8 Yield Cache (Separate Namespace)
Structured-feature embeddings (§7.3) index recurring (crop×region×season×soil) combinations for
near-duplicate lookup ahead of the GBT model — a cache layer, not a replacement (§5.3). Conservative
threshold (cosine > 0.92); seasonal TTL (30–45 days); results labeled "approximate/similar
conditions" when cache-served.

### 9.9 Relevance Tiers and Threshold Calibration

| Tier | Raw score | Behaviour |
|---|---|---|
| `grounded` | ≥ 0.66 | Answer and cite sources |
| `fallback_with_disclaimer` | 0.56 – 0.66 | Answer, appended with "verify with your local KVK" |
| `abstain_out_of_scope` | < 0.56 | Refuse — state the query is outside the knowledge base |

**Thresholds are model-specific and must not be carried across models.** The values inherited from
Milestone 1 (0.85 / 0.65) were calibrated against a different embedder. Similarity scales are not
comparable between models — 0.86 from one means nothing like 0.86 from another — and applied
unchanged to bge-m3 they would have refused **every query, including exact matches**. Thresholds are
now derived from the measured score distribution on the real index and written into `manifest.json`,
from which the serving path reads them (§9.11).

Observed separation: real queries 0.590–0.811, junk queries 0.480–0.530. The upper threshold is set
at the **lower-quartile** point of real-query scores, not the median — using the median downgrades
half of all good answers to the disclaimer tier by construction.

The search entry point never raises: failures return `tier: "error"` rather than propagating an
exception into the request path.

### 9.10 Validation Status and the Health-Check Failure
The index passes filter, routing, and refusal checks; refuses off-domain queries correctly; and
returns substantive answers with concrete figures (e.g. "urea 30 kg per acre") at **30–330ms**
typical latency.

How the original health check failed is worth recording, because the check itself was the problem.
It compared "rice blast disease" against "how to win at chess" and required a clear margin — MuRIL
cleared it by **+0.874** while unable to distinguish wheat from mango. The gate tested
*cross-domain* separation, the one thing the failing model could do, and never tested the in-domain
separation the system depends on.

The replacement is a real ranking task: five farm topics (wheat urea, wheat rust, paddy nursery,
PM-KISAN, mango pest), each in English, Hindi, and Hinglish, with the correct document required to
rank first. The build **fails outright** if fewer than 4 of 5 rank first, if the margin is under
0.01 (correct by luck, not discrimination), or if unrelated documents score above 0.97 with each
other (representation collapse). The gate was itself validated against a deliberately broken model,
which it blocked on all three conditions.

**Principle adopted:** a health check must exercise the actual task, not a more tractable proxy. A
check that only ever passes is indistinguishable from no check.

### 9.11 Artifacts and Configuration Coupling

| Artifact | Size | Required to serve |
|---|---|---|
| `agri_knowledge-*.snapshot` | 3.80 GB | **Yes** — 723,439 vectors with text, metadata, HNSW index |
| `manifest.json` | ~1 KB | **Yes** — model name, prefix policy, vector dim, thresholds, weights |
| Embedding cache (~37 `.npy` shards) | ~1.5 GB | No — enables rebuild without repeating ~3h GPU work; FP16, measured accuracy loss 1.2e-7 |
| Eval/inspection JSON | small | No — evaluation records |

**The snapshot and manifest are a matched pair; the snapshot alone is unsafe.** Without the manifest
the serving path must guess prefix policy, vector dimension, and thresholds, and each wrong guess
fails silently. Serving reads all three from the manifest and asserts that the loaded model's
output dimension matches the index.

Rebuild cost: ~1 hour from the embedding cache, ~4 hours from scratch (~3h GPU encoding). Vectors
are checkpointed every 20,000 chunks, so an interrupted build resumes at the last completed batch —
validated against a simulated crash.

---

## 10. Prompt Engineering

**System prompt.** Defines the model as an agri-advisory assistant that responds in the farmer's
language, cites only supplied retrieved/tool data, never invents dosage or price figures, keeps
responses to 60–100 tokens, and adds a safety disclaimer where relevant.

**Template structure.**
```
[System Instructions — role, safety rules, output format]
[Retrieved Chunks — with citation IDs and relevance tier]
[Structured Tool Data — yield estimate, price, profitability range]
[Farmer Query — original text, post-ASR if voice]
[Output Format Instructions]
```

**Few-shot vs. zero-shot.** Primarily zero-shot, since the distilled model is fine-tuned for this
behaviour (§7.5). Two or three exemplars reinforce the profitability-framing rule (always a range
with caveats, never a binary success claim — §14). Guided decoding constrains output into light
structure (delimited sections for disease advice, policy info, yield estimate, disclaimer),
shortening it and making guardrail parsing more reliable than free-form prose.

**Hallucination mitigation, three layers.** (1) Retrieval-side: queries below the abstain threshold
(§9.9) never reach generation. (2) Generation-side: only values present in the supplied context may
be referenced. (3) Post-check: any dosage or price figure not traceable to source context is
flagged and replaced with a safe fallback. Guardrails are two-stage (pre-filter on retrieved/tool
data, post-check on generated text) and rule/classifier-based rather than extra LLM calls, to avoid
paying generation latency twice.

**Function calling.** A deliberate departure from a classic ReAct loop: tool selection happens
**upfront** via the intent/entity extractor, and all needed tools fire in one parallel batch before
the LLM runs — the LLM only consumes pre-fetched, pre-validated outputs. A dynamic LLM-driven loop
is retained only as a **slow-path fallback** for novel queries outside the intent taxonomy.

## 11. System Integration

**Communication.** Capstone: plain REST/JSON between backend and a single Python ML service; models
within it are called in-process (§1.5). Production: gRPC over Protobuf/HTTP2 once models are split
into per-model microservices.

**Shared schema.** One "enriched context" contract (retrieved chunks + tool outputs + guardrail
flags) is shared between orchestration and LLM input assembly, so every module producing context
data conforms to it.

**Data access.** Qdrant client (server mode, §9.3) and Redis client. GBT is in-process. No direct
LLM-to-database access — all access is mediated by the orchestration layer.

**Orchestration.** No agent framework on the low-latency path: the deterministic fan-out is
implemented directly, avoiding overhead for a case needing no dynamic planning. LangGraph is a
candidate **only** for the slow-path fallback (§10).

---

## 12. Computational Requirements

### 12.1 Hardware
| Workload | Recommended hardware |
|---|---|
| Vision, intent extractor, embedder | GPU (T4/L4-class) or CPU with ONNX Runtime |
| Synthesis LLM (2–4B, FP8) | Capstone: single GPU, batched. Production: A100/H100-class + vLLM |
| GBT yield model | CPU-only |
| Qdrant | **Server mode required** (§9.3); CPU, memory sized to the index |

### 12.2 Memory
The built Policy/KCC collection holds 723,439 chunks × **1024-dim** vectors; the full snapshot
(text, metadata, HNSW graph) is **3.80 GB**. The 1024-dim figure supersedes the 768-dim assumption
in earlier drafts — ~33% more vector storage (§7.0.4, note 4). The yield cache and Redis working
set are both much smaller.

### 12.3 Inference Latency
*Estimates except where marked **measured**.*

| Path | Latency |
|---|---|
| Fast path (cache hit) | ~35–50ms (estimate) |
| Slow path (cache miss) | ~210–290ms (estimate) |
| **Retrieval alone, full 723k index** | **30–330ms typical; p50 466ms; p95 903ms — measured** |
| Retrieval in Qdrant local mode (rejected, §9.3) | ~25s extrapolated at 723k (3.7s measured at 107k) |
| Voice input *(not built)* | +150–400ms ASR finalization |

**These figures do not reconcile, and that is the most important open item in this report.**
Retrieval was budgeted as one component inside a 210–290ms slow path, but measured retrieval alone
reaches p50 466ms and p95 903ms — the retrieval step by itself can exceed the entire end-to-end
target before vision, generation, or tool calls are counted.

Two things are true and should not be conflated. The measurement environment is not the target
environment — these come from a free Colab session with 2 CPU cores, and the 30–330ms typical figure
suggests the tail, not the common case, breaches the budget. But that is a *hypothesis, not a
result*: no measurement on representative hardware exists. Until one does, the §1.2 target should be
read as unvalidated against the one component actually measured. Benchmarking on target-class
hardware is therefore a first-order M4 task — better to learn whether the budget holds before the
remaining components are tuned around it.

### 12.4 Storage
| Artifact | Size |
|---|---|
| Distilled LLM checkpoint (2–4B) | ~4–8GB FP16 (~2–4GB FP8) |
| ViT-Small checkpoint | ~90MB |
| bge-m3 checkpoint (FP16) | ~568MB |
| **Vector index snapshot (measured)** | **3.80 GB** + ~1 KB manifest (required together, §9.11) |
| Embedding cache (rebuild insurance, not needed to serve) | ~1.5 GB |

---

## 13. Design Decisions

| Decision | Alternative rejected | Why | Confidence |
|---|---|---|---|
| **bge-m3 as embedder** | MuRIL (original choice); multilingual-e5 | MuRIL cannot separate topics *within* agriculture (unrelated farm sentences at 1.000 cosine); e5 leaves a 0.018 in/out-of-domain gap, too narrow to place a refusal threshold. bge-m3 gives 0.072 | **Highest in this report** — the only §5 choice decided by our own controlled experiment (§5.2) |
| **Qdrant server mode; embedded local mode rejected** | Run embedded for simplicity | Local mode ignores HNSW and scans linearly: ~25s/query at 723k vs. 466ms | High — measured at two corpus sizes, scaling linear (§9.3) |
| **Confidence tier on raw score, pre-weighting** | Tier on the weighted score | Source weighting is a ranking device; feeding it to the threshold lets a multiplier manufacture unsupported confidence | High — structural, not empirical |
| **Per-model thresholds derived from the live index** | Carry forward M1's 0.85/0.65 | Similarity scales are not comparable across models; M1 values applied to bge-m3 refuse every query including exact matches | High — verified directly (§9.9) |
| **Frozen embedder; fine-tune deprioritized** | Fine-tune on KCC pairs before indexing | The frozen model clears every gate; fine-tuning would invalidate calibrated thresholds for unmeasured gain | Medium — no fine-tuned comparison was run |
| Distill 12B → 2–4B | Keep 12B in critical path | 12B cannot fit the 150–180ms generation budget under any realistic batching | High — decode-time physics |
| GBT for yield; vector cache only as lookup | Embed yield outcomes in the RAG index | k-NN averaging underperforms GBT on tabular regression and would pollute the semantic index | High — well established |
| Upfront intent extraction + parallel fan-out | ReAct dynamic tool-calling for all queries | Sequential agent turns stack 150–180ms each — unaffordable for compound queries | High — same decode-time argument |
| Rule/classifier guardrails, two-stage | LLM-based guardrails (second generation pass) | Avoids paying generation latency twice; rule checks are more auditable in a safety-critical domain | Medium — recall vs. an LLM baseline still needs comparison |
| Single ML service; plain REST | Per-model microservices over gRPC | Removes network hops, proto maintenance, and per-service ops for one team and one deployment target | High — final (§1.5) |

---

## 14. Risks and Limitations

| Risk | Description | Mitigation |
|---|---|---|
| **Hinglish technical vocabulary fails retrieval** | "gehu me pila ratua" (wheat yellow rust) returns potato leaf-yellowing: the model matches *pila* (yellow) and misses *ratua* (rust). Confirmed **not** a coverage gap — the correct PDF is indexed and never surfaces, partly because 716k KCC chunks outnumber 7k PDF chunks. Plain Hindi, plain English, and everyday Hinglish all work; the failure is specific to transliterated technical terms | Enable bge-m3's sparse/lexical vectors — literal token matching is exactly what dense similarity blurs, and the capability is already in the chosen checkpoint. Until then, treat these as a known-weak path and prefer the disclaimer tier |
| **Thresholds rest on a thin calibration set** | The 0.66/0.56 boundaries (§9.9) derive from 14 self-authored questions. The separation is real, but the sample is small and not drawn from real usage | Recalibrate against real farmer queries with human judgement on answer usability. Because these gate whether dosage advice is emitted at all, this is a safety task, not a tuning task |
| **Corpus imbalance skews source selection** | 716,303 KCC vs. 7,136 PDF chunks (100:1); document-grounded queries can be crowded out even when the authoritative PDF is indexed | Intent weighting (§9.6) partly compensates; expanding the PDF corpus is the structural fix |
| **Bilingual retrieval check is invalid** | It compares whether English and Hindi phrasings return the *same record IDs*. Across 112,269 rice chunks both can return entirely correct but different records and score 0.00 | Rewrite to compare retrieved *topics*, not record identity. Its output is disregarded meanwhile — recorded rather than silently ignored, since a check known to be wrong is worse than none if someone later reads it as signal |
| **Retrieval latency measured only on Colab** | p50 466ms / p95 903ms on 2 CPU cores against a 200–300ms end-to-end target (§12.3) | Benchmark on target-class hardware early in M4, before optimizing around a budget that may not hold |
| **Intent/guardrail training data does not exist** | Every other capstone model maps to a prepared M2 dataset; this one does not (§7.0.3). Guardrail labels have no natural source and must be hand-authored, and the extractor gates tool selection for the whole fast path | Start label bootstrapping at the beginning of M4, not alongside model training; treat the authored adversarial set as a deliverable in its own right |
| Distillation accuracy loss | 2–4B student may lose reasoning breadth vs. the 12B teacher | Held-out agri-QA eval set (§7.5) to bound acceptable loss before deployment |
| Hallucination residual risk | Grounding and guardrails reduce but do not eliminate it | Abstain tier (§9.9) plus explicit user-facing disclaimer on dosage/financial answers |
| Data/representation bias | Training data skews toward well-represented crops, regions, languages | Monitor per-subgroup performance as standing practice. M2 quantified the starting skew: wheat and paddy are 31.9% of KCC queries; the vision corpus is 5.6:1 imbalanced, with `rice__leaf_smut` at only 40 images |

---

## 15. Deliverables

**Design documents** (`docs/references/`): HLD; LLD critical path (sequence diagram, API contracts,
cache/timeout config, failure modes); low-latency and academic architecture variants; consolidated
model inventory; and the **RAG Vector DB Audit & Runbook** — embedder bake-off, measured results,
bug post-mortems, artifact inventory, and retrieval instructions, and the source for §5.2, §9, and
the retrieval rows in §13–§14.

**Built artifacts (retrieval).** Unlike the rest of this milestone, retrieval exists as running code
and data rather than design:

| Artifact | Purpose |
|---|---|
| `08c_rag_vector_db_bge_m3.ipynb` | Builds the index. Run once (~4h); already done |
| `09_rag_retrieval.ipynb` | Searches the index. Builds nothing; ~10 min to load |
| `agri_knowledge-*.snapshot` (3.80 GB) + `manifest.json` | Production index and its configuration — required together (§9.11) |
| Embedding cache (~1.5 GB) | Rebuild without repeating GPU encoding |

Build and query paths are deliberately separate: building needs a GPU and hours, querying needs
neither. The e5 comparison notebooks were removed after selection; results are preserved in the
audit.

**Repository structure.** `services/` (backend-core; ml-mesh as one deployed service with each model
as a module), `docs-only/voice/` (designed, not built), `infra/` (Qdrant, Redis), `prompts/`,
`configs/`, `docs/`, `eval/`.

---

## 16. Summary and Next Steps

### 16.1 Architecture Summary
A three-tier architecture — fast classifiers/embedders, tool-based fact retrieval/computation, and a
single distilled-LLM synthesis pass — designed so the generative model never performs fact retrieval
or computation. Retrieval, yield prediction, and guardrails are purpose-built non-LLM components.

### 16.2 Readiness for Milestone 4
§7.0 maps every model to the M2 dataset that trains it. Against that inventory, the design is
training-ready pending: (a) the Intent/Entity Extractor label set, the only model with no prepared
dataset (§7.0.3); (b) two derivation steps from existing data — triplet mining and distillation-pair
construction; (c) selection of exact base checkpoints for the teacher/student pair; and (d) an
evaluation harness for held-out agri-QA and per-subgroup fairness checks. The vision corpus and both
yield datasets need nothing further.

**Retrieval has moved past readiness into completion.** The chunking re-validation and deferred PDF
chunking listed as pending in v14 are done: the index is built and validated over 723,439 chunks
(§9.10). It is no longer a design assumption — §5.2 and §9 report measured behaviour, where the
rest of this report reports intent. It also produced the report's most significant unresolved
problem: measured retrieval latency sits above the entire end-to-end budget (§12.3). That surfaced
*only* because the component was built, which is itself the argument for building the rest early
rather than deepening the design around unvalidated numbers.

What building it taught: three failures here — MuRIL's collapsed embedding space, the crop filter
matching zero rows, and thresholds inherited across incompatible models — were **silent**, each
producing clean runs and plausible output with no error. The health check that should have caught
the first passed comfortably because it tested an easier question than the real one (§9.10). M4
should assume the same holds for components not yet built, and design checks that can fail.

### 16.3 Planned Activities
1. **Benchmark the built index on target-class hardware before other optimization** — the §12.3 gap
   gates the §1.2 latency claim, and downstream tuning assumes a budget that may not survive
   measurement.
2. Reconcile the 716,303 vs. 1,468,625 KCC chunk-count discrepancy (§7.0.4, note 2).
3. Start Intent/Entity Extractor label bootstrapping (§7.0.3) — the long pole.
4. Enable and evaluate hybrid dense+sparse retrieval against the Hinglish failure (§14);
   recalibrate §9.9 thresholds against real farmer queries; rewrite the bilingual check to compare
   topics, not record IDs.
5. Run vision fine-tuning (per-class F1 baseline); build the distillation pipeline (§7.5) and the
   guardrail head (§7.9).
6. Stand up the fast-path template cache; integrate and load-test end to end.
7. Work through Appendix C — each row is a falsifiable claim this design depends on.

## Appendix A — Model Configuration

*Capstone build; see §1.5 for the production column.*

| Model | Precision | Capstone serving | Production serving |
|---|---|---|---|
| ViT-Small | FP16 | ONNX Runtime | + TensorRT if profiling justifies |
| BAAI/bge-m3 (1024-dim, no prefix) | FP16 | ONNX Runtime | + TensorRT if profiling justifies |
| Feature Embedder | FP16 | ONNX Runtime | ONNX Runtime |
| Intent/Entity Extractor + guardrail head | FP16 | ONNX Runtime | ONNX Runtime |
| Synthesis LLM | FP8 | Simple batched (single-GPU) | vLLM (continuous batching, guided decoding) |
| Speculative draft / ASR / TTS | — | *Not built* | vLLM spec. decoding; Faster-Whisper; streaming TTS |

## Appendix B — Hyperparameter Summary

| Model | Optimizer | LR | Batch | Epochs |
|---|---|---|---|---|
| Vision Classifier | AdamW | 3e-4 head / 3e-5 backbone | 64–128 | 20–30 |
| Text Embedder *(only if fine-tuned — not planned)* | AdamW | 2e-5 | — | — |
| Feature Embedder | Adam | 1e-3 | 256 | until recall@k plateau |
| Synthesis LLM (distillation) | AdamW | 1e-5–5e-5 | ~256 effective | 2–3 |
| Intent/Entity Extractor + guardrail head | AdamW | 3e-5 | 32 | 10–15 |

## Appendix C — Assumptions Requiring Validation

| Assumption | Current basis | Validation method | When |
|---|---|---|---|
| ~~bge-m3 outperforms alternatives on code-mixed retrieval~~ | ~~Known pretraining properties~~ | **RESOLVED (§5.2).** Controlled three-model comparison on identical corpus and tests. Unrelated-doc similarity 0.404 vs. 0.844 (e5-large); in/out-of-domain gap +0.072 vs. +0.018. Caveat: the code-mixed *technical* case remains weak (§14) | **Done** |
| **Retrieval fits the 200–300ms budget** | Contradicted by current evidence: p50 466ms / p95 903ms on 2-core Colab | Re-benchmark the built index on target-class hardware under concurrent load; if the gap persists, revise the budget or the retrieval design | **M4, first — gates the §1.2 claim** |
| **716,303 indexed KCC chunks is the correct corpus size** | M2 projected 1,468,625 from the same records — roughly double | Reconcile the chunking accounting; confirm the difference is the larger context window, not silent data loss at ingest | M4, before the corpus is called complete |
| Sparse vectors fix Hinglish technical retrieval | Mechanism plausible (lexical matching catches literal *ratua*), untested | Build the hybrid dense+sparse path; evaluate on a Hinglish technical query set against the dense-only baseline | M4 |
| Thresholds (0.66/0.56) generalize beyond the calibration set | 14 self-authored questions (§9.9) | Recalibrate on real farmer queries with human usability judgements | M4 |
| Intent weights (2.0/0.5) are correctly tuned | Chosen by reasoning about source suitability, never swept | Grid against a labelled policy-vs-practice query set | M4 |
| 2–4B student fits the ~150–180ms generation budget | Published token/sec for comparable sizes | Throughput benchmark under realistic concurrent load once the checkpoint exists | M4 |
| Distilled student retains acceptable accuracy | Standard distillation literature | Held-out agri-QA eval, automated + human-reviewed (§7.5) | M4 |
| GBT outperforms k-NN/neural nets for yield | General tabular-ML literature, not this dataset | Cross-validated comparison on the actual yield dataset | M4 |
| Guardrail rule-based recall is sufficient | Design assumption, not measured | Adversarial red-team set; compare recall against an LLM-based baseline | M4 |

## Appendix D — Change Log

| Version | Change |
|---|---|
| v1–v8 | Architecture evolution: single-service → caching/parallel fan-out/guardrail split → LLD critical path → tiered fast/slow path → compound-query decomposition and yield cache → consolidated model inventory and voice I/O |
| v9–v11 | Full Milestone 3 report (16 sections + appendices); added scope/assumptions section, decision-confidence ratings, and the validation-plan appendix; team sign-off |
| v12–v13 | Addressed TA feedback on architectural complexity: added capstone-vs-production tiering (§1.5); folded guardrails into the Intent/Entity Extractor as a multi-task head; deferred Kafka, TensorRT, and speculative decoding to production; finalized voice as excluded, one ML service, plain REST — no conditional language left in scope decisions |
| v14 | Added §7.0 Dataset-to-Model Mapping: explicit M2→M3 join, coverage summary, the Intent/Entity Extractor data gap and its bootstrapping plan, and consistency notes covering the EfficientNet→ViT, MuRIL→bge-m3, and FAISS→Qdrant changes |
| **v15 (current)** | **Incorporated the RAG Vector DB Audit & Runbook.** (a) **Corrected the MuRIL→bge-m3 substitution**, which in v14 swapped the model name while leaving MuRIL's properties and justification in place — §4.1 params, §4.2 architecture, §6 tokenizer/length/dimension and the no-prefix requirement, §9.2, §12.2 dimension; §7.0.4 note 4 marked applied. (b) **Rewrote §5.2** from asserted properties to the measured three-model comparison, including MuRIL's build failure and the refusal-headroom argument. (c) **Rewrote §9 against the built system** — intent-based source weighting and raw-score confidence (§9.6), relevance tiers and threshold calibration (§9.9), validation status and health-check post-mortem (§9.10), artifact/manifest coupling (§9.11), plus the crop-filter bug and Qdrant server-mode finding. (d) **Surfaced the latency conflict** between measured retrieval and the 200–300ms target (§12.3). (e) Added measured storage figures, five retrieval decision rows (§13), six retrieval risk rows (§14); resolved one Appendix C row and added five; flagged the unreconciled KCC chunk count. (f) **Condensed the report from ~1,510 to ~800 lines**, cutting redundant cross-reference stubs and verbose restatement while preserving all measured results, decisions, and open problems. |

---

## Team Review & Sign-Off

Sign-off was collected on v11; v12–v14 made scope revisions. **v15 requires fresh sign-off rather
than re-confirmation** — it changes a §5 justification on measured evidence, corrects model
specifications that were wrong in v14 (§7.0.4, note 4), and records a measured result conflicting
with the §1.2 target (§12.3). Reviewers should read §5.2, §9.9–9.11, §12.3, and the §14 retrieval
rows specifically.

| # | Team Member | Approved | Date |
|:-:|-------------|:--------:|:----:|
| 1 | Mahesh | Yes | 23 Jul 2026 |
| 2 | Harliv | Yes | 23 Jul 2026 |
| 3 | Lokesh | Yes | 23 Jul 2026 |
| 4 | Aneeqa | Yes | 23 Jul 2026 |
| 5 | Tanmay | Yes | 23 Jul 2026 |

**Document version:** Milestone 3 — v15 · **Prepared:** 23 July 2026 · **Revised:** 28 July 2026
(retrieval findings incorporated from the RAG Vector DB Audit & Runbook, 27 July 2026)
