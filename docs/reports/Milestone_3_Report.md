# FarmerVision — Milestone 3 Report
## Model Architecture & End-to-End Pipeline Design

---

## 1. Introduction

### 1.1 Project Recap
FarmerVision is a multimodal AI advisory system for farmers, primarily
targeting Hindi and regional-dialect speakers in Uttar Pradesh (with
multi-state expansion as a stretch goal). Farmers interact via text,
photo, or voice to get help with three core needs: **crop disease
diagnosis** (from a photo), **government policy/incentive lookup**
(scheme eligibility, subsidy info), and **yield/profitability guidance**
(what yield to expect, whether a crop choice is likely to pay off).

### 1.2 Objectives of Milestone 3
This milestone selects and justifies the model(s) powering each module,
and designs a complete, defensible pipeline from raw farmer input (text,
image, or voice) to final delivered output (text and/or spoken advice) —
including retrieval, tool-use, guardrails, and generation. The target
serving environment assumes reliable connectivity (an academic framing
that isolates model/serving choices from network-quality confounds), with
an end-to-end **latency target of 200–300ms** for the core reasoning path.

### 1.3 Relationship Between Model Architecture and Project Goals
The domain is narrower than general open-domain assistance: a finite set
of crops, diseases, districts, and government schemes, with numeric
data (yield, price) that has known, computable ground truth. This
structure is what makes the 200–300ms target achievable at all — it
allows most of the "intelligence" to live in fast retrieval and small
purpose-built models (Tier 1/2), with a single large generative model
invocation reserved only for turning already-validated facts into fluent
language (Tier 3). A single large general-purpose LLM handling everything
end-to-end — retrieval, computation, and generation — would not meet the
latency target and would introduce unnecessary hallucination risk on
numeric/financial claims. The architecture below is built around that
constraint.

### 1.4 Scope, Assumptions, and Validation Status
This is a **design-phase document**. Model choices, pipeline structure,
and the 200–300ms latency target are grounded in published benchmarks for
comparable model sizes/hardware and standard engineering practice — they
are **not yet measured on this specific system**. Real implementation
will surface constraints this document cannot anticipate: actual GPU
availability, real training-data quality and volume, true distillation
accuracy retention, and production traffic patterns will all cause
specific numbers (latency, hyperparameters, cache-hit rates) to shift.

Where this matters most:
- **Latency figures** (§3, §12.3) are architectural estimates — they show
  what the *design* targets, not a benchmarked result.
- **Hyperparameters** (§7, Appendix C) are standard starting points for
  each model class, expected to change after real tuning.
- **Model choices** (§5) are justified against known properties of each
  model family, not against this project's own comparative experiments,
  which haven't been run yet.

Appendix G lists the specific assumptions that most need empirical
validation before this design should be treated as final, along with how
and when each will be tested.

---

### 2.1 High-Level Diagram

```
   [React/Next.js or Flutter Client] ── text / image / voice input
                │
                ▼
   [Java Backend Core — Spring Boot] ── AuthN/Z, orchestration,
                │                        rate limiting, trace context
                │
      ┌─────────┼──────────────────────────────────────┐
      │         │                                        │
      ▼         ▼                                        ▼
 [Redis      [Vector DB Cluster            [Python ML Mesh]
  Cache]      (Qdrant) — Policy/KCC         ├─ ASR / TTS (Indic)
              + Yield Cache namespaces       ├─ Vision Classifier (ViT/MobileViT)
                                              ├─ MuRIL Embedder
                                              ├─ Intent/Entity Extractor
                                              ├─ Yield Prediction (GBT)
                                              ├─ Profitability Estimator
                                              ├─ Guardrails (pre/post)
                                              └─ Synthesis LLM (distilled, vLLM)
      │
      ▼
 [External APIs: Agmarknet Mandi / Weather] ── circuit breaker, cache-first
      │
      ▼ (async, out-of-band)
 [Kafka / Pub-Sub] → [Offline Data Lake] → [Drift Detection + Retraining]
```

### 2.2 Major Modules and Interactions
| Module | Role |
|---|---|
| Client (React/Next.js, Flutter) | Captures text/image/voice, renders streamed response, TTS playback |
| Java Backend Core | Auth, orchestration, request fan-out, circuit breaking, trace propagation |
| Python ML Mesh | Hosts all model inference (vision, ASR/TTS, embedding, LLM, classifiers) |
| Vector DB Cluster (Qdrant) | Policy/KCC semantic retrieval + separate yield-answer cache |
| Redis | Embedding cache, vector-search cache, mandi/weather cache, session state |
| External APIs | Live mandi price, weather |
| Kafka / Pub-Sub | Telemetry, drift signals, retraining triggers — decoupled from serving path |

### 2.3 Data Flow
Text/image/voice → (voice converted to text via ASR) → parallel vision +
intent/entity extraction → fast-path cache check → (on miss) parallel
tool fan-out (policy retrieval, yield cache/GBT, mandi/weather,
profitability) → guardrails pre-filter → LLM synthesis → guardrails
post-check → response (text, streamed; optionally spoken via TTS).

### 2.4 External Services/APIs
- Agmarknet Mandi Price API (government)
- Weather forecast API
- (Optional) Bhashini as an alternative/supplementary ASR-MT service for
  dialect coverage

### 2.5 Technology Stack
| Layer | Technology |
|---|---|
| Frontend | React/Next.js (web), Flutter/Kotlin (mobile) |
| Backend orchestration | Java, Spring Boot 3.x |
| Inter-service transport | gRPC over Protobuf / HTTP2 |
| ML serving | Python, vLLM (LLM), TensorRT / ONNX Runtime (small models), Faster-Whisper/CTranslate2 (ASR) |
| Vector DB | Qdrant (HNSW, cosine similarity) |
| Cache | Redis |
| Messaging | Apache Kafka / Cloud Pub/Sub |
| Drift & retraining | EvidentlyAI, Prefect |
| Observability | OpenTelemetry (cross-language trace propagation) |

---

## 3. End-to-End Workflow

### 3.1 Complete Workflow Diagram

```
        [Farmer Query]
   ┌───────────┼───────────┐
 typed        image       spoken
 text        (photo)         │
   │           │             ▼
   │           │      #ASR (streaming) → #Dialect Normalizer
   │           │      → text (same downstream as typed)
   │           │             │
   └───────────┴─────────────┘
               │
               ▼
   [text + optional image, entry point]
               │
     ┌─────────┴─────────┐
     ▼                   ▼
[Vision Classifier]  [Intent/Entity Extractor]     (parallel, both Tier 1)
     │                   │
     └─────────┬─────────┘
               ▼
   [Fast-path key: disease_label + intent_flags]
               │
      ┌────────┴────────┐
   cache HIT          cache MISS
      │                   │
      ▼                   ▼
 [Template          [Parallel Tool Fan-Out]
  response]          ├─ Policy Vector DB (Qdrant)
      │               ├─ Yield Cache DB → (miss) → GBT Yield Model
      │               ├─ Mandi/Weather Tool
      │               └─ Profitability Estimator
      │                   │
      │                   ▼
      │            [Guardrails Pre-Filter]
      │                   │
      │                   ▼
      │            [Synthesis LLM, streamed]
      │                   │
      │                   ▼
      │            [Guardrails Post-Check]
      │                   │
      └─────────┬─────────┘
                 ▼
        [Response: text, streamed]
                 │
                 ▼ (if voice reply requested)
              [TTS, streamed]
                 │
                 ▼
           [Farmer receives answer]
```

### 3.2 Inputs and Outputs of Each Module
See §6 (Model Inputs and Outputs) for the full per-model table.

### 3.3 Sequence Diagram
See Appendix A for the detailed multi-actor sequence diagram (client,
backend, cache, vision, embedder, vector DB, LLM, external APIs,
guardrails).

### 3.4 Error Handling and Fallback Mechanisms
| Failure | Fallback |
|---|---|
| Vector DB unavailable | Skip retrieval, synthesize from tool outputs only, flag `sources: []` |
| Mandi/Weather API down (circuit open) | Serve last cached value, flag `stale: true` |
| LLM synthesis timeout (>6s hard cap) | Return retrieved facts/labels without synthesized prose |
| Guardrails service down | Fail-closed — never return unvalidated advice; `escalate_to_expert` |
| ASR low confidence on key entity | Trigger short spoken confirmation before proceeding |
| Network drop mid-upload | Client queues request locally, retries with same idempotent trace ID |

### 3.5 Storage and Retrieval Components
- **Qdrant** — two logical collections: Policy/KCC (MuRIL embeddings) and
  Yield Cache (structured-feature embeddings), kept separate.
- **Redis** — embedding cache, vector-search result cache, external API
  cache, session/conversation state, fast-path template store.

### 3.6 User Interaction Flow
1. Farmer opens app, chooses text, camera, or mic input.
2. For voice: partial transcript builds as they speak; response begins
   streaming almost immediately after they stop.
3. Provisional/streamed response appears progressively rather than after
   a full wait — text fills in, and audio (if enabled) begins on the
   first completed sentence.
4. Farmer can follow up conversationally; session context persists via
   Redis session state.

---

## 4. Model Architecture Selection

### 4.1 Models Selected per Module

| Module | Model | Pretrained / Custom | Params |
|---|---|---|---|
| Vision (disease classification) | ViT-Small (fallback: MobileViT-XXS for edge) | Pretrained (ImageNet) → fine-tuned | ~22–86M |
| Text embedding (retrieval) | MuRIL-base | Pretrained → optionally fine-tuned | ~236M |
| Structured feature embedding (yield cache) | Custom two-tower encoder | Custom, trained from scratch | ~5–10M |
| Intent/Entity extraction | DistilBERT-class, multi-task head | Pretrained → fine-tuned | ~66M |
| Yield prediction | XGBoost/LightGBM (GBT) | Custom, trained from scratch | N/A (tree ensemble) |
| Profitability estimation | Rule-based composite | Custom (non-ML initially) | N/A |
| Guardrails (pre/post) | Rule-based + small classifier | Hybrid | small |
| Synthesis LLM | Gemma, distilled from 12B teacher to 2–4B student | Pretrained teacher → distilled custom student | 2–4B |
| Speculative draft model | Small model, same tokenizer family | Pretrained/distilled | 350–500M |
| ASR | IndicWhisper / Bhashini-class | Pretrained → optionally fine-tuned on dialect data | ~250M–1.5B |
| Dialect normalization | IndicTrans2 (or fine-tune) | Pretrained | ~1.2B (distilled variant preferred) |
| TTS | VITS/FastSpeech2-class, Indic multi-speaker | Pretrained | ~30–90M |

### 4.2 Model Architecture Notes
- **ViT-Small**: standard Vision Transformer, patch-based tokenization
  (16×16 patches), 12 transformer encoder layers, fine-tuned classification
  head sized to the number of disease classes in the dataset.
- **MuRIL**: BERT-base architecture (12 layers, 768 hidden, 12 attention
  heads), pretrained on Indic languages with both native-script and
  transliterated text — this transliteration robustness is the specific
  reason it's chosen over a generic multilingual embedder (see §5).
- **Distilled synthesis LLM**: same architecture family as the teacher
  (transformer decoder), reduced layer count/hidden size to hit the 2–4B
  parameter range; exact configuration depends on which base checkpoint
  the teacher lineage supports for distillation — to be finalized in
  Milestone 4 alongside training-data preparation.
- **GBT yield model**: not a neural architecture — gradient-boosted
  decision trees, chosen specifically because the yield-prediction task is
  tabular, not sequential or spatial (see §5 for the "why not a neural
  net or vector retrieval" comparison).

### 4.3 Integration Between Multiple Models
Vision and intent/entity extraction run independently and in parallel;
their outputs are combined into a fast-path cache key. On a cache miss,
outputs from the embedder, GBT model, external tools, and profitability
estimator are all assembled into a single structured context object that
becomes the LLM's input — the LLM never calls these tools itself in the
low-latency path (see §10.7 for why this diverges from a classic
agentic/function-calling design).

---

## 5. Justification of Model Choices

### 5.1 Vision: ViT-Small (vs. CNN alternatives)
- **Why chosen**: transformers generalize better than CNNs on
  fine-grained texture differences (e.g., early-stage fungal vs.
  bacterial lesions) given sufficient fine-tuning data, and patch
  attention gives some interpretability (attention maps over the leaf).
- **Alternatives considered**: ResNet/EfficientNet (CNN) — cheaper to
  train and run, strong baseline, but historically weaker on fine-grained
  classification without heavy augmentation.
- **Trade-off accepted**: ViT needs more training data to reach CNN-level
  performance; mitigated by starting from an ImageNet-pretrained
  checkpoint rather than training from scratch.

### 5.2 Text Embedding: MuRIL (vs. multilingual-e5, LaBSE, IndicBERT)
- **Why chosen**: farmer queries are frequently code-mixed
  (Hindi-English, transliterated Hindi in Latin script) — MuRIL was
  specifically pretrained on this mix, which general multilingual
  embedders are not.
- **Alternatives considered**: multilingual-e5 (strong general retrieval
  benchmark performance, but not transliteration-aware), LaBSE
  (translation-focused, not tuned for retrieval-style similarity).
- **Trade-off accepted**: MuRIL is India-specific and would need
  replacement or a second embedder if the product expands beyond Indic
  languages.

### 5.3 Yield Prediction: GBT (vs. neural net, vs. vector k-NN retrieval)
- **Why chosen**: yield prediction is a structured/tabular regression
  problem. Gradient-boosted trees are the established strong baseline on
  tabular data, generally outperforming both deep nets and k-NN retrieval
  approaches at this data scale, and are cheap to serve on CPU with no
  GPU dependency.
- **Alternatives considered**: this was explicitly discussed and rejected
  earlier in the design process — embedding yield outcomes directly into
  the RAG vector index and answering via nearest-neighbor averaging.
  Rejected because (a) k-NN averaging is a weaker regression method than
  GBT for continuous tabular targets, and (b) it would require sharing an
  embedding space with the semantically-oriented policy/KCC index, which
  degrades both. Vector retrieval is instead used only as a **cache layer
  in front of** the GBT model (§9.5), not a replacement for it.

### 5.4 Synthesis LLM: Distilled 2–4B (vs. keeping 12B, vs. off-the-shelf small model)
- **Why chosen**: a 12B model cannot generate a response within the
  150–180ms slice of the latency budget available to it (decode is
  sequential per token; even batched serving has a throughput floor).
  Distilling from the existing 12B teacher preserves domain behavior
  learned during any prior tuning of that model, rather than starting
  from a generic small model with no agri-domain grounding.
- **Alternatives considered**: adopting an off-the-shelf small model
  (e.g., a 2–3B open instruction-tuned model) directly, skipping
  distillation. This is a legitimate fallback if distillation
  infrastructure/timeline is constrained, but is expected to underperform
  the distilled option on domain-specific accuracy without additional
  fine-tuning of its own — effectively similar engineering cost either
  way.
- **Trade-off accepted**: the distilled student will have narrower
  general reasoning ability than the 12B teacher; this is acceptable
  because the model's job is constrained to synthesis over already-
  retrieved facts, not open-ended reasoning (see §10 prompt design).

### 5.5 Vector DB: Qdrant (vs. ChromaDB, Pinecone, Weaviate)
- **Why chosen**: native HNSW performance, self-hostable (relevant for
  data sovereignty on government policy data), supports multiple
  collections/namespaces cleanly (needed to keep Policy/KCC and Yield
  Cache separate), and supports read-replica configurations for the
  read/write isolation described in §9.5 and the earlier HLD.
- **Alternatives considered**: Pinecone/Weaviate (managed, less
  infrastructure burden, but less control over co-location/latency and
  recurring hosted cost at scale); ChromaDB (simpler, weaker
  clustering/replication story at production scale).

### 5.6 ASR/TTS: Self-hosted Indic models (vs. cloud APIs)
- **Why chosen**: streaming is core to hitting acceptable perceived
  latency (§3, §10) — self-hosted models give direct control over
  streaming behavior and chunking; cloud APIs vary in streaming support
  and add per-call cost at farmer-scale usage volumes.
- **Trade-off accepted**: self-hosting requires more MLOps investment
  (model updates, GPU capacity planning) than a managed API call; this is
  accepted given the latency requirement is a hard product constraint,
  not a nice-to-have.

---

## 6. Model Inputs and Outputs

| Model | Input | Preprocessing | Output |
|---|---|---|---|
| Vision Classifier | Image (JPEG/PNG) | Resize to model input size, normalize (ImageNet mean/std), client-side compression before upload | `disease_label` (categorical), `confidence` (float), optional bbox |
| MuRIL Embedder | Raw text (query or document chunk) | WordPiece tokenization, max sequence length ~128 tokens, lowercase/normalize for Hindi-English code-mixing | 768-dim float vector |
| Structured Feature Embedder | Crop ID, region ID, soil type, season, acreage | Categorical encoding + numeric normalization | 64–128-dim float vector |
| Intent/Entity Extractor | Raw query text | Tokenization (same family as backbone) | Multi-label intent flags + NER entity spans (crop, region, scheme) |
| Yield Prediction (GBT) | Structured features (crop, region, soil, rainfall, sowing date, variety) | Standard tabular preprocessing (encoding, scaling as needed for the specific boosting library) | Yield estimate (continuous, with optional quantile bands) |
| Synthesis LLM | Enriched prompt: system instructions + retrieved chunks + tool outputs + query | SentencePiece tokenization (model's native tokenizer), prefix caching for shared system/guardrail instructions | Token sequence, capped 60–100 tokens, decoded to text |
| ASR | Raw audio waveform | 16kHz mono PCM, streaming chunked | Text transcript + per-token confidence scores |
| Dialect Normalizer | Raw transcript text | — | Normalized standard-language text |
| TTS | Text (sentence-chunked) | Sentence segmentation for streaming | Audio waveform, streamed |

---

## 7. Training Strategy

*Note: hyperparameters below are recommended starting points based on
standard practice for each model class; final values are expected to be
tuned empirically once training data is finalized in Milestone 4.*

### 7.1 Vision Classifier
- **Approach**: transfer learning — ImageNet-pretrained ViT-Small,
  fine-tuned on a labeled crop-disease dataset (e.g., PlantVillage-style
  data augmented with India-region-specific images).
- **Frozen vs. trainable**: freeze patch-embedding + early encoder
  layers initially; progressively unfreeze (discriminative fine-tuning)
  as validation accuracy plateaus.
- **Loss**: label-smoothed cross-entropy.
- **Optimizer**: AdamW.
- **LR strategy**: cosine schedule with linear warmup; base LR ~3e-4 for
  the classification head, ~3e-5 for unfrozen backbone layers.
- **Batch size**: 64–128 (GPU-memory dependent).
- **Epochs**: ~20–30, with early stopping on validation macro-F1
  (macro, not accuracy, to avoid bias toward over-represented disease
  classes).
- **Checkpointing**: save best-validation-F1 checkpoint.

### 7.2 Text Embedder (MuRIL)
- **Approach**: used pretrained/frozen by default; optionally fine-tuned
  via contrastive learning (InfoNCE/triplet loss) on query–chunk pairs
  from the policy/KCC corpus to improve retrieval relevance.
- **Frozen vs. trainable**: if fine-tuned, unfreeze only the top N
  transformer layers.
- **Optimizer/LR**: AdamW, LR ~2e-5.

### 7.3 Structured Feature Embedder (Yield Cache)
- **Approach**: trained from scratch via metric learning (triplet loss,
  hard-negative mining) so agronomically similar inputs cluster in
  embedding space.
- **Optimizer/LR**: Adam, LR ~1e-3, batch size 256.
- **Stopping criterion**: validation retrieval recall@k plateau.

### 7.4 Yield Prediction (GBT)
- **Approach**: standard gradient boosting (XGBoost/LightGBM), objective
  = regression (squared error, or quantile loss if uncertainty bands are
  required).
- **Hyperparameter search**: cross-validated grid/Bayesian search (Optuna)
  over `max_depth`, `learning_rate`, `n_estimators`, `subsample`.
- **Stopping criterion**: early stopping on validation RMSE.

### 7.5 Synthesis LLM (12B → 2–4B Distillation)
- **Approach**: sequence-level knowledge distillation — student
  initialized from a smaller pretrained checkpoint in the same model
  family, fine-tuned on a combination of (a) teacher-generated
  synthetic Q&A over agri-advisory scenarios and (b) real
  retrieved-context + human-reviewed answer pairs.
- **Loss**: weighted combination of KL-divergence (student vs. teacher
  output distribution) and standard cross-entropy against ground-truth
  answers.
- **Frozen vs. trainable**: full fine-tuning if compute allows;
  LoRA/QLoRA as a parameter-efficient fallback if not.
- **Optimizer/LR**: AdamW, warmup + cosine decay, LR ~1e-5–5e-5.
- **Batch size**: gradient accumulation to reach an effective batch size
  of ~256, actual per-step batch constrained by GPU memory.
- **Epochs**: 2–3 (LLM fine-tuning typically needs few epochs to avoid
  catastrophic forgetting of general language ability).
- **Stopping criterion**: validation perplexity plus a held-out
  agri-QA evaluation set (automated + human-reviewed scoring).
- **Checkpointing**: periodic step-based checkpoints, best-eval retained.

### 7.6 Speculative Draft Model
- Either a further-distilled (smaller) checkpoint from the same lineage
  as §7.5, or a small pretrained model sharing the same tokenizer —
  required for speculative decoding compatibility.

### 7.7 ASR
- **Approach**: fine-tune an IndicWhisper/Bhashini base checkpoint on
  domain vocabulary (crop/scheme names) and, where available, dialect
  audio data.
- **Frozen vs. trainable**: freeze encoder initially, fine-tune decoder
  and top encoder layers.
- **Regularization**: SpecAugment.
- **LR**: ~1e-5, small batch size (audio memory constraints).

### 7.8 TTS
- Used pretrained by default; fine-tune only if domain terms
  (crop/scheme names) are mispronounced, using a small targeted
  domain-glossary audio set.

### 7.9 Guardrails Classifiers
- **Approach**: lightweight classifier (logistic regression or small
  transformer) trained on labeled safe/violating examples (dosage bounds,
  banned terms).
- **Loss**: cross-entropy with heavy class weighting (violations are
  rare — imbalanced data).
- **Stopping criterion**: early stopping on validation **recall**
  (prioritized over precision, since this is a safety-critical
  classifier — a missed violation is worse than a false alarm).

### 7.10 Intent/Entity Extractor
- **Approach**: fine-tune DistilBERT-class backbone with a joint head —
  multi-label intent classification (BCE loss) + token-level NER
  (cross-entropy or CRF loss).
- **LR/batch/epochs**: LR ~3e-5, batch 32, ~10–15 epochs, early stopping
  on combined F1 across both heads.

---

## 8. Model Pipeline

### 8.1 Data Flow Into Models
Raw input (image/text/audio) → modality-specific preprocessing (§6) →
parallel model inference (vision + intent/entity, and ASR if voice) →
structured context assembly (retrieved chunks + tool outputs) →
guardrails pre-filter → LLM input assembly → generation → guardrails
post-check → output formatting.

### 8.2 Preprocessing Before Inference
See §6 per-model preprocessing column.

### 8.3 Feature Engineering
- Guardrails pre-filter derives boolean flags (e.g., `stale_price`,
  `dosage_out_of_range`) that are injected into the LLM's structured
  context — this counts as feature engineering for the generation step,
  turning raw retrieved facts into pre-validated, labeled inputs.

### 8.4 Intermediate Outputs
- Disease label + confidence (vision)
- Intent flags + entities (extractor)
- Retrieved chunks + citation IDs (vector DB)
- Yield estimate (GBT or cache)
- Price/weather data (tool)
- Profitability range (estimator)

### 8.5 Post-Processing
- Guardrails post-check: hallucination/dosage/banned-term scan on
  generated text.
- Template slot-filling (fast path) or structured-output parsing (slow
  path — see §10.4).
- Sentence segmentation for TTS streaming.

### 8.6 Final Prediction Generation
Final response = guardrail-approved text, optionally rendered to audio
via streaming TTS, delivered to the client alongside structured metadata
(disease label, sources, price context, guardrails flag).

---

## 9. Retrieval and Knowledge Components

### 9.1 Retrieval Pipeline
Query → MuRIL embedding → Qdrant HNSW search (cosine similarity) → top-K
chunks → guardrails pre-filter → included in LLM context.

### 9.2 Embedding Model
MuRIL-base (see §5.2 for justification).

### 9.3 Vector Database
Qdrant, two separate collections: **Policy/KCC** and **Yield Cache**
(different embedding spaces — see §9.5).

### 9.4 Similarity Search Algorithm
HNSW indexing, cosine similarity, top-K=5 for policy retrieval.

### 9.5 Chunking Strategy
- Policy/KCC documents chunked into ~200–400 token passages with ~50
  token overlap between adjacent chunks, preferring paragraph/section
  boundaries over arbitrary token cuts.
- Each chunk tagged with metadata (state, scheme name, applicable
  crop(s), document date) to support metadata-filtered retrieval
  alongside semantic search.

### 9.6 RAG Workflow
Retrieve top-K chunks → guardrails pre-filter validates content (e.g.,
dosage figures within safe bounds) → filtered chunks + citation IDs
inserted into the LLM's structured context → LLM cites retrieved
material rather than generating facts from parametric knowledge.

### 9.7 Re-ranking Strategy
Given the tight latency budget, re-ranking is **conditional, not
default**: retrieve a larger candidate pool (e.g., top-20) only when the
top-1 similarity score falls below a confidence threshold, then apply a
lightweight cross-encoder re-ranker to select the final top-5. This adds
~10–20ms only on ambiguous queries, keeping the common case at the
cheaper single-pass retrieval cost.

### 9.8 Yield Cache (Separate Namespace)
Structured-feature embeddings (§4.1, §7.3) index recurring
(crop×region×season×soil) combinations for near-duplicate lookup ahead
of the GBT model — see §5.3 for why this is a cache layer, not a
replacement for regression-based yield prediction. Conservative
similarity threshold (e.g., cosine > 0.92); cache entries carry a
seasonal TTL (30–45 days) and are labeled as "approximate/similar
conditions" rather than farmer-specific when served from cache.

---

## 10. Prompt Engineering

### 10.1 System Prompt (Design Summary)
Defines the model's role as an agri-advisory assistant that: responds in
the farmer's language, cites only supplied retrieved/tool data, never
invents dosage or price figures, keeps responses concise (60–100 tokens),
and includes a safety disclaimer where relevant (e.g., recommending
consultation with a local agri officer for exact dosage).

### 10.2 Prompt Template Structure
```
[System Instructions — role, safety rules, output format]
[Retrieved Policy/KCC Chunks — with citation IDs]
[Structured Tool Data — yield estimate, price, profitability range]
[Farmer Query — original text, post-ASR if voice]
[Output Format Instructions]
```

### 10.3 Few-Shot vs. Zero-Shot Strategy
Primarily zero-shot, since the distilled model is fine-tuned specifically
for this behavior (§7.5). A small number (2–3) of few-shot exemplars are
retained in the system prompt template specifically to reinforce the
profitability-framing rule (always a range with caveats, never a binary
success claim — see §14).

### 10.4 Structured Output Format
Generation is constrained via guided/grammar-constrained decoding (vLLM
guided decoding) into light structure (e.g., delimited sections for
disease advice, policy info, yield estimate, disclaimer) — this both
shortens output and makes the post-processing/guardrails step more
reliable than parsing free-form prose.

### 10.5 Hallucination Mitigation
Generation is restricted to referencing only values present in the
supplied structured context; the guardrails post-check regex-flags any
numeric dosage/price figure that does not trace back to the source
context and strips or replaces it with a safe fallback phrase.

### 10.6 Guardrails
Two-stage: a **pre-filter** on retrieved/tool data before synthesis
(dosage-range checks, stale-data flagging) and a lightweight **post-check**
on generated text (hallucinated-number detection, banned-term blocklist,
language-consistency check) — both rule/classifier-based, not additional
LLM calls, to avoid re-incurring generation latency for validation.

### 10.7 Function Calling / Tool Use
This is a deliberate departure from a classic agentic ReAct loop where
the LLM decides which tools to call turn-by-turn. For the low-latency
path, **tool selection happens upfront** via the intent/entity extractor
(§4.3, §9), and all needed tools fire in one parallel batch before the
LLM ever runs — the LLM only consumes pre-fetched, pre-validated tool
outputs as context. A dynamic, LLM-driven tool-calling loop is retained
only as a **slow-path fallback** for genuinely novel queries outside the
covered intent taxonomy, where the latency target is relaxed and
exploratory reasoning is worth the cost.

---

## 11. System Integration

### 11.1 How Models Communicate
gRPC over Protobuf/HTTP2 between the Java backend and the Python ML mesh;
internal services within the ML mesh communicate via direct function
calls or lightweight internal gRPC, depending on deployment topology.

### 11.2 APIs Between Modules
See Appendix D for full request/response schemas (`/v2/diagnose` REST
contract, internal gRPC service definitions for Vision, Embedding, and
Synthesis services).

### 11.3 Shared Schemas
A common "enriched context" schema (retrieved chunks + tool outputs +
guardrail flags) is shared between the orchestration layer and the LLM
input assembly step, so any module producing context data conforms to
one contract.

### 11.4 Database Interactions
Qdrant client (vector search), Redis client (caching/session), GBT model
served as a lightweight internal microservice (REST or gRPC) — no direct
LLM-to-database access; all data access is mediated through the
orchestration layer.

### 11.5 Orchestration Framework
No general-purpose agent framework (LangChain/LangGraph/CrewAI) is used
for the low-latency fast/compound path — the deterministic parallel
fan-out is implemented directly in the Java backend (or FastAPI router)
for full latency control, avoiding the overhead such frameworks add for
a case that doesn't need dynamic multi-step planning. A framework such as
LangGraph is a reasonable candidate **only** for the slow-path agentic
fallback (§10.7) where genuine dynamic tool orchestration is needed and
the latency budget is already relaxed.

---

## 12. Computational Requirements

### 12.1 Hardware Requirements
| Workload | Recommended hardware |
|---|---|
| Vision, intent extractor, embedders, guardrails classifiers | GPU (T4/L4-class) or CPU with ONNX Runtime, regionally deployed |
| Synthesis LLM (2–4B, FP8, vLLM) | GPU (A100/H100-class), batched serving |
| GBT yield model | CPU-only, no GPU required |
| ASR/TTS | GPU (T4/L4-class), streaming-capable |

### 12.2 Memory Requirements
- Vector DB: scales with corpus size (policy/KCC chunk count × 768-dim
  float vectors); yield-cache collection is much smaller (structured
  feature dimension, §7.3).
- Redis: sized to hot-query working set (embedding cache, price/weather
  cache, session state) — modest relative to vector DB.

### 12.3 Expected Inference Latency (Summary)
*Architectural estimates — see §1.4 and Appendix G. Not yet benchmarked
on real hardware/traffic.*

| Path | Latency |
|---|---|
| Fast path (cache hit, image+intent) | ~35–50ms |
| Slow path (cache miss, simple or compound query) | ~210–290ms |
| Voice input, added before pipeline starts | +150–400ms ASR finalization (+20–40ms if dialect normalization triggered) |
| Voice output | streamed alongside generation, not stacked after |

### 12.4 Storage Requirements
| Artifact | Approx. size |
|---|---|
| Distilled LLM checkpoint (2–4B, FP16) | ~4–8GB (~2–4GB at FP8) |
| ViT-Small checkpoint | ~90MB |
| MuRIL checkpoint (FP16) | ~900MB |
| Vector DB | scales with document/chunk corpus size |

---

## 13. Design Decisions and Trade-offs

| Decision | Alternative rejected | Why | Confidence |
|---|---|---|---|
| Distill 12B → 2–4B for synthesis | Keep 12B in critical path | 12B cannot fit the 150–180ms generation budget under any realistic batching | High — decode-time physics, not an assumption |
| GBT for yield, vector cache only as a lookup layer | Embed yield outcomes directly in RAG index | k-NN averaging underperforms GBT on tabular regression; would also pollute the semantic policy index | High — well-established for tabular data |
| Upfront intent/entity extraction + parallel fan-out | Classic ReAct dynamic tool-calling for all queries | Sequential agent reasoning turns stack 150–180ms each — unaffordable for compound queries under the latency target | High — same decode-time argument as above |
| Custom lightweight orchestration for fast/compound path | General agent framework (LangChain/LangGraph) everywhere | Framework overhead not justified when tool selection is deterministic and known upfront | Medium — depends on actual framework overhead, not yet measured on this stack |
| Self-hosted streaming ASR/TTS | Cloud speech APIs | Streaming control is required to make voice interactions feel responsive; cloud APIs vary in streaming support and add per-call cost at scale | Medium — cost/quality trade-off depends on actual vendor pricing and self-hosted model quality, both to be validated |
| Two-stage rule/classifier-based guardrails | LLM-based guardrails (a second generation pass) | Avoids paying generation latency twice; rule-based checks are also more auditable for a safety-critical domain | Medium — recall of the rule-based approach vs. an LLM-based check needs empirical comparison (Appendix H) |
| Regional/edge model deployment (in real-world, non-academic setting) | Single central data center | Network RTT dominates the budget on rural connectivity — noted here for completeness even though this milestone assumes good connectivity | Low — out of scope for this milestone's good-connectivity assumption |

---

## 14. Risks and Limitations

| Risk | Description | Mitigation |
|---|---|---|
| Distillation accuracy loss | 2–4B student may lose reasoning breadth vs. 12B teacher | Held-out agri-QA eval set (§7.5) to quantify and bound acceptable loss before deployment |
| Vision accuracy on rare/underrepresented diseases | Long-tail classes have less training data | Track per-class F1, not just aggregate accuracy; flag low-confidence predictions for expert review |
| Dialect ASR immaturity | Non-standard dialects degrade ASR accuracy | Ship standard-language voice first (per original stretch-goal scoping); collect dialect data via usage logs to fine-tune later |
| Guardrails false negatives | A violation could slip past both filter stages | Recall-prioritized training (§7.9); periodic red-teaming of guardrails with adversarial test queries |
| Cache staleness (yield/price) | Seasonal/market shifts make cached answers outdated | TTL + weather-event-triggered invalidation (§9.8) |
| Hallucination residual risk | Grounding + guardrails reduce but don't eliminate hallucination | Explicit user-facing disclaimer ("verify with local agri officer") on financial/dosage-adjacent answers |
| Data/representation bias | Training data likely skews toward well-represented crops, regions, and languages | Monitor per-subgroup performance (crop, region, language) as a standing evaluation practice, not a one-time check |
| Scalability of fast-path template coverage | Requires ongoing curation as new (disease×intent) combos emerge | Log cache-miss patterns; promote recurring novel combos into the template set on a review cadence |

---

## 15. Deliverables Produced

### 15.1 Documents
- FarmerVision_HLD_v2.md — high-level architecture, bottleneck analysis
- FarmerVision_LLD_CriticalPath.md — sequence diagram, API contracts,
  cache/timeout config, guardrails logic, failure modes
- FarmerVision_AIML_LowLatency_Architecture.md — real-world tiered
  fast/slow-path design
- FarmerVision_AIML_Academic_LowLatency.md — ideal-network compute
  architecture, compound-query handling, yield cache design
- FarmerVision_Consolidated_AIML_Architecture.md — master model
  inventory and integrated diagram, including voice I/O
- FarmerVision_Milestone3_Report.md — this report

### 15.2 Proposed Repository Structure
```
farmervision/
├── services/
│   ├── backend-core/          # Java Spring Boot orchestration
│   ├── ml-mesh/
│   │   ├── vision/
│   │   ├── embedding/
│   │   ├── intent-extractor/
│   │   ├── yield-model/
│   │   ├── profitability/
│   │   ├── guardrails/
│   │   ├── synthesis-llm/
│   │   └── voice/ (asr, tts, dialect-normalizer)
├── infra/
│   ├── vector-db/
│   ├── redis/
│   └── kafka/
├── prompts/                   # system prompt + templates (§10)
├── configs/                   # model/hyperparameter configs (§7)
├── docs/                      # this report + supporting docs
└── eval/                      # held-out eval sets, drift monitoring
```

---

## 16. Summary and Next Steps

### 16.1 Summary of Architecture Decisions
A three-tier model architecture — fast classifiers/embedders, tool-based
fact retrieval/computation, and a single distilled-LLM synthesis pass —
built to fit a 200–300ms latency target by keeping the generative model
out of fact retrieval/computation entirely. Retrieval, yield prediction,
and guardrails are handled by purpose-built, non-LLM components; voice
input/output is layered on top via streaming ASR/TTS without altering the
core reasoning pipeline.

### 16.2 Readiness for Milestone 4 (Training)
The architecture and model selections in this report are training-ready
pending: (a) finalized labeled datasets (disease images, agri-QA pairs
for distillation, dialect audio where available), (b) selection of the
exact base checkpoints for the LLM teacher/student pair, and (c) an
evaluation harness for the held-out agri-QA and per-subgroup fairness
checks flagged in §14.

### 16.3 Planned Implementation Activities
1. Assemble and clean training datasets per §7.
2. Run vision classifier fine-tuning; establish per-class F1 baseline.
3. Build the distillation pipeline (teacher inference → synthetic data →
   student fine-tuning) per §7.5.
4. Implement guardrails classifiers and validate recall on adversarial
   test cases.
5. Stand up the fast-path template cache with an initial curated set of
   common (disease×intent) pairs.
6. Integrate and load-test the full pipeline against the latency targets
   in §12.3.
7. Work through Appendix G systematically — each row is a design
   assumption this report currently treats as given; Milestone 4 should
   produce a measured result for each, not just a working system.

---

## Appendix A — Sequence Diagram (Critical Path)

```
Farmer      Backend      Redis      Vision      MuRIL     Vector DB    LLM        Mandi API   Guardrails
  │            │           │          │           │           │         │            │            │
  │─POST /v2/diagnose──────►│          │           │           │         │            │            │
  │            │─embed cache check────►│           │           │         │            │            │
  │            │◄─MISS──────────────────│          │           │         │            │            │
  │            │─ClassifyImage──────────────────────►│         │         │            │            │
  │            │◄─{label, confidence}────────────────│         │         │            │            │
  │            │─Embed(text)─────────────────────────────────►│         │            │            │
  │            │◄─{vector}────────────────────────────────────│         │            │            │
  │            │─VectorSearch (parallel)──────────────────────────────►│            │            │
  │            │─GetMandiPrice (parallel)──────────────────────────────────────────►│            │
  │            │◄─chunks────────────────────────────────────────────────│            │            │
  │            │◄─price/fallback──────────────────────────────────────────────────────│            │
  │            │─GuardrailsPreFilter────────────────────────────────────────────────────────────►│
  │            │◄─{ok, filteredChunks}───────────────────────────────────────────────────────────│
  │            │─Synthesize(enrichedPrompt)────────────────────────────────────────►│            │
  │            │◄─{advice_text}──────────────────────────────────────────────────────│            │
  │            │─GuardrailsPostCheck────────────────────────────────────────────────────────────►│
  │            │◄─{ok}────────────────────────────────────────────────────────────────────────────│
  │◄─200 OK {advice, label, sources}──│
```

## Appendix B — Model Configuration Table

| Model | Precision | Serving framework |
|---|---|---|
| ViT-Small | FP16 | TensorRT |
| MuRIL | FP16 | TensorRT |
| Feature Embedder | FP16 | ONNX Runtime |
| Intent/Entity Extractor | FP16 | ONNX Runtime |
| Synthesis LLM | FP8 | vLLM (continuous batching, PagedAttention, guided decoding) |
| Speculative draft model | FP8 | vLLM speculative decoding |
| ASR | FP16 | Faster-Whisper / CTranslate2 |
| TTS | FP16 | Streaming synthesis, sentence-chunked |
| Guardrails classifiers | INT8 | ONNX Runtime |

## Appendix C — Hyperparameter Summary Table

| Model | Optimizer | LR | Batch size | Epochs |
|---|---|---|---|---|
| Vision Classifier | AdamW | 3e-4 (head) / 3e-5 (backbone) | 64–128 | 20–30 |
| Text Embedder (if fine-tuned) | AdamW | 2e-5 | — | — |
| Feature Embedder | Adam | 1e-3 | 256 | until recall@k plateau |
| Synthesis LLM (distillation) | AdamW | 1e-5–5e-5 | ~256 (effective) | 2–3 |
| ASR fine-tune | AdamW | 1e-5 | small (audio memory) | — |
| Intent/Entity Extractor | AdamW | 3e-5 | 32 | 10–15 |

## Appendix D — API Specifications

See LLD document (FarmerVision_LLD_CriticalPath.md) §2–3 for full
`/v2/diagnose` REST contract and internal gRPC proto definitions
(VisionService, EmbeddingService, ReActAgentService).

## Appendix E — Prompt Template

See §10.2 for the structural template. Full worked prompt examples to be
finalized alongside the distillation training-data preparation in
Milestone 4.

## Appendix F — References to Selected Models
- ViT (Vision Transformer) — image classification backbone
- MuRIL — Indic multilingual text embedding
- Gemma family — synthesis LLM teacher/student lineage
- IndicWhisper / Bhashini — ASR
- IndicTrans2 — dialect/language normalization
- XGBoost / LightGBM — yield prediction

## Appendix G — Assumptions Requiring Empirical Validation

| Assumption | Current basis | Validation method | When |
|---|---|---|---|
| 2–4B distilled model fits ~150–180ms generation budget | Published token/sec figures for comparable model sizes on H100/vLLM | Direct throughput benchmark under realistic concurrent load once distilled checkpoint exists | Milestone 4 |
| Distilled student retains acceptable accuracy vs. 12B teacher | Standard distillation literature | Held-out agri-QA eval set, automated + human-reviewed scoring (§7.5) | Milestone 4 |
| GBT outperforms k-NN/neural nets for yield prediction | General tabular-ML literature, not this dataset | Cross-validated comparison against alternatives on the actual yield dataset | Milestone 4 |
| Fast-path template cache covers ~70–80% of real traffic | Assumed Zipfian distribution of farmer queries | Query-log analysis after initial deployment/beta; revise coverage target based on observed distribution | Post-beta |
| Guardrails rule-based recall is sufficient (no second LLM pass needed) | Design assumption, not measured | Adversarial red-team test set; compare recall against an LLM-based guardrail baseline | Milestone 4 |
| MuRIL outperforms alternatives on code-mixed query retrieval | Known MuRIL pretraining properties | Retrieval eval (recall@k) comparing MuRIL vs. multilingual-e5 on a code-mixed query set | Milestone 4 |
| Vision model (ViT-Small) accuracy target | ImageNet-pretrained transfer learning assumption | Per-class F1 on labeled disease dataset, especially long-tail/rare classes | Milestone 4 |
| Re-ranking threshold (§9.7) correctly identifies ambiguous queries | Design heuristic | Measure retrieval quality with/without conditional re-ranking on a query set with known ambiguity | Milestone 4 |
| Custom orchestration outperforms a framework (LangChain/LangGraph) on latency | Assumed framework overhead, not measured | Benchmark both approaches on the same deterministic fan-out task | Optional, if time allows |

This table is the intended bridge between this design document and
Milestone 4: each row is a specific, falsifiable claim this architecture
depends on, not a general disclaimer that "results may vary."

## Appendix H — Change Log
| Version | Change |
|---|---|
| v1 | Initial single-service architecture (PNG diagram) |
| v2 | HLD revision: caching, parallel tool-calls, guardrails split |
| v3 | LLD critical path with full API/cache/timeout specification |
| v4 | Low-latency (real-world) tiered fast/slow-path architecture |
| v5 | Academic (ideal-network) compute-focused architecture |
| v6 | Compound-query decomposition + yield vector cache added |
| v7 | Consolidated model inventory; voice I/O (ASR/TTS) added |
| v8 | Voice entry point corrected in diagram; ASR confidence confirmation step added |
| v9 | Full Milestone 3 report — all 16 sections + appendices |
| v10 (current) | Added §1.4 Scope/Assumptions, confidence ratings on design decisions (§13), and Appendix G validation plan mapping specific claims to test methods and milestones |
