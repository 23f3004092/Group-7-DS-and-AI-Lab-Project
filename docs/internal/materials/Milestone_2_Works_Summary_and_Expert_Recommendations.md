# Milestone 2 — Comprehensive Works Summary & Expert AI Engineering Recommendations

This document provides a complete inventory and architectural significance assessment of all data engineering, exploratory data analysis (EDA), preprocessing, and dataset integration work completed across all four streams of the **Decoupled Agentic Multimodal Crop Advisory System for Uttar Pradesh** during Milestone 2. 

In addition, it provides an executive priority matrix and detailed engineering recommendations ranked from **Highest to Lowest Priority**, evaluated against **Architectural Impact**, **Implementation Complexity**, and the **Milestone 3 Project Timeline**.

---

## Part 1: Comprehensive Inventory of All Works Done & Their Architectural Significance

```
Decoupled Agentic Crop Advisory System — Milestone 2 Data Streams
├── 1. KCC Agronomic Q&A Stream      (1.46M Chunks · Cleaned & Tagged for RAG)
├── 2. Government Policy PDF Stream  (170 PDFs · ~1,450 Chunks · Authoritative Domain Knowledge)
├── 3. Tabular Yield Prediction      (440,962 Rows · 16 Cols · MissForest Harmonized Time-Series)
└── 4. Field-Robust Vision Dataset   (12,859 Images · 20 Classes · Group-Aware Zero-Leakage Split)
```

---

### Stream 1: Kisan Call Centre (KCC) Agronomic Q&A & RAG Pipeline Preparation
**Key Documents:** [KCC Data EDA.md](../../../outputs/reports/KCC%20Data%20EDA.md), [KCC Preprocessing.md](../../../outputs/reports/KCC%20Preprocessing.md)

#### 1. What Was Done
- **Data Ingestion & Geographic Scope Verification:** Loaded 3,123,029 raw records spanning 2020–2025 and verified 100% origin from Uttar Pradesh (`StateName == 'Uttar Pradesh'`).
- **Domain Noise Exclusion:** Applied rigorous inclusion/exclusion filtering to eliminate ~1.4M non-agronomic records (government financial schemes, market prices, subsidies, credit, insurance, weather reports), retaining **1,701,442 strictly agronomic records** (Cereals, Vegetables, Oilseeds, Pulses, Fruits, Plant Protection, Nutrient Management).
- **Data Governance & PII Redaction:** Executed `remove_pii()` to strip farmer phone numbers, email addresses, and identification numbers across all records to guarantee privacy compliance.
- **Missing Value Imputation & Indic Normalization:**
  - Solved a 100% missing `Season` column across 1.7M rows by deterministically inferring agricultural seasons (*Kharif*, *Rabi*, *Zaid*) from call timestamp months.
  - Applied Unicode NFC normalization (`normalize_indic_text()`) to resolve Devanagari/Hinglish script encoding issues and zero-width artifacts.
- **Exact & Pair Deduplication:** Removed 241,620 exact and Q&A-pair duplicates (12.2% reduction), resulting in **1,459,692 unique records**.
- **MuRIL-Optimized Semantic Chunking & Rich Metadata Tagging:**
  - Formatted records into structured pairs (`Question: {query}\nAnswer: {answer}`).
  - Implemented sentence-boundary aware chunking at 512 characters with 50-character overlap (**98.8% fit within a single chunk**), generating **1,468,625 RAG-ready chunks** (`kcc_chunks_rag.jsonl`, 1.18 GB).
  - Tagged every chunk with structured metadata (`crop`, `district`, `block`, `season`, `query_type`, `category`, `year`, `month`, `language`).

#### 2. Architectural Significance
- **Direct Support for Grounded Generation (Milestone 1 Objective O4):** By removing financial/scheme noise and tagging fine-grained metadata, the system enables deterministic metadata filtering during vector retrieval (`WHERE crop = 'Wheat' AND season = 'Rabi'`), directly preventing cross-crop and cross-scheme hallucinations.
- **Embedding Efficiency:** Aligning chunk lengths to 512 characters prevents MuRIL transformer token truncation, ensuring full semantic representation of code-mixed farmer queries.

---

### Stream 2: RAG Government Policy PDF Knowledge Base
**Key Documents:** [rag_pdf_report.md](../../../outputs/reports/rag_pdf_report.md)

#### 1. What Was Done
- **Corpus Sourcing & Acquisition:** Sourced **187 authoritative government PDFs** across four distinct domains:
  - PPQS (Plant Protection, Quarantine & Storage) Advisories: 90 PDFs
  - Uttar Pradesh Agriculture Contingency Plans (ACP): 74 PDFs
  - Official Government Agricultural Schemes: 11 PDFs
  - ICAR Publications & Farming Handbooks: 12 PDFs
- **Automated Validation & Corrupted PDF Quarantining:** Built a dual-stage extraction pipeline (`pdfplumber` + OCR verification) that audited every PDF and quarantined **17 unreadable, empty, or corrupted files** (`excluded_unreadable_docs.csv`), retaining **170 pristine documents**.
- **Metadata Extraction & Semantic Chunking:**
  - Extracted document-level metadata (`filename`, `source_folder`, `page_count`, `word_count`, `language`, `year`, `garbage_character_ratio`).
  - Executed sentence-aware semantic chunking centered around **450–520 tokens** with overlap, producing **~1,450 RAG chunks** (`pdf_chunks.jsonl`).

#### 2. Architectural Significance
- **Exceeded Milestone 2 Target:** The Milestone 2 implementation plan flagged UP Policy PDFs as an open question with a baseline target of 20–30 PDFs. Delivering 170 authoritative government PDFs provides the Agentic LLM with an authoritative ground-truth reference for pesticide schedules and regional drought/flood contingency measures.
- **Dual-Corpus Complementarity:** While KCC call logs provide colloquial, experiential agronomic Q&A, the PDF corpus provides institutional regulatory grounding (chemical concentrations, safety intervals), enabling high RAGAS Faithfulness scores.

---

### Stream 3: Classical Machine Learning Yield Prediction Dataset
**Key Documents:** [yield_report.md](../../../outputs/reports/yield_report.md), [UP Crop Yield Data Research.md](../../../outputs/reports/UP%20Crop%20Yield%20Data%20Research.md)

#### 1. What Was Done
- **Multi-Source Time-Series Harmonization:** Combined four disparate historical agriculture datasets spanning 1997–2024 (`Crop Recommendation dataset.csv`, `crop_yield.csv`, `crop-wise-area-production-yield.csv`, `DES-District-Data-For-2024-25.csv`) into a unified 16-column schema (`crop`, `year`, `state`, `district`, `season`, `area`, `production`, `yield`, `annual_rainfall`, `fertilizer`, `pesticide`).
- **Granularity Resolution & Unit Normalization:** Deduplicated overlapping records by prioritizing district-level observations over state-level aggregates, and standardized unit discrepancies (e.g., converting coconut counts from pieces to tonnes).
- **Non-Parametric Multivariate Imputation:** Applied Ordinal Encoding followed by non-parametric random forest imputation (`MissForest`) over **440,962 rows across 124 crops and 35 states**, preserving non-linear co-dependencies between rainfall, fertilizer, area, and yield (`production_unified_imputed.csv`).

#### 2. Architectural Significance
- **Agentic Tool Calling Readiness (Milestone 1 Objective O5):** Establishes the foundational tabular schema required for the Agentic LLM to execute Python tool calls (`predict_yield(crop, area, district)`).
- **Preserved Non-Linear Feature Interactions:** Using `MissForest` instead of univariate mean imputation prevents artificial flattening of the relationship between fertilizer intensity and yield.

---

### Stream 4: Field-Robust Computer Vision Disease Dataset
**Key Documents:** [Rice_Leaf_Disease_EDA.md](../../../outputs/reports/Rice_Leaf_Disease_EDA.md), [Rice_leaf_disease_dataset_set2_documentation.md](../../../outputs/reports/Rice_leaf_disease_dataset_set2_documentation.md), [rice_set1_preprocessing_documentation.md](../../../outputs/reports/rice_set1_preprocessing_documentation.md), [rice_set2_preprocessing_documentation.md](../../../outputs/reports/rice_set2_preprocessing_documentation.md), [wheat_disease_dataset_EDA.md](../../../outputs/reports/wheat_disease_dataset_EDA.md), [wheat_preprocessing_documentation.md](../../../outputs/reports/wheat_preprocessing_documentation.md), [notebookD_merge_split_documentation.md](../../../outputs/reports/notebookD_merge_split_documentation.md), [notebook_training_pipeline_design.md](../../../outputs/reports/notebook_training_pipeline_design.md)

#### 1. What Was Done
- **Unified 20-Class Taxonomy & Scope Enforcement:** Integrated three independent datasets into a unified **12,859-image dataset across 20 classes** (15 Wheat diseases + 5 Rice diseases: `bacterial_blight`, `blast`, `brown_spot`, `leaf_smut`, `tungro`), standardized to 256×256 RGB letterboxed images (`273 MB`).
- **Label Repair & De-Corruption:** Identified and repaired severe folder-name inconsistencies in the Wheat dataset that originally inflated the class count to 45 (e.g., collapsing `Black Rust`, `black_rust_test`, and `black_rust_valid` into canonical `wheat__black_rust`).
- **Group-Aware Zero-Leakage Splitting:**
  - Discovered severe internal near-duplicate clusters (~44% in Wheat; ~79–83% in Rice Set 1) and 646 cross-split collisions in raw Kaggle splits.
  - Executed a central **Group-Aware Stratified 80/10/10 Split** (`GroupShuffleSplit` on perceptual hash `group_id`s), guaranteeing **zero cross-split data leakage** while keeping class proportions tight.
- **Domain-Gap & Shortcut Defense (`rice__tungro`):**
  - Uncovered a critical spurious correlation where `rice__tungro` images were photographed zoomed-out against brown soil at 4:3 native resolutions (unlike 300×300 leaf close-ups for other classes).
  - Formulated a 3-part defense in Notebook E design: 256×256 letterboxed standardization, background-oriented augmentation, and explicit non-soil Tungro robustness testing.
- **Rare-Class Safeguards (`rice__leaf_smut`):** Protected the 40-image `leaf_smut` class by enforcing a minimum evaluation floor ($\ge 8$ images in Val and Test) and designing a combined `WeightedRandomSampler` + Class-Weighted Loss strategy for live training.

#### 2. Architectural Significance
- **Closing the Lab-to-Field Gap (Milestone 1 Objective O1 & O2):** Eliminating cross-split near-duplicate leakage ensures that validation/test F1-scores reflect true generalization rather than memorized video frames.
- **Decoupled Vision Handoff Integrity:** Establishing a frozen `label_to_idx.json` and canonical `crop__disease` prefixes guarantees deterministic label handoffs from the CNN Semantic Router to the reasoning LLM Agent.

---

## Part 2: Prioritized Recommendations Matrix (Highest to Lowest Priority)

Every recommendation below is evaluated against **Architectural Impact**, **Implementation Complexity / Effort**, and the **Milestone 3 Project Timeline**.

| Priority | Recommendation | Stream | Impact | Complexity | M3 Timeline Phase |
|:---:|:---|:---:|:---:|:---:|:---:|
| **P1** | **Temporal & Geographic Splitting Before Imputation** | Yield (S3) | **CRITICAL** | Low | Sprint 1 (Pre-Training Setup) |
| **P2** | **Structured Top-2 Diagnosis JSON + TTA Entropy Abstention** | Vision (S4) | **HIGH** | Medium | Sprint 1 (Vision Service API) |
| **P3** | **MinHash LSH Near-Duplicate Compression (~70% DB Reduction)** | KCC (S1) | **HIGH** | Low–Med | Sprint 1 (Vector DB Indexing) |
| **P4** | **Parent-Child ("Small-to-Big") & Table-Preserving Indexing** | PDF (S2) | **HIGH** | Medium | Sprint 1–2 (RAG Ingestion) |
| **P5** | **Hierarchical Multi-Task Classification Head (Crop → Disease)** | Vision (S4) | **HIGH** | Medium | Sprint 2 (Model Training) |
| **P6** | **Query Expansion / Dual-Script Router for Vernacular Retrieval** | RAG (S1+S2) | **HIGH** | Medium | Sprint 2 (RAG Retrieval Tuning) |
| **P7** | **Agronomic Interaction & Lag Feature Engineering** | Yield (S3) | **MEDIUM** | Low–Med | Sprint 2 (Yield ML Modeling) |
| **P8** | **Deterministic Dosage Lookup Table Tool (Function Calling)** | PDF (S2) | **MEDIUM** | Low | Sprint 2–3 (Agent Tooling) |
| **P9** | **Supervised Contrastive Learning (SupCon) Pre-Training** | Vision (S4) | **MEDIUM** | High | Sprint 3 (Fine-Tuning Polish) |
| **P10** | **Offline LLM Synthetic FAQ Canonicalization** | KCC (S1) | **LOW–MED** | High | Stretch / Post-Pilot |

---

## Part 3: Detailed Breakdown of Recommendations

### Tier 1: Highest Priority (Must Implement in Milestone 3 Baseline)

#### P1. Eliminate Lookahead Bias in Yield Imputation via Temporal & Geographic Splitting
- **Target Stream:** Stream 3 (Yield Prediction)
- **Why it is Highest Priority:** Currently, `production_unified_imputed.csv` contains all 35 states and was imputed across all years (1997–2024) *before* splitting. A random forest imputer (`MissForest`) trained on 2024 data can subtly leak future agricultural trends into historical rows. Furthermore, the Agent requires a model localized strictly to Uttar Pradesh Rice and Wheat.
- **Implementation Plan:**
  1. Filter raw records to `StateName == 'Uttar Pradesh'` and `Crop IN ('Rice', 'Wheat')`.
  2. Perform a strict **Temporal Split**: Train ($\le 2020$), Validation ($2021\text{--}2022$), Test ($2023+$).
  3. Fit `MissForest` exclusively on the historical Training split ($\le 2020$) and apply it forward to Validation and Test.

#### P2. Structured Top-2 Diagnosis JSON Payload + TTA Entropy Abstention
- **Target Stream:** Stream 4 (Computer Vision Router)
- **Why it is Highest Priority:** Passing a flat string label (`[Wheat Yellow Rust, Conf: 91%]`) to the Agentic LLM limits diagnostic reasoning. Furthermore, standard CNN softmax scores can be overconfident on out-of-distribution field images.
- **Implementation Plan:**
  1. At inference time, pass 4–8 Test-Time Augmentation (TTA) views (center crop, horizontal flip, slight brightness shift) through the CNN backbone.
  2. Compute mean probability and **predictive entropy across TTA views**. If entropy exceeds threshold $\tau$, trigger `"abstention_triggered": true`.
  3. Emit a structured JSON payload to the ReAct Agent:
     ```json
     {
       "vision_diagnosis": {
         "primary_label": "wheat__yellow_rust",
         "primary_confidence": 0.88,
         "secondary_label": "wheat__brown_rust",
         "secondary_confidence": 0.09,
         "abstention_triggered": false
       }
     }
     ```

#### P3. MinHash LSH Near-Duplicate Compression of KCC Chunks
- **Target Stream:** Stream 1 (KCC RAG Knowledge Base)
- **Why it is Highest Priority:** While exact duplicates were removed, indexing 1.46M chunks where ~70% are minor rephrasings (*"gehun me pila rog"* vs. *"gehun ki patti pili ho rahi hai"*) clutters the vector database's `Top-K=3` context window with redundant responses and increases VRAM/disk requirements.
- **Implementation Plan:**
  1. Run **MinHash LSH** (Locality Sensitive Hashing on character 3-grams) on `QueryText` within each `(Crop, QueryType)` bucket.
  2. Merge near-duplicate clusters into a single canonical chunk per cluster.
  3. Compress the vector index from **1.46M chunks down to ~400K unique chunks** (~70% reduction in VRAM and index search latency).

#### P4. Parent-Child ("Small-to-Big") & Layout-Aware Table Indexing
- **Target Stream:** Stream 2 (Government PDF RAG Corpus)
- **Why it is Highest Priority:** Standard 500-token chunking across PPQS and Contingency Plan PDFs can slice through structured pesticide dosage tables or isolate chunks without their regional document context.
- **Implementation Plan:**
  1. Index **Child Chunks** (~150 tokens or individual table rows) in MuRIL vector space for highly precise similarity search.
  2. Map each child chunk to its **Parent Document Section** (~600–1,000 tokens). When a child matches a query, pass the complete Parent Section to the LLM context window.

---

### Tier 2: Medium Priority (High Value for Milestone 3 System Polish)

#### P5. Hierarchical Multi-Task Classification Head (Crop Species → Disease)
- **Target Stream:** Stream 4 (Computer Vision Classifier)
- **Why it is High Value:** A flat 20-class softmax head permits cross-crop misclassifications (e.g., predicting a Wheat disease on a Rice leaf if unusual soil lighting confuses the backbone).
- **Implementation Plan:**
  1. Replace the 20-class linear classification head with a 2-stage Hierarchical Head:
     - **Head 1 (Coarse):** Predicts Crop Species (`Rice` vs. `Wheat` vs. `OOD`).
     - **Head 2 (Fine):** Predicts the specific disease conditional on the predicted crop species.

#### P6. Query Expansion & Dual-Script Router for Vernacular-to-English Retrieval
- **Target Stream:** Streams 1 & 2 (RAG Retrieval)
- **Why it is High Value:** Nearly 100% of official policy PDFs are in English, while farmers query in Hinglish/Hindi. Technical chemical names in formal PDFs (*"Azoxystrobin 18.2% SC"*) may not cleanly match informal farmer queries (*"patti par dhaabe ke liye dawai"*).
- **Implementation Plan:**
  1. Configure the LLM ReAct Agent to perform an automatic query expansion pre-processing step before vector search:
     - *Raw Query:* `"gehun me bhura rog"`
     - *Expanded RAG Query:* `"gehun me bhura rog | Wheat Brown Rust | Puccinia triticina foliar fungicide treatment UP"`

#### P7. Agronomic Interaction & Lag Feature Engineering
- **Target Stream:** Stream 3 (Yield Prediction ML Model)
- **Why it is High Value:** Agricultural yield is governed by input intensity ratios, seasonal weather anomalies, and historical momentum rather than raw area alone.
- **Implementation Plan:**
  1. Engineer domain-specific interaction variables:
     - Input Intensity: $\text{Fertilizer per Ha} = \frac{\text{fertilizer}}{\text{area}}$, $\text{Pesticide per Ha} = \frac{\text{pesticide}}{\text{area}}$
     - Climate Anomaly: Standardized Z-score of annual rainfall relative to each UP district's 20-year historical mean.
     - Momentum Lags: 1-year and 2-year lagged district yield per crop.

#### P8. Deterministic Pesticide/Dosage Lookup Table Tool
- **Target Stream:** Stream 2 (Agent Tooling)
- **Why it is High Value:** Generative LLMs can occasionally transpose numeric figures or hallucinate dosage rates when summarizing dense PDF prose.
- **Implementation Plan:**
  1. Extract structured chemical dosage schedules from PPQS advisories into an offline **SQLite / JSON Lookup Table**.
  2. Expose a deterministic Python tool (`lookup_pesticide_schedule(crop, disease)`) for the ReAct Agent to invoke alongside vector RAG.

---

### Tier 3: Stretch / Longer-Term Enhancements (Low Priority / Higher Complexity)

#### P9. Supervised Contrastive Learning (SupCon) & Copy-Paste Augmentation
- **Target Stream:** Stream 4 (Computer Vision)
- **Why it is Stretch:** Requires significant GPU training time and specialized segmentation masks.
- **Implementation Plan:** Run 5–10 epochs of SupCon pre-training on the CNN backbone to sharpen decision boundaries between visually similar rust lesions (`black_rust`, `brown_rust`, `yellow_rust`). Optionally use FastSAM masks to paste diseased leaves onto random backgrounds to counteract texture/soil bias.

#### P10. Offline LLM Synthetic FAQ Canonicalization
- **Target Stream:** Stream 1 (KCC RAG Knowledge Base)
- **Why it is Stretch:** High API/compute cost to rewrite hundreds of thousands of clusters.
- **Implementation Plan:** Run an offline batch pipeline to rewrite cluster centroids of KCC call logs into self-contained, canonical Q&A pairs.

---

## 4. Summary Verification & Sign-Off Checklist for M3 Entry

- [x] **Stream 1 (KCC RAG):** 1.46M chunks prepared (`kcc_chunks_rag.jsonl`); MinHash LSH compression staged for M3 indexing.
- [x] **Stream 2 (Policy PDFs):** 170 authoritative PDFs chunked (~1,450 chunks); Parent-Child indexing strategy documented.
- [x] **Stream 3 (Yield Tabular):** 440,962 rows harmonized; temporal UP Rice/Wheat split rule documented before M3 modeling.
- [x] **Stream 4 (Vision Dataset):** 12,859 images across 20 classes split 80/10/10 with **zero leakage** (`label_to_idx.json` frozen); TTA abstention and hierarchical head staged.
