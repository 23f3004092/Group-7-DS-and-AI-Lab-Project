# Milestone 2 — Dataset Identification, Understanding & Preparation (Executive Short Report)

## A Decoupled, Agentic Multimodal Crop Advisory System for Uttar Pradesh

> [!NOTE]
> **About this Document:** This executive report synthesizes the core data engineering, exploratory data analysis (EDA), preprocessing, dataset integration, group-aware splitting, and training readiness across all four subsystems of AgriAssist. For unabridged raw distribution tables, cell-by-cell EDA charts, and detailed algorithmic proofs, refer to the **Unabridged Report** ([`Milestone_2_Report.md`](./Milestone_2_Report.md)) and the **Technical Appendices** ([`Milestone_2_work/`](./Milestone_2_work/)).

---

## Table of Contents
1. [Introduction & Objectives](#1-introduction--objectives)
2. [Dataset Identification & Architecture Overview](#2-dataset-identification--architecture-overview)
3. [Dataset Description Summary](#3-dataset-description-summary)
4. [Data Governance & Ethical Compliance](#4-data-governance--ethical-compliance)
5. [Key Exploratory Data Analysis (EDA) Findings](#5-key-exploratory-data-analysis-eda-findings)
6. [System Preprocessing Pipeline & Integration](#6-system-preprocessing-pipeline--integration)
7. [Dataset Splitting & Leakage Prevention](#7-dataset-splitting--leakage-prevention)
8. [Final Prepared Datasets & Training Readiness](#8-final-prepared-datasets--training-readiness)
9. [Challenges Encountered & Technical Resolutions](#9-challenges-encountered--technical-resolutions)
10. [Deliverables Produced](#10-deliverables-produced)
11. [Summary & Milestone 3 Activities](#11-summary--milestone-3-activities)
12. [Team Review & Sign-Off](#12-team-review--sign-off)

---

## 1. Introduction & Objectives

In **Milestone 1**, we proposed a **decoupled, agentic multimodal crop advisory system** for smallholder farmers in Uttar Pradesh, focusing on **rice and wheat** cultivation. Building upon that architecture, **Milestone 2** establishes and validates the data foundation across three primary operational subsystems:
1. **Vision Disease Detection Subsystem:** Identifying foliar and spike diseases across Rice and Wheat using controlled laboratory and real-world field imagery.
2. **RAG & NLP Advisory Subsystem:** Grounding agronomic advice and government scheme eligibility using authoritative PDF corpora and Kisan Call Center (KCC) historical Q&A logs.
3. **Crop Yield Prediction Subsystem:** Estimating seasonal district-level yields using national multi-crop and localized Uttar Pradesh agricultural histories enriched with meteorological covariates.

---

## 2. Dataset Identification & Architecture Overview

The system utilizes seven primary and holdout data sources structured across four technical appendices:

| # | Subsystem | Primary Dataset Name | Scope & Volume | Appendix Reference |
|:--:|-----------|----------------------|----------------|--------------------|
| **1** | **Vision (Train/Val/Test)** | **Unified 20-Class Vision Dataset** *(Merged Wheat + Rice Set 1 + Set 2)* | **12,859 images** across 20 canonical classes | [`Appendix A: Vision Preprocessing`](./Milestone_2_work/notebookD_merge_split_documentation.md) |
| **2** | **Vision (Evaluation Only)** | **PlantDoc Field Holdout Dataset** | **2,598 images** (real-world farmer imagery) | Section 9.2 / Unabridged Report |
| **3** | **RAG PDF Corpus** | **Government Agriculture PDF Knowledge Base** | **170 clean PDFs / 1,451 semantic chunks** | [`Appendix B: RAG PDF Report`](./Milestone_2_work/rag_pdf_report.md) |
| **4** | **RAG & NLP Advisory** | **Kisan Call Center (KCC) Q&A Dataset (2020–2025)** | **3,123,029 records** across UP farmer queries | [`Appendix C: KCC Data EDA`](./Milestone_2_work/KCC%20Data%20EDA.md) |
| **5** | **Yield Prediction (Primary)** | **Unified Pan-India Multi-Crop Dataset** | **440,962 records** (124 crops, 35 States/UTs) | [`Appendix D: Multi-Crop Yield Report`](./Milestone_2_work/yield_report.md) |
| **6** | **Yield Prediction (Domain)** | **Localized Uttar Pradesh Rice/Wheat Subset** | **3,996 records** (75 districts, IMD/ICRISAT enriched) | Notebooks 05 & 06 / Unabridged Report |
| **7** | **System Architecture** | **Milestone 2 Implementation Blueprint** | Pipeline workflows & sprint planning | [`Appendix E: Implementation Plan`](./Milestone_2_Implementation_Plan.md) |

---

## 3. Dataset Description Summary

Below is an executive summary of the schema, input formats, and target variables across the system's datasets. For complete column-level data dictionaries, refer to the unabridged report.

```
+----------------------------------------------------------------------------------------------------+
|                                      AGRIASSIST DATASET METRICS                                    |
+--------------------------+--------------------+------------------------+---------------------------+
| Dataset                  | Clean Volume       | Input Format           | Primary Target / Role     |
+--------------------------+--------------------+------------------------+---------------------------+
| Unified Vision Training  | 12,859 images      | 256x256 RGB JPEG       | 20 Disease & Pest Classes |
| PlantDoc Field Holdout   | 2,598 images       | Variable RGB Image     | Field Robustness Test     |
| RAG Agricultural PDFs    | 1,451 chunks       | 512-token text chunks  | MuRIL Vector Index        |
| KCC Farmer Advisories    | 3,123,029 records  | Tabular + Text (EN/HI) | Q&A Retrieval & Intent    |
| Pan-India Multi-Crop     | 440,962 records    | Tabular numeric/cat    | Yield (kg/ha) Regression  |
| UP Localized Rice/Wheat  | 3,996 records      | Tabular numeric/cat    | District Yield Regression |
+--------------------------+--------------------+------------------------+---------------------------+
```

---

## 4. Data Governance & Ethical Compliance

All data sources strictly adhere to open-data licensing and ethical compliance standards:
* **Licensing:** Vision datasets (Rice Set 1/2, Wheat, PlantDoc) are licensed under open research terms (Kaggle/CC-BY). RAG PDFs and KCC records originate from public government portals (data.gov.in, PPQS, UP Agriculture). Yield datasets derive from official Directorate of Economics and Statistics releases.
* **Personally Identifiable Information (PII):** KCC Call IDs are anonymized system identifiers; no farmer phone numbers, names, or addresses are stored.
* **Advisory Safety:** Vision and RAG subsystems incorporate explicit confidence thresholds ($\tau$) and abstention mechanisms to prevent generating unsafe pesticide or treatment recommendations.

---

## 5. Key Exploratory Data Analysis (EDA) Findings

The table below synthesizes the highest-impact quality discoveries uncovered during EDA and their architectural resolutions.

| Subsystem | Critical EDA Finding | Architectural & Preprocessing Action |
|-----------|----------------------|--------------------------------------|
| **Vision (Wheat)** | **45 noisy raw labels** with redundant synonyms and severe class imbalance (`5.6:1`). | Canonicalized raw labels into **15 clean biological classes**; designed class-weighted CrossEntropy + `WeightedRandomSampler`. |
| **Vision (Wheat)** | **646 perceptual hashes appearing across splits** in the original Kaggle folder split. | Discarded original splits; mandated centralized pHash group-aware splitting. |
| **Vision (Rice Set 1)** | **33.3% of Blast images are near-duplicates** (burst-captured frames); Tungro class exhibited a green-background dimension shortcut. | Applied MD5 + pHash Hamming distance ($\le 6$ bits) union-find clustering (`5,932 -> 2,066` sharp frames); letterboxed all images to 256×256 RGB. |
| **RAG PDF Corpus** | Document lengths range from 4.9-page advisories to >100-page policy manuals. | Confirmed document-level retrieval fails; implemented sentence-aware 512-token semantic chunking (`1,451 chunks`). |
| **KCC Dataset** | **99.98% English/Hinglish queries vs. 98.80% Hindi answers**; `98.9%` of Q&A fit within 512 chars. | Mandated cross-lingual multilingual embedding alignment (`MuRIL`); verified 512-character chunk compatibility. |
| **Yield (Multi-Crop)** | Extreme sparsity across secondary crop attributes; coconut units reported in pieces rather than weight. | Converted coconut figures to metric tonnes; applied non-parametric **Random Forest Imputation (`MissForest`)**. |
| **Yield (UP Subset)** | **164 district discontinuity records** caused by administrative district bifurcations post-1997. | Reconstructed historical continuity using parent-district area proportional apportionment (~28% share). |

---

## 6. System Preprocessing Pipeline & Integration

Every subsystem executed an automated preprocessing workflow designed to eliminate leakage, ensure format standardization, and guarantee training readiness:

```
+--------------------------------------------------------------------------------------------------+
|                                    PREPROCESSING PIPELINE SUMMARY                                |
+------------------------------------+--------------------------------+----------------------------+
| Pipeline Stream                    | Primary Transformations        | Final Artifact Location    |
+------------------------------------+--------------------------------+----------------------------+
| Vision (Wheat + Rice S1 + S2)      | Canonicalize labels (45->15),  | data/final/train|val|test  |
|                                    | pHash deduplication, RGB       | master_manifest.csv        |
|                                    | 256x256 letterboxing           | label_to_idx.json          |
+------------------------------------+--------------------------------+----------------------------+
| RAG PDF Knowledge Base             | OCR fallback (300 DPI), clean  | pdf_inventory_clean.csv    |
|                                    | deduplication, sentence-aware  | pdf_chunks.csv             |
|                                    | 512-token semantic chunking    | pdf_chunks.jsonl           |
+------------------------------------+--------------------------------+----------------------------+
| KCC Advisory Logs                  | Drop empty Season col, drop    | kcc_combined_2020_2025.csv |
|                                    | boilerplate weather duplicates |                            |
|                                    | 512-char cross-lingual chunks  |                            |
+------------------------------------+--------------------------------+----------------------------+
| Pan-India Multi-Crop Yield         | Unit standardization, MissForest| production_unified_imputed |
|                                    | Random Forest ML imputation    | .csv                       |
+------------------------------------+--------------------------------+----------------------------+
| UP Localized Rice/Wheat Yield      | District boundary harmonization| train/val/test_yield.csv   |
|                                    | IMD/ICRISAT weather enrichment |                            |
+------------------------------------+--------------------------------+----------------------------+
```

---

## 7. Dataset Splitting & Leakage Prevention

To ensure evaluation metrics reflect genuine real-world generalization, strict leakage-prevention splitting protocols were executed across all datasets:

### 7.1 Unified Vision Group-Aware Stratified Split (Notebook D)
* **Centralized Pool:** Concatenated Wheat (`10,673`), Rice Set 1 (`2,066`), and Rice Set 2 (`120`) into a single **12,859-image pool across 20 classes**.
* **Split Ratios & Small-Class Floor:** 80% Train (`10,275` images) / 10% Validation (`1,292` images) / 10% Test (`1,292` images), enforcing an **8-image minimum evaluation floor** per class (`rice__leaf_smut` split 24 Train / 8 Val / 8 Test).
* **Leakage Proof:** Entire pHash `group_id` clusters (`11,530` unique groups) were assigned indivisibly to a single split. **Verified `LEAKAGE = 0 groups spanning splits`**.

### 7.2 Yield Prediction Chronological Partitioning
To prevent temporal autocorrelation leakage across agricultural seasons, yield data is split strictly chronologically by agricultural start year:
* **Train Set (1997–2018):** 81.5% (`3,256` records) — Historical baseline training cohort.
* **Validation Set (2019–2020):** 7.4% (`296` records) — Out-of-time tuning cohort.
* **Holdout Test Set (2021–2023):** 11.1% (`444` records) — Strict out-of-time evaluation cohort.

---

## 8. Final Prepared Datasets & Training Readiness

All Milestone 2 datasets have been audited, materialized to disk, and verified as **100% Training-Ready** for Milestone 3 execution:

| Dataset / Deliverable | Status | Prepared Volume / Artifact |
|-----------------------|:------:|----------------------------|
| **Unified Vision Training Dataset** | ✅ **Complete** | **12,859 images across 20 classes** (`final/train|val|test`, 273 MB disk footprint) |
| **PlantDoc Holdout Evaluation Set** | ✅ **Complete** | **2,598 field images** reserved strictly for out-of-distribution robustness testing |
| **RAG PDF Knowledge Base Chunks** | ✅ **Complete** | **1,451 semantic chunks** (`pdf_chunks.jsonl`) ready for `MuRIL` vector indexing |
| **KCC Farmer Advisory Logs** | ✅ **Complete** | **3,123,029 records** (`kcc_combined_2020_2025.csv`) ready for Q&A embedding |
| **Pan-India Multi-Crop Yield Dataset** | ✅ **Complete** | **440,962 complete imputed records** (`production_unified_imputed.csv`) across 124 crops |
| **UP Localized Rice/Wheat Yield Subset** | ✅ **Complete** | **3,996 enriched records** partitioned chronologically (`train/val/test_yield.csv`) |

---

## 9. Challenges Encountered & Technical Resolutions

1. **Wheat Label Noise & Synonyms:** Solved via canonical mapping dictionaries combining biological synonyms into 15 robust classes.
2. **Burst-Frame Duplicate Explosion (Rice Set 1):** Solved via union-find perceptual hashing clusters (`5,932 -> 2,066` sharp frames).
3. **Tungro Background Shortcut:** Solved via 256×256 RGB letterboxing and targeted training-time background randomization.
4. **Scanned PDF Extraction Failures:** Solved via automated 300 DPI Tesseract OCR fallback (`pytesseract`).
5. **District Boundary Changes (UP Yield):** Solved via parent-district proportional area apportionment.

---

## 10. Deliverables Produced

1. **System Architecture & Implementation Plan:** [`docs/Milestone_2_Implementation_Plan.md`](./Milestone_2_Implementation_Plan.md)
2. **Unabridged Comprehensive Report:** [`docs/Milestone_2_Report.md`](./Milestone_2_Report.md)
3. **Vision Preprocessing & Integration Reports (Mahesh's Work):**
   * [`docs/Milestone_2_work/wheat_preprocessing_documentation.md`](./Milestone_2_work/wheat_preprocessing_documentation.md)
   * [`docs/Milestone_2_work/rice_set1_preprocessing_documentation.md`](./Milestone_2_work/rice_set1_preprocessing_documentation.md)
   * [`docs/Milestone_2_work/rice_set2_preprocessing_documentation.md`](./Milestone_2_work/rice_set2_preprocessing_documentation.md)
   * [`docs/Milestone_2_work/notebookD_merge_split_documentation.md`](./Milestone_2_work/notebookD_merge_split_documentation.md)
   * [`docs/Milestone_2_work/notebook_training_pipeline_design.md`](./Milestone_2_work/notebook_training_pipeline_design.md)
4. **RAG PDF Corpus Technical Report (Harliv's Work):**
   * [`docs/Milestone_2_work/rag_pdf_report.md`](./Milestone_2_work/rag_pdf_report.md)
5. **KCC Advisory Dataset EDA Report (Aneeqa's Work):**
   * [`docs/Milestone_2_work/KCC Data EDA.md`](./Milestone_2_work/KCC%20Data%20EDA.md)
6. **Pan-India Multi-Crop Yield Technical Report (Tanmay's Work):**
   * [`docs/Milestone_2_work/yield_report.md`](./Milestone_2_work/yield_report.md)
7. **Executable Notebooks:**
   * Vision: `rice-leaf-disease-dataset-EDA.ipynb`, `rice-leaf-disease-dataset-set-2-eda.ipynb`, `wheat-dataset-EDA.ipynb`
   * RAG & KCC: `PDF_Corpus_EDA.ipynb`, `PDF_Chunking.ipynb`, `03_kcc_rag_eda.ipynb`
   * Yield: `05_yield_eda.ipynb`, `06_yield_preprocessing.ipynb`, `07_Yield_EDA+ preprocessing.ipynb`

---

## 11. Summary & Milestone 3 Activities

With all datasets cleaned, integrated, deduplicated, and split with zero cross-split leakage, **Milestone 3 (Training & System Evaluation)** will execute:
1. Fine-tuning `EfficientNet-B0` on the 20-class Unified Vision Dataset using class-weighted loss and `WeightedRandomSampler`.
2. Evaluating lab-to-field robustness on the held-out `PlantDoc` dataset and conducting Tungro Grad-CAM saliency audits.
3. Embedding all 1,451 PDF chunks and deduplicated KCC Q&A pairs using `MuRIL` into a ChromaDB vector store.
4. Benchmarking retrieval performance (`Recall@5`, `MRR`, `nDCG`).
5. Training crop yield regression models (`XGBoost`, `LightGBM`, `Random Forest`) on both national and UP domain datasets.

---

## 12. Team Review & Sign-Off

| # | Team Member | Role | Reviewed & Approved | Date | Signature |
|:-:|-------------|------|:-------------------:|:----:|-----------|
| 1 | **Mahesh** | Comprehensive Vision EDA, Preprocessing, Integration & Training Pipeline Design | ☑ | 2026-07-09 | `Signed - Mahesh` |
| 2 | **Harliv** | Comprehensive RAG PDF Corpus Collection, EDA, Cleaning & Semantic Chunking | ☑ | 2026-07-09 | `Signed - Harliv` |
| 3 | **Lokesh** | Milestone 2 Implementation Plan, Identifying Vision Data Sources, Primary Vision EDA, UP Rice/Wheat Yield Subset & Report Authoring | ☑ | 2026-07-09 | `Signed - Lokesh` |
| 4 | **Aneeqa** | Comprehensive KCC Dataset Aggregation (3.12M), EDA & Multilingual RAG Preparation | ☑ | 2026-07-09 | `Signed - Aneeqa` |
| 5 | **Tanmay** | Primary Multi-Crop Yield Dataset Unification, EDA & MissForest Preprocessing | ☑ | 2026-07-09 | `Signed - Tanmay` |

**Document Version:** Executive Short Report (Milestone 2) · **Prepared:** July 2026
