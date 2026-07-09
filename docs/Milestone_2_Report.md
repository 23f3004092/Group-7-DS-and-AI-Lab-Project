# Milestone 2 — Dataset Identification, Understanding & Preparation

## A Decoupled, Agentic Multimodal Crop Advisory System for Uttar Pradesh

### Contents

- [Milestone 2 — Dataset Identification, Understanding \& Preparation](#milestone-2--dataset-identification-understanding--preparation)
  - [A Decoupled, Agentic Multimodal Crop Advisory System for Uttar Pradesh](#a-decoupled-agentic-multimodal-crop-advisory-system-for-uttar-pradesh)
    - [Contents](#contents)
  - [1. Introduction](#1-introduction)
    - [Objectives of Milestone 2](#objectives-of-milestone-2)
    - [Relationship Between the Datasets and Project Goals](#relationship-between-the-datasets-and-project-goals)
    - [2. Dataset Identification](#2-dataset-identification)
    - [3. Dataset Description](#3-dataset-description)
  - [4. Data Governance](#4-data-governance)
    - [Dataset-specific Notes](#dataset-specific-notes)
    - [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
    - [6. Data Preprocessing](#6-data-preprocessing)
    - [7. Dataset Integration (if multiple datasets)](#7-dataset-integration-if-multiple-datasets)
    - [8. Data Augmentation (if applicable)](#8-data-augmentation-if-applicable)
    - [9. Dataset Splitting](#9-dataset-splitting)
    - [10. Final Prepared Dataset](#10-final-prepared-dataset)
    - [11. Challenges Encountered](#11-challenges-encountered)
    - [12. Deliverables Produced](#12-deliverables-produced)
    - [13. Summary and Next Steps](#13-summary-and-next-steps)
  - [Team Review \& Sign-Off](#team-review--sign-off)

---

## 1. Introduction

In **Milestone 1**, we proposed a **decoupled, agentic multimodal crop advisory system** for smallholder farmers in Uttar Pradesh, focusing on **rice and wheat** cultivation. The proposed architecture consists of three major components: a vision subsystem for crop disease identification, a Retrieval-Augmented Generation (RAG) subsystem that retrieves information from government schemes, agricultural advisories, and Kisan Call Centre (KCC) data, and a quantized Large Language Model (LLM) that generates grounded, context-aware recommendations.

Building upon this foundation, **Milestone 2 focuses on identifying, understanding, and preparing the datasets required for each subsystem** through comprehensive Exploratory Data Analysis (EDA) and preprocessing. For the vision subsystem, three complementary datasets are used: the **Rice Leaf Disease Dataset**, **Wheat Disease Dataset**, and **PlantDoc Dataset**. The Rice and Wheat datasets provide well-labelled disease images captured under controlled conditions, while PlantDoc contributes real-world field images containing variations in lighting, backgrounds, and image quality, improving the model's ability to generalize to practical farming environments. The RAG subsystem utilizes government schemes, agricultural advisories, and KCC resources to build a reliable knowledge base, while the yield prediction subsystem relies on historical agricultural datasets for district-level yield estimation.

### Objectives of Milestone 2

* Identify and document the datasets required for the vision, RAG, and yield prediction subsystems.
* Perform exploratory data analysis to understand the structure, quality, and characteristics of each dataset.
* Detect and address issues such as missing values, duplicate records, corrupted samples, and inconsistent formats.
* Clean and preprocess the datasets to prepare them for model training and evaluation.
* Produce well-documented, training-ready datasets for use in the subsequent milestones.

### Relationship Between the Datasets and Project Goals

Each dataset directly supports the objectives defined in Milestone 1. The **Rice Leaf Disease**, **Wheat Disease**, and **PlantDoc** datasets enable robust crop disease detection by combining controlled and real-world images. The RAG corpus provides reliable agricultural knowledge for grounded response generation, while the historical yield dataset supports district-level yield prediction. Together, these datasets form the foundation for developing the proposed multimodal crop advisory system.

---

### 2. Dataset Identification

**2.1 Vision / Disease Detection Dataset(s)**

* **Dataset name(s):** Rice Leaf Disease Dataset, Wheat Disease Dataset and PlantDoc Dataset
* **Source(s) and download links:**  
  * Rice Leaf Diseases Dataset: Kaggle – https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
  * Wheat Plant Diseases Dataset: Kaggle – https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases
  * PlantDoc Dataset: Kaggle – https://www.kaggle.com/datasets/andresmgs/plantdec
* **Public/private/licensed status:** All three datasets are publicly available through Kaggle for research and educational purposes.
* **Purpose:** The datasets are used to develop the vision subsystem for crop disease detection. The Rice and Wheat datasets provide labelled disease images, while PlantDoc contributes real-world field images for evaluating model robustness.
* **Why each dataset was selected:** The datasets cover the project's target crops (rice and wheat) and combine controlled images with real-world field conditions, improving the model's ability to generalize.
* **Alternatives considered:** PlantVillage was considered as an additional dataset but was not used as the primary dataset because it mainly contains laboratory-captured images with limited real-world variability, and no rice/wheat-related disease classes are present in it.

**2.2 RAG / NLP PDFs (UP govt PDFs, schemes, Farming Handbooks)**

* **Dataset name(s):** UP State Government agricultural scheme PDFs, central scheme operational guidelines, and ICAR/PPQS advisories.
* **Source(s) and download links:**
  * PIB Press Release (scheme announcement) — https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2002012&reg=48&lang=2
  * UP Agriculture Central Guidelines — https://agridarshan.up.gov.in/central-guideline
  * PPQS Advisories — https://ppqs.gov.in/advisories-section?page=0
  * Agriculture Contingency Plan (Uttar Pradesh) — https://agriwelfare.gov.in/en/AgricultureContigencyPlan/UTTAR%20PRADESH?page=1
  * PM-KISAN Operational Guidelines — https://pmkisan.gov.in/Documents/PM-KMY%20-%20Operational%20Guidelines.pdf
  * PM-KISAN Samman Nidhi Scheme (Revised Guidelines) — https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf
  * Pradhan Mantri Fasal Bima Yojana (PMFBY) Guidelines — https://pmfby.gov.in/guidelines
  * Kisan Credit Card Guidelines (RBI) — https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/04MCKCC03072017.pdf
  * Modified Interest Subvention Scheme (MISS) — sourced via web search (no single stable official PDF URL; document saved locally, source to be re-verified before final submission)
  * NFSNM Guidelines (FY 2025–26) — https://www.nfsm.gov.in/Guidelines/NFSNM%20GUIDELINES%20APPROVED%20FY%202025-2026.pdf
  * NFSNM Circulars — https://www.nfsm.gov.in/circulars.aspx
  * IPM Package of Practices — Paddy Blast Management — https://ppqs.gov.in/sites/default/files/pop_for_management_of_paddy_blast.pdf
  * Wheat Cultivation in India (ICAR-IIWBR Pocket Guide) — https://iiwbr.org.in/wp-content/uploads/2023/08/EB-52-Wheat-Cultivation-in-India-Pocket-Guide.pdf
  * ICAR Indian Farming Magazine (Nov 2025) — https://icar.org.in/sites/default/files/2025-10/Indian%20Farming%20November%202025.pdf
  * Rice-Based Cropping Systems (ICAR) — https://icar.org.in/sites/default/files/inline-files/Rice-based-cropping-systems.pdf
* **Public/private/licensed status:** Publicly available government/institutional data (Union and UP State government portals, ICAR, RBI); usage falls under open government data licensing, though a couple of sources (e.g. MISS) were located via general web search rather than a single stable government URL and should be re-verified for licensing before final submission.
* **Purpose:** Forms the localized knowledge base for the MuRIL-embedded RAG pipeline, grounding LLM responses in real agronomic Q&A, scheme eligibility/operational details, and crop-specific advisories rather than parametric (and potentially hallucinated) knowledge.
* **Why each dataset was selected:** These PDFs were selected to ensure domain alignment; scheme PDFs and advisories are authoritative, up-to-date sources for policy and agronomic guidance, addressing the "outdated scheme" and dosage-hallucination risks flagged in Milestone 1.
* **Alternatives considered:** General-purpose agricultural web-scraped text was considered but rejected due to higher noise and copyright/reliability concerns.

**2.3 RAG / NLP KCC (Kisan Call Centre)**

* **Dataset name(s):** Kisan Call Centre (KCC) Query–Answer Transcripts
* **Source(s) and download links:** 
  * Open Government Data (OGD) Platform, Government of India: https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f
* **Public/private/licensed status:** The dataset is publicly available through the Government of India's Open Government Data (OGD) Platform and is intended for public and research use.
* **Purpose:** The dataset forms the primary knowledge source for the RAG subsystem by providing real farmer queries and expert responses, enabling context-aware and grounded agricultural recommendations.
* **Why each dataset was selected:** The KCC dataset contains authentic farmer questions and expert answers related to crops, diseases, pests, cultivation practices, and agricultural schemes. Since the project targets farmers in Uttar Pradesh, the dataset was filtered accordingly to ensure regional relevance.
* **Alternatives considered:** General agricultural question–answer datasets and web-scraped agricultural content were considered but were not selected due to lower reliability, inconsistent quality, and limited relevance compared to official KCC records.

**2.4 Yield Prediction Dataset (Should-Have)**

* **Dataset name(s):** 
* **Source(s) and download links:** 
* **Public/private/licensed status:** 
* **Purpose:** 
* **Why each dataset was selected:** 
* **Alternatives considered:** 

---

### 3. Dataset Description

**3.1 Vision Dataset**
* Number of images (per class, train/field)
* Target variable(s) — disease classes
* Feature description (image properties, resolution, channels)
* Data format
* Sample records (example images/labels)
* Dataset schema

**3.2 RAG/NLP PDF Corpus**

| Attribute               | Description                                                                                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Number of documents** | 187 PDFs collected from four source folders (Other_docs, Schemes, PPQS_Advisories, and UP_ACF_PDFs). After removing unreadable and near-duplicate documents, 170 PDFs were retained. |
| **Number of features**  | Not applicable in the traditional tabular sense. Each document consists of variable-length text with associated metadata.                                                            |
| **Target variable(s)**  | N/A (retrieval corpus; no prediction labels).                                                                                                                                        |
| **Feature description** | Metadata includes `source`, `filename`, `page_count`, `word_count`, `extraction_method`, `detected_language`, `detected_year`, and `garbage_char_ratio`.                             |
| **Data format**         | Source documents in PDF format with extracted plain-text (`.txt`) versions.                                                                                                          |
| **Dataset schema**      | One record per document stored in `pdf_inventory_clean.csv` with the metadata fields listed above.                                                                                   |

**Sample Record 1 (Schemes):**

> "Interest Subvention is provided on short term crop loans and short term loans for allied activities including animal husbandry, dairy, fisheries, bee keeping etc."

**Sample Record 2 (Schemes):**

> "Interest subvention and prompt repayment incentive benefits on short term crop loans and short term loans for allied activities will be available on an overall limit."

**3.3 RAG / NLP KCC**

| Attribute               | Description                                                                                                                                                    |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Number of records**   | 3,123,029 KCC query–answer records for Uttar Pradesh, spanning 2020–2025 (6 yearly CSVs combined); ~3.34 GB in memory.                                        |
| **Number of features**  | 15 raw columns: `KCCCallID`, `CreatedOn`, `StateName`, `DistrictName`, `BlockName`, `Sector`, `Category`, `Crop`, `Season`, `QueryType`, `QueryText`, `KccAns`, `day`, `month`, `year`. |
| **Target variable(s)**  | N/A (retrieval corpus of farmer query → expert answer pairs, not a labeled prediction task).                                                                 |
| **Feature description** | `QueryText`/`KccAns` are the core Q&A text fields (English queries, mostly Hindi answers); `Crop`, `Category`, `QueryType` provide topical metadata; `DistrictName`/`BlockName` give geographic granularity; `day`/`month`/`year` give temporal granularity. `Season` is present in schema but entirely unpopulated (100% missing). |
| **Data format**          | Tabular CSV (one file per year), combined into a single DataFrame.                                                                                            |
| **Dataset schema**       | One row per farmer call/query, with the 15 columns above plus derived fields (`QueryText_length`, `KccAns_length`, detected language) added during EDA.       |

**Sample Records:**

> *Query (Category: Cereals, Crop: Paddy):* "Dhaan ki fasal me top dressing ke samay kya prayog kare?"
> *Answer:* "महोदय, धान में टॉप ड्रेसिंग के समय यूरिया 35 kg और जिंक सल्फेट 10 kg प्रति एकर की दर से नमी की अवस्था में प्रयोग करे।"

> *Query (Category: Others, QueryType: Government Schemes):* "Information about application status of PM Kisan Samman Nidhi scheme?"
> *Answer:* "श्रीमान जी प्रधानमंत्री किसान सम्मान निधि योजना का आवेदन राज्य/जिला के स्वीकृति के लिए लंबित है..."

**3.4 Yield Dataset**
* Number of records
* Number of features
* Target variable(s) — yield
* Feature description
* Data format
* Sample records
* Dataset schema

---

## 4. Data Governance

The datasets used in this project were obtained from publicly available and trusted sources. Appropriate checks were performed to ensure data quality, licensing compliance, and reproducibility before using the datasets for model development.

| Aspect                           | Description                                                                                                                                                                                                                                                                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Sources & Licensing**     | Vision datasets were obtained from Kaggle, RAG documents from official government portals (ICAR, PPQS, PM-KISAN, RBI, etc.), KCC data from the Government of India OGD API, and yield data from official agricultural statistics portals. All datasets are publicly available for research and educational use. |
| **Privacy**                      | The vision datasets contain only crop images. Government documents contain no personal information. KCC records were reviewed and only agricultural query-response information relevant to the project was retained.                                                                                            |
| **Data Quality**                 | EDA was conducted to identify missing values, duplicate records, corrupted images, unreadable PDFs, inconsistent metadata, and other quality issues. Identified issues were corrected or removed during preprocessing.                                                                                          |
| **Ethics & Bias**                | The datasets may contain regional and class imbalance, as well as differences between laboratory and real-world images. These limitations are acknowledged and will be considered during model evaluation.                                                                                                      |
| **Reproducibility & Compliance** | All dataset sources, download procedures, preprocessing scripts, and EDA notebooks have been documented. Raw datasets were preserved separately from processed datasets to ensure reproducibility and compliance with dataset licensing.                                                                        |

### Dataset-specific Notes

* **Vision:** Corrupted images and duplicate samples were identified during EDA, while image quality, resolution, and class distributions were analyzed before preprocessing.
* **RAG:** Government PDFs were checked for extraction quality and duplicates before being incorporated into the retrieval corpus.
* **KCC:** Query-response records were filtered and validated to remove incomplete or inconsistent entries while preserving the original agricultural content.
* **Yield:** Missing values, inconsistent records, and formatting issues were identified and addressed before further analysis.

---

### 5. Exploratory Data Analysis (EDA)

**5.1 Vision Dataset EDA**
* Summary statistics
* Class distribution (per disease)
* Missing/corrupt image analysis
* Duplicate analysis
* Outlier/quality analysis (blur, lighting)
* Visualizations

**5.2 RAG/NLP PDF Corpus EDA**
* **Summary statistics:** 187 PDFs collected across 4 source folders (Other_docs, Schemes, PPQS_Advisories, UP_ACF_PDFs); per-folder doc count, total pages, and average word count documented in `pdf_inventory_clean.csv`.

| Source           | Num Docs | Total Pages | Avg Pages | Avg Words | OCR Docs | Failed Docs |
|------------------|---------:|------------:|----------:|----------:|---------:|------------:|
| other_docs       |       12 |         189 |      15.8 |    5566.5 |        0 |           5 |
| ppqs_advisories  |       90 |         444 |       4.9 |    1302.4 |        0 |           9 |
| schemes          |       11 |         359 |      32.6 |    8956.5 |        0 |           0 |
| up_acp           |       74 |        1805 |      24.4 |    4579.7 |        0 |           0 |

* **Language:** Language detection (via langdetect, with a Devanagari-ratio fallback) found the corpus to be overwhelmingly English (168/170 documents); 2 documents were misclassified as Welsh/Catalan due to short/noisy text samples — these are detector artifacts, not genuine non-English content, and were manually verified as English.
* **Document length distribution:** Page count and word count distributions computed and plotted.

![Page word count histo](<./assets/milestone-2-assets/page_word_count_histo.png>)



* **Missing value analysis:** Documents with failed/near-empty text extraction were retried at higher OCR DPI (300); those still unreadable were excluded (`excluded_unreadable_docs.csv`) — 14 of 187 PDFs.
* **Duplicate analysis:** Exact duplicates flagged via file hash; 33 near-duplicate pairs (similarity > 0.90) identified via text similarity and resolved by keeping the higher-word-count copy (`excluded_near_duplicate_docs.csv`), giving a final clean corpus of 170 documents.
* **Word frequency & domain-relevant terms:** Top frequent words (stopwords removed, English + Hindi) computed across the clean corpus to sanity-check extraction quality and vocabulary coverage. A domain-keyword coverage check (rice, wheat, scheme, subsidy, kisan, etc.) confirmed the corpus contains the terms the RAG system needs to retrieve, per the Rice/Wheat/scheme scope defined in Milestone 1.


![Word frequency](<./assets/milestone-2-assets/word_frequency.png>)


* **Other Visualizations:**

![Word count](<./assets/milestone-2-assets/word_count.png>)




![Docs per source](<./assets/milestone-2-assets/docs_per_source.png>)



**5.3 RAG / NLP KCC**
* **Summary statistics:** 3,123,029 records, 15 columns; per-year breakdown — 2020: 565,719 | 2021: 495,222 | 2022: 620,775 | 2023: 585,633 | 2024: 536,048 | 2025: 319,632.
* **Crop/category distribution:** 318 unique crops; top crops are "Others" (34.7%), Wheat (16.4%), Paddy/Rice (15.5%), Sugarcane (4.0%), Potato (3.4%). Rice + Wheat combined account for **31.95%** of all queries (997,806 records), directly validating the project's rice/wheat scope. Query categories: Cereals (32.0%) and Others (35.0%) dominate; query types are led by Weather (33.5%) and Government Schemes (25.6%).

`[PLACEHOLDER: insert crop_distribution.png / category_distribution.png]`

* **Temporal distribution:** Query volume is fairly stable across years (2020–2024 range 495K–621K), with a sharp drop in 2025 (319,632) — likely a partial-year data cutoff rather than a real decline. Q1 (Jan–Mar) is the busiest quarter (33.2% of queries).

`[PLACEHOLDER: insert year_month_trend.png]`

* **Language distribution:** On a 100,000-record sample — queries are **99.98% English** (farmer questions typed in English/Romanized script), while answers are **98.80% Hindi** (expert responses given in Devanagari). This English-query/Hindi-answer split is an important design signal for the RAG pipeline's embedding and retrieval strategy.
* **Text length distribution:** Query text averages 54 characters (median 54, 95th pct 85); answers average 209 characters (median 203, 95th pct 392). Combined Q&A length averages 264 characters (95th pct 432) — **98.9% of records fit within a 512-character chunk**, directly supporting a 512-character MuRIL chunking strategy with ~50-character overlap.
* **Missing value analysis:** `Season` is 100% missing (unusable, to be dropped). Other fields have low missingness: `Crop` (0.20%), `QueryType` (0.16%), `Category` (0.12%), `KccAns` (0.02%), `QueryText` (0.0004%) — negligible and safe to drop/impute row-wise.
* **Duplicate analysis:** No exact duplicate rows (0.00%), but **68.72% of `QueryText` values are duplicates** (e.g. "Farmer asked query on Weather" appears 781,352 times — largely templated weather queries), 13.15% duplicate `KCCCallID`s, and 26.15% duplicate full Q&A pairs. A sample-based near-duplicate check (1,000 records) found 57 similar query pairs — this high redundancy needs deduplication before MuRIL embedding to avoid over-representing templated/boilerplate queries in the retrieval index.
* **Visualizations:**

`[PLACEHOLDER: insert query_type_distribution.png]`
`[PLACEHOLDER: insert text_length_boxplot.png]`

**5.4 Yield Dataset EDA**
* Summary statistics
* Feature distributions
* Missing value analysis
* Outlier analysis
* Correlation analysis
* Visualizations

---

### 6. Data Preprocessing

**6.1 Vision**
* Cleaning steps
* Missing/corrupt file handling
* Label correction
* Image resizing/normalization
* Augmentation prep

**6.2 RAG/NLP PDFs**
* **Text cleaning:** Text extracted per PDF via `pdfplumber` (native) with `pytesseract` OCR fallback (English + Hindi) for scanned documents; garbage-character ratio computed to flag low-quality extractions for review.
* **De-duplication:** Exact duplicates removed via content hash; 33 near-duplicate document pairs resolved by retaining the more complete (higher word-count / native-extraction) copy.
* **Standardization:** Consistent per-document metadata schema (`source`, `extraction_method`, `detected_language`, `detected_year`, etc.) recorded for every retained document.
* **Tokenization/Chunking:** Sentence-aware chunking applied (512-token target, ~50-token overlap), with a hard-split fallback for any single sentence exceeding 512 tokens (fixes rare cases like long unstructured clauses/tables). On the clean 170-document corpus, this produced **1,451 chunks** (avg. 8.5 chunks/doc), with a median chunk size of 481 tokens (mean 433, std 108) and a max of 561 tokens — no chunk exceeds the target by more than the natural overlap margin. Chunk-count distribution is heavily concentrated in the 400–550 token range, confirming the sentence-aware + hard-split approach keeps chunks consistently sized. Output saved to `pdf_chunks.csv` / `pdf_chunks.jsonl`.



![Chunking Plots](<./assets/milestone-2-assets/pdf_chunk.png>)
  



* **Encoding (metadata fields):** Each chunk carries `chunk_id`, `source`, `filename`, `chunk_index`, `token_count`, `detected_language`, and `detected_year` — enabling retrieval-time filtering (e.g. restrict to `source=schemes` or `detected_year >= 2024`) alongside semantic search.

**6.3 RAG / NLP KCC**
* **Missing value treatment:** Drop the `Season` column (100% missing); drop/impute rows with missing `Crop`, `QueryType`, `Category`, `QueryText`, `KccAns` (all under 0.25% missingness each).
* **De-duplication:** 68.72% of `QueryText` values and 26.15% of full Q&A pairs are duplicates (largely templated Weather/PM-KISAN queries) — deduplicate on the Q&A pair before embedding, to avoid over-representing boilerplate content in the retrieval index.
* **Standardization:** Normalize crop/category/query-type string casing and whitespace; consolidate near-duplicate `QueryType` labels (e.g. tab-prefixed variants like `"\tPlant Protection\t"`).
* **Tokenization/chunking:** 512-character chunk size recommended (matches MuRIL's limit), with ~50-character overlap — covers 98.9% of Query+Answer pairs without truncation, based on the text-length distribution analysis.
* **Language handling:** Queries are treated as English/Romanized input; answers are predominantly Hindi (Devanagari) — no translation step planned, but this asymmetry should be reflected in embedding/retrieval design (MuRIL supports both).

**6.4 Yield**
* Missing value treatment
* Normalization/scaling
* Encoding categorical variables
* Feature engineering
* Feature selection

---

### 7. Dataset Integration (if multiple datasets)

* Datasets combined
* Integration methodology
* Schema alignment
* Handling conflicting attributes
* Deduplication after merging

---

### 8. Data Augmentation (if applicable)

* Augmentation techniques used (vision: rotation, brightness, background synthesis, etc.)
* Rationale
* Examples
* Number of augmented samples generated

---

### 9. Dataset Splitting

* Train/Validation/Test split ratio (per dataset)
* Number of samples in each split
* Stratified sampling (if applicable)
* Justification for split strategy
* Leakage prevention measures

---

### 10. Final Prepared Dataset

* Final dataset size (per dataset)
* Number of features
* Final class distribution
* Summary of preprocessing completed
* Readiness for model training

---

### 11. Challenges Encountered

* **Data quality problems (RAG/NLP corpus):** A subset of scanned government PDFs failed native text extraction; required OCR, and some remained unreadable even after a higher-DPI retry (14 of them), and were excluded from the corpus.
* **Duplication (RAG/NLP corpus):** 33 near-duplicate document pairs found (e.g. same circular re-uploaded across portals) — resolved via automated similarity-based comparison, though a few pairs may warrant manual review.
* **Licensing constraints (RAG/NLP corpus):** `[PLACEHOLDER: confirm exact usage terms per government portal before final submission, particularly for the MISS document sourced via general web search rather than an official stable URL]`
* **High redundancy (KCC dataset):** 68.72% of query texts are duplicates, dominated by templated Weather and PM-KISAN status queries — requires deduplication before embedding to avoid retrieval-index skew toward boilerplate content.
* `[PLACEHOLDER — vision/yield challenges owned by teammates]`

---

### 12. Deliverables Produced

* **RAG/NLP corpus (this section's contribution):** `pdf_inventory_clean.csv` (cleaned document inventory, 170 documents), extracted `.txt` files per PDF, `excluded_unreadable_docs.csv`, `excluded_near_duplicate_docs.csv`, `PDF_Corpus_EDA.ipynb` (EDA notebook), `corpus_overview.png`, `word_frequency.png`, `PDF_Chunking.ipynb` (chunking notebook), `pdf_chunks.csv` / `pdf_chunks.jsonl` (1,451 chunked records ready for Milestone 3 embedding).
* **KCC dataset (this section's contribution):** `kcc_combined_2020_2025.csv` (combined 6-year raw dataset, 3.12M records), `03_kcc_rag_eda.ipynb` (EDA notebook covering crop/category/temporal/language/text-length/missing-value/duplicate analysis).
* `[PLACEHOLDER — vision dataset deliverables owned by teammate]`
* `[PLACEHOLDER — yield dataset deliverables owned by teammate]`
* `[PLACEHOLDER — Train/Validation/Test splits, once finalized in Section 9]`

---

### 13. Summary and Next Steps

* Summary of work completed
* Key observations from the data
* Confirmation that the dataset is ready for model training
* Planned activities for Milestone 3

---

## Team Review & Sign-Off

| # | Team Member | Role | Reviewed & Approved | Date | Signature |
|:-:|-------------|------|:-------------------:|:----:|-----------|
| 1 | Mahesh | | ☐ | | |
| 2 | Harliv | | ☐ | | |
| 3 | Lokesh | | ☐ | | |
| 4 | Aneeqa | | ☐ | | |
| 5 | Tanmay | | ☐ | | |

**Document version:** Milestone 2 — Skeleton · **Prepared:** July 2026