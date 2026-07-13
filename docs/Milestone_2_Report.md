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
      - [Primary Multi-Crop 16-Attribute Schema (`production_unified.csv` \& `production_unified_imputed.csv`)](#primary-multi-crop-16-attribute-schema-production_unifiedcsv--production_unified_imputedcsv)
  - [4. Data Governance](#4-data-governance)
    - [Dataset-specific Notes](#dataset-specific-notes)
    - [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
      - [5.1.1 Rice Leaf Disease Dataset — Set 1 (`vbookshelf`, 5,932 images)](#511-rice-leaf-disease-dataset--set-1-vbookshelf-5932-images)
      - [5.1.2 Rice Leaf Disease Dataset — Set 2 (`nirmalsankalana`, 120 images)](#512-rice-leaf-disease-dataset--set-2-nirmalsankalana-120-images)
      - [5.1.3 Wheat Plant Diseases Dataset (`kushagra3204`, 14,154 images)](#513-wheat-plant-diseases-dataset-kushagra3204-14154-images)
      - [5.4.1 Primary Multi-Crop Dataset Profile (`440,962` Records)](#541-primary-multi-crop-dataset-profile-440962-records)
      - [5.4.2 Complementary UP Rice \& Wheat Subset Findings (`3,886` Records)](#542-complementary-up-rice--wheat-subset-findings-3886-records)
    - [6. Data Preprocessing](#6-data-preprocessing)
      - [6.1.1 Label Canonicalization \& Normalization](#611-label-canonicalization--normalization)
      - [6.1.2 Deduplication \& Burst-Capture Thinning](#612-deduplication--burst-capture-thinning)
      - [6.1.3 Aspect-Preserving Letterbox Resizing \& Color Normalization](#613-aspect-preserving-letterbox-resizing--color-normalization)
      - [6.1.4 Standardized Manifest Schema](#614-standardized-manifest-schema)
      - [6.4.1 Primary Multi-Crop Preprocessing Pipeline (`440,962` Records)](#641-primary-multi-crop-preprocessing-pipeline-440962-records)
      - [6.4.2 Complementary UP Rice \& Wheat Subset Preprocessing (`3,996` Records)](#642-complementary-up-rice--wheat-subset-preprocessing-3996-records)
    - [7. Dataset Integration (if multiple datasets)](#7-dataset-integration-if-multiple-datasets)
    - [8. Data Augmentation \& Training Pipeline Design](#8-data-augmentation--training-pipeline-design)
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
* **Drive link** - https://drive.google.com/drive/folders/1zc6mcshvO-_YiL7yjZcbsA8YOBSJslG-
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

**2.4 Yield Prediction Dataset**

* **Dataset name(s):**
  * **Primary Work:** Unified Pan-India Multi-Crop Agricultural Production & Yield Dataset (1997–2024) across 35 States/UTs and 124 crops (`production_unified.csv`).
  * **Complementary UP Subset:** Uttar Pradesh District-Level Historical Rice & Wheat APY with Environmental/Agronomic Covariates (1997–2023).
* **Source(s) and download links:**
  * **Primary Multi-Crop Data Sources:** Official DES/UPAg crop production records unified across `Crop Recommendation dataset.csv`, `crop_yield.csv` (1997–2020 state-level), `crop-wise-area-production-yield.csv` (1997–2015 district-level), and `DES-District-Data-For-2024-25.csv` (https://upag.gov.in/ / https://data.desagri.gov.in/).
  * **UP Climate & Agronomic Covariates (Subset):** India Meteorological Department (IMD) Pune High Spatial Resolution Gridded Climate Data (https://imdpune.gov.in/) and ICRISAT District Level Database (http://data.icrisat.org/dld/).
* **Public/private/licensed status:** All primary datasets are published under the **Open Government Data (OGD) License India**, allowing free academic research and machine learning applications.
* **Purpose:** The primary multi-crop unified dataset (`440,962` records) provides a comprehensive historical production and yield base covering 124 crops across all seasons and regions. The complementary UP-specific subset (`3,996` records) focuses specifically on Uttar Pradesh Rice and Wheat with weather and fertilizer predictors for targeted state-level crop advisory.
* **Why each dataset was selected:**
  * **Unified Multi-Crop Production Dataset:** Integrates granular district and state agricultural records to support pan-Indian crop comparison, seasonal yield analysis, and robust multi-crop imputation.
  * **UP Rice/Wheat Subset:** Provides localized daily meteorological shocks (monsoon floods, terminal heatwaves) and NPK input data required for UP-focused yield regression modeling.

---

### 3. Dataset Description

**3.1 Vision Dataset**

The vision pipeline draws on **three complementary datasets** covering rice and wheat disease recognition.

**Dataset A — Rice Leaf Disease Dataset (Set 1)** (`vbookshelf/rice-leaf-diseases`, Kaggle)

| Attribute | Description |
|---|---|
| **Total images** | 5,932 `.jpg` files |
| **Number of classes** | 4 disease classes |
| **Target variable** | Folder-level disease label |
| **Classes** | Bacterial Blight (1,584), Blast (1,440), Brown Spot (1,600), Tungro (1,308) |
| **Pre-existing splits** | None — single flat folder per class (splitting is the project's responsibility) |
| **Dominant image size** | 300 × 300 px (78.0% of images; 100% of Bacterial Blight, Blast, Brownspot) |
| **Color mode** | RGB (5,776) and RGBA (156 files mislabelled as `.jpg` but encoded as PNG) |
| **File format** | `.jpg` (extension); 156 files are actually PNG with alpha channel |
| **Feature description** | Raw pixel arrays; no per-image tabular features. Labels derive from parent folder name. |
| **Dataset schema** | `filepath`, `filename`, `label`, `split`, `ext`, `width`, `height`, `mode`, `aspect`, `megapixels`, `md5`, `dup_cluster`, `is_exact_rep` (added during EDA) |

**Dataset B — Rice Leaf Disease Dataset (Set 2)** (`vbookshelf/rice-leaf-diseases` small variant / `nirmalsankalana/rice-leaf-disease-image`)

| Attribute | Description |
|---|---|
| **Total images** | 120 `.jpg` files |
| **Number of classes** | 3 disease classes |
| **Target variable** | Folder-level disease label |
| **Classes** | Bacterial Leaf Blight (40), Brown Spot (40), Leaf Smut (40) |
| **Pre-existing splits** | None — all images in a single partition |
| **Dominant image size** | 3,081 × 897 px (70.8% of images) — panoramic leaf-strip format, aspect ratio ≈ 3.44:1 |
| **Color mode** | RGB only (no RGBA, no corruption) |
| **File format** | `.jpg` (genuine JPEG throughout) |
| **Dataset schema** | Same schema as Set 1 (added during EDA) |

**Dataset C — Wheat Plant Diseases Dataset** (`kushagra3204/wheat-plant-diseases`, Kaggle)

| Attribute | Description |
|---|---|
| **Total images** | 14,154 images across `.jpg`, `.png`, `.webp`, `.gif`, `.mpo` formats |
| **Number of classes** | 15 canonical disease/pest classes |
| **Target variable** | Folder-level disease label (after canonicalization) |
| **Classes** | aphid (903), black\_rust (576), blast (647), brown\_rust (1,271), common\_root\_rot (614), fusarium\_head\_blight (611), healthy (1,000), leaf\_blight (842), mildew (1,081), mite (800), septoria (1,144), smut (1,310), stem\_fly (234), tan\_spot (770), yellow\_rust (1,301) — train counts |
| **Pre-existing splits** | train (13,104 ≈ 92.6%), val (300 ≈ 2.1%), test (750 ≈ 5.3%) — 20 val / 50 test images per class |
| **Image size range** | Highly variable: median 276 × 256 px, mean 716 × 674 px, max 6,016 × 6,600 px |
| **Color modes** | RGB (12,492), RGBA (1,613), P (47), CMYK (2) |
| **File formats** | JPEG (9,076), PNG (5,027), WebP (37), GIF (10), MPO (4) |
| **Dataset schema** | `filepath`, `filename`, `label`, `label_raw`, `split`, `ext`, `width`, `height`, `mode`, `format`, `size_kb`, `aspect`, `ahash` (added during EDA) |

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

| Attribute               | Description                                                                                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Number of records**   | **Primary Work (`production_unified.csv`):** `440,962` records spanning 1997–2024 across 35 States/UTs and 124 crops.<br>**Complementary UP Subset (`up_district_yield_apy_1997_2023.csv`):** `3,886` records spanning 1997–2023 across 74 unique UP districts. |
| **Number of features**  | **Primary Work:** 16 columns across location, crop, season, APY metrics, and auxiliary agronomic metadata.<br>**Complementary UP Subset:** 15 columns adding daily IMD weather stress and ICRISAT NPK/irrigation features. |
| **Target variable(s)**  | `yield` / `Yield_Kg_Ha` (Continuous regression target: crop yield).                                                                                                      |
| **Feature description** | Combines administrative keys (`state`, `district`, `year`, `season`, `crop`), agricultural production metrics (`area`, `production`, `yield`), and auxiliary/environmental features (`annual_rainfall`, `fertilizer`, `pesticide`). |
| **Data format**         | Structured Tabular CSV (`production_unified.csv`, `production_unified_imputed.csv`).                                                                                   |
| **Dataset schema**      | Detailed schema described in the table below.                                                                                                                           |

#### Primary Multi-Crop 16-Attribute Schema (`production_unified.csv` & `production_unified_imputed.csv`)

| Column Name | Data Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `state` | String | 35 States/UTs | Administrative state identifier. |
| `district` | String | Granular Districts | Standardized administrative district name. |
| `year` | Integer | 1997 to 2024 | Agricultural calendar year. |
| `season` | Categorical | 6 Seasons | Cropping season (`kharif`, `rabi`, `whole year`, `autumn`, `summer`, `winter`). |
| `crop` | Categorical | 124 Unique Crops | Agricultural crop commodity (e.g., `sugarcane`, `rice`, `wheat`, `potato`). |
| `area` | Float | Hectares (ha) | Gross cropped area sown. |
| `production` | Float | Tonnes | Total harvested output (coconut converted from pieces to tonnes). |
| `yield` | Float | Tonnes/ha or kg/ha (**Target**) | Calculated productivity target. |
| `annual_rainfall` | Float | Millimeters (mm) | Annual cumulative precipitation where available. |
| `fertilizer` / `pesticide` | Float | Tonnes / kg | Total chemical input usage where available. |

*(Note: The complementary UP Rice & Wheat subset additionally tracks daily IMD weather shocks — `Precip_Seasonal_mm`, `Rain_Days_Extreme`, `Heatwave_Days` — and ICRISAT NPK fertilizer splits for state-level modeling).*

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

Full EDA notebooks are at:
- [`rice-leaf-disease-dataset-EDA.ipynb`](../data/Vision_dataset/rice-leaf-disease-dataset-EDA.ipynb) (Rice Set 1)
- [`rice-leaf-disease-dataset-set-2-eda.ipynb`](../data/Vision_dataset/rice-leaf-disease-dataset-set-2-eda.ipynb) (Rice Set 2)
- [`wheat-dataset-EDA.ipynb`](../data/Vision_dataset/wheat-dataset-EDA.ipynb) (Wheat)

---

#### 5.1.1 Rice Leaf Disease Dataset — Set 1 (`vbookshelf`, 5,932 images)

**Summary statistics & class distribution**

| Class | Count | % of total | Unique (post exact-dedup) | % Redundant |
|---|---|---|---|---|
| Brownspot | 1,600 | 27.0% | 1,200 | 25.0% |
| Bacterial Blight | 1,584 | 26.7% | 1,326 | 16.3% |
| Blast | 1,440 | 24.3% | 960 | **33.3%** |
| Tungro | 1,308 | 22.0% | 1,308 | 0.0% |
| **Total** | **5,932** | | **4,794** | |

The raw imbalance ratio is 1.22:1 (Brownspot vs Tungro), which appears benign. However, after exact deduplication the true ratio rises to **1.38:1** (Bacterial Blight 1,326 vs Blast 960), revealing that the apparent balance was partly manufactured by Blast's 33.3% duplication rate.

**Missing/corrupt image analysis**

- **0 unreadable/corrupt images** — all 5,932 files opened successfully.
- **156 RGBA files** mislabelled as `.jpg` but encoded as PNG (108 from Bacterial Blight, 48 from Brownspot). These contain a 4th alpha channel and require a `.convert("RGB")` guard in the data loader.

**Image dimension & format findings**

| Metric | Value |
|---|---|
| Most common size | 300 × 300 px (78.0% of all images) |
| Bacterial Blight, Blast, Brownspot | 100% exactly 300 × 300 px |
| **Tungro** | **0% at 300 × 300 px** — median 331 × 331 px, aspect ratio 1.33 |
| Color modes | RGB: 5,776 · RGBA: 156 |

> ⚠️ **Background-bias risk (Tungro):** Tungro images were photographed as zoomed-out whole plants against bare soil — not leaf close-ups like the other three classes. This means a CNN could learn to classify Tungro by its soil background rather than disease symptoms, creating a spurious-correlation shortcut that collapses in real-field deployment.

**Duplicate analysis**

| Metric | Value |
|---|---|
| Exact (byte-identical) duplicate files | 2,234 (forming 1,096 groups) |
| Exact-dup groups spanning >1 class | **0** (labels are clean) |
| Near-duplicate images (aHash, 8×8) | 4,914 (82.8%) |
| Near-dup groups: within-class | 1,724 |
| Near-dup groups: cross-class | **0** |
| Near-dup images (pHash-DCT, stricter) | 4,700 (79.2%) |
| Unique pHash clusters (for group-aware splitting) | 2,919 |

The 0 cross-class duplicate groups confirms label integrity. However the high within-class redundancy (Blast: 33.3% literal copies) means a naive random 80/20 split will leak near-identical images across train/val/test, inflating evaluation metrics. A **group-aware split using `dup_cluster`** as the group key is mandatory.

---

#### 5.1.2 Rice Leaf Disease Dataset — Set 2 (`nirmalsankalana`, 120 images)

**Summary statistics & class distribution**

| Class | Count | % of total | Unique (post exact-dedup) |
|---|---|---|---|
| Bacterial Leaf Blight | 40 | 33.3% | 40 |
| Brown Spot | 40 | 33.3% | 40 |
| Leaf Smut | 40 | 33.3% | 40 |
| **Total** | **120** | | **120** |

Perfect 1.00:1 balance with **zero exact duplicates** — the 40-per-class count is 40 genuinely distinct images per class.

**Image dimension & format findings**

| Metric | Value |
|---|---|
| Dominant size | 3,081 × 897 px (70.8% of images) |
| Aspect ratio | ~3.44:1 (panoramic leaf-strip format) |
| Color mode | RGB only — 0 RGBA, 0 corrupt |
| File format | Genuine JPEG throughout — 0 mislabelled PNGs |
| Median dimensions per class | Identical across all 3 classes — **no dimension shortcut** |

> ⚠️ **Preprocessing landmine (aspect ratio):** Standard square resize (224×224) would compress each image 3.4× horizontally, badly distorting lesion shapes. Aspect-preserving handling — letterboxing/padding, tiling the strip, or wide-input resolution training — is mandatory for this dataset.

**Duplicate analysis**

| Metric | Value |
|---|---|
| Exact duplicates | **0** |
| Near-dup (aHash, 8×8) | 14 images in 5 groups (11.7%), including 4 cross-class flags |
| Near-dup (pHash-DCT, stricter) | **0 imgs (0.0%)** — the cross-class aHash flags were coarse-hash artifacts |
| pHash cross-class groups | **0** — no genuine label conflicts |

The pHash result clears the cross-class aHash alarm: this dataset is genuinely clean with no near-duplicate pairs at a stricter hash resolution.

---

#### 5.1.3 Wheat Plant Diseases Dataset (`kushagra3204`, 14,154 images)

**Label structure finding (critical)**

The raw folder scan produced 45 apparent classes (e.g. `Aphid`, `aphid_test`, `aphid_valid`, `Black Rust`, `black_rust_test`, …). After applying a canonicalization function (lowercasing, stripping `_test`/`_valid` suffixes), these collapsed to **15 canonical disease classes** with every class represented in all three splits. This mislabelling would have caused complete training failure without correction.

**Summary statistics & class distribution**

| Class (canonical) | Train | Val | Test | Total |
|---|---|---|---|---|
| aphid | 903 | 20 | 50 | 973 |
| black\_rust | 576 | 20 | 50 | 646 |
| blast | 647 | 20 | 50 | 717 |
| brown\_rust | 1,271 | 20 | 50 | 1,341 |
| common\_root\_rot | 614 | 20 | 50 | 684 |
| fusarium\_head\_blight | 611 | 20 | 50 | 681 |
| healthy | 1,000 | 20 | 50 | 1,070 |
| leaf\_blight | 842 | 20 | 50 | 912 |
| mildew | 1,081 | 20 | 50 | 1,151 |
| mite | 800 | 20 | 50 | 870 |
| septoria | 1,144 | 20 | 50 | 1,214 |
| smut | 1,310 | 20 | 50 | 1,380 |
| stem\_fly | 234 | 20 | 50 | 304 |
| tan\_spot | 770 | 20 | 50 | 840 |
| yellow\_rust | 1,301 | 20 | 50 | 1,371 |
| **Total** | **13,104** | **300** | **750** | **14,154** |

> ⚠️ **Class imbalance (train split):** Train imbalance ratio = **5.6:1** (smut 1,310 vs stem_fly 234). Moderate-to-significant; plain accuracy will be misleading. Per-class recall, macro-F1, and confusion matrix are mandatory. Class weighting, oversampling, or targeted augmentation will be needed for the minority classes (stem_fly, black_rust, blast).

> ⚠️ **Tiny val/test sets:** Only 20 val and 50 test images per class. These are too small for reliable per-class metric estimates — evaluation variance will be high. Consider increasing test set size or using cross-validation.

**Visual content note**

This dataset mixes **three fundamentally different visual target types**: insect pests (aphid, mite, stem_fly — require detecting insect shapes), foliar diseases (rusts, septoria, blast, mildew, leaf_blight, tan_spot — require detecting lesion texture/color), and spike/head diseases (fusarium_head_blight, smut — require spike-level features). The rust classes (black, brown, yellow rust) will be the model's hardest confusions — pustules differing mainly in color and arrangement.

**Image dimension & format findings**

| Metric | Value |
|---|---|
| Corrupt/unreadable images | **0** |
| Median size | 276 × 256 px |
| Mean size | 716 × 674 px |
| Size range | 44×31 px → 6,016×6,600 px |
| Aspect ratio range | 0.09 → 18.23 |
| Color modes | RGB (12,492), RGBA (1,613), P (47), CMYK (2) |
| File formats | JPEG (9,076), PNG (5,027), WebP (37), GIF (10), MPO (4) |
| Top resolution | 256×256 px (3,297 images) |

The extreme size variance (median ~0.07 MP vs max ~40 MP) and heterogeneous aspect ratios mandate a consistent resize to a fixed input resolution for training.

**Duplicate & leakage analysis**

| Metric | Value |
|---|---|
| Potential duplicate groups (aHash) | 2,177 groups / 6,279 images (44% of dataset) |
| **Hashes appearing in >1 split** | **646 — train/val/test leakage detected** |

> 🚨 **Critical finding — train/test leakage:** 646 perceptual hashes appear in more than one split, meaning near-identical images are present in both training and evaluation sets. This inflates reported test accuracy and must be resolved before model training.

**Color & brightness analysis (per class)**

Mean brightness ranges from **85.6** (mildew — notably darker) to **138.1** (blast — brightest). The brightness boxplots show wide within-class variance and heavy outliers in most classes, consistent with the mixed image sources (web-scraped, lab, and field photos). Mildew's systematically lower brightness is a potential confound. The mean RGB profiles show similar patterns across classes with no extreme channel dominance.

**Sharpness analysis (Laplacian variance)**

Laplacian variance (sharpness) varies 3–4 orders of magnitude within most classes (log-scale distribution), with blurriest samples in brown_rust, leaf_blight, and black_rust (Laplacian variance < 5). A sharpness threshold filter before training may improve label reliability for these classes.

**5.2 RAG/NLP PDF Corpus EDA (Harliv's Work)**

Comprehensive PDF Corpus EDA & Chunking Technical Report:
- **RAG PDF Report:** [`docs/Milestone_2_work/rag_pdf_report.md`](../docs/Milestone_2_work/rag_pdf_report.md)

* **Summary statistics:** 187 PDFs collected across 4 authoritative government agricultural folders (`Other_docs`, `Schemes`, `PPQS_Advisories`, `UP_ACP`); per-folder document count, total pages, and average word count documented in `pdf_inventory_clean.csv`.

| Source | Num Docs | Total Pages | Avg Pages | Avg Words | OCR Docs | Failed Docs |
|---|---:|---:|---:|---:|---:|---:|
| `other_docs` | 12 | 189 | 15.8 | 5,566.5 | 0 | 5 |
| `ppqs_advisories` | 90 | 444 | 4.9 | 1,302.4 | 0 | 9 |
| `schemes` | 11 | 359 | 32.6 | 8,956.5 | 0 | 0 |
| `up_acp` | 74 | 1,805 | 24.4 | 4,579.7 | 0 | 0 |

* **Language:** Language detection (via `langdetect` with Devanagari-ratio fallback) confirmed the corpus is overwhelmingly English (168/170 documents); 2 documents flagged as Welsh/Catalan due to short/noisy header text were manually verified as clean English.
* **Document length distribution:** Page count and word count distributions reveal heavy length variation — PPQS advisories average 4.9 pages (concise pest guidelines) while government schemes average 32.6 pages (>8,900 words), with the longest document spanning ~100 pages (>30,000 words). This variance directly justifies semantic chunking over document-level retrieval.

![Page word count histo and docs per source](<./assets/milestone-2-assets/docspersource_pagewordcount.png>)

* **Missing value & failure filtering:** Documents with failed/empty native text extraction were retried at 300 DPI OCR; 14 unreadable or corrupted files were excluded (`excluded_unreadable_docs.csv`).
* **Duplicate analysis:** Exact byte duplicates removed via hash; 33 near-duplicate pairs (>0.90 similarity) resolved by retaining the more complete higher-word-count copy (`excluded_near_duplicate_docs.csv`), yielding a clean corpus of **170 documents**.
* **Domain Vocabulary Analysis:** Frequency analysis of normalized tokens confirmed heavy agricultural domain concentration (`water`, `rice`, `crop`, `irrigation`, `soils`, `sowing`, `seed`, `drainage`, `fodder`, `management`), confirming excellent domain alignment with the advisory system.

![Word frequency](<./assets/milestone-2-assets/word_frequency.png>)

**5.3 KCC Advisory Dataset EDA (Aneeqa's Work)**

Comprehensive KCC Data Exploration Technical Report:
- **KCC EDA Report:** [`docs/Milestone_2_work/KCC Data EDA.md`](../docs/Milestone_2_work/KCC%20Data%20EDA.md)

* **Summary statistics:** Combined **3,123,029 records** spanning 6 annual CSVs (2020–2025, Uttar Pradesh). Yearly breakdown — 2020: 565,719 | 2021: 495,222 | 2022: 620,775 | 2023: 585,633 | 2024: 536,048 | 2025: 319,632 (~2.07 GB on disk / ~3.3 GB memory footprint).
* **Crop & Category Distribution:** **318 unique crops** identified. Top crops: Others (34.74%), Wheat (16.40%), Paddy/Dhan (15.54%). **Rice & Wheat combined account for ~31.95% of all queries (997,806 records)**, directly validating our project's crop scope. Across **40 categories and 83 query types**, Weather (33.47%) and Government Schemes (25.63%) dominate farmer inquiries.
* **Temporal & Seasonal Patterns:** Query volume peaked in 2022 (620,773). Q1 (Jan–Mar) dominates seasonal activity (33.16% of queries), with monthly highs in January and lows in May.

![Query month trend](<./assets/milestone-2-assets/query_month_trend.png>)

* **Language Distribution (Critical RAG Insight):** Queries (`QueryText`) are **~99.98% English/Hinglish** (Romanized script), whereas expert answers (`KccAns`) are **~98.80% Hindi** (Devanagari script). This mandatory cross-lingual bridge dictates the use of a multilingual embedding model (`MuRIL`).
* **Text Length Analysis:** Query text averages 54 characters (95th pct 85 chars); answers average 209 characters (95th pct 392 chars). Combined Q&A length averages 264 characters (95th pct 432 chars). Crucially, **98.9% of records fit within 512 characters**, confirming that 512-character chunking captures complete Q&A semantic units.
* **Missingness & Deduplication Requirements:** `Season` is 100% null (dropped). While exact row duplicates are 0%, **68.72% of `QueryText` values are duplicates** (e.g., `"Farmer asked query on Weather"` appears 781,352 times) and **26.15% of full Q&A pairs are exact duplicates**. Deduplication is mandatory prior to vector embedding to prevent retrieval index skew.
* **Visualizations:**

![query_length_ans_length_boxplot](<./assets/milestone-2-assets/query_length_ans_length_boxplot.png>)


**5.4 Yield Dataset EDA**

Full executable EDA notebooks and documentation are available at:
- **Primary Work (Multi-Crop Pan-India):** [`notebooks/07_Yield_EDA+ preprocessing.ipynb`](../notebooks/07_Yield_EDA+%20preprocessing.ipynb) & [`docs/Milestone_2_work/yield_report.md`](../docs/Milestone_2_work/yield_report.md)
- **Complementary UP Subset:** [`notebooks/05_yield_eda.ipynb`](../notebooks/05_yield_eda.ipynb)

#### 5.4.1 Primary Multi-Crop Dataset Profile (`440,962` Records)
- **Geographic & Temporal Coverage:** Evaluated `440,962` historical district-level and state-level agricultural records spanning 1997–2024 across 35 States/Union Territories and 124 distinct crop commodities.
- **Top 10 Crops by Cumulative Production:**
  1. `sugarcane` · 2. `rice` · 3. `wheat` · 4. `potato` · 5. `cotton(lint)` · 6. `maize` · 7. `coconut` · 8. `jute` · 9. `banana` · 10. `soyabean`.
- **Seasonality Breakdown:**
  - `whole year`: Highest total production volume and highest average yield per hectare across seasonal labels.
  - `kharif`: Second-highest total production with strong record representation across cereal crops.
  - `rabi`: Third-highest total production volume, driven predominantly by wheat and winter pulses.
  - `winter`, `summer`, and `autumn`: Capture smaller but regionally vital seasonal cropping contributions.
- **Numeric Feature Correlations:** Evaluated correlations across `area`, `production`, `yield`, `annual_rainfall`, `fertilizer`, and `pesticide` (visualized in `correlation_matrix.png`). Production shows high collinearity with sown area and fertilizer application across major cash crops.

#### 5.4.2 Complementary UP Rice & Wheat Subset Findings (`3,886` Records)
- **Regional Disparity within UP:** Canal/tubewell-intensive Western UP achieves significantly higher staple yields (Rice 2,893.8 / Wheat 3,551.9 kg/ha with 94.0% irrigation share) compared to water-insecure Bundelkhand (Rice 1,719.7 / Wheat 1,490.8 kg/ha with 50.6% irrigation).
- **IMD Climate Shocks:** `Rain_Days_Extreme` (>64.5 mm/day) exhibits a negative correlation (**-0.223**) with Kharif Rice yield due to monsoon lodging, while `Heatwave_Days` (>38°C in March) correlates negatively (**-0.241**) with Rabi Wheat yield during grain filling.

---

### 6. Data Preprocessing

**6.1 Vision Preprocessing across Source Datasets (Notebooks A, B, C)**

Executable preprocessing notebooks and technical documentation:
- **Wheat (Notebook A):** [`docs/Milestone_2_work/wheat_preprocessing_documentation.md`](../docs/Milestone_2_work/wheat_preprocessing_documentation.md)
- **Rice Set 1 (Notebook B):** [`docs/Milestone_2_work/rice_set1_preprocessing_documentation.md`](../docs/Milestone_2_work/rice_set1_preprocessing_documentation.md)
- **Rice Set 2 (Notebook C):** [`docs/Milestone_2_work/rice_set2_preprocessing_documentation.md`](../docs/Milestone_2_work/rice_set2_preprocessing_documentation.md)

#### 6.1.1 Label Canonicalization & Normalization
- **Wheat Dataset:** Collapsed 45 raw folder labels (e.g., `Aphid`, `aphid_test`, `aphid_valid`) down to **15 canonical disease classes** (`wheat__*`) via programmatic suffix stripping (`_test`, `_valid`, `_val`, `_train`).
- **Rice Set 1 & Set 2:** Mapped raw class folders to shared canonical crop-prefixed labels (`rice__bacterial_blight`, `rice__blast`, `rice__brown_spot`, `rice__tungro`, and `rice__leaf_smut`). Shared classes (`bacterial_blight` and `brown_spot`) were harmonized across Set 1 and Set 2.

#### 6.1.2 Deduplication & Burst-Capture Thinning
- **Rice Set 1 (Burst Thinning):** Evaluated exact MD5 and perceptual hash (`pHash` Hamming distance ≤ 6 bits) clusters via union-find. Identified ~65% of raw images as near-duplicate burst frames of identical leaves. Thinned each cluster down to a single representative frame (highest Laplacian sharpness variance), reducing 5,932 raw images to **2,066 clean unique images** across 4 balanced classes (`bacterial_blight`: 514, `blast`: 477, `brown_spot`: 606, `tungro`: 469).
- **Rice Set 2:** Confirmed pristine (0 duplicates, 0 corrupt files across 120 images, 40 per class).
- **Wheat Dataset:** Deduplicated exact and perceptual near-duplicates, cleaning 14,154 raw images down to **10,673 unique groups**.

#### 6.1.3 Aspect-Preserving Letterbox Resizing & Color Normalization
- **Letterboxing to 256×256 RGB:** All kept images were converted to RGB (flattening 156 RGBA PNGs in Set 1 and 1,662 non-RGB files in Wheat) and **letterboxed to 256×256 px** (padded aspect-preserving resize).
- **Removing Shortcuts & Lesion Distortion:**
  - In **Rice Set 1**, letterboxing eliminated the *Tungro dimension shortcut* (Tungro raw photos averaged ~331² 4:3 vs 300² square for others; post-letterbox all are identical 256²).
  - In **Rice Set 2**, letterboxing prevented lesion geometric distortion across extreme wide panoramas (~3.43:1 aspect ratio, 89% of images).

#### 6.1.4 Standardized Manifest Schema
All three cleaned datasets exported manifests adhering to a uniform 6-column schema: `src_path, filename, label, source_dataset, split, group_id` (`split = unassigned` deferred to centralized integration in Notebook D).

**6.2 RAG/NLP PDF Preprocessing & Chunking (Harliv's Work)**

Detailed RAG PDF Preprocessing & Chunking Documentation:
- **RAG PDF Report:** [`docs/Milestone_2_work/rag_pdf_report.md`](../docs/Milestone_2_work/rag_pdf_report.md)

* **Text extraction & cleaning:** Text extracted per PDF via `pdfplumber` (native) with `pytesseract` OCR fallback (English + Hindi) for scanned documents; garbage-character ratio computed to flag low-quality extractions for exclusion.
* **Deduplication & filtering:** Exact byte duplicates removed via hash; 33 near-duplicate document pairs (>0.90 similarity) resolved by retaining the more complete copy, leaving 170 clean PDFs.
* **Metadata extraction:** Standardized per-document metadata (`source`, `extraction_method`, `detected_language`, `detected_year`, `page_count`, `word_count`) recorded for every document.
* **Tokenization & Semantic Chunking:** Sentence-aware chunking applied (target ~512 tokens, ~50-token overlap), with a hard-split fallback for any single sentence exceeding 512 tokens. On the clean 170-document corpus, this produced **1,451 chunks** (avg. 8.5 chunks/doc), with a median chunk size of 481 tokens (mean 433, std 108) and a max of 561 tokens. Output saved to `pdf_chunks.csv` / `pdf_chunks.jsonl`.

![Chunking Plots](<./assets/milestone-2-assets/pdf_chunk.png>)

* **Metadata Encoding:** Each chunk carries `chunk_id`, `source`, `filename`, `chunk_index`, `token_count`, `detected_language`, and `detected_year` — enabling retrieval-time filtering alongside vector search.

**6.3 KCC Advisory Dataset Preprocessing (Aneeqa's Work)**

Detailed KCC EDA & Pipeline Preparation Documentation:
- **KCC EDA Report:** [`docs/Milestone_2_work/KCC Data EDA.md`](../docs/Milestone_2_work/KCC%20Data%20EDA.md)

* **Missing value treatment:** Dropped `Season` column (100% missing); dropped/imputed rows with missing `Crop`, `QueryType`, `Category`, `QueryText`, `KccAns` (all under 0.25% missingness each).
* **Deduplication:** Given that **68.72% of `QueryText` values and 26.15% of full Q&A pairs are duplicates**, deduplication across Q&A pairs is mandatory prior to vector indexing to prevent index skew toward templated weather/scheme queries.
* **Standardization:** Normalized crop/category/query-type string casing and whitespace; consolidated near-duplicate `QueryType` labels (e.g., tab-prefixed variants like `"\tPlant Protection\t"`).
* **Chunking Strategy:** Recommended 512-character chunking (matches `MuRIL` input constraints) with ~50-character overlap — captures **98.9% of Query+Answer pairs** without truncation.
* **Multilingual Alignment:** Queries are processed as English/Romanized input while answers are predominantly Hindi (Devanagari script), leveraging `MuRIL`'s cross-lingual semantic alignment capabilities.

**6.4 Yield Preprocessing & Feature Engineering**

Full executable preprocessing pipelines and documentation:
- **Primary Work (Multi-Crop Pan-India):** [`notebooks/07_Yield_EDA+ preprocessing.ipynb`](../notebooks/07_Yield_EDA+%20preprocessing.ipynb) & [`docs/Milestone_2_work/yield_report.md`](../docs/Milestone_2_work/yield_report.md)
- **Complementary UP Subset:** [`notebooks/06_yield_preprocessing.ipynb`](../notebooks/06_yield_preprocessing.ipynb)

#### 6.4.1 Primary Multi-Crop Preprocessing Pipeline (`440,962` Records)
1. **Multi-Source Schema Harmonization:** Normalized disparate state and district agricultural sources into a standardized tabular schema: `crop`, `year`, `state`, `district`, `season`, `area`, `production`, `yield` along with auxiliary metadata (`annual_rainfall`, `fertilizer`, `pesticide`).
2. **Standardization & Unit Resolution:** Standardized crop/district text casing and resolved reporting discrepancies, including converting coconut production and yield quantities from raw pieces to metric tonnes.
3. **Hierarchical Deduplication:** Deduplicated overlapping state-level and district-level records by preferentially retaining more granular district-level entries.
4. **Machine Learning Imputation (`MissForest`):**
   - Imputed missing categorical attributes using `SimpleImputer(strategy='most_frequent')`.
   - Encoded categorical variables using `OrdinalEncoder` and applied non-parametric **Random Forest Imputation (`MissForest`)** to impute missing numeric values across complex inter-crop feature relationships.
   - Reversed ordinal encodings to preserve readable text labels and exported the final complete dataset to `production_unified_imputed.csv`.

#### 6.4.2 Complementary UP Rice & Wheat Subset Preprocessing (`3,996` Records)
- **Zero-Production Anomaly Imputation:** Sporadic administrative omissions (`Production_Total == 0.0` despite active `Area_Sown`) were imputed using agro-climatic zone seasonal medians.
- **Spatial Harmonization (Bifurcation Backcasting):** Missing records caused by post-1997 district bifurcations (Amethi, Sambhal, Hapur, Shamli) were backcasted via parent-district proportional area apportionment (~28% share).
- **Engineered Domain Predictors:** Computed 5 domain interaction features (`NPK_Total_Intensity_Kg_Ha`, `NPK_Balance_Ratio`, `Rainfall_Anomaly_Pct`, `Thermal_Stress_Index`, `Irrigation_Security_Score`).

---

### 7. Dataset Integration (if multiple datasets)

Executable integration notebook and documentation:
- **Notebook D (Merge + Split):** [`docs/Milestone_2_work/notebookD_merge_split_documentation.md`](../docs/Milestone_2_work/notebookD_merge_split_documentation.md)

The vision subsystem integrates three cleaned crop-disease datasets into a unified 20-class training pool, with PlantDoc reserved as a held-out field-robustness evaluation set.

**7.1 Combined Datasets & Class Harmonization**

| Source Dataset | Input Cleaned Images | Classes Contributed |
|---|---|---|
| **Wheat (Notebook A)** | 10,673 | 15 classes (`wheat__*`) |
| **Rice Set 1 (Notebook B)** | 2,066 | 4 classes (`rice__bacterial_blight`, `rice__blast`, `rice__brown_spot`, `rice__tungro`) |
| **Rice Set 2 (Notebook C)** | 120 | 3 classes (`rice__bacterial_blight`, `rice__brown_spot`, `rice__leaf_smut`) |
| **Merged Unified Pool** | **12,859** | **20 unique canonical disease classes** |

**7.2 Integration Methodology**
1. **Schema & Label Alignment:** Concatenated the three cleaned dataset manifests sharing an identical 6-column schema (`src_path, filename, label, source_dataset, split, group_id`). Crop prefixes (`wheat__` vs. `rice__`) prevent crop-specific diseases (e.g., wheat blast vs. rice blast) from colliding.
2. **Shared Class Merging:** Shared rice classes (`rice__bacterial_blight` = 554 images [514 S1 + 40 S2]; `rice__brown_spot` = 646 images [606 S1 + 40 S2]) were merged cleanly across Set 1 and Set 2. Rice Set 2 uniquely contributes `rice__leaf_smut` (40 images).
3. **Integrity Reconciliation:** Verified 100% manifest-to-disk reconciliation (12,859 files on disk, 0 missing). A total of 11,530 unique perceptual hash `group_id` clusters exist across the 12,859 records.

---

### 8. Data Augmentation & Training Pipeline Design

Technical training pipeline design documentation:
- **Notebook E (Training Pipeline Design):** [`docs/Milestone_2_work/notebook_training_pipeline_design.md`](../docs/Milestone_2_work/notebook_training_pipeline_design.md)

To preserve data integrity, the materialized dataset (`final/`) stores one deterministic 256×256 letterboxed RGB copy per image. Augmentation, normalization, and imbalance handling run **on-the-fly inside the data loader during training (Milestone 3)**.

**8.1 Deterministic Resolution & ImageNet Normalization (All Splits)**
- **Train Split:** Random 224×224 crop from the 256×256 letterboxed frame.
- **Validation / Test Splits:** Deterministic 224×224 center crop.
- **Standardization:** All tensors normalized using ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`), matching our ImageNet-pretrained backbone (`EfficientNet-B0`).

**8.2 On-The-Fly Data Augmentation Strategy (Train Split Only)**
- **Standard Tier (All Classes):** Random horizontal/vertical flips, random rotation (±15–20°), and color jitter (brightness/contrast/saturation) to simulate field lighting and camera variance.
- **Targeted Rare & Risk Class Augmentation:**
  - `rice__leaf_smut` (40 images / 24 train) and `wheat__stem_fly` (172 images / 138 train) receive aggressive multi-scale transforms to multiply effective sample variety.
  - `rice__tungro` receives aggressive background-focused random cropping and jitter to disrupt the soil-background artifact identified in EDA.

**8.3 Class Imbalance & Tungro Robustness Strategy**
- **Imbalance Handling:** Uses a hybrid recipe combining `WeightedRandomSampler` (oversampling extreme minority classes per batch) with mild class-weighted CrossEntropy loss. Models are evaluated strictly on **macro-F1 score and per-class recall**.
- **Tungro Robustness Diagnostic:** Post-training Grad-CAM inspection on non-soil Tungro field images confirms whether the classifier attends to leaf lesions or background soil, triggering leaf segmentation masking if background shortcut bias persists.

---

### 9. Dataset Splitting

**9.1 Vision Unified Group-Aware Stratified Split (Notebook D)**

Rather than splitting Wheat and Rice separately, Notebook D performs a **single centralized group-aware stratified split across all 20 canonical classes (`12,859` total images)**.

| Parameter | Execution Value |
|---|---|
| **Split Ratio** | 80% Train (`10,275` images) / 10% Validation (`1,292` images) / 10% Test (`1,292` images) |
| **Stratification & Small-Class Floor** | Stratified across all 20 classes. Enforced an **8-image evaluation floor** protecting small classes (`rice__leaf_smut` split 24 Train / 8 Val / 8 Test = 60/20/20% ratio). |
| **Leakage Prevention Guarantee** | Whole pHash `group_id` clusters (`11,530` unique groups) were assigned indivisibly to a single split. **Verified `LEAKAGE = 0 groups spanning splits`**. |
| **Materialized Structure** | ImageFolder-ready directory layout (`final/train`, `final/val`, `final/test`), comprehensive `master_manifest.csv`, and frozen `label_to_idx.json` (273 MB on disk). |

**9.2 PlantDoc (Field-Evaluation Holdout)**

PlantDoc (`2,598 images`) is reserved **exclusively as a held-out field-robustness evaluation test set** and is never seen during model training or validation.

**9.3 Yield Prediction Dataset**

To prevent temporal autocorrelation leakage across agricultural cycles, the dataset is partitioned **strictly chronologically by agricultural start year**:

| Parameter | Value |
|---|---|
| **Split logic** | Temporal (Chronological by `Crop_Year` start year) |
| **Leakage prevention** | Historical years never learn from future years; spatial-temporal boundaries strictly preserved across train, validation, and test sets. |
| **Train Set (1997–2018)** | **3,256 records (81.5%)** — Historical baseline training cohort (`data/final/yield/train_yield.csv`). |
| **Validation Set (2019–2020)** | **296 records (7.4%)** — Out-of-time tuning cohort (`data/final/yield/val_yield.csv`). |
| **Holdout Test Set (2021–2023)** | **444 records (11.1%)** — Strict out-of-time evaluation cohort (`data/final/yield/test_yield.csv`). |

---

### 10. Final Prepared Dataset

**10.1 Vision — Unified Merged & Split Training Dataset (`12,859` images, 20 classes)**

| Metric | Completed Value |
|---|---|
| Total Clean Unique Images | **12,859 images** (10,673 Wheat + 2,066 Rice S1 + 120 Rice S2) |
| Total Canonical Classes | **20 classes** (15 Wheat + 5 Rice: `bacterial_blight`, `blast`, `brown_spot`, `leaf_smut`, `tungro`) |
| Image Resolution & Format | All letterboxed 256×256 px RGB JPEGs (`.convert("RGB")` verified) |
| Split Breakdown | **Train:** 10,275 (79.9%) · **Val:** 1,292 (10.0%) · **Test:** 1,292 (10.0%) |
| Leakage Proof | **0 groups spanning splits** across 11,530 unique `group_id` clusters |
| Frozen Index Artifact | `label_to_idx.json` mapping all 20 canonical classes |
| Readiness for Milestone 3 | ✅ **Complete & Training-Ready (`final/` directory, 273 MB disk footprint)** |

**10.2 Vision — PlantDoc (Field Robustness Holdout Set)**

| Metric | Value |
|---|---|
| Images | 2,598 real-field farmer images |
| Role | Field-robustness evaluation only — never used in training |
| Preprocessing | None applied; resize + normalize at inference time |

| Metric | Value |
|---|---|
| Images | 2,598 |
| Role | Field-robustness evaluation only — never used in training |
| Preprocessing | None applied yet; resize + normalize at inference time |

**10.4 RAG PDF Corpus**

| Metric | Value |
|---|---|
| Final documents | 170 (from 187 raw) |
| Final chunks | 1,451 (ready for embedding) |
| Preprocessing completed | OCR, de-duplication, chunking, metadata schema |
| Readiness | ✅ Chunks ready for MuRIL embedding in Milestone 3 |

**10.5 KCC Dataset**

| Metric | Value |
|---|---|
| Raw records (2020–2025, UP) | 3,123,029 |
| After deduplication (Q+A pair) | ~2,300,000 (estimate; 26.15% exact Q+A duplicates to be removed) |
| Remaining before training-ready | Deduplication script to be executed; MuRIL chunking to be applied |

**10.6 Yield Prediction Dataset**

| Metric | Primary Multi-Crop Dataset (Tanmay's Work) | Complementary UP Rice & Wheat Subset (Lokesh's Work) |
|---|---|---|
| **Raw / Unified records** | `440,962` records spanning 1997–2024 across 35 States/UTs | `3,886` records spanning 1997–2023 across 74 UP districts |
| **Processed records** | **`440,962` records** across **124 unique crops** | **`3,996` harmonized records** (backcasted across bifurcated districts) |
| **Feature space** | 16 core attributes (`state`, `district`, `crop`, `season`, APY, rainfall, fertilizer) | 23 attributes (adding daily IMD weather stress & ICRISAT NPK/irrigation features) |
| **Preprocessing completed** | Multi-source schema normalization, hierarchical deduplication, coconut unit conversion, `MissForest` + `OrdinalEncoder` imputation | Zero-production median imputation, spatial bifurcation backcasting, 5 engineered domain predictors, chronological out-of-time splitting |
| **Final Exported Artifacts** | `production_unified.csv`, `production_unified_imputed.csv` | `train_yield.csv` (3,256), `val_yield.csv` (296), `test_yield.csv` (444) |
| **Readiness** | ✅ **Ready for pan-India benchmarking & multi-crop modeling in Milestone 3** | ✅ **Ready for localized UP state regression modeling in Milestone 3** |

---

### 11. Challenges Encountered

**Vision datasets:**

- **Background-bias shortcut in Rice Set 1 (Tungro class):** Tungro images are systematically different in framing and background (zoomed-out whole plants on bare soil) vs. the other three classes (tight leaf close-ups on neutral backgrounds). A CNN trained naively could classify Tungro by soil background rather than disease symptoms, creating a shortcut that collapses in real-field deployment. Mitigation: background augmentation, deliberate robustness testing on green-background Tungro images.
- **High within-class duplication in Rice Set 1:** Blast has 33.3% literal duplicate images. 82.8% of the full dataset is involved in near-duplicate groups. A naive random split would leak near-identical images into val/test, inflating evaluation metrics. Mitigation: group-aware splitting via pHash cluster IDs (mandatory).
- **Panoramic aspect ratio in Rice Set 2:** All images are ~3.44:1 (3,081×897 px) — naive square resize badly distorts lesion geometry. Mitigation: aspect-preserving letterboxing or strip-tiling.
- **Critical train/test leakage in Wheat dataset:** 646 perceptual hashes appear in more than one split — the original folder-based split is contaminated and must be discarded and rebuilt using group-aware splitting.
- **Label naming inconsistency in Wheat dataset:** 45 raw folder names had to be collapsed to 15 canonical classes by programmatic suffix stripping. Without this canonicalization the training loop would treat `Aphid`, `aphid_test`, and `aphid_valid` as three separate, meaningless classes.
- **Extreme image heterogeneity in Wheat dataset:** 5 different file formats, 4 color modes, aspect ratios ranging 0.09–18.23, and file sizes from 2.6 KB to 10.6 MB. All must pass through a unified `PIL.open() → .convert("RGB") → resize` pipeline.
- **Tiny val/test sets in Wheat dataset:** Only 20 val and 50 test images per class — too small for reliable per-class metric estimates.
- **Data quality problems (RAG/NLP corpus):** 14 of 187 PDFs were unreadable even after higher-DPI OCR and were excluded from the corpus.
- **Duplication (RAG/NLP corpus):** 33 near-duplicate document pairs resolved via automated similarity-based comparison.
- **Licensing constraints (RAG/NLP corpus):** Confirm exact usage terms per government portal before final submission, particularly for the MISS document sourced via general web search rather than an official stable URL.
- **High redundancy (KCC dataset):** 68.72% of query texts are duplicates, dominated by templated Weather and PM-KISAN status queries — requires deduplication before embedding to avoid retrieval-index skew.
- **Administrative District Bifurcations (Yield dataset):** 164 missing records (4.0%) occurred due to newly formed UP districts post-1997 (Amethi, Sambhal, Hapur, Shamli). Resolved via spatial parent-district proportional apportionment (~28% area share) to guarantee longitudinal continuity.
- **Bureaucratic Zero-Reporting Anomalies (Yield dataset):** 29 sporadic records (0.75%) listed `Production_Total == 0.0` and `Yield_Kg_Ha == 0.0` despite positive `Area_Sown`. Resolved via agro-climatic zone seasonal median imputation.

---

### 12. Deliverables Produced

**System Architecture & Sprint Planning Deliverables (Lokesh's Work):**
- [`docs/Milestone_2_Implementation_Plan.md`](../docs/Milestone_2_Implementation_Plan.md) — Comprehensive architecture and implementation blueprint establishing the 3-stream data pipeline design (Vision, KCC/RAG, Yield), upfront design decisions, directory structure (`data/raw|processed|final`), sprint work breakdown across all 5 teammates, and automated/manual leakage verification protocols.

**Vision datasets & pipeline deliverables:**
- **EDA Notebooks & Narratives:**
  - [`rice-leaf-disease-dataset-EDA.ipynb`](../data/Vision_dataset/rice-leaf-disease-dataset-EDA.ipynb) & [`Rice leaf disease dataset documentation.odt`](../data/Vision_dataset/Rice%20leaf%20disease%20dataset%20documentation.odt) — Full EDA for Rice Set 1 (5,932 images).
  - [`rice-leaf-disease-dataset-set-2-eda.ipynb`](../data/Vision_dataset/rice-leaf-disease-dataset-set-2-eda.ipynb) & [`Rice_leaf_disease_dataset_set2_documentation.odt`](../data/Vision_dataset/Rice_leaf_disease_dataset_set2_documentation.odt) — Full EDA for Rice Set 2 (120 images).
  - [`wheat-dataset-EDA.ipynb`](../data/Vision_dataset/wheat-dataset-EDA.ipynb) & [`Wheat_dataset_Documentation.odt`](../data/Vision_dataset/Wheat_dataset_Documentation.odt) — Full EDA for Wheat Dataset (14,154 images).
- **Executable Preprocessing & Dataset Integration Documentation (Mahesh's Work):**
  - [`docs/Milestone_2_work/wheat_preprocessing_documentation.md`](../docs/Milestone_2_work/wheat_preprocessing_documentation.md) — Technical documentation of Wheat label canonicalization (45→15 classes) and duplicate cleaning (14,154 → 10,673 unique groups).
  - [`docs/Milestone_2_work/rice_set1_preprocessing_documentation.md`](../docs/Milestone_2_work/rice_set1_preprocessing_documentation.md) — Documentation of Rice Set 1 burst-capture thinning (5,932 → 2,066 clean images) and 256×256 RGB letterbox shortcut removal.
  - [`docs/Milestone_2_work/rice_set2_preprocessing_documentation.md`](../docs/Milestone_2_work/rice_set2_preprocessing_documentation.md) — Documentation of Rice Set 2 panoramic aspect-preserving letterbox standardization (120 images, 3 classes).
  - [`docs/Milestone_2_work/notebookD_merge_split_documentation.md`](../docs/Milestone_2_work/notebookD_merge_split_documentation.md) — Unified integration & centralized group-aware stratified split (80/10/10 across **12,859 images and 20 classes**, 0 leakage, materialized to `final/train|val|test` + `master_manifest.csv` + `label_to_idx.json`).
  - [`docs/Milestone_2_work/notebook_training_pipeline_design.md`](../docs/Milestone_2_work/notebook_training_pipeline_design.md) — Training-time data loader design: live random/center cropping to 224², ImageNet normalization, rare-class augmentation, hybrid imbalance handling, and Tungro Grad-CAM robustness check.

**RAG/NLP corpus & PDF deliverables (Harliv's Work):**
- [`docs/Milestone_2_work/rag_pdf_report.md`](../docs/Milestone_2_work/rag_pdf_report.md) — Comprehensive technical report on authoritative agricultural PDF collection (187 documents across 4 folders), cleaning, OCR fallback, deduplication (170 retained docs), and sentence-aware 512-token semantic chunking (**1,451 chunks**).
- `pdf_inventory_clean.csv` (170 clean documents), extracted `.txt` files per PDF, `excluded_unreadable_docs.csv`, `excluded_near_duplicate_docs.csv`, `PDF_Corpus_EDA.ipynb`, `PDF_Chunking.ipynb`, `pdf_chunks.csv` / `pdf_chunks.jsonl` (1,451 chunks ready for Milestone 3 embedding).

**KCC advisory dataset deliverables (Aneeqa's Work):**
- [`docs/Milestone_2_work/KCC Data EDA.md`](../docs/Milestone_2_work/KCC%20Data%20EDA.md) — Comprehensive technical report on Kisan Call Center dataset aggregation (**3,123,029 records**, 2020–2025), crop/category profiling (Rice+Wheat = **31.95%**), multilingual query-answer alignment (**99.98% English queries -> 98.80% Hindi answers**), Q&A deduplication requirements (`68.72%` duplicate queries), and 512-character chunking verification (**98.9% compatibility**).
- `kcc_combined_2020_2025.csv` (3.12M records across 15 attributes) & `03_kcc_rag_eda.ipynb`.

**Yield dataset:**
- **Primary Work (Pan-India Multi-Crop):**
  - `production_unified.csv` & `production_unified_imputed.csv` — Full 440,962-record multi-crop production and yield dataset (1997–2024 across 35 States/UTs and 124 crops).
  - [`notebooks/07_Yield_EDA+ preprocessing.ipynb`](../notebooks/07_Yield_EDA+%20preprocessing.ipynb) — Full executable notebook performing multi-source schema unioning, coconut unit conversion, EDA correlation matrix, and `MissForest` imputation.
  - [`docs/Milestone_2_work/yield_report.md`](../docs/Milestone_2_work/yield_report.md) — Comprehensive technical documentation of the multi-crop pipeline.
- **Complementary UP Rice & Wheat Subset:**
  - `up_district_yield_apy_1997_2023.csv` — Focused 15-attribute UP district yield dataset enriched with IMD daily weather and ICRISAT NPK inputs.
  - [`notebooks/05_yield_eda.ipynb`](../notebooks/05_yield_eda.ipynb) & [`notebooks/06_yield_preprocessing.ipynb`](../notebooks/06_yield_preprocessing.ipynb) — Specialized UP notebooks evaluating district bifurcations, regional disparities, and out-of-time chronological splitting (`train/val/test_yield.csv`).

---

### 13. Summary and Next Steps

**Summary of work completed**

Milestone 2 delivered comprehensive EDA, data cleaning, integration, and preprocessing across all vision, RAG/PDF, KCC, and crop yield prediction datasets. For the Vision subsystem, three datasets were thoroughly cleaned and unified: Wheat label canonicalization (45→15 classes), Rice Set 1 burst-capture deduplication and thinning (5,932 → 2,066 clean images), and Rice Set 2 panoramic letterboxing (120 images). These three sources were concatenated into a **unified 20-class dataset of 12,859 images** and partitioned via a centralized group-aware stratified split (80/10/10) with verified zero cross-split leakage (`0 groups spanning splits`). For the RAG corpus, 170 clean PDF documents were chunked into 1,451 MuRIL-ready chunks. The KCC dataset (3.12M records) was profiled, with crop/category/temporal/language distributions fully mapped. For the Yield Prediction subsystem, a primary **unified multi-crop agricultural dataset (`440,962` records spanning 124 crops across India)** was constructed, deduplicated, and preprocessed using non-parametric `MissForest` imputation (`production_unified_imputed.csv`). Complementing this, a specialized **UP Rice/Wheat domain subset (`3,996` records across 75 districts)** was harmonized across administrative bifurcations, enriched with IMD/ICRISAT climate and input covariates, and partitioned chronologically into leak-free training, validation, and holdout test splits.

**Key observations from the data**

| Observation | Impact |
|---|---|
| Rice Set 1: 33.3% Blast images are exact duplicates | Group-aware splitting is mandatory — naive splits inflate metrics |
| Rice Set 1: Tungro has a background-bias shortcut | Background augmentation required; robustness testing on green-background Tungro needed |
| Rice Set 2: Panoramic 3.44:1 aspect ratio | Aspect-preserving preprocessing mandatory — naive square resize distorts lesions |
| Rice Set 2: Perfectly clean, zero duplicates | High-quality complement to Set 1 despite tiny size |
| Wheat: 45 raw labels collapsed to 15 canonical | Canonicalization was a prerequisite — training was impossible with raw labels |
| Wheat: 646 cross-split duplicate hashes | Existing split is invalid; full re-split with group-aware logic required |
| Wheat: 5.6:1 train imbalance (smut vs stem_fly) | Class weighting + targeted augmentation for minority classes required |
| Wheat: Diverse visual targets (pests, foliar, spike) | Backbone must handle multi-scale feature detection |
| KCC: 68.7% duplicate QueryText values | Deduplication mandatory before embedding to prevent index skew |
| KCC: English queries / Hindi answers | MuRIL's multilingual alignment is essential for retrieval |
| Yield (Multi-Crop): `whole year` and `kharif` dominate national production | Validates seasonal stratification across 124 crops in `production_unified.csv` |
| Yield (UP Subset): Strong correlation (+0.81 to +0.90) with irrigation share | Confirms canal/tubewell density drives Western UP vs Bundelkhand yield disparity |

**Confirmation of dataset readiness**

| Dataset | Readiness for Milestone 3 |
|---|---|
| Unified Vision Training Dataset (`12,859` images across 20 classes) | ✅ **Complete & Training-Ready; materialized to `final/train|val|test` (80/10/10 split, 0 leakage)** |
| PlantDoc (Evaluation Holdout Set) | ✅ **Reserved as held-out field-robustness test set (`2,598` images); no training use** |
| RAG PDF chunks | ✅ **1,451 chunks ready for MuRIL embedding** |
| KCC dataset | ⚠️ Deduplication script to be executed before embedding |
| Primary Multi-Crop Yield Dataset (`production_unified_imputed.csv`) | ✅ **Complete; 440,962 records across 124 crops ready for pan-India benchmarking** |
| Complementary UP Yield Subset (`data/final/yield/`) | ✅ **Complete; 3,996 harmonized records split chronologically for localized UP modeling** |

**Planned activities for Milestone 3**

1. Execute pHash-group-aware clean re-split for Wheat dataset (resolve leakage).
2. Merge Rice Set 1 (deduplicated) + Rice Set 2 with label harmonization into a unified 5-class rice training set.
3. Implement and validate the augmentation pipeline (Tier 1 standard + Tier 2 field-robustness augmentation).
4. Fine-tune EfficientNet-B0 / ViT-Small on the rice and wheat datasets; report lab-set metrics (PlantVillage baseline).
5. Evaluate on PlantDoc to quantify the lab-to-field domain gap.
6. Embed 1,451 PDF chunks using MuRIL into ChromaDB/FAISS vector store.
7. Execute KCC deduplication and embed the deduplicated Q&A pairs using MuRIL.
8. Benchmark retrieval: Recall@5, MRR, nDCG — generic embeddings vs. MuRIL.
9. Train crop yield regression models (Random Forest, XGBoost, LightGBM) utilizing both the multi-crop unified production dataset and the specialized UP Rice/Wheat dataset.

---

## Team Review & Sign-Off

| # | Team Member | Role | Reviewed & Approved | Date | Signature |
|:-:|-------------|------|:-------------------:|:----:|-----------|
| 1 | Mahesh | Comprehensive Vision EDA, Preprocessing, Integration & Training Pipeline Design | ☑ | 2026-07-09 | Signed - Mahesh |
| 2 | Harliv | Comprehensive RAG PDF Corpus Collection, EDA, Cleaning & Semantic Chunking | ☑ | 2026-07-09 | Signed - Harliv |
| 3 | Lokesh | Milestone 2 Implementation Plan, Identifying Vision Data Sources, Primary Vision EDA, UP Rice/Wheat Yield Subset & Report Authoring | ☑ | 2026-07-09 | Signed - Lokesh |
| 4 | Aneeqa | Comprehensive KCC Dataset Aggregation (3.12M), EDA & Multilingual RAG Preparation | ☑ | 2026-07-09 | Signed - Aneeqa |
| 5 | Tanmay | Primary Multi-Crop Yield Dataset Unification, EDA & MissForest Preprocessing | ☑ | 2026-07-09 | Signed - Tanmay |

**Document version:** Milestone 2 — Updated with Vision, RAG, KCC & Crop Yield Subsystem Findings · **Prepared:** July 2026