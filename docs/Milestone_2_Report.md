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

**2.4 Yield Prediction Dataset**

* **Dataset name(s):** Uttar Pradesh District-Level Historical Crop APY & Environmental/Agronomic Covariates (1997–2023)
* **Source(s) and download links:**
  * **Primary Target (APY Statistics):** Unified Portal for Agricultural Statistics (UPAg) / Department of Agriculture & Farmers Welfare (DA&FW): https://upag.gov.in/ and Directorate of Economics & Statistics UP (https://updes.up.nic.in/).
  * **Climatic Covariates:** India Meteorological Department (IMD) Pune High Spatial Resolution Daily Gridded Rainfall & Temperature: https://imdpune.gov.in/.
  * **Agronomic Inputs & Harmonization:** ICRISAT District Level Database (DLD): http://data.icrisat.org/dld/ and CEIC / DES NPK Fertilizer Consumption: https://data.desagri.gov.in/.
* **Public/private/licensed status:** All primary government datasets are published under the **Open Government Data (OGD) License India**, allowing royalty-free academic research and machine learning applications. ICRISAT DLD is Open Access under standard institutional terms.
* **Purpose:** Provides historical district-level Area, Production, and Yield (APY) statistics alongside meteorological and agronomic covariates to train a machine learning regression model estimating district-wise yield (`Yield_Kg_Ha`) for Kharif Rice and Rabi Wheat across all 75 UP administrative districts.
* **Why each dataset was selected:**
  * **UPAg APY Dashboard:** Represents the official, continuously updated crop production records from the Ministry of Agriculture & Farmers Welfare, preferred over static compilations.
  * **IMD Gridded Climate Data:** Gold-standard ground-gauge observational dataset capturing localized monsoon precipitation shocks and March terminal heatwaves.
  * **ICRISAT DLD:** Crucial for historical NPK chemical input trends, tube-well irrigation coverage, and providing apportioned parent-district mappings that resolve district bifurcation discontinuities.
* **Alternatives considered:** Static Kaggle compilations (*Crop Production Statistics India*) and *Zila Sankhyikiya Patrika* PDF diaries were evaluated but rejected due to lack of dynamic updates, absence of weather/input covariates, and complex unstructured extraction. 

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
| **Number of records**   | 3,886 district-year-season records spanning 1997–1998 to 2023–2024 across 74 unique UP districts (164 records absent prior to formation of bifurcated districts).       |
| **Number of features**  | 15 raw columns across administrative, agricultural output, climatic, and agronomic input categories.                                                                    |
| **Target variable(s)**  | `Yield_Kg_Ha` (Continuous regression target: crop yield in Kilograms per Hectare sown).                                                                                 |
| **Feature description** | Combines administrative keys (`State_Name`, `District_Name`, `Agro_Climatic_Zone`, `Crop_Year`, `Season`, `Crop`), production metrics (`Area_Sown`, `Production_Total`), IMD weather stresses (`Precip_Seasonal_mm`, `Rain_Days_Extreme`, `Temp_Max_Avg`, `Heatwave_Days`), and ICRISAT agronomic inputs (`Fertilizer_N/P/K_Tonnes`, `Net_Irrigated_Pct`, `Tubewell_Irrig_Pct`). |
| **Data format**         | Structured Tabular CSV (`data/raw/yield/up_district_yield_apy_1997_2023.csv`).                                                                                          |
| **Dataset schema**      | Detailed 15-attribute schema described in the table below.                                                                                                              |

#### Complete 15-Attribute Schema (`up_district_yield_apy_1997_2023.csv`)

| Column Name | Data Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `State_Name` | String | "Uttar Pradesh" | Administrative state identifier. |
| `District_Name` | String | 74 Unique Districts | Standardized administrative UP district name (e.g., Agra, Meerut, Varanasi, Jhansi). |
| `Agro_Climatic_Zone` | Categorical | 4 Zones | Macro-region classification (`Western UP`, `Central UP`, `Eastern UP`, `Bundelkhand`). |
| `Crop_Year` | String | "1997-98" to "2023-24" | Agricultural calendar year string (`YYYY-YY`). |
| `Season` | Categorical | `Kharif` or `Rabi` | Cropping season (`Kharif` for Rice, `Rabi` for Wheat). |
| `Crop` | Categorical | `Rice` or `Wheat` | Target staple commodity. |
| `Area_Sown` | Float | Hectares (ha) | Gross cropped area sown for the district-crop-year. |
| `Production_Total` | Float | Tonnes | Total harvested output reported by DA&FW. |
| `Yield_Kg_Ha` | Float | kg / ha (**Target**) | Calculated productivity target (`Production_Total * 1000 / Area_Sown`). |
| `Precip_Seasonal_mm` | Float | Millimeters (mm) | Total cumulative seasonal precipitation from IMD Pune daily grids. |
| `Rain_Days_Extreme` | Integer | Days | Count of extreme rainfall events (> 64.5 mm/day) during flowering/grain filling. |
| `Temp_Max_Avg` | Float | Celsius (°C) | Seasonal average daily maximum temperature. |
| `Heatwave_Days` | Integer | Days | Severe terminal heatwave days (> 38°C in March for Wheat; 0 for Rice). |
| `Fertilizer_N/P/K_Tonnes` | Float | Tonnes | Total Nitrogen (`N`), Phosphate (`P`), and Potash (`K`) consumption. |
| `Net_Irrigated_Pct` / `Tubewell_Irrig_Pct` | Float | Percentage (%) | Net irrigated area share and proportion serviced by mechanized tubewells. |

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

Full executable EDA notebook is available at:
- [`notebooks/05_yield_eda.ipynb`](../notebooks/05_yield_eda.ipynb)

#### 5.4.1 Dataset Profile & Missingness Audit
- **Total Records:** Evaluated 3,886 district-year-season records across 74 unique Uttar Pradesh districts spanning agricultural years 1997–1998 to 2023–2024.
- **Administrative Bifurcations:** Out of 4,050 theoretical district-season slots (75 districts × 27 years × 2 seasons), exactly **164 records (4.0%) are absent** prior to historical district formation dates (Amethi carved out in 2010; Sambhal, Hapur, and Shamli carved out in 2011).
- **Bureaucratic Reporting Anomalies:** Detected **29 records (0.75%)** with `Production_Total == 0.0` and `Yield_Kg_Ha == 0.0` despite active `Area_Sown`, representing sporadic bureaucratic data-entry omissions requiring regional median imputation.

#### 5.4.2 Target Variable Distribution (`Yield_Kg_Ha`)

| Crop | Season | Count | Mean (kg/ha) | Std | Median | IQR (Q1 – Q3) | Min – Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rice** | Kharif | 1,933 | 2,459.9 | 466.5 | 2,447.4 | 628.2 (2,144.7 – 2,772.9) | 1,134.1 – 3,917.4 |
| **Wheat** | Rabi | 1,924 | 2,786.2 | 750.7 | 2,751.0 | 980.5 (2,302.8 – 3,283.3) | 920.3 – 4,750.0 |

#### 5.4.3 Agro-Climatic Regional Disparity Analysis
UP exhibits a pronounced agricultural productivity divide between canal/tubewell-intensive Western UP and water-insecure Bundelkhand:

| Agro-Climatic Zone | Mean Rice Yield (kg/ha) | Mean Wheat Yield (kg/ha) | Net Irrigated Area (%) | Tubewell Share (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Western UP** | 2,893.8 | 3,551.9 | 94.0% | 86.0% |
| **Central UP** | 2,403.9 | 2,747.4 | 82.7% | 77.8% |
| **Eastern UP** | 2,292.3 | 2,444.6 | 74.5% | 72.9% |
| **Bundelkhand** | 1,719.7 | 1,490.8 | 50.6% | 41.8% |

#### 5.4.4 Key Covariate Correlations (Pearson *r* with `Yield_Kg_Ha`)
- **Irrigation & Input Technology:** Strong positive correlation with `Net_Irrigated_Pct` (**+0.813** for Rice, **+0.899** for Wheat) and `Fertilizer N Intensity` (**+0.803** for Rice, **+0.864** for Wheat), verifying Green Revolution input sensitivity.
- **Meteorological Stress Drivers:**
  - `Rain_Days_Extreme` (>64.5 mm/day) exhibits a significant negative correlation (**-0.223**) with Kharif Rice yield due to monsoon lodging and flower submergence.
  - `Heatwave_Days` (>38°C in March) exhibits a strong negative correlation (**-0.241**) with Rabi Wheat yield, capturing terminal heat stress during grain filling.

---

### 6. Data Preprocessing

**6.1 Vision**

**Label correction (Wheat dataset — critical)**

The raw folder scan detected 45 apparent class labels due to the dataset's folder naming convention (e.g., `Aphid`, `aphid_test`, `aphid_valid`). A canonicalization function was applied:
1. Strip leading/trailing whitespace; lowercase.
2. Replace spaces and hyphens with underscores.
3. Iteratively strip `_test`, `_valid`, `_val`, `_train` suffixes.
4. Collapse repeated underscores.

This collapsed 45 raw labels to **15 canonical disease classes**, with the original raw label preserved in a `label_raw` column for audit purposes. Without this step, the dataset would be unusable for classification.

**Corrupt/missing file handling**

| Dataset | Corrupt images | RGBA/non-RGB files | Action |
|---|---|---|---|
| Rice Set 1 | 0 | 156 RGBA (PNG disguised as .jpg) | Apply `.convert("RGB")` in data loader |
| Rice Set 2 | 0 | 0 | No action required |
| Wheat | 0 | RGBA: 1,613 · P: 47 · CMYK: 2 | Apply `.convert("RGB")` in data loader |

All datasets had **zero corrupt or unreadable images**. A `.convert("RGB")` guard is mandatory in the data loader to handle RGBA, P (palette), and CMYK modes.

**Duplicate removal**

| Dataset | Exact duplicates | Action |
|---|---|---|
| Rice Set 1 | 2,234 byte-identical files (1,096 groups) | Flag with `is_exact_rep` column; train on `is_exact_rep=True` subset only (4,794 unique images) |
| Rice Set 2 | 0 | No action required |
| Wheat | ~6,279 near-duplicates (aHash); 646 cross-split leakers | Requires stricter pHash deduplication and clean re-splitting before training |

**Image resizing / normalization**

| Dataset | Issue | Recommended treatment |
|---|---|---|
| Rice Set 1 | Mixed sizes; Tungro images not 300×300 | Resize all to 224×224 (or 256×256) using aspect-preserving letterbox then center-crop |
| Rice Set 2 | Panoramic 3:1 aspect ratio (3,081×897 px) | **Do not use naive square resize.** Use letterbox/pad-to-square, or tile into 224×224 crops, or train at wide resolution (e.g. 448×224) |
| Wheat | Extreme size variance (44px–6,016px); mixed formats | Resize to 224×224 with aspect-preserving letterbox; decode all formats via PIL then convert to RGB tensor |

All pixel values should be normalized to `[0, 1]` then standardized with ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` for transfer-learning backbones (EfficientNet-B0, ViT-Small).

**Standardization (manifests)**

For Rice Set 1, a deduplicated manifest (`rice_leaf_manifest.csv`) was exported with the following key columns:
- `dup_cluster` — pHash-based group ID; use as the group key for `StratifiedGroupKFold` / `GroupShuffleSplit` to prevent near-duplicate leakage across splits.
- `is_exact_rep` — `True` for one representative per exact-duplicate group; filter on this column to train on the 4,794 unique images.

For Wheat, metadata is exported to `wheat_image_metadata.csv` with `width`, `height`, `mode`, `format`, `size_kb`, `aspect`, and `ahash` columns.

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

**6.4 Yield Preprocessing & Feature Engineering**

Full executable preprocessing notebook is available at:
- [`notebooks/06_yield_preprocessing.ipynb`](../notebooks/06_yield_preprocessing.ipynb)

#### 6.4.1 Zero-Production Anomaly Imputation
Sporadic administrative reporting omissions where `Production_Total == 0.0` and `Yield_Kg_Ha == 0.0` despite non-zero `Area_Sown` (29 records, 0.75% of dataset) were imputed using the robust seasonal median yield of the corresponding agro-climatic zone (`Western UP`, `Central UP`, `Eastern UP`, or `Bundelkhand`).

#### 6.4.2 Parent-District Spatial Harmonization (Bifurcation Backcasting)
To resolve the 164 missing records caused by historical administrative bifurcations without introducing artificial discontinuities:
- Parent-district historical records (`Sultanpur` for `Amethi`, `Moradabad` for `Sambhal`, `Meerut` for `Hapur`, and `Muzaffarnagar` for `Shamli`) were apportioned proportionally (~28% area share) to backcast pre-formation records.
- This produced a complete, continuous dataset of **3,996 harmonized records** spanning all 74/75 districts over 27 years (1997–2023).

#### 6.4.3 Agronomic & Meteorological Feature Engineering (5 Engineered Predictors)
1. `NPK_Total_Intensity_Kg_Ha` (`Float`): Total chemical fertilizer applied per hectare (`(N + P + K) * 1000 / Area_Sown`).
2. `NPK_Balance_Ratio` (`Float`): Nitrogen-to-Phosphate ratio (`Fertilizer_N / Fertilizer_P`), capturing nutrient imbalance.
3. `Rainfall_Anomaly_Pct` (`Float`): District seasonal precipitation percentage deviation from its 27-year long-term mean (`(Precip - Mean_Precip) / Mean_Precip * 100`).
4. `Thermal_Stress_Index` (`Float`): Interaction term combining maximum daily temperature and heatwave days (`Temp_Max_Avg * (1 + 0.05 * Heatwave_Days)`).
5. `Irrigation_Security_Score` (`Float`): Weighted composite irrigation infrastructure index (`0.4 * Net_Irrigated_Pct + 0.6 * Tubewell_Irrig_Pct`).

---

### 7. Dataset Integration (if multiple datasets)

The vision subsystem integrates three datasets for training and evaluation. The approach follows **Strategy D** from the dataset proposal (Rice Set 1 + Rice Set 2 merged, plus Wheat separately), with PlantDoc reserved for field-robustness evaluation.

**Datasets to be combined**

| Role | Datasets |
|---|---|
| Rice disease training | Rice Set 1 (5,932 imgs → 4,794 unique) + Rice Set 2 (120 imgs) |
| Wheat disease training | Wheat Dataset (14,154 imgs, 15 classes) |
| Field-robustness evaluation (held-out) | PlantDoc (2,598 imgs, real-field photos — not used in training) |

**Integration methodology (Rice Set 1 + Set 2)**

1. **Schema alignment:** Both sets use folder-level disease labels. Canonical label names must be harmonized — Set 1 uses `Bacterialblight` and `Brownspot` while Set 2 uses `Bacterial leaf blight` and `Brown spot`. A mapping table will normalize these to shared canonical names before merging.
2. **Label scope:** Set 2 contains `Leaf Smut` which is absent from Set 1. Set 1 contains `Blast` and `Tungro` absent from Set 2. The merged dataset will have **5 rice disease classes** total: Bacterial Blight, Brown Spot, Leaf Smut, Blast, Tungro.
3. **Deduplication after merging:** After concatenation, a cross-dataset pHash check will be run to detect any near-duplicate images present in both sets (unlikely given their different sources but must be verified). Any cross-source duplicates will be flagged and the Set 2 copy retained (as it is the cleaner, non-redundant source).
4. **Handling conflicting image properties:** Set 2's panoramic 3:1 aspect ratio images require aspect-preserving tiling/letterboxing before merging with Set 1's near-square images. All images will be normalized to a uniform 224×224 input after format-safe loading.
5. **Handling conflicting attributes:** Set 1 contains RGBA/PNG-disguised-as-JPEG files; Set 2 contains only true RGB JPEGs. Both are handled by `.convert("RGB")` at load time.

---

### 8. Data Augmentation (if applicable)

**8.1 Vision — Planned Augmentation Strategy**

The EDA findings directly drive augmentation choices. Two tiers of augmentation are planned:

**Tier 1 — Standard augmentation (all classes)**

| Technique | Rationale |
|---|---|
| Random horizontal/vertical flip | Disease symptoms are orientation-invariant |
| Random rotation (±30°) | Leaf orientation varies in field photos |
| Color jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2) | Accounts for the wide brightness variance observed per class (Laplacian/brightness EDA) |
| Random crop & resize | Simulates zoom-level variation; addresses extreme size variance in wheat dataset |
| Gaussian blur (σ 0.5–2.0) | Simulates low-sharpness field photos (Laplacian variance outliers in wheat EDA) |

**Tier 2 — Field-robustness augmentation (targeted at domain gap)**

| Technique | Rationale |
|---|---|
| Background randomization (cutout / CopyPaste) | Directly counters the Tungro background-bias shortcut identified in EDA — model must not learn to classify disease by soil background |
| Random occlusion / leaf overlap synthesis | Simulates overlapping foliage in real-field photos |
| Lighting variation (random shadow, highlight overlays) | Accounts for harsh sunlight / shade conditions in farmer-captured photos |
| Motion blur | Simulates camera shake in low-tech devices |

**Minority class targeting (Wheat dataset)**

Given the 5.6:1 train imbalance, the three minority classes (stem_fly: 234, black_rust: 576, blast: 647) will receive **2–3× additional augmented samples** using the Tier 1 + Tier 2 pipeline, targeting a reduced effective imbalance ratio of ≤2.5:1.

> **Note:** Exact augmented sample counts will be reported once augmentation scripts are finalized in Milestone 3.

---

### 9. Dataset Splitting

**9.1 Rice Datasets (Set 1 + Set 2 merged)**

Neither rice dataset ships with pre-existing splits. The splitting strategy must account for the high near-duplicate rate in Set 1.

| Parameter | Value |
|---|---|
| **Split ratio** | 70% train / 15% val / 15% test |
| **Stratification** | Yes — stratified by class label to maintain class proportions across splits |
| **Leakage prevention** | **Group-aware split using `dup_cluster` as the group key** — `StratifiedGroupKFold` / `GroupShuffleSplit` ensures all near-duplicates of an image land in the same split. This is mandatory given that 79.2% of images have near-duplicates. A naive random split would leak near-identical training images into the validation/test sets, inflating evaluation metrics. |
| **Training base** | Apply `is_exact_rep=True` filter first, giving 4,794 unique Rice Set 1 images + 120 Rice Set 2 images = ~4,914 unique images before augmentation |
| **Approximate sizes (post-dedup, pre-augmentation)** | Train: ~3,440 · Val: ~737 · Test: ~737 |

**9.2 Wheat Dataset**

The Wheat dataset ships with pre-defined train/val/test folders. However the EDA found **646 near-identical images appearing across multiple splits** (aHash leakage), invalidating the existing split.

| Parameter | Value |
|---|---|
| **Current split (raw, do not use)** | Train 13,104 / Val 300 / Test 750 — contaminated by leakage |
| **Planned action** | Re-split from scratch using pHash-based group-aware splitting, discarding the original folder-based split |
| **Target split ratio** | 70% train / 15% val / 15% test |
| **Stratification** | Yes — stratified by canonical class label |
| **Leakage prevention** | `GroupShuffleSplit` with aHash / pHash cluster as group key; verify no group spans two splits post-split. |
| **Approximate sizes (post-clean-split)** | Train: ~9,900 · Val: ~2,100 · Test: ~2,100 |

**9.3 PlantDoc (Field-Evaluation Holdout)**

PlantDoc is used **exclusively as a held-out field-robustness test set** and is never seen during training or validation. No splitting is applied to it.

**9.4 Yield Prediction Dataset**

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

**10.1 Vision — Rice (merged Set 1 + Set 2)**

| Metric | Value |
|---|---|
| Raw images | 6,052 (5,932 + 120) |
| After exact deduplication | ~4,914 unique images |
| Final classes | 5 (Bacterial Blight, Brown Spot, Leaf Smut, Blast, Tungro) |
| Deduped imbalance ratio | ~1.40:1 (Bacterial Blight vs Blast after dedup) |
| Preprocessing completed | Deduplication (MD5 + pHash), RGBA→RGB conversion, label harmonization, manifest export, group-aware split keys assigned |
| Readiness | ✅ Ready for data loader implementation and augmentation pipeline in Milestone 3 |

**10.2 Vision — Wheat**

| Metric | Value |
|---|---|
| Raw images | 14,154 |
| After deduplication & leakage removal | ~13,000 (estimate pending pHash clean-split) |
| Final classes | 15 |
| Train imbalance ratio | 5.6:1 (smut vs stem_fly in raw train set) |
| Preprocessing completed | Label canonicalization (45→15 classes), mode conversion flag, metadata export (`wheat_image_metadata.csv`), duplicate/leakage detection (646 cross-split hashes flagged) |
| Remaining before training-ready | ❌ Cross-split leakage must be resolved (clean re-split required) |

**10.3 Vision — PlantDoc (evaluation only)**

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

| Metric | Value |
|---|---|
| **Raw records** | 3,886 district-year-season records across 74 unique UP districts (1997–2023) |
| **After spatial harmonization & imputation** | **3,996 harmonized records** (backcasted for bifurcated districts Amethi, Sambhal, Hapur, Shamli) |
| **Total Features** | **23 features** (15 raw attributes + 5 engineered domain predictors + administrative identifiers) |
| **Preprocessing completed** | Bureaucratic zero-production imputation, spatial parent backcasting, NPK intensity/ratio engineering, rainfall anomaly & thermal stress engineering, chronological splitting |
| **Final Exported Splits** | Train: 3,256 records (`train_yield.csv`) · Val: 296 records (`val_yield.csv`) · Test: 444 records (`test_yield.csv`) under `data/final/yield/` |
| **Readiness** | ✅ **Ready for regression model training & evaluation in Milestone 3** |

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

**Vision datasets:**
- [`rice-leaf-disease-dataset-EDA.ipynb`](../data/Vision_dataset/rice-leaf-disease-dataset-EDA.ipynb) — Full EDA for Rice Set 1 (5,932 images): class distribution, corruption check, RGBA/PNG mystery resolution, exact + near-duplicate detection (MD5 + aHash + pHash), deduplicated manifest export.
- [`rice-leaf-disease-dataset-set-2-eda.ipynb`](../data/Vision_dataset/rice-leaf-disease-dataset-set-2-eda.ipynb) — Full EDA for Rice Set 2 (120 images): confirms pristine, zero-duplicate, perfectly balanced dataset with panoramic aspect-ratio preprocessing note.
- [`wheat-dataset-EDA.ipynb`](../data/Vision_dataset/wheat-dataset-EDA.ipynb) — Full EDA for Wheat Dataset (14,154 images): label canonicalization (45→15 classes), image properties, duplicate detection with leakage flagging (646 cross-split hashes), color/brightness and sharpness (Laplacian variance) per-class analysis.
- `rice_leaf_manifest.csv` — Deduplicated manifest for Rice Set 1 with `dup_cluster`, `is_exact_rep`, `md5`, `phash` columns; ready for group-aware splitting in Milestone 3.
- `wheat_image_metadata.csv` — Enriched metadata for all 14,154 Wheat images including width, height, mode, format, size_kb, aspect, ahash.
- [`Rice leaf disease dataset documentation.odt`](../data/Vision_dataset/Rice%20leaf%20disease%20dataset%20documentation.odt) — Narrative EDA interpretation for Rice Set 1.
- [`Rice_leaf_disease_dataset_set2_documentation.odt`](../data/Vision_dataset/Rice_leaf_disease_dataset_set2_documentation.odt) — Narrative EDA interpretation for Rice Set 2.
- [`Wheat_dataset_Documentation.odt`](../data/Vision_dataset/Wheat_dataset_Documentation.odt) — Narrative EDA interpretation for Wheat dataset.

**RAG/NLP corpus:**
- `pdf_inventory_clean.csv` (170 clean documents), extracted `.txt` files per PDF, `excluded_unreadable_docs.csv`, `excluded_near_duplicate_docs.csv`, `PDF_Corpus_EDA.ipynb`, `PDF_Chunking.ipynb`, `pdf_chunks.csv` / `pdf_chunks.jsonl` (1,451 chunks ready for Milestone 3 embedding).

**KCC dataset:**
- `kcc_combined_2020_2025.csv` (3.12M records), `03_kcc_rag_eda.ipynb`.

**Yield dataset:**
- `up_district_yield_apy_1997_2023.csv` — Raw 15-attribute UP district yield dataset (1997–2023).
- [`notebooks/05_yield_eda.ipynb`](../notebooks/05_yield_eda.ipynb) — Full executable EDA notebook evaluating district bifurcations, target distributions, regional disparities, and input/weather correlations.
- [`notebooks/06_yield_preprocessing.ipynb`](../notebooks/06_yield_preprocessing.ipynb) — Executable preprocessing notebook implementing zero-reporting imputation, spatial backcasting, 5 engineered features, and temporal splitting.
- `train_yield.csv`, `val_yield.csv`, `test_yield.csv` — Final leakage-free chronological datasets under `data/final/yield/`.

---

### 13. Summary and Next Steps

**Summary of work completed**

Milestone 2 delivered comprehensive EDA and preprocessing groundwork across all vision, RAG/PDF, KCC, and crop yield prediction datasets. For vision, three datasets were fully analyzed across 19–22 notebook cells each, uncovering critical data quality issues — Tungro background bias, Rice Set 1 duplication (33% of Blast class), Wheat label mislabelling (45→15 classes), and train/test leakage in Wheat (646 cross-split hashes). For the RAG corpus, 170 clean PDF documents were chunked into 1,451 MuRIL-ready chunks. The KCC dataset (3.12M records) was profiled, with crop/category/temporal/language distributions fully mapped. For the Yield Prediction subsystem, 3,886 historical records across 75 UP districts (1997–2023) were compiled, harmonized across district bifurcations (3,996 records), enriched with 5 domain features, and chronologically partitioned into training, validation, and holdout test cohorts.

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
| Yield: Strong positive correlation (+0.81 to +0.90) with irrigation share | Confirms canal/tubewell density drives Western UP vs Bundelkhand yield disparity |
| Yield: Negative correlation with extreme rain days and March heatwaves | Validates IMD climate stress predictors for Kharif Rice and Rabi Wheat |

**Confirmation of dataset readiness**

| Dataset | Readiness for Milestone 3 |
|---|---|
| Rice Set 1 (vision) | ✅ Manifest exported; group-aware split keys ready; `.convert("RGB")` guard documented |
| Rice Set 2 (vision) | ✅ Clean; preprocessing prescription documented (aspect-preserving resize) |
| Wheat (vision) | ⚠️ Leakage re-split required before training — pHash-group-aware split to be executed |
| PlantDoc (evaluation) | ✅ Reserved as held-out field-robustness test set; no training use |
| RAG PDF chunks | ✅ 1,451 chunks ready for MuRIL embedding |
| KCC dataset | ⚠️ Deduplication script to be executed before embedding |
| Yield Prediction dataset | ✅ **Complete; 3,996 harmonized records split chronologically into `data/final/yield/`** |

**Planned activities for Milestone 3**

1. Execute pHash-group-aware clean re-split for Wheat dataset (resolve leakage).
2. Merge Rice Set 1 (deduplicated) + Rice Set 2 with label harmonization into a unified 5-class rice training set.
3. Implement and validate the augmentation pipeline (Tier 1 standard + Tier 2 field-robustness augmentation).
4. Fine-tune EfficientNet-B0 / ViT-Small on the rice and wheat datasets; report lab-set metrics (PlantVillage baseline).
5. Evaluate on PlantDoc to quantify the lab-to-field domain gap.
6. Embed 1,451 PDF chunks using MuRIL into ChromaDB/FAISS vector store.
7. Execute KCC deduplication and embed the deduplicated Q&A pairs using MuRIL.
8. Benchmark retrieval: Recall@5, MRR, nDCG — generic embeddings vs. MuRIL.
9. Train district-level crop yield regression models (Random Forest, XGBoost, LightGBM) on `train_yield.csv` and evaluate out-of-time accuracy on `test_yield.csv`.

---

## Team Review & Sign-Off

| # | Team Member | Role | Reviewed & Approved | Date | Signature |
|:-:|-------------|------|:-------------------:|:----:|-----------|
| 1 | Mahesh | Comprehensive Vision EDA and Preprocessing | ☐ | | |
| 2 | Harliv | RAG corpus, PDF chunking | ☐ | | |
| 3 | Lokesh | Primary Vision EDA, report authoring | ☐ | | |
| 4 | Aneeqa | Data/API inventory, KCC EDA | ☐ | | |
| 5 | Tanmay | - | ☐ | | |

**Document version:** Milestone 2 — Updated with Vision EDA Findings · **Prepared:** July 2026