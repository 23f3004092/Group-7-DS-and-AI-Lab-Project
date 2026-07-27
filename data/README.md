# Data Directory — AgriAssist

> **Important:** Raw datasets are large and are **excluded from Git** via `.gitignore`. Follow the setup instructions below to download them locally.

---

## Directory Structure

```
data/
├── raw/                    # ⛔ .gitignored — original downloads
│   ├── rice_diseases/      # Strategy A: Primary Rice Leaf Diseases dataset
│   ├── wheat_diseases/     # Strategy A: Primary Wheat Plant Diseases dataset
│   ├── rice_diseases_extra/# Strategy D: Extra Rice Leaf Disease images
│   ├── plantvillage/       # Strategy D: Supplementary transfer learning images
│   ├── plantdoc/           # Field evaluation / domain gap benchmark
│   ├── kcc/                # Kisan Call Centre Q&A CSV dump
│   ├── yield/              # District-level crop yield CSVs
│   └── pdfs/               # UP government advisory PDFs
│
├── processed/              # ✅ Tracked — cleaned & filtered data
│   ├── vision/             # Standardized Rice/Wheat images
│   ├── kcc/                # Filtered & cleaned KCC text
│   └── yield/              # Cleaned yield tabular data
│
└── final/                  # ✅ Tracked — training-ready splits
    ├── vision/
    │   ├── train/          # Combined Rice & Wheat train split
    │   ├── val/            # Validation split
    │   └── test/           # Held-out evaluation set / PlantDoc test
    ├── kcc/                # Chunked text, ready for MuRIL embedding
    └── yield/              # Train/val/test CSVs (temporal split)
```

---

## Dataset Inventory

### 1. Vision — Strategy A Core (Primary Rice & Wheat Datasets)

#### A. Rice Leaf Diseases Dataset
| Property | Value |
|---|---|
| **Purpose** | Core training & validation for Rice leaf disease classification |
| **Source** | [Kaggle — Rice Leaf Diseases (vbookshelf)](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) |
| **Classes** | Brown Spot, Blast, Bacterial Blight, Healthy |
| **Total Size** | ~38 MB (120 images per class, highly curated) |
| **License** | CC0 1.0 (Public Domain) |

#### B. Wheat Plant Diseases Dataset
| Property | Value |
|---|---|
| **Purpose** | Core training & validation for Wheat leaf disease classification |
| **Source** | [Kaggle — Wheat Plant Diseases (kushagra3204)](https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases) |
| **Classes** | Brown Rust, Yellow Rust, Stem Rust, Septoria, Blast, Powdery Mildew, Healthy, etc. |
| **Total Size** | ~14,000+ high-resolution images |
| **License** | Open Access / CC BY 4.0 |

---

### 2. Vision — Strategy D Expansion (Supplementary Vision Data)

#### A. Rice Leaf Disease Images (Extra)
| Property | Value |
|---|---|
| **Purpose** | Expansion dataset to enrich Rice disease variety and robustness |
| **Source** | [Kaggle — Rice Leaf Disease Images (nirmalsankalana)](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image) |
| **Total Size** | ~205 MB |

#### B. PlantVillage Dataset (Supplementary Pretraining)
| Property | Value |
|---|---|
| **Purpose** | General crop disease pretraining / transfer learning baseline |
| **Source** | [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Total Size** | ~3.0 GB (~54,000 images, 38 classes across various crops) |

---

### 3. Vision — Field Benchmark (PlantDoc)

| Property | Value |
|---|---|
| **Purpose** | In-the-wild / field-condition evaluation (measures lab-to-field domain gap) |
| **Source** | [Kaggle — PlantDoc Dataset](https://www.kaggle.com/datasets/andresmgs/plantdec) |
| **Total Size** | ~74 MB (30 classes across multiple crops) |
| **License** | CC BY 4.0 |

---

### 4. NLP / RAG — Kisan Call Centre (KCC) Q&A Logs

| Property | Value |
|---|---|
| **Purpose** | Agronomic knowledge base for RAG retrieval & Q&A pipeline |
| **Source** | [Official Government Portal — data.gov.in API (cef25fe2-9231-4128-8aec-2c948fedd43f)](https://data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f) |
| **Format** | `json` (JSONL + combined Parquet), `csv` (for EDA notebooks), `xml` |
| **Total Records** | ~3,123,029 records across 2020–2025 (for Uttar Pradesh) |
| **Filtering Applied** | StateName (`UTTAR PRADESH`), Year (`2020-2025`), optional Month (`1-12`) |
| **Authentication** | Requires `DATA_GOV_KEY` set inside the `.env` file at project root |

---

### 5. Yield Prediction — District-Level Crop Data

| Property | Value |
|---|---|
| **Purpose** | Train classical ML model for district-level yield estimation |
| **Recommended Sources** | [UPAg Portal](https://upag.gov.in/), [DES Reports](https://aps.dac.gov.in/APY/Public_Report1.aspx), [ICRISAT](http://data.icrisat.org/dld/src/crops.html) |
| **Format** | CSV / Excel |
| **Filtering Applied** | State = Uttar Pradesh, Crops = Rice/Wheat |

---

## Quick Setup & Example Usage

### 1. API Keys Configuration
Before downloading data, make sure your credentials are configured:
- **KCC API (`data.gov.in`)**: Add your API key to the `.env` file located at the project root (`d:\Group-7-DS-and-AI-Lab-Project\.env`):
  ```env
  DATA_GOV_KEY=your_actual_data_gov_in_api_key
  ```
- **Vision / Kaggle Datasets**: Ensure your Kaggle API token is present at `~/.kaggle/kaggle.json`.

---

### 2. Automated Download Script (`scripts/download_data.py`)

#### A. KCC Dataset (Official `data.gov.in` API)
The script automatically handles pagination, rate-limiting, retries, and compiles raw files into notebook-ready formats (`CSV`, `JSONL`, `Parquet`).

```bash
# 1. Download all 6 years (2020–2025) as CSV for EDA Notebook (notebooks/03_kcc_rag_eda.ipynb)
# Note: Omitting --kcc-months fetches all records across the year cleanly without looping over months.
python scripts/download_data.py --kcc --kcc-state "UTTAR PRADESH" --kcc-year "2020-2025" --kcc-format csv

# 2. Download a single year as JSONL + combined Parquet for RAG embedding
python scripts/download_data.py --kcc --kcc-state "UTTAR PRADESH" --kcc-year 2025 --kcc-format json

# 3. Download specific months only (e.g., January, June, December 2025)
python scripts/download_data.py --kcc --kcc-state "UTTAR PRADESH" --kcc-year 2025 --kcc-months "1,6,12"
```

#### B. Vision & Combo Downloads (Kaggle)
```bash
# Download Strategy A Core (Rice + Wheat + PlantDoc + KCC + Yield instructions)
python scripts/download_data.py --all

# Download Strategy A vision datasets only
python scripts/download_data.py --rice --wheat --plantdoc

# Download Strategy D expansion datasets (Rice Extra + PlantVillage)
python scripts/download_data.py --expand

# Download everything (All Vision + KCC + Yield instructions)
python scripts/download_data.py --everything
```

---

## Data Flow

```
raw/ ──[EDA Notebooks]──> processed/ ──[Preprocessing Notebooks]──> final/
```

---

## RAG PDF Corpus (`data/raw/pdfs/`)

187 UP government advisory/scheme PDFs across four source folders (`Other_docs`, `PPQS_Advisories`, `Schemes`, `UP_ACP_PDFs`), consumed by `notebooks/04_pdfs_rag_eda.ipynb`. Outputs mirror the KCC layout: intermediate artifacts in `data/processed/pdfs/`, the final chunk artifact `pdf_chunks_final.jsonl` in `data/final/pdfs/`.

```bash
python scripts/download_data.py --pdfs
```

Setup (one-time): zip the shared `DS_AI_RAG_pdfs` Drive folder, share the zip as "Anyone with the link", and put its file id in `.env` at the project root as `PDF_ZIP_DRIVE_ID=<file-id>` (a `PDF_FOLDER_DRIVE_ID` fallback exists, but gdown caps folder downloads at ~50 files per folder). Requires `pip install gdown`. Without an id, the script prints manual placement instructions.
