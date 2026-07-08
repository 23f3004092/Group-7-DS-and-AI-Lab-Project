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
| **Purpose** | Agronomic knowledge base for RAG retrieval |
| **Source** | [Kaggle — Farmers Call Query (KCC) Data Q&A](https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa) |
| **Format** | CSV (`questions`, `answers` columns) |
| **Total Size** | ~4.5 MB compressed (~178,939 Q&A records) |
| **Filtering Applied** | Agronomic queries related to Rice/Wheat & crop protection |
| **License** | CC0 1.0 |

---

### 5. Yield Prediction — District-Level Crop Data

| Property | Value |
|---|---|
| **Purpose** | Train classical ML model for district-level yield estimation |
| **Recommended Sources** | [UPAg Portal](https://upag.gov.in/), [DES Reports](https://aps.dac.gov.in/APY/Public_Report1.aspx), [ICRISAT](http://data.icrisat.org/dld/src/crops.html) |
| **Format** | CSV / Excel |
| **Filtering Applied** | State = Uttar Pradesh, Crops = Rice/Wheat |

---

## Quick Setup

### Automated Download Script

We provide a script to download datasets via Kaggle API:

```bash
# Download Strategy A Core + KCC + Yield instructions
python scripts/download_data.py --all

# Download Strategy A vision datasets only (Rice + Wheat + PlantDoc)
python scripts/download_data.py --rice --wheat --plantdoc

# Download Strategy D expansion datasets (Rice Extra + PlantVillage)
python scripts/download_data.py --expand

# Download everything
python scripts/download_data.py --everything
```

> **Note:** Kaggle downloads require a valid API token placed at `~/.kaggle/kaggle.json`.

---

## Data Flow

```
raw/ ──[EDA Notebooks]──> processed/ ──[Preprocessing Notebooks]──> final/
```
