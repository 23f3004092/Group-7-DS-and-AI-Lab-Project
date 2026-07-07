# Data Directory — AgriAssist

> **Important:** Raw datasets are large (PlantVillage ~3GB, KCC ~500MB+) and are
> **excluded from Git** via `.gitignore`. Follow the setup instructions below to
> download them locally.

---

## Directory Structure

```
data/
├── raw/                    # ⛔ .gitignored — original downloads
│   ├── plantvillage/       # PlantVillage leaf disease images
│   ├── plantdoc/           # PlantDoc field leaf disease images
│   ├── kcc/                # Kisan Call Centre Q&A CSV dump
│   ├── yield/              # District-level crop yield CSVs
│   └── pdfs/               # UP government advisory PDFs
│
├── processed/              # ✅ Tracked — cleaned & filtered data
│   ├── vision/             # Filtered Rice/Wheat images only
│   ├── kcc/                # Filtered & cleaned KCC text
│   └── yield/              # Cleaned yield tabular data
│
└── final/                  # ✅ Tracked — training-ready splits
    ├── vision/
    │   ├── train/          # 80% PlantVillage (stratified)
    │   ├── val/            # 20% PlantVillage (stratified)
    │   └── test/           # PlantDoc (independent test set)
    ├── kcc/                # Chunked text, ready for MuRIL embedding
    └── yield/              # Train/val/test CSVs (temporal split)
```

---

## Dataset Inventory

### 1. Vision — PlantVillage (Training & Validation)

| Property | Value |
|---|---|
| **Purpose** | Lab-baseline image classification for crop disease detection |
| **Source** | [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Alt Source** | [Kaggle — New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) |
| **Format** | JPEG images organised in class-labeled subdirectories |
| **Total Size** | ~3.0 GB (~54,000 images, 38 classes) |
| **Classes Used** | Rice and Wheat disease classes only (filtered during preprocessing) |
| **License** | CC0 1.0 (Public Domain) |
| **Citation** | Hughes, D., Salathé, M. (2016). *An Open Access Repository of Images on Plant Health.* |

### 2. Vision — PlantDoc (Independent Test Set)

| Property | Value |
|---|---|
| **Purpose** | In-the-wild / field-condition evaluation (measures lab-to-field domain gap) |
| **Source** | [GitHub — PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) |
| **Alt Source** | [Kaggle — PlantDoc](https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset) |
| **Format** | JPEG/PNG images in class-labeled subdirectories |
| **Total Size** | ~300 MB (~2,598 images, 27 classes across 13 plant species) |
| **Classes Used** | Rice and Wheat classes only (filtered during preprocessing) |
| **License** | CC BY-SA 4.0 |
| **Citation** | Singh, D., et al. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* ACM CoDS-COMAD. |

### 3. NLP/RAG — Kisan Call Centre (KCC) Q&A Logs

| Property | Value |
|---|---|
| **Purpose** | Agronomic knowledge base for RAG retrieval |
| **Source** | [data.gov.in — KCC Dataset](https://data.gov.in) (search "Kisan Call Centre") |
| **Alt Source** | [Kaggle — Farmers Call Query (KCC) Data](https://www.kaggle.com/datasets) (search "KCC") |
| **Alt Source** | [AIKosh / IndiaAI](https://www.indiaai.gov.in/) — "Kisan Call Centre Transcripts" |
| **Format** | CSV |
| **Total Size** | ~500 MB+ (millions of records across all states) |
| **Filtering Applied** | State = "Uttar Pradesh", Crops = Rice/Wheat, Category = agronomic only, Date ≥ 2020 |
| **License** | Government Open Data License — India (OGL-India) |
| **Privacy Notes** | Contains farmer names/phone numbers in some versions — must be stripped during preprocessing |

### 4. Yield Prediction — District-Level Crop Data

| Property | Value |
|---|---|
| **Purpose** | Train classical ML model for district-level yield estimation (Should-Have) |
| **Source** | [UPAg — Unified Portal for Agricultural Statistics](https://upag.gov.in/) |
| **Alt Source** | [DES — Area, Production, Yield Reports](https://aps.dac.gov.in/APY/Public_Report1.aspx) |
| **Alt Source** | [ICRISAT District-Level Database](http://data.icrisat.org/dld/src/crops.html) |
| **Alt Source** | [data.gov.in](https://data.gov.in) — search "crop production statistics" |
| **Format** | CSV / Excel |
| **Features** | District, Year, Season, Crop, Area (ha), Production (tonnes), Yield (tonnes/ha), Rainfall (mm), Fertilizer consumption |
| **Filtering Applied** | State = Uttar Pradesh, Crops = Rice/Wheat |
| **License** | NDSAP (National Data Sharing and Accessibility Policy) / OGL-India |

### 5. RAG Corpus — UP Government Policy PDFs (TBD)

| Property | Value |
|---|---|
| **Purpose** | Ground LLM responses in official scheme/advisory documents |
| **Suggested Sources** | UP Agriculture Dept (upagripardarshi.gov.in), ICAR-IIRR, ICAR-IIWBR, KVK advisories, PM-KISAN UP circulars |
| **Status** | ⚠️ **Not yet sourced** — research in progress |
| **Target** | 20–100 PDFs covering disease advisories, treatment guidelines, and government schemes |
| **License** | Government publications (OGL-India) |

---

## Quick Setup

### Option A: Automated Download Script

```bash
# From the project root:
python scripts/download_data.py --all
```

See [`scripts/download_data.py`](../scripts/download_data.py) for options:
- `--plantvillage` — Download PlantVillage only
- `--plantdoc` — Download PlantDoc only
- `--kcc` — Download KCC dataset only
- `--yield` — Download yield dataset only
- `--all` — Download everything

> **Note:** PlantVillage and KCC downloads from Kaggle require a valid
> `~/.kaggle/kaggle.json` API token. See
> [Kaggle API docs](https://www.kaggle.com/docs/api) for setup.

### Option B: Manual Download

1. Download datasets from the links in the inventory table above
2. Extract them into the corresponding `data/raw/<dataset>/` directories
3. Run the preprocessing notebooks to generate `data/processed/` and `data/final/`

### Option C: Team Google Drive (Fallback)

If automated download fails, contact the team for the shared Google Drive link
containing pre-downloaded raw datasets.

> **Google Drive Link:** *TBD — will be added once datasets are uploaded*

---

## Data Flow

```
raw/ ──[EDA notebooks]──> processed/ ──[Preprocessing notebooks]──> final/
```

| Stage | Description | Notebooks |
|---|---|---|
| `raw/` | Original downloads, untouched | — |
| `processed/` | Filtered to Rice/Wheat, cleaned, deduplicated | `01_vision_eda.ipynb`, `03_kcc_rag_eda.ipynb`, `05_yield_eda.ipynb` |
| `final/` | Training-ready splits (Train/Val/Test) | `02_vision_preprocessing.ipynb`, `04_kcc_preprocessing.ipynb`, `06_yield_preprocessing.ipynb` |
