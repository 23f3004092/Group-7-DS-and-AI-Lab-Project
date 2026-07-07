# Milestone 2 — Dataset Preparation & EDA Implementation Plan

## Goal

Produce **training-ready datasets** for all three data streams of the Decoupled Agentic Multimodal Crop Advisory System, with comprehensive EDA, preprocessing pipelines, and a final report following the instructor's 13-section structure. **Deadline: 3 days (July 10, 2026).**

## Resolved Design Decisions

| Decision | Resolution |
|---|---|
| Vision datasets | PlantVillage (train/val) + PlantDoc (independent test); Rice & Wheat classes only |
| Vision split | 80/20 stratified Train/Val on PlantVillage; PlantDoc = held-out test set |
| Augmentation | Phased: baseline → standard augmentations → field-noise simulation |
| KCC/RAG data | Download full KCC from data.gov.in; filter UP + Rice/Wheat + agronomic |
| UP Policy PDFs | **Not yet sourced** — research task required |
| RAG scope in M2 | Clean, chunk, and describe text data; defer MuRIL embedding to M3; document chunk structure & metadata schema |
| Yield data | Include with full EDA + preprocessing, but lowest priority |
| Tooling | Jupyter notebooks (one per stream per phase) |
| Data storage | `data/raw/`, `data/processed/`, `data/final/`; raw in .gitignore |
| Report format | Single Markdown (13-section); may add notebook appendices (pending instructor) |
| Image resolution | 224×224 default; document as tunable hyperparameter for M3 |
| Libraries | pandas, matplotlib, seaborn, OpenCV, PIL, scikit-learn, PyTorch |

---

## Open Questions

> [!IMPORTANT]
> **UP Government PDF Sources:** No specific sources have been identified yet. Suggested starting points:
> - [UP Agriculture Department](https://upagripardarshi.gov.in/)
> - ICAR-IIRR (Rice) and ICAR-IIWBR (Wheat) advisory circulars
> - PM-KISAN UP district-level circulars
> - KVK advisories from UP districts
>
> **Team member(s) assigned to this research should aim to identify at least 20-30 PDFs by Day 2** to include in the report's "Dataset Identification" section, even if full processing is deferred.

> [!NOTE]
> **Report format flexibility:** The default plan produces a single Markdown file. If the instructor confirms they want notebook appendices, the notebooks are already structured to be export-ready — no rework needed, just add an appendix section with links.

---

## Proposed Changes

### Phase 1: Project Setup & Data Acquisition (Day 1 — Morning)

#### [NEW] scripts/download_data.py
- Automated script to download PlantVillage from Kaggle, PlantDoc from GitHub/Kaggle, KCC dataset from data.gov.in, and yield data from data.gov.in
- Includes checksum validation and directory setup
- Documents fallback Google Drive links

#### [MODIFY] .gitignore
- Add `data/raw/` to .gitignore (large datasets)
- Keep `data/processed/` and `data/final/` tracked (smaller, derived files)

#### [NEW] data/README.md
- Document all dataset sources, download links, expected sizes, and checksums
- Directory structure explanation
- Setup instructions for team members

#### [NEW] Data directory structure
```
data/
├── raw/                    # .gitignored — original downloads
│   ├── plantvillage/       # Full PlantVillage dataset
│   ├── plantdoc/           # Full PlantDoc dataset
│   ├── kcc/                # KCC CSV dump
│   ├── yield/              # data.gov.in yield CSVs
│   └── pdfs/               # UP government PDFs (when sourced)
├── processed/              # Cleaned, filtered data
│   ├── vision/             # Filtered Rice/Wheat images
│   ├── kcc/                # Filtered & cleaned KCC text
│   └── yield/              # Cleaned yield tabular data
└── final/                  # Training-ready splits
    ├── vision/
    │   ├── train/          # 80% PlantVillage (stratified)
    │   ├── val/            # 20% PlantVillage (stratified)
    │   └── test/           # PlantDoc (independent)
    ├── kcc/                # Chunked, ready-for-embedding text
    └── yield/              # Train/val/test CSVs
```

---

### Phase 2: Vision Data — EDA & Preprocessing (Day 1 Afternoon – Day 2 Morning)

#### [NEW] notebooks/01_vision_eda.ipynb

**EDA Sections:**
1. **Dataset overview** — Total images, classes available, filter to Rice & Wheat diseases
2. **Class distribution** — Bar charts showing sample counts per disease class (PlantVillage vs PlantDoc)
3. **Class mapping** — Align PlantVillage and PlantDoc class labels (they may differ in naming)
4. **Image statistics** — Resolution distribution (histograms of width/height), aspect ratios, file sizes
5. **Sample visualization** — Grid of sample images per class (both PlantVillage and PlantDoc side by side)
6. **Channel statistics** — Mean/std of RGB channels per class (needed for normalization)
7. **Duplicate detection** — Perceptual hashing to find near-duplicate images
8. **Quality analysis** — Blur detection (Laplacian variance), brightness distribution
9. **Domain gap visualization** — t-SNE or PCA of feature embeddings (using pretrained EfficientNet) showing PlantVillage vs PlantDoc clusters → this directly illustrates the lab-to-field gap from your Milestone 1

#### [NEW] notebooks/02_vision_preprocessing.ipynb

**Preprocessing Steps:**
1. **Class filtering** — Extract only Rice and Wheat disease classes from both datasets
2. **Duplicate removal** — Remove near-duplicates identified in EDA
3. **Quality filtering** — Remove severely blurry or corrupted images
4. **Resize** — Standardize to 224×224 (document original sizes for reference)
5. **Normalization** — Compute and document per-channel mean/std for the filtered training set
6. **Stratified split** — 80/20 Train/Val from PlantVillage using scikit-learn
7. **Augmentation pipeline definition** — Define (but don't mass-apply) the augmentation transforms using PyTorch:
   - Phase 1: Random rotation (±15°), horizontal flip, brightness/contrast jitter
   - Phase 2: Gaussian blur, color jitter (field-noise simulation)
   - Phase 3 (M3): CutMix/MixUp
8. **Augmentation preview** — Show before/after examples of augmented images
9. **Save final splits** — Write to `data/final/vision/{train,val,test}/` in class-labeled subdirectories
10. **Generate dataset manifest** — CSV with `filepath, class, split, original_dataset, original_resolution`

---

### Phase 3: KCC / RAG Data — EDA & Preprocessing (Day 2)

#### [NEW] notebooks/03_kcc_rag_eda.ipynb

**EDA Sections:**
1. **Dataset overview** — Total records, columns, data types
2. **State filtering preview** — Distribution of records by state (show UP proportion)
3. **Crop distribution** — Crop-wise record counts (focus on Rice/Wheat)
4. **Query category analysis** — Distribution of query types (pest, disease, scheme, market, etc.)
5. **Temporal analysis** — Records by year/season to identify outdated data
6. **Language analysis** — Detect Hindi/English/code-mixed entries; character set analysis
7. **Text length distribution** — Query and response length histograms
8. **Missing value analysis** — Nulls per column, patterns
9. **Duplicate analysis** — Exact and near-duplicate detection
10. **Sample records** — Display representative examples from each category
11. **Document structure analysis** — For future RAG: analyze typical Q&A pair lengths, identify natural chunk boundaries, propose metadata schema (district, crop, season, query_type, date)

#### [NEW] notebooks/04_kcc_preprocessing.ipynb

**Preprocessing Steps:**
1. **State filter** — Keep only `State == "Uttar Pradesh"` records
2. **Crop filter** — Keep only Rice and Wheat related queries
3. **Category filter** — Exclude financial/subsidy/market entries (per Milestone 1 risk mitigation)
4. **Temporal filter** — Remove records older than a relevance threshold (e.g., pre-2020)
5. **Missing value treatment** — Handle nulls (drop or impute based on EDA findings)
6. **Duplicate removal** — Remove exact and fuzzy duplicates
7. **Text cleaning** — Normalize whitespace, fix encoding issues, standardize transliteration
8. **Chunk preparation** — Split long Q&A pairs into chunks suitable for embedding (target: 256-512 tokens); document chunking strategy and overlap
9. **Metadata tagging** — Add structured metadata columns: `district`, `crop`, `season`, `query_type`
10. **Save processed data** — Write to `data/final/kcc/` as cleaned CSV + chunked JSONL (ready for MuRIL in M3)
11. **RAG readiness documentation** — Document: chunk size rationale, metadata schema, recommended embedding model (MuRIL), suggested similarity thresholds (Tier 1/2/3 from M1)

---

### Phase 4: Yield Data — EDA & Preprocessing (Day 2 Evening – Day 3 Morning, if time permits)

#### [NEW] notebooks/05_yield_eda.ipynb

**EDA Sections:**
1. **Dataset overview** — Records, features, districts covered, time range
2. **Feature descriptions** — Yield, rainfall, fertilizer usage, area, production
3. **Missing value analysis** — Heatmap of missing data by district × year
4. **Distribution analysis** — Histograms/boxplots for each feature
5. **Temporal trends** — Yield over time by district and crop
6. **Correlation analysis** — Heatmap of feature correlations
7. **Outlier detection** — IQR/Z-score analysis for yield and input features
8. **District coverage** — Map or bar chart of data completeness by UP district

#### [NEW] notebooks/06_yield_preprocessing.ipynb

**Preprocessing Steps:**
1. **Filter to UP + Rice/Wheat**
2. **Handle missing values** — Interpolation or district-mean imputation
3. **Outlier treatment** — Clip or remove based on EDA
4. **Feature engineering** — Yield per hectare, rainfall deviation from mean, fertilizer intensity
5. **Normalization/scaling** — StandardScaler or MinMaxScaler (document choice)
6. **Train/Val/Test split** — Temporal split (e.g., train: ≤2020, val: 2021-2022, test: 2023+) to prevent leakage
7. **Save final splits** — Write to `data/final/yield/`

---

### Phase 5: Report Writing (Day 3)

#### [NEW] docs/Milestone_2_Report.md

Following the instructor's 13-section structure exactly:

| Section | Content Source | Owner |
|---|---|---|
| 1. Introduction | Recap from M1 + M2 objectives | Report lead |
| 2. Dataset Identification | All three datasets: sources, links, licensing | Data acquisition team |
| 3. Dataset Description | Records, features, schema from EDA notebooks | EDA team |
| 4. Data Governance | Licensing (PlantVillage: CC0, KCC: OGL-India, data.gov.in: NDSAP), privacy, ethics | Report lead |
| 5. EDA | Key visualizations exported from notebooks | EDA team |
| 6. Data Preprocessing | Steps from preprocessing notebooks | Preprocessing team |
| 7. Dataset Integration | PlantVillage + PlantDoc alignment; KCC filtering | Preprocessing team |
| 8. Data Augmentation | Augmentation strategy + preview images | Vision team |
| 9. Dataset Splitting | Ratios, sample counts, stratification, leakage prevention | Preprocessing team |
| 10. Final Prepared Dataset | Summary stats of final splits | All |
| 11. Challenges Encountered | Issues found during EDA/preprocessing | All |
| 12. Deliverables Produced | List of notebooks, scripts, datasets | Report lead |
| 13. Summary & Next Steps | Readiness confirmation + M3 plan | Report lead |

---

## Suggested Team Work Breakdown (3-Day Sprint)

> [!IMPORTANT]
> This is a suggestion — adjust based on team members' strengths and availability.

### Day 1 (July 8)

| Team Member | Task | Deliverable |
|---|---|---|
| **Lokesh** | Project setup: directory structure, .gitignore, download script, data README | `scripts/download_data.py`, `data/README.md`, updated `.gitignore` |
| **Mahesh** | Download & begin Vision EDA (PlantVillage + PlantDoc) | `notebooks/01_vision_eda.ipynb` (in progress) |
| **Aneeqa** | Download KCC dataset from data.gov.in, begin KCC EDA | `notebooks/03_kcc_rag_eda.ipynb` (in progress) |
| **Tanmay** | Download yield data from data.gov.in, begin Yield EDA | `notebooks/05_yield_eda.ipynb` (in progress) |
| **Harliv** | Research UP government PDF sources; begin report skeleton (Sections 1-4) | PDF source list, `docs/Milestone_2_Report.md` (skeleton) |

### Day 2 (July 9)

| Team Member | Task | Deliverable |
|---|---|---|
| **Lokesh** | Vision preprocessing pipeline | `notebooks/02_vision_preprocessing.ipynb` |
| **Mahesh** | Complete Vision EDA + domain gap visualization | `notebooks/01_vision_eda.ipynb` (complete) |
| **Aneeqa** | KCC preprocessing pipeline | `notebooks/04_kcc_preprocessing.ipynb` |
| **Tanmay** | Yield EDA completion + yield preprocessing | `notebooks/05_yield_eda.ipynb`, `notebooks/06_yield_preprocessing.ipynb` |
| **Harliv** | Continue report (Sections 5-9), integrate EDA visuals | `docs/Milestone_2_Report.md` (sections 1-9) |

### Day 3 (July 10)

| Team Member | Task | Deliverable |
|---|---|---|
| **All** | Morning: Complete any remaining preprocessing, export final datasets | `data/final/` populated |
| **Harliv + Lokesh** | Afternoon: Complete report (Sections 10-13), integrate all visuals | `docs/Milestone_2_Report.md` (complete) |
| **Mahesh + Aneeqa + Tanmay** | Afternoon: Review report, cross-check numbers, sign-off | Final review |
| **All** | Evening: Final polish, team sign-off | Submission-ready |

---

## Verification Plan

### Automated Checks
- Run all notebooks end-to-end to ensure reproducibility
- Verify `data/final/` directory contains expected splits with correct sample counts
- Validate no data leakage between train/val/test (check for overlapping filenames in vision, overlapping records in tabular)
- Confirm class distribution is preserved after stratified splitting

### Manual Verification
- Each team member reviews at least one notebook they didn't author
- Cross-check report statistics against notebook outputs
- Verify all download links in `data/README.md` are functional
- Review augmented image samples for visual correctness
- Confirm report covers all 13 instructor sections

### Dataset Readiness Checklist
- [ ] Vision: Train/Val/Test directories populated with correctly labeled images
- [ ] Vision: Normalization constants (mean/std) documented
- [ ] Vision: Augmentation pipeline defined and previewed
- [ ] KCC: Filtered, cleaned, chunked text ready for embedding
- [ ] KCC: Metadata schema documented for vector DB indexing
- [ ] Yield: Clean tabular data with temporal train/val/test split
- [ ] All: Dataset manifests/data dictionaries produced
