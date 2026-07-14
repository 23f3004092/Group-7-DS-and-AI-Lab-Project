# Sample Datasets & Pipeline Verification (`data/sample/`)

In accordance with our repository governance (`docs/instructor_feedback/on_repo_structure.md`), large raw and preprocessed datasets (`data/raw/`, `data/processed/`, `data/final/`) are excluded from Git version control to prevent repository bloat and ensure PII compliance.

This directory (`data/sample/`) is reserved for **lightweight, representative data samples** to allow instructors, reviewers, and automated testing suites (`tests/`) to verify pipeline functionality without downloading multi-gigabyte files.

---

## 1. Directory Sub-Structure

When populating sample data for testing, maintain the following hierarchy:

```text
data/sample/
├── vision/
│   ├── rice_sample/        # 2–3 sample images per class (Bacterial Leaf Blight, Brown Spot, Leaf Smut)
│   └── wheat_sample/       # 2–3 sample images per class (Healthy, Yellow Rust, Brown Rust, etc.)
├── kcc/
│   └── kcc_sample.csv      # ~100 anonymized sample farmer queries and Devanagari responses
└── yield/
    └── yield_sample.csv    # ~100 rows of historical district-level yield covariate records
```

---

## 2. Full Dataset Acquisition

To execute complete training or exploratory data analysis against full datasets (`data/raw/`), run our automated ingestion script:

```bash
python scripts/download_data.py --dataset all
```

For complete dataset documentation, schema definitions, and citation records, consult **[`data/README.md`](../README.md)**.
