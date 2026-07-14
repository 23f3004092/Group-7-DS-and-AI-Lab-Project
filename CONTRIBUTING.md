# Contributing to AgriAssist

Thank you for your interest in contributing to **AgriAssist (Agentic Farmer Query Assistant)** — a field-robust, multimodal agricultural decision support system designed for smallholder farmers in rural India.

This guide outlines the technical, architectural, and operational standards for developing code, managing data, and collaborating within the repository in accordance with our academic governance guidelines (`docs/instructor_feedback/on_repo_structure.md`).

---

## 📌 Academic Work Logs & Assessment Reference
If you are looking for the chronological log of individual student contributions across milestones for academic evaluation and grading, please refer to:
👉 **[`docs/TEAM_CONTRIBUTIONS.md`](docs/TEAM_CONTRIBUTIONS.md)**

---

## 1. Repository Architecture & Storage Hierarchy

To maintain a clean, maintainable, and reproducible codebase, all additions must adhere to our strict directory separation:

| Directory | Purpose & Rules |
|---|---|
| **`src/`** | Reusable, production-grade Python/Java source code (`models/`, `retrieval/`, `inference/`, `evaluation/`, `utils/`). |
| **`notebooks/`** | Exploratory Data Analysis (EDA), initial experimentation, and visual exploration (`01_vision_eda.ipynb`, `03_kcc_rag_eda.ipynb`, etc.). *Must have cell outputs cleared before committing.* |
| **`data/`** | Local dataset partitions (`raw/`, `processed/`, `final/`). **Never commit large datasets or PII to Git.** See [`data/README.md`](data/README.md) for download protocols. |
| **`outputs/`** | Generated figures, charts, and technical preprocessing/EDA sub-reports (`outputs/figures/`, `outputs/reports/`). |
| **`docs/`** | Complete project history, authoritative milestone deliverables, architecture diagrams, and academic logs (`reports/`, `architecture/`, `manuals/`, `TEAM_CONTRIBUTIONS.md`). |
| **`scripts/`** | Executable standalone automation utilities (`download_data.py`, `run_yield_eda_analysis.py`). |

---

## 2. Environment Setup & Dependency Management

Before beginning development, set up your local Python environment:

```bash
# 1. Clone the repository
git clone <repository-url>
cd Group-7-DS-and-AI-Lab-Project

# 2. Create and activate a Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install project dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> [!IMPORTANT]
> If you introduce a new Python package required for your subsystem, you must pin its version and update `requirements.txt`:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 3. Data Governance & Privacy Standards

Our RAG knowledge base utilizes real farmer query logs from the **Kisan Call Centre (KCC)** and agricultural yield records. Strict data governance is mandatory:

1. **Zero Personally Identifiable Information (PII):** All farmer phone numbers, names, email addresses, and identification numbers must be redacted using our preprocessing pipelines (`remove_pii()`) before storage or processing.
2. **Git Ignore Compliance:** Large data files (`.csv`, `.parquet`, `.jsonl`, `.jpg`, `.png` image dumps) are strictly excluded via `.gitignore`. Do not override these rules using `git add -f`.
3. **Data Acquisition:** To populate your local `data/raw/` directory, use our automated ingestion tool:
   ```bash
   python scripts/download_data.py --dataset all
   ```
   For full instructions, consult [`data/README.md`](data/README.md).

---

## 4. Git Workflow & Branching Strategy

We follow a structured branching model to prevent merge conflicts and maintain stable core pipelines:

### Branch Naming Conventions
- **`feature/`** — New features, models, or data pipelines (e.g., `feature/yolo-vision-pipeline`, `feature/muril-embedding-service`).
- **`fix/`** — Bug fixes or preprocessing corrections (e.g., `fix/missforest-imputation-nan`).
- **`docs/`** — Documentation updates or milestone reports (e.g., `docs/milestone-3-draft`).
- **`test/`** — Unit test additions or evaluation verification (e.g., `test/ragas-faithfulness-suite`).

### Commit Message Standards
Commit messages should follow the **Conventional Commits** specification (`type(scope): description`):
```text
feat(vision): integrate pHash deduplication for Rice Set 2 images
fix(yield): resolve MissForest ordinal encoding edge case for coconut units
docs(reports): update Milestone 2 report link references to outputs hierarchy
refactor(rag): optimize sentence-boundary chunking to 512-character target
```

---

## 5. Code & Notebook Hygiene

### Python Source Code (`src/` & `scripts/`)
- Follow **PEP8** formatting guidelines.
- Provide comprehensive docstrings (`docstrings`) for all classes and functions detailing inputs, outputs, and edge-case behaviors.
- Use explicit type hinting where applicable (`def predict_yield(crop: str, area: float) -> float:`).

### Jupyter Notebooks (`notebooks/`)
- **Clear Outputs Before Committing:** Jupyter notebooks with large embedded output tables, base64 images, and interactive traces create massive Git diffs and repository bloat. Before committing any `.ipynb` file, run:
  ```bash
  # Clear outputs manually in your IDE, or via command line if nbconvert/nbstripout is installed:
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
  ```
- **Exporting Figures:** If a notebook generates key visual charts required for reports or presentations, save them programmatically to `outputs/figures/`:
  ```python
  import matplotlib.pyplot as plt
  plt.savefig("../../outputs/figures/my_chart_name.png", dpi=300, bbox_inches="tight")
  ```

---

## 6. Pull Request & Sign-Off Process

When submitting a pull request to `main`:
1. Ensure all new or modified pipelines have been tested against local sample splits (`train/val/test`).
2. Verify that zero data leakage occurs across temporal or spatial splits.
3. Request peer review from at least one subsystem lead (`Vision`, `RAG/NLP`, or `Yield`) before merging.
