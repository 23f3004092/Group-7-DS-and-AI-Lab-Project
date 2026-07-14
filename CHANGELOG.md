# Changelog — AgriAssist: Farmer Query Assistant
**Group 7 | DS and AI Lab | IIT Madras**

All notable changes to this project will be documented here.

Format: [Milestone/Version] — Date | What changed and why.

---

## [Repository Governance & Realignment] — 2026-07-15

**Document:** Repository Architecture & Governance Alignment  
**Author:** Lokesh  
**Purpose:** Records structural realignment of documentation, operational manuals, and grading logs across the repository to strictly satisfy academic software engineering standards and instructor feedback (`docs/instructor_feedback/on_repo_structure.md`).

| # | Area / File | Key Technical Changes & Rationale |
|---|-------------|-----------------------------------|
| 1 | **Internal Path Standardisation**<br>(`Milestone_2_Report.md`, etc.) | • **Zero Broken Links:** Updated all relative image and report links across `Milestone_2_Report.md`, `Milestone_2_Report_Updated.md`, and internal recommendation documents to match the canonical `outputs/reports/`, `outputs/figures/`, and `docs/internal/` hierarchy. |
| 2 | **Manuals Roadmap Specification**<br>(`docs/manuals/README.md`) | • **Directory Instantiation:** Created `docs/manuals/README.md` with strict academic specifications outlining where M2 data engineering manuals reside (`data/README.md`) and defining the technical roadmap for M3 (`model_training_manual.md`, `evaluation_manual.md`) and M4 (`user_manual.md`, `deployment_manual.md`). |
| 3 | **Academic Grading Log Realignment**<br>(`docs/TEAM_CONTRIBUTIONS.md`) | • **Consolidated Work Logs:** Migrated and merged the complete individual work logs for both Milestone 1 and Milestone 2 into `docs/TEAM_CONTRIBUTIONS.md`, establishing a single, comprehensive academic assessment log across all five teammates. |
| 4 | **Developer Guide Transformation**<br>(`CONTRIBUTING.md`) | • **Root Developer Guide:** Re-architected root `CONTRIBUTING.md` into an authoritative onboarding specification covering repository hierarchy, Python virtual environment setup, PII redaction standards (`remove_pii()`), `.gitignore` compliance, and Jupyter notebook output sanitization (`nbstripout`). |
| 5 | **Data Hierarchy & Tracking Realignment**<br>(`data/`, `.gitignore`) | • **Large Dataset Untracking (`git rm --cached`):** Untracked 6,159 previously committed raw and preprocessed data files (`data/processed/`, `data/final/`, `data/Vision_dataset/`, `data/raw/`) from Git's index without deleting local d
isk assets.<br>• **Strict `.gitignore` Enforcement:** Expanded `.gitignore` rules to exclude all non-sample datasets and preprocessed outputs while explicitly tracking only `data/sample/` and `data/README.md`.<br>• **Sample Directory Instantiation:** Created `data/sample/README.md` establishing the instructor-mandated structure (`docs/instructor_feedback/on_repo_structure.md`) for storing lightweight verification subsets (`vision/`, `kcc/`, `yield/`) for testing. |

---

## [Milestone 2] — 2026-07-09

**Document:** Milestone 2 – Comprehensive Exploratory Data Analysis (EDA) & Data Preprocessing  
**Version:** Final Submission (Post Integration & Team Sign-Off)  
**Purpose:** Documents the complete data engineering, quality auditing, preprocessing, integration, group-aware splitting, and deliverable readiness across all four system pipelines (Vision, RAG PDF Corpus, Kisan Call Center Advisory Dataset, and Crop Yield Prediction).

| # | Subsystem & Lead(s) | Key Deliverables & Technical Changes Made |
|---|---------------------|-------------------------------------------|
| 1 | **Vision Subsystem**<br>*(Mahesh & Lokesh)* | • **Comprehensive EDA across 3 Datasets:** Performed deep quality auditing on Wheat (`14,154` images), Rice Set 1 (`5,932` images), and Rice Set 2 (`120` images).<br>• **Wheat Preprocessing & Canonicalization:** Collapsed 45 noisy/raw labels into 15 canonical classes and audited image sharpness/brightness profiles.<br>• **Rice Burst-Capture Deduplication:** Applied MD5 + pHash Hamming distance ≤ 6 bits union-find clustering to thin Set 1 from `5,932 -> 2,066` sharpest unique frames.<br>• **Aspect-Preserving Letterboxing:** Standardized images to 256×256 RGB padded letterboxes (`.convert("RGB")`), eliminating background-dimension shortcuts on Tungro and preserving ~3.43:1 panoramic aspect ratios on Set 2.<br>• **Unified Integration & Group-Aware Split (Notebook D):** Concatenated Wheat (`10,673`), Rice Set 1 (`2,066`), and Rice Set 2 (`120`) into a **20-Class Unified Dataset (`12,859` images)**. Performed centralized group-aware stratified splitting (80% Train / 10% Val / 10% Test) with verified **zero cross-split leakage (`0 groups spanning splits`)** and an 8-image evaluation floor per class (`final/` directory, 273 MB).<br>• **Training Pipeline Design (Notebook E):** Designed live data loaders with random/center crops to 224², ImageNet normalization, rare-class augmentation, hybrid `WeightedRandomSampler`, and Tungro Grad-CAM field-robustness checks. |
| 2 | **RAG PDF Corpus**<br>*(Harliv)* | • **Authoritative Corpus Collection:** Curated 187 government agricultural documents across 4 folders (`PPQS Advisories`, `UP ACP`, `Government Schemes`, `Other Docs`).<br>• **Extraction & Cleaning:** Extracted native text via `pdfplumber` with 300 DPI OCR fallback (`pytesseract`); filtered 14 unreadable and 33 near-duplicate documents to yield a clean **170-document corpus** (`pdf_inventory_clean.csv`).<br>• **Semantic Chunking:** Applied sentence-aware chunking (~512-token target, ~50-token overlap) with hard-split fallbacks, producing **1,451 retrieval-ready chunks** (`pdf_chunks.csv` / `pdf_chunks.jsonl`). |
| 3 | **KCC Advisory Dataset**<br>*(Aneeqa)* | • **3.12M Record Aggregation:** Combined 6 annual Uttar Pradesh Kisan Call Center datasets (2020–2025) into `kcc_combined_2020_2025.csv` (~2.07 GB).<br>• **Scope Validation:** Profiled 318 crops and confirmed Rice (`15.54%`) and Wheat (`16.40%`) together account for **31.95% (~997,806 queries)** of farmer inquiries.<br>• **Multilingual RAG Architecture Signal:** Discovered queries are **~99.98% English/Hinglish** while answers are **~98.80% Hindi (Devanagari script)**, establishing `MuRIL` cross-lingual embedding alignment as mandatory.<br>• **Text Length & Deduplication:** Confirmed **98.9% of records fit within 512 characters**, and audited high repetition (`68.72% duplicate queries`, `26.15% duplicate Q&A pairs`) requiring pre-embedding deduplication. |
| 4 | **Yield Prediction Subsystem**<br>*(Tanmay & Lokesh)* | • **Primary Multi-Crop Pipeline (Tanmay):** Unified disparate state/district sources into a national dataset (**440,962 records** across 124 crops, 35 States/UTs, 1997–2024). Standardized coconut units from raw pieces to metric tonnes, resolved zero-reporting anomalies, and applied non-parametric **Random Forest Imputation (`MissForest`)** (`production_unified_imputed.csv`).<br>• **Complementary UP Domain Pipeline (Lokesh):** Harmonized administrative district bifurcations across 75 UP districts (**3,996 records**), enriched district yield histories with IMD weather covariates and ICRISAT NPK input data, and partitioned data chronologically (`train/val/test_yield.csv`) to prevent temporal autocorrelation leakage across agricultural cycles. |
| 5 | **Report & Team Sign-Off**<br>*(All Teammates)* | • **Formal Team Sign-Off:** Complete review and sign-off documented for all five team members (`Mahesh`, `Harliv`, `Lokesh`, `Aneeqa`, `Tanmay`) with explicit responsibilities and approval dates (`2026-07-09`). |

---

## [Milestone 1] — 2025-07-02
# Change Log

**Document:** Milestone 1 – Problem Definition & Literature Review  
**Version:** Revision 2 (Post Review Meeting)  
**Purpose:** This document records the revisions made following the review meeting and confirms that the reviewers' recommendations have been incorporated into the revised submission.

| # | Reviewer Recommendation | Changes Made |
|---|-------------------------|--------------|
| 1 | Reduce the length of the main report by moving detailed scenarios to an appendix. Keep the report simple in language and presentation. | The main report was significantly condensed. All illustrative system scenarios were moved to **Appendix: System Scenarios**, allowing the core report to focus on the proposal, methodology and evaluation. |
| 2 | Use a more academic writing style and reduce marketing-style language such as "The Three Trust Failures", "attacks all three head-on", "goldmine", and "strongest contribution". | The report was rewritten using a formal academic tone. Promotional headings and persuasive language were removed and replaced with objective technical descriptions throughout the document. |
| 3 | Reduce the number of badges, emojis and callout boxes to improve professionalism. | Decorative badges, emojis, coloured callout boxes and excessive visual styling were removed. The revised report adopts a cleaner academic format. |
| 4 | Project scope is still too ambitious; clearly prioritise Must Have, Should Have, and Stretch Goals. | The project scope was restructured into **Must-Have**, **Should-Have**, **Stretch Goals**, and **Out-of-Scope** sections with accompanying justifications to clearly communicate implementation priorities. |
| 5 | Explain how your approach improves upon each major work reviewed, not just their strengths and weaknesses. | The literature review and comparative analysis were revised to explicitly explain how the proposed architecture addresses the limitations of prior systems and extends existing research. |
| 6 | Define target values for evaluation metrics instead of only listing the metrics. | Quantitative target values were introduced for the primary evaluation metrics (e.g., Macro F1, Recall@5, RAGAS Faithfulness, False Discovery Rate, RMSE, R²). |
| 7 | Justify why ResNet, EfficientNet and ViT were shortlisted as candidate models. | The candidate model discussion was revised to explain the architectural rationale behind model selection. The revised proposal narrows the shortlist to lightweight architectures that align with project constraints and deployment feasibility. |
| 8 | Describe how the confidence threshold (τ) for Abstention will be selected and validated. | A dedicated calibration section now explains the use of **temperature scaling**, validation on a held-out dataset, and optimisation of the abstention threshold based on performance metrics. |
| 9 | Define the relevance-score thresholds for Tier 1, Tier 2 and Tier 3 responses. | Explicit similarity thresholds were introduced for the three RAG response tiers, together with the behaviour associated with each tier. |
| 10 | Describe how the RAG knowledge base will handle duplicate, conflicting and outdated advisories. | The RAG implementation was revised to include knowledge-base curation strategies, including filtering outdated KCC entries, prioritising official government documents, and limiting the corpus to verified regional sources. |
| 11 | Revisit dataset scope not only for training and fine tuning but also RAG implementation. The quantity of data required is not clear and may prove to be overwhelming. | Dataset scope was substantially narrowed. The project now focuses on **Uttar Pradesh**, **Rice**, and **Wheat**, with a clearly defined RAG corpus consisting of filtered KCC data and a manageable number of regional policy documents. |
| 12 | Add inline citations for all important quantitative claims (such as model performance values). | Quantitative statements throughout the report were revised to include appropriate inline citations to the supporting literature. |
| 13 | Specify the baseline models against which improvements will be measured. | Baseline datasets, benchmark methods and comparison targets were clarified within the evaluation methodology and literature review to support quantitative performance comparison. |
| 14 | Include a Risk Assessment section with possible risks and mitigation strategies. | A dedicated **Risk Assessment & Mitigation** section was added, identifying major technical risks together with corresponding mitigation strategies. |
| 15 | Include a brief section on ethical considerations. | A new **Ethical Considerations** section was introduced covering transparency of AI-generated advice and safeguards against unsafe agronomic recommendations. |
| 16 | Add a Limitations section describing known constraints of the proposed system. | A dedicated **System Limitations** section was added to explicitly describe the project's intended scope, technical constraints and deployment limitations. |
| 17 | Include computational requirements such as hardware, training time and deployment feasibility. | A new **Computational Requirements** section was added detailing expected hardware requirements, GPU memory constraints, approximate training time and deployment feasibility using consumer-grade hardware. |

---

## Summary

All recommendations discussed during the review meeting have been addressed in the revised report. The updated submission presents a more focused project scope, adopts a formal academic writing style, improves evaluation planning with measurable targets, clarifies architectural decisions, introduces additional sections requested during the review (Risk Assessment, Ethical Considerations, Limitations, and Computational Requirements), and improves the overall professionalism and feasibility of the proposal.