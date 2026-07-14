# Milestone 1 — Team Contribution Log

## Working Model

The team worked in a **fully collaborative, non-siloed manner** for Milestone 1. No member was locked to a fixed sub-task; instead, everyone was free to research any aspect, contribute to any component, and review each other's work. Research documentation was shared openly across the team so that findings compounded rather than duplicated. The entries below record each member's primary areas of contribution within that shared effort.

---

## Mahesh — Contributions
- Helped drive the team from a broad "farmer assistant" concept toward a **sharp, research-worthy problem statement** — reframing a generic agri-chatbot into a system defined by three *measurable* trust failures: field-robustness of disease detection, faithfulness/hallucination of advice, and confidence calibration/abstention.
- Contributed to defining the **project scope and boundaries**.

- **Designed and built the system architecture diagram on Excalidraw** (the updated multimodal routing pipeline), finalizing it by incorporating suggestions and inputs from the team.
- Shaped the core design decisions through detailed analysis and probing questions, including:
  - Why intent understanding can be LLM-native vs. a separately trained classifier, and where a fine-tuned model is genuinely justified.
  - The **VLM-aware routing** for image inputs (route leaves to the specialized detector; let the VLM handle non-leaf images) rather than forcing every image into a disease class.
  - The **context-elicitation / follow-up question** flow for personalizing responses (location, crop stage, farming method).
  - The **tiered response strategy** (grounded RAG → transparent LLM fallback → out-of-scope redirect) and how RAG retrieval actually works mechanically (embedding → similarity search → grounded generation).
  - Identifying **legitimate, non-forced trainable components** — settling on the field-robust disease classifier and the domain-adapted embedding model as the two core trainable models.

- Explored and compiled the **candidate datasets**.

- Reviewed current solutions, tools, and academic work relevant to the problem.
- Assembled key references and industry context (WEF deep-tech reports, FarmerChat/GAIA, government initiatives, Syngenta, Plantix, Intello Labs, Wadhwani AI) into the research documentation.

- Authored the primary research document (*Farm_assistant_research_doc_Mahesh*), capturing the problem framing, interaction scenarios, design doubts and their resolutions, dataset notes, and literature review — and **shared it with the team** as a common reference.

- **Prepared the final Milestone-1 report**, consolidating inputs and research from all team members into a single structured deliverable covering the problem statement, scope, stakeholders, objectives, architecture, scenarios, literature review, comparative analysis, datasets, and evaluation framework.

---

## Harliv — Contributions
- Contributed to defining and refining the initial problem statement and project direction.

- Assisted in preparing and designing the milestone presentation slides.

- Reviewed the project documentation for technical accuracy, clarity, and completeness.

- Identified architectural inconsistencies and potential design issues within the proposed system.

- Suggested improvements and solutions to strengthen the system architecture and overall project design.

- Participated in discussions to refine the problem statement, objectives, and technical approach.
---

## Lokesh — Contributions
- Participated in team discussions to narrow the broad concept into a focused problem statement and technical approach.
- Authored the comprehensive technical summary (Detailed_Summary_by_Lokesh), synthesizing the full solution into a structured reference — ingestion/intent layer, computer-vision lab-to-field challenges and interventions, context-aware retrieval and sensor integration, the generative layer (fluency vs. faithfulness, RAGAS), multi-turn dialogue and voice interfaces, and baseline comparisons.
- Researched specialized Indian-language models (MuRIL, IndicBERT) and evaluation methodology (RAGAS metrics, faithfulness).
- Explored architectural interventions for the disease-detection domain gap (segmentation-first, domain adaptation, YOLOv8, vision-language models).
- Reviewed and contributed to research documentation and the final Milestone-1 report for technical accuracy and completeness.
---

## Aneeqa — Contributions
- Participated in team discussions to narrow the broad "farmer assistant" concept into a focused problem statement.

- Contributed to defining project boundaries, identifying what is in scope versus explicitly out of scope.

- Researched existing solutions including DigiGreen, KisanSarathi.

- Explored government and industry initiatives including India's Digital Agriculture Mission, Kisan e-Mitra and Bharat-VISTAAR.

- Reviewed and contributed to research documentation and final Milestone-1 report, ensuring technical accuracy and completeness
---

## Tanmay — Contributions
 - Participated in team discussions regarding architecture of the product focusing on system design and reliability of API calls, mitigate latency bottlenecks before implementation begins
 - Proposed and structured the architectural shift from a pure Python setup to a split-responsibility model, introducing a highly concurrent Java (Spring Boot) core for I/O routing and keeping Python strictly sandboxed for ML execution.
 - Directed the integration of local language components (such as speech streaming and direct multilingual vector mapping via MuRIL) to make the system resilient for rural Indian demographics.

---

## Shared / Collective Activities
- Joint discussion and refinement of the problem statement and scope.
- Cross-review of each other's research documents.
- Collective narrowing of the idea from a broad concept to a focused, measurable, milestone-aligned project.
- Shared identification of gaps and opportunities across current solutions.

---

# Milestone 2 — Team Contribution Log

## Working Model

For Milestone 2, the team operated with structured ownership across the core subsystems of the Agricultural Decision Support System—**Computer Vision (Disease Detection)**, **RAG / NLP Knowledge Base (PDF Corpus & Kisan Call Centre Q&A)**, and **Crop Yield Prediction**—while maintaining continuous cross-functional collaboration and rigorous peer review.

Commit history analysis over the Milestone 2 period (verified via both commit logs and detailed commit diff inspection where non-descriptive messages like *"Add files via upload"* or *"yield_asset"* were used) confirms extensive individual ownership combined with collective synthesis:

---

## Mahesh — Contributions
- **Vision Subsystem Preprocessing & Pipeline Engineering**:
  - Developed end-to-end preprocessing pipelines and notebooks for the **Wheat Disease Dataset**, **Rice Leaf Disease Set 1**, and **Rice Leaf Disease Set 2** (`preprocessing-wheat-dataset.ipynb`, `preprocessing-rice-dataset-set-1-b.ipynb`, `preprocessing-rice-dataset-set-2-c.ipynb`).
  - Authored comprehensive EDA and technical documentation for the vision datasets (`outputs/reports/wheat_disease_dataset_EDA.md`, `outputs/reports/Rice_leaf_disease_dataset_set2_documentation.md`, and `outputs/reports/wheat_preprocessing_documentation.md`), detailing class distributions, resolution variations, and data augmentation strategies.
- **KCC Data Ingestion Tooling**:
  - Built the automated data downloader script (`scripts/download_data.py`) supporting scalable streaming ingestion of Kisan Call Centre (KCC) records in JSONL and Parquet formats from data.gov.in.
- **Visual Analytics & Dashboard Assets**:
  - Created key dataset visualizations including sample grids, class distribution charts, and summary dashboards for Rice Set 2 (`outputs/figures/rice_set2_class_distribution.png`, `outputs/figures/rice_set2_dashboard.png`, `outputs/figures/rice_set2_sample_grid.png`).

---

## Harliv — Contributions
- **RAG / NLP PDF Corpus Architecture & EDA**:
  - Built and executed the complete PDF corpus EDA pipeline (`05499e3`) for the RAG system, extracting, cleaning, and cataloging text across agricultural schemes, pest/disease advisories, and Uttar Pradesh District Agricultural Contingency Plans (`up_acp`).
  - Authored the detailed RAG PDF corpus technical report (`outputs/reports/rag_pdf_report.md`) and curated the cleaned inventory (`pdf_inventory_clean.csv`).
- **Semantic Chunking & Embedding Preparation**:
  - Implemented semantic chunking analysis (`b83b1c6`) and preprocessed domain documents for downstream multilingual vector representation (MuRIL/ChromaDB).
- **Report Restructuring & Multi-Modal Visual Integration**:
  - Structured the initial Milestone 2 report framework (`e36595b`, `bebbb33`) and integrated core visual EDA outputs across both Vision (`outputs/figures/rice_dist_and_image_resolution.png`, `outputs/figures/wheat_dist_and_image_resolution.png`) and KCC datasets (`outputs/figures/docspersource_pagewordcount.png`, query/answer length distributions, monthly trends).

---

## Lokesh — Contributions
- **Milestone 2 Architecture & Technical Sprint Planning**:
  - Authored the comprehensive [Milestone 2 Implementation Plan](file:///d:/Group-7-DS-and-AI-Lab-Project/docs/internal/Milestone_2_Implementation_Plan.md) (`docs/internal/Milestone_2_Implementation_Plan.md`, 262 lines) at sprint initiation (`commit 61ab4b3`), architecting the data pipelines across all three subsystems (Vision, KCC/RAG, and Crop Yield).
  - Established the 3-day sprint work breakdown across all five team members, defined the repository storage hierarchy (`data/raw/`, `data/processed/`, `data/final/`), resolved core design decisions upfront, and established the verification checklist for data leakage prevention.
- **Crop Yield Subsystem Preparation, EDA & Preprocessing**:
  - Conducted comprehensive domain research on Uttar Pradesh crop yield data (`outputs/reports/UP Crop Yield Data Research.md`), built the yield EDA and preprocessing notebooks (`notebooks/05_yield_eda.ipynb`, `notebooks/06_yield_preprocessing.ipynb`), and generated structured train/validation/test datasets (`data/final/yield/`).
  - Developed Python automation scripts for yield data fetching, exploratory analysis, and preprocessing (`scripts/fetch_prepare_yield_data.py`, `scripts/run_yield_eda_analysis.py`, `scripts/run_yield_preprocessing.py`).
- **Initial Vision EDA & Pipeline Consolidation**:
  - Developed the initial Rice & Wheat exploratory data analysis notebook (`notebooks/01_vision_eda.ipynb`) and configured project dependencies (`requirements.txt`, `data/README.md`).
  - Consolidated and structured the Vision training pipeline design and merge specifications (`outputs/reports/notebookD_merge_split_documentation.md`, `outputs/reports/notebook_training_pipeline_design.md`).
- **Milestone 2 Report Synthesis & Sign-Off Management**:
  - Consolidated findings from all subsystems into the authoritative `docs/reports/Milestone_2_Report.md`, standardizing technical sections, tables, and team sign-off verification.

---

## Aneeqa — Contributions
- **Kisan Call Centre (KCC) Knowledge Base EDA**:
  - Developed the comprehensive KCC Exploratory Data Analysis notebook (`notebooks/03_kcc_rag_eda.ipynb`, 2,366 lines), analyzing query distributions, farmer query patterns, crop/topic clustering, and seasonal frequency across agricultural advisory records.
- **RAG Knowledge Preparation & Documentation**:
  - Authored structured technical documentation for KCC data analysis (`outputs/reports/KCC Data EDA.md`), establishing the foundation for multilingual query-answer retrieval and deduplication.

---

## Tanmay — Contributions
- **Unified Multi-Crop Yield EDA & Advanced Preprocessing**:
  - Developed the comprehensive multi-crop yield exploratory data analysis and preprocessing notebook (`notebooks/07_Yield_EDA+ preprocessing.ipynb`, 1,544 lines), implementing missing value treatment (MissForest) and feature engineering across historical yield records.
- **Yield Visual Analytics & Correlation Studies**:
  - Generated and integrated key statistical charts and analytical assets (`outputs/figures/Avg_yield.png`, `outputs/figures/correlation_matrix.png`, `outputs/figures/season_summary.png`, `outputs/figures/top_crops_production.png`, `outputs/figures/yearly_trends.png`).
- **Yield Subsystem Documentation**:
  - Authored detailed narrative summaries and technical reports for the crop yield prediction subsystem (`outputs/reports/yield_report.md`, `notebooks/yield_notebook_summary.md`).

---

## Shared / Collective Activities (Milestone 2)
- **Subsystem Integration**: Collaborative harmonization of datasets across Computer Vision, RAG / NLP Knowledge Base, and Crop Yield Prediction into a unified architecture.
- **Quality Assurance & Data Leakage Prevention**: Cross-checking dataset splits, pHash deduplication strategies across image sets, and schema consistency across tabular yield and KCC data.
- **Report Review & Sign-Off**: Joint technical review and formal sign-off of the complete Milestone 2 submission.
---
