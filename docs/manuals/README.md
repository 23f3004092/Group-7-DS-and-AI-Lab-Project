# System and User Manuals

## 1. Overview
This directory (`docs/manuals/`) serves as the repository for operational and end-user documentation for the **Agentic Farmer Query Assistant (AgriAssist)**, in strict alignment with [`docs/instructor_feedback/on_repo_structure.md`](../instructor_feedback/on_repo_structure.md).

---

## 2. Milestone 2 (Data Engineering Phase)
Current execution guidelines and preprocessing specifications reside alongside their primary scripts and data assets:
* **Ingestion & Schema Protocols:** [`data/README.md`](../../data/README.md)
* **Environment Configuration:** [`README.md`](../../README.md) and [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
* **Subsystem Preprocessing Guides:** [`outputs/reports/`](../../outputs/reports/)

---

## 3. Scheduled Deliverables (Milestones 3 and 4)

| Milestone | Document Name | Target Audience | Technical Scope |
|---|---|---|---|
| **Milestone 3** | `model_training_manual.md` | ML Engineers | Execution protocols for Vision (YOLO/ResNet), RAG (`MuRIL`), and Yield (`MissForest`/RF) training pipelines. |
| **Milestone 3** | `evaluation_manual.md` | QA / Reviewers | Procedures for quantitative benchmarking, RAGAS faithfulness evaluations, and Grad-CAM salience verification. |
| **Milestone 4** | `user_manual.md` | Extension Officers / Farmers | Operating instructions for submitting code-mixed queries, uploading diagnostic leaf images, and retrieving yield forecasts. |
| **Milestone 4** | `deployment_manual.md` | DevOps Engineers | Specifications for containerization (`Docker`), ChromaDB vector indexing, and API security/telemetry. |
