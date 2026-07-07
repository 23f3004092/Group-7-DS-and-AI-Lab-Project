
# Milestone 2 — Dataset Identification & Preparation

## 1. Introduction

* **Brief recap of the project:** [Insert 2-3 sentences about the decoupled, agentic multimodal crop advisory system for UP.]
* **Objectives of Milestone 2:** [Outline the goals of data gathering, cleaning, and prep.]
* **Relationship between the dataset and project goals:** [Explain how datasets solve the vision gap, linguistic incompatibility, and hallucination issues.]

## 2. Dataset Identification

| Dataset Name | Source / Link | Status (Public/Private) | Purpose | Selection Rationale | Alternatives Considered |
| --- | --- | --- | --- | --- | --- |
| PlantVillage | [Link] | Public | [Purpose] | [Rationale] | [Alternatives] |
| PlantDoc | [Link] | Public | [Purpose] | [Rationale] | [Alternatives] |
| KCC Q&A Logs | [Link] | Public | [Purpose] | [Rationale] | [Alternatives] |
| Agri-Policy PDFs | [Link] | Public | [Purpose] | [Rationale] | [Alternatives] |
| Yield Data | [Link] | Public | [Purpose] | [Rationale] | [Alternatives] |

## 3. Dataset Description

* **PlantVillage:** [Records, features, target variable, format, schema]
* **PlantDoc:** [Records, features, target variable, format, schema]
* **KCC Q&A Logs:** [Records, features, target variable, format, schema, sample record]
* **Agri-Policy PDFs:** [Records, format, document hierarchy/schema]
* **UP District Yield Data:** [Records, features, target variable, format, schema]

## 4. Data Governance

* **Data Source & Licensing:** [Detail licenses for Kaggle, GitHub, and Govt data]
* **Privacy:** [Note on anonymizing KCC caller data, absence of PII]
* **Data Quality:** [Steps for handling broken images or corrupted PDF text]
* **Ethics & Bias:** [Address regional bias toward UP, language bias]
* **Reproducibility & Compliance:** [Link to download scripts and version control details]

## 5. Exploratory Data Analysis (EDA)

* **Summary statistics:** [Table or text summary of numerical data]
* **Feature distributions:** [Notes on image sizes, text lengths]
* **Class distribution:** [Healthy vs. Diseased ratios for Rice/Wheat]
* **Missing value & duplicate analysis:** [Findings in KCC logs or Yield data]
* **Visualizations:** [Placeholders for class distribution charts, sample images, text word clouds]

## 6. Data Preprocessing

* **Cleaning steps performed:** [Text lowercasing, artifact removal]
* **Missing value & duplicate treatment:** [How nulls were handled]
* **Standardization/Normalization:** [Image scaling to 224x224, text Unicode standardization]
* **Tokenization/Chunking:** [Chunking strategy for PDFs (e.g., recursive character split)]
* **Feature engineering/selection:** [Extracting specific columns from KCC/Yield data]

## 7. Dataset Integration

* **Datasets combined:** [e.g., Merging multiple policy PDFs into one vector store, or aligning different KCC CSV versions]
* **Integration methodology:** [Schema alignment, deduplication]

## 8. Data Augmentation

* **Techniques used:** [Rotation, flip, brightness adjustment for PlantVillage to simulate field conditions]
* **Rationale:** [Bridging the lab-to-field domain gap]
* **Generated samples:** [Total number of augmented images]

## 9. Dataset Splitting

* **Split ratio:** [e.g., 70% Train / 15% Val / 15% Test]
* **Number of samples:** [Counts per split]
* **Stratified sampling:** [Ensuring equal class distribution across splits]
* **Leakage prevention:** [Ensuring augmented images stay in the train set; keeping policy chunks isolated by document if testing retrieval]

## 10. Final Prepared Dataset

* **Final sizes and features:** [Total processed records ready for the pipeline]
* **Final class distribution:** [Post-augmentation balance]
* **Summary of readiness:** [Confirmation that data is ready for CNN training, Vector DB ingestion, and ML modeling]

## 11. Challenges Encountered

* **Data availability/quality:** [e.g., Scarcity of UP-specific vernacular queries, poor resolution in PlantDoc]
* **Integration/Limitation:** [Handling unstructured PDFs vs. structured CSVs]

## 12. Deliverables Produced

* Cleaned and preprocessed datasets (Images, Text, Tabular)
* Train/Validation/Test splits
* Automated download scripts (`download_data.py`) and preprocessing notebooks (`.ipynb`)
* Updated Data Dictionary

## 13. Summary and Next Steps

* **Summary of work completed:** [Brief recap of the dataset preparation phase]
* **Key observations:** [Insights gained during EDA]
* **Planned activities for Milestone 3:** [e.g., Model training (CNN), MuRIL embedding generation, LLM prompt engineering]