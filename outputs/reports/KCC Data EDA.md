**KCC Data EDA & RAG Pipeline Preparation - Summary**

**Project Overview**

This notebook conducts a comprehensive Exploratory Data Analysis (EDA) on Kisan Call Center (KCC) data spanning 2020-2025, with the goal of informing the design of an NLP/RAG (Retrieval-Augmented Generation) pipeline.

**Key Work Completed**

**1. Data Loading & Initial Processing**

- **Mounted Google Drive** and loaded **6 CSV files** (kcc_up_2020.csv through kcc_up_2025.csv)

- **Combined** 3,123,029 records into a single dataframe with 15 columns

- **Saved** the combined dataset locally and to Google Drive (\~2.07 GB)

**2. Dataset Overview Analysis**

- **Profile**: 3.1M records, 15 columns, \~3.3 GB memory usage

- **Key Columns Identified**: KCCCallID, QueryText, KccAns, Category, QueryType, Crop, DistrictName, StateName, Year, etc.

- **Note**: Season column found to be completely empty (100% null)

- **Dataset**: [kcc_combined_2020_2025.csv](https://drive.google.com/file/d/17F0_P5di9IXNkE3rrTK3DUsX8km0Yhli/view?usp=drive_link)

**3. Crop Distribution Analysis**

- **318 unique crops** identified

- **Top Crops**: Others (34.74%), Wheat (16.40%), Paddy/Dhan (15.54%)

- **Rice & Wheat combined** account for \~31.95% of queries

**4. Query Category Analysis**

- **40 categories** and **83 query types** identified

- **Top Categories**: Others (35.04%), Cereals (32.05%), Vegetables (10.78%)

- **Top Query Types**: Weather (33.47%), Government Schemes (25.63%), Plant Protection (13.87%)

**5. Temporal Analysis**

- **Yearly trends**: Peak in 2022 (620,773 queries), declining trend since

- **Monthly patterns**: Highest in January, lowest in May

- **Quarterly analysis**: Q1 dominates (33.16%), followed by Q3 (25.07%)

- **Seasonal patterns**: Clear agricultural seasonality with peaks during key farming periods

**6. Language Analysis**

- **Queries**: \~99.98% in English (including Hinglish)

- **Answers**: \~98.80% in Hindi (Devanagari script)

- **Key Insight**: Multilingual processing needed (English queries → Hindi responses)

**7. Text Length Analysis**

- **Query Length**: Mean \~54 chars, Median \~54 chars, 95th percentile \~85 chars

- **Answer Length**: Mean \~209 chars, Median \~203 chars, 95th percentile \~392 chars

- **Combined Q&A**: 95th percentile \~432 characters

- **Critical Finding**: 98.9% of records fit within 512 characters

**8. Data Quality Assessment**

- **Missing Values**: Minimal except Season (100% null)

- **Duplicates**:

  - Exact duplicate rows: 0%

  - Duplicate queries: 68.72%

  - Duplicate Q&A pairs: 26.15%

  - Duplicate IDs: 13.15%

- **Most Common Duplicate**: \"Farmer asked query on Weather\" appears 781,352 times

**9. Sample Record Analysis**

- Examined samples by category and query type

- Validated data structure and content quality

**Key Insights for RAG Pipeline**

**Critical Findings**

1.  **Language**: System must support English/Hinglish queries with Hindi responses

2.  **Text Length**: Simple chunking strategy possible (512 chars covers 98.9% of data)

3.  **Deduplication**: Essential due to high query repetition (68.72% duplicates)

4.  **Model Choice**: MuRIL or similar multilingual model recommended

5.  **Domain Focus**: Weather and Government Schemes are dominant topics
