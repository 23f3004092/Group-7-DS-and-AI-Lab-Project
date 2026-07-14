**KCC Preprocessing Notebook: Detailed Work Report**

------------------------------------------------------------------------

**1. Data Loading**

**Work Done:**

- Mounted Google Drive and loaded the raw KCC dataset (kcc_combined_2020_2025.csv) containing 3,123,029 records.

- Verified the dataset structure with 15 columns including query text, expert answers, crop information, district, category, and temporal data.

- Created necessary directory structures for processed and final outputs.

**Findings:**

- Dataset successfully loaded with all columns intact.

- Data spans from 2020-2025 and is exclusively for Uttar Pradesh state.

- Key columns: QueryText, KccAns, Crop, Category, DistrictName, QueryType, year, month.

**Data Location:**

- **Input Raw Data:** [/content/drive/MyDrive/kcc_raw/kcc_combined_2020_2025.csv ](https://drive.google.com/file/d/17F0_P5di9IXNkE3rrTK3DUsX8km0Yhli/view?usp=drive_link)(\~2.02 GB)

------------------------------------------------------------------------

**2. Data Overview & Quality Check**

**Work Done:**

- Examined data types, column structures, and basic statistics.

- Verified geographic and temporal filters (Uttar Pradesh only, years ≥ 2020).

- Analyzed crop distribution across the dataset.

**Findings:**

- State filter confirmed: All records are from Uttar Pradesh.

- Year range verified: 2020-2025.

- Top 5 crops identified: Others (1.08M), Wheat (512K), Paddy (485K), Sugarcane (126K), Potato (107K).

- Season column found to be entirely null (100% missing) - requires inference or imputation.

------------------------------------------------------------------------

**3. Filtering for Agronomic Categories**

**Work Done:**

- Applied inclusion filter for agronomic categories: Cereals, Pulses, Oilseeds, Vegetables, Fruits, Plant Protection, Nutrient Management, etc.

- Applied exclusion filter to remove: Government Schemes, Market Information, Crop Insurance, Credit, Subsidy, Weather, etc.

**Findings:**

- **1,701,442 records** retained (54.48% of original data).

- Removed \~1.4M non-agronomic records (financial, schemes, weather-related).

- Remaining categories distribution:

  - Cereals: 1,000,881

  - Vegetables: 336,545

  - Oilseeds: 153,852

  - Pulses: 109,358

  - Fruits: 100,707

- This filter successfully mitigated the risk of including non-agronomic content as specified in Milestone 1.

------------------------------------------------------------------------

**4. Handling Missing Values**

**Work Done:**

- Identified columns with missing values and quantified missing percentages.

- Dropped redundant columns: CreatedOn, Sector, StateName, KCCCallID, day, BlockName.

- Handled critical field missing values:

  - Dropped records missing QueryText (7 records) and KccAns (111 records).

- Imputed missing values for non-critical fields:

  - Filled Crop with \'Unknown\' (1,909 records).

  - Filled DistrictName with \'Unknown\' (1 record).

  - Inferred Season from month column using predefined season mapping (June-Oct = Kharif, Nov-March = Rabi, April-May = Zaid) - filled 1,701,324 records.

  - Filled QueryType with \'Other\' (798 records).

  - Dropped records missing month (2 records) and year (0 records).

**Findings:**

- **1,701,322 records** remained after handling all missing values.

- Season column successfully populated from month data, eliminating the initial 100% missing issue.

- All critical fields (QueryText, KccAns, Crop, DistrictName) verified to have zero missing values.

- Columns reduced from 15 to 9 essential columns.

------------------------------------------------------------------------

**5. Duplicate Removal**

**Work Done:**

- Removed exact duplicates across all columns (39,111 records removed).

- Removed duplicate Q&A pairs for the same crop using QueryText, KccAns, and Crop columns (202,509 records removed).

**Findings:**

- **1,459,702 records** remained after deduplication.

- Total reduction: 12.2% (241,620 duplicates removed).

- This ensures diversity in the dataset by preserving different answers for the same query type while removing identical Q&A pairs.

------------------------------------------------------------------------

**6. Text Cleaning**

**Work Done:**

- Applied comprehensive text cleaning functions:

  - clean_text(): Removed extra whitespace, fixed encoding issues, retained Devanagari and regional script characters.

  - remove_pii(): Redacted phone numbers, email addresses, and ID patterns.

  - normalize_indic_text(): Normalized Unicode for Indian scripts (NFC normalization), removed zero-width characters.

  - detect_language(): Identified language of text (Hindi, English, Mixed) based on character ratio.

- Removed records with very short text (\<5 characters) - 10 records removed.

**Findings:**

- **1,459,692 records** remained after cleaning.

- Language distribution:

  - English: 1,459,452 (100%)

  - Mixed: 243 (0.0%)

  - Unknown: 7 (0.0%)

- All personal identification information successfully redacted.

- Text encoding issues resolved through normalization.

------------------------------------------------------------------------

**7. Metadata Tagging**

**Work Done:**

- Created structured metadata dictionary for each record containing:

  - crop, district, block, season, query_type, category, year, month, language

- Ensured all fields are in appropriate data types (integers for year/month).

**Findings:**

- Metadata successfully attached to all 1,459,692 records.

- Enables filtered retrieval in RAG pipeline (filter by crop, district, season, etc.).

- Metadata structure aligned with project requirements for Milestone 2.

------------------------------------------------------------------------

**8. Chunk Preparation for RAG**

**Work Done:**

- Configured chunking parameters: 512-character chunks with 50-character overlap.

- Implemented smart chunking strategy:

  - Combined Q&A pairs in format: \"Question: {query}\nAnswer: {answer}\"

  - Single chunk if text ≤ 512 characters.

  - Multi-chunk splitting at sentence boundaries (.!?) to preserve meaning.

  - Overlap implemented using last 3-5 words for context preservation.

- Processed 1,459,692 records in batches of 10,000.

**Findings:**

- **1,468,625 chunks** created from 1,459,692 records.

- Average chunk length: 288 characters.

- **98.8%** of records (1,451,474) fit in a single chunk.

- Only **1.2%** (17,151) required multiple chunks.

- Successful chunking ensures MuRIL\'s 512-token limit is respected while preserving contextual coherence.

------------------------------------------------------------------------

**9. Data Export & Storage**

**Work Done:**

- Created Google Drive folders: /processed/ and /final/.

- Exported multiple file formats with organized folder structure.

**Folder Structure:**

**Processed Folder [(/content/drive/MyDrive/kcc_raw/processed/)](https://drive.google.com/drive/folders/1qd1EOWQDVcR4skw2hc7cdbcFT8PybLvB?usp=drive_link)**

Contains intermediate data files for reference and validation:

| **File Name** | **Description** | **Size** | **Records** |
|----|----|----|----|
| [kcc_cleaned_all_crops.csv](https://drive.google.com/file/d/1cxxBJ-xEl-DnBeLMzkRfjf2JLDD9AmIu/view?usp=drive_link) | Final cleaned dataset after all preprocessing steps (filtering, missing value handling, deduplication, text cleaning). Contains all records with cleaned text and metadata. | \~2.0 GB | 1,459,692 |
| [kcc_qa_pairs.csv](https://drive.google.com/file/d/1bp_6n3LQ0TCoUI7QpbfsAsWx17893-Bd/view?usp=drive_link) | Query-Answer pairs in CSV format for easy viewing and manual review. Contains cleaned queries, answers, and key metadata. | \~894 MB | 1,459,692 |
| [kcc_chunks_sample_1000.jsonl](https://drive.google.com/file/d/11Y9pL0UReae40YZZ_-26kW571_G5Jj3I/view?usp=drive_link) | Sample of 1000 chunks for quick testing and validation. | \~762 KB | 1,000 |

**Final Folder [(/content/drive/MyDrive/kcc_raw/final/)](https://drive.google.com/drive/folders/1tE6Nke-R451CcPB6Ik-91q7YRi3R4kUY?usp=drive_link)**

Contains RAG-ready data for embedding pipeline:

| **File Name** | **Description** | **Size** | **Records** |
|----|----|----|----|
| [kcc_chunks_rag.jsonl](https://drive.google.com/file/d/1UsfqVJNdDTI4NQ4F_4z2UPQ33IH1a9s6/view?usp=drive_link) | Main chunked dataset ready for MuRIL embedding. Each record contains Q&A text split into 512-character chunks with complete metadata for filtered retrieval. | \~1.18 GB | 1,468,625 |
| [metadata_schema.json](https://drive.google.com/file/d/1HTyGAv94SXhz5NZAfYifPyRjdWF7XQyy/view?usp=drive_link) | Complete documentation of dataset structure, chunk configuration, metadata schema, embedding model specifications, and RAG tiered relevance thresholds. | \~7 KB | N/A |

------------------------------------------------------------------------

**Overall Summary**

| **Metric**                   | **Value**                    |
|------------------------------|------------------------------|
| Original Records             | 3,123,029                    |
| After Agronomic Filtering    | 1,701,442                    |
| After Missing Value Handling | 1,701,322                    |
| After Deduplication          | 1,459,702                    |
| After Text Cleaning          | 1,459,692                    |
| Total Chunks Created         | 1,468,625                    |
| Data Reduction               | 53.3%                        |
| Dominant Language            | English (100%)               |
| Top Crops                    | Wheat (29.6%), Paddy (26.2%) |
| Chunking Efficiency          | 98.8% single-chunk records   |

------------------------------------------------------------------------

**Risk Mitigation & Quality Assurance**

1.  **PII Removal**: Successfully redacted phone numbers, emails, and IDs to ensure data privacy compliance.

2.  **Agronomic Focus**: Excluded 45.5% of non-agronomic records, aligning with project objectives.

3.  **Data Quality**: All critical missing values handled; zero missing in essential fields.

4.  **Duplication**: 12.2% duplication removed ensuring diverse training data.

5.  **Chunk Quality**: 98.8% records fit in single chunks, minimizing context fragmentation.

------------------------------------------------------------------------

**Data Storage Summary**

| **Location** | **Path** | **Total Size** |
|----|----|----|
| **Raw Data** | /content/drive/MyDrive/kcc_raw/ | \~2.02 GB |
| **Processed Data** | /content/drive/MyDrive/kcc_raw/processed/ | \~2.9 GB |
| **Final Data** | /content/drive/MyDrive/kcc_raw/final/ | \~1.18 GB |
| **Total Storage** | All KCC Data | \~6 GB |

------------------------------------------------------------------------

**Recommendations for Milestone 3:**

- Use the kcc_chunks_rag.jsonl file (1.18 GB) directly for MuRIL embedding generation.

- Metadata fields (crop, district, season, query_type) can be used for filtered retrieval.

- Consider sampling strategies for training vs. evaluation splits.

- The sample chunks file (1000 records, 762 KB) can be used for quick pipeline testing before full-scale processing.

- All files are available in both Google Drive and local paths for redundancy and accessibility.
