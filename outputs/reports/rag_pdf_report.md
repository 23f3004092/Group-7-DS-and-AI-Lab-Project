# RAG PDF Corpus Preparation and Exploratory Data Analysis

## Overview

A major component of the project is the Retrieval-Augmented Generation (RAG) subsystem, which requires a high-quality knowledge base built from reliable agricultural documents rather than general internet sources. To support this, a complete document collection, cleaning, preprocessing, and chunking pipeline was developed.

The objective was not only to collect relevant agricultural PDFs but also to ensure that the corpus is clean, searchable, consistent, and suitable for embedding generation and semantic retrieval.

---

# 1. Document Collection

## Sources

The document corpus was created by collecting agricultural documents from multiple authoritative sources.

The collected documents include:

- PPQS (Plant Protection, Quarantine & Storage) advisories
- Uttar Pradesh Agriculture Contingency Plan (ACP)
- Government agricultural schemes
- Farming handbooks
- ICAR publications
- Other official agriculture-related documents

Collection involved a combination of:

- Manual downloading from official government websites
- Automated PDF scraping scripts
- Metadata verification
- Folder-wise organization

---

## Corpus Composition

| Source | Documents |
|---------|----------:|
| PPQS Advisories | 90 |
| UP ACP Documents | 74 |
| Government Schemes | 11 |
| Other Agriculture Documents | 12 |

**Total PDFs collected:** **187**

The documents cover multiple aspects of crop management including:

- Crop production
- Disease management
- Irrigation
- Government schemes
- Pest management
- Fertilizer recommendations
- Water management
- Soil health
- Cropping systems

This diversity provides broad domain coverage required for the RAG system.

---

# 2. Metadata Extraction

A custom preprocessing pipeline was developed to automatically extract metadata from every PDF.

Extracted metadata includes:

- filename
- source folder
- page count
- word count
- language
- extraction method
- detected publication year
- OCR status
- garbage character ratio

The metadata was stored in a structured inventory for further analysis.

---

# 3. Text Extraction Pipeline

A robust PDF processing pipeline was implemented.

For each document:

1. Native text extraction was attempted using pdfplumber.
2. If extraction quality was poor, OCR fallback was available.
3. Extracted text was cleaned.
4. Metadata was generated.
5. Documents were validated before inclusion.

This ensured maximum text quality while avoiding corrupted extractions.

---

# 4. Exploratory Data Analysis

Extensive EDA was performed to understand the quality and characteristics of the corpus before building the retrieval system.

The following analyses were performed.

---

## 4.1 Source Distribution

The collected PDFs originated from four different source folders.

### Findings

- PPQS advisories contributed the highest number of documents.
- UP ACP formed the second largest source.
- Scheme documents were fewer in number but significantly longer.
- The corpus contains information from multiple government agencies, improving topic diversity.

---

## 4.2 Document Length Analysis

Page count analysis was performed to understand document size variation.

### Findings

- Majority of documents contain fewer than 10 pages.
- Government schemes and farming manuals are considerably larger.
- A small number of documents exceed 50 pages.
- The longest document contains nearly 100 pages.

This justified the need for chunk-based retrieval rather than document-level retrieval.

---

## 4.3 Word Count Distribution

Word counts were computed for every document.

### Findings

- Most documents contain between 500 and 6000 words.
- Scheme documents are significantly longer than advisory documents.
- A few documents exceed 30,000 words.
- Large variation in document length confirmed that fixed-document retrieval would be ineffective.

---

## 4.4 Corpus Statistics

| Metric | Value |
|---------|------:|
| Total PDFs collected | 187 |
| Successfully processed | 170 |
| Excluded documents | 17 |
| OCR processed documents | 0 |
| Source folders | 4 |

Average statistics:

| Source | Avg Pages | Avg Words |
|---------|----------:|----------:|
| Other Documents | 15.8 | 5566 |
| PPQS Advisories | 4.9 | 1302 |
| Schemes | 32.6 | 8956 |
| UP ACP | 24.4 | 4579 |

These statistics indicate that different document categories contribute complementary knowledge.

---

## 4.5 Failed Document Detection

During processing every document was validated.

The pipeline automatically detected:

- unreadable PDFs
- extraction failures
- empty documents
- corrupted files

### Findings

| Source | Failed Documents |
|---------|----------------:|
| Other Documents | 5 |
| PPQS Advisories | 9 |
| Schemes | 0 |
| UP ACP | 0 |

These documents were excluded from the final retrieval corpus.

---

## 4.6 Language Analysis

Language detection was performed for every extracted document.

### Findings

- Nearly all documents are written in English.
- A very small number were incorrectly classified because of short text.
- Manual verification confirmed that the retained documents are suitable for English embedding models.

---

## 4.7 Domain Vocabulary Analysis

After cleaning:

- stopwords were removed
- punctuation removed
- words normalized

The most frequent agricultural terms were computed.

Top domain words include:

- water
- rice
- crop
- irrigation
- soils
- sowing
- seed
- drainage
- fodder
- management

### Interpretation

The frequent occurrence of agricultural terminology confirms that the collected corpus is highly domain specific and well aligned with the objectives of the crop advisory system.

---

# 5. Document Cleaning

Several preprocessing steps were applied before the documents were accepted into the knowledge base.

These include:

- removal of unreadable PDFs
- duplicate detection
- metadata validation
- text normalization
- whitespace cleanup
- Unicode normalization
- removal of extraction artefacts

The resulting corpus is significantly cleaner than the raw downloaded documents.

---

# 6. Chunk Generation

After document cleaning, every document was converted into retrieval chunks.

The chunking pipeline performs:

- sentence-aware splitting
- token counting
- overlap generation
- metadata preservation

Chunk metadata includes:

- chunk id
- document id
- source
- filename
- token count
- chunk index

---

## Chunk Statistics

The clean corpus produced approximately **1450 semantic chunks**.

### Observations

- Most chunks contain approximately 450–520 tokens.
- Token distribution is tightly centered around the target chunk size.
- Only a few chunks required hard splitting.
- Chunk overlap preserves contextual continuity across retrieval boundaries.

This chunk size is well suited for transformer embedding models.

---

# 7. Final Outputs Produced

The preprocessing pipeline generated the following artifacts.

### Metadata

- pdf_inventory_clean.csv

### Clean Text

- Extracted text files for every PDF

### Exclusion Lists

- excluded_unreadable_docs.csv
- excluded_near_duplicate_docs.csv

### Chunk Files

- pdf_chunks.csv
- pdf_chunks.jsonl

### EDA

- PDF EDA Notebook
- PDF Chunking Notebook

---


# 8. Key Findings

The major observations from the analysis are summarized below.

- 187 agricultural PDFs were collected from authoritative government sources.
- 170 high-quality documents were retained after cleaning.
- The corpus spans advisories, government schemes, farming manuals, and contingency plans.
- Document sizes vary considerably, validating the use of chunk-level retrieval.
- PPQS contributes the largest number of documents, while government schemes contribute the largest documents.
- The vocabulary is strongly agriculture-focused, confirming good domain coverage.
- Failed and low-quality PDFs were successfully filtered before indexing.
- Approximately 1450 retrieval-ready chunks were generated with consistent token lengths.
- The final corpus is suitable for embedding generation and semantic retrieval in the RAG pipeline.

---

# Conclusion

The RAG corpus preparation process transformed a heterogeneous collection of government agricultural PDFs into a structured, validated, and retrieval-ready knowledge base. Through automated metadata extraction, document quality analysis, cleaning, normalization, and semantic chunking, the corpus is now prepared for embedding generation and integration into the Retrieval-Augmented Generation subsystem developed in subsequent milestones.