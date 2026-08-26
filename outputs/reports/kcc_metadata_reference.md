# KCC Chunk Metadata & Intent Mapping Reference

This document outlines how KCC records are processed, mapped to intents, and finally inserted into the Qdrant RAG database.

## 1. Intent Mapping Logic (`scripts/01_prepare_datasets.py`)
During dataset preparation, the raw KCC `QueryType` is mapped to an `intent_label` for training the IEG model. This mapping is strictly for the **training dataset** and is never saved into the chunks meant for RAG.

**The `QTYPE_TO_INTENT` dictionary:**
- `disease_pest`: Plant Protection, Weed Management, Insect Management, Pathogenic Disease Management
- `nutrition_fertilizer`: Nutrient Management, Fertilizer Use and Availability, Nutrient Deficiency/Excessiveness Management, Bio-Pesticides and Bio-Fertilizers
- `cultivation_practice`: Cultural Practices, Varieties, Varietal Selection, Seeds and Planting Material, Seeds, Seed Sowing And Treatment, Field Preparation, Water Management, Irrigation Management, Soil Testing, Abiotic Stress Management
- `post_harvest_storage`: Post Harvest Preservation, Storage, Cold Storage
- `specialty_other`: Organic Farming, Floriculture, Beekeeping
- *(Unmapped types are labeled `'other'`)*

**Regex Overrides:**
After the dictionary mapping, three regexes override the intent if matching keywords are found in the query or answer:
- `weather` (e.g. mausam, rain, monsoon)
- `policy` (e.g. pm kisan, beneficiary status)
- `market` (e.g. mandi price, modal price)

---

## 2. RAG Chunk Extraction (`scripts/01_prepare_datasets.py`)
When exporting KCC records for the RAG database (`kcc_chunks_rag.jsonl`), the script deliberately ignores the `intent_label`. It only exports the following raw fields into the `metadata` block:

```json
{
    "crop": "str",         // Standardized crop name (e.g. "rice", "wheat", "maize") or None
    "district": "str",     // Raw KCC district name, later canonicalized
    "block": "str",        // Raw KCC block name or "unknown"
    "season": "str",       // "Kharif", "Rabi", "Zaid", or "unknown" (computed from CreatedOn month)
    "query_type": "str",   // Raw KCC QueryType (e.g. "Plant Protection", "Fertilizer Use")
    "category": "str",     // Raw KCC Category (e.g. "Agriculture", "Horticulture")
    "year": "int",         // Parsed from CreatedOn (e.g. 2023) or None
    "month": "int",        // Parsed from CreatedOn (1-12) or None
    "language": "str"      // "en", "hi", or "mixed" (computed from char ratio)
}
```

---

## 3. Qdrant Payload Injection (`scripts/build_rag_artifacts.py`)
When `build_rag_artifacts.py` ingests the JSONL file to build the Qdrant snapshot, it canonicalizes some fields (like crops and districts) and drops others. 

Here is the **exact payload dictionary** that gets attached to every `kcc_qa` chunk in Qdrant:

```python
{
    "source_type":  "kcc",                   # Hardcoded
    "source":       "kcc_qa",                # Hardcoded (This is what INTENT_SOURCE_MAP should use)
    "language":     detect_language(text),   # "en", "hi", or "mixed"
    "year":         int(year),
    "crop":         canon_crop(crop_raw),    # Standardized lowercase crop name
    "district":     canon_district(m.get("district")), 
    "district_raw": m.get("district"),
    "season":       m.get("season"),
    "query_type":   m.get("query_type"),     # Raw KCC QueryType
    "category":     m.get("category"),
    "month":        int(month)
}
```

### The Root Cause of the Bug
Because `intent_label` is dropped in step 2, Qdrant cannot filter on it. The retriever relies entirely on filtering the `source` field. 

Since `build_rag_artifacts.py` hardcodes `source: "kcc_qa"`, but `run_e2e_eval.py`'s `INTENT_SOURCE_MAP` maps to `source: "kcc"`, any query mapped to `disease_pest` or `cultivation_practice` completely drops the 716k KCC chunks from its search!

---

## 4. PDF Chunk Metadata (`notebooks/02_pdfs_rag_eda.ipynb` & `scripts/build_rag_artifacts.py`)
PDF documents (such as circulars, advisories, and schemes) are parsed and chunked. Below is the exact payload structure injected into Qdrant for PDF chunks, ensuring provenance and structural context are preserved for the retrieval agent:

```python
{
    # --- Shared Fields (with KCC) ---
    "chunk_id":        "str",                    # Unique identifier (hash of chunk)
    "source_type":     "pdf",                    # Hardcoded to "pdf"
    "source":          "str",                    # "other_docs", "ppqs_advisories", "schemes", or "up_acp"
    "language":        "str",                    # "en", "hi", or "mixed" (computed via langdetect/heuristics)
    "year":            "int",                    # Extracted from filename or text (e.g. 2023) or None
    "crop":            None,                     # Not pre-tagged for PDFs currently (reserved for future use)
    "district":        "str",                    # Extracted from filename for "up_acp" docs (e.g. "ayodhya"), else None
    "chunk_index":     "int",                    # Sequential index within the document (0-indexed)
    "n_chunks_in_doc": "int",                    # Total chunks in the parent document
    
    # --- PDF-Specific Fields ---
    "filename":           "str",                 # Original PDF filename (e.g. "SONBHADRA.pdf")
    "doc_category":       "str",                 # Broad category inferred from source/path (e.g. "Advisory", "Scheme")
    "heading_hierarchy":  "str",                 # Markdown headers path (e.g. "# Title > ## Subtitle")
    "page_start":         "int",                 # Provenance: starting page number (1-indexed)
    "page_end":           "int",                 # Provenance: ending page number (1-indexed)
    "has_table":          "bool",                # True if this chunk contains a parsed markdown table
    "extraction_method":  "str",                 # "pymupdf4llm" (native text) or "pytesseract" (OCR fallback)
    "source_pdf_sha256":  "str"                  # Checksum of the source PDF for integrity
}
```
