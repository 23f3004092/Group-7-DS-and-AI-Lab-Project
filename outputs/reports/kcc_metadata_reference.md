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
    "crop": "str",
    "district": "str",
    "block": "str",
    "season": "str",
    "query_type": "str",   // The raw KCC QueryType, NOT the IEG intent_label
    "category": "str",
    "year": "int",
    "month": "int",
    "language": "str"
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
