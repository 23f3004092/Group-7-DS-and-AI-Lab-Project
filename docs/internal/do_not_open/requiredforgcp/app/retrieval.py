"""Retrieval = query embedding (BGE-M3 on GPU) + Qdrant HNSW search.

This is a faithful port of `search_agri_knowledge` from notebook
08b_rag_vector_db_bge_m3.ipynb. Differences on purpose:
  * SentenceTransformers runs the exact BGE-M3 query encoder on CUDA. FastEmbed
    does not support BAAI/bge-m3 and silently falling back to CPU added seconds
    to every request.
  * The tier thresholds / fusion weights / prefixes are read from manifest.json,
    so they stay identical to the built index.
"""
from typing import Optional

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from . import config as C

_qdrant: Optional[QdrantClient] = None
_st = None                  # exact BGE-M3 encoder used to build the index
_backend = None


def load():
    """Load exact-match BGE-M3 on CUDA and connect to Qdrant."""
    global _qdrant, _st, _backend
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible for BGE-M3 query embedding.")
    _qdrant = QdrantClient(url=C.QDRANT_URL, timeout=120)
    from sentence_transformers import SentenceTransformer
    # Keep the encoder in float32 to preserve the calibrated similarity scores.
    # Only its device changes; model, prefixes and normalization remain identical.
    _st = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    _st.max_seq_length = C.MAX_SEQ_LENGTH
    _backend = "sentence-transformers-cuda-fp32"
    _ = embed_query("warmup")     # avoid a slow first request
    torch.cuda.synchronize()
    print(f"[retrieval] embedder backend = {_backend}")
    return _qdrant


@torch.inference_mode()
def embed_query(text: str):
    """Encode one query to a 1024-dim vector. bge-m3 uses NO prefix (manifest)."""
    t = C.QUERY_PREFIX + text
    return _st.encode(t, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False).tolist()


# --- canonicalizers -------------------------------------------------------
# The build notebook used canon_crop / canon_district to normalize filter values
# to the exact spellings stored in the payload. If you pass crop/district filters,
# replace these with the real maps from the preprocessing notebooks.
def canon_crop(x: str) -> str:
    return (x or "").strip().lower()


def canon_district(x: str) -> str:
    return (x or "").strip().title()


def _citation(p: dict) -> dict:
    if p.get("source_type") == "pdf":
        return {"corpus": "pdf", "file": p.get("filename"),
                "pages": [p.get("page_start"), p.get("page_end")],
                "section": p.get("heading_hierarchy") or None,
                "doc_category": p.get("doc_category"),
                "district": p.get("district"), "year": p.get("year")}
    return {"corpus": "kcc", "record": "KCC Q&A", "crop": p.get("crop"),
            "district": p.get("district"), "season": p.get("season"),
            "query_type": p.get("query_type"), "year": p.get("year")}


def search_agri_knowledge(query, top_k=None, intent="general", source_type=None,
                          doc_category=None, query_type=None, crop=None, district=None,
                          season=None, language=None, year_from=None, only_tables=None):
    """JSON-in / JSON-out. Never raises. Returns {query, intent, tier, top_score, results}."""
    top_k = top_k or C.TOP_K_DEFAULT
    weights = C.FUSION_WEIGHTS.get(intent, C.FUSION_WEIGHTS["general"])

    def sub_search(stype, qvec):
        must = [FieldCondition(key="source_type", match=MatchValue(value=stype))]
        if doc_category: must.append(FieldCondition(key="doc_category", match=MatchValue(value=doc_category)))
        if query_type:   must.append(FieldCondition(key="query_type", match=MatchValue(value=query_type)))
        if crop:         must.append(FieldCondition(key="crop", match=MatchValue(value=canon_crop(crop))))
        if district:     must.append(FieldCondition(key="district", match=MatchValue(value=canon_district(district))))
        if season:       must.append(FieldCondition(key="season", match=MatchValue(value=season)))
        if language:     must.append(FieldCondition(key="language", match=MatchValue(value=language)))
        if year_from:    must.append(FieldCondition(key="year", range=Range(gte=year_from)))
        if only_tables:  must.append(FieldCondition(key="has_table", match=MatchValue(value=True)))
        return _qdrant.query_points(
            collection_name=C.COLLECTION, query=qvec,
            query_filter=Filter(must=must), limit=top_k, with_payload=True,
        ).points

    try:
        qvec = embed_query(query)
        sources = [source_type] if source_type else ["pdf", "kcc"]
        hits = []
        for stype in sources:
            for h in sub_search(stype, qvec):
                hits.append({
                    "raw_score": round(float(h.score), 4),
                    "fused_score": round(float(h.score) * weights.get(stype, 1.0), 4),
                    "text": h.payload.get("text", ""),
                    "source_type": stype,
                    "has_table": bool(h.payload.get("has_table", False)),
                    "chunk_id": h.payload.get("chunk_id"),
                    "citation": _citation(h.payload),
                })
        hits.sort(key=lambda x: x["fused_score"], reverse=True)
        hits = hits[:top_k]
    except Exception as e:  # never raise into the request path
        return {"query": query, "tier": "error", "top_score": 0.0, "results": [], "error": str(e)}

    best_raw = max((h["raw_score"] for h in hits), default=0.0)   # tier on RAW cosine
    tier = ("grounded" if best_raw >= C.TIER_GROUNDED
            else "fallback_with_disclaimer" if best_raw >= C.TIER_FALLBACK
            else "abstain_out_of_scope")
    return {"query": query, "intent": intent, "tier": tier,
            "top_score": round(best_raw, 4), "results": hits}


