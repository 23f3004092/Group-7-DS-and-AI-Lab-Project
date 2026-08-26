"""Retrieval = query embedding (BGE-M3 on GPU) + Qdrant HNSW search.

This is a faithful port of `search_agri_knowledge` from notebook
08b_rag_vector_db_bge_m3.ipynb. Differences on purpose:
  * SentenceTransformers runs the exact BGE-M3 query encoder on CUDA. FastEmbed
    does not support BAAI/bge-m3 and silently falling back to CPU added seconds
    to every request.
  * The tier thresholds / fusion weights / prefixes are read from manifest.json,
    so they stay identical to the built index.

MIGRATION NOTE (2026-08): backported improvements from scripts/run_e2e_eval.py:
  * Intent → source routing (INTENT_SOURCE_MAP) + KCC query_type sub-filtering
    with confidence-gated loosening (CONF_GATE_LOOSEN).
  * Dynamic TOP_K scaling with intent count.
  * Multi-query retrieval (batched embedding + parallel Qdrant searches).
  * Post-retrieval filtering: score floor, min chunk length, per-source cap.
  * Cross-encoder reranking (lazy-loaded, default-on).
These activate ONLY when the collection uses the ROUTED payload schema
(payload key "source", e.g. "kcc_qa"). The legacy schema ("source_type"
= pdf/kcc + fusion weights) keeps working unchanged — detected at load().
"""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range

from . import config as C
from .log import get as _get_log

log = _get_log("retrieval")

_qdrant: Optional[QdrantClient] = None
_st = None                  # exact BGE-M3 encoder used to build the index
_backend = None
_schema = "legacy"          # "legacy" (source_type pdf/kcc) | "routed" (source kcc_qa/...)
_reranker = None            # lazy cross-encoder (only touched on first multi-hit search)

# --- intent routing tables (backported from run_e2e_eval.py lines 86-119) ---
# Note: ROUTED-schema KCC chunks use source="kcc_qa".
INTENT_SOURCE_MAP = {
    "disease_pest":         ["ppqs_advisories", "kcc_qa"],
    "cultivation_practice": ["up_acp", "kcc_qa"],
    "nutrition_fertilizer": ["ppqs_advisories", "kcc_qa"],
    "yield_estimation":     ["kcc_qa"],
    "post_harvest_storage": ["kcc_qa"],
    "specialty_other":      ["kcc_qa"],
    "policy":               ["schemes"],   # UP govt scheme circulars live here
    # fine gateway intents map onto the same sources:
    "field_practice":       None,
    "weather":              None,
    "market_price":         None,
    "mandi_prices":         None,
    "general":              None,   # no filter → search all
}

# Reverse mapping from IEG intent back to raw KCC query_type.
# Prevents searching ~716k KCC chunks when we only care about a specific intent.
INTENT_QTYPE_MAP = {
    "disease_pest": [
        "Plant Protection", "Weed Management", "Insect Management",
        "Pathogenic Disease Management",
    ],
    "nutrition_fertilizer": [
        "Nutrient Management", "Fertilizer Use and Availability",
        "Nutrient Deficiency/Excessiveness Management",
        "Bio-Pesticides and Bio-Fertilizers",
    ],
    "cultivation_practice": [
        "Cultural Practices", "Varieties", "Varietal Selection",
        "Seeds and Planting Material", "Seeds", "Seed Sowing And Treatment",
        "Field Preparation", "Water Management",
        "Water Management, Micro Irrigation", "Irrigation Management",
        "Soil Testing", "Abiotic Stress Management",
    ],
    "post_harvest_storage": [
        "Post Harvest Preservation", "Storage", "Cold Storage",
    ],
    "specialty_other": [
        "Organic Farming", "Floriculture", "Beekeeping",
    ],
}


def _get_reranker():
    """Lazy-load the cross-encoder reranker (downloaded once, cached). Optional."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def load():
    """Load exact-match BGE-M3 on CUDA and connect to Qdrant."""
    global _qdrant, _st, _backend, _schema
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible for BGE-M3 query embedding.")
    _qdrant = QdrantClient(url=C.QDRANT_URL, timeout=120)
    from sentence_transformers import SentenceTransformer
    # Keep the encoder in float32 to preserve the calibrated similarity scores.
    # Only its device changes; model, prefixes and normalization remain identical.
    _st = SentenceTransformer(C.EMBED_MODEL, device="cuda")
    _st.max_seq_length = C.MAX_SEQ_LENGTH
    _backend = "sentence-transformers-cuda-fp32"

    # --- payload schema detection -------------------------------------------
    # legacy snapshot: payload key "source_type" (pdf | kcc)
    # routed snapshot: payload key "source"     (kcc_qa | ppqs_advisories | ...)
    try:
        pts, _ = _qdrant.scroll(collection_name=C.COLLECTION, limit=1,
                                with_payload=True, with_vectors=False)
        if pts and "source" in (pts[0].payload or {}):
            _schema = "routed"
        else:
            _schema = "legacy"
    except Exception as e:
        log.warning("schema detection failed (%s); assuming legacy", e)
        _schema = "legacy"
    print(f"[retrieval] payload schema = {_schema}")

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


def _citation_routed(p: dict) -> dict:
    """Citation builder for the ROUTED schema (payload key 'source')."""
    src = p.get("source", "unknown")
    if src == "kcc_qa":
        return {"corpus": "kcc", "record": "KCC Q&A", "source": src,
                "crop": p.get("crop"), "district": p.get("district"),
                "season": p.get("season"), "query_type": p.get("query_type"),
                "year": p.get("year")}
    # pdf-family sources under the routed schema (ppqs_advisories, up_acp, schemes, ...)
    return {"corpus": "pdf", "source": src, "file": p.get("filename"),
            "pages": [p.get("page_start"), p.get("page_end")],
            "section": p.get("heading_hierarchy") or None,
            "doc_category": p.get("doc_category"),
            "district": p.get("district"), "year": p.get("year")}


def _build_routed_filter(intents: list, loosen_qtypes: bool):
    """Intent → source routing filter for the ROUTED schema.
    Backported from run_e2e_eval.py retrieve() lines 422-465."""
    sources, qtypes = set(), set()
    allow_all_kcc = False
    for intent in intents:
        srcs = INTENT_SOURCE_MAP.get(intent)
        if not srcs:
            continue
        sources.update(srcs)
        if "kcc_qa" in srcs:
            if intent in INTENT_QTYPE_MAP:
                qtypes.update(INTENT_QTYPE_MAP[intent])
            else:
                allow_all_kcc = True

    if not sources:
        return None

    should_clauses = []
    non_kcc = [s for s in sources if s != "kcc_qa"]
    if non_kcc:
        should_clauses.append(
            Filter(must=[FieldCondition(key="source", match=MatchAny(any=non_kcc))]))

    if "kcc_qa" in sources:
        if allow_all_kcc or not qtypes or loosen_qtypes:
            # Confidence gate fired (or unmapped intent): drop the qtype sub-filter,
            # keep the source-level filter. The reranker recovers precision; it cannot
            # recover chunks that were filtered out.
            should_clauses.append(
                Filter(must=[FieldCondition(key="source", match=MatchValue(value="kcc_qa"))]))
        else:
            should_clauses.append(Filter(must=[
                FieldCondition(key="source", match=MatchValue(value="kcc_qa")),
                FieldCondition(key="query_type", match=MatchAny(any=list(qtypes))),
            ]))

    return Filter(should=should_clauses) if should_clauses else None


def _search_routed(query: str, intents: list, top_k: int, loosen_qtypes: bool):
    """Experimental retrieval engine (run_e2e_eval.py retrieve()): dynamic TOP_K,
    multi-query batched+parallel search, post-filters, cross-encoder reranking.
    Returns list of qdrant ScoredPoints (already filtered/reranked)."""
    # Dynamic TOP_K scaling with intent count
    active_top_k = max(C.TOP_K_BASE, len(intents) * 5) if len(intents) > 2 else max(top_k, C.TOP_K_BASE)
    query_filter = _build_routed_filter(intents, loosen_qtypes)

    if len(intents) <= 1:
        q_vec = embed_query(query)
        hits = _qdrant.query_points(
            collection_name=C.COLLECTION, query=q_vec,
            query_filter=query_filter, limit=active_top_k,
            with_payload=True).points
    else:
        # Multi-query retrieval: BATCHED embedding + parallel searches
        limit_per_intent = max(2, active_top_k // len(intents))
        sub_queries = [f"{i.replace('_', ' ')}: {query}" for i in intents]
        vecs = _st.encode(sub_queries, batch_size=len(sub_queries),
                          normalize_embeddings=True, show_progress_bar=False)

        def _search(vec):
            return _qdrant.query_points(
                collection_name=C.COLLECTION, query=vec.tolist(),
                query_filter=query_filter, limit=limit_per_intent,
                with_payload=True).points

        with ThreadPoolExecutor(max_workers=len(vecs)) as pool:
            results = list(pool.map(_search, vecs))

        hits, seen = [], set()
        for sub_hits in results:
            for h in sub_hits:
                if h.id not in seen:
                    seen.add(h.id)
                    hits.append(h)
        hits = sorted(hits, key=lambda x: x.score, reverse=True)[:active_top_k]

    if not hits:
        return []

    # Per-chunk score floor
    hits = [h for h in hits if h.score >= C.MIN_CHUNK_SCORE]
    if not hits:
        return []

    # Minimum chunk length (OCR noise / empty chunks)
    hits = [h for h in hits if len(h.payload.get("text", "")) >= C.MIN_CHUNK_CHARS]
    if not hits:
        return []

    # Source diversity cap
    source_counts, kept = Counter(), []
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        src = h.payload.get("source", "unknown")
        if source_counts[src] < C.MAX_PER_SOURCE:
            kept.append(h)
            source_counts[src] += 1
    hits = kept[:active_top_k]
    if not hits:
        return []

    # Cross-encoder reranking (optional — falls back to vector score order)
    if len(hits) > 1:
        try:
            reranker = _get_reranker()
            pairs = [(query, h.payload.get("text", "")[:512]) for h in hits]
            scores = reranker.predict(pairs)
            hits = [h for _, h in sorted(zip(scores, hits),
                                         key=lambda x: x[0], reverse=True)]
            hits = hits[:C.RERANK_TOP_N]
        except Exception as e:
            log.warning("reranker unavailable (%s); using vector order", e)

    return hits


def search_agri_knowledge(query, top_k=None, intent="general", source_type=None,
                          doc_category=None, query_type=None, crop=None, district=None,
                          season=None, language=None, year_from=None, only_tables=None,
                          top_confidence=None, intents=None):
    """JSON-in / JSON-out. Never raises. Returns {query, intent, tier, top_score, results}.

    NEW optional params (backward compatible):
      top_confidence -- IEG top-1 softmax confidence; enables confidence-gated
                        qtype-filter loosening when < C.CONF_GATE_LOOSEN.
      intents        -- multi-intent list from IEG; enables multi-query retrieval
                        and dynamic TOP_K scaling (routed schema only).
    """
    t0 = time.time()
    top_k = top_k or C.TOP_K_DEFAULT

    # Confidence-gated loosening decision (0 ms — conf is already computed by IEG)
    loosen_qtypes = bool(top_confidence is not None and top_confidence < C.CONF_GATE_LOOSEN)
    intents = [i for i in (intents or []) if i] or None

    if loosen_qtypes:
        log.info("LOOSEN_GATE q=%r top_conf=%.3f threshold=%.3f intents=%s",
                 (query or "")[:80], top_confidence, C.CONF_GATE_LOOSEN, intents)

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
        if _schema == "routed":
            pts = _search_routed(query, intents or [intent], top_k, loosen_qtypes)
            hits = [{
                "raw_score": round(float(h.score), 4),
                "fused_score": round(float(h.score), 4),   # no fusion weights in routed schema
                "text": h.payload.get("text", ""),
                "source_type": h.payload.get("source", "unknown"),
                "has_table": bool(h.payload.get("has_table", False)),
                "chunk_id": h.payload.get("chunk_id"),
                "citation": _citation_routed(h.payload),
            } for h in pts]
        else:
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
        log.exception("search failed")
        return {"query": query, "tier": "error", "top_score": 0.0, "results": [], "error": str(e)}

    best_raw = max((h["raw_score"] for h in hits), default=0.0)   # tier on RAW cosine
    tier = ("grounded" if best_raw >= C.TIER_GROUNDED
            else "fallback_with_disclaimer" if best_raw >= C.TIER_FALLBACK
            else "abstain_out_of_scope")
    result = {"query": query, "intent": intent, "tier": tier,
              "top_score": round(best_raw, 4), "results": hits}
    # NEW diagnostic fields (backward compatible):
    if intents:
        result["intents"] = intents
    result["filters_loosened"] = loosen_qtypes
    result["retrieval_ms"] = round((time.time() - t0) * 1000)
    return result


