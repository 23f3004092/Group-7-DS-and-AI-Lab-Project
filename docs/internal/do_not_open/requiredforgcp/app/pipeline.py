"""Text pipeline (multi-turn) + photo diagnosis."""
import time

from . import ieg, retrieval, generation
from .log import get as _get_log

log = _get_log("pipeline")

try:
    from . import vision
except Exception:
    vision = None

SESSIONS = {}
_MAX_MSGS = 8
_MAX_SESSIONS = 500

REFUSAL = {
    "non_agricultural": "I can only help with farming and agriculture questions.",
    "restricted": ("That product is restricted/regulated. I can't help source it — please consult "
                   "your local KVK or agriculture officer for approved, safe alternatives and correct use."),
}


def _ms(t0):
    return round((time.time() - t0) * 1000)


def _contextualize(history, query):
    if not history:
        return query
    prev = [h["content"] for h in history if h.get("role") == "user" and h.get("content")][-2:]
    return " ".join(prev + [query]) if prev else query


def _remember(session_id, history, query, answer):
    history = list(history or [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    history = history[-_MAX_MSGS:]
    if session_id is not None:
        SESSIONS[session_id] = history
        if len(SESSIONS) > _MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))
    return history


def answer_query(query, intent=None, top_k=None, filters=None, live_data=None,
                 skip_retrieval=False, history=None, include_content=False,
                 session_id=None):
    t0 = time.time()
    filters = filters or {}
    if session_id is not None:
        history = SESSIONS.get(session_id, [])
    history = list(history or [])

    g = ieg.classify(query)
    if g["blocked"]:
        log.info("BLOCKED q=%r reason=%s backend=%s",
                 (query or "")[:80], g["block_reason"], g["guardrail_backend"])
        kind = "restricted" if (g["block_reason"] or "").startswith("restricted") else "non_agricultural"
        return {"tier": "blocked", "blocked": True, "block_reason": g["block_reason"],
                "answer": REFUSAL[kind], "sources": [], "intent": g["intent"],
                "guardrail_backend": g["guardrail_backend"],
                "session_id": session_id, "history": history, "latency_ms": _ms(t0)}

    retr_intent = intent or g["retrieval_intent"]
    if skip_retrieval:
        r = {"tier": "skipped", "results": [], "top_score": 0.0}
    else:
        r = retrieval.search_agri_knowledge(_contextualize(history, query),
                                            top_k=top_k, intent=retr_intent, **filters)
    has_context = bool(r["results"]) and r["tier"] not in ("abstain_out_of_scope", "error")

    hints = ieg.external_hints(query)
    use_live = live_data if (live_data and hints) else None
    greeting = ieg.is_greeting(query)
    grounded = (not greeting) and (has_context or bool(use_live))
    log.info("Q=%r intent=%s tier=%s top=%.3f n_ctx=%d hints=%s greet=%s -> %s",
             (query or "")[:90], retr_intent, r.get("tier"), r.get("top_score") or 0.0,
             len(r.get("results") or []), hints, greeting,
             "grounded" if grounded else "chat")
    if live_data:
        log.info("  live_data=%s", {k: str(v)[:140] for k, v in live_data.items()})
    if r.get("results"):
        log.info("  chunks=%s", [(h.get("source_type"), (h.get("citation") or {}).get("crop"),
                                  (h.get("citation") or {}).get("query_type"), round(h.get("raw_score", 0), 3))
                                 for h in r["results"][:5]])

    if not grounded:
        gen = generation.generate(query, [], history=history, grounded=False)
        answer = gen["answer"]
        new_history = _remember(session_id, history, query, answer)
        return {"tier": "greeting" if greeting else r["tier"], "blocked": False,
                "answer": answer, "grounded": False, "sources": [], "top_score": r["top_score"],
                "intent": retr_intent, "lang": gen["lang"], "guardrail_backend": g["guardrail_backend"],
                "gen_ms": gen["gen_ms"], "out_tokens": gen["out_tokens"],
                "error": r.get("error"), "session_id": session_id,
                "history": new_history, "latency_ms": _ms(t0)}

    drop_ctx = (bool(hints) and bool(use_live) and not ieg.is_agri_advice(query))
    ctx = [] if drop_ctx else (r["results"] if has_context else [])
    disclaimer = bool(ctx) and (r["tier"] == "fallback_with_disclaimer")
    gen = generation.generate(query, ctx, disclaimer=disclaimer,
                              live_data=use_live, history=history, grounded=True)
    answer = gen["answer"]
    if disclaimer:
        answer = answer + "\n\n" + generation.KVK_DISCLAIMER.get(gen["lang"], generation.KVK_DISCLAIMER["en"])

    new_history = _remember(session_id, history, query, answer)
    sources = [{"n": i + 1, "score": h["raw_score"], "source_type": h["source_type"],
                "citation": h["citation"],
                **({"content": h["text"]} if include_content else {})}
               for i, h in enumerate(ctx)]
    return {"tier": r["tier"], "blocked": False, "answer": answer, "grounded": bool(ctx),
            "sources": sources,
            "live_data_used": sorted(use_live.keys()) if use_live else [],
            "top_score": r["top_score"], "intent": retr_intent, "lang": gen["lang"],
            "guardrail_backend": g["guardrail_backend"], "gen_ms": gen["gen_ms"],
            "out_tokens": gen["out_tokens"], "session_id": session_id,
            "history": new_history, "latency_ms": _ms(t0)}


def answer_query_stream(query, intent=None, top_k=None, filters=None, live_data=None,
                        skip_retrieval=False, history=None, include_content=False,
                        session_id=None):
    """SSE-friendly version of answer_query with an identical authoritative result."""
    t0 = time.time()
    filters = filters or {}
    if session_id is not None:
        history = SESSIONS.get(session_id, [])
    history = list(history or [])
    yield {"type": "status", "stage": "classifying"}

    g = ieg.classify(query)
    if g["blocked"]:
        kind = "restricted" if (g["block_reason"] or "").startswith("restricted") else "non_agricultural"
        result = {"tier": "blocked", "blocked": True,
                  "block_reason": g["block_reason"], "answer": REFUSAL[kind],
                  "sources": [], "intent": g["intent"],
                  "guardrail_backend": g["guardrail_backend"],
                  "session_id": session_id, "history": history,
                  "latency_ms": _ms(t0)}
        yield {"type": "final", "data": result}
        return

    retr_intent = intent or g["retrieval_intent"]
    yield {"type": "status", "stage": "retrieving"}
    if skip_retrieval:
        r = {"tier": "skipped", "results": [], "top_score": 0.0}
    else:
        r = retrieval.search_agri_knowledge(_contextualize(history, query),
                                            top_k=top_k, intent=retr_intent, **filters)
    has_context = bool(r["results"]) and r["tier"] not in ("abstain_out_of_scope", "error")
    hints = ieg.external_hints(query)
    use_live = live_data if (live_data and hints) else None
    greeting = ieg.is_greeting(query)
    grounded = (not greeting) and (has_context or bool(use_live))
    drop_ctx = (bool(hints) and bool(use_live) and not ieg.is_agri_advice(query))
    ctx = [] if drop_ctx else (r["results"] if has_context else [])
    disclaimer = bool(ctx) and (r["tier"] == "fallback_with_disclaimer")

    yield {"type": "status", "stage": "generating",
           "tier": r["tier"], "top_score": r["top_score"]}
    gen = None
    stream = generation.generate_stream(
        query, ctx if grounded else [], disclaimer=disclaimer if grounded else False,
        live_data=use_live if grounded else None, history=history, grounded=grounded)
    for event in stream:
        if event.get("type") == "generation_final":
            gen = event["data"]
        else:
            yield event
    if gen is None:
        raise RuntimeError("generation stream ended without a final payload")

    answer = gen["answer"]
    if disclaimer:
        answer += "\n\n" + generation.KVK_DISCLAIMER.get(
            gen["lang"], generation.KVK_DISCLAIMER["en"])
    new_history = _remember(session_id, history, query, answer)
    sources = [{"n": i + 1, "score": h["raw_score"],
                "source_type": h["source_type"], "citation": h["citation"],
                **({"content": h["text"]} if include_content else {})}
               for i, h in enumerate(ctx)]

    if not grounded:
        result = {"tier": "greeting" if greeting else r["tier"],
                  "blocked": False, "answer": answer, "grounded": False,
                  "sources": [], "top_score": r["top_score"],
                  "intent": retr_intent, "lang": gen["lang"],
                  "guardrail_backend": g["guardrail_backend"],
                  "gen_ms": gen["gen_ms"], "out_tokens": gen["out_tokens"],
                  "error": r.get("error"), "session_id": session_id,
                  "history": new_history, "latency_ms": _ms(t0)}
    else:
        result = {"tier": r["tier"], "blocked": False, "answer": answer,
                  "grounded": bool(ctx), "sources": sources,
                  "live_data_used": sorted(use_live.keys()) if use_live else [],
                  "top_score": r["top_score"], "intent": retr_intent,
                  "lang": gen["lang"], "guardrail_backend": g["guardrail_backend"],
                  "gen_ms": gen["gen_ms"], "out_tokens": gen["out_tokens"],
                  "session_id": session_id, "history": new_history,
                  "latency_ms": _ms(t0)}
    yield {"type": "final", "data": result}


def diagnose_image(image_bytes, question=None, top_k=None):
    t0 = time.time()
    if vision is None or vision._model is None:
        return {"error": "vision module not loaded"}
    v = vision.predict(image_bytes)
    
    # ALWAYS include the crop and disease in the query, even if the user provided a question.
    crop_disease = f"{v['crop']} {v['disease']}"
    if question:
        q = f"{crop_disease} {question}"
    else:
        q = f"{crop_disease} treatment"
    
    r = retrieval.search_agri_knowledge(q, top_k=top_k, intent="field_practice")
    if r["tier"] in ("abstain_out_of_scope", "error"):
        return {"diagnosis": v, "tier": r["tier"], "answer": None,
                "message": "Diagnosis done, but no matching treatment found in the knowledge base.",
                "sources": r["results"], "error": r.get("error"), "latency_ms": _ms(t0)}
    diagnosis = {"crop": v["crop"], "disease": v["disease"], "confidence": v["confidence"]}
    gen = generation.generate(question or f"{v['crop']} {v['disease']} ke liye upchar",
                              r["results"], disclaimer=(r["tier"] == "fallback_with_disclaimer"),
                              diagnosis=diagnosis)
    return {"diagnosis": v, "tier": r["tier"], "answer": gen["answer"],
            "sources": [{"n": i + 1, "score": h["raw_score"], "source_type": h["source_type"],
                         "citation": h["citation"]} for i, h in enumerate(r["results"])],
            "gen_ms": gen["gen_ms"], "out_tokens": gen["out_tokens"], "latency_ms": _ms(t0)}
