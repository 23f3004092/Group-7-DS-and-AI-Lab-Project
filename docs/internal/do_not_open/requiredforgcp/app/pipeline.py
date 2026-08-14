"""End-to-end pipelines:
  * answer_query   : text -> guardrail -> (contextualized) retrieve -> generate,
                     multi-turn aware (client `history` OR server `session_id`).
  * diagnose_image : photo -> classify -> retrieve -> grounded advice (single-shot).
"""
import time

from . import ieg, retrieval, generation

try:
    from . import vision
except Exception:                       # optional module
    vision = None

# In-memory conversation store for server-managed sessions. Lost on gateway
# restart; fine for a single-instance demo. Client-managed `history` needs none.
SESSIONS = {}
_MAX_MSGS = 8                           # keep last 8 messages (~4 turns) per conversation
_MAX_SESSIONS = 500

REFUSAL = {
    "non_agricultural": "I can only help with farming and agriculture questions.",
    "restricted": ("That product is restricted/regulated. I can't help source it — please consult "
                   "your local KVK or agriculture officer for approved, safe alternatives and correct use."),
}


def _ms(t0):
    return round((time.time() - t0) * 1000)


def _contextualize(history, query):
    """For follow-ups, prepend the last couple of user turns so retrieval has topic."""
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


def answer_query(query: str, intent: str = None, top_k: int = None, filters: dict = None,
                 live_data: dict = None, skip_retrieval: bool = False,
                 history: list = None, session_id: str = None) -> dict:
    """`history` = [{"role":"user"/"assistant","content":...}, ...] (client-managed) OR
    pass `session_id` to let the gateway remember the conversation. The response echoes
    `session_id` and the updated `history` so the client can persist it."""
    t0 = time.time()
    filters = filters or {}
    if session_id is not None:                       # server memory wins if a session id is given
        history = SESSIONS.get(session_id, [])
    history = list(history or [])

    # 1) Guardrail + intent
    g = ieg.classify(query)
    if g["blocked"]:
        kind = "restricted" if (g["block_reason"] or "").startswith("restricted") else "non_agricultural"
        return {"tier": "blocked", "blocked": True, "block_reason": g["block_reason"],
                "answer": REFUSAL[kind], "sources": [], "intent": g["intent"],
                "guardrail_backend": g["guardrail_backend"],
                "session_id": session_id, "history": history, "latency_ms": _ms(t0)}

    retr_intent = intent or g["retrieval_intent"]

    # 2) Retrieve (contextualized with recent turns; skippable for pure live-data)
    if skip_retrieval:
        r = {"tier": "skipped", "results": [], "top_score": 0.0}
    else:
        r = retrieval.search_agri_knowledge(_contextualize(history, query),
                                            top_k=top_k, intent=retr_intent, **filters)
    has_context = bool(r["results"]) and r["tier"] not in ("abstain_out_of_scope", "error")

    # 3) Abstain only if there's neither context nor injected live data
    if not has_context and not live_data:
        lang = generation.detect_lang(query)
        return {"tier": r["tier"], "blocked": False, "answer": None,
                "message": generation.ABSTAIN_MSG.get(lang, generation.ABSTAIN_MSG["en"]),
                "sources": r["results"], "top_score": r["top_score"], "intent": retr_intent,
                "error": r.get("error"), "session_id": session_id, "history": history,
                "latency_ms": _ms(t0)}

    # 4) Generate (grounded + language-forced + multi-turn)
    disclaimer = (r["tier"] == "fallback_with_disclaimer")
    gen = generation.generate(query, r["results"], disclaimer=disclaimer,
                              live_data=live_data, history=history)
    answer = gen["answer"]
    if disclaimer:                                   # appended by rule, not by the model
        answer = answer + "\n\n" + generation.KVK_DISCLAIMER.get(gen["lang"], generation.KVK_DISCLAIMER["en"])

    new_history = _remember(session_id, history, query, answer)
    return {"tier": r["tier"], "blocked": False, "answer": answer,
            "sources": [{"n": i + 1, "score": h["raw_score"], "source_type": h["source_type"],
                         "citation": h["citation"]} for i, h in enumerate(r["results"])],
            "live_data_used": sorted(live_data.keys()) if live_data else [],
            "top_score": r["top_score"], "intent": retr_intent, "lang": gen["lang"],
            "guardrail_backend": g["guardrail_backend"], "gen_ms": gen["gen_ms"],
            "out_tokens": gen["out_tokens"], "session_id": session_id,
            "history": new_history, "latency_ms": _ms(t0)}


def diagnose_image(image_bytes: bytes, question: str = None, top_k: int = None) -> dict:
    """Photo -> disease -> grounded, cited treatment advice (single-shot)."""
    t0 = time.time()
    if vision is None or vision._model is None:      # noqa: SLF001
        return {"error": "vision module not loaded"}

    v = vision.predict(image_bytes)
    q = question or f"{v['crop']} {v['disease']} treatment control medicine dose"
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
