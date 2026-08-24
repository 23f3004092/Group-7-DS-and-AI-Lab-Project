"""Text pipeline (multi-turn) + photo diagnosis."""
import re
import time

from . import ieg, retrieval, generation
from .log import get as _get_log

log = _get_log("pipeline")

try:
    from . import vision
except Exception:
    vision = None

REFUSAL = {
    "non_agricultural": "I can only help with farming and agriculture questions.",
    "restricted": ("That product is restricted/regulated. I can't help source it — please consult "
                   "your local KVK or agriculture officer for approved, safe alternatives and correct use."),
}

OUT_OF_DOMAIN = {
    "en": "I can only help with farming and agriculture questions.",
    "hi": "मैं केवल खेती और कृषि से जुड़े सवालों में मदद कर सकता हूँ।",
    "hinglish": "Main sirf kheti aur agriculture se jude sawaalon mein madad kar sakta hoon.",
}

SESSIONS = {}
_MAX_SESSIONS = 500
_MAX_MESSAGES = 12

_REFERENCE = re.compile(
    r"(?i)(?:\b(?:it|its|that|this|these|those|they|their|same|above|previous|there|them|again|"
    r"dose|kitna|kitni|wahi|uska|uski|uske|iska|iski|iske|inme|isme)\b|"
    r"(?:इसका|इसके|इसमें|उसका|उसके|उसकी|वही|इनमें|पहले|ऊपर|खुराक|कितना|कितनी))")
_QUESTION_START = re.compile(
    r"(?i)^\s*(?:and\s+)?(?:what|how|when|where|which|why|can|should|will|is|are|"
    r"kitna|kitni|kaise|kab|kahan|kya|kaunsa|kaunsi|ab|aur|toh|तो|अब|और|क्या|"
    r"कैसे|कब|कितना|कितनी|कौनसा|कौनसी)\b")
_SMALL_TALK = re.compile(
    r"(?i)^\s*(?:hi|hello|hey|thanks|thank you|ok|okay|bye|namaste|नमस्ते|धन्यवाद)[!. ]*$")
_EXPLICIT_TOPIC = re.compile(
    r"(?i)(?:\b(?:wheat|rice|paddy|maize|corn|sugarcane|mustard|potato|tomato|onion|"
    r"cotton|gram|chickpea|lentil|soybean|bajra|jowar|gehu|gehun|dhan|makka|ganna|"
    r"sarson|aloo|tamatar|pyaz|kapas|mandi|weather|mausam|scheme|yojana|pm-?kisan)\b|"
    r"(?:गेहूं|गेहूँ|धान|चावल|मक्का|गन्ना|सरसों|आलू|टमाटर|प्याज|कपास|मंडी|मौसम|योजना))")
_NAME_INTRO = re.compile(
    r"(?i)(?:\bmy name is\s+|\bmera naam\s+|\bमेरा नाम\s+|"
    r"(?:^|\b(?:hi|hello|hey)\s+)(?:i am|i'm)\s+[a-z])")
_NAME_ASK = re.compile(
    r"(?i)(?:\bwhat(?:'s| is) my name\b|\bdo you (?:know|remember) my name\b|"
    r"\bmera naam kya (?:hai|tha)\b|(?:मेरा नाम क्या है|क्या आपको मेरा नाम याद है))")
_CHAT_EXACT = re.compile(
    r"(?i)^\s*(?:hi|hello|hey|namaste|नमस्ते|thanks|thank you|धन्यवाद|how are you|"
    r"who are you|what is your name|bye|good morning|good evening)[?!. ]*$")
_MEMORY_CHAT = re.compile(
    r"(?i)(?:\b(?:what|which) .{0,30}\bdid i (?:say|tell|mention)\b|"
    r"\bwhat did i (?:say|tell you|mention)\b|\bdo you remember\b|"
    r"\bwhat do you (?:know|remember) about me\b|\bwhat did you (?:say|tell|recommend|suggest)\b|"
    r"\b(?:you|we) (?:said|discussed|recommended|suggested) (?:earlier|before|previously)\b|"
    r"\b(?:earlier|previously|last time) (?:you|i|we)\b|"
    r"मैंने क्या बताया|मैंने .* क्या कहा|आपने .* क्या बताया|क्या आपको याद है|पहले हमने)")
_DOSAGE_QUERY = re.compile(
    r"(?i)(?:\b(?:dose|dosage|quantity|how much|kitna|kitni|matra)\b|"
    r"(?:खुराक|कितना|कितनी|मात्रा))")
_PRODUCT_MENTION = re.compile(
    r"(?i)(?:\b(?:urea|dap|npk|neem oil|sulphur|sulfur|copper|fungicide|insecticide|"
    r"herbicide|pesticide)\b|\b[a-z][a-z0-9-]{3,}(?:azole|mectin|thrin|phos|fos)\b|"
    r"\b\d+(?:\.\d+)?\s*%?\s*(?:ec|wp|sc|wg|sl|sp|gr|cs)\b|(?:नीम तेल|यूरिया|डीएपी))")
_AGRI_SIGNAL = re.compile(
    r"(?i)(?:\b(?:crop|farm|field|leaf|leaves|disease|treatment|control|manage|fungal|"
    r"fungus|pest|insect|spray|fertili[sz]er|"
    r"seed|soil|irrigat|harvest|yield|mandi|weather|scheme|wheat|rice|paddy|maize|"
    r"sugarcane|mustard|potato|tomato|cotton)\b|(?:फसल|खेत|पत्ती|रोग|कीट|दवा|स्प्रे|"
    r"खाद|बीज|मिट्टी|सिंचाई|मंडी|मौसम|गेहूं|धान))")
_SELF_CONTAINED_REQUEST = re.compile(
    r"(?i)(?:\b(?:for|of|in)\s+(?!(?:it|this|that|them|those|same)\b)[a-z][\w-]{2,}|"
    r"\b[a-z][\w-]{2,}(?:\s+[a-z][\w-]{2,}){0,3}\s+"
    r"(?:treatment|control|management|disease|symptoms?|remedy)\b|"
    r"(?:का इलाज|का उपचार|की रोकथाम|ke liye upchar|ka ilaj))")
def _ms(t0):
    return round((time.time() - t0) * 1000)


def _sanitize_history(history):
    if not isinstance(history, (list, tuple)):
        return []
    cleaned = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            message = {"role": item["role"], "content": content.strip()[:4000]}
            if cleaned and cleaned[-1] == message:
                continue
            cleaned.append(message)
    return cleaned[-_MAX_MESSAGES:]


def _is_chat_only(query):
    """Route social and conversation-memory turns around agricultural RAG."""
    text = query or ""
    name_only = bool(_NAME_INTRO.search(text) and not _AGRI_SIGNAL.search(text))
    return bool(_CHAT_EXACT.match(text) or name_only or _NAME_ASK.search(text)
                or _MEMORY_CHAT.search(text))


def _extract_name(history):
    patterns = (
        re.compile(r"(?i)\bmy name is\s+([a-z][a-z .'-]{0,40}?)(?=[,.!?]|$|\s+(?:and|but)\b)"),
        re.compile(r"(?i)\bmera naam\s+([a-z][a-z .'-]{0,40}?)(?=\s+hai\b|[,.!?]|$)"),
        re.compile(r"मेरा नाम\s+([^,.!?]{1,40}?)(?=\s+है|[,.!?]|$)"),
        re.compile(r"(?i)(?:^|\b(?:hi|hello|hey)\s+)(?:i am|i'm)\s+"
                   r"([a-z][a-z'-]{1,30})(?=[,.!?]|$)"),
    )
    for item in reversed(_sanitize_history(history)):
        if item["role"] != "user":
            continue
        for pattern in patterns:
            match = pattern.search(item["content"])
            if match:
                words = match.group(1).strip().split()
                # Natural fillers are not part of the name: "my name is Harliv ok".
                while words and words[-1].lower().strip(".,!?") in {
                    "ok", "okay", "alright", "bro", "bhai", "please", "pls", "ji", "hai"
                }:
                    words.pop()
                return " ".join(words).title() if words else None
    return None


def _name_answer(query, history):
    if not _NAME_ASK.search(query or ""):
        return None
    name = _extract_name(history)
    lang = generation.detect_lang(query)
    if name:
        return ({"hi": f"आपका नाम {name} है।",
                 "hinglish": f"Aapka naam {name} hai.",
                 "en": f"Your name is {name}."}[lang], lang)
    return ({"hi": "आपने अभी तक अपना नाम नहीं बताया है।",
             "hinglish": "Aapne abhi tak apna naam nahi bataya hai.",
             "en": "You have not told me your name yet."}[lang], lang)


def _needs_product_clarification(query, history):
    if not _DOSAGE_QUERY.search(query or "") or _PRODUCT_MENTION.search(query or ""):
        return False
    latest = next((h["content"] for h in reversed(_sanitize_history(history))
                   if h["role"] == "assistant"), "")
    return (not latest or generation._looks_like_refusal(latest)
            or not _PRODUCT_MENTION.search(latest))


def _dose_clarification(query):
    lang = generation.detect_lang(query)
    return ({
        "en": "Which product or spray do you mean? Please share its exact name and formulation so I can give the correct dose.",
        "hi": "आप किस दवा या स्प्रे की खुराक पूछ रहे हैं? सही मात्रा बताने के लिए उसका पूरा नाम और फॉर्मूलेशन बताइए।",
        "hinglish": "Aap kis dawa ya spray ki dose pooch rahe hain? Sahi matra batane ke liye uska exact naam aur formulation batayein.",
    }[lang], lang)


def _blocked_answer(kind, query):
    if kind == "non_agricultural":
        return OUT_OF_DOMAIN[generation.detect_lang(query)]
    return REFUSAL["restricted"]


def _contextualize(history, query):
    history = _sanitize_history(history)
    if not history or not isinstance(query, str) or _SMALL_TALK.match(query):
        return query
    words = query.split()
    follow_up = bool(_REFERENCE.search(query))
    last_assistant = next((item["content"] for item in reversed(history)
                           if item["role"] == "assistant"), "")
    answered_clarification = bool("?" in last_assistant and "?" not in query
                                  and len(words) <= 16)
    self_contained = bool(_EXPLICIT_TOPIC.search(query) or _SELF_CONTAINED_REQUEST.search(query))
    if not follow_up and not self_contained:
        # Only genuinely short/elliptical questions inherit the previous topic.
        # Complete questions such as "what to do for loose smut" stand alone.
        follow_up = bool((_QUESTION_START.search(query) and len(words) <= 5)
                         or len(words) <= 4 or answered_clarification)
    if not follow_up:
        return query
    # Retrieval sees only farmer-authored facts. Previous model answers—especially
    # refusals or wrong guesses—must never steer the next vector search.
    lines = [f"Earlier farmer message: {item['content'][:600]}"
             for item in history if item["role"] == "user"][-3:]
    if _DOSAGE_QUERY.search(query or ""):
        latest = next((item["content"] for item in reversed(history)
                       if item["role"] == "assistant"), "")
        products = []
        if latest and not generation._looks_like_refusal(latest):
            for match in _PRODUCT_MENTION.finditer(latest):
                product = match.group(0)
                if product.lower() not in {p.lower() for p in products}:
                    products.append(product)
        if products:
            lines.append("Product(s) from the prior answer to verify, not trust: "
                         + ", ".join(products[:3]))
    lines.append(f"Farmer's current follow-up: {query}")
    return "Conversation context:\n" + "\n".join(lines)


def _remember(session_id, history, query, answer):
    prior = _sanitize_history(history)
    updated = _sanitize_history(
        prior + [{"role": "user", "content": query},
                 {"role": "assistant", "content": answer}])
    if session_id is not None:
        SESSIONS[session_id] = updated
        if len(SESSIONS) > _MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))
    return updated


def answer_query(query, intent=None, top_k=None, filters=None, live_data=None,
                 skip_retrieval=False, history=None, include_content=False,
                 session_id=None):
    t0 = time.time()
    filters = filters or {}
    supplied_history = _sanitize_history(history)
    stored_history = SESSIONS.get(session_id, []) if session_id is not None else []
    # A supplied history can seed a new in-memory session. Existing server history
    # stays authoritative so clients do not accidentally duplicate turns.
    history = stored_history or supplied_history
    chat_only = _is_chat_only(query)
    contextual_query = query if chat_only else _contextualize(history, query)

    g = ieg.classify(query if chat_only else contextual_query)

    if chat_only:
        recalled = _name_answer(query, history)
        if recalled:
            answer, lang = recalled
            gen_ms, out_tokens = 0, 0
        else:
            gen = generation.generate(query, [], history=history, grounded=False)
            answer, lang = gen["answer"], gen["lang"]
            gen_ms, out_tokens = gen["gen_ms"], gen["out_tokens"]
        new_history = _remember(session_id, history, query, answer)
        return {"tier": "chat", "blocked": False, "answer": answer, "grounded": False,
                "sources": [], "top_score": 0.0,
                "intent": intent or g.get("retrieval_intent") or g.get("intent") or "general",
                "lang": lang, "guardrail_backend": g.get("guardrail_backend"),
                "gen_ms": gen_ms, "out_tokens": out_tokens, "error": None,
                "session_id": session_id, "history": new_history, "latency_ms": _ms(t0)}

    if g["blocked"]:
        log.info("BLOCKED q=%r reason=%s backend=%s",
                 (query or "")[:80], g["block_reason"], g["guardrail_backend"])
        kind = "restricted" if (g["block_reason"] or "").startswith("restricted") else "non_agricultural"
        answer = _blocked_answer(kind, query)
        new_history = _remember(session_id, history, query, answer)
        return {"tier": "blocked", "blocked": True, "block_reason": g["block_reason"],
                "answer": answer, "sources": [], "intent": g["intent"],
                "guardrail_backend": g["guardrail_backend"],
                "session_id": session_id, "history": new_history, "latency_ms": _ms(t0)}

    if _needs_product_clarification(query, history):
        answer, lang = _dose_clarification(query)
        new_history = _remember(session_id, history, query, answer)
        return {"tier": "chat", "blocked": False, "answer": answer, "grounded": False,
                "sources": [], "top_score": 0.0,
                "intent": intent or g["retrieval_intent"], "lang": lang,
                "guardrail_backend": g["guardrail_backend"], "gen_ms": 0, "out_tokens": 0,
                "error": None, "session_id": session_id, "history": new_history,
                "latency_ms": _ms(t0)}

    retr_intent = intent or g["retrieval_intent"]
    if skip_retrieval:
        r = {"tier": "skipped", "results": [], "top_score": 0.0}
    else:
        r = retrieval.search_agri_knowledge(contextual_query,
                                            top_k=top_k, intent=retr_intent, **filters)
    has_context = bool(r["results"]) and r["tier"] not in ("abstain_out_of_scope", "error")

    hints = ieg.external_hints(contextual_query)
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
    if not gen.get("used_context", bool(ctx)):
        ctx, disclaimer = [], False
    if disclaimer:
        answer = answer + "\n\n" + generation.KVK_DISCLAIMER.get(gen["lang"], generation.KVK_DISCLAIMER["en"])

    new_history = _remember(session_id, history, query, answer)
    sources = [{"n": i + 1, "score": h["raw_score"], "source_type": h["source_type"],
                "citation": h["citation"],
                **({"content": h["text"]} if include_content else {})}
               for i, h in enumerate(ctx)]
    return {"tier": r["tier"], "blocked": False, "answer": answer,
            "grounded": bool(ctx or use_live),
            "sources": sources,
            "live_data_used": sorted(use_live.keys()) if use_live else [],
            "top_score": r["top_score"], "intent": retr_intent, "lang": gen["lang"],
            "guardrail_backend": g["guardrail_backend"], "gen_ms": gen["gen_ms"],
            "out_tokens": gen["out_tokens"], "session_id": session_id,
            "history": new_history, "latency_ms": _ms(t0)}


def diagnose_image(image_bytes, question=None, top_k=None, session_id=None, history=None):
    t0 = time.time()
    supplied_history = _sanitize_history(history)
    stored_history = SESSIONS.get(session_id, []) if session_id is not None else []
    history = stored_history or supplied_history
    if vision is None or vision._model is None:
        return {"error": "vision module not loaded"}
    v = vision.predict(image_bytes)
    
    # ALWAYS include the crop and disease in the query, even if the user provided a question.
    crop_disease = f"{v['crop']} {v['disease']}"
    if question:
        q = f"{crop_disease} {question}"
    else:
        q = f"{crop_disease} treatment"
    image_turn = (f"I uploaded a crop image. The vision classification is {v['crop']} "
                  f"{v['disease']}.")
    if question:
        image_turn += f" My question: {question.strip()}"
    
    r = retrieval.search_agri_knowledge(q, top_k=top_k, intent="field_practice")
    if r["tier"] in ("abstain_out_of_scope", "error"):
        message = "Diagnosis done, but no matching treatment found in the knowledge base."
        new_history = _remember(session_id, history, image_turn, message)
        return {"diagnosis": v, "tier": r["tier"], "answer": None,
                "message": message,
                "sources": r["results"], "error": r.get("error"),
                "session_id": session_id, "history": new_history,
                "latency_ms": _ms(t0)}
    diagnosis = {"crop": v["crop"], "disease": v["disease"], "confidence": v["confidence"]}
    gen = generation.generate(question or f"{v['crop']} {v['disease']} ke liye upchar",
                              r["results"], disclaimer=(r["tier"] == "fallback_with_disclaimer"),
                              diagnosis=diagnosis, history=history)
    answer = gen["answer"]
    new_history = _remember(session_id, history, image_turn, answer)
    return {"diagnosis": v, "tier": r["tier"], "answer": gen["answer"],
            "sources": [{"n": i + 1, "score": h["raw_score"], "source_type": h["source_type"],
                         "citation": h["citation"]} for i, h in enumerate(r["results"])],
            "gen_ms": gen["gen_ms"], "out_tokens": gen["out_tokens"],
            "session_id": session_id, "history": new_history, "latency_ms": _ms(t0)}
