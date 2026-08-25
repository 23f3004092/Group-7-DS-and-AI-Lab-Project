"""Distilled generator = merged Gemma (or base + LoRA) in BF16, on GPU.
Multi-turn aware, language-controlled, grounded."""
import os
import re
import threading
import time

import torch

from . import config as C
from .log import get as _get_log

log = _get_log("generation")
_tok = None
_model = None

# The distilled model sometimes over-refuses ("I don't have info") even when the
# context is clearly relevant (M5 known limitation). Detect that so we can retry.
_REFUSAL_PAT = re.compile(
    r"(i (don'?t|do not) have|i (don'?t|do not) know|not available|no (information|data|details)"
    r"|outside .{0,25}(knowledge|scope|database)|cannot (help|answer|provide)|can'?t (help|answer)"
    r"|unable to (help|answer|provide)|insufficient (information|context|data)"
    r"|जानकारी नहीं|उपलब्ध नहीं|पता नहीं|मेरे पास .{0,25}नहीं|मुझे .{0,25}नहीं"
    r"|jaa?nkaa?ri nahi|pata nahi|nahi pata|maloom nahi|nahi maloom"
    r"|mujhe .{0,30}(nahi|nahin)|information nahi)",
    re.I)

# Clarification questions are NOT refusals — they are valid responses when the
# query is ambiguous (e.g., missing crop name). We exclude them from the retry loop.
_CLARIFICATION_PAT = re.compile(
    r"(?i)(kripya|please)\s*(batayein|tell|specify|batao)\s*(?:kis|which)\s*(?:fasal|crop)",
    re.I)

# Prompt scaffolding that must never appear in a reply (the model occasionally echoes the
# trailing language directive or a section header).
_WRITE_DIRECTIVE = re.compile(r"#{2,}\s*WRITE YOUR ENTIRE ANSWER IN[^\n]*", re.I)
_SCAFFOLD_PAT = re.compile(
    r"(?im)^\s*(#{2,}.*|(FORMAT|CONTEXT|LIVE DATA|OUTPUT|HOW TO ANSWER|"
    r"FARMER'?S (QUESTION|MESSAGE))\s*:.*)$")


def _looks_like_refusal(text: str) -> bool:
    t = (text or "").strip()
    return len(t) < 400 and bool(_REFUSAL_PAT.search(t)) and not bool(_CLARIFICATION_PAT.search(t))


def _clean_answer(text: str) -> str:
    """Strip any leaked prompt scaffolding (### WRITE ... ###, section headers) from a reply."""
    if not text:
        return text
    text = _WRITE_DIRECTIVE.sub("", text)
    text = _SCAFFOLD_PAT.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load():
    global _tok, _model
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForImageTextToText)
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible.")
    cap = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    token = C.HF_TOKEN
    use_merged = os.path.isdir(C.MERGED_MODEL_DIR) and os.path.isfile(
        os.path.join(C.MERGED_MODEL_DIR, "config.json"))
    source = C.MERGED_MODEL_DIR if use_merged else C.GEN_MODEL_ID
    cfg = AutoConfig.from_pretrained(source, token=token)
    multimodal = (hasattr(cfg, "vision_config")
                  or "text_config" in (getattr(cfg, "sub_configs", None) or {}))
    order = ([AutoModelForImageTextToText, AutoModelForCausalLM] if multimodal
             else [AutoModelForCausalLM, AutoModelForImageTextToText])
    # The merged 4B model fits on an L4. Loading it directly in BF16 avoids the
    # per-token BitsAndBytes NF4 dequantization path. Force the complete model
    # onto GPU 0 so device_map cannot silently offload layers to CPU.
    attn_impl = os.environ.get("GEN_ATTN_IMPLEMENTATION", "sdpa")
    kw = dict(device_map={"": 0}, token=token, torch_dtype=compute_dtype,
              attn_implementation=attn_impl, low_cpu_mem_usage=True)
    base, last_err = None, None
    for cls in order:
        try:
            base = cls.from_pretrained(source, **kw); break
        except Exception as e:
            last_err = e
    if base is None:
        raise RuntimeError(f"could not load {source}: {last_err}")
    if use_merged:
        _tok = AutoTokenizer.from_pretrained(C.MERGED_MODEL_DIR, token=token)
        _model = base
        print("[gen] loaded MERGED model:", C.MERGED_MODEL_DIR)
    else:
        from peft import PeftModel
        tok_src = C.ADAPTER_DIR if os.path.exists(os.path.join(C.ADAPTER_DIR, "tokenizer_config.json")) else C.GEN_MODEL_ID
        _tok = AutoTokenizer.from_pretrained(tok_src, token=token)
        _model = PeftModel.from_pretrained(base, C.ADAPTER_DIR) if os.path.isdir(C.ADAPTER_DIR) else base
        print("[gen] loaded base + LoRA adapter")
    _model.eval()
    _model.config.use_cache = True
    if getattr(_model, "generation_config", None) is not None:
        _model.generation_config.use_cache = True
    print(f"[gen] device={next(_model.parameters()).device} dtype={compute_dtype} "
          f"attention={attn_impl} gpu={torch.cuda.get_device_name(0)}")
    return _tok, _model


@torch.inference_mode()
def warmup():
    """Initialize CUDA kernels and the KV-cache path before the first real query."""
    if _model is None or _tok is None:
        raise RuntimeError("generation model is not loaded")
    device = next(_model.parameters()).device
    msgs = [{"role": "user", "content": "Reply with OK."}]
    enc = _tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(device)
    am = enc.get("attention_mask")
    am = am.to(device) if am is not None else None
    t0 = time.time()
    _model.generate(input_ids=ids, attention_mask=am, max_new_tokens=1,
                    do_sample=False, use_cache=True,
                    pad_token_id=_tok.eos_token_id)
    torch.cuda.synchronize()
    print(f"[gen] warmup complete in {round((time.time() - t0) * 1000)} ms")


_HINGLISH = re.compile(
    r"(?<!\w)(?:ki|ka|ke|kya|kaise|kaun|kaunsi|hai|hain|nahi|nhi|kitna|kitni|dawa|dawai|"
    r"fasal|kheti|khet|bhav|mandi|paidawar|barish|mausam|batao|bataye|karein|kare|chahiye|"
    r"ilaj|upchar|se|me|mein|ko|par|aur|kab|kahan|kyun)(?!\w)", re.I)


def _deva_ratio(s):
    letters = [c for c in (s or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def detect_lang(text):
    if _deva_ratio(text) > 0.30:
        return "hi"
    if len(_HINGLISH.findall(text or "")) >= 2:
        return "hinglish"
    return "en"


LANG_NAME = {"hi": "Hindi (Devanagari script)", "en": "English",
             "hinglish": "Hinglish (Hindi words written in Roman/Latin script)"}
KVK_DISCLAIMER = {"en": "Please verify with your local KVK before applying.",
                  "hi": "कृपया उपयोग से पहले अपने नजदीकी केवीके से पुष्टि कर लें।",
                  "hinglish": "Upyog se pehle apne nazdeeki KVK se confirm kar lein."}
ABSTAIN_MSG = {
    "en": ("This is outside the agricultural knowledge base I can answer from. Please ask about "
           "crops, pests, fertilisers, or Uttar Pradesh government farm schemes."),
    "hi": ("यह प्रश्न मेरे कृषि ज्ञान आधार से बाहर है। कृपया फसल, कीट, उर्वरक या सरकारी कृषि योजनाओं के बारे में पूछें।"),
    "hinglish": ("Ye sawaal krishi knowledge base se bahar hai. Kripya fasal, keet, khaad ya "
                 "sarkari krishi yojana ke baare mein poochein.")}

GROUNDED_RULES = (
    "You are FarmerVision, a warm, knowledgeable agricultural advisor for farmers in Uttar Pradesh, "
    "India. Answer the FARMER'S QUESTION directly and helpfully using the LIVE DATA and CONTEXT below.\n"
    "HOW TO ANSWER:\n"
    "1. Answer the actual question that was asked. If it is about prices / MSP / mandi, weather, or "
    "yield, the LIVE DATA has the current answer — state those exact values. If it is about agronomy or goverenment policy "
    "(diseases, pests, dosage, fertiliser, cultivation, government schemes), use the CONTEXT records.\n"
    "2. Do NOT reply that you lack information or that it is outside your knowledge when the answer is "
    "present in the LIVE DATA or CONTEXT — use it and give answer. But if a CONTEXT item is NOT relevant to the "
    "question, ignore it; never answer a different question than the one that was asked.\n"
    "3. Don't fabricate specific figures (dosages, prices, dates, scheme names) that are not in the "
    "LIVE DATA or CONTEXT; you MAY add general agronomic explanation to make advice clear and complete.\n"
    "4. The CONTEXT items are reference records — many are past Kisan Call Centre Q&A written in "
    "Hindi. The 'question' text inside a context item is NOT the farmer's current question; use its "
    "answer / advice content as knowledge.\n"
    "5. LIVE DATA (mandi price / weather / yield) is current and fresh — use those exact values to answer Farmer's Question involving mandi price/weather/yield.\n"
    "6. Do NOT copy the CONTEXT's language. Your reply language is stated at the very END and must "
    "match the farmer's question.\n"
    "7. Cite the context numbers you actually used, like [1] or [2] (only when you used CONTEXT). Never cite context numbers if you didn't use context to generate final answer.\n"
    "8. In a multi-turn chat, use earlier turns to resolve references (e.g. 'it', 'that dose', 'wahi').\n"
    "9. Be thorough, clear and practical — explain what to do, how, and why. Use short paragraphs or "
    "bullet points. Avoid needless repetition and filler.\n"
    "10. If the Farmer's question is broad and asking for lets say Different Diseases in wheat crop, then use all the retrived chunks to answer these kinda queries and extract name of diseases required to answer.\n"
   
)
CHAT_RULES = (
    "You are FarmerVision, a warm, friendly agricultural advisory assistant for farmers in Uttar "
    "Pradesh, India. No reference records were retrieved for this message — just be genuinely helpful.\n"
    "- If the farmer greets you or makes small talk (hello, hi, namaste, thanks, how are you), greet "
    "them warmly, briefly introduce what you can help with (crop diseases and pests, fertiliser and "
    "cultivation advice, mandi prices, weather, and government farm schemes), and invite their question.\n"
    "- If they ask a general farming question, answer helpfully and thoroughly from your general "
    "knowledge, in a clear, well-structured way (short paragraphs or a few bullet points).\n"
    "- Do NOT invent a specific dosage, market price, exact date, or official scheme name/amount. For "
    "those precise details, gently suggest they ask specifically or confirm with their local KVK / "
    "agriculture officer.\n"
    "- Reply in the same language as the farmer (stated at the very END of this message).\n"
    "Be friendly, natural, and complete — never reply with just one terse line.")
OUTPUT_FORMAT = ("FORMAT: Give a clear, helpful answer — a few short paragraphs or bullet points as "
                 "needed. End with a line 'Sources: [n]' listing the context numbers you used.")
_LIVE_LABELS = {"mandi_prices": "Mandi prices", "mandi": "Mandi prices", "market": "Market prices",
                "prices": "Prices", "weather": "Weather", "forecast": "Weather forecast",
                "yield": "Predicted yield", "yield_prediction": "Predicted yield"}


def _format_context(results, max_chunks=5, char_cap=700):
    lines = []
    for i, h in enumerate(results[:max_chunks], 1):
        txt = re.sub(r"\s+", " ", h.get("text", "")).strip()[:char_cap]
        lines.append(f"[{i}] {txt}")
    return "\n\n".join(lines) if lines else "(no context)"


def _format_live_data(live_data):
    import json
    out = []
    for k, v in live_data.items():
        label = _LIVE_LABELS.get(k, str(k).replace("_", " ").title())
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        out.append(f"- {label}: {v}")
    return "\n".join(out)


def build_messages(query, results, disclaimer=False, diagnosis=None, live_data=None,
                   history=None, grounded=True):
    lang = detect_lang(query)
    parts = [GROUNDED_RULES if grounded else CHAT_RULES]
    if diagnosis:
        conf = diagnosis.get("confidence")
        cs = f" ({round(float(conf) * 100)}% confidence)" if conf is not None else ""
        parts.append(f"PHOTO DIAGNOSIS: likely {diagnosis.get('crop','')} "
                     f"{diagnosis.get('disease','')}{cs}. Treat as likely, not certain.")
    if live_data:
        parts.append("LIVE DATA (authoritative, current):\n" + _format_live_data(live_data))
    if grounded:
        if results:
            parts.append("CONTEXT:\n" + _format_context(results))
        parts.append("FARMER'S QUESTION:\n" + query)
        if results:
            parts.append(OUTPUT_FORMAT)
        else:
            parts.append("FORMAT: Answer the question directly using the LIVE DATA above. "
                         "Do not add a 'Sources' line.")
        parts.append(f"### WRITE YOUR ENTIRE ANSWER IN {LANG_NAME[lang].upper()}. ###\n"
                     f"Match the farmer's question language exactly — do not switch languages.")
    else:
        parts.append("FARMER'S MESSAGE:\n" + query)
        parts.append(f"### WRITE YOUR ENTIRE ANSWER IN {LANG_NAME[lang].upper()}. ###\n"
                     f"Match the farmer's language exactly — do not switch languages.")
    current = "\n\n".join(parts)
    msgs = []
    if history:
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": current})
    return msgs


def _encode_messages(messages, query, results, disclaimer, diagnosis, live_data,
                     grounded):
    """Apply the same chat template for regular and streaming generation."""
    try:
        enc = _tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
    except Exception as e:
        log.warning("chat_template failed (%s); retrying without history", e)
        enc = _tok.apply_chat_template(
            build_messages(query, results, disclaimer, diagnosis, live_data, None, grounded),
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
    device = next(_model.parameters()).device
    ids = enc["input_ids"].to(device)
    am = enc.get("attention_mask")
    am = am.to(device) if am is not None else None
    return ids, am


def _generate_kwargs(ids, am, max_new_tokens, min_new):
    """Single source of truth for quality-sensitive decoding settings."""
    return dict(input_ids=ids, attention_mask=am,
                max_new_tokens=max_new_tokens, min_new_tokens=min_new,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=50,
                repetition_penalty=1.1, use_cache=True,
                pad_token_id=_tok.eos_token_id)


@torch.inference_mode()
def generate(query, results, disclaimer=False, diagnosis=None, live_data=None,
             history=None, grounded=True, max_new_tokens=512):
    lang = detect_lang(query)

    def _run(msgs, min_new=8):
        ids, am = _encode_messages(msgs, query, results, disclaimer, diagnosis,
                                   live_data, grounded)
        # Sampling (temperature 0.8), NOT greedy: greedy deterministically collapses to the
        # "safe" refusal token even with good context. min_new_tokens stops an instant EOS,
        # repetition_penalty avoids loops.
        o = _model.generate(**_generate_kwargs(ids, am, max_new_tokens, min_new))
        txt = _clean_answer(_tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True))
        return txt, int(ids.shape[1]), int(o.shape[1] - ids.shape[1])

    n_ctx = len(results or [])
    messages = build_messages(query, results, disclaimer, diagnosis, live_data, history, grounded)
    log.info("generate: grounded=%s lang=%s n_ctx=%d live=%s q=%r",
             grounded, lang, n_ctx, bool(live_data), (query or "")[:80])
    t0 = time.time()
    text, n_in, n_out = _run(messages)
    log.info("  -> in_tokens=%d out_tokens=%d chars=%d ans=%r", n_in, n_out, len(text), text[:160])

    # Over-refusal guard: if the model refuses despite relevant context, retry once
    # with an explicit anti-refusal instruction + light sampling.
    # EXCEPTION: If the answer is a clarification question (e.g., asking for the crop name),
    # we treat it as valid and DO NOT retry.
    if grounded and n_ctx and _looks_like_refusal(text):
        log.warning("  REFUSAL despite %d context chunks — retrying forcefully.", n_ctx)
        forced = build_messages(query, results, disclaimer, diagnosis, live_data, history, grounded)
        forced[-1]["content"] += ("\n\nIMPORTANT: The CONTEXT above IS relevant to this question. "
                                  "Answer it directly and helpfully using that context. Do NOT reply "
                                  "that you lack information or that it is outside your knowledge.")
        text2, _, n_out2 = _run(forced, min_new=24)
        log.info("  retry -> chars=%d ans=%r", len(text2), text2[:160])
        if text2 and not _looks_like_refusal(text2):
            text, n_out = text2, n_out2

    if not text:                                     # never return an empty answer
        log.warning("  empty answer — retrying.")
        text2, _, n_out2 = _run(messages, min_new=24)
        if text2:
            text, n_out = text2, n_out2

    gen_ms = round((time.time() - t0) * 1000)
    log.info("generate done: out_tokens=%d gen_ms=%d", n_out, gen_ms)
    log.info("  FULL ANSWER: %s", text)
    if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
        log.debug("  PROMPT >>>\n%s\n<<<", messages[-1]["content"][:4000])
    return {"answer": text, "gen_ms": gen_ms, "out_tokens": n_out, "lang": lang}


def generate_stream(query, results, disclaimer=False, diagnosis=None, live_data=None,
                    history=None, grounded=True, max_new_tokens=512):
    """Yield decoded deltas, then the same cleaned final generation payload.

    Streaming is opt-in. If the existing refusal guard retries, a ``reset``
    event tells the client to clear the draft before replacement tokens arrive.
    The final event is always authoritative and contains the cleaned answer.
    """
    from transformers import TextIteratorStreamer

    lang = detect_lang(query)
    n_ctx = len(results or [])
    messages = build_messages(query, results, disclaimer, diagnosis, live_data,
                              history, grounded)
    t0 = time.time()

    def _attempt(msgs, min_new=8):
        ids, am = _encode_messages(msgs, query, results, disclaimer, diagnosis,
                                   live_data, grounded)
        streamer = TextIteratorStreamer(_tok, skip_prompt=True,
                                        skip_special_tokens=True, timeout=180.0)
        kw = _generate_kwargs(ids, am, max_new_tokens, min_new)
        kw["streamer"] = streamer
        state = {"output": None, "error": None}

        def _worker():
            try:
                with torch.inference_mode():
                    state["output"] = _model.generate(**kw)
            except Exception as e:  # unblock the iterator and surface the error
                state["error"] = e
                streamer.end()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        pieces = []
        for piece in streamer:
            if piece:
                pieces.append(piece)
                yield {"type": "delta", "text": piece}
        worker.join()
        if state["error"] is not None:
            raise state["error"]
        output = state["output"]
        n_out = int(output.shape[1] - ids.shape[1])
        return _clean_answer("".join(pieces)), int(ids.shape[1]), n_out

    text, n_in, n_out = yield from _attempt(messages)

    if grounded and n_ctx and _looks_like_refusal(text):
        log.warning("  REFUSAL despite %d context chunks — streaming retry.", n_ctx)
        yield {"type": "reset", "reason": "refusal_retry"}
        forced = build_messages(query, results, disclaimer, diagnosis, live_data,
                                history, grounded)
        forced[-1]["content"] += ("\n\nIMPORTANT: The CONTEXT above IS relevant to this question. "
                                  "Answer it directly and helpfully using that context. Do NOT reply "
                                  "that you lack information or that it is outside your knowledge.")
        text2, _, n_out2 = yield from _attempt(forced, min_new=24)
        if text2 and not _looks_like_refusal(text2):
            text, n_out = text2, n_out2

    if not text:
        yield {"type": "reset", "reason": "empty_retry"}
        text2, _, n_out2 = yield from _attempt(messages, min_new=24)
        if text2:
            text, n_out = text2, n_out2

    gen_ms = round((time.time() - t0) * 1000)
    log.info("stream generate done: in_tokens=%d out_tokens=%d gen_ms=%d",
             n_in, n_out, gen_ms)
    yield {"type": "generation_final",
           "data": {"answer": text, "gen_ms": gen_ms,
                    "out_tokens": n_out, "lang": lang}}
