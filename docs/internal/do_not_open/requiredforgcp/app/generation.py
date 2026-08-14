"""Distilled generator = merged Gemma (or base + LoRA) in 4-bit, on GPU.

Multi-turn aware, language-controlled, grounded. Key behaviours:
  * Reply language is FORCED to match the farmer's question (Hindi / English /
    Hinglish) and stated at the very END of the prompt — the context is mostly
    Hindi and the model must NOT copy its language.
  * Context items are treated as reference records: the question text inside a
    KCC Q&A chunk is NOT the farmer's current question (only its answer is knowledge).
  * Prior conversation turns are passed as chat history so follow-ups resolve.
See RUNBOOK §4 for how to edit the prompt.
"""
import os
import re
import time

import torch

from . import config as C

_tok = None
_model = None


def load():
    """Prefer a MERGED model (fast, no PEFT); else base Gemma + LoRA adapter. 4-bit."""
    global _tok, _model
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForImageTextToText, BitsAndBytesConfig)

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible. Set LLM to a GPU host, or use CPU build.")

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

    kw = dict(device_map="auto", token=token,
              quantization_config=BitsAndBytesConfig(
                  load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True))

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
    return _tok, _model


# ---------------------------------------------------------------------------
# Language handling — reply MUST match the query, not the (mostly Hindi) context
# ---------------------------------------------------------------------------
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
    """Returns 'hi', 'hinglish', or 'en' for the farmer's query."""
    if _deva_ratio(text) > 0.30:
        return "hi"
    if len(_HINGLISH.findall(text or "")) >= 2:
        return "hinglish"
    return "en"


LANG_NAME = {
    "hi": "Hindi (Devanagari script)",
    "en": "English",
    "hinglish": "Hinglish (Hindi words written in Roman/Latin script)",
}
KVK_DISCLAIMER = {
    "en": "Please verify with your local KVK before applying.",
    "hi": "कृपया उपयोग से पहले अपने नजदीकी केवीके से पुष्टि कर लें।",
    "hinglish": "Upyog se pehle apne nazdeeki KVK se confirm kar lein.",
}
ABSTAIN_MSG = {
    "en": ("This is outside the agricultural knowledge base I can answer from. Please ask about "
           "crops, pests, fertilisers, or Uttar Pradesh government farm schemes."),
    "hi": ("यह प्रश्न मेरे कृषि ज्ञान आधार से बाहर है। कृपया फसल, कीट, उर्वरक या सरकारी कृषि योजनाओं के बारे में पूछें।"),
    "hinglish": ("Ye sawaal krishi knowledge base se bahar hai. Kripya fasal, keet, khaad ya "
                 "sarkari krishi yojana ke baare mein poochein."),
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_RULES = (
    "You are FarmerVision, an agricultural advisory assistant for farmers in Uttar Pradesh, India.\n"
    "RULES:\n"
    "1. Answer ONLY from the numbered CONTEXT and any LIVE DATA below. Never use outside knowledge.\n"
    "2. The CONTEXT items are reference records — many are past Kisan Call Centre Q&A written in "
    "Hindi. The question text inside a context item is NOT the farmer's current question; use only "
    "its factual answer content as knowledge.\n"
    "3. Do NOT copy the CONTEXT's language. Your reply language is stated at the very END of this "
    "message and must match the farmer's question.\n"
    "4. Never invent a dosage, price, date, or scheme name. If a specific figure is not in the "
    "context or live data, say it is not available. LIVE DATA (mandi price / weather / yield) is "
    "current and authoritative — use those exact values.\n"
    "5. Cite the context numbers you used, like [1] or [2].\n"
    "6. In a multi-turn chat, use the earlier turns to resolve references (e.g. 'it', 'that dose', "
    "'wahi'), but still ground every fact in the CONTEXT / LIVE DATA of THIS turn.\n"
    "7. Be concise and practical: 2-4 sentences, no preamble, no repetition."
)
OUTPUT_FORMAT = ("FORMAT: one short paragraph of practical advice, then a final line "
                 "'Sources: [n]' listing the context numbers you used.")

_LIVE_LABELS = {
    "mandi_prices": "Mandi prices", "mandi": "Mandi prices", "market": "Market prices",
    "prices": "Prices", "weather": "Weather", "forecast": "Weather forecast",
    "yield": "Predicted yield", "yield_prediction": "Predicted yield",
}


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


def build_messages(query, results, disclaimer=False, diagnosis=None, live_data=None, history=None):
    """Chat messages: prior turns (if any) + a final user turn carrying the rules,
    context, live data, question, and the FORCED reply-language instruction (last)."""
    lang = detect_lang(query)
    parts = [SYSTEM_RULES]
    if diagnosis:
        conf = diagnosis.get("confidence")
        cs = f" ({round(float(conf) * 100)}% confidence)" if conf is not None else ""
        parts.append(f"PHOTO DIAGNOSIS: likely {diagnosis.get('crop','')} "
                     f"{diagnosis.get('disease','')}{cs}. Treat as likely, not certain.")
    if live_data:
        parts.append("LIVE DATA (authoritative, current):\n" + _format_live_data(live_data))
    parts.append("CONTEXT:\n" + _format_context(results))
    parts.append("FARMER'S QUESTION:\n" + query)
    parts.append(OUTPUT_FORMAT)
    parts.append(f"### WRITE YOUR ENTIRE ANSWER IN {LANG_NAME[lang].upper()}. ###\n"
                 f"Even though the context above is mostly in Hindi, you MUST answer in "
                 f"{LANG_NAME[lang]}, matching the farmer's question — do not switch languages.")
    current = "\n\n".join(parts)

    msgs = []
    if history:
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": current})
    return msgs


@torch.inference_mode()
def generate(query, results, disclaimer=False, diagnosis=None, live_data=None,
             history=None, max_new_tokens=256):
    device = next(_model.parameters()).device
    messages = build_messages(query, results, disclaimer, diagnosis, live_data, history)
    try:
        enc = _tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
    except Exception:
        # If the template rejects the history sequence, retry single-turn.
        messages = build_messages(query, results, disclaimer, diagnosis, live_data, None)
        enc = _tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else None

    t0 = time.time()
    out = _model.generate(input_ids=input_ids, attention_mask=attn,
                          max_new_tokens=max_new_tokens, do_sample=False,
                          temperature=None, top_p=None, top_k=None,
                          pad_token_id=_tok.eos_token_id)
    gen_ms = (time.time() - t0) * 1000
    text = _tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return {"answer": text, "gen_ms": round(gen_ms),
            "out_tokens": int(out.shape[1] - input_ids.shape[1]), "lang": detect_lang(query)}
