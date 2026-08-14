"""Intent / Entity / Guardrail (DistilBERT multilingual, 3 heads).

IMPORTANT: the guardrail ships as MODEL + RULES, never the head alone
(Milestone-5 §9.2: the head alone recalls only 0.571 on adversarial input).
The rule layer here is self-contained and always runs. The model layer is
pluggable: paste your exact model class + forward from notebook 11 /
intent-entity-guard-M5.ipynb where marked, so the .pt checkpoint can load.
If the model can't be loaded, the gateway degrades to RULES + a keyword intent
guess and keeps serving (the RAG path still works).
"""
import re

from . import config as C

_model = None
MODEL_OK = False

# fine intent -> retrieval fusion intent (policy / field_practice / general)
_INTENT_TO_RETRIEVAL = {
    "nutrition_fertilizer": "field_practice",
    "cultivation_practice": "field_practice",
    "disease_pest":         "field_practice",
    "post_harvest_storage": "field_practice",
    "specialty_other":      "general",
    "general":              "general",
    "non_agri":             "general",
}

# --- RULE LAYER (always on) -------------------------------------------------
# Restricted / banned or heavily-regulated substances: block procurement-style
# queries with a safety message instead of answering "where to buy".
_RESTRICTED = ["monocrotophos", "phorate", "methyl parathion", "phosphamidon",
               "carbofuran", "endosulfan", "paraquat"]
_BUY_INTENT = re.compile(r"\b(buy|purchase|kahan\s*se|kaha\s*se|khareed|kharid|order|price of)\b", re.I)

# Obvious non-agriculture topics the model must never route into RAG.
_NON_AGRI = re.compile(
    r"\b(movie|cricket|football|bitcoin|stock|flight|hotel|bike|motorcycle|laptop|"
    r"iphone|visa|passport|loan\s+app|dating|song|lyrics)\b", re.I)


def rule_block(text: str):
    t = text or ""
    for sub in _RESTRICTED:
        if sub in t.lower() and _BUY_INTENT.search(t):
            return True, f"restricted_substance:{sub}"
    if _NON_AGRI.search(t):
        return True, "non_agricultural"
    return False, None


# --- external-data hints ----------------------------------------------------
# The main app fetches mandi prices / weather / yield itself. These patterns let
# it (via /classify) know which of those a query is asking for, so it can fetch
# the right thing and inject it into /query as live_data. English + Hindi + Hinglish.
_HINT_PATTERNS = {
    "mandi_prices": re.compile(
        r"\b(mandi|bhav|bhaav|price|prices|rate|rates|daam|dam|bhaw|market|msp|"
        r"quintal|kimat|keemat|bech|bechna|sell|selling)\b", re.I),
    "weather": re.compile(
        r"\b(weather|mausam|mosam|rain|rains|rainfall|barish|baarish|forecast|"
        r"temperature|humidity|climate|storm|baadal|aandhi)\b", re.I),
    "yield": re.compile(
        r"\b(yield|paidawar|paidavar|utpadan|production|expected\s+yield|"
        r"kitni\s+(fasal|paidawar)|estimate|estimated)\b", re.I),
}


def external_hints(text: str):
    """Return which external data sources a query likely needs, e.g. ['mandi_prices']."""
    t = text or ""
    return [k for k, p in _HINT_PATTERNS.items() if p.search(t)]


def _keyword_intent(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(scheme|subsidy|yojana|loan|insurance|bima|kcc card|pm[- ]?kisan)\b", t):
        return "field_practice"  # scheme questions -> fusion handles pdf weighting via 'policy' below
    if re.search(r"\b(rust|blight|pest|disease|rog|keet|dawa|spray|fungicid)\b", t):
        return "disease_pest"
    if re.search(r"\b(urea|dap|npk|fertiliz|khaad|nutrient|dose|kg\/)\b", t):
        return "nutrition_fertilizer"
    return "general"


# --- MODEL LAYER (pluggable) ------------------------------------------------
def load():
    """Try to load the 3-head DistilBERT checkpoint. Never fatal."""
    global _model, MODEL_OK
    if not C.HAS_IEG:
        print("[ieg] no checkpoint found — running RULES + keyword intent only.")
        return
    try:
        # ---------------------------------------------------------------
        # TODO: paste your model class + load code from notebook 11 here, e.g.:
        #   import torch
        #   from your_ieg_module import IntentEntityGuardModel   # the class you trained
        #   _model = IntentEntityGuardModel(...);
        #   _model.load_state_dict(torch.load(f"{C.IEG_DIR}/intent_entity_guardrail_model.pt",
        #                                     map_location="cpu"))
        #   _model.eval()
        #   MODEL_OK = True
        # ---------------------------------------------------------------
        raise NotImplementedError("paste IEG model class from notebook 11")
    except Exception as e:
        print(f"[ieg] model not loaded ({e}). RULES + keyword intent only — RAG still works.")
        MODEL_OK = False


def classify(text: str) -> dict:
    """Returns intent + guardrail decision. Guardrail = model OR rules."""
    blocked, reason = rule_block(text)

    if MODEL_OK:
        # ---------------------------------------------------------------
        # TODO: run the model, combine with rules:
        #   intent, model_blocked, entities = _model.predict(text)
        #   blocked = blocked or model_blocked
        # ---------------------------------------------------------------
        intent = _keyword_intent(text)   # placeholder until the block above is filled
        entities = {}
    else:
        intent = _keyword_intent(text)
        entities = {}

    return {
        "intent": intent,
        "retrieval_intent": _INTENT_TO_RETRIEVAL.get(intent, "general"),
        "blocked": blocked,
        "block_reason": reason,
        "entities": entities,
        "guardrail_backend": "model+rules" if MODEL_OK else "rules-only",
    }
