"""Intent / Entity / Guardrail (DistilBERT multilingual, 3 heads).

The guardrail ships as MODEL + RULES (Milestone-5 §9.2: the head alone recalls
only 0.571 on adversarial input, so the rule layer below ALWAYS runs too). The
model layer mirrors scripts/run_e2e_eval.py (IEGModel / load_ieg / ieg_run) and
notebook 11 / intent-entity-guard-M5.ipynb exactly, so the checkpoint loads as-is.
If the checkpoint can't be loaded, the gateway degrades to RULES + a keyword
intent guess and keeps serving (the RAG path still works).

Artifacts (staged to /artifacts/ieg by 04_vm_setup.sh, pulled from the GCS bucket):
  intent_entity_guardrail_model.pt   # torch state_dict of the 3-head model
  label_maps.json                    # {intent_classes, ner_labels, model_name, max_len}

Runs on CPU on purpose (like vision) so the GPU stays free for the LLM.
"""
import json
import os
import re

from . import config as C

_model = None
_tok = None
_id2intent = {}
_ner_labels = []
_max_len = 64
MODEL_OK = False

# Matches run_e2e_eval.ieg_run: intent head is read multi-label via a sigmoid gate
# so compound queries ("price AND disease") surface every intent above threshold.
INTENT_THRESHOLD = 0.30

# fine intent -> retrieval fusion intent (policy / field_practice / general).
# Unknown intents fall back to 'general' (and C.FUSION_WEIGHTS.get also falls back
# to 'general'), so retrieval never breaks even if the trained label set differs.
_INTENT_TO_RETRIEVAL = {
    "nutrition_fertilizer": "field_practice",
    "cultivation_practice": "field_practice",
    "disease_pest":         "field_practice",
    "post_harvest_storage": "field_practice",
    "weather":              "general",
    "market_price":         "general",
    "mandi_prices":         "general",
    "yield_estimation":     "general",
    "specialty_other":      "general",
    "general":              "general",
    "non_agri":             "general",
    "other":                "general",
}

# --- RULE LAYER (always on) -------------------------------------------------
# Restricted / banned or heavily-regulated substances: block procurement-style
# queries with a safety message instead of answering "where to buy".
_RESTRICTED = ["monocrotophos", "phorate", "methyl parathion", "phosphamidon",
               "carbofuran", "endosulfan", "paraquat"]
_BUY_INTENT = re.compile(r"\b(buy|purchase|kahan\s*se|kaha\s*se|khareed|kharid|order|price of)\b", re.I)

# Obvious non-agriculture topics the model must never route into RAG.
_NON_AGRI = re.compile(
    r"\b(movie|cricket|football|chess|board\s+game|video\s+game|gaming|bitcoin|stock|"
    r"flight|hotel|bike|motorcycle|car|automobile|engine\s+repair|vehicle\s+repair|"
    r"phone\s+repair|computer\s+repair|laptop|iphone|programming|coding|software|"
    r"visa|passport|loan\s+app|dating|song|lyrics)\b", re.I)


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


# Greeting / small-talk detection. Used to (a) never let the model guardrail block a
# greeting, and (b) route chit-chat to conversational mode even when the app has attached
# live_data. It must NOT fire on a real question that merely opens with a greeting word
# ("namaste, wheat rust ki dawa") — so we require a greeting word AND no agri/data content.
_GREET_WORDS = re.compile(
    r"\b(hi|hey|hello|helo|yo|namaste|namaskar|namaskaram|pranam|ram\s*ram|sat\s*sri\s*akal|"
    r"salaam|salam|greetings|good\s*(morning|afternoon|evening|night)|"
    r"how\s*(are|r)\s*(you|u|ya)|kaise\s*(ho|hain|hai)|kaisi\s*ho|kya\s*haal|kya\s*chal|"
    r"who\s*are\s*(you|u)|aap\s*kaun|tum\s*kaun|"
    r"what\s*can\s*(you|u)\s*do|what\s*do\s*(you|u)\s*do|how\s*can\s*(you|u)\s*help|"
    r"introduce\s*yourself|thanks|thank\s*you|dhanyavad|shukriya|welcome)\b", re.I)

# Agri / data topic words — if any appear, treat it as a real question, not chit-chat.
_CONTENT_WORDS = re.compile(
    r"\b(wheat|rice|paddy|maize|bajra|sugarcane|potato|tomato|onion|mustard|crop|crops|fasal|"
    r"gehu|gehun|dhan|makka|dawa|dawai|spray|rust|blight|pest|pests|disease|rog|keet|fungicid|"
    r"insecticid|urea|dap|npk|khaad|fertiliz|nutrient|dose|dosage|mandi|bhav|bhaav|price|prices|"
    r"rate|rates|daam|msp|quintal|market|weather|mausam|mosam|rain|barish|temperature|forecast|"
    r"yield|paidawar|utpadan|production|scheme|schemes|yojana|subsidy|loan|insurance|bima|"
    r"pm[- ]?kisan|irrigation|sinchai|seed|beej|soil|mitti|water|paani|harvest|sow|bijai)\b", re.I)


def is_greeting(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 60:              # long messages are real questions, not greetings
        return False
    if _CONTENT_WORDS.search(t):          # mentions a crop / price / scheme / etc. → real question
        return False
    return bool(_GREET_WORDS.search(t))


# Agronomy-advice topics (disease / pest / dosage / cultivation). Used to decide that a query
# genuinely needs the knowledge base — so a mixed "blight ka ilaj AND mandi bhav" question keeps
# its (relevant) KB chunks instead of being treated as a pure price/weather/yield lookup.
_AGRI_ADVICE = re.compile(
    r"\b(disease|blight|rust|blast|spot|smut|wilt|rot|mildew|mould|mold|aphid|mite|borer|"
    r"pest|pests|insect|keet|rog|bimari|bimaari|"
    r"dawa|dawai|davai|spray|chhidkav|chidkav|fungicid|insecticid|pesticid|herbicid|"
    r"ilaj|ilaaj|upchar|upchaar|treatment|control|manage|prevent|cure|"
    r"urea|dap|npk|khaad|khad|fertiliz|nutrient|dose|dosage|khurak|matra|"
    r"cultivation|sowing|bijai|buvai|irrigation|sinchai|weed|kharpatwar|"
    r"variety|kism|qism|beej|seed|pruning|spacing)\b", re.I)


def is_agri_advice(text: str) -> bool:
    """True if the query asks for agronomy advice (disease/pest/dosage/practice)."""
    return bool(_AGRI_ADVICE.search(text or ""))


def _keyword_intent(text: str) -> str:
    """Fallback intent guess used only when the model isn't loaded."""
    t = (text or "").lower()
    if re.search(r"\b(scheme|subsidy|yojana|loan|insurance|bima|kcc card|pm[- ]?kisan)\b", t):
        return "field_practice"  # scheme questions -> fusion handles pdf weighting via 'policy' below
    if re.search(r"\b(rust|blight|pest|disease|rog|keet|dawa|spray|fungicid)\b", t):
        return "disease_pest"
    if re.search(r"\b(urea|dap|npk|fertiliz|khaad|nutrient|dose|kg\/)\b", t):
        return "nutrition_fertilizer"
    return "general"


# --- MODEL LAYER (DistilBERT, 3 heads) --------------------------------------
def load():
    """Load the 3-head DistilBERT checkpoint from /artifacts/ieg. Never fatal.

    Mirrors run_e2e_eval.load_ieg: reads label_maps.json for the class lists,
    rebuilds the exact IEGModel, loads the state_dict (bare or wrapped), CPU/eval.
    """
    global _model, _tok, _id2intent, _ner_labels, _max_len, MODEL_OK
    if not C.HAS_IEG:
        print("[ieg] no checkpoint at IEG_DIR — running RULES + keyword intent only.")
        return
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        # ---- exact architecture from run_e2e_eval.IEGModel / notebook 11 -----
        class IEGModel(nn.Module):
            def __init__(self, n_intents, n_ner, model_name):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(model_name)
                h = self.backbone.config.hidden_size
                self.intent_head = nn.Linear(h, n_intents)       # on [CLS] pooled token
                self.ner_head = nn.Linear(h, n_ner)               # per-token
                self.guardrail_head = nn.Linear(h, 2)             # on [CLS] pooled token

            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                pooled = out[:, 0, :]
                return self.intent_head(pooled), self.ner_head(out), self.guardrail_head(pooled)

        # ---- label maps (saved next to the checkpoint) -----------------------
        lm_path = None
        for name in ("label_maps.json", "intent_entity_label_maps.json"):
            p = os.path.join(C.IEG_DIR, name)
            if os.path.isfile(p):
                lm_path = p
                break
        label_maps = json.load(open(lm_path, encoding="utf-8")) if lm_path else {}

        intent_classes = label_maps.get("intent_classes", [
            "cultivation_practice", "disease_pest", "general", "non_agri",
            "nutrition_fertilizer", "post_harvest_storage", "specialty_other", "yield_estimation"])
        _ner_labels = label_maps.get("ner_labels", ["O", "B-CROP", "I-CROP", "B-DISTRICT", "I-DISTRICT"])
        model_name = label_maps.get("model_name", "distilbert-base-multilingual-cased")
        _max_len = int(label_maps.get("max_len", 64))

        # ---- build + load the state_dict (bare or wrapped) -------------------
        ckpt = os.path.join(C.IEG_DIR, "intent_entity_guardrail_model.pt")
        model = IEGModel(len(intent_classes), len(_ner_labels), model_name)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            sd = state.get("model_state_dict", state.get("state_dict", state))
        else:
            sd = state
        model.load_state_dict(sd)
        model.eval()

        _model = model
        _tok = AutoTokenizer.from_pretrained(model_name)
        _id2intent = {i: c for i, c in enumerate(intent_classes)}
        MODEL_OK = True
        print(f"[ieg] model loaded — {len(intent_classes)} intents, {len(_ner_labels)} NER tags, on CPU.")
    except Exception as e:
        MODEL_OK = False
        print(f"[ieg] model not loaded ({e}). RULES + keyword intent only — RAG still works.")


def _decode_entities(text, ner_ids, offsets):
    """Merge BIO token tags back into {crop:[...], district:[...]} spans via offsets."""
    ents = {}
    cur = {"type": None, "s": None, "e": None}

    def flush():
        if cur["type"] is not None:
            val = text[cur["s"]:cur["e"]].strip()
            if val:
                ents.setdefault(cur["type"].lower(), []).append(val)
        cur["type"], cur["s"], cur["e"] = None, None, None

    for tid, (s, e) in zip(ner_ids, offsets):
        if s == e:                       # special / padding token
            continue
        tag = _ner_labels[tid] if 0 <= tid < len(_ner_labels) else "O"
        if tag == "O":
            flush()
            continue
        pos, _, etype = tag.partition("-")     # "B-CROP" -> ("B","-","CROP")
        if pos == "B" or cur["type"] != etype:
            flush()
            cur["type"], cur["s"], cur["e"] = etype, s, e
        else:                            # "I-" continuation of the same type
            cur["e"] = e
    flush()
    return ents


def _model_infer(text: str):
    """Run the 3 heads. Returns (primary_intent, intents_list, entities, off_domain)."""
    import torch

    enc = _tok(text, truncation=True, max_length=_max_len, padding="max_length",
               return_offsets_mapping=True, return_tensors="pt")
    off = enc.pop("offset_mapping", None)              # only a fast tokenizer returns this
    enc.pop("token_type_ids", None)                    # DistilBERT doesn't use them
    with torch.no_grad():
        intent_logits, ner_logits, guard_logits = _model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    # intent: primary = argmax (the trained softmax objective); list = e2e sigmoid gate
    primary = _id2intent.get(int(intent_logits[0].argmax(-1).item()), "general")
    probs = torch.sigmoid(intent_logits[0])
    intents = [_id2intent[i] for i, p in enumerate(probs.tolist())
               if p > INTENT_THRESHOLD and i in _id2intent]
    if not intents:
        intents = [primary]

    off_domain = int(guard_logits[0].argmax(-1).item()) == 1   # class 1 = off-domain
    # entities are a bonus: only if the tokenizer gave char offsets (fast tokenizer)
    entities = {}
    if off is not None:
        ner_ids = ner_logits[0].argmax(-1).tolist()
        entities = _decode_entities(text, ner_ids, off[0].tolist())
    return primary, intents, entities, off_domain


def classify(text: str) -> dict:
    """Returns intent + entities + guardrail decision. Guardrail = model OR rules."""
    blocked, reason = rule_block(text)          # rules always run

    if MODEL_OK:
        try:
            intent, intents, entities, off_domain = _model_infer(text)
            if off_domain and not blocked and not is_greeting(text):
                blocked, reason = True, "off_domain_model"
        except Exception as e:                  # a bad request never kills the path
            print(f"[ieg] inference failed ({e}); using keyword intent for this query.")
            intent = _keyword_intent(text)
            intents, entities = [intent], {}
    else:
        intent = _keyword_intent(text)
        intents, entities = [intent], {}

    return {
        "intent": intent,
        "intents": intents,
        "retrieval_intent": _INTENT_TO_RETRIEVAL.get(intent, "general"),
        "blocked": blocked,
        "block_reason": reason,
        "entities": entities,
        "guardrail_backend": "model+rules" if MODEL_OK else "rules-only",
    }
