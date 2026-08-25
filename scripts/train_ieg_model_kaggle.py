import json
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path("/kaggle/input/datasets/lokeshvns/e2e-dataset/data/processed/kcc/kcc_train_99.csv")

ROOT      = Path(__file__).resolve().parent
OUT_DIR   = ROOT / "outputs" / "ieg_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# l3cube-pune/hing-mbert-mixed: mBERT fine-tuned on Hinglish corpus by L3Cube Pune.
# Better than vanilla mBERT for KCC data which is heavily Hinglish/Hindi.
# Hidden size = 768 (same as mBERT) — no architecture changes needed.
MODEL_NAME = "l3cube-pune/hing-mbert-mixed"
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
MAX_LEN         = 48
BATCH_SIZE      = 128
EPOCHS          = 10         # early stopping will likely fire before this
LR              = 3e-5       # head learning rate
BACKBONE_LR     = 1e-5       # backbone LR (LLRD: lower to prevent forgetting)
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.05      # slight smoothing on intent loss
WARMUP_RATIO    = 0.15      # 10 % of total steps for linear warmup
PATIENCE        = 2         # early stopping patience (composite val score)

# Data flags
# Curated synthetic guardrail examples are ALWAYS added regardless of this flag.
#
# NON_AGRI_CSV — Kaggle dataset aneeqasiddiqui377/non-agricultural-data
#   File values: guardrail_label=0 (safe, not dangerous), _is_real_kcc=0
#   _is_real_kcc overridden to 1 at training time → enables non_agri intent training
#   Guardrail semantics: 1=DANGEROUS (block), 0=SAFE (route via intent). NOT off-domain.
#   Inference: combined_flag = ml_model_flag OR rule_flag(text)
NON_AGRI_CSV = Path("/kaggle/input/datasets/aneeqasiddiqui377/non-agricultural-data/final_non_agri_data01.csv")

# HOLDOUT_EVAL_NAME — real-scenario holdout (45 rows: 5 per intent ×
# English / Devanagari Hindi / Hinglish). Tests isolated IEG performance
# AND the hing-mbert cross-lingual transfer objective.
HOLDOUT_EVAL_NAME = "kcc_eval_1_augmented.csv"

# ADD_AG_NEWS_NEGATIVES — fallback extra off-domain negatives from AG News.
#   Only useful if NON_AGRI_CSV is not available and you want more diversity.
#   Requires: pip install datasets  (~120 MB download on first use).
ADD_TOXICITY_NEGATIVES = True
ADD_PARAPHRASING_GUARDRAIL = True 
PARA_MAX_LEN = MAX_LEN 
GUARDRAIL_CACHE_PATH = ROOT / "guardrail_cache.json" 

# NER labels — DISTRICT added for LangGraph district-resolution benefit
# (app.py NER_LABELS must also be updated to match when loading this checkpoint)
NER_LABELS   = ["O", "B-CROP", "I-CROP", "B-DISTRICT", "I-DISTRICT"]
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC GUARDRAIL — banned pesticides, dangerous overdose, unsafe practices
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_GUARDRAIL = [
    # Banned pesticides (English)
    "can I use monocrotophos on my tomato crop",
    "endosulfan spray for cotton bollworms",
    "is phorate available for sugarcane",
    "methyl parathion dose for rice",
    "DDT application in vegetable crops",
    "aldrin pesticide for termite control in fields",
    "chlorpyrifos 20% EC for pest control",
    "where to buy monocrotophos in bulk",
    "endosulfan supplier near my village",
    "phorate 10% granules for groundnut",
    # Banned pesticides (Hinglish)
    "is monocrotophos sold anywhere near me for crop spraying",
    "endosulfan ka istemal kar sakte hain kya is fasal mein",
    "can I still buy methyl parathion for pest control",
    "aldrin dawa milegi kya sasti keemat mein",
    "chlorpyrifos spray dalni hai cotton par",
    # Dangerous overdose (English)
    "apply 10x the recommended pesticide dose",
    "double strength pesticide for quick results",
    "triple dose of urea per acre",
    "500 kg urea per acre for faster growth",
    "use extra pesticide to be sure it works",
    "spray 3 times the label dose",
    "increase fertilizer by 4x for better yield",
    "apply 400kg urea in one acre plot this season",
    "using five times the normal pesticide strength on infested wheat",
    "what is the maximum urea dose per acre",
    "can I double the fertilizer dose",
    # Dangerous overdose (Hinglish)
    "kitna zyada dawa dalu taaki fasal jaldi theek ho jaye",
    "spray double the recommended dose of this pesticide to be sure it works",
    "triple dose of urea per acre for faster growth is that fine",
    "mix extra fertilizer beyond the label amount for better yield",
    "zyada dawa dalne se keede jaldi maarte hain kya",
    # Unsafe application practices
    "mix all pesticides together to save time",
    "spray pesticide without mask and gloves",
    "store pesticides in food containers",
    "use pesticide during rain for better coverage",
    "give pesticide overdose to kill all insects at once",
]

MORE_GUARDRAIL_TEMPLATES = [
    # Banned pesticides (more chemicals)
    "is {chem} allowed on {crop}",
    "can I still buy {chem} for my farm",
    "{chem} ki dostavai kahan milti hai",
    # Dangerous overdose (more patterns)
    "use {multiplier}x the recommended dose of {chem}",
    "apply {amount} kg of {chem} per acre",
    "spray {multiplier} times the label strength of {chem}",
    # Unsafe practices (more)
    "mix {chem} with water and drink it",
    "store {chem} in a kitchen container",
    "apply {chem} during rain for better spread",
]

# Helper to instantiate the templates
def _expand_guardrail():
    chem_list = ["monocrotophos", "endosulfan", "phorate", "methyl parathion",
                 "ddt", "aldrin", "chlorpyrifos", "heptachlor", "lindane"]
    crop_list = ["wheat", "paddy", "tomato", "potato", "mustard", "onion",
                 "maize", "sugarcane", "chickpea", "cotton"]
    multiplier = ["double", "triple", "4x", "5x", "10x"]
    amount = ["200", "400", "600", "800", "1000"]  # kg per acre

    extras = []
    for t in MORE_GUARDRAIL_TEMPLATES:
        if "{chem}" in t and "{crop}" in t:
            for c in chem_list:
                for cr in crop_list:
                    extras.append(t.format(chem=c, crop=cr))
        elif "{chem}" in t and "{multiplier}" in t:
            for c in chem_list:
                for m in multiplier:
                    extras.append(t.format(chem=c, multiplier=m))
        elif "{chem}" in t and "{amount}" in t:
            for c in chem_list:
                for a in amount:
                    extras.append(t.format(chem=c, amount=a))
        elif "{multiplier}" in t and "{chem}" in t:
            for m in multiplier:
                for c in chem_list:
                    extras.append(t.format(multiplier=m, chem=c))
        else:  # no placeholders – just keep the string as is
            extras.append(t)
    return extras

# Append the generated examples to the existing list
SYNTHETIC_GUARDRAIL.extend(_expand_guardrail())
print(f"✅ Guardrail list now has {len(SYNTHETIC_GUARDRAIL)} entries")


# ----------------------------------------------------------------------
#  OPTIONAL: Paraphrase the existing SYNTHETIC_GUARDRAIL seeds
#  with automatic caching to avoid re‑doing the work on every run.
#  Uses AutoModelForSeq2SeqLM + AutoTokenizer (no pipeline task needed).
# ----------------------------------------------------------------------
if ADD_PARAPHRASING_GUARDRAIL:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from tqdm.auto import tqdm
    import json
    import torch

    # --------------------------------------------------------------
    # 0️⃣  Try to load a cached version first
    # --------------------------------------------------------------
    if GUARDRAIL_CACHE_PATH.is_file():
        try:
            with open(GUARDRAIL_CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and all(isinstance(s, str) for s in cached):
                SYNTHETIC_GUARDRAIL = cached
                print(
                    f"🔄 Loaded {len(SYNTHETIC_GUARDRAIL)} guardrail seeds from cache "
                    f"({GUARDRAIL_CACHE_PATH})"
                )
            else:
                raise ValueError("Cache file does not contain a list of strings.")
        except Exception as e:
            print(f"⚠️  Failed to read cache ({e}); will regenerate.")
    else:
        print(f"ℹ️  No cache found at {GUARDRAIL_CACHE_PATH}; will generate paraphrases.")

    # --------------------------------------------------------------
    # 1️⃣  If we still need to paraphrase (cache miss or forced reload)
    # --------------------------------------------------------------
    original_len = len(SYNTHETIC_GUARDRAIL)

    # If we already loaded a cache we are done.
    if GUARDRAIL_CACHE_PATH.is_file():
        pass   # cached list is already in SYNTHETIC_GUARDRAIL
    else:
        # ----- Load paraphraser model (once) -----
        # <<<--- USE A LOCAL VARIABLE NAME, NOT MODEL_NAME --->>>
        PARAPHRASER_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"   # <‑‑ changed
        tokenizer = AutoTokenizer.from_pretrained(PARAPHRASER_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_MODEL_NAME)
        model.to(DEVICE)          # DEVICE is already defined in the script
        model.eval()

        def _paraphrase_batch(texts, num_return=2):
            """Return a flat list of paraphrases (num_return per input)."""
            inputs = [f"paraphrase: {t}" for t in texts]
            enc = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,   # keep same length as training
            ).to(DEVICE)

            gen_ids = model.generate(
                **enc,
                max_length=MAX_LEN,   # also cap generated length
                num_beams=4,
                num_return_sequences=num_return,
                do_sample=True,
                temperature=0.7,
                early_stopping=True,
            )
            return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # ---- Configuration -------------------------------------------------
        PARA_NUM_RETURN   = 3          # paraphrases per seed (1‑3)
        PARA_CHUNK_SIZE   = 500        # batch size for the paraphraser

        # ---- Generate paraphrases in chunks -------------------------------
        paraphrased_seeds = []
        for start in tqdm(
            range(0, len(SYNTHETIC_GUARDRAIL), PARA_CHUNK_SIZE),
            desc="Paraphrasing synthetic guardrail seeds",
        ):
            chunk = SYNTHETIC_GUARDRAIL[start : start + PARA_CHUNK_SIZE]
            paras = _paraphrase_batch(chunk, num_return=PARA_NUM_RETURN)
            for original, para in zip(chunk, paras):
                paraphrased_seeds.append(para)

        # ---- Append and (optionally) save to cache -----------------------
        if paraphrased_seeds:
            SYNTHETIC_GUARDRAIL.extend(paraphrased_seeds)
            try:
                with open(GUARDRAIL_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(SYNTHETIC_GUARDRAIL, f, ensure_ascii=False, indent=2)
                print(
                    f"💾 Saved expanded guardrail list ({len(SYNTHETIC_GUARDRAIL)} total) "
                    f"to {GUARDRAIL_CACHE_PATH}"
                )
            except Exception as e:
                print(f"⚠️  Could not write cache: {e}")
            print(
                f"✅ Guardrail list expanded: {len(SYNTHETIC_GUARDRAIL)} total entries "
                f"(+{len(paraphrased_seeds)} paraphrased)"
            )
        else:
            print("⚠️  Paraphrasing produced no new seeds – check the model/flags.")
# ----------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# HINGLISH TEMPLATES — rare intent class augmentation (post_harvest, specialty, general)
# ─────────────────────────────────────────────────────────────────────────────

HINGLISH_RARE_TEMPLATES: dict[str, list[str]] = {
    "post_harvest_storage": [
        "{crop} ko store karne ka sahi tarika kya hai",
        "{crop} ki katai ke baad kya karna chahiye",
        "{crop} ko kitne din tak rakh sakte hain",
        "{crop} ki storage mein koi problem ho rahi hai",
        "{crop} post harvest loss kaise kam karen",
        "{crop} ko warehouse mein rakhne ka sahi tarika",
    ],
    "specialty_other": [
        "{crop} ki organic kheti kaise karen",
        "{crop} mein jeevamrit ka prayog kaise karen",
        "{crop} ke liye jaivik kheti ke fayde kya hain",
        "{crop} mein bio fertilizer ka upyog kaise karein",
        "{crop} ki natural farming ki jankari chahiye",
    ],
    "other": [
        "{crop} ki kheti se judi koi jankari chahiye",
        "{crop} ke baare mein kuch poochna tha",
        "{crop} ke liye koi salah chahiye",
        "{crop} ki kheti ke baare mein guide kijiye",
    ],
}

# 'other' matches kcc_train_99.csv catch-all (01_prepare_datasets.py: fillna('other'))
RARE_CLASSES       = ["post_harvest_storage", "specialty_other", "other"]
TOP_CROPS_FALLBACK = ["wheat", "paddy", "tomato", "potato", "mustard",
                        "onion", "maize", "sugarcane", "chickpea", "cotton"]

# ─────────────────────────────────────────────────────────────────────────────
# EDA — language, token length, NER coverage, intent imbalance
# ─────────────────────────────────────────────────────────────────────────────

DEVANAGARI_RE    = re.compile(r"[\u0900-\u097F]")
HINGLISH_MARKERS = {
    "ki", "ke", "ka", "ko", "se", "me", "mein", "hai", "hain", "kya",
    "kaise", "kab", "kyun", "aur", "bhi", "karen", "kare", "karna",
    "chahiye", "wala", "sakte", "sakta", "fasal", "khaad", "dawa",
    "prayog", "jankari", "upaj", "paidawar",
}


def detect_lang(text: str) -> str:
    """Classify query language as hindi / hinglish / english."""
    text = str(text)
    if DEVANAGARI_RE.search(text):
        return "hindi"
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return "hinglish" if words & HINGLISH_MARKERS else "english"


def run_eda(df: pd.DataFrame, tokenizer) -> None:
    """
    EDA complementing 01_prepare_datasets.py.
    Reports: language distribution, token lengths, NER span coverage,
    intent imbalance. Run BEFORE augmentation on KCC rows only.
    """
    print("\n" + "=" * 70)
    print("EDA (run on kcc_train_99.csv rows, before augmentation)")
    print("=" * 70)

    # [1] Language distribution per intent
    df = df.copy()
    df["_lang"] = df["QueryText"].apply(detect_lang)
    print("\n[1] Language distribution per intent:")
    lang_cross = pd.crosstab(df["intent_label"], df["_lang"])
    print(lang_cross.to_string())
    hinglish_pct = (df["_lang"] == "hinglish").mean() * 100
    hindi_pct    = (df["_lang"] == "hindi").mean()    * 100
    english_pct  = 100 - hinglish_pct - hindi_pct
    print(f"\n  Overall → English={english_pct:.1f}%  "
          f"Hinglish={hinglish_pct:.1f}%  Hindi={hindi_pct:.1f}%")
    if hinglish_pct < 10:
        print("  ⚠️  Low Hinglish coverage — synthetic templates will help")

    # [2] Token length distribution
    print(f"\n[2] Token length distribution (MAX_LEN={MAX_LEN}):")
    lengths = df["QueryText"].fillna("").apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=True))
    )
    p95 = int(lengths.quantile(0.95))
    print(f"  mean={lengths.mean():.1f}  median={lengths.median():.0f}  "
          f"p95={p95}  max={lengths.max()}")
    exceed_pct = (lengths > MAX_LEN).mean() * 100
    print(f"  Truncated at MAX_LEN={MAX_LEN}: {exceed_pct:.1f}%")
    if exceed_pct > 5:
        print(f"  ⚠️  >5% of queries will be truncated — consider increasing MAX_LEN")
    if p95 < MAX_LEN - 10:
        print(f"  ℹ️  p95={p95} well below MAX_LEN={MAX_LEN} — could reduce for speed")

    # [3] NER span coverage
    print("\n[3] NER span coverage:")

    def _has_entity(spans_val, label: str) -> bool:
        try:
            spans = json.loads(spans_val) if isinstance(spans_val, str) else spans_val
            return any(len(s) >= 3 and s[2] == label for s in spans)
        except Exception:
            return False

    crop_cov = df["_char_spans"].apply(lambda s: _has_entity(s, "CROP")).mean()     * 100
    dist_cov = df["_char_spans"].apply(lambda s: _has_entity(s, "DISTRICT")).mean() * 100
    print(f"  CROP span matched:     {crop_cov:.1f}% of queries")
    print(f"  DISTRICT span matched: {dist_cov:.1f}% of queries")
    if crop_cov < 40:
        print("  ⚠️  Low CROP coverage — alias map in 01_prepare_datasets.py may need expansion")

    # [4] Intent class imbalance
    print("\n[4] Intent class imbalance:")
    counts  = df["intent_label"].value_counts()
    max_cnt = counts.max()
    print(f"  {'Intent':<30} {'Count':>7}  {'Ratio':>6}")
    print("  " + "-" * 48)
    for intent, n in counts.items():
        ratio = n / max_cnt
        flag  = "  ⚠️  RARE" if ratio < 0.25 else ""
        print(f"  {intent:<30} {n:>7,}  {ratio:>6.2f}{flag}")

    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _build_guardrail_df() -> pd.DataFrame:
    """35 authored dangerous examples (guardrail=1, _is_real_kcc=0; intent masked via -100)."""
    return pd.DataFrame([{
        "QueryText":       text,
        "intent_label":    "non_agri",   # masked by _is_real_kcc=0; value is irrelevant
        "guardrail_label": 1,
        "_char_spans":     "[]",
        "_is_real_kcc":    0,
    } for text in SYNTHETIC_GUARDRAIL])


def _build_hinglish_df(df: pd.DataFrame) -> pd.DataFrame:
    """Template-based Hinglish rows for rare intent classes."""
    # Use top crops from the actual training data; fall back to hardcoded list
    if "Crop" in df.columns:
        top_crops = df["Crop"].dropna().value_counts().head(20).index.tolist()
        # Simplify: take the part before parentheses, lowercase, deduplicate
        simple_crops = list({c.split("(")[0].strip().lower()
                             for c in top_crops if len(c.split("(")[0].strip()) > 2})[:15]
    else:
        simple_crops = TOP_CROPS_FALLBACK

    if not simple_crops:
        simple_crops = TOP_CROPS_FALLBACK

    rows = []
    for intent, templates in HINGLISH_RARE_TEMPLATES.items():
        for crop in simple_crops:
            for tmpl in templates:
                rows.append({
                    "QueryText":       tmpl.format(crop=crop),
                    "intent_label":    intent,
                    "guardrail_label": 0,
                    "_char_spans":     "[]",
                    "_is_real_kcc":    1,  # valid intent label → train intent head; NER=all-O (no spans)
                })
    return pd.DataFrame(rows)


def _build_non_agri_df() -> pd.DataFrame | None:
    """
    Load non-agricultural data (e.g., aneeqasiddiqui377/non-agricultural-data).
    """
    if not NON_AGRI_CSV.exists():
        print(f"  ℹ️  NON_AGRI_CSV not found at {NON_AGRI_CSV} — skipping.")
        return None

    import ast
    ndf = pd.read_csv(NON_AGRI_CSV)

    # Rename text→QueryText; also accepts 'cleaned_query' for other dataset versions
    for src_col in ("text", "cleaned_query"):
        if src_col in ndf.columns and "QueryText" not in ndf.columns:
            ndf = ndf.rename(columns={src_col: "QueryText"})
            break

    if "QueryText" not in ndf.columns:
        print(f"  ⚠️  No recognised text column (checked: text, cleaned_query, QueryText).")
        print(f"     Columns found: {list(ndf.columns)}")
        return None

    # Parse _char_spans from Python literal string → re-serialise as JSON
    if "_char_spans" in ndf.columns:
        ndf["_char_spans"] = ndf["_char_spans"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else (v or [])
        )
        ndf["_char_spans"] = ndf["_char_spans"].apply(json.dumps)
    else:
        ndf["_char_spans"] = "[]"

    # guardrail_label=0 trusted from file. Override _is_real_kcc→1 to enable non_agri intent training.
    ndf["_is_real_kcc"]    = 1
    ndf["intent_label"]    = ndf["intent_label"].fillna("non_agri").astype(str)

    print(f"  Loaded {len(ndf):,} rows from final_non_agri_data01.csv")
    print(f"  guardrail=0 | _is_real_kcc=1 (intent training enabled)")
    return ndf



def load_data(tokenizer):
    """
    Load, augment, and split data. Returns (train_df, val_df, test_df, intent2id, id2intent).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {DATA_PATH}\n"
            "Run scripts/01_prepare_datasets.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows from kcc_train_99.csv")

    # Standardise columns
    df["intent_label"]    = df["intent_label"].fillna("general").astype(str)
    df["guardrail_label"] = 0
    df["_is_real_kcc"]    = 1
    df["_char_spans"]     = df["_char_spans"].fillna("[]")

    print(f"\nIntent distribution (KCC rows):\n{df['intent_label'].value_counts().to_string()}\n")

    run_eda(df, tokenizer)

    # Intent classes derived from KCC only; non_agri added for the non-agri CSV split
    intent_classes = sorted(df["intent_label"].unique().tolist())
    if "non_agri" not in intent_classes:
        intent_classes.append("non_agri")
    intent2id = {c: i for i, c in enumerate(intent_classes)}
    id2intent  = {i: c for c, i in intent2id.items()}

    COLS = ["QueryText", "intent_label", "guardrail_label", "_char_spans", "_is_real_kcc"]
    all_dfs = [df[COLS]]

    guardrail_df = _build_guardrail_df()
    all_dfs.append(guardrail_df)
    print(f"Added {len(guardrail_df)} curated synthetic guardrail rows")

    hinglish_df = _build_hinglish_df(df)
    all_dfs.append(hinglish_df)
    print(f"Added {len(hinglish_df)} Hinglish template rows for rare classes")

    # Non-agri curated dataset (preferred over AG News)
    non_agri_df = _build_non_agri_df()
    if non_agri_df is not None:
        non_agri_cols = [c for c in COLS if c in non_agri_df.columns]
        all_dfs.append(non_agri_df[non_agri_cols])
        print(f"Added {len(non_agri_df):,} non-agri curated rows "
              f"(intent routing, guardrail=0, _is_real_kcc=1)")

    if ADD_TOXICITY_NEGATIVES:          # keep the flag, just change the source
        from datasets import load_dataset
        print("Loading toxic comments (civil_comments)…")
        toxic = load_dataset("civil_comments", split="train[:20000]")  # adjust size
        # The dataset has a 'toxicity' float 0‑1; treat >0.6 as dangerous
        toxic_df = toxic.filter(lambda x: x["toxicity"] > 0.6).to_pandas()
        toxic_df = toxic_df.rename(columns={"text": "QueryText"})
        toxic_df["intent_label"]    = "non_agri"   # masked later (_is_real_kcc=0)
        toxic_df["guardrail_label"] = 1
        toxic_df["_char_spans"]     = "[]"
        toxic_df["_is_real_kcc"]    = 1
        all_dfs.append(toxic_df[COLS])
        print(f"Added {len(toxic_df)} toxic‑comment rows")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df["intent_label"] = full_df["intent_label"].fillna("general").astype(str)

    print(f"\nCombined dataset: {len(full_df):,} rows")
    print(f"Guardrail distribution:\n{full_df['guardrail_label'].value_counts().to_string()}")

    # ── 80 / 10 / 10 split — joint stratification by intent × guardrail ────
    # Combining labels keeps rare intent classes proportionally represented
    # across splits (guardrail-only stratification let them clump).
    def _strat_key(d: pd.DataFrame) -> pd.Series:
        key = d["intent_label"].astype(str) + "|" + d["guardrail_label"].astype(str)
        # sklearn requires ≥2 samples per class; collapse ultra-rare combos
        counts = key.value_counts()
        return key.mask(key.map(counts) < 2, "_rare_")

    strat_full = _strat_key(full_df)
    train_df, temp_df = train_test_split(
        full_df, test_size=0.20, random_state=SEED,
        stratify=strat_full,
    )
    strat_temp = _strat_key(temp_df)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=strat_temp,
    )
    print("Split intent distributions:")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = (split["intent_label"].value_counts(normalize=True) * 100).round(2)
        print(f"  {name}: {dist.to_dict()}")
    print(f"\nSplit → train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}\n")

    return train_df, val_df, test_df, intent2id, id2intent



# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED GUARDRAIL — inference-time layer; combined_flag = ml_flag OR rule_flag(text)
# ─────────────────────────────────────────────────────────────────────────────

_BANNED_TERMS = [
    "monocrotophos", "endosulfan", "phorate", "methyl parathion",
    "ddt", "aldrin", "chlorpyrifos",
]
_DOSAGE_RE = [
    re.compile(r"\b(\d+)\s*(x|times)\b.*\b(dose|dosage|dawa|spray)\b", re.I),
    re.compile(r"\b(double|triple|4x|5x|10x)\b.*\b(dose|dosage|strength|dawa)\b", re.I),
    re.compile(r"\bz?jyada\b.*\bdawa\b", re.I),
    re.compile(r"\b(\d{3,})\s*kg\b.*\b(per acre|prati ekad|ekad)\b", re.I),
]


def rule_flag(text: str) -> int:
    """1 if banned chemical or dangerous overdose pattern, else 0.
    Inference: combined = int(ml_flag == 1 or rule_flag(text) == 1)
    """
    t = str(text).lower()
    if any(term in t for term in _BANNED_TERMS):
        return 1
    return int(any(pat.search(t) for pat in _DOSAGE_RE))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _spans_to_tags(spans: list, offsets: list) -> list[int]:
    """
    Convert character-level entity spans → per-token BIO tag IDs.
    Special / padding tokens (offset == (0,0)) are marked -100 (ignored by loss).
    """
    tags = [NER_LABEL2ID["O"]] * len(offsets)

    for start, end, label in spans:
        b_tag = f"B-{label}"
        i_tag = f"I-{label}"
        if b_tag not in NER_LABEL2ID:
            continue   # unknown entity type (e.g. future labels) — skip cleanly
        first = True
        for i, (s, e) in enumerate(offsets):
            if s == e:  # special token — skip here; masked to -100 below
                continue
            if s >= start and e <= end:
                tags[i] = NER_LABEL2ID[b_tag if first else i_tag]
                first   = False

    # Mask special tokens (CLS, SEP, PAD) so their loss is ignored
    for i, (s, e) in enumerate(offsets):
        if s == e:
            tags[i] = -100

    return tags


class IEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, intent2id: dict):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.intent2id = intent2id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        text = str(row["QueryText"])

        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()

        # NER: only on real-KCC rows; -100 elsewhere
        if row["_is_real_kcc"]:
            try:
                spans = (
                    json.loads(row["_char_spans"])
                    if isinstance(row["_char_spans"], str)
                    else row["_char_spans"]
                )
            except Exception:
                spans = []
            ner_tags = _spans_to_tags(spans, offsets)
        else:
            ner_tags = [-100] * MAX_LEN

        # Intent: -100 for non-KCC rows; fallback to 'other' (not 'general') on unknown labels
        intent_id = (
            self.intent2id.get(row["intent_label"],
                               self.intent2id.get("other", 0))
            if row["_is_real_kcc"]
            else -100
        )

        return {
            "input_ids":       enc["input_ids"][0],
            "attention_mask":  enc["attention_mask"][0],
            "intent_label":    torch.tensor(intent_id,              dtype=torch.long),
            "ner_tags":        torch.tensor(ner_tags,               dtype=torch.long),
            "guardrail_label": torch.tensor(row["guardrail_label"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class IEGModel(nn.Module):
    def __init__(self, n_intents: int, n_ner: int):
        super().__init__()
        self.backbone       = AutoModel.from_pretrained(MODEL_NAME)
        h                   = self.backbone.config.hidden_size
        self.intent_head    = nn.Linear(h, n_intents)
        self.ner_head       = nn.Linear(h, n_ner)
        self.guardrail_head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        out    = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 0, :]   # [CLS] pooling
        return (
            self.intent_head(pooled),   # (B, n_intents)
            self.ner_head(out),         # (B, seq_len, n_ner)
            self.guardrail_head(pooled) # (B, 2)
        )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bio_to_spans(tags: list[str]) -> set[tuple]:
    """Convert a BIO tag sequence to a set of (start, end, label) spans."""
    spans, start, label = [], None, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, label))
            start, label = i, tag[2:]
        elif tag.startswith("I-") and label == tag[2:]:
            continue
        else:
            if start is not None:
                spans.append((start, i, label))
            start, label = None, None
    if start is not None:
        spans.append((start, len(tags), label))
    return set(spans)


def _entity_f1(true_seqs: list, pred_seqs: list) -> float:
    """Span-level entity F1 (more meaningful than token-level for short spans)."""
    tp = fp = fn = 0
    for t_seq, p_seq in zip(true_seqs, pred_seqs):
        t_spans = _bio_to_spans(t_seq)
        p_spans = _bio_to_spans(p_seq)
        tp += len(t_spans & p_spans)
        fp += len(p_spans - t_spans)
        fn += len(t_spans - p_spans)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate(model: IEGModel, loader: DataLoader) -> dict:
    """
    Returns a dict with:
      intent_accuracy, intent_macro_f1,
      ner_entity_f1,
      guardrail_precision, guardrail_recall, guardrail_f1
    """
    model.eval()
    intent_true, intent_pred = [], []
    ner_true,    ner_pred    = [], []
    guard_true,  guard_pred  = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            il, nl, gl     = model(input_ids, attention_mask)

            # Intent — exclude non-KCC rows (intent_label == -100)
            i_true = batch["intent_label"].numpy()
            i_pred = il.argmax(-1).cpu().numpy()
            mask   = i_true != -100
            intent_true.extend(i_true[mask].tolist())
            intent_pred.extend(i_pred[mask].tolist())

            # NER — span-level evaluation
            n_pred = nl.argmax(-1).cpu().numpy()
            n_true = batch["ner_tags"].numpy()
            for p_seq, t_seq in zip(n_pred, n_true):
                valid = t_seq != -100
                ner_pred.append([NER_LABELS[x] for x in p_seq[valid]])
                ner_true.append([NER_LABELS[x] for x in t_seq[valid]])

            # Guardrail
            guard_pred.extend(gl.argmax(-1).cpu().numpy().tolist())
            guard_true.extend(batch["guardrail_label"].numpy().tolist())

    return {
        "intent_accuracy":     accuracy_score(intent_true, intent_pred),
        "intent_macro_f1":     f1_score(intent_true, intent_pred,
                                        average="macro", zero_division=0),
        "ner_entity_f1":       _entity_f1(ner_true, ner_pred),
        "guardrail_precision": precision_score(guard_true, guard_pred, zero_division=0),
        "guardrail_recall":    recall_score(guard_true, guard_pred, zero_division=0),
        "guardrail_f1":        f1_score(guard_true, guard_pred, zero_division=0),
    }


def _print_metrics(metrics: dict, prefix: str = "") -> None:
    for k, v in metrics.items():
        print(f"  {prefix}{k:<30}: {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# HOLDOUT EVALUATION — kcc_eval_1_augmented.csv (real-scenario, cross-lingual)
# ─────────────────────────────────────────────────────────────────────────────

def _find_holdout_csv() -> Path | None:
    """Locate the holdout CSV locally or in the Kaggle input mount."""
    candidates = [
        ROOT / "data" / "processed" / "kcc" / HOLDOUT_EVAL_NAME,
        Path.cwd() / HOLDOUT_EVAL_NAME,
        Path("/kaggle/working") / HOLDOUT_EVAL_NAME,
    ]
    candidates += sorted(Path("/kaggle/input").glob(f"**/{HOLDOUT_EVAL_NAME}"))
    return next((p for p in candidates if p.exists()), None)


def evaluate_holdout(model: IEGModel, tokenizer, intent2id: dict) -> dict | None:
    """
    Evaluate the best checkpoint on the curated holdout set, overall and
    per language (English vs Devanagari Hindi vs Hinglish) — directly measures
    cross-lingual transfer from English-dominant training data.
    Guardrail metrics are omitted (holdout has no dangerous examples).
    """
    eval_path = _find_holdout_csv()
    if eval_path is None:
        print(f"\nℹ️  Holdout eval skipped — {HOLDOUT_EVAL_NAME} not found")
        return None

    hdf = pd.read_csv(eval_path)
    hdf["intent_label"]    = hdf["intent_label"].fillna("other").astype(str)
    hdf["guardrail_label"] = 0
    hdf["_is_real_kcc"]    = 1
    if "_char_spans" not in hdf.columns:
        hdf["_char_spans"] = "[]"
    hdf["_char_spans"]     = hdf["_char_spans"].fillna("[]")

    def _eval_subset(sub: pd.DataFrame) -> dict:
        loader = DataLoader(
            IEGDataset(sub.reset_index(drop=True), tokenizer, intent2id),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=(DEVICE == "cuda"),
        )
        m = evaluate(model, loader)
        return {k: v for k, v in m.items()
                if k in ("intent_accuracy", "intent_macro_f1", "ner_entity_f1")}

    print("\n" + "=" * 70)
    print(f"Holdout Evaluation — {HOLDOUT_EVAL_NAME} (n={len(hdf)}, "
          f"real-scenario, unseen at train time)")
    print("=" * 70)

    results: dict = {"overall": _eval_subset(hdf)}
    print("\nOverall:")
    _print_metrics(results["overall"])

    lang_col = "language" if "language" in hdf.columns else None
    if lang_col:
        for lang, sub in hdf.groupby(lang_col):
            results[str(lang)] = _eval_subset(sub)
            print(f"\n{lang}  (n={len(sub)}):")
            _print_metrics(results[str(lang)])

    print("=" * 70 + "\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 70)
    print("IEG Training Script v2")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Backbone   : {MODEL_NAME}")
    print(f"NER labels : {NER_LABELS}")
    print(f"Epochs     : {EPOCHS}  LR={LR}  BS={BATCH_SIZE}  "
          f"WD={WEIGHT_DECAY}  LS={LABEL_SMOOTHING}  Patience={PATIENCE}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load & prepare data ──────────────────────────────────────────────────
    train_df, val_df, test_df, intent2id, id2intent = load_data(tokenizer)

    # Save label maps early — safe even if training crashes later
    label_maps = {
        "intent_classes": list(intent2id.keys()),
        "ner_labels":     NER_LABELS,
    }
    with open(OUT_DIR / "label_maps.json", "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"Label maps saved → {OUT_DIR / 'label_maps.json'}")
    print(f"Intent classes ({len(intent2id)}): {list(intent2id.keys())}\n")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    # num_workers=0 is safest on Windows; increase on Linux/Kaggle
    _loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(DEVICE == "cuda"))
    train_loader = DataLoader(IEGDataset(train_df, tokenizer, intent2id),
                              shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(IEGDataset(val_df,   tokenizer, intent2id),
                              shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(IEGDataset(test_df,  tokenizer, intent2id),
                              shuffle=False, **_loader_kwargs)

    # ── Model ────────────────────────────────────────────────────────────────
    model = IEGModel(n_intents=len(intent2id), n_ner=len(NER_LABELS)).to(DEVICE)

    # ── Baseline evaluation (untrained backbone) ─────────────────────────────
    print("─" * 50)
    print("Baseline — untrained backbone (test set):")
    baseline_metrics = evaluate(model, test_loader)
    _print_metrics(baseline_metrics)
    print()

    # ── Optimizer + Scheduler (LLRD) ────────────────────────────────────────
    # Backbone trains at BACKBONE_LR (1e-5) to preserve cross-lingual pretraining.
    # Classification heads train at LR (3e-5) for faster task adaptation.
    no_decay = ["bias", "LayerNorm.weight"]
    backbone_params = list(model.backbone.named_parameters())
    head_params     = (
        list(model.intent_head.named_parameters()) +
        list(model.ner_head.named_parameters()) +
        list(model.guardrail_head.named_parameters())
    )
    optimizer = AdamW([
        {"params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
         "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in backbone_params if     any(nd in n for nd in no_decay)],
         "lr": BACKBONE_LR, "weight_decay": 0.0},
        {"params": [p for n, p in head_params     if not any(nd in n for nd in no_decay)],
         "lr": LR,          "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in head_params     if     any(nd in n for nd in no_decay)],
         "lr": LR,          "weight_decay": 0.0},
    ])
    print(f"Optimizer: LLRD — backbone_lr={BACKBONE_LR}, head_lr={LR}")
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Scheduler: linear warmup over {warmup_steps} steps "
          f"then decay to 0 over {total_steps - warmup_steps} steps")

    # ── Loss functions ───────────────────────────────────────────────────────
    intent_loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=LABEL_SMOOTHING,
    )
    ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Class-weighted guardrail loss — upweights the positive (dangerous) class
    n_pos = int((train_df["guardrail_label"] == 1).sum())
    n_neg = int((train_df["guardrail_label"] == 0).sum())
    if n_pos > 0 and n_neg > 0:
        pos_weight = (n_neg / n_pos) * 2
        weight = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(DEVICE)
        guardrail_loss_fn = nn.CrossEntropyLoss(weight=weight)
        print(f"Guardrail class weights: [neg=1.00, pos={pos_weight:.2f}]  "
              f"(n_pos={n_pos}, n_neg={n_neg})")
    else:
        guardrail_loss_fn = nn.CrossEntropyLoss()
        print("⚠️  Using unweighted guardrail loss — check data mix")

    # ── Training loop ────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    best_val_score   = -1.0   # composite: intent_macro_f1 + ner_entity_f1 + guardrail_recall
    patience_counter = 0
    history          = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            intent_labels  = batch["intent_label"].to(DEVICE)
            ner_tags       = batch["ner_tags"].to(DEVICE)
            guardrail_labs = batch["guardrail_label"].to(DEVICE)

            il, nl, gl = model(input_ids, attention_mask)

            loss_intent    = intent_loss_fn(il, intent_labels)
            loss_ner       = ner_loss_fn(nl.view(-1, len(NER_LABELS)), ner_tags.view(-1))
            loss_guardrail = guardrail_loss_fn(gl, guardrail_labs)
            loss           = loss_intent + loss_ner + loss_guardrail

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                      f"| loss={loss.item():.4f} "
                      f"(intent={loss_intent.item():.3f} "
                      f"ner={loss_ner.item():.3f} "
                      f"guard={loss_guardrail.item():.3f})")

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_metrics  = evaluate(model, val_loader)
        val_score    = (val_metrics["intent_macro_f1"]
                        + val_metrics["ner_entity_f1"]
                        + val_metrics["guardrail_recall"])   # same composite as notebook

        print(f"\nEpoch {epoch}/{EPOCHS}  avg_train_loss={avg_train_loss:.4f}  "
              f"val_composite={val_score:.4f}")
        _print_metrics(val_metrics, prefix="  val_")

        # -------------------  VALIDATION LOSS  -------------------
        # Re‑compute the loss on the whole validation set (no grads)
        model.eval()
        val_intent_loss = 0.0
        val_ner_loss    = 0.0
        val_guard_loss  = 0.0
        n_val_batches   = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                intent_labels  = batch["intent_label"].to(DEVICE)
                ner_tags       = batch["ner_tags"].to(DEVICE)
                guardrail_labs = batch["guardrail_label"].to(DEVICE)

                il, nl, gl = model(input_ids, attention_mask)

                val_intent_loss += intent_loss_fn(il, intent_labels).item()
                val_ner_loss    += ner_loss_fn(nl.view(-1, len(NER_LABELS)),
                                               ner_tags.view(-1)).item()
                val_guard_loss  += guardrail_loss_fn(gl, guardrail_labs).item()
                n_val_batches += 1
        val_loss = (val_intent_loss + val_ner_loss + val_guard_loss) / max(n_val_batches, 1)


        history.append({
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "val_score":  val_score,
            "val_loss":   val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Checkpoint + early stopping
        if val_score > best_val_score:
            best_val_score   = val_score
            patience_counter = 0
            ckpt_path        = OUT_DIR / "ieg_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✅ Best model saved (val_composite={val_score:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement — patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("  🛑 Early stopping triggered")
                break

        print()

    # ── Final test evaluation ────────────────────────────────────────────────
    print("─" * 50)
    print("Final Test Evaluation (best checkpoint):")
    model.load_state_dict(
        torch.load(OUT_DIR / "ieg_best.pt", map_location=DEVICE, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader)
    _print_metrics(test_metrics)

    # ── Holdout eval — real-scenario + cross-lingual breakdown ───────────────
    holdout_metrics = evaluate_holdout(model, tokenizer, intent2id)

    # Save final (last epoch) checkpoint as well
    torch.save(model.state_dict(), OUT_DIR / "ieg_adamw.pt")

    # ── eval_summary.json ────────────────────────────────────────────────────
    eval_summary = {
        "baseline_metrics": baseline_metrics,
        "test_metrics":     test_metrics,
        "training_history": history,
        **({"holdout_metrics": holdout_metrics} if holdout_metrics else {}),
        "config": {
            "model":             MODEL_NAME,
            "epochs_run":        len(history),
            "lr":                LR,
            "batch_size":        BATCH_SIZE,
            "weight_decay":      WEIGHT_DECAY,
            "label_smoothing":   LABEL_SMOOTHING,
            "warmup_ratio":      WARMUP_RATIO,
            "patience":          PATIENCE,
            "ner_labels":        NER_LABELS,
            "augment_toxicity":   ADD_TOXICITY_NEGATIVES,
        },
    }
    with open(OUT_DIR / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSaved artifacts to {OUT_DIR}:")
    for fname in ["ieg_best.pt", "ieg_adamw.pt", "label_maps.json", "eval_summary.json"]:
        p = OUT_DIR / fname
        size = f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "missing"
        print(f"  {fname:<28} {size}")

    print("\nDone.")


if __name__ == "__main__":
    train()