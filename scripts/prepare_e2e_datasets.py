import os
import re
import json
import time
import argparse
import random
import pandas as pd
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Setup roots
ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "kcc" / "kcc_combined_2020_2025.csv"

# Outputs
OUT_DIR = ROOT / "data" / "processed" / "kcc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRAIN = OUT_DIR / "kcc_train_99.csv"
OUT_EVAL = OUT_DIR / "kcc_eval_1.csv"
OUT_RAG = OUT_DIR / "kcc_chunks_rag.jsonl"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Configuration Constants
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller model for 4GB VRAM
LOG_INTERVAL = 10  # Number of samples before logging progress
EVAL_SAMPLES_PER_INTENT = 5  # Reduced number for local testing
TRAIN_SAMPLES_PER_INTENT = 4000  # Number of training samples per intent for IEG model


# Agronomic intent mapping based on the legacy notebooks
QTYPE_TO_INTENT = {
    'Plant Protection': 'disease_pest',
    'Weed Management': 'disease_pest',
    'Insect Management': 'disease_pest',
    'Pathogenic Disease Management': 'disease_pest',
    'Nutrient Management': 'nutrition_fertilizer',
    'Fertilizer Use and Availability': 'nutrition_fertilizer',
    'Nutrient Deficiency/Excessiveness Management': 'nutrition_fertilizer',
    'Bio-Pesticides and Bio-Fertilizers': 'nutrition_fertilizer',
    'Cultural Practices': 'cultivation_practice',
    'Varieties': 'cultivation_practice',
    'Varietal Selection': 'cultivation_practice',
    'Seeds and Planting Material': 'cultivation_practice',
    'Seeds': 'cultivation_practice',
    'Seed Sowing And Treatment': 'cultivation_practice',
    'Field Preparation': 'cultivation_practice',
    'Water Management': 'cultivation_practice',
    'Water Management, Micro Irrigation': 'cultivation_practice',
    'Irrigation Management': 'cultivation_practice',
    'Soil Testing': 'cultivation_practice',
    'Abiotic Stress Management': 'cultivation_practice',
    'Post Harvest Preservation': 'post_harvest_storage',
    'Storage': 'post_harvest_storage',
    'Cold Storage': 'post_harvest_storage',
    'Organic Farming': 'specialty_other',
    'Floriculture': 'specialty_other',
    'Beekeeping': 'specialty_other',
}

# Regexes for weather, market, policy (Temporal/Admin)
qt_weather_regex = r'\bweather\b|weather forecast|mausam|barish|baaris|badal|baadal|tapan|tappman|tapman|monsoon|\brain\b|rainfall|cloud|humidity|cyclone|toofan|\bole\b|hailstorm|dhoop|kohra|\bfog\b|\bfrost\b|\bpala\b|मौसम|बारिश|वर्षा|पूर्वानुमान|बादल|तापमान|मानसून|ओलावृष्टि|कोहरा|पाला'
ans_weather_regex = r'मौसम विभाग|मौसम का पूर्वानुमान|मौसम ज्ञात हो रहा|मौसम साफ रहेगा|बारिश होने की|बारिश की संभावना|वर्षा होने की|बादल छाए|बूंदाबांदी|तापमान|अधिकतम तापमान|न्यूनतम तापमान|मानसून|ओला|कोहरा|पाला|weather forecast|meteorological department|rain forecast|cloudy sky'

qt_policy_regex = r'beneficiary status|pm kisan|pm-kisan|pradhan mantri kisan|samman nidhi|kisan samman|प्रधानमंत्री किसान'
ans_policy_regex = r'प्रधानमंत्री किसान सम्मान|लाभार्थी स्थिति|स्टेटस में'

qt_market_regex = r'farmer asked price detail|\(modal price\):|mandi price|\bbhav\b|\bbhaav\b|rate of|price of|मंडी भाव|मोडल प्राइस'
ans_market_regex = r'मंडी भाव|मोडल प्राइस|modal price|\bquintal\b|क्विंटल|रु/क्विंटल|रु\./क्विंटल'

qt_admin_land_animal_regex = (
    r'khatauni|khasra|lekhpal|tehsildar|mutation|gata no|bhu-naksha|kisan credit card|kcc loan|bank branch|patwari|'
    r'kisaan call center|toll free|helpline|complaint|district agriculture officer|up krishi nideshak|adhikari se sampark|1800-|'
    r'solar pump|boring|tube well|tubewell|subsidy|subsidies|anudan|farm machinery bank|chc application|custom hiring|tractor|rotavator|electricity|'
    r'\bcow\b|\bcattle\b|\bgoat\b|\bpoultry\b|\bhen\b|\bdog\b|\bmilk\b|\bmastitis\b|dairy|veterinary|pashudhan|pashu|fish\b|fishery|thanaula|'
    r'खतौनी|खसरा|लेखपाल|तहसीलदार|दाखिल ख़ारिज|भू-नक्शा|पटवारी|टोल फ्री|हेल्पलाइन|शिकायत करें|जिला कृषि अधिकारी से संपर्क|'
    r'सब्सिडी|अनुदान|सोलर पम्प|बोरिंग|ट्यूबवेल|बिजली|गाय|भैंस|बकरी|मुर्गी|थनैला|दूध|पशु|मछली'
)
ans_admin_land_animal_regex = (
    r'खतौनी|खसरा|लेखपाल|तहसीलदार|दाखिल ख़ारिज|दाखिल खारिज|भू-नक्शा|पटवारी|किसान क्रेडिट कार्ड|बैंक शाखा|'
    r'1800-|टोल फ्री|हेल्पलाइन|शिकायत करें|जिला कृषि अधिकारी से संपर्क|उप कृषि निदेशक कार्यालय|'
    r'सब्सिडी|अनुदान|सोलर पम्प|सोलर पंप|बोरिंग|ट्यूबवेल|बिजली|फार्म मशीनरी|कृषि यंत्रों|'
    r'गाय|भैंस|बकरी|मुर्गी|थनैला|दूध|पशु|मछली|veterinary|dairy'
)

agri_categories = [
    'Cereals', 'Pulses', 'Oilseeds', 'Vegetables', 'Fruits',
    'Millets', 'Sugar and Starch Crops', 'Medicinal and Aromatic Plants',
    'Condiments and Spices', 'Fodder Crops', 'Flowers', 'Fiber Crops',
    'Plantation Crops', 'Green Manure', 'Drug and Narcotics', 'Sugar Crop', 'Fiber Crop',
    'Annual Spices', 'Forage', 'Plantation Fruit Crop', 
]


def _has_pattern(dataframe, qt_pattern, ans_pattern):
    return (
        dataframe['QueryText'].fillna('').str.contains(qt_pattern, case=False) |
        dataframe['KccAns'].fillna('').str.contains(ans_pattern, case=False)
    )


def build_alias_map(series):
    alias_map = {}
    for val in series.dropna().unique():
        val = str(val).strip()
        if not val:
            continue
        forms = [val.lower()]
        m = re.match(r'^(.*?)\s*\((.*?)\)\s*$', val)
        if m:
            main, bracket = m.group(1).strip(), m.group(2).strip()
            forms.append(main.lower())
            forms.extend(a.strip().lower() for a in bracket.split('/'))
        alias_map[val] = [f for f in set(forms) if len(f) > 2]
    return alias_map


def find_entity_spans(text, canonical_value, alias_forms):
    text_lower = text.lower()
    for form in sorted(alias_forms, key=len, reverse=True):
        idx = text_lower.find(form)
        if idx != -1:
            return idx, idx + len(form)
    return None



def deduplicate_keep_recent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-stage deduplication that retains the most recent row.

    Stage 1 – Exact dedup
    ---------------------
    Drops rows where both QueryText AND KccAns are identical strings.
    Within each exact-duplicate group the row with the most recent
    CreatedOn timestamp is kept.

    Stage 2 (structural near-dedup) was evaluated and removed:
    99.1% of candidate groups had >1 unique KccAns, meaning the same
    question received different expert answers across years (2020-2025).
    Collapsing those would silently discard valid answer diversity.
    """
    before = len(df)

    # --- sort by CreatedOn so the most recent row is always first ---
    if 'CreatedOn' in df.columns:
        df = df.copy()
        df['_created_dt'] = pd.to_datetime(df['CreatedOn'], errors='coerce')
        df = df.sort_values('_created_dt', ascending=False)
        df = df.drop(columns=['_created_dt'])

    # ── Stage 1: exact (QueryText, KccAns) dedup ────────────────────────
    df = df.drop_duplicates(subset=['QueryText', 'KccAns'], keep='first')
    after_s1 = len(df)
    print(f"  Exact dedup (Stage-1) : removed {before - after_s1:,} rows  "
          f"({before:,} -> {after_s1:,})")

    return df.reset_index(drop=True)


def stratified_sample(df, target_n, combo_cols):
    """
    Sample target_n rows using a stratified approach based on the frequency
    of specified combinations (e.g. DistrictName, Crop, Category).
    - 35% from the 90-100th percentile of combo frequencies
    - 50% from the 10-90th percentile of combo frequencies
    - 15% from the 0-10th percentile of combo frequencies
    
    Sampling is done by first randomly selecting combos within the percentile bucket,
    and then randomly sampling 1 row from each selected combo.
    """
    if len(df) <= target_n:
        return df.copy()
        
    df = df.copy()
    
    df['combo'] = df[combo_cols[0]].astype(str)
    for col in combo_cols[1:]:
        df['combo'] = df['combo'] + "_" + df[col].astype(str)
        
    combo_counts = df['combo'].value_counts()
    
    p90 = combo_counts.quantile(0.90)
    p10 = combo_counts.quantile(0.10)
    
    def get_strata(count):
        if count >= p90:
            return 'high'
        elif count <= p10:
            return 'low'
        else:
            return 'med'
            
    combo_strata = combo_counts.apply(get_strata)
    
    # Calculate sample allocations (e.g. 5 samples -> 2 high, 2 med, 1 low)
    n_high = int(round(target_n * 0.35))
    n_low = int(round(target_n * 0.15))
    n_med = target_n - n_high - n_low
    
    samples = []
    
    for strata, n in [('high', n_high), ('med', n_med), ('low', n_low)]:
        if n <= 0:
            continue
            
        strata_combos = combo_strata[combo_strata == strata].index.tolist()
        
        # If bucket is empty, we'll fall short and make it up at the end
        if not strata_combos:
            continue
            
        # Randomly select 'n' combos (with replacement if we need more samples than available combos)
        import random
        replace = len(strata_combos) < n
        selected_combos = pd.Series(strata_combos).sample(n=n, replace=replace, random_state=RANDOM_SEED).tolist()
        
        from collections import Counter
        combo_counts_to_sample = Counter(selected_combos)
        
        # Efficiently sample using groupby instead of full dataframe scans
        groups = df[df['combo'].isin(combo_counts_to_sample.keys())].groupby('combo')
        for c, k in combo_counts_to_sample.items():
            if c in groups.groups:
                combo_rows = groups.get_group(c)
                k_actual = min(k, len(combo_rows))
                if k_actual > 0:
                    samples.append(combo_rows.sample(n=k_actual, random_state=RANDOM_SEED))
            
    if samples:
        res = pd.concat(samples)
        res = res.drop_duplicates()  # Fix the duplicate sampling bug
    else:
        res = pd.DataFrame(columns=df.columns)
    
    # Make up any shortfall randomly from the remaining rows
    shortfall = target_n - len(res)
    if shortfall > 0:
        remaining_df = df.drop(res.index, errors='ignore')
        if len(remaining_df) >= shortfall:
            res = pd.concat([res, remaining_df.sample(n=shortfall, random_state=RANDOM_SEED)])
        else:
            res = pd.concat([res, remaining_df])
            
    return res.drop(columns=['combo'])


def load_qwen_for_data_gen(device_id="cuda"):
    print(f"Loading {LLM_MODEL_ID} on {device_id} for E2E dataset augmentation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_id,
    )
    return tokenizer, model


def apply_multi_turn_llm_worker(args):
    df_chunk, device_id = args
    tokenizer, model = load_qwen_for_data_gen(device_id)
    return apply_multi_turn_llm(df_chunk, tokenizer, model, device_id)


def apply_multi_turn_llm(df, tokenizer, model, device_id="cuda"):
    """
    Generates single/multi-turn variations for the eval split.
    Enforces deterministic language distribution: 33% English, 33% Devanagari Hindi, 33% Hinglish
    by directly embedding the target language requirement into the system prompt.
    """
    print(f"Applying Qwen generation on {len(df)} rows for eval split...")
    results = []
    
    # Must use left-padding for batched decoder-only generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    langs = ["English", "Devanagari Hindi", "Hinglish (Hindi written in English alphabet)"]
    
    prompts = []
    for i, row in df.iterrows():
        target_lang = langs[i % 3]
        
        q = row['QueryText']
        a = row['KccAns']
        
        user_prompt = (
            "You are an expert data annotator. The following is a raw query and answer from the Kisan Call Centre.\n"
            f"Query: {q}\n"
            f"Answer: {a}\n\n"
            f"Rewrite this Q&A into a 2-turn conversation in {target_lang}. Make the user sound like a real farmer. "
            "Output ONLY valid JSON in this exact format:\n"
            "[\n"
            "  {\"role\": \"user\", \"content\": \"...\"},\n"
            "  {\"role\": \"assistant\", \"content\": \"...\"},\n"
            "  {\"role\": \"user\", \"content\": \"...\"},\n"
            "  {\"role\": \"assistant\", \"content\": \"...\"}\n"
            "]"
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that strictly follows output format constraints."},
            {"role": "user", "content": user_prompt}
        ]
        
        # Qwen2.5 works best with its chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append((i, row, text, target_lang))
        
    batch_size = 4  # Appropriate batch size for 1.5B model on 4GB VRAM
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_texts = [b[2] for b in batch]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device_id)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, temperature=0.7, do_sample=True, top_p=0.9
            )
        
        # Properly decode each output in the batch
        for j, b in enumerate(batch):
            row = b[1].copy()
            target_lang = b[3]
            q = row['QueryText']
            a = row['KccAns']
            
            text_out = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            try:
                # Regex to pull JSON array
                json_str = re.search(r'\[.*\]', text_out, re.DOTALL).group(0)
                json_parsed = json.loads(json_str)
                row['multi_turn_json'] = json.dumps(json_parsed, ensure_ascii=False)
            except Exception:
                # Fallback if json parsing fails
                row['multi_turn_json'] = json.dumps([
                    {"role": "user", "content": f"(Translated/Rephrased manually to {target_lang}) {q}"},
                    {"role": "assistant", "content": a}
                ])
                
            row['language'] = target_lang
            results.append(row)
        
        if (i+batch_size) % LOG_INTERVAL < batch_size:
            print(f"Processed {min(i+batch_size, len(prompts))} rows...")
            try:
                print(f"Latest Sample generated:\n{results[-1]['multi_turn_json']}\n{'-'*40}")
            except UnicodeEncodeError:
                print(f"Latest Sample generated: [Unicode Output Hidden to Prevent Console Encoding Error]\n{'-'*40}")
            
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip Qwen multi-turn generation (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("Executing prepare_e2e_datasets.py")
    print("=" * 60)
    
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_CSV}")
        
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"Loaded {len(df):,} raw rows.")
    
    if 'QueryType' in df.columns:
        df['QueryType'] = df['QueryType'].str.strip()

    # Apply standard category filter
    std_agri_mask = df['Category'].str.contains('|'.join(agri_categories), case=False, na=False)
    
    # We want to identify the exact rows that are weather, market, policy 
    mask_weather = _has_pattern(df, qt_weather_regex, ans_weather_regex)
    mask_policy = _has_pattern(df, qt_policy_regex, ans_policy_regex)
    mask_market = _has_pattern(df, qt_market_regex, ans_market_regex)
    
    # Find non-agronomic admin/animal that we actually DO want to drop
    mask_admin_animal = _has_pattern(df, qt_admin_land_animal_regex, ans_admin_land_animal_regex)
    ignore_pat = r'fodder|chara|straw|भूसा|चारा|cowpea|नील गाय|नीलगाय|neelgai|blue bull'
    mask_animal_protected = _has_pattern(df, ignore_pat, ignore_pat)
    drop_admin_mask = mask_admin_animal & (~mask_animal_protected)

    # Assign mapped intent labels
    df['intent_label'] = df['QueryType'].map(QTYPE_TO_INTENT).fillna('other')
    
    # Overwrite intent labels for the temporal categories
    df.loc[mask_weather, 'intent_label'] = 'weather'
    df.loc[mask_policy, 'intent_label'] = 'policy'
    df.loc[mask_market, 'intent_label'] = 'market'
    
    # Filter dataset
    filtered_df = df[std_agri_mask & (~drop_admin_mask)].copy()
    print(f"Post-filter dataset: {len(filtered_df):,} rows.")

    # Drop NA query text
    filtered_df = filtered_df.dropna(subset=['QueryText', 'KccAns'])
    
    # Strictly remove any rows that are completely empty or just whitespace
    filtered_df = filtered_df[filtered_df['QueryText'].astype(str).str.strip() != '']
    filtered_df = filtered_df[filtered_df['KccAns'].astype(str).str.strip() != '']
    
    invalid_queries = ['INCOMPLETE CALL', 'IRRELEVANT CALL']
    filtered_df = filtered_df[~filtered_df['QueryText'].astype(str).str.strip().str.upper().isin(invalid_queries)]
    filtered_df['Crop'] = filtered_df['Crop'].fillna('Unknown')
    filtered_df['DistrictName'] = filtered_df['DistrictName'].fillna('Unknown')

    # ── Deduplication: keep most recent row per duplicate group ──────────
    print("\n--- Deduplicating dataset (keeping most recent by year) ---")
    filtered_df = deduplicate_keep_recent(filtered_df)
    print(f"Post-dedup dataset: {len(filtered_df):,} rows.")

    print("\n--- Generating Char Spans for NER ---")
    crop_alias_map = build_alias_map(filtered_df['Crop'])
    district_alias_map = build_alias_map(filtered_df['DistrictName'])
    
    # Fast vectorized loop instead of slow apply(axis=1)
    char_spans_list = []
    for text, c, d in zip(filtered_df['QueryText'], filtered_df['Crop'], filtered_df['DistrictName']):
        text_str = str(text)
        spans = []
        if c and c in crop_alias_map:
            span = find_entity_spans(text_str, c, crop_alias_map[c])
            if span:
                spans.append((span[0], span[1], "CROP"))
        if d and d in district_alias_map:
            span = find_entity_spans(text_str, d, district_alias_map[d])
            if span:
                spans.append((span[0], span[1], "DISTRICT"))
        char_spans_list.append(json.dumps(spans))
        
    filtered_df['_char_spans'] = char_spans_list

    print("\n--- Sampling for Evaluation (kcc_eval_1.csv) ---")
    eval_dfs = []
    train_dfs = []
    
    intent_classes = filtered_df['intent_label'].unique()
    for intent in intent_classes:
        intent_df = filtered_df[filtered_df['intent_label'] == intent]
        
        # 1. E2E eval pool
        eval_sample = stratified_sample(intent_df, EVAL_SAMPLES_PER_INTENT, ['QueryType', 'Crop', 'DistrictName'])
        eval_dfs.append(eval_sample)
        
        # 2. Drop eval samples from intent_df
        rem_df = intent_df.drop(eval_sample.index)
        
        # 3. Train pool
        train_sample = stratified_sample(rem_df, TRAIN_SAMPLES_PER_INTENT, ['QueryType', 'Crop', 'DistrictName'])
        train_dfs.append(train_sample)

    eval_df_concat = pd.concat(eval_dfs)
    eval_used_indices = eval_df_concat.index
    
    eval_df_final = eval_df_concat.reset_index(drop=True)
    train_df_final = pd.concat(train_dfs).reset_index(drop=True)

    print(f"kcc_eval_1.csv raw target: {len(eval_df_final):,} rows.")
    print(f"kcc_train_99.csv raw target: {len(train_df_final):,} rows.")

    print("\n" + "=" * 120)
    print("DATASET STATS: POST-DEDUP, POST-SAMPLE")
    print("=" * 120)

    total_rows = len(filtered_df)
    intent_counts = filtered_df['intent_label'].value_counts()

    # -- [1] Rows per intent --------------------------------------------------
    print("\n[1] Rows per intent (post-dedup pool)")
    print(f"  {'Intent':<25} {'Rows':>8}  {'Share':>6}")
    print("  " + "-" * 44)
    for intent in intent_classes:
        n = intent_counts.get(intent, 0)
        print(f"  {intent:<25} {n:>8,}  {n/total_rows*100:>5.1f}%")
    print(f"  {'TOTAL':<25} {total_rows:>8,}  100.0%")

    # -- [2] Per-intent: independent column unique counts --------------------
    print("\n[2] Unique values per column within each intent")
    print(f"  {'Intent':<25} {'Crops':>8} {'Districts':>10} {'Categories':>12}")
    print("  " + "-" * 60)
    for intent in intent_classes:
        sub = filtered_df[filtered_df['intent_label'] == intent]
        print(f"  {intent:<25} {sub['Crop'].nunique():>8,}"
              f" {sub['DistrictName'].nunique():>10,}"
              f" {sub['Category'].nunique():>12,}")

    # -- [3] Per-intent: 2-combo unique counts --------------------------------
    print("\n[3] Unique 2-combos within each intent")
    print(f"  {'Intent':<25} {'Crop x Dist':>12} {'Crop x Cat':>12} {'Dist x Cat':>12}")
    print("  " + "-" * 66)
    for intent in intent_classes:
        sub = filtered_df[filtered_df['intent_label'] == intent].copy()
        cd = (sub['Crop'].astype(str) + "|" + sub['DistrictName'].astype(str)).nunique()
        cc = (sub['Crop'].astype(str) + "|" + sub['Category'].astype(str)).nunique()
        dc = (sub['DistrictName'].astype(str) + "|" + sub['Category'].astype(str)).nunique()
        print(f"  {intent:<25} {cd:>12,} {cc:>12,} {dc:>12,}")

    # -- [4] Per-intent: Unique combos + sample captures ---------------
    print("\n[4] Unique Combos + sample captures")
    print(f"  {'Intent':<25} {'Cat x Crop x Dist':>18} {'QT x Crop x Dist':>18} {'QT x Cat x Crop x Dist':>22} {'Train sampled':>15} {'Eval sampled':>14}")
    print("  " + "-" * 111)
    for intent in intent_classes:
        sub = filtered_df[filtered_df['intent_label'] == intent].copy()
        
        cat_crop_dist = (sub['Category'].astype(str) + "|" +
                         sub['Crop'].astype(str) + "|" +
                         sub['DistrictName'].astype(str)).nunique()
                         
        qt_crop_dist = (sub['QueryType'].astype(str) + "|" +
                        sub['Crop'].astype(str) + "|" +
                        sub['DistrictName'].astype(str)).nunique()
                        
        qt_cat_crop_dist = (sub['QueryType'].astype(str) + "|" +
                            sub['Category'].astype(str) + "|" +
                            sub['Crop'].astype(str) + "|" +
                            sub['DistrictName'].astype(str)).nunique()
                            
        tr_n = len(train_df_final[train_df_final['intent_label'] == intent])
        ev_n = len(eval_df_final[eval_df_final['intent_label'] == intent])
        print(f"  {intent:<25} {cat_crop_dist:>18,} {qt_crop_dist:>18,} {qt_cat_crop_dist:>22,} {tr_n:>15,} {ev_n:>14,}")
    print("  " + "-" * 111)
    print(f"  {'TOTAL':<25} {'':>18} {'':>18} {'':>22} {len(train_df_final):>15,} {len(eval_df_final):>14,}")

    print("\n" + "=" * 120)

    if not args.skip_llm:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Found {num_gpus} GPUs, splitting eval generation workload across processes...")
            mp.set_start_method('spawn', force=True)
            
            chunk_size = int(np.ceil(len(eval_df_final) / num_gpus))
            splits = [eval_df_final.iloc[i:i + chunk_size] for i in range(0, len(eval_df_final), chunk_size)]
            worker_args = [(split, f"cuda:{i}") for i, split in enumerate(splits)]
            
            with mp.Pool(processes=num_gpus) as pool:
                results = pool.map(apply_multi_turn_llm_worker, worker_args)
                
            eval_df_final = pd.concat(results).reset_index(drop=True)
        else:
            device = "cuda:0" if num_gpus > 0 else "cpu"
            print(f"Single GPU/CPU detected. Running sequentially on {device}...")
            tokenizer, model = load_qwen_for_data_gen(device)
            eval_df_final = apply_multi_turn_llm(eval_df_final, tokenizer, model, device)
    else:
        print("Skipping Qwen multi-turn generation (testing mode).")
        eval_df_final['multi_turn_json'] = "[]"
        eval_df_final['language'] = "English"

    eval_df_final.to_csv(OUT_EVAL, index=False)
    print(f"Saved {OUT_EVAL}")

    train_df_final.to_csv(OUT_TRAIN, index=False)
    print(f"Saved {OUT_TRAIN}")
    
    print("\n--- Building RAG chunks (kcc_chunks_rag.jsonl) ---")
    # Remaining rows after taking Eval (keep Train in RAG)
    rag_base = filtered_df.drop(eval_used_indices, errors='ignore')
    
    # RAG shouldn't contain weather, market, policy (they are temporal and shouldn't be retrieved for static advice)
    rag_base = rag_base[~rag_base['intent_label'].isin(['weather', 'market', 'policy'])]
    
    # Simple JSONL chunking (simplified chunk generation matching legacy format)
    rag_records = []
    for _, row in rag_base.iterrows():
        rag_records.append({
            "id": str(row.name),
            "question": str(row['QueryText']),
            "answer": str(row['KccAns']),
            "metadata": {
                "crop": row['Crop'],
                "category": row['Category'],
                "district": row['DistrictName']
            }
        })
        
    with open(OUT_RAG, 'w', encoding='utf-8') as f:
        for record in rag_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Saved {OUT_RAG} with {len(rag_records):,} chunks.")
    print("\nDone.")

if __name__ == "__main__":
    main()
