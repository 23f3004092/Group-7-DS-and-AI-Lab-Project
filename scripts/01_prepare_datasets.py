import os
import re
import json
import time
import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

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
    before = len(df)
    if 'CreatedOn' in df.columns:
        df = df.copy()
        df['_created_dt'] = pd.to_datetime(df['CreatedOn'], errors='coerce')
        df = df.sort_values('_created_dt', ascending=False)
        df = df.drop(columns=['_created_dt'])
    df = df.drop_duplicates(subset=['QueryText', 'KccAns'], keep='first')
    after_s1 = len(df)
    print(f"  Exact dedup (Stage-1) : removed {before - after_s1:,} rows  ({before:,} -> {after_s1:,})")
    return df.reset_index(drop=True)

def stratified_sample(df, target_n, combo_cols):
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
    
    n_high = int(round(target_n * 0.35))
    n_low = int(round(target_n * 0.15))
    n_med = target_n - n_high - n_low
    
    samples = []
    
    for strata, n in [('high', n_high), ('med', n_med), ('low', n_low)]:
        if n <= 0:
            continue
            
        strata_combos = combo_strata[combo_strata == strata].index.tolist()
        
        if not strata_combos:
            continue
            
        import random
        replace = len(strata_combos) < n
        selected_combos = pd.Series(strata_combos).sample(n=n, replace=replace, random_state=RANDOM_SEED).tolist()
        
        from collections import Counter
        combo_counts_to_sample = Counter(selected_combos)
        
        groups = df[df['combo'].isin(combo_counts_to_sample.keys())].groupby('combo')
        for c, k in combo_counts_to_sample.items():
            if c in groups.groups:
                combo_rows = groups.get_group(c)
                k_actual = min(k, len(combo_rows))
                if k_actual > 0:
                    samples.append(combo_rows.sample(n=k_actual, random_state=RANDOM_SEED))
            
    if samples:
        res = pd.concat(samples)
        res = res.drop_duplicates()
    else:
        res = pd.DataFrame(columns=df.columns)
    
    shortfall = target_n - len(res)
    if shortfall > 0:
        remaining_df = df.drop(res.index, errors='ignore')
        if len(remaining_df) >= shortfall:
            res = pd.concat([res, remaining_df.sample(n=shortfall, random_state=RANDOM_SEED)])
        else:
            res = pd.concat([res, remaining_df])
            
    return res.drop(columns=['combo'])

def main():
    print("=" * 60)
    print("Executing 01_prepare_datasets.py")
    print("=" * 60)
    
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_CSV}")
        
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"Loaded {len(df):,} raw rows.")
    
    if 'QueryType' in df.columns:
        df['QueryType'] = df['QueryType'].str.strip()

    std_agri_mask = df['Category'].str.contains('|'.join(agri_categories), case=False, na=False)
    
    mask_weather = _has_pattern(df, qt_weather_regex, ans_weather_regex)
    mask_policy = _has_pattern(df, qt_policy_regex, ans_policy_regex)
    mask_market = _has_pattern(df, qt_market_regex, ans_market_regex)
    
    mask_admin_animal = _has_pattern(df, qt_admin_land_animal_regex, ans_admin_land_animal_regex)
    ignore_pat = r'fodder|chara|straw|भूसा|चारा|cowpea|नील गाय|नीलगाय|neelgai|blue bull'
    mask_animal_protected = _has_pattern(df, ignore_pat, ignore_pat)
    drop_admin_mask = mask_admin_animal & (~mask_animal_protected)

    df['intent_label'] = df['QueryType'].map(QTYPE_TO_INTENT).fillna('other')
    df.loc[mask_weather, 'intent_label'] = 'weather'
    df.loc[mask_policy, 'intent_label'] = 'policy'
    df.loc[mask_market, 'intent_label'] = 'market'
    
    filtered_df = df[std_agri_mask & (~drop_admin_mask)].copy()
    print(f"Post-filter dataset: {len(filtered_df):,} rows.")

    filtered_df = filtered_df.dropna(subset=['QueryText', 'KccAns'])
    
    # Strictly remove any rows that are completely empty or just whitespace
    filtered_df = filtered_df[filtered_df['QueryText'].astype(str).str.strip() != '']
    filtered_df = filtered_df[filtered_df['KccAns'].astype(str).str.strip() != '']
    
    invalid_queries = ['INCOMPLETE CALL', 'IRRELEVANT CALL']
    filtered_df = filtered_df[~filtered_df['QueryText'].astype(str).str.strip().str.upper().isin(invalid_queries)]
    filtered_df['Crop'] = filtered_df['Crop'].fillna('Unknown')
    filtered_df['DistrictName'] = filtered_df['DistrictName'].fillna('Unknown')

    print("\n--- Deduplicating dataset (keeping most recent by year) ---")
    filtered_df = deduplicate_keep_recent(filtered_df)
    print(f"Post-dedup dataset: {len(filtered_df):,} rows.")

    print("\n--- Generating Char Spans for NER ---")
    crop_alias_map = build_alias_map(filtered_df['Crop'])
    district_alias_map = build_alias_map(filtered_df['DistrictName'])
    
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
        
        eval_sample = stratified_sample(intent_df, EVAL_SAMPLES_PER_INTENT, ['QueryType', 'Crop', 'DistrictName'])
        eval_dfs.append(eval_sample)
        
        rem_df = intent_df.drop(eval_sample.index)
        
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

    print("\n[1] Rows per intent (post-dedup pool)")
    print(f"  {'Intent':<25} {'Rows':>8}  {'Share':>6}")
    print("  " + "-" * 44)
    for intent in intent_classes:
        n = intent_counts.get(intent, 0)
        print(f"  {intent:<25} {n:>8,}  {n/total_rows*100:>5.1f}%")
    print(f"  {'TOTAL':<25} {total_rows:>8,}  100.0%")

    eval_df_final.to_csv(OUT_EVAL, index=False)
    print(f"\nSaved un-augmented evaluation split to {OUT_EVAL}")

    train_df_final.to_csv(OUT_TRAIN, index=False)
    print(f"Saved training split to {OUT_TRAIN}")
    
    print("\n--- Building RAG chunks (kcc_chunks_rag.jsonl) ---")
    rag_base = filtered_df.drop(eval_used_indices, errors='ignore')
    rag_base = rag_base[~rag_base['intent_label'].isin(['weather', 'market', 'policy'])]

    def _detect_language(text, sample_chars=1000):
        """Same logic as detect_language() in build_rag_artifacts.py.
        KCC chunks are English question + Devanagari answer -> 'mixed'.
        Returns: 'en' | 'hi' | 'mixed'
        """
        import re as _re
        s = text[:sample_chars]
        dev = len(_re.findall(r'[\u0900-\u097F]', s))
        lat = len(_re.findall(r'[a-zA-Z]', s))
        total = max(dev + lat, 1)
        if dev / total > 0.15 and lat / total > 0.15:
            return "mixed"
        return "hi" if dev / total > 0.3 else "en"

    def _season_from_month(month):
        """Map calendar month -> agronomic season (Kharif / Rabi / Zaid)."""
        if month in (6, 7, 8, 9, 10):
            return "Kharif"
        elif month in (11, 12, 1, 2, 3):
            return "Rabi"
        elif month in (4, 5):
            return "Zaid"
        return "unknown"

    rag_records = []
    for _, row in rag_base.iterrows():
        q = str(row['QueryText']).strip()
        a = str(row['KccAns']).strip()
        text = f"Question: {q}\nAnswer: {a}"

        # Year / month from CreatedOn (best-effort)
        year_val, month_val, season_val = None, None, "unknown"
        if 'CreatedOn' in row and pd.notna(row['CreatedOn']):
            try:
                dt = pd.to_datetime(row['CreatedOn'], errors='coerce')
                if pd.notna(dt):
                    year_val  = int(dt.year)
                    month_val = int(dt.month)
                    season_val = _season_from_month(month_val)
            except Exception:
                pass

        crop_val     = str(row.get('Crop', '') or '').strip() or None
        district_val = str(row.get('DistrictName', '') or '').strip() or None
        block_val    = str(row.get('Block', '') or '').strip() or "unknown"
        qtype_val    = str(row.get('QueryType', '') or '').strip() or None
        cat_val      = str(row.get('Category', '') or '').strip() or None

        rag_records.append({
            "text": text,
            "metadata": {
                "crop":       crop_val,
                "district":   district_val,
                "block":      block_val,
                "season":     season_val,
                "query_type": qtype_val,
                "category":   cat_val,
                "year":       year_val,
                "month":      month_val,
                "language":   _detect_language(text),
            },
            "chunk_number":  1,
            "total_chunks":  1,
        })

    with open(OUT_RAG, 'w', encoding='utf-8') as f:
        for record in rag_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {OUT_RAG} with {len(rag_records):,} chunks.")
    print("  Sample record:")
    if rag_records:
        import sys
        sys.stdout.buffer.write((json.dumps(rag_records[0], indent=4, ensure_ascii=False) + '\n').encode('utf-8', errors='replace'))
    print("\nDone with Part 1: Preprocessing and Splitting.")


if __name__ == "__main__":
    main()
