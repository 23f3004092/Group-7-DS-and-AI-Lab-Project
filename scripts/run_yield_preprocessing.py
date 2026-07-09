#!/usr/bin/env python3
"""
run_yield_preprocessing.py — Complete Preprocessing, Spatial Harmonization & Feature Engineering
for UP District Yield Dataset (1997-2023).

Performs:
  1. Parent-District Spatial Harmonization for bifurcated districts (Amethi, Sambhal, Hapur, Shamli)
  2. Imputation of bureaucratic zero-production reporting anomalies
  3. Feature Engineering (NPK intensity, balance ratio, rainfall anomaly, thermal stress, irrigation security)
  4. Chronological Train / Validation / Test splitting (1997-2018 / 2019-2020 / 2021-2023)
  5. Exports to data/final/yield/train_yield.csv, val_yield.csv, test_yield.csv
"""

import os
import csv
import math
from collections import defaultdict

# District bifurcation mapping: child district -> parent district for back-casting pre-formation years
PARENT_DISTRICT_MAP = {
    "Amethi": "Sultanpur",
    "Sambhal": "Moradabad",
    "Hapur": "Meerut",
    "Shamli": "Muzaffarnagar"
}


def main():
    in_path = "data/raw/yield/up_district_yield_apy_1997_2023.csv"
    records = []
    with open(in_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"[*] Loaded {len(records)} raw yield records.")

    # 1. Compute Zone-Season Median Yields for Imputing Zero Reporting Anomalies
    zone_season_yields = defaultdict(list)
    for r in records:
        y = float(r["Yield_Kg_Ha"])
        if y > 0:
            zone_season_yields[(r["Agro_Climatic_Zone"], r["Crop"])].append(y)

    medians = {}
    for key, vals in zone_season_yields.items():
        s = sorted(vals)
        medians[key] = s[len(s) // 2]

    # Clean zero anomalies
    imputed_count = 0
    for r in records:
        if float(r["Yield_Kg_Ha"]) == 0.0:
            r["Yield_Kg_Ha"] = str(round(medians[(r["Agro_Climatic_Zone"], r["Crop"])], 1))
            imputed_count += 1
    print(f"[+] Imputed {imputed_count} bureaucratic zero-production reporting anomalies using zone medians.")

    # 2. Spatial Harmonization: Backcast missing bifurcated districts (Amethi, Sambhal, Hapur, Shamli)
    # Index records by (District_Name, Crop_Year, Season)
    rec_index = {(r["District_Name"], r["Crop_Year"], r["Season"]): r for r in records}
    years = [f"{yr}-{str(yr+1)[-2:]}" for yr in range(1997, 2024)]
    seasons = [("Kharif", "Rice"), ("Rabi", "Wheat")]

    apportioned_added = 0
    for child, parent in PARENT_DISTRICT_MAP.items():
        for yr in years:
            for season, crop in seasons:
                if (child, yr, season) not in rec_index:
                    parent_rec = rec_index.get((parent, yr, season))
                    if parent_rec:
                        # Create apportioned child record with proportional area (~28% of parent area)
                        child_rec = dict(parent_rec)
                        child_rec["District_Name"] = child
                        child_rec["Area_Sown"] = str(round(float(parent_rec["Area_Sown"]) * 0.28, 1))
                        child_rec["Production_Total"] = str(round(float(parent_rec["Production_Total"]) * 0.28, 2))
                        # Yield (kg/ha), weather, and input rates remain identical to parent baseline
                        records.append(child_rec)
                        apportioned_added += 1

    print(f"[+] Spatial Harmonization: Apportioned {apportioned_added} pre-formation records for bifurcated districts.")
    print(f"[+] Complete harmonized dataset size: {len(records)} records (75 districts x 27 yrs x 2 seasons = 4,050).")

    # 3. Compute District Long-Term Mean Rainfall for Anomaly Feature
    dist_season_rain = defaultdict(list)
    for r in records:
        dist_season_rain[(r["District_Name"], r["Season"])].append(float(r["Precip_Seasonal_mm"]))
    mean_rain = {k: sum(v)/len(v) for k, v in dist_season_rain.items()}

    # 4. Feature Engineering
    processed_records = []
    for r in records:
        area = max(1.0, float(r["Area_Sown"]))
        fn = float(r["Fertilizer_N_Tonnes"])
        fp = float(r["Fertilizer_P_Tonnes"])
        fk = float(r["Fertilizer_K_Tonnes"])

        npk_intensity = ((fn + fp + fk) * 1000.0) / area
        npk_balance = fn / max(0.1, fp)
        rain_mean = mean_rain[(r["District_Name"], r["Season"])]
        rain_anomaly = ((float(r["Precip_Seasonal_mm"]) - rain_mean) / rain_mean) * 100.0
        thermal_stress = float(r["Temp_Max_Avg"]) * (1.0 + 0.05 * float(r["Heatwave_Days"]))
        irrig_score = 0.4 * float(r["Net_Irrigated_Pct"]) + 0.6 * float(r["Tubewell_Irrig_Pct"])

        # Extract start year integer for temporal splitting
        start_yr = int(r["Crop_Year"].split("-")[0])

        new_r = dict(r)
        new_r["NPK_Total_Intensity_Kg_Ha"] = round(npk_intensity, 2)
        new_r["NPK_Balance_Ratio"] = round(npk_balance, 2)
        new_r["Rainfall_Anomaly_Pct"] = round(rain_anomaly, 2)
        new_r["Thermal_Stress_Index"] = round(thermal_stress, 2)
        new_r["Irrigation_Security_Score"] = round(irrig_score, 2)
        new_r["Start_Year"] = start_yr
        processed_records.append(new_r)

    # 5. Chronological Train / Val / Test Partitioning
    train_recs = [r for r in processed_records if r["Start_Year"] <= 2018]
    val_recs = [r for r in processed_records if 2019 <= r["Start_Year"] <= 2020]
    test_recs = [r for r in processed_records if r["Start_Year"] >= 2021]

    out_dir = os.path.join("data", "final", "yield")
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "State_Name", "District_Name", "Agro_Climatic_Zone", "Crop_Year", "Season", "Crop",
        "Area_Sown", "Production_Total", "Yield_Kg_Ha",
        "Precip_Seasonal_mm", "Rain_Days_Extreme", "Temp_Max_Avg", "Heatwave_Days",
        "Fertilizer_N_Tonnes", "Fertilizer_P_Tonnes", "Fertilizer_K_Tonnes",
        "Net_Irrigated_Pct", "Tubewell_Irrig_Pct",
        "NPK_Total_Intensity_Kg_Ha", "NPK_Balance_Ratio", "Rainfall_Anomaly_Pct",
        "Thermal_Stress_Index", "Irrigation_Security_Score"
    ]

    for name, subset in [("train_yield.csv", train_recs), ("val_yield.csv", val_recs), ("test_yield.csv", test_recs)]:
        out_path = os.path.join(out_dir, name)
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)

    print("\n=== TEMPORAL SPLITTING & EXPORT SUMMARY ===")
    print(f"[+] Train Set (1997-2018): {len(train_recs)} records ({len(train_recs)/len(processed_records)*100:.1f}%) -> data/final/yield/train_yield.csv")
    print(f"[+] Val Set   (2019-2020): {len(val_recs)} records ({len(val_recs)/len(processed_records)*100:.1f}%) -> data/final/yield/val_yield.csv")
    print(f"[+] Test Set  (2021-2023): {len(test_recs)} records ({len(test_recs)/len(processed_records)*100:.1f}%) -> data/final/yield/test_yield.csv")
    print(f"[+] Total Engineered Columns: {len(fieldnames)}")


if __name__ == "__main__":
    main()
