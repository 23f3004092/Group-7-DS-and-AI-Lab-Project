#!/usr/bin/env python3
"""
run_yield_eda_analysis.py — Comprehensive EDA statistical profiler for UP District Yield Dataset
Computes exact summary metrics across 3,886 district-year records (1997-2023) for Milestone 2 Report.
"""

import csv
import math
from collections import defaultdict


def compute_mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)


def compute_median_iqr(values):
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    med = s[n // 2]
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return med, q1, q3


def pearson_corr(xs, ys):
    mx, _ = compute_mean_std(xs)
    my, _ = compute_mean_std(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den != 0 else 0.0


def main():
    path = "data/raw/yield/up_district_yield_apy_1997_2023.csv"
    records = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"=== UP YIELD DATASET EDA REPORT (Total Records: {len(records)}) ===")

    # 1. Missingness / Bifurcation Analysis
    districts = set(r["District_Name"] for r in records)
    print(f"\n[+] Unique UP Districts represented: {len(districts)}")
    # Expected total if all 75 existed for 27 years (1997-2023) * 2 seasons = 75 * 27 * 2 = 4,050
    expected_total = 75 * 27 * 2
    missing_due_to_bifurcation = expected_total - len(records)
    print(f"[+] Total expected records (75 districts x 27 yrs x 2 seasons): {expected_total}")
    print(f"[+] Records absent due to historical district bifurcations prior to formation: {missing_due_to_bifurcation} ({missing_due_to_bifurcation/expected_total*100:.1f}%)")

    # Zero production reporting anomalies
    zero_prod = [r for r in records if float(r["Production_Total"]) == 0.0]
    print(f"[+] Sporadic zero-production bureaucratic reporting anomalies: {len(zero_prod)} ({len(zero_prod)/len(records)*100:.2f}%)")

    # 2. Per-Crop Yield Summary Statistics (excluding zero reporting errors)
    for crop in ["Rice", "Wheat"]:
        yields = [float(r["Yield_Kg_Ha"]) for r in records if r["Crop"] == crop and float(r["Yield_Kg_Ha"]) > 0]
        mean, std = compute_mean_std(yields)
        med, q1, q3 = compute_median_iqr(yields)
        print(f"\n[+] {crop.upper()} Yield (kg/ha) Summary:")
        print(f"    Count: {len(yields)} | Mean: {mean:.1f} | Std: {std:.1f}")
        print(f"    Median: {med:.1f} | Q1: {q1:.1f} | Q3: {q3:.1f} | IQR: {q3-q1:.1f}")
        print(f"    Min: {min(yields):.1f} | Max: {max(yields):.1f}")

    # 3. Regional Disparity Analysis (Western UP vs Central vs Eastern vs Bundelkhand)
    print("\n[+] Regional Yield & Irrigation Disparities (Mean values by Agro-Climatic Zone):")
    zones = ["Western UP", "Central UP", "Eastern UP", "Bundelkhand"]
    print(f"{'Zone':<15} | {'Rice Yield':<11} | {'Wheat Yield':<11} | {'Net Irrig %':<11} | {'Tubewell %':<11}")
    print("-" * 68)
    for zone in zones:
        rice_y = [float(r["Yield_Kg_Ha"]) for r in records if r["Agro_Climatic_Zone"] == zone and r["Crop"] == "Rice" and float(r["Yield_Kg_Ha"]) > 0]
        wheat_y = [float(r["Yield_Kg_Ha"]) for r in records if r["Agro_Climatic_Zone"] == zone and r["Crop"] == "Wheat" and float(r["Yield_Kg_Ha"]) > 0]
        irrig = [float(r["Net_Irrigated_Pct"]) for r in records if r["Agro_Climatic_Zone"] == zone and r["Crop"] == "Wheat"]
        tube = [float(r["Tubewell_Irrig_Pct"]) for r in records if r["Agro_Climatic_Zone"] == zone and r["Crop"] == "Wheat"]
        print(f"{zone:<15} | {compute_mean_std(rice_y)[0]:<11.1f} | {compute_mean_std(wheat_y)[0]:<11.1f} | {compute_mean_std(irrig)[0]:<11.1f} | {compute_mean_std(tube)[0]:<11.1f}")

    # 4. Correlations with Weather & Inputs
    print("\n[+] Correlations with Target Yield (Yield_Kg_Ha > 0):")
    for crop in ["Rice", "Wheat"]:
        crop_recs = [r for r in records if r["Crop"] == crop and float(r["Yield_Kg_Ha"]) > 0]
        ys = [float(r["Yield_Kg_Ha"]) for r in crop_recs]
        rain = [float(r["Precip_Seasonal_mm"]) for r in crop_recs]
        irrig = [float(r["Net_Irrigated_Pct"]) for r in crop_recs]
        extreme = [float(r["Rain_Days_Extreme"]) for r in crop_recs]
        heat = [float(r["Heatwave_Days"]) for r in crop_recs]
        n_fert = [float(r["Fertilizer_N_Tonnes"]) / max(1.0, float(r["Area_Sown"])) * 1000 for r in crop_recs]
        print(f"  * {crop}:")
        print(f"      Seasonal Precip corr:   {pearson_corr(ys, rain):+.3f}")
        print(f"      Net Irrigated % corr:   {pearson_corr(ys, irrig):+.3f}")
        print(f"      Extreme Rain Days corr: {pearson_corr(ys, extreme):+.3f}")
        if crop == "Wheat":
            print(f"      March Heatwaves corr:   {pearson_corr(ys, heat):+.3f}")
        print(f"      Fertilizer N (kg/ha):   {pearson_corr(ys, n_fert):+.3f}")


if __name__ == "__main__":
    main()
