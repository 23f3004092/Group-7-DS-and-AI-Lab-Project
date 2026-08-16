import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from ..config import settings

ROOT = Path(__file__).resolve().parent.parent.parent

# Trained model (LGBMRegressor) serialized with joblib
YIELD_MODEL_PATH = (Path(__file__).resolve().parent / "models" / "lightgbm_tuned.joblib")

# Legacy LightGBM booster path from sample_app.py (fallback)
LEGACY_BOOSTER_PATH = (ROOT / "outputs" / "Yeild_output_files" / "kaggle" /
                       "working" / "saved_models" / "lightgbm_tuned.txt")

# Feature columns the model was trained on
YIELD_FEATURES = [
    "crop", "state", "district", "season", "data_source", "crop_type",
    "year", "area", "annual_rainfall", "fertilizer", "pesticide",
]
YIELD_CATEGORICAL = ["crop", "state", "district", "season", "data_source", "crop_type"]

# Regional adjustments for UP districts (Western vs Bundelkhand vs Eastern)
DISTRICT_MODIFIERS = {
    # Western UP (High yield)
    "meerut": 1.25, "bulandshahr": 1.20, "aligarh": 1.15, "baghpat": 1.22, "muzaffarnagar": 1.26,
    # Bundelkhand (Low yield, dry)
    "jhansi": 0.70, "lalitpur": 0.65, "mahoba": 0.62, "hamirpur": 0.68, "banda": 0.72,
    # Central/Eastern (Average)
    "lucknow": 1.0, "kanpur": 0.98, "varanasi": 0.95, "gorakhpur": 0.92, "prayagraj": 0.96,
}

class YieldService:
    def __init__(self):
        self.model = None
        self.booster = None
        self.initialized = False

        # Preferred: load trained joblib model
        if os.path.exists(YIELD_MODEL_PATH):
            try:
                import joblib
                self.model = joblib.load(str(YIELD_MODEL_PATH))
                self.initialized = True
                print(f"Loaded yield model from {YIELD_MODEL_PATH}")
            except Exception as e:
                print(f"Could not load joblib model: {e}. Trying legacy LightGBM booster.")
        else:
            print(f"Yield model file not found at {YIELD_MODEL_PATH}. Trying legacy LightGBM booster.")

        # Fallback: legacy LightGBM booster
        if not self.initialized and os.path.exists(LEGACY_BOOSTER_PATH):
            try:
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=str(LEGACY_BOOSTER_PATH))
                self.initialized = True
                print(f"Loaded LightGBM booster from {LEGACY_BOOSTER_PATH}")
            except Exception as e:
                print(f"Could not load LightGBM booster: {e}. Bypassing to math fallback.")
        elif not self.initialized:
            print("Yield model file not found. Using math-based fallback predictor.")

    def predict(self, crop: str, district: str, area_ha: float,
                annual_rainfall: Optional[float] = None, year: int = 2023) -> Tuple[float, float]:
        """
        Predict yield in tonnes per hectare (t/ha) and total yield (tonnes).
        Also calculates a simulated profit breakdown.
        """
        crop_n = crop.lower().strip()
        dist_n = district.lower().strip()
        rainfall = annual_rainfall if annual_rainfall is not None else 850.0

        # If joblib model is loaded, use it
        if self.initialized and self.model:
            try:
                import pandas as pd
                X = pd.DataFrame([{
                    "crop":            crop_n,
                    "state":           "uttar pradesh",
                    "district":        dist_n,
                    "season":          "Rabi" if crop_n == "wheat" else "Kharif",
                    "data_source":     "area_production",
                    "crop_type":       "cereals",
                    "year":            year,
                    "area":            area_ha,
                    "annual_rainfall": rainfall,
                    "fertilizer":      1200000.0,
                    "pesticide":       2400.0,
                }])
                for c in YIELD_CATEGORICAL:
                    X[c] = X[c].astype("category")

                pred_t_ha = float(self.model.predict(X)[0])
                total_t = round(pred_t_ha * area_ha, 2)
                return pred_t_ha, total_t
            except Exception as e:
                print(f"Joblib model prediction failed: {e}. Falling back.")

        # If LightGBM booster is loaded, use it
        if self.initialized and self.booster:
            try:
                import pandas as pd
                # Replicate feature frame from sample_app.py
                X = pd.DataFrame([{
                    "crop":            crop_n,
                    "state":           "uttar pradesh",
                    "district":        dist_n,
                    "season":          "Rabi" if crop_n == "wheat" else "Kharif",
                    "data_source":     "area_production",
                    "crop_type":       "cereals",
                    "year":            year,
                    "area":            area_ha,
                    "annual_rainfall": rainfall,
                    "fertilizer":      1200000.0,
                    "pesticide":       2400.0,
                }])
                for c in YIELD_CATEGORICAL:
                    X[c] = X[c].astype("category")
                
                pred_t_ha = float(self.booster.predict(X)[0])
                total_t = round(pred_t_ha * area_ha, 2)
                return pred_t_ha, total_t
            except Exception as e:
                print(f"LightGBM prediction failed: {e}. Falling back.")

        # Fallback physics-based yield predictor
        # Baseline yield (t/ha)
        if crop_n in ["wheat"]:
            base_yield = 3.65 # Wheat average in UP
        elif crop_n in ["rice", "paddy"]:
            base_yield = 2.80 # Rice average in UP
        elif crop_n in ["maize"]:
            base_yield = 2.20
        else:
            base_yield = 1.80
            
        # Apply district modifier
        modifier = DISTRICT_MODIFIERS.get(dist_n, 0.95) # Default slightly below central UP
        
        # Area sizing scaling - larger farms might have slight efficiency drop or rise
        area_factor = 1.0 - (min(area_ha, 100.0) / 1000.0)
        
        # Calculate yield per hectare
        pred_t_ha = base_yield * modifier * area_factor
        
        # Clamp to realistic ranges
        pred_t_ha = max(0.8, min(pred_t_ha, 6.5))
        total_t = round(pred_t_ha * area_ha, 2)
        
        return pred_t_ha, total_t

    def estimate_profitability(self, crop: str, yield_t: float, area_ha: float,
                               market_price_per_quintal: Optional[float] = None,
                               cost_per_ha: Optional[float] = None,
                               price_source: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimate costs, revenues, and net profit. Uses the live mandi market price
        per quintal when provided, otherwise falls back to CACP/static MSP values.
        Cost per hectare comes from CACP/DES when provided, else static baselines.
        """
        crop_n = crop.lower().strip()
        
        # Minimum Support Price (MSP) in INR per quintal (1 quintal = 100kg = 0.1 tonnes)
        # So 1 tonne = 10 quintals
        # 2024-2025 MSP values
        if crop_n == "wheat":
            msp_per_quintal = 2275
        elif crop_n in ["rice", "paddy"]:
            msp_per_quintal = 2183
        elif crop_n == "maize":
            msp_per_quintal = 2090
        else:
            msp_per_quintal = 1800

        if market_price_per_quintal is not None:
            price_per_quintal = market_price_per_quintal
            price_source = price_source or "mandi"
        else:
            price_per_quintal = msp_per_quintal
            price_source = price_source or "msp"

        if cost_per_ha is not None:
            cost_per_ha = cost_per_ha
            cost_source = "cacp"
        else:
            if crop_n == "wheat":
                cost_per_ha = 32000  # Seed, land prep, fertilizer, irrigation
            elif crop_n in ["rice", "paddy"]:
                cost_per_ha = 38000  # Higher water/labor costs
            elif crop_n == "maize":
                cost_per_ha = 25000
            else:
                cost_per_ha = 20000
            cost_source = "static"

        total_cost = cost_per_ha * area_ha
        total_revenue = (yield_t * 10) * price_per_quintal
        net_profit = total_revenue - total_cost
        
        return {
            "crop": crop,
            "area_ha": area_ha,
            "price_per_quintal": price_per_quintal,
            "price_source": price_source,
            "msp_per_quintal": msp_per_quintal,
            "cost_per_ha": cost_per_ha,
            "cost_source": cost_source,
            "total_cost": round(total_cost, 2),
            "total_revenue": round(total_revenue, 2),
            "net_profit": round(net_profit, 2),
            "roi_percent": round((net_profit / total_cost) * 100, 1) if total_cost > 0 else 0
        }

yield_service = YieldService()
