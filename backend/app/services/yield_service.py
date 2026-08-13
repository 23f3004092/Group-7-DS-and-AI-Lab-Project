import os
from pathlib import Path
from ..config import settings

ROOT = Path(__file__).resolve().parent.parent.parent

# Path from sample_app.py
YIELD_MODEL_PATH = (ROOT / "outputs" / "Yeild_output_files" / "kaggle" /
                    "working" / "saved_models" / "lightgbm_tuned.txt")

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
        self.booster = None
        self.initialized = False
        
        # Try to load LightGBM booster
        if os.path.exists(YIELD_MODEL_PATH):
            try:
                import lightgbm as lgb
                self.booster = lgb.Booster(model_file=str(YIELD_MODEL_PATH))
                self.initialized = True
                print(f"Loaded LightGBM model from {YIELD_MODEL_PATH}")
            except Exception as e:
                print(f"Could not load LightGBM booster: {e}. Bypassing to math fallback.")
        else:
            print(f"Yield model file not found at {YIELD_MODEL_PATH}. Using math-based fallback predictor.")

    def predict(self, crop: str, district: str, area_ha: float) -> Tuple[float, float]:
        """
        Predict yield in tonnes per hectare (t/ha) and total yield (tonnes).
        Also calculates a simulated profit breakdown.
        """
        crop_n = crop.lower().strip()
        dist_n = district.lower().strip()
        
        # If LightGBM is loaded, use it
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
                    "year":            2023,
                    "area":            area_ha,
                    "annual_rainfall": 850.0,
                    "fertilizer":      1200000.0,
                    "pesticide":       2400.0,
                }])
                for c in ["crop", "state", "district", "season", "data_source", "crop_type"]:
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

    def estimate_profitability(self, crop: str, yield_t: float, area_ha: float) -> Dict[str, Any]:
        """
        Estimate costs, revenues, and net profit based on standard crop prices (MSP) and inputs.
        """
        crop_n = crop.lower().strip()
        
        # Minimum Support Price (MSP) in INR per quintal (1 quintal = 100kg = 0.1 tonnes)
        # So 1 tonne = 10 quintals
        # 2024-2025 MSP values
        if crop_n == "wheat":
            msp_per_quintal = 2275
            cost_per_ha = 32000 # Seed, land prep, fertilizer, irrigation
        elif crop_n in ["rice", "paddy"]:
            msp_per_quintal = 2183
            cost_per_ha = 38000 # Higher water/labor costs
        elif crop_n == "maize":
            msp_per_quintal = 2090
            cost_per_ha = 25000
        else:
            msp_per_quintal = 1800
            cost_per_ha = 20000
            
        total_cost = cost_per_ha * area_ha
        total_revenue = (yield_t * 10) * msp_per_quintal
        net_profit = total_revenue - total_cost
        
        return {
            "crop": crop,
            "area_ha": area_ha,
            "msp_per_quintal": msp_per_quintal,
            "cost_per_ha": cost_per_ha,
            "total_cost": round(total_cost, 2),
            "total_revenue": round(total_revenue, 2),
            "net_profit": round(net_profit, 2),
            "roi_percent": round((net_profit / total_cost) * 100, 1) if total_cost > 0 else 0
        }

yield_service = YieldService()
