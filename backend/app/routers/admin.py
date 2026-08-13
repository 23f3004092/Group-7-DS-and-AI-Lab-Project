import json
import random
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from ..database import get_db
from ..models import QueryLog, SystemConfig
from ..schemas import FeedbackSubmit, ConfigUpdate, LogResponseSchema
from ..config import settings

router = APIRouter(prefix="/api/admin", tags=["Admin Dashboard"])

def seed_mock_data_if_empty(db: Session):
    """Seed SQLite database with realistic FarmerVision telemetry logs to immediately populate charts."""
    if db.query(QueryLog).count() > 0:
        return
        
    print("Seeding SQLite with mock telemetry logs...")
    
    # 20 Mock Queries representing the past 7 days
    now = datetime.datetime.utcnow()
    
    mock_records = [
        # Pathway A: Text queries
        QueryLog(
            timestamp=now - datetime.timedelta(days=6, hours=4),
            pathway="A",
            input_text="गेहूं की पत्तियां पीली हो रही हैं, क्या करें?",
            intent=json.dumps(["disease_pest"]),
            detected_crop="wheat",
            detected_disease="wheat__yellow_rust",
            predicted_yield=None,
            retrieved_chunks=json.dumps([{"text": "Yellow Rust in wheat... use Propiconazole 25% EC", "crop": "wheat", "source_type": "pdf_policy"}]),
            synthesis_response="गेहूं में पीला रतुआ (Yellow Rust) के नियंत्रण के लिए प्रोपिकोनाजोल 25% EC @ 200 मिली प्रति एकड़ की दर से छिड़काव करें [1]।",
            latency_ms=310,
            is_blocked=False,
            feedback_score=1,
            feedback_text="धन्यवाद, दवा से फायदा हुआ।"
        ),
        QueryLog(
            timestamp=now - datetime.timedelta(days=5, hours=2),
            pathway="A",
            input_text="how much urea for 1 hectare rice",
            intent=json.dumps(["nutrition_fertilizer"]),
            detected_crop="rice",
            detected_disease=None,
            predicted_yield=None,
            retrieved_chunks=json.dumps([{"text": "Paddy nitrogen requirement is 120 kg urea per hectare", "crop": "rice", "source_type": "pdf_policy"}]),
            synthesis_response="The recommended nitrogen dose for rice is 120 kg urea per hectare, split into 3 applications [1].",
            latency_ms=250,
            is_blocked=False,
            feedback_score=1,
            feedback_text="Very helpful dosage info"
        ),
        # Blocked query (Guardrails)
        QueryLog(
            timestamp=now - datetime.timedelta(days=5, hours=6),
            pathway="A",
            input_text="spray double dose of monocrotophos to kill all insects quickly",
            intent=json.dumps(["disease_pest"]),
            detected_crop=None,
            detected_disease=None,
            predicted_yield=None,
            retrieved_chunks=None,
            synthesis_response="This query cannot be answered safely. Please contact a KVK officer.",
            latency_ms=45,
            is_blocked=True,
            guardrail_reason="Banned chemical 'monocrotophos' and hazardous dosage 'double dose' detected.",
            feedback_score=None
        ),
        # Pathway B: Image queries
        QueryLog(
            timestamp=now - datetime.timedelta(days=4, hours=1),
            pathway="B",
            image_path="/uploads/rice_leaf_spot_1.jpg",
            intent=json.dumps(["disease_pest"]),
            detected_crop="rice",
            detected_disease="rice__brown_spot",
            predicted_yield=None,
            retrieved_chunks=json.dumps([{"text": "Brown Spot in paddy... spray Hexaconazole 5% EC", "crop": "rice", "source_type": "pdf_policy"}]),
            synthesis_response="धान में भूरा धब्बा (Brown Spot) रोग लगा है। इसके रासायनिक उपचार के लिए हेक्साकोनाजोल 5% EC @ 2 मिली प्रति लीटर का छिड़काव करें [1]।",
            latency_ms=640,
            is_blocked=False,
            feedback_score=1
        ),
        QueryLog(
            timestamp=now - datetime.timedelta(days=3, hours=3),
            pathway="B",
            image_path="/uploads/wheat_rust_2.jpg",
            intent=json.dumps(["disease_pest"]),
            detected_crop="wheat",
            detected_disease="wheat__yellow_rust",
            predicted_yield=None,
            retrieved_chunks=json.dumps([{"text": "Yellow Rust in wheat... use Propiconazole 25% EC", "crop": "wheat", "source_type": "pdf_policy"}]),
            synthesis_response="Your wheat crop is diagnosed with Yellow Rust. Spray Propiconazole 25% EC @ 200ml/acre [1].",
            latency_ms=590,
            is_blocked=False,
            feedback_score=-1,
            feedback_text="Image was a bit blurry, but diagnosis was correct."
        ),
        # Pathway C: Yield queries
        QueryLog(
            timestamp=now - datetime.timedelta(days=2, hours=8),
            pathway="C",
            input_text="crop: wheat, district: jhansi, area_ha: 5.5",
            intent=json.dumps(["general"]),
            detected_crop="wheat",
            detected_disease=None,
            predicted_yield=14.19,  # ~2.58 t/ha
            retrieved_chunks=None,
            synthesis_response="Expected yield for wheat on 5.5 ha in jhansi: ~2.58 t/ha (14.2 tonnes total). Estimate based on historical UP district data (LightGBM).",
            latency_ms=15,
            is_blocked=False,
            feedback_score=1
        ),
        QueryLog(
            timestamp=now - datetime.timedelta(days=1, hours=5),
            pathway="C",
            input_text="crop: rice, district: meerut, area_ha: 10.0",
            intent=json.dumps(["general"]),
            detected_crop="rice",
            detected_disease=None,
            predicted_yield=33.5,  # ~3.35 t/ha
            retrieved_chunks=None,
            synthesis_response="Expected yield for rice on 10.0 ha in meerut: ~3.35 t/ha (33.5 tonnes total). Estimate based on historical UP district data (LightGBM).",
            latency_ms=12,
            is_blocked=False,
            feedback_score=None
        ),
        # Pathway AB: Multimodal
        QueryLog(
            timestamp=now - datetime.timedelta(days=0, hours=2),
            pathway="AB",
            input_text="what spray can cure this leaf spot immediately?",
            image_path="/uploads/rice_leaf_spot_2.jpg",
            intent=json.dumps(["disease_pest"]),
            detected_crop="rice",
            detected_disease="rice__brown_spot",
            predicted_yield=None,
            retrieved_chunks=json.dumps([{"text": "Brown Spot in paddy... spray Hexaconazole 5% EC", "crop": "rice", "source_type": "pdf_policy"}]),
            synthesis_response="This leaf spot is Rice Brown Spot. Apply Hexaconazole 5% EC @ 2 ml/liter of water [1].\n\n⚠ Caution: Wear protective clothing when spraying.",
            latency_ms=780,
            is_blocked=False,
            feedback_score=1
        )
    ]
    
    # Add more queries for scale
    for i in range(15):
        days_ago = random.randint(0, 6)
        hours_ago = random.randint(1, 23)
        pw = random.choice(["A", "B", "C", "AB"])
        lat = random.choice([20, 45, 180, 290, 420, 680, 950])
        block = random.random() < 0.08
        feed = random.choice([1, 1, 1, -1, None])
        crop = random.choice(["wheat", "rice", "maize"])
        
        log = QueryLog(
            timestamp=now - datetime.timedelta(days=days_ago, hours=hours_ago),
            pathway=pw,
            input_text=f"Sample query {i} for {crop}",
            intent=json.dumps(["disease_pest" if pw in ["B","AB"] else "general"]),
            detected_crop=crop,
            detected_disease=f"{crop}__healthy" if random.random() > 0.4 else f"{crop}__disease_{i}",
            predicted_yield=round(random.uniform(2.5, 38.0), 2) if pw == "C" else None,
            synthesis_response=f"This is a simulated synthesized response for crop advisory on {crop}.",
            latency_ms=lat,
            is_blocked=block,
            guardrail_reason="Banned pesticide check triggered" if block else None,
            feedback_score=feed
        )
        mock_records.append(log)
        
    db.add_all(mock_records)
    db.commit()

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Retrieve statistics for dashboard visualization."""
    seed_mock_data_if_empty(db)
    
    # Basic counts
    total_queries = db.query(QueryLog).count()
    blocked_queries = db.query(QueryLog).filter(QueryLog.is_blocked == True).count()
    
    # Feedback rates
    positive_feedback = db.query(QueryLog).filter(QueryLog.feedback_score == 1).count()
    negative_feedback = db.query(QueryLog).filter(QueryLog.feedback_score == -1).count()
    total_feedback = positive_feedback + negative_feedback
    sat_rate = round((positive_feedback / total_feedback * 100), 1) if total_feedback > 0 else 100.0

    # Average latency
    avg_latency = 0.0
    logs = db.query(QueryLog).all()
    if logs:
        avg_latency = round(sum(log.latency_ms for log in logs) / len(logs), 1)
        
    # Latencies by pathway
    pathway_latencies = {}
    for pw in ["A", "B", "C", "AB"]:
        pw_logs = [log for log in logs if log.pathway == pw]
        pathway_latencies[pw] = round(sum(log.latency_ms for log in pw_logs) / len(pw_logs), 1) if pw_logs else 0
        
    # Pathway volume breakdown
    pathway_counts = {}
    for pw in ["A", "B", "C", "AB"]:
        pathway_counts[pw] = sum(1 for log in logs if log.pathway == pw)

    # Disease frequency count
    disease_counts = {}
    for log in logs:
        if log.detected_disease:
            disease_counts[log.detected_disease] = disease_counts.get(log.detected_disease, 0) + 1
            
    # Format disease data for charts
    disease_breakdown = [{"name": k.replace("__", " ").title(), "value": v} for k, v in disease_counts.items()]
    disease_breakdown = sorted(disease_breakdown, key=lambda x: x["value"], reverse=True)[:5]
    
    # Latency distribution buckets
    latency_buckets = {"< 100ms": 0, "100-300ms": 0, "300-600ms": 0, "> 600ms": 0}
    for log in logs:
        if log.latency_ms < 100:
            latency_buckets["< 100ms"] += 1
        elif log.latency_ms <= 300:
            latency_buckets["100-300ms"] += 1
        elif log.latency_ms <= 600:
            latency_buckets["300-600ms"] += 1
        else:
            latency_buckets["> 600ms"] += 1
            
    latency_dist = [{"bucket": k, "count": v} for k, v in latency_buckets.items()]

    # Volume trend over the last 7 days
    volume_trend = []
    now = datetime.datetime.utcnow().date()
    for d in range(6, -1, -1):
        target_date = now - datetime.timedelta(days=d)
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Count queries on this day
        cnt = db.query(QueryLog).filter(
            QueryLog.timestamp >= datetime.datetime.combine(target_date, datetime.time.min),
            QueryLog.timestamp <= datetime.datetime.combine(target_date, datetime.time.max)
        ).count()
        
        volume_trend.append({"date": date_str, "queries": cnt})

    return {
        "summary": {
            "total_queries": total_queries,
            "blocked_queries": blocked_queries,
            "safety_violation_rate": round((blocked_queries / total_queries * 100), 1) if total_queries > 0 else 0,
            "average_latency_ms": avg_latency,
            "satisfaction_rate": sat_rate,
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback
        },
        "pathway_counts": pathway_counts,
        "pathway_latencies": pathway_latencies,
        "disease_breakdown": disease_breakdown,
        "latency_distribution": latency_dist,
        "volume_trend": volume_trend
    }

@router.get("/logs", response_model=List[LogResponseSchema])
def get_logs(
    pathway: Optional[str] = None,
    is_blocked: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Retrieve detailed list of query logs."""
    query = db.query(QueryLog)
    
    if pathway:
        query = query.filter(QueryLog.pathway == pathway)
    if is_blocked is not None:
        query = query.filter(QueryLog.is_blocked == is_blocked)
        
    logs = query.order_by(QueryLog.timestamp.desc()).limit(limit).offset(offset).all()
    return logs

@router.get("/config")
def get_configs(db: Session = Depends(get_db)):
    """Retrieve active configuration thresholds and weights."""
    configs = {}
    for key in [
        "tier_grounded", "tier_fallback", 
        "weight_pdf_policy", "weight_kcc_policy",
        "weight_pdf_practice", "weight_kcc_practice",
        "mock_models"
    ]:
        db_cfg = db.query(SystemConfig).filter(SystemConfig.key == key).first()
        if db_cfg:
            # Parse numeric/bool values
            val = db_cfg.value
            if val.lower() in ["true", "false"]:
                configs[key] = val.lower() == "true"
            else:
                try:
                    configs[key] = float(val)
                except ValueError:
                    configs[key] = val
        else:
            # Return defaults from config settings
            cfg_name = key.upper()
            configs[key] = getattr(settings, cfg_name, None)
            
    return configs

@router.post("/config")
def update_configs(cfg_data: ConfigUpdate, db: Session = Depends(get_db)):
    """Update dynamic configuration variables in database."""
    updates = cfg_data.model_dump(exclude_unset=True)
    
    for key, value in updates.items():
        db_cfg = db.query(SystemConfig).filter(SystemConfig.key == key).first()
        if db_cfg:
            db_cfg.value = str(value)
        else:
            db_cfg = SystemConfig(key=key, value=str(value))
            db.add(db_cfg)
            
    db.commit()
    
    # Reload/override current memory settings temporarily if needed or return success
    return {"status": "success", "updated_keys": list(updates.keys())}

@router.get("/vectordb")
def check_vector_db():
    """Retrieve connection health status and counts for the Vector DB."""
    from .query import get_qdrant_status
    status = get_qdrant_status()
    return status
