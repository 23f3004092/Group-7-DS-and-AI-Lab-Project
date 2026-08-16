import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from ..database import get_db
from ..models import QueryLog, SystemConfig
from ..schemas import FeedbackSubmit, ConfigUpdate, LogResponseSchema
from ..config import settings

router = APIRouter(prefix="/api/admin", tags=["Admin Dashboard"])

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Retrieve statistics for dashboard visualization."""
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
