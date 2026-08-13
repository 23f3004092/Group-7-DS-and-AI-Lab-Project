import os
import json
import time
import uuid
import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from ..database import get_db
from ..models import QueryLog, SystemConfig
from ..schemas import TextQuery, YieldQuery, QueryResponse
from ..services.cloud_models import CloudAIService
from ..services.qdrant_service import qdrant_service
from ..services.yield_service import yield_service
from ..config import settings

router = APIRouter(prefix="/api/query", tags=["Advisory Query Pipelines"])

# Create upload directory
UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_current_settings(db: Session) -> Dict[str, Any]:
    """Helper to fetch dynamic configurations from SQLite settings table."""
    configs = {}
    for key in [
        "tier_grounded", "tier_fallback", 
        "weight_pdf_policy", "weight_kcc_policy",
        "weight_pdf_practice", "weight_kcc_practice",
        "mock_models"
    ]:
        db_cfg = db.query(SystemConfig).filter(SystemConfig.key == key).first()
        if db_cfg:
            val = db_cfg.value
            if val.lower() in ["true", "false"]:
                configs[key] = val.lower() == "true"
            else:
                try:
                    configs[key] = float(val)
                except ValueError:
                    configs[key] = val
        else:
            cfg_name = key.upper()
            configs[key] = getattr(settings, cfg_name, None)
    return configs

def get_qdrant_status() -> Dict[str, Any]:
    """Helper to check vector DB connection status and count."""
    if qdrant_service.initialized and qdrant_service.client:
        try:
            col_info = qdrant_service.client.get_collection(settings.COLLECTION_NAME)
            return {
                "status": "connected",
                "mode": "server",
                "collection": settings.COLLECTION_NAME,
                "points_count": col_info.points_count,
                "vector_size": col_info.config.params.vector_size if hasattr(col_info.config, 'params') else 1024,
                "error": None
            }
        except Exception as e:
            return {
                "status": "degraded",
                "mode": "fallback_local",
                "collection": settings.COLLECTION_NAME,
                "points_count": 8,
                "vector_size": 1024,
                "error": str(e)
            }
    return {
        "status": "connected",
        "mode": "fallback_local",
        "collection": settings.COLLECTION_NAME,
        "points_count": 8,
        "vector_size": 1024,
        "error": "Qdrant server not running. Running in local retrieval mode."
    }

def record_telemetry_log(
    db: Session,
    pathway: str,
    latency_ms: int,
    input_text: Optional[str] = None,
    image_path: Optional[str] = None,
    intent: Optional[List[str]] = None,
    detected_crop: Optional[str] = None,
    detected_disease: Optional[str] = None,
    predicted_yield: Optional[float] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    synthesis_response: Optional[str] = None,
    is_blocked: bool = False,
    guardrail_reason: Optional[str] = None
) -> int:
    """Save the query telemetry data into database logs."""
    try:
        log = QueryLog(
            pathway=pathway,
            input_text=input_text,
            image_path=image_path,
            intent=json.dumps(intent) if intent else "[]",
            detected_crop=detected_crop,
            detected_disease=detected_disease,
            predicted_yield=predicted_yield,
            retrieved_chunks=json.dumps(retrieved_chunks) if retrieved_chunks else "[]",
            synthesis_response=synthesis_response,
            latency_ms=latency_ms,
            is_blocked=is_blocked,
            guardrail_reason=guardrail_reason
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log.id
    except Exception as e:
        print(f"Failed to record telemetry log: {e}")
        return 0

def get_caveat_text(tier: str) -> Optional[str]:
    """Return caveat disclaimers according to relevance tiers (§9.9)."""
    if tier == "fallback":
        return "Advice based on general guidance. Please confirm with your local KVK officer."
    if tier == "abstain":
        return "Your query is outside the system's knowledge base. Please contact a KVK officer."
    return None

@router.post("/text", response_model=QueryResponse)
async def query_text(q: TextQuery, db: Session = Depends(get_db)):
    """Pathway A — Text-only query. Runs Intent, RAG vector retrieval, and LLM Synthesis."""
    t0 = time.time()
    text = q.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    cfg = get_current_settings(db)
    
    # 1. Run IEG (Intent, Entity, Guardrails)
    intents, blocked, block_reason = await CloudAIService.run_intent_entity_guardrails(text)
    
    if blocked:
        answer = "This query cannot be answered safely. Please contact a KVK officer."
        latency = int((time.time() - t0) * 1000)
        log_id = record_telemetry_log(
            db, pathway="A", latency_ms=latency, input_text=text, 
            intent=intents, is_blocked=True, guardrail_reason=block_reason,
            synthesis_response=answer
        )
        return QueryResponse(
            pathway="A", intent=intents, blocked=True, tier="blocked",
            top_score=0.0, answer=answer, latency_ms=latency
        )
        
    # 2. Identify crop entities mentioned
    detected_crop = None
    for crop in ["wheat", "rice", "paddy", "maize", "mustard", "sugarcane", "potato", "mango"]:
        if crop in text.lower():
            detected_crop = crop
            break

    # 3. Retrieve documents from Vector DB
    hits, tier, top_score = qdrant_service.retrieve(text, intents=intents, current_settings=cfg)
    caveat = get_caveat_text(tier)
    
    # Format sources list
    sources = [{
        "rank": idx + 1,
        "score": round(hit["score"], 3),
        "text": hit["text"][:150] + "...",
        "crop": hit["crop"],
        "source_type": hit["source_type"]
    } for idx, hit in enumerate(hits[:5])]

    # 4. Synthesize answer
    if tier == "abstain":
        answer = caveat
    else:
        answer = await CloudAIService.synthesize_response(text, hits[:5])
        if caveat:
            answer = f"{answer}\n\n⚠ {caveat}"

    latency = int((time.time() - t0) * 1000)
    
    # 5. Save log
    log_id = record_telemetry_log(
        db, pathway="A", latency_ms=latency, input_text=text,
        intent=intents, detected_crop=detected_crop, 
        retrieved_chunks=hits[:5], synthesis_response=answer,
        is_blocked=False
    )
    
    return QueryResponse(
        pathway="A", intent=intents, blocked=False, tier=tier,
        top_score=round(top_score, 4), answer=answer, sources=sources,
        latency_ms=latency, detected_crop=detected_crop
    )

@router.post("/image", response_model=QueryResponse)
async def query_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Pathway B — Leaf photo upload. Runs image disease classification, RAG retrieval, and Synthesis."""
    t0 = time.time()
    cfg = get_current_settings(db)
    
    # Save image
    ext = os.path.splitext(file.filename)[1]
    saved_name = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    
    image_bytes = await file.read()
    with open(saved_path, "wb") as f:
        f.write(image_bytes)
        
    web_image_path = f"/uploads/{saved_name}"

    # 1. Run Vision Classifier
    result = await CloudAIService.run_vision_diagnosis(image_bytes, file.filename)
    
    if result.get("rejected", False):
        answer = "Image rejected — please submit a clear, close-up crop leaf photo."
        latency = int((time.time() - t0) * 1000)
        record_telemetry_log(
            db, pathway="B", latency_ms=latency, image_path=web_image_path,
            synthesis_response=answer, guardrail_reason="Image rejected (OOD)"
        )
        return QueryResponse(
            pathway="B", intent=["disease_pest"], blocked=False, tier="rejected",
            top_score=0.0, answer=answer, latency_ms=latency
        )
        
    label = result["label"]
    confidence = result["confidence"]
    crop = label.split("__")[0] if "__" in label else "crop"
    disease = label.split("__")[1].replace("_", " ") if "__" in label else "disease"
    
    # 2. Form advisory text query based on disease label
    query = f"{crop} {disease} treatment and management advice"
    
    # 3. Retrieve
    hits, tier, top_score = qdrant_service.retrieve(query, intents=["disease_pest"], current_settings=cfg)
    caveat = get_caveat_text(tier)
    
    sources = [{
        "rank": idx + 1,
        "score": round(hit["score"], 3),
        "text": hit["text"][:150] + "...",
        "crop": hit["crop"],
        "source_type": hit["source_type"]
    } for idx, hit in enumerate(hits[:5])]

    # 4. Generate
    if tier == "abstain":
        answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Conf: {confidence:.1%})\n\n{caveat}"
    else:
        answer = await CloudAIService.synthesize_response(query, hits[:5])
        # Inject details from classifier if not in context
        organic = result.get("organic_treatment", "")
        chemical = result.get("chemical_treatment", "")
        
        treatment_section = f"\n\n**Organic Remedy:** {organic}\n**Chemical Remedy:** {chemical}"
        answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Confidence: {confidence:.1%})\n\n{answer}{treatment_section}"
        
        if caveat:
            answer = f"{answer}\n\n⚠ {caveat}"

    latency = int((time.time() - t0) * 1000)
    
    record_telemetry_log(
        db, pathway="B", latency_ms=latency, image_path=web_image_path,
        intent=["disease_pest"], detected_crop=crop, detected_disease=label,
        retrieved_chunks=hits[:5], synthesis_response=answer
    )
    
    return QueryResponse(
        pathway="B", intent=["disease_pest"], blocked=False, tier=tier,
        top_score=round(top_score, 4), answer=answer, sources=sources,
        latency_ms=latency, detected_crop=crop, detected_disease=label
    )

@router.post("/yield")
async def query_yield(q: YieldQuery, db: Session = Depends(get_db)):
    """Pathway C — Crop Yield & Profitability estimation using LightGBM and price/cost multipliers."""
    t0 = time.time()
    
    crop = q.crop.lower().strip()
    district = q.district.lower().strip()
    area = q.area_ha
    
    # 1. Run LightGBM yield prediction
    pred_t_ha, total_yield = yield_service.predict(crop, district, area)
    
    # 2. Run economic profitability calculator
    economics = yield_service.estimate_profitability(crop, total_yield, area)
    
    answer = (
        f"Expected yield for {crop.title()} on {area} ha in {district.title()}: "
        f"~{pred_t_ha:.2f} t/ha ({total_yield:.2f} tonnes total).\n\n"
        f"**Economic Projections:**\n"
        f"- Est. Cultivation Cost: ₹{economics['total_cost']:,}\n"
        f"- Est. Gross Revenue (MSP): ₹{economics['total_revenue']:,}\n"
        f"- Projected Net Income: ₹{economics['net_profit']:,}\n"
        f"- Return on Investment: {economics['roi_percent']}%"
    )
    
    latency = int((time.time() - t0) * 1000)
    
    record_telemetry_log(
        db, pathway="C", latency_ms=latency, 
        input_text=f"crop: {crop}, district: {district}, area_ha: {area}",
        intent=["general"], detected_crop=crop, predicted_yield=total_yield,
        synthesis_response=answer
    )
    
    return {
        "pathway": "C",
        "intent": ["general"],
        "blocked": False,
        "tier": "grounded",
        "top_score": 1.0,
        "answer": answer,
        "latency_ms": latency,
        "detected_crop": crop,
        "predicted_yield": total_yield,
        "economics": economics
    }

@router.post("/multimodal", response_model=QueryResponse)
async def query_multimodal(
    text: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Pathway AB — Multimodal query (leaf photo + text query). Blends vision prediction and text advisory."""
    t0 = time.time()
    cfg = get_current_settings(db)
    text = text.strip()
    
    # Save image
    ext = os.path.splitext(file.filename)[1]
    saved_name = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    
    image_bytes = await file.read()
    with open(saved_path, "wb") as f:
        f.write(image_bytes)
        
    web_image_path = f"/uploads/{saved_name}"

    # 1. Run IEG on text
    intents, blocked, block_reason = await CloudAIService.run_intent_entity_guardrails(text)
    
    if blocked:
        answer = "This query cannot be answered safely. Please contact a KVK officer."
        latency = int((time.time() - t0) * 1000)
        record_telemetry_log(
            db, pathway="AB", latency_ms=latency, input_text=text, image_path=web_image_path,
            intent=intents, is_blocked=True, guardrail_reason=block_reason,
            synthesis_response=answer
        )
        return QueryResponse(
            pathway="AB", intent=intents, blocked=True, tier="blocked",
            top_score=0.0, answer=answer, latency_ms=latency
        )

    # 2. Run Vision Classifier
    result = await CloudAIService.run_vision_diagnosis(image_bytes, file.filename)
    
    if result.get("rejected", False):
        # Fallback: Treat as Pathway A text query
        hits, tier, top_score = qdrant_service.retrieve(text, intents=intents, current_settings=cfg)
        caveat = get_caveat_text(tier)
        answer = await CloudAIService.synthesize_response(text, hits[:5])
        if caveat:
            answer = f"{answer}\n\n⚠ {caveat}"
            
        latency = int((time.time() - t0) * 1000)
        record_telemetry_log(
            db, pathway="AB", latency_ms=latency, input_text=text, image_path=web_image_path,
            intent=intents, retrieved_chunks=hits[:5], synthesis_response=answer,
            guardrail_reason="Image rejected (OOD) - Processed text only"
        )
        
        sources = [{
            "rank": idx + 1,
            "score": round(hit["score"], 3),
            "text": hit["text"][:150] + "...",
            "crop": hit["crop"],
            "source_type": hit["source_type"]
        } for idx, hit in enumerate(hits[:5])]
        
        return QueryResponse(
            pathway="AB", intent=intents, blocked=False, tier=tier,
            top_score=round(top_score, 4), answer=answer, sources=sources,
            latency_ms=latency
        )

    label = result["label"]
    confidence = result["confidence"]
    crop = label.split("__")[0] if "__" in label else "crop"
    disease = label.split("__")[1].replace("_", " ") if "__" in label else "disease"
    
    # 3. Combine inputs for retrieval
    combined_query = f"{crop} {disease}. {text}"
    
    # Add disease_pest to intents
    search_intents = list(set(intents + ["disease_pest"]))
    
    # 4. Retrieve
    hits, tier, top_score = qdrant_service.retrieve(combined_query, intents=search_intents, current_settings=cfg)
    caveat = get_caveat_text(tier)
    
    sources = [{
        "rank": idx + 1,
        "score": round(hit["score"], 3),
        "text": hit["text"][:150] + "...",
        "crop": hit["crop"],
        "source_type": hit["source_type"]
    } for idx, hit in enumerate(hits[:5])]

    # 5. Generate
    if tier == "abstain":
        answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Conf: {confidence:.1%})\n\n{caveat}"
    else:
        answer = await CloudAIService.synthesize_response(combined_query, hits[:5])
        organic = result.get("organic_treatment", "")
        chemical = result.get("chemical_treatment", "")
        
        treatment_section = f"\n\n**Organic Remedy:** {organic}\n**Chemical Remedy:** {chemical}"
        answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Confidence: {confidence:.1%})\n\n{answer}{treatment_section}"
        
        if caveat:
            answer = f"{answer}\n\n⚠ {caveat}"

    latency = int((time.time() - t0) * 1000)
    
    record_telemetry_log(
        db, pathway="AB", latency_ms=latency, input_text=text, image_path=web_image_path,
        intent=search_intents, detected_crop=crop, detected_disease=label,
        retrieved_chunks=hits[:5], synthesis_response=answer
    )
    
    return QueryResponse(
        pathway="AB", intent=search_intents, blocked=False, tier=tier,
        top_score=round(top_score, 4), answer=answer, sources=sources,
        latency_ms=latency, detected_crop=crop, detected_disease=label
    )

@router.post("/logs/{log_id}/feedback")
def submit_feedback(log_id: int, fb: FeedbackSubmit, db: Session = Depends(get_db)):
    """Allow client to submit feedback (+1/-1 rating and comment) for a past query log."""
    log = db.query(QueryLog).filter(QueryLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log record not found.")
        
    log.feedback_score = fb.feedback_score
    log.feedback_text = fb.feedback_text
    db.commit()
    return {"status": "success", "message": "Feedback submitted successfully."}
