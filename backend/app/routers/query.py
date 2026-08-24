import os
import json
import re
import time
import uuid
import asyncio
import datetime
import unicodedata
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from ..database import get_db
from ..models import QueryLog, SystemConfig
from ..schemas import TextQuery, YieldQuery, QueryResponse, FeedbackSubmit
from ..services.cloud_models import CloudAIService
from ..services.qdrant_service import qdrant_service
from ..services.yield_service import yield_service
from ..services.weather_service import weather_service
from ..services.mandi_service import mandi_service, CROP_ALIASES
from ..services.cost_service import cost_service
from ..config import settings

router = APIRouter(prefix="/api/query", tags=["Advisory Query Pipelines"])


@router.get("/source/{source_id}")
async def get_source_detail(source_id: str):
    """Return the full text + metadata of a single RAG chunk by id.

    The chat source chips only carry a 150-char preview; the citation page calls
    this endpoint to display the complete advisory content.
    """
    source = qdrant_service.get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source

# Create upload directory
UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

CROP_ALIAS_MAP = {
    "wheat": ["wheat", "gehu", "गेहूं", "गेहूँ"],
    "rice": ["rice", "paddy", "dhan", "चावल", "धान"],
    "maize": ["maize", "makka", "makkai", "मक्का"],
    "mustard": ["mustard", "sarson", "सरसों"],
    "sugarcane": ["sugarcane", "ganna", "गन्ना"],
    "potato": ["potato", "aloo", "आलू"],
    "mango": ["mango", "aam", "आम"],
}
LIVE_DATA_TRIGGERS = {
    "mandi_prices": [
        r"\bmandi\w*", r"\bbhav\b", r"\bbhaav\b", r"\brate\b", r"\bprice\w*",
        r"\bsell\w*", r"\bbech\w*", r"\bsale\b", r"\bbazar\b", r"\bmarket\b",
        r"\bभाव\b", r"\bमंडी\b", r"\bरेट\b", r"\bदाम\b", r"\bकीमत\b", r"\bबेच\b",
        r"\bबाज़ार\b", r"\bबाजार\b",
    ],
    "weather": [
        r"\bweather\b", r"\bmausam\b", r"\btemperature\b", r"\brainy?\b",
        r"\bforecast\b", r"\bupcoming days\b", r"\bsunny\b",
        r"\bbarish\b", r"\bbaarish\b", r"\bbarsaat\w*", r"\bpani\b", r"\bgarmi\b",
        r"\bगर्मी\b", r"\bमौसम\b", r"\bबारिश\b", r"\bतापमान\b", r"\bबरसात\b", r"\bपानी\b",
        r"\bबादल\b", r"\bधूप\b",
    ],
    "yield": [
        r"\byield\b", r"\bproduction\b", r"\bupaj\b",
        r"\bquintal\w*", r"\buple\b",
        r"\bउपज\b", r"\bपैदावार\b", r"\bउत्पादन\b", r"\bक्विंटल\b",
    ],
}

def detect_live_data_needs(text: str) -> List[str]:
    """Bilingual (English/Hindi/Hinglish) rule-based detector for live-data needs.

    Text and patterns are NFD-normalized so precomposed vs decomposed nukta
    forms (फ़/ज़/ड़...) always match each other.
    """
    def _nfd(s: str) -> str:
        return unicodedata.normalize("NFD", s)

    t = _nfd(text.lower())
    needs = []
    for key, patterns in LIVE_DATA_TRIGGERS.items():
        if any(re.search(_nfd(p), t) for p in patterns):
            needs.append(key)
    return needs

def detect_crop_entity(text: str) -> Optional[str]:
    """Word-boundary crop detection (English + Hinglish/Hindi aliases).

    Uses regex boundaries so substrings like 'rice' inside 'prices' do not
    produce a false crop match.
    """
    text_lower = text.lower()
    for crop, aliases in CROP_ALIAS_MAP.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", text_lower) for alias in aliases):
            return crop
    return None

def _format_mandi_fact(crop: Optional[str], mandi: Dict[str, Any], msp: Optional[float]) -> Optional[str]:
    """Format 'Wheat MSP: Rs 2275/quintal, today's price: Rs 2450/quintal (2026-08-16)'.

    When no crop is named, summarize the top available commodities instead."""
    if not isinstance(mandi, dict):
        return None
    prices = mandi.get("prices") or []
    date = (mandi.get("fetched_at") or "")[:10]

    def _fmt(v: float) -> str:
        return f"{v:g}" if v == int(v) else f"{v:.2f}"

    if crop:
        commodity = CROP_ALIASES.get(crop, crop.title())
        row = None
        for p in prices:
            if p.get("crop", "").lower() == commodity.lower() and p.get("modal_price") is not None:
                row = p
                break
        msp_val = msp if msp is not None else 0
        today = row["modal_price"] if row else None
        date = ((row.get("arrival_date") if row else None) or date)[:10]
        label = commodity.title()
        if msp_val > 0:
            fact = f"{label} MSP: Rs {_fmt(msp_val)}/quintal"
            if today is not None:
                fact += f", today's price: Rs {_fmt(today)}/quintal ({date})"
        elif today is not None:
            fact = f"{label} mandi price today: Rs {_fmt(today)}/quintal ({date})"
        else:
            return None
        return fact

    # No specific crop -> list up to 3 distinct commodities that have a live modal price.
    parts = []
    seen_crops = set()
    for p in prices:
        name = p.get("crop")
        if not name or p.get("modal_price") is None:
            continue
        key = name.lower()
        if key in seen_crops:
            continue
        seen_crops.add(key)
        market = p.get("market")
        loc = f" ({market})" if market else ""
        parts.append(f"{name}{loc}: Rs {_fmt(p['modal_price'])}/quintal")
        if len(parts) >= 3:
            break
    if parts:
        return f"Mandi prices ({mandi.get('district', '')}, {date}): " + ", ".join(parts)
    return None

def _format_weather_fact(weather: Dict[str, Any]) -> Optional[str]:
    """Format 'little rainfall next 3 days, temperature 28-33C'."""
    forecast = weather.get("forecast") or []
    rainy = 0
    for f in forecast[:3]:
        prob = f.get("rain_probability")
        if prob is not None and float(prob) > 50:
            rainy += 1
    if rainy == 0:
        rain_txt = "no rainfall next 3 days"
    elif rainy <= 1:
        rain_txt = "little rainfall next 3 days"
    else:
        rain_txt = f"rain likely on {rainy} of next 3 days"
    lo, hi = weather.get("min_temp_c"), weather.get("max_temp_c")
    temp_txt = f"temperature {lo:.0f}-{hi:.0f}C" if lo is not None and hi is not None else ""
    if temp_txt:
        return f"{rain_txt}, {temp_txt}"
    return rain_txt

def _format_yield_fact(crop: str, pred_t_ha: float) -> str:
    """Format 'Estimated yield: 42 quintal/acre' (1 t/ha = 10 qtl/2.47 acre)."""
    qtl_per_acre = pred_t_ha * 10 / 2.47105
    return f"Estimated {crop} yield: {qtl_per_acre:.1f} quintal/acre"

async def build_live_data_facts(
    needs: List[str], crop: Optional[str], state: Optional[str],
    district: Optional[str], lat: Optional[float], lon: Optional[float],
) -> Dict[str, str]:
    """Fetch and format live mandi/weather/yield facts for the advisory answer."""
    state = (state or settings.MANDI_STATE).strip()
    district = (district or settings.MANDI_DISTRICT).strip()
    facts: Dict[str, str] = {}
    weather = None

    wants_weather = "weather" in needs
    wants_mandi = "mandi_prices" in needs
    wants_yield = "yield" in needs and crop is not None

    tasks = []
    if wants_weather:
        tasks.append(weather_service.get_current(lat=lat, lon=lon, city=district or None))
    if wants_mandi:
        tasks.append(mandi_service.get_prices(crop=crop, state=state, district=district))
        # Keep result indices aligned: msp slot resolves to None when no crop
        # is named (generic "what are the mandi prices" queries).
        tasks.append(cost_service.get_msp(crop) if crop else asyncio.sleep(0, result=None))
    if wants_yield:
        tasks.append(cost_service.get_cost_per_ha(crop, state=state))

    results = list(await asyncio.gather(*tasks)) if tasks else []

    idx = 0
    if wants_weather:
        weather = results[idx]
        facts["weather"] = _format_weather_fact(weather)
        idx += 1
    if wants_mandi:
        facts["mandi_prices"] = _format_mandi_fact(crop, results[idx], results[idx + 1])
        idx += 2
    if wants_yield:
        annual_rainfall = None
        if weather is not None:
            annual_rainfall = weather.get("precipitation_mm")
            if annual_rainfall is None and weather.get("rain_probability") is not None:
                annual_rainfall = weather["rain_probability"] * 0.1
        pred_t_ha, _ = yield_service.predict(crop, district, 1.0, annual_rainfall=annual_rainfall)
        facts["yield"] = _format_yield_fact(crop, pred_t_ha)

    return facts
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
        
    # 2. Identify crop entities mentioned (English + Hinglish/Hindi aliases)
    detected_crop = detect_crop_entity(text)

    # 3. Retrieve documents from Vector DB
    hits, tier, top_score = qdrant_service.retrieve(text, intents=intents, current_settings=cfg)
    caveat = get_caveat_text(tier)
    
    # Format sources list
    sources = [{
        "id": hit["id"],
        "rank": idx + 1,
        "score": round(hit["score"], 3),
        "text": hit["text"][:150] + "...",
        "full_text": hit["text"],
        "crop": hit["crop"],
        "source_type": hit["source_type"]
    } for idx, hit in enumerate(hits[:5])]

    # 4. Detect live-data needs (mandi / weather / yield) and fetch facts
    live_data = {}
    needs = detect_live_data_needs(text)
    if needs:
        live_data = await build_live_data_facts(
            needs, detected_crop, q.state, q.district, q.lat, q.lon
        )

    # 5. Synthesize answer (live-data questions are answered even when RAG abstains)
    if tier == "abstain" and not live_data:
        answer = caveat
    else:
        answer = await CloudAIService.synthesize_response(text, hits[:5], live_data=live_data or None)
        if caveat and not live_data:
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
        latency_ms=latency, detected_crop=detected_crop,
        live_data=live_data or None
    )

@router.post("/text_stream")
async def query_text_stream(q: TextQuery, db: Session = Depends(get_db)):
    """Pathway A with Server-Sent Events — same pipeline as /text but streams the
    synthesis token-by-token. Event protocol matches the GCP gateway (API_SPEC.md):
    `status` (classifying/retrieving/generating) -> `delta`* -> `final`."""
    t0 = time.time()
    text = q.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    def _sse(event: str, payload: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def event_gen():
        cfg = get_current_settings(db)

        yield _sse("status", {"type": "status", "stage": "classifying"})
        intents, blocked, block_reason = await CloudAIService.run_intent_entity_guardrails(text)

        if blocked:
            answer = "This query cannot be answered safely. Please contact a KVK officer."
            latency = int((time.time() - t0) * 1000)
            log_id = record_telemetry_log(
                db, pathway="A", latency_ms=latency, input_text=text,
                intent=intents, is_blocked=True, guardrail_reason=block_reason,
                synthesis_response=answer
            )
            yield _sse("final", {"type": "final", "data": {
                "pathway": "A", "intent": intents, "blocked": True, "tier": "blocked",
                "top_score": 0.0, "answer": answer, "sources": [],
                "latency_ms": latency, "log_id": log_id,
            }})
            return

        # Crop entity detection (shared alias map)
        detected_crop = detect_crop_entity(text)

        yield _sse("status", {"type": "status", "stage": "retrieving"})
        hits, tier, top_score = qdrant_service.retrieve(text, intents=intents, current_settings=cfg)
        caveat = get_caveat_text(tier)

        sources = [{
            "id": hit["id"],
            "rank": idx + 1,
            "score": round(hit["score"], 3),
            "text": hit["text"][:150] + "...",
            "full_text": hit["text"],
            "crop": hit["crop"],
            "source_type": hit["source_type"]
        } for idx, hit in enumerate(hits[:5])]

        # Live-data facts (mandi / weather / yield)
        needs = detect_live_data_needs(text)
        live_data = {}
        if needs:
            live_data = await build_live_data_facts(
                needs, detected_crop, q.state, q.district, q.lat, q.lon
            )

        yield _sse("status", {"type": "status", "stage": "generating",
                              "tier": tier, "top_score": round(top_score, 4)})

        deltas: List[str] = []
        if tier == "abstain" and not live_data:
            answer = caveat
            yield _sse("delta", {"type": "delta", "text": answer})
        else:
            async for delta in CloudAIService.synthesize_response_stream(
                text, hits[:5], live_data=live_data or None
            ):
                if not delta:
                    continue
                deltas.append(delta)
                yield _sse("delta", {"type": "delta", "text": delta})
            answer = "".join(deltas)
            if caveat and not live_data:
                answer = f"{answer}\n\n⚠ {caveat}"
                yield _sse("delta", {"type": "delta", "text": f"\n\n⚠ {caveat}"})

        latency = int((time.time() - t0) * 1000)
        log_id = record_telemetry_log(
            db, pathway="A", latency_ms=latency, input_text=text,
            intent=intents, detected_crop=detected_crop,
            retrieved_chunks=hits[:5], synthesis_response=answer,
            is_blocked=False
        )

        yield _sse("final", {"type": "final", "data": {
            "pathway": "A", "intent": intents, "blocked": False, "tier": tier,
            "top_score": round(top_score, 4), "answer": answer, "sources": sources,
            "latency_ms": latency, "detected_crop": detected_crop,
            "live_data": live_data or None, "log_id": log_id,
        }})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.post("/image", response_model=QueryResponse)
async def query_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Pathway B — Leaf photo upload. Runs image disease classification, RAG retrieval, and Synthesis."""
    t0 = time.time()
    cfg = get_current_settings(db)
    
    # Save image
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    saved_name = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    
    image_bytes = await file.read()
    with open(saved_path, "wb") as f:
        f.write(image_bytes)
        
    web_image_path = f"/uploads/{saved_name}"

    # 1. Run Vision Classifier
    result = await CloudAIService.run_vision_diagnosis(image_bytes, file.filename or saved_name)
    
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
        "id": hit["id"],
        "rank": idx + 1,
        "score": round(hit["score"], 3),
        "text": hit["text"][:150] + "...",
        "full_text": hit["text"],
        "crop": hit["crop"],
        "source_type": hit["source_type"]
    } for idx, hit in enumerate(hits[:5])]

    # 4. Generate
    if tier == "abstain":
        answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Conf: {confidence:.1%})\n\n{caveat}"
    else:
        answer = await CloudAIService.synthesize_response(query, hits[:5])
        # Inject details from classifier if not in context (only when available)
        organic = result.get("organic_treatment", "") or ""
        chemical = result.get("chemical_treatment", "") or ""

        header = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Confidence: {confidence:.1%})\n\n{answer}"
        if organic or chemical:
            header += f"\n\n**Organic Remedy:** {organic}\n**Chemical Remedy:** {chemical}"
        answer = header

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

@router.post("/vision", response_model=QueryResponse)
async def query_vision(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """VIT-only classification for the Leaf Scanner — no LLM synthesis.

    The photo goes to the ViT (/vision) and the result is returned as-is;
    treatment/advisory text is intentionally NOT generated here. Chat uses
    /image or the gateway /diagnose for synthesis + VIT.
    """
    t0 = time.time()

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    saved_name = f"{uuid.uuid4()}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    image_bytes = await file.read()
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    web_image_path = f"/uploads/{saved_name}"

    # 1. Run the ViT classifier (real model via gateway /vision, mock offline)
    result = await CloudAIService.run_vision_diagnosis(image_bytes, file.filename or saved_name)
    latency = int((time.time() - t0) * 1000)

    if result.get("rejected", False):
        answer = "Image rejected — please submit a clear, close-up crop leaf photo."
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

    answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Confidence: {confidence:.1%})"

    record_telemetry_log(
        db, pathway="B", latency_ms=latency, image_path=web_image_path,
        intent=["disease_pest"], detected_crop=crop, detected_disease=label,
        synthesis_response=answer
    )

    return QueryResponse(
        pathway="B", intent=["disease_pest"], blocked=False, tier="vision",
        top_score=round(confidence, 4), answer=answer,
        latency_ms=latency, detected_crop=crop, detected_disease=label
    )

@router.post("/yield")
async def query_yield(q: YieldQuery, db: Session = Depends(get_db)):
    """Pathway C — Crop Yield & Profitability estimation using the trained model,
    live weather (rainfall) and live mandi prices."""
    t0 = time.time()
    
    crop = q.crop.lower().strip()
    district = q.district.lower().strip()
    area = q.area_ha
    
    # 1. Fetch live weather, mandi prices and CACP cost data in parallel
    weather, mandi, cacp_msp, cost_per_ha = await asyncio.gather(
        weather_service.get_current(lat=q.lat, lon=q.lon, city=district),
        mandi_service.get_prices(crop=crop, state="Uttar Pradesh", district=district),
        cost_service.get_msp(crop),
        cost_service.get_cost_per_ha(crop, state="Uttar Pradesh"),
    )
    
    # 2. Feed live rainfall into the yield model (current precipitation, else rain probability)
    annual_rainfall = weather.get("precipitation_mm")
    if annual_rainfall is None and weather.get("rain_probability") is not None:
        annual_rainfall = weather["rain_probability"] * 0.1
    pred_t_ha, total_yield = yield_service.predict(crop, district, area, annual_rainfall=annual_rainfall)
    
    # 3. Price: live mandi > CACP MSP > static constants
    live_price = None
    price_source = None
    if mandi.get("source") == "live":
        commodity = CROP_ALIASES.get(crop, crop.title())
        for p in mandi.get("prices", []):
            if p.get("crop", "").lower() == commodity.lower() and p.get("modal_price") is not None:
                live_price = p["modal_price"]
                price_source = "mandi"
                break
    if live_price is None and cacp_msp is not None:
        live_price = cacp_msp
        price_source = "cacp_msp"
    economics = yield_service.estimate_profitability(
        crop, total_yield, area,
        market_price_per_quintal=live_price, cost_per_ha=cost_per_ha,
        price_source=price_source,
    )
    
    price_label = {
        "mandi": f"Mandi ₹{economics['price_per_quintal']}",
        "cacp_msp": f"CACP MSP ₹{economics['price_per_quintal']}",
        "msp": f"MSP ₹{economics['price_per_quintal']}",
    }.get(economics["price_source"], f"MSP ₹{economics['price_per_quintal']}")
    cost_label = f"CACP (₹{economics['cost_per_ha']}/ha)" if economics["cost_source"] == "cacp" else "Estimate"
    answer = (
        f"Expected yield for {crop.title()} on {area} ha in {district.title()}: "
        f"~{pred_t_ha:.2f} t/ha ({total_yield:.2f} tonnes total).\n\n"
        f"**Economic Projections ({price_label}/quintal, cost: {cost_label}):**\n"
        f"- Est. Cultivation Cost: ₹{economics['total_cost']:,}\n"
        f"- Est. Gross Revenue: ₹{economics['total_revenue']:,}\n"
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
        "predicted_yield_t_ha": round(pred_t_ha, 2),
        "total_yield_t": total_yield,
        "economics": economics,
        "weather": weather,
        "mandi": mandi
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
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
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
    result = await CloudAIService.run_vision_diagnosis(image_bytes, file.filename or saved_name)
    
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
            "id": hit["id"],
            "rank": idx + 1,
            "score": round(hit["score"], 3),
            "text": hit["text"][:150] + "...",
            "full_text": hit["text"],
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
            "id": hit["id"],
            "rank": idx + 1,
            "score": round(hit["score"], 3),
            "text": hit["text"][:150] + "...",
            "full_text": hit["text"],
            "crop": hit["crop"],
            "source_type": hit["source_type"]
        } for idx, hit in enumerate(hits[:5])]

    # 5. Generate
    if tier == "abstain":
        answer = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Conf: {confidence:.1%})\n\n{caveat}"
    else:
        answer = await CloudAIService.synthesize_response(combined_query, hits[:5])
        organic = result.get("organic_treatment", "") or ""
        chemical = result.get("chemical_treatment", "") or ""

        header = f"Diagnosed Crop/Disease: {crop.title()} - {disease.title()} (Confidence: {confidence:.1%})\n\n{answer}"
        if organic or chemical:
            header += f"\n\n**Organic Remedy:** {organic}\n**Chemical Remedy:** {chemical}"
        answer = header

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
def submit_feedback(log_id: int, fb: FeedbackSubmit = Body(...), db: Session = Depends(get_db)):
    """Allow client to submit feedback (+1/-1 rating and comment) for a past query log."""
    log = db.query(QueryLog).filter(QueryLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log record not found.")
        
    log.feedback_score = fb.feedback_score
    log.feedback_text = fb.feedback_text
    db.commit()
    return {"status": "success", "message": "Feedback submitted successfully."}
