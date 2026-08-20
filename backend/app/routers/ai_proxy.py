"""CORS-friendly reverse proxy for the GCP-deployed FarmerVision AI service.

The GCP deployment (see API_SPEC.md) does not send CORS headers, so browsers
block direct calls from the Expo web app. This router forwards /ai/* requests
to the GCP service with the API key attached server-side; the local backend
already sends the CORS headers that make browser calls work.
"""
import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ..config import settings

router = APIRouter(prefix="/ai", tags=["AI Proxy"])

AI_BASE = settings.AI_API_URL.rstrip("/") if settings.AI_API_URL else ""
AI_KEY = settings.AI_API_KEY
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


async def _forward(method: str, path: str, **kwargs):
    if not AI_BASE:
        raise HTTPException(status_code=503, detail="AI proxy not configured (set AI_API_URL in .env)")
    headers = kwargs.pop("headers", {})
    if AI_KEY:
        headers["X-API-Key"] = AI_KEY
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.request(method, f"{AI_BASE}{path}", headers=headers, **kwargs)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"AI service unreachable: {exc}")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


@router.get("/health")
async def proxy_health():
    # NOTE: GCP's /health needs no API key, so a bare forward returns "ok" even
    # when the proxy is missing the key that /vision and /query require. Attach
    # the proxy's own readiness so clients can tell "reachable" apart from
    # "can actually run authenticated vision/query requests".
    data = await _forward("GET", "/health")
    if isinstance(data, dict):
        data["proxy"] = {
            "ai_url_configured": bool(AI_BASE),
            "ai_key_configured": bool(AI_KEY),
            "vision_capable": bool(AI_BASE and AI_KEY),
        }
    return data


@router.post("/classify")
async def proxy_classify(payload: dict):
    return await _forward("POST", "/classify", json=payload)


@router.post("/query")
async def proxy_query(payload: dict):
    return await _forward("POST", "/query", json=payload)


@router.post("/diagnose")
async def proxy_diagnose(
    file: UploadFile = File(...),
    question: str = Form(default=""),
):
    data = {"question": question} if question else None
    files = {"file": (file.filename or "leaf.jpg", await file.read(), file.content_type or "image/jpeg")}
    return await _forward("POST", "/diagnose", data=data, files=files)


@router.post("/vision")
async def proxy_vision(file: UploadFile = File(...)):
    files = {"file": (file.filename or "leaf.jpg", await file.read(), file.content_type or "image/jpeg")}
    return await _forward("POST", "/vision", files=files)
