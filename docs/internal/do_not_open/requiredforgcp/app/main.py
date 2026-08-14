"""FastAPI gateway — the ONE API your main app calls.

This gateway serves ONLY: vision, intent/entity/guardrail (IEG), and RAG
(retrieval + generation). Mandi prices, weather, and yield prediction are
fetched by YOUR app and injected into /query via `live_data`.

Endpoints (every POST needs header  X-API-Key: <your API_KEY>):
  GET  /health     -> liveness + what's loaded (no key)
  POST /classify   -> IEG: intent + entities + guardrail + external-data hints (JSON)
  POST /query      -> RAG: guardrail -> retrieve -> generate, with injected live_data (JSON)
  POST /diagnose   -> photo: classify -> retrieve -> grounded advice (image upload)
  POST /vision     -> photo: leaf-disease classification only (image upload)
"""
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from . import config as C
from . import retrieval, generation, ieg, pipeline

# optional module
try:
    from . import vision
except Exception:
    vision = None

STATE = {"ready": False, "errors": []}


def _try(name, fn):
    try:
        fn()
    except Exception as e:  # keep serving whatever loaded
        STATE["errors"].append(f"{name}: {e}")
        print(f"[startup] {name} failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # cheap CPU stuff first, LLM last
    _try("retrieval", retrieval.load)
    _try("ieg", ieg.load)
    if C.HAS_VISION and vision is not None:
        _try("vision", vision.load)
    _try("generation", generation.load)
    STATE["ready"] = True
    yield


app = FastAPI(title="FarmerVision Gateway", version="1.2", lifespan=lifespan)


def require_key(x_api_key: Optional[str]):
    if not C.API_KEY:
        return  # no key configured -> open (dev only)
    if x_api_key != C.API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


# ------------------------------ models -------------------------------------
class QueryIn(BaseModel):
    query: str
    intent: Optional[str] = None           # policy | field_practice | general (optional override)
    top_k: Optional[int] = None
    filters: Optional[dict] = None          # e.g. {"crop": "wheat", "district": "Varanasi"}
    # Facts YOUR app fetched and wants woven into the answer. Keys are free-form;
    # known ones: mandi_prices, weather, yield. Values can be strings or JSON.
    live_data: Optional[dict] = None
    # Skip the vector search entirely (e.g. a pure weather/mandi question).
    skip_retrieval: Optional[bool] = False
    # Multi-turn: either pass prior turns yourself (client-managed) ...
    history: Optional[list] = None          # [{"role":"user"/"assistant","content":"..."}, ...]
    # ... OR pass a session_id and let the gateway remember the conversation.
    session_id: Optional[str] = None


class ClassifyIn(BaseModel):
    query: str


# ------------------------------ endpoints ----------------------------------
@app.get("/health")
def health():
    points = None
    try:
        points = retrieval._qdrant.get_collection(C.COLLECTION).points_count  # noqa: SLF001
    except Exception:
        pass
    return {
        "status": "ok" if STATE["ready"] else "starting",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "collection": C.COLLECTION,
        "points": points,
        "modules": {
            "retrieval": retrieval._qdrant is not None,                        # noqa: SLF001
            "generation": generation._model is not None,                      # noqa: SLF001
            "ieg_model": ieg.MODEL_OK,
            "vision": bool(vision and vision._model is not None),             # noqa: SLF001
        },
        "note": "mandi/weather/yield are provided by the caller via live_data",
        "errors": STATE["errors"],
    }


@app.post("/classify")
def classify(body: ClassifyIn, x_api_key: Optional[str] = Header(default=None)):
    """IEG only: intent + entities + guardrail, plus which external data (mandi/
    weather/yield) the query likely needs — so your app knows what to fetch."""
    require_key(x_api_key)
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    out = ieg.classify(body.query)
    out["suggested_external"] = ieg.external_hints(body.query)
    return out


@app.post("/query")
def query(body: QueryIn, x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    result = pipeline.answer_query(
        body.query, intent=body.intent, top_k=body.top_k, filters=body.filters,
        live_data=body.live_data, skip_retrieval=bool(body.skip_retrieval),
        history=body.history, session_id=body.session_id,
    )
    # tell the caller what external data this query looked like it wanted, so they
    # can notice if they forgot to fetch/inject something.
    result["suggested_external"] = ieg.external_hints(body.query)
    return result


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...), question: Optional[str] = Form(default=None),
                   x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    if not (vision and vision._model is not None):                            # noqa: SLF001
        raise HTTPException(status_code=501, detail="vision model not deployed")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    return pipeline.diagnose_image(image_bytes, question=question)


@app.post("/vision")
async def vision_only(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    require_key(x_api_key)
    if not (vision and vision._model is not None):                            # noqa: SLF001
        raise HTTPException(status_code=501, detail="vision model not deployed")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    return vision.predict(image_bytes)
