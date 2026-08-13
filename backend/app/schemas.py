from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class TextQuery(BaseModel):
    text: str = Field(..., description="Query text from farmer")

class YieldQuery(BaseModel):
    crop: str = Field(..., description="Crop name (e.g. wheat, rice)")
    district: str = Field(..., description="District in Uttar Pradesh")
    area_ha: float = Field(..., description="Acreage/area in hectares")

class FeedbackSubmit(BaseModel):
    feedback_score: int = Field(..., description="1 for thumbs up, -1 for thumbs down")
    feedback_text: Optional[str] = Field(None, description="Optional text review")

class ConfigUpdate(BaseModel):
    tier_grounded: Optional[float] = None
    tier_fallback: Optional[float] = None
    weight_pdf_policy: Optional[float] = None
    weight_kcc_policy: Optional[float] = None
    weight_pdf_practice: Optional[float] = None
    weight_kcc_practice: Optional[float] = None
    mock_models: Optional[bool] = None

class RetrievedChunkSchema(BaseModel):
    id: str
    score: float
    text: str
    crop: Optional[str] = None
    district: Optional[str] = None
    source_type: Optional[str] = None
    page: Optional[int] = None
    year: Optional[int] = None

class QueryResponse(BaseModel):
    pathway: str
    intent: List[str]
    blocked: bool
    tier: str
    top_score: float
    answer: str
    sources: List[Dict[str, Any]] = []
    latency_ms: int
    detected_crop: Optional[str] = None
    detected_disease: Optional[str] = None
    predicted_yield: Optional[float] = None

class LogResponseSchema(BaseModel):
    id: int
    timestamp: datetime
    pathway: str
    input_text: Optional[str]
    image_path: Optional[str]
    intent: List[str]
    detected_crop: Optional[str]
    detected_disease: Optional[str]
    predicted_yield: Optional[float]
    retrieved_chunks: List[Dict[str, Any]]
    synthesis_response: Optional[str]
    latency_ms: int
    is_blocked: bool
    guardrail_reason: Optional[str]
    feedback_score: Optional[int]
    feedback_text: Optional[str]

    class Config:
        from_attributes = True
