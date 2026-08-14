import json
from pydantic import BaseModel, Field, model_validator, field_validator
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

    @field_validator("feedback_score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("feedback_score must be 1 (thumbs up) or -1 (thumbs down)")
        return v

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
    input_text: Optional[str] = None
    image_path: Optional[str] = None
    intent: List[str] = []
    detected_crop: Optional[str] = None
    detected_disease: Optional[str] = None
    predicted_yield: Optional[float] = None
    retrieved_chunks: List[Dict[str, Any]] = []
    synthesis_response: Optional[str] = None
    latency_ms: int = 0
    is_blocked: bool = False
    guardrail_reason: Optional[str] = None
    feedback_score: Optional[int] = None
    feedback_text: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def deserialize_json_fields(cls, values):
        """Convert JSON-string columns from SQLite into Python lists/dicts."""
        # Support both dict (from ORM to_dict) and ORM object
        if hasattr(values, '__dict__'):
            values = values.__dict__
        if isinstance(values, dict):
            for field in ('intent', 'retrieved_chunks'):
                val = values.get(field)
                if isinstance(val, str):
                    try:
                        values[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        values[field] = []
                elif val is None:
                    values[field] = []
        return values

    model_config = {'from_attributes': True}
