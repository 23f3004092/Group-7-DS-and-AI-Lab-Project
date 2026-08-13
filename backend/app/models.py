import json
import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from .database import Base

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    pathway = Column(String, index=True)  # "A", "B", "C", "AB"
    input_text = Column(Text, nullable=True)
    image_path = Column(String, nullable=True)
    intent = Column(Text, nullable=True)  # JSON-serialized array of intents
    detected_crop = Column(String, nullable=True)
    detected_disease = Column(String, nullable=True)
    predicted_yield = Column(Float, nullable=True)
    retrieved_chunks = Column(Text, nullable=True)  # JSON-serialized chunks list
    synthesis_response = Column(Text, nullable=True)
    latency_ms = Column(Integer)
    is_blocked = Column(Boolean, default=False)
    guardrail_reason = Column(String, nullable=True)
    feedback_score = Column(Integer, nullable=True)  # +1 (thumbs up), -1 (thumbs down)
    feedback_text = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "pathway": self.pathway,
            "input_text": self.input_text,
            "image_path": self.image_path,
            "intent": json.loads(self.intent) if self.intent else [],
            "detected_crop": self.detected_crop,
            "detected_disease": self.detected_disease,
            "predicted_yield": self.predicted_yield,
            "retrieved_chunks": json.loads(self.retrieved_chunks) if self.retrieved_chunks else [],
            "synthesis_response": self.synthesis_response,
            "latency_ms": self.latency_ms,
            "is_blocked": self.is_blocked,
            "guardrail_reason": self.guardrail_reason,
            "feedback_score": self.feedback_score,
            "feedback_text": self.feedback_text,
        }

class SystemConfig(Base):
    __tablename__ = "system_configs"

    key = Column(String, primary_key=True, index=True)
    value = Column(String)
