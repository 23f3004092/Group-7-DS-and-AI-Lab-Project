import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FarmerVision Backend API"
    VERSION: str = "0.2.0"
    
    # Database settings
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///backend/data/farmervision.db")
    
    # Qdrant Vector DB Settings
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = "agri_knowledge"
    BGE_MODEL_ID: str = "BAAI/bge-m3"
    
    # RAG Tiers & Parameters
    TIER_GROUNDED: float = 0.66
    TIER_FALLBACK: float = 0.56
    TOP_K: int = 10
    
    # Source Weights for Intent-based blending
    WEIGHT_PDF_POLICY: float = 2.0
    WEIGHT_KCC_POLICY: float = 0.5
    WEIGHT_PDF_PRACTICE: float = 0.5
    WEIGHT_KCC_PRACTICE: float = 2.0
    
    # Cloud AI Model Credentials
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    
    # Execution Flags
    SKIP_GENERATOR: bool = bool(os.environ.get("SKIP_GENERATOR", ""))
    MOCK_MODELS: bool = True  # Enable mockup when keys are missing or local execution requested

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure backend/data folder exists
os.makedirs("backend/data", exist_ok=True)
