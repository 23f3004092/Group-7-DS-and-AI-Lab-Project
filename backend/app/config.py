import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    # GCP-deployed FarmerVision AI service (RAG + vision, see API_SPEC.md).
    # Used by the /ai proxy so browsers can reach it despite missing CORS headers.
    AI_API_URL: str = Field("", validation_alias=AliasChoices("AI_API_URL"))
    AI_API_KEY: str = Field("", validation_alias=AliasChoices("AI_API_KEY"))
    
    # Mandi (market prices) API — data.gov.in Agmarknet.
    # Accepts the key under either name (docs use MANDI_API_KEY; the project's
    # .env historically stores it as live_mandi_api).
    MANDI_API_KEY: str = Field(
        "",
        validation_alias=AliasChoices("MANDI_API_KEY", "live_mandi_api"),
    )
    MANDI_API_URL: str = os.environ.get(
        "MANDI_API_URL",
        "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    )
    MANDI_STATE: str = os.environ.get("MANDI_STATE", "Uttar Pradesh")
    MANDI_DISTRICT: str = os.environ.get("MANDI_DISTRICT", "Meerut")
    MANDI_CACHE_TTL: int = int(os.environ.get("MANDI_CACHE_TTL", "21600"))  # 6h (rate-limited to 100 req/day)
    
    # Weather API — indianapi.in (IMD-backed for Indian cities)
    WEATHER_API_KEY: str = os.environ.get("WEATHER_API_KEY", "")
    WEATHER_API_URL: str = os.environ.get("WEATHER_API_URL", "https://weather.indianapi.in")
    WEATHER_CACHE_TTL: int = int(os.environ.get("WEATHER_CACHE_TTL", "1800"))  # 30 min
    # 'open_meteo' (default, keyless, live) or 'indian' (indianapi.in/IMD, needs key)
    WEATHER_PROVIDER: str = os.environ.get("WEATHER_PROVIDER", "open_meteo")
    
    # Execution Flags
    SKIP_GENERATOR: bool = bool(os.environ.get("SKIP_GENERATOR", ""))
    MOCK_MODELS: bool = True  # Enable mockup when keys are missing or local execution requested

settings = Settings()

# Ensure backend/data folder exists
os.makedirs("backend/data", exist_ok=True)
