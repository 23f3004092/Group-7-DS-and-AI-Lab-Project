from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .database import engine, Base
from .config import settings
from .routers import query, admin, mcp, mandi, weather
from .services.qdrant_service import qdrant_service
from .services.yield_service import yield_service
from .services.mandi_service import mandi_service

# Create SQLAlchemy Database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="FarmerVision API: Multimodal agricultural advisory for Indian farmers.",
    version=settings.VERSION
)

# Configure CORS Middleware
# Enable react frontend and mobile connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. Change to specific domains in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads folder exists
os.makedirs("backend/uploads", exist_ok=True)

# Mount upload files as static endpoint so clients can download uploaded photos
app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")

# Include Routers
app.include_router(query.router)
app.include_router(admin.router)
app.include_router(mcp.router)
app.include_router(mandi.router)
app.include_router(weather.router)

@app.get("/health")
def health_check():
    """Verify backend system status and connection parameters."""
    qdrant_status = query.get_qdrant_status()
    
    components = {
        "sqlite_db": "OK",
        "vector_db": qdrant_status,
        "yield_model": "OK (LightGBM loaded)" if yield_service.initialized else "OK (Math-fallback active)",
        "cloud_ai_models": "OK (Active)" if settings.GEMINI_API_KEY else "OK (Running with Mockups)",
        "mandi_prices": "OK (Live)" if settings.MANDI_API_KEY else "OK (MSP Fallback)",
        "weather": "OK (Live Open-Meteo)" if settings.WEATHER_PROVIDER != "indian" else ("OK (IMD)" if settings.WEATHER_API_KEY else "OK (Static)"),
    }
    
    # Check overall state
    all_ok = qdrant_status["status"] == "connected"
    
    return {
        "status": "ok" if all_ok else "degraded",
        "version": settings.VERSION,
        "components": components
    }

@app.get("/")
def read_root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs"
    }
