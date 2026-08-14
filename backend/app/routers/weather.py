from fastapi import APIRouter, Query
from typing import Optional
from ..services.weather_service import weather_service

router = APIRouter(prefix="/api/weather", tags=["Weather"])

@router.get("/current")
async def get_current_weather(
    lat: Optional[float] = Query(None, description="Latitude from GPS (preferred for hyper-local data)"),
    lon: Optional[float] = Query(None, description="Longitude from GPS (preferred for hyper-local data)"),
    city: Optional[str] = Query(None, description="City or district name (used when no GPS fix)"),
):
    """Return current weather + 3-day forecast for the farmer's location.

    Location is client-driven: GPS coordinates when available, otherwise the
    selected district/city name (fuzzy-matched). Falls back to static data when
    the live providers are unreachable.
    """
    return await weather_service.get_current(lat=lat, lon=lon, city=city)