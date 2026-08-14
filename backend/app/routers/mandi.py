from fastapi import APIRouter, Query
from typing import Optional
from ..services.mandi_service import mandi_service

router = APIRouter(prefix="/api/mandi", tags=["Mandi Market Prices"])

@router.get("/prices")
async def get_mandi_prices(
    crop: Optional[str] = Query(None, description="Crop name (e.g. wheat, rice, maize, mustard)"),
    state: Optional[str] = Query(None, description="State name (default from server config: Uttar Pradesh)"),
    district: Optional[str] = Query(None, description="District name (default from server config: Meerut)"),
    market: Optional[str] = Query(None, description="Market/mandi name filter"),
):
    """Return the latest mandi prices for the farmer's location, optionally filtered by crop or market.

    Location is client-driven (GPS or manual selection). Falls back to static MSP
    reference prices when the live data.gov.in API is unavailable.
    """
    return await mandi_service.get_prices(crop=crop, state=state, district=district, market=market)

@router.get("/districts")
async def get_mandi_districts(
    state: Optional[str] = Query(None, description="State name (default: Uttar Pradesh)"),
):
    """Return the district pick-list for a state to populate the farmer's location selector."""
    districts = mandi_service.list_districts(state)
    return {"state": state or "Uttar Pradesh", "district_count": len(districts), "districts": districts}

@router.get("/states")
async def get_mandi_states():
    """Return all supported states/UTs to populate the farmer's location (state) selector."""
    states = mandi_service.list_states()
    return {"state_count": len(states), "states": states}