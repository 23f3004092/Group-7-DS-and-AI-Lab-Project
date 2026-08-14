"""Weather service: Open-Meteo live data with the indianapi.in (IMD) provider ready to swap in.

Open-Meteo (keyless, free) serves current conditions + a 3-day forecast for GPS
coordinates or city names, including Indian locations. The indianapi.in provider
is fully wired up and selected with WEATHER_PROVIDER=indian, but its data
endpoints are broken upstream (Aug 2026) — Open-Meteo is the live default and
static data is the last-resort fallback, mirroring the mandi service pattern.
"""
import time
import httpx
from typing import Optional, Dict, Any, List
from ..config import settings

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"

# WMO weather codes -> (condition label)
WMO_CODES = {
    0: ("Clear sky", "☀️"),
    1: ("Mostly clear", "🌤"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫"),
    48: ("Rime fog", "🌫"),
    51: ("Light drizzle", "🌦"),
    53: ("Drizzle", "🌦"),
    55: ("Heavy drizzle", "🌧"),
    56: ("Freezing drizzle", "🌧"),
    57: ("Heavy freezing drizzle", "🌧"),
    61: ("Light rain", "🌧"),
    63: ("Rain", "🌧"),
    65: ("Heavy rain", "🌧"),
    66: ("Freezing rain", "🌧"),
    67: ("Heavy freezing rain", "🌧"),
    71: ("Light snow", "🌨"),
    73: ("Snow", "🌨"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "❄️"),
    80: ("Light showers", "🌧"),
    81: ("Rain showers", "🌧"),
    82: ("Heavy showers", "⛈"),
    85: ("Snow showers", "🌨"),
    86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈"),
    96: ("Thunderstorm with hail", "⛈"),
    99: ("Heavy thunderstorm with hail", "⛈"),
}

STATIC_FALLBACK = {
    "source": "fallback",
    "provider": "static",
    "location": "Uttar Pradesh, India",
    "temperature_c": 31.0,
    "apparent_temperature_c": None,
    "condition": "Sunny",
    "humidity": None,
    "wind_speed_kmh": None,
    "precipitation_mm": 0.0,
    "rain_probability": None,
    "max_temp_c": None,
    "min_temp_c": None,
    "forecast": [],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}


def _condition_for_wmo(code: Optional[int]) -> tuple:
    return WMO_CODES.get(code, ("Weather data", "🌡"))


class WeatherService:
    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = settings.WEATHER_CACHE_TTL
        self._fallback_ttl = 300  # retry a failed live API after 5 min

    def _cache_key(self, lat: Optional[float], lon: Optional[float], city: Optional[str]) -> str:
        if lat is not None and lon is not None:
            return f"coords|{lat:.4f}|{lon:.4f}"
        return f"city|{(city or '').strip().lower()}"

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry:
            ttl = self._fallback_ttl if entry[1].get("source") == "fallback" else self._cache_ttl
            if time.time() - entry[0] < ttl:
                return entry[1]
        return None

    def _set_cached(self, key: str, data: Dict[str, Any]):
        self._cache[key] = (time.time(), data)

    async def get_current(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        city: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return current weather for GPS coordinates (preferred) or a city name.

        Coordinates give the farmer's hyper-local conditions; the city path is
        used when only a district is known (IMD/Open-Meteo city fuzzy match).
        """
        key = self._cache_key(lat, lon, city)
        cached = self._get_cached(key)
        if cached:
            return cached

        data = None
        try:
            if settings.WEATHER_PROVIDER == "indian":
                data = await self._fetch_indian(lat, lon, city)
            else:
                data = await self._fetch_open_meteo(lat, lon, city)
        except Exception as e:
            print(f"Weather fetch failed: {e}. Using static fallback.")

        if not data:
            data = dict(STATIC_FALLBACK)
            data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        self._set_cached(key, data)
        return data

    # --- Open-Meteo (default live provider) ---

    async def _fetch_open_meteo(
        self, lat: Optional[float], lon: Optional[float], city: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if lat is None or lon is None:
            coords = await self._geocode(city or "")
            if coords is None:
                return None
            lat, lon, resolved_name = coords
        else:
            resolved_name = f"{lat:.2f}, {lon:.2f}"

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 3,
            "timezone": "auto",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(OPEN_METEO_FORECAST_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()

        current = payload.get("current") or {}
        daily = payload.get("daily") or {}
        if current.get("temperature_2m") is None:
            return None

        label, icon = _condition_for_wmo(current.get("weather_code"))
        forecast = []
        dates = daily.get("time") or []
        for i, d in enumerate(dates[:3]):
            forecast.append({
                "date": d,
                "max_temp_c": (daily.get("temperature_2m_max") or [None] * 3)[i],
                "min_temp_c": (daily.get("temperature_2m_min") or [None] * 3)[i],
                "rain_probability": (daily.get("precipitation_probability_max") or [None] * 3)[i],
            })

        return {
            "source": "live",
            "provider": "open_meteo",
            "location": resolved_name,
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "condition": f"{icon} {label}",
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_probability": (daily.get("precipitation_probability_max") or [None])[0],
            "max_temp_c": (daily.get("temperature_2m_max") or [None])[0],
            "min_temp_c": (daily.get("temperature_2m_min") or [None])[0],
            "forecast": forecast,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    async def _geocode(self, city: str) -> Optional[tuple]:
        """City/district name -> (lat, lon, display name) via Open-Meteo geocoding."""
        params = {
            "name": city.strip(),
            "count": 1,
            "language": "en",
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(OPEN_METEO_GEO_URL, params=params)
            resp.raise_for_status()
            results = (resp.json() or {}).get("results") or []
        if not results:
            return None
        first = results[0]
        parts = [p for p in [first.get("name"), first.get("admin1"), first.get("country")] if p]
        return first["latitude"], first["longitude"], ", ".join(parts)

    # --- indianapi.in / IMD provider (wired up; their data endpoints are down Aug 2026) ---

    async def _fetch_indian(
        self, lat: Optional[float], lon: Optional[float], city: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not settings.WEATHER_API_KEY:
            raise ValueError("WEATHER_API_KEY is not set for the indian weather provider")

        headers = {"x-api-key": settings.WEATHER_API_KEY}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            if lat is not None and lon is not None:
                resp = await client.get(f"{settings.WEATHER_API_URL}/global/current",
                                        params={"location": f"{lat},{lon}"})
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, dict) and payload.get("detail"):
                    raise ValueError(f"indianapi.in: {payload['detail']}")
                return {
                    "source": "live",
                    "provider": "indian",
                    "location": f"{lat:.2f}, {lon:.2f}",
                    "temperature_c": payload.get("temperature"),
                    "apparent_temperature_c": payload.get("feels_like"),
                    "condition": payload.get("condition"),
                    "humidity": payload.get("humidity"),
                    "wind_speed_kmh": payload.get("wind_speed"),
                    "precipitation_mm": None,
                    "rain_probability": None,
                    "max_temp_c": None,
                    "min_temp_c": None,
                    "forecast": [],
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

            resp = await client.get(f"{settings.WEATHER_API_URL}/india/weather",
                                    params={"city": (city or "").strip()})
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("detail"):
                raise ValueError(f"indianapi.in: {payload['detail']}")

            weather = payload.get("weather") or {}
            current = weather.get("current") or {}
            temps = current.get("temperature") or {}
            humidity = current.get("humidity") or {}
            forecast = []
            for f in (weather.get("forecast") or [])[:3]:
                forecast.append({
                    "date": f.get("date"),
                    "max_temp_c": f.get("max_temp"),
                    "min_temp_c": f.get("min_temp"),
                    "rain_probability": None,
                })
            astron = weather.get("astronomical") or {}
            condition = ""
            if forecast:
                condition = forecast[0].get("description") or ""
            if astron.get("sunrise"):
                condition = f"{condition} (☀️ {astron['sunrise']} – 🌇 {astron['sunset']})".strip()
            return {
                "source": "live",
                "provider": "indian",
                "location": payload.get("city") or (city or ""),
                "temperature_c": temps.get("max", {}).get("value"),
                "apparent_temperature_c": None,
                "condition": condition,
                "humidity": humidity.get("morning"),
                "wind_speed_kmh": None,
                "precipitation_mm": current.get("rainfall"),
                "rain_probability": None,
                "max_temp_c": temps.get("max", {}).get("value"),
                "min_temp_c": temps.get("min", {}).get("value"),
                "forecast": forecast,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }


weather_service = WeatherService()