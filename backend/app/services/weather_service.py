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
WTTR_URL = "https://wttr.in/{query}?format=j1"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
MET_NO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

_OSM_UA = {"User-Agent": "FarmerVision/1.0 (agricultural advisory app)"}


def _condition_for_symbol(code: Optional[str]) -> tuple:
    """met.no symbol_code -> (label, icon)."""
    c = (code or "").lower()
    for suffix in ("_day", "_night", "_polartwilight"):
        c = c.replace(suffix, "")
    if "thunder" in c:
        return ("Thunderstorm", "⛈")
    if "heavyrain" in c:
        return ("Heavy rain", "🌧")
    if "rain" in c or "sleet" in c or "drizzle" in c or "showers" in c:
        return ("Light rain" if c.startswith("light") else "Rain", "🌧")
    if "snow" in c:
        return ("Snow", "🌨")
    if "fog" in c:
        return ("Fog", "🌫")
    if "overcast" in c:
        return ("Overcast", "☁️")
    if "cloudy" in c or "partlycloudy" in c:
        return ("Partly cloudy", "⛅")
    if "clearsky" in c or "fair" in c:
        return ("Clear sky", "☀️")
    return ("Weather data", "🌡")


def _symbol_rain_chance(code: Optional[str]) -> int:
    """Heuristic rain probability (%) from a met.no symbol_code."""
    c = (code or "").lower()
    if any(w in c for w in ("rain", "sleet", "snow", "thunder", "drizzle")):
        return 85
    if any(w in c for w in ("cloudy", "overcast", "fog")):
        return 30
    return 10

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
    "wind_gusts_kmh": None,
    "wind_direction_deg": None,
    "wind_direction_label": None,
    "pressure_hpa": None,
    "dew_point_c": None,
    "cloud_cover_pct": None,
    "uv_index": None,
    "wmo_code": None,
    "sunrise": None,
    "sunset": None,
    "precipitation_mm": 0.0,
    "rain_probability": None,
    "max_temp_c": None,
    "min_temp_c": None,
    "forecast": [],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass_label(deg: Optional[float]) -> Optional[str]:
    """Degrees -> 16-point compass label (N, NNE, NE, ...)."""
    if deg is None or deg < 0:
        return None
    return WIND_DIRS[int(round((float(deg) % 360) / 22.5)) % 16]


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
            print(f"Weather fetch failed ({settings.WEATHER_PROVIDER}): {e}. Trying fallback providers.")

        # Fallback chain — Open-Meteo rate-limits (429) shared datacenter egress
        # IPs (seen on Render free tier), so degrade through met.no then wttr.in
        # before giving up and serving static data.
        if not data:
            try:
                data = await self._fetch_met_no(lat, lon, city)
            except Exception as e:
                print(f"met.no fallback failed: {e}.")
        if not data:
            try:
                data = await self._fetch_wttr(lat, lon, city)
            except Exception as e:
                print(f"wttr.in fallback failed: {e}. Using static fallback.")

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
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
                "wind_gusts_10m,surface_pressure,dew_point_2m,cloud_cover,uv_index"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                "weather_code,sunrise,sunset"
            ),
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
        wmo_daily = daily.get("weather_code") or []
        for i, d in enumerate(dates[:3]):
            forecast.append({
                "date": d,
                "wmo_code": (wmo_daily or [None] * 3)[i],
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
            "wind_gusts_kmh": current.get("wind_gusts_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "wind_direction_label": _compass_label(current.get("wind_direction_10m")),
            "pressure_hpa": current.get("surface_pressure"),
            "dew_point_c": current.get("dew_point_2m"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "uv_index": current.get("uv_index"),
            "wmo_code": current.get("weather_code"),
            "sunrise": (daily.get("sunrise") or [None])[0],
            "sunset": (daily.get("sunset") or [None])[0],
            "precipitation_mm": current.get("precipitation"),
            "rain_probability": (daily.get("precipitation_probability_max") or [None])[0],
            "max_temp_c": (daily.get("temperature_2m_max") or [None])[0],
            "min_temp_c": (daily.get("temperature_2m_min") or [None])[0],
            "forecast": forecast,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # --- met.no Locationforecast (primary fallback; keyless, coords-native) ---

    async def _fetch_met_no(
        self, lat: Optional[float], lon: Optional[float], city: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fallback provider: Norwegian met.no Locationforecast 2.0 compact.

        Reliable and keyless (User-Agent required). Accepts raw GPS
        coordinates, so it works even when wttr.in's place-name lookups fail.
        """
        if lat is None or lon is None:
            place = await self._search_place(city or "")
            if place is None:
                return None
            lat, lon, resolved_name = place
        else:
            resolved_name = f"{lat:.2f}, {lon:.2f}"

        async with httpx.AsyncClient(timeout=20.0, headers=_OSM_UA) as client:
            resp = await client.get(MET_NO_URL, params={"lat": lat, "lon": lon})
            resp.raise_for_status()
            payload = resp.json()

        series = ((payload.get("properties") or {}).get("timeseries")) or []
        if not series:
            return None

        def details(entry: Dict[str, Any]) -> Dict[str, Any]:
            return (((entry.get("data") or {}).get("instant") or {}).get("details")) or {}

        def symbol_of(entry: Dict[str, Any]) -> Optional[str]:
            data = entry.get("data") or {}
            for h in ("next_1_hours", "next_6_hours"):
                sym = ((data.get(h) or {}).get("summary") or {}).get("symbol_code")
                if sym:
                    return sym
            return None

        def precip_of(entry: Dict[str, Any]) -> Optional[float]:
            data = entry.get("data") or {}
            for h in ("next_1_hours", "next_6_hours"):
                amt = ((data.get(h) or {}).get("details") or {}).get("precipitation_amount")
                if amt is not None:
                    return float(amt)
            return None

        current_entry = series[0]
        cur = details(current_entry)
        cur_sym = symbol_of(current_entry)
        label, icon = _condition_for_symbol(cur_sym)

        # Group the next 3 days: max/min temp + max symbol rain chance per day.
        by_day: Dict[str, Dict[str, Any]] = {}
        for entry in series:
            day = str(entry.get("time", ""))[:10]
            if not day or (by_day and len(by_day) >= 3 and day not in by_day):
                continue
            d = details(entry)
            t = d.get("air_temperature")
            bucket = by_day.setdefault(day, {"max": None, "min": None, "chance": 0, "precip": 0.0})
            if isinstance(t, (int, float)):
                bucket["max"] = t if bucket["max"] is None else max(bucket["max"], t)
                bucket["min"] = t if bucket["min"] is None else min(bucket["min"], t)
            bucket["chance"] = max(bucket["chance"], _symbol_rain_chance(symbol_of(entry)))
            p = precip_of(entry)
            if p:
                bucket["precip"] += p

        forecast = [{
            "date": day,
            "wmo_code": None,
            "max_temp_c": b["max"],
            "min_temp_c": b["min"],
            "rain_probability": b["chance"] or None,
        } for day, b in sorted(by_day.items())[:3]]

        wind_deg = cur.get("wind_from_direction")
        wind_kmh = cur.get("wind_speed")
        if isinstance(wind_kmh, (int, float)):
            wind_kmh = round(wind_kmh * 3.6, 1)

        first = forecast[0] if forecast else {}
        return {
            "source": "live",
            "provider": "met_no",
            "location": resolved_name,
            "temperature_c": cur.get("air_temperature"),
            "apparent_temperature_c": None,
            "condition": f"{icon} {label}",
            "humidity": cur.get("relative_humidity"),
            "wind_speed_kmh": wind_kmh,
            "wind_gusts_kmh": None,
            "wind_direction_deg": wind_deg,
            "wind_direction_label": _compass_label(wind_deg),
            "pressure_hpa": cur.get("air_pressure_at_sea_level"),
            "dew_point_c": cur.get("dew_point_temperature"),
            "cloud_cover_pct": cur.get("cloud_area_fraction"),
            "uv_index": None,
            "wmo_code": None,
            "sunrise": None,
            "sunset": None,
            "precipitation_mm": precip_of(current_entry),
            "rain_probability": first.get("rain_probability"),
            "max_temp_c": first.get("max_temp_c"),
            "min_temp_c": first.get("min_temp_c"),
            "forecast": forecast,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    async def _search_place(self, name: str) -> Optional[tuple]:
        """City/district name -> (lat, lon, display name) via OSM Nominatim."""
        params = {"q": name.strip(), "format": "json", "limit": 1, "countrycodes": "in"}
        async with httpx.AsyncClient(timeout=15.0, headers=_OSM_UA) as client:
            resp = await client.get(NOMINATIM_SEARCH_URL, params=params)
            resp.raise_for_status()
            results = resp.json() or []
        if not results:
            return None
        first = results[0]
        display = first.get("display_name") or name
        return float(first["lat"]), float(first["lon"]), ", ".join(display.split(", ")[:3])

    # --- wttr.in (secondary keyless provider) ---

    async def _fetch_wttr(
        self, lat: Optional[float], lon: Optional[float], city: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fallback provider: wttr.in JSON API (keyless, 3-day forecast).

        Used when Open-Meteo is unreachable/rate-limited from the server's IP.
        wttr.in only resolves place names reliably, so GPS coordinates are
        reverse-geocoded to a city via Nominatim first.
        """
        if lat is not None and lon is not None:
            city = await self._reverse_geocode(lat, lon) or city
            if not city:
                return None
            query = (city or "").strip()
            resolved_name = f"{lat:.2f}, {lon:.2f}"
        else:
            query = (city or "").strip()
            if not query:
                return None
            resolved_name = query.title()

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(WTTR_URL.format(query=query), headers={"User-Agent": "curl/8.0.1"})
            resp.raise_for_status()
            payload = resp.json()

        current = (payload.get("current_condition") or [None])[0]
        if not current or current.get("temp_C") is None:
            return None

        def _desc(cc: Dict[str, Any]) -> str:
            d = (cc.get("weatherDesc") or [{}])[0].get("value")
            return d or "Weather data"

        days = payload.get("weather") or []
        forecast = []
        for d in days[:3]:
            hourly = d.get("hourly") or []
            chances = [int(h.get("chanceofrain") or 0) for h in hourly]
            forecast.append({
                "date": d.get("date"),
                "wmo_code": None,
                "max_temp_c": float(d["maxtempC"]) if d.get("maxtempC") not in (None, "") else None,
                "min_temp_c": float(d["mintempC"]) if d.get("mintempC") not in (None, "") else None,
                "rain_probability": max(chances) if chances else None,
            })

        wind_deg = float(current["winddirDegree"]) if current.get("winddirDegree") not in (None, "") else None

        return {
            "source": "live",
            "provider": "wttr",
            "location": resolved_name,
            "temperature_c": float(current["temp_C"]),
            "apparent_temperature_c": float(current["FeelsLikeC"]) if current.get("FeelsLikeC") not in (None, "") else None,
            "condition": _desc(current),
            "humidity": int(current["humidity"]) if current.get("humidity") not in (None, "") else None,
            "wind_speed_kmh": float(current["windspeedKmph"]) if current.get("windspeedKmph") not in (None, "") else None,
            "wind_gusts_kmh": None,
            "wind_direction_deg": wind_deg,
            "wind_direction_label": _compass_label(wind_deg),
            "pressure_hpa": float(current["pressure"]) if current.get("pressure") not in (None, "") else None,
            "dew_point_c": None,
            "cloud_cover_pct": int(current["cloudcover"]) if current.get("cloudcover") not in (None, "") else None,
            "uv_index": None,
            "wmo_code": None,
            "sunrise": (days[0].get("astronomy") or [{}])[0].get("sunrise") if days else None,
            "sunset": (days[0].get("astronomy") or [{}])[0].get("sunset") if days else None,
            "precipitation_mm": float(current["precipMM"]) if current.get("precipMM") not in (None, "") else None,
            "rain_probability": forecast[0]["rain_probability"] if forecast else None,
            "max_temp_c": forecast[0]["max_temp_c"] if forecast else None,
            "min_temp_c": forecast[0]["min_temp_c"] if forecast else None,
            "forecast": forecast,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    async def _reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """GPS coords -> nearest place name via OSM Nominatim (keyless)."""
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10,  # city/district level
        }
        headers = {"User-Agent": "FarmerVision/1.0 (agricultural advisory app)"}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get(NOMINATIM_REVERSE_URL, params=params)
            resp.raise_for_status()
            data = resp.json() or {}
        address = data.get("address") or {}
        for key in ("city", "town", "village", "municipality", "state_district", "county"):
            if address.get(key):
                return address[key]
        return None

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
                    "wind_gusts_kmh": None,
                    "wind_direction_deg": None,
                    "wind_direction_label": None,
                    "pressure_hpa": None,
                    "dew_point_c": None,
                    "cloud_cover_pct": None,
                    "uv_index": None,
                    "wmo_code": None,
                    "sunrise": None,
                    "sunset": None,
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
                "wind_gusts_kmh": None,
                "wind_direction_deg": None,
                "wind_direction_label": None,
                "pressure_hpa": None,
                "dew_point_c": None,
                "cloud_cover_pct": None,
                "uv_index": None,
                "wmo_code": None,
                "sunrise": astron.get("sunrise"),
                "sunset": astron.get("sunset"),
                "precipitation_mm": current.get("rainfall"),
                "rain_probability": None,
                "max_temp_c": temps.get("max", {}).get("value"),
                "min_temp_c": temps.get("min", {}).get("value"),
                "forecast": forecast,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }


weather_service = WeatherService()