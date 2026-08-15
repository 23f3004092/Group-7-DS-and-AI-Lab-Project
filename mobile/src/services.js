// Pure mappers + fetchers for the mandi prices and weather endpoints.
// The AppShell calls these with its apiUrl / locationInfo; they never touch
// component state directly — callers decide how to apply the results.

// Normalize raw mandi API rows into display-ready cards (one per crop, latest price).
export function rowsFromMandiData(data) {
  const rows = (data.prices || []).map(p => ({
    crop: p.crop,
    tag: p.variety === 'MSP Reference'
      ? 'MSP'
      : (p.variety && p.variety !== '—' ? p.variety : null),
    market: p.market && p.market !== '—' ? p.market : null,
    price: p.modal_price != null
      ? `₹${Math.round(p.modal_price).toLocaleString('en-IN')}/qtl`
      : '—',
    change: p.change_per_quintal != null
      ? `${p.change_per_quintal >= 0 ? '+' : '-'}₹${Math.round(Math.abs(p.change_per_quintal))}`
      : '—',
  }));
  // One row per crop (first = latest price) so every available crop is shown
  const seenCrops = new Set();
  return rows.filter(r => {
    const key = r.crop.toLowerCase();
    if (seenCrops.has(key)) return false;
    seenCrops.add(key);
    return true;
  });
}

// Fetch fresh mandi rows for the selected location; returns rows or null
export async function fetchMandiRows(apiUrl, locationInfo) {
  if (!apiUrl || (!locationInfo.state && !locationInfo.district)) return null;
  try {
    const params = new URLSearchParams({
      state: locationInfo.state || 'Uttar Pradesh',
      district: locationInfo.district || '',
    });
    const res = await fetch(`${apiUrl}/api/mandi/prices?${params.toString()}`);
    if (!res.ok) throw new Error('bad mandi response');
    const data = await res.json();
    if (!data.prices?.length) return null;
    return { rows: rowsFromMandiData(data), source: data.source };
  } catch (e) {
    // Server unreachable or rate-limited -> keep the static MSP fallback list
    return null;
  }
}

// Map raw weather API payload into the app's flat weather object
export function weatherObjectFromApi(data, locationInfo) {
  return {
    temp: Math.round(data.temperature_c),
    condition: data.condition || '—',
    location: data.location || ([locationInfo.district, locationInfo.state].filter(Boolean).join(', ') || 'Uttar Pradesh'),
    rain: data.precipitation_mm != null ? `${data.precipitation_mm} mm` : '—',
    humidity: data.humidity != null ? `${data.humidity}%` : null,
    forecast: data.forecast || [],
    source: data.source,
    feelsLike: data.apparent_temperature_c != null ? Math.round(data.apparent_temperature_c) : null,
    maxTemp: data.max_temp_c != null ? Math.round(data.max_temp_c) : null,
    minTemp: data.min_temp_c != null ? Math.round(data.min_temp_c) : null,
    rainProb: data.rain_probability != null ? Math.round(data.rain_probability) : null,
    windKmh: data.wind_speed_kmh != null ? Math.round(data.wind_speed_kmh) : null,
    windGusts: data.wind_gusts_kmh != null ? Math.round(data.wind_gusts_kmh) : null,
    windDirLabel: data.wind_direction_label || null,
    windDirDeg: data.wind_direction_deg != null ? Math.round(data.wind_direction_deg) : null,
    pressure: data.pressure_hpa != null ? Math.round(data.pressure_hpa) : null,
    dewPoint: data.dew_point_c != null ? Math.round(data.dew_point_c) : null,
    cloud: data.cloud_cover_pct != null ? Math.round(data.cloud_cover_pct) : null,
    uvIndex: data.uv_index != null ? Math.round(data.uv_index) : null,
    wmoCode: data.wmo_code != null ? data.wmo_code : null,
    sunrise: data.sunrise || null,
    sunset: data.sunset || null,
    updatedAt: data.updated_at || null,
  };
}

// Fetch fresh weather snapshot (GPS coords preferred, else district city); returns object or null
export async function fetchWeatherSnapshot(apiUrl, locationInfo) {
  if (!apiUrl) return null;
  try {
    const params = new URLSearchParams();
    if (locationInfo.lat != null && locationInfo.lon != null) {
      params.set('lat', locationInfo.lat);
      params.set('lon', locationInfo.lon);
    } else if (locationInfo.district) {
      params.set('city', locationInfo.district);
    }
    if (!params.toString()) return null;
    const res = await fetch(`${apiUrl}/api/weather/current?${params.toString()}`);
    if (!res.ok) throw new Error('bad weather response');
    const data = await res.json();
    if (data.temperature_c == null) return null;
    return weatherObjectFromApi(data, locationInfo);
  } catch (e) {
    // Server unreachable -> keep the static default weather card
    return null;
  }
}