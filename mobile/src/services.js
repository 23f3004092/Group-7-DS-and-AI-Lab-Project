// Pure mappers + fetchers for the mandi prices and weather endpoints.
// The AppShell calls these with its apiUrl / locationInfo; they never touch
// component state directly — callers decide how to apply the results.

// Known Indian locations so chat can fetch weather/mandi for the place
// mentioned in the question instead of the user's saved location.
// [name, state, district]: district goes to the mandi API, city to weather geocoding.
const KNOWN_LOCATIONS = [
  // Metros / major cities
  ['Delhi', 'Delhi', 'Delhi'],
  ['Mumbai', 'Maharashtra', 'Mumbai'],
  ['Kolkata', 'West Bengal', 'Kolkata'],
  ['Chennai', 'Tamil Nadu', 'Chennai'],
  ['Bangalore', 'Karnataka', 'Bengaluru'],
  ['Bengaluru', 'Karnataka', 'Bengaluru'],
  ['Hyderabad', 'Telangana', 'Hyderabad'],
  ['Ahmedabad', 'Gujarat', 'Ahmedabad'],
  ['Pune', 'Maharashtra', 'Pune'],
  ['Nagpur', 'Maharashtra', 'Nagpur'],
  ['Indore', 'Madhya Pradesh', 'Indore'],
  ['Bhopal', 'Madhya Pradesh', 'Bhopal'],
  ['Jaipur', 'Rajasthan', 'Jaipur'],
  ['Patna', 'Bihar', 'Patna'],
  ['Ranchi', 'Jharkhand', 'Ranchi'],
  ['Guwahati', 'Assam', 'Kamrup'],
  ['Chandigarh', 'Chandigarh', 'Chandigarh'],
  ['Amritsar', 'Punjab', 'Amritsar'],
  ['Ludhiana', 'Punjab', 'Ludhiana'],
  ['Jalandhar', 'Punjab', 'Jalandhar'],
  ['Dehradun', 'Uttarakhand', 'Dehradun'],
  ['Haridwar', 'Uttarakhand', 'Haridwar'],
  ['Nainital', 'Uttarakhand', 'Nainital'],
  ['Shimla', 'Himachal Pradesh', 'Shimla'],
  ['Srinagar', 'Jammu and Kashmir', 'Srinagar'],
  ['Jammu', 'Jammu and Kashmir', 'Jammu'],
  ['Surat', 'Gujarat', 'Surat'],
  ['Vadodara', 'Gujarat', 'Vadodara'],
  ['Coimbatore', 'Tamil Nadu', 'Coimbatore'],
  ['Madurai', 'Tamil Nadu', 'Madurai'],
  ['Tiruchirappalli', 'Tamil Nadu', 'Tiruchirappalli'],
  ['Kochi', 'Kerala', 'Ernakulam'],
  ['Thiruvananthapuram', 'Kerala', 'Thiruvananthapuram'],
  ['Vijayawada', 'Andhra Pradesh', 'Krishna'],
  ['Visakhapatnam', 'Andhra Pradesh', 'Visakhapatnam'],
  ['Aurangabad', 'Maharashtra', 'Aurangabad'],
  ['Dhanbad', 'Jharkhand', 'Dhanbad'],
  ['Jamshedpur', 'Jharkhand', 'East Singhbhum'],
  ['Bhubaneswar', 'Odisha', 'Khordha'],
  ['Cuttack', 'Odisha', 'Cuttack'],
  ['Raipur', 'Chhattisgarh', 'Raipur'],
  ['Kota', 'Rajasthan', 'Kota'],
  ['Udaipur', 'Rajasthan', 'Udaipur'],
  ['Jodhpur', 'Rajasthan', 'Jodhpur'],
  ['Gwalior', 'Madhya Pradesh', 'Gwalior'],
  ['Jabalpur', 'Madhya Pradesh', 'Jabalpur'],
  ['Karnal', 'Haryana', 'Karnal'],
  ['Hisar', 'Haryana', 'Hisar'],
  ['Kurukshetra', 'Haryana', 'Kurukshetra'],
  ['Ambala', 'Haryana', 'Ambala'],
  ['Rohtak', 'Haryana', 'Rohtak'],
  ['Panipat', 'Haryana', 'Panipat'],
  ['Faridabad', 'Haryana', 'Faridabad'],
  ['Gurugram', 'Haryana', 'Gurugram'],
  ['Noida', 'Uttar Pradesh', 'Gautam Buddha Nagar'],
  // Uttar Pradesh districts (mandi coverage)
  ['Agra', 'Uttar Pradesh', 'Agra'],
  ['Aligarh', 'Uttar Pradesh', 'Aligarh'],
  ['Ambedkar Nagar', 'Uttar Pradesh', 'Ambedkar Nagar'],
  ['Amethi', 'Uttar Pradesh', 'Amethi'],
  ['Amroha', 'Uttar Pradesh', 'Amroha'],
  ['Auraiya', 'Uttar Pradesh', 'Auraiya'],
  ['Ayodhya', 'Uttar Pradesh', 'Ayodhya'],
  ['Allahabad', 'Uttar Pradesh', 'Prayagraj'],
  ['Prayagraj', 'Uttar Pradesh', 'Prayagraj'],
  ['Azamgarh', 'Uttar Pradesh', 'Azamgarh'],
  ['Bagpat', 'Uttar Pradesh', 'Bagpat'],
  ['Bahraich', 'Uttar Pradesh', 'Bahraich'],
  ['Ballia', 'Uttar Pradesh', 'Ballia'],
  ['Balrampur', 'Uttar Pradesh', 'Balrampur'],
  ['Banda', 'Uttar Pradesh', 'Banda'],
  ['Barabanki', 'Uttar Pradesh', 'Barabanki'],
  ['Bareilly', 'Uttar Pradesh', 'Bareilly'],
  ['Basti', 'Uttar Pradesh', 'Basti'],
  ['Bhadohi', 'Uttar Pradesh', 'Bhadohi'],
  ['Bijnor', 'Uttar Pradesh', 'Bijnor'],
  ['Budaun', 'Uttar Pradesh', 'Budaun'],
  ['Bulandshahr', 'Uttar Pradesh', 'Bulandshahr'],
  ['Chandauli', 'Uttar Pradesh', 'Chandauli'],
  ['Chitrakoot', 'Uttar Pradesh', 'Chitrakoot'],
  ['Deoria', 'Uttar Pradesh', 'Deoria'],
  ['Etah', 'Uttar Pradesh', 'Etah'],
  ['Etawah', 'Uttar Pradesh', 'Etawah'],
  ['Farrukhabad', 'Uttar Pradesh', 'Farrukhabad'],
  ['Fatehpur', 'Uttar Pradesh', 'Fatehpur'],
  ['Firozabad', 'Uttar Pradesh', 'Firozabad'],
  ['Ghaziabad', 'Uttar Pradesh', 'Ghaziabad'],
  ['Ghazipur', 'Uttar Pradesh', 'Ghazipur'],
  ['Gonda', 'Uttar Pradesh', 'Gonda'],
  ['Gorakhpur', 'Uttar Pradesh', 'Gorakhpur'],
  ['Hamirpur', 'Uttar Pradesh', 'Hamirpur'],
  ['Hapur', 'Uttar Pradesh', 'Hapur'],
  ['Hardoi', 'Uttar Pradesh', 'Hardoi'],
  ['Hathras', 'Uttar Pradesh', 'Hathras'],
  ['Jalaun', 'Uttar Pradesh', 'Jalaun'],
  ['Jaunpur', 'Uttar Pradesh', 'Jaunpur'],
  ['Jhansi', 'Uttar Pradesh', 'Jhansi'],
  ['Kannauj', 'Uttar Pradesh', 'Kannauj'],
  ['Kanpur', 'Uttar Pradesh', 'Kanpur Nagar'],
  ['Kasganj', 'Uttar Pradesh', 'Kasganj'],
  ['Kaushambi', 'Uttar Pradesh', 'Kaushambi'],
  ['Khiri', 'Uttar Pradesh', 'Lakhimpur Kheri'],
  ['Kushinagar', 'Uttar Pradesh', 'Kushinagar'],
  ['Lalitpur', 'Uttar Pradesh', 'Lalitpur'],
  ['Lucknow', 'Uttar Pradesh', 'Lucknow'],
  ['Maharajganj', 'Uttar Pradesh', 'Maharajganj'],
  ['Mahoba', 'Uttar Pradesh', 'Mahoba'],
  ['Mainpuri', 'Uttar Pradesh', 'Mainpuri'],
  ['Mathura', 'Uttar Pradesh', 'Mathura'],
  ['Mau', 'Uttar Pradesh', 'Mau'],
  ['Meerut', 'Uttar Pradesh', 'Meerut'],
  ['Mirzapur', 'Uttar Pradesh', 'Mirzapur'],
  ['Moradabad', 'Uttar Pradesh', 'Moradabad'],
  ['Muzaffarnagar', 'Uttar Pradesh', 'Muzaffarnagar'],
  ['Pilibhit', 'Uttar Pradesh', 'Pilibhit'],
  ['Pratapgarh', 'Uttar Pradesh', 'Pratapgarh'],
  ['Raebareli', 'Uttar Pradesh', 'Raebareli'],
  ['Rampur', 'Uttar Pradesh', 'Rampur'],
  ['Saharanpur', 'Uttar Pradesh', 'Saharanpur'],
  ['Sambhal', 'Uttar Pradesh', 'Sambhal'],
  ['Sant Kabir Nagar', 'Uttar Pradesh', 'Sant Kabir Nagar'],
  ['Shahjahanpur', 'Uttar Pradesh', 'Shahjahanpur'],
  ['Shamli', 'Uttar Pradesh', 'Shamli'],
  ['Shravasti', 'Uttar Pradesh', 'Shravasti'],
  ['Siddharthnagar', 'Uttar Pradesh', 'Siddharthnagar'],
  ['Sitapur', 'Uttar Pradesh', 'Sitapur'],
  ['Sonbhadra', 'Uttar Pradesh', 'Sonbhadra'],
  ['Sultanpur', 'Uttar Pradesh', 'Sultanpur'],
  ['Unnao', 'Uttar Pradesh', 'Unnao'],
  ['Varanasi', 'Uttar Pradesh', 'Varanasi'],
];

const LOCATION_RE = KNOWN_LOCATIONS.map(([name]) => {
  const escaped = name.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return { name, re: new RegExp(`\\b${escaped}\\b`) };
});

// Detect a place mentioned in a chat question ("weather in chennai", "mandi
// price at varanasi"). Returns {name, state, district} or null so callers can
// fall back to the user's saved location.
export function locationFromQuery(text) {
  if (!text) return null;
  const lower = text.toLowerCase();
  let best = null;
  for (const { name, re } of LOCATION_RE) {
    if (re.test(lower) && (!best || name.length > best.name.length)) {
      best = name;
    }
  }
  if (!best) return null;
  const entry = KNOWN_LOCATIONS.find(([n]) => n.toLowerCase() === best.toLowerCase());
  return { name: entry[0], state: entry[1], district: entry[2] };
}

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
    // Server unreachable or rate-limited -> no live prices are shown
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

// Yield estimate fact string for the AI chat's live_data. Detects the crop
// (EN/Hinglish/Hindi) and area (hectares/acres) from the farmer's message, then
// asks the local backend for a model-based prediction.
const CROP_ALIASES = {
  wheat: ['wheat', 'gehu', 'गेहूं', 'गेहूँ'],
  rice: ['rice', 'paddy', 'dhan', 'धान', 'चावल'],
  maize: ['maize', 'makka', 'मक्का'],
  mustard: ['mustard', 'sarson', 'सरसों'],
  sugarcane: ['sugarcane', 'ganna', 'गन्ना'],
  potato: ['potato', 'aloo', 'आलू'],
};

export async function fetchYieldFact(userText, apiUrl, locationInfo) {
  if (!apiUrl || !userText) return undefined;
  const text = userText.toLowerCase();
  let crop;
  for (const [slug, words] of Object.entries(CROP_ALIASES)) {
    if (words.some(w => text.includes(w))) { crop = slug; break; }
  }
  const district = (locationInfo?.district || '').toLowerCase();
  if (!crop || !district) return undefined;
  const m = text.match(/(\d+(?:\.\d+)?)\s*(hectare|hectares|ha|acre|acres|एकड़)/i);
  let areaHa = 1;
  if (m) {
    const value = parseFloat(m[1]);
    const unit = (m[2] || '').toLowerCase();
    areaHa = (unit.startsWith('acre') || unit === 'एकड़') ? value * 0.404686 : value;
  }
  try {
    const res = await fetch(`${apiUrl}/api/query/yield`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ crop, district, area_ha: areaHa }),
    });
    if (!res.ok) return undefined;
    const data = await res.json();
    const tHa = data.predicted_yield_t_ha;
    if (tHa == null) return undefined;
    const qPerAcre = tHa * 10 / 2.47105;
    const areaTxt = m ? ` (${areaHa} ha farm)` : '';
    return `Estimated ${crop} yield: ${qPerAcre.toFixed(1)} quintal/acre, ${tHa.toFixed(2)} t/ha${areaTxt}`;
  } catch (e) {
    return undefined;
  }
}