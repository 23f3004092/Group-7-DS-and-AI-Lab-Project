import Ionicons from '@react-native-vector-icons/ionicons';
import MaterialCommunityIcons from '@react-native-vector-icons/material-design-icons';

// Time-of-day greeting used on the Home screen
export const greeting = (t) => {
  const h = new Date().getHours();
  if (h < 12) return t('greetings.morning');
  if (h < 17) return t('greetings.afternoon');
  return t('greetings.evening');
};

// Localized label for scanner/yield crop slugs
export const cropLabel = (t, slug) => {
  const key = { wheat: 'Wheat', rice: 'Paddy' }[slug] || slug;
  return t(`crops.${key}`, { defaultValue: key });
};

// WMO weather code -> vector icon used on the weather card and detail view
export const weatherIcon = (code) => {
  if (code == null) return 'partly-sunny-outline';
  if (code === 0) return 'sunny-outline';
  if (code === 1 || code === 2) return 'partly-sunny-outline';
  if (code === 3 || code === 45 || code === 48) return 'cloudy-outline';
  if (code >= 51 && code <= 57) return 'rainy-outline';
  if (code >= 61 && code <= 67) return 'rainy-outline';
  if (code >= 71 && code <= 77) return 'snow-outline';
  if (code >= 80 && code <= 82) return 'rainy-outline';
  if (code === 85 || code === 86) return 'snow-outline';
  if (code >= 95) return 'thunderstorm-outline';
  return 'partly-sunny-outline';
};

// Render from the right icon family ('windy' lives in MaterialDesignIcons)
export const renderIcon = (name, size, color, extra) =>
  name === 'windy'
    ? <MaterialCommunityIcons name="weather-windy" size={size} color={color} style={extra} />
    : <Ionicons name={name} size={size} color={color} style={extra} />;

export const fmtTime = (iso) => (iso ? String(iso).slice(11, 16) : '—');

export const fmtDay = (dateStr) => {
  if (!dateStr) return '—';
  const d = new Date(String(dateStr) + 'T00:00:00');
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric' }).replace(',', '');
};

// Rule-based field advisory generated from live conditions (farmer + crop guidance)
export const buildAdvisory = (weather) => {
  const tips = [];
  const pct = (v) => (v != null ? Math.round(v) : null);
  if (pct(weather.humidity) >= 70) {
    tips.push({ icon: 'water-outline', key: 'advisoryHumidity', values: { humidity: pct(weather.humidity) } });
  }
  if (pct(weather.rainProb) >= 60) {
    tips.push({ icon: 'umbrella-outline', key: 'advisoryRainChance', values: { prob: pct(weather.rainProb) } });
  }
  if (pct(weather.windKmh) >= 25) {
    tips.push({ icon: 'windy', key: 'advisoryWind', values: { wind: pct(weather.windKmh) } });
  }
  if (pct(weather.uvIndex) >= 7) {
    tips.push({ icon: 'sunny-outline', key: 'advisoryUv', values: { uv: pct(weather.uvIndex) } });
  }
  if (weather.temp >= 35) tips.push({ icon: 'thermometer-outline', key: 'advisoryHeat', values: { temp: weather.temp } });
  if (weather.temp <= 10) tips.push({ icon: 'snow-outline', key: 'advisoryCold', values: { temp: weather.temp } });
  if (pct(weather.dewPoint) >= 20) tips.push({ icon: 'water-outline', key: 'advisoryDew', values: { dew: pct(weather.dewPoint) } });
  if (tips.length === 0) tips.push({ icon: 'leaf-outline', key: 'advisoryFine' });
  return tips.slice(0, 5);
};