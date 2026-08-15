import Constants from 'expo-constants';

// Default API Server. In dev, point at the machine running the Expo/Metro bundler
// (its LAN IP) so physical phones & emulators can reach the FastAPI backend.
// Falls back to localhost when no dev server host is available (e.g. a built app).
const FALLBACK_API_URL = 'http://127.0.0.1:8000';

function resolveDefaultApiUrl() {
  try {
    const hostUri = Constants.expoConfig?.hostUri || Constants.expoGoConfig?.debuggerHost || '';
    const host = (hostUri || '').split(':')[0];
    if (host) return `http://${host}:8000`;
  } catch (e) {
    // Constants unavailable -> fall through to the localhost default
  }
  return FALLBACK_API_URL;
}

export const DEFAULT_API_URL = resolveDefaultApiUrl();

export const DEFAULT_SETTINGS = {
  theme: 'light',
  accent: 'emerald',
  fontSize: 'medium',
  location: { state: '', district: '', lat: null, lon: null }, // empty -> ask for permission / manual picker on first run
};

export const SETTINGS_KEY = 'farmervision.settings.v2';

// Bottom navigation destinations (icon glyphs switch between outline/filled)
export const TABS = [
  { key: 'home', icon: 'home-outline', iconActive: 'home', labelKey: 'home' },
  { key: 'scanner', icon: 'scan-outline', iconActive: 'scan', labelKey: 'scanner' },
  { key: 'chat', icon: 'chatbubble-ellipses-outline', iconActive: 'chatbubble-ellipses', labelKey: 'chat' },
  { key: 'yield', icon: 'stats-chart-outline', iconActive: 'stats-chart', labelKey: 'yield' },
  { key: 'settings', icon: 'settings-outline', iconActive: 'settings', labelKey: 'settings' },
];