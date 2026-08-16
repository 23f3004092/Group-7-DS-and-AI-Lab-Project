import Constants from 'expo-constants';

// Default API Server.
// 1) EXPO_PUBLIC_API_URL (set in mobile/.env or EAS secrets) overrides everything —
//    this is the deployed backend URL baked into the APK.
// 2) In dev, auto-detect the machine running the Expo/Metro bundler (its LAN IP)
//    so physical phones & emulators can reach the FastAPI backend.
// 3) Falls back to localhost when no dev server host is available.
const FALLBACK_API_URL = 'http://127.0.0.1:8000';

function resolveDefaultApiUrl() {
  const explicit = (process.env.EXPO_PUBLIC_API_URL || '').trim().replace(/\/+$/, '');
  if (explicit) return explicit;
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