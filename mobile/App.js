import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Image,
  ActivityIndicator,
  Alert,
  Modal,
  FlatList,
  Platform,
  RefreshControl,
  KeyboardAvoidingView,
} from 'react-native';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';
import Storage from 'expo-sqlite/kv-store';
import * as Location from 'expo-location';
import * as ImagePicker from 'expo-image-picker';
import Constants from 'expo-constants';
import Ionicons from '@react-native-vector-icons/ionicons';
import MaterialCommunityIcons from '@react-native-vector-icons/material-design-icons';
import { useFonts } from 'expo-font';
import {
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
  Inter_800ExtraBold,
} from '@expo-google-fonts/inter';
import { initI18n, setLanguage, SUPPORTED_LANGUAGES, LANG_NAMES } from './i18n';
import {
  AI_CONFIGURED,
  AI_BASE_URL_VALUE,
  ask as aiAsk,
  classify as aiClassify,
  diagnose as aiDiagnose,
  checkHealth as aiCheckHealth,
  normalizeSources,
  setAiProxyUrl,
} from './aiClient';

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

const DEFAULT_API_URL = resolveDefaultApiUrl();

// ---------- TYPOGRAPHY ----------

const FONT = {
  regular: 'Inter_400Regular',
  medium: 'Inter_500Medium',
  semibold: 'Inter_600SemiBold',
  bold: 'Inter_700Bold',
  extrabold: 'Inter_800ExtraBold',
};

// ---------- THEME SYSTEM ----------

// Accent palettes available for personalization
const ACCENTS = {
  emerald: {
    label: 'Emerald',
    main: '#10b981',
    strong: '#059669',
    soft: 'rgba(16, 185, 129, 0.14)',
    softText: '#059669',
  },
  sky: {
    label: 'Sky',
    main: '#0ea5e9',
    strong: '#0284c7',
    soft: 'rgba(14, 165, 233, 0.14)',
    softText: '#0284c7',
  },
  amber: {
    label: 'Amber',
    main: '#f59e0b',
    strong: '#d97706',
    soft: 'rgba(245, 158, 11, 0.16)',
    softText: '#b45309',
  },
  violet: {
    label: 'Violet',
    main: '#8b5cf6',
    strong: '#7c3aed',
    soft: 'rgba(139, 92, 246, 0.14)',
    softText: '#7c3aed',
  },
  rose: {
    label: 'Rose',
    main: '#f43f5e',
    strong: '#e11d48',
    soft: 'rgba(244, 63, 94, 0.14)',
    softText: '#e11d48',
  },
};

// Base color tokens per theme mode
const BASE_THEMES = {
  light: {
    name: 'Light',
    bg: '#f4f6fa',
    surface: '#ffffff',
    surfaceAlt: '#eef2f8',
    surfaceDeep: '#e6ebf2',
    text: '#0f172a',
    textMuted: '#64748b',
    border: '#e7ecf3',
    inputBg: '#f1f4f9',
    placeholder: '#94a3b8',
    shadow: '#0f172a',
    shadowOpacity: 0.08,
    statusBar: 'dark',
    weatherGradient: ['#10b981', '#047857'],
    danger: '#ef4444',
    dangerBg: '#fef2f2',
    success: '#10b981',
    successBg: '#ecfdf5',
    warning: '#d97706',
    warningBg: '#fffbeb',
  },
  dark: {
    name: 'Dark',
    bg: '#0b0f17',
    surface: '#131b28',
    surfaceAlt: '#1a2333',
    surfaceDeep: '#0d1320',
    text: '#e7edf6',
    textMuted: '#8ea0bb',
    border: '#253049',
    inputBg: '#0e1420',
    placeholder: '#5c6d88',
    shadow: '#000000',
    shadowOpacity: 0.4,
    statusBar: 'light',
    weatherGradient: ['#0f9d8f', '#0b6e66'],
    danger: '#f87171',
    dangerBg: '#2a1416',
    success: '#34d399',
    successBg: '#12241d',
    warning: '#fbbf24',
    warningBg: '#2b2412',
  },
  highContrast: {
    name: 'High Contrast',
    bg: '#000000',
    surface: '#000000',
    surfaceAlt: '#1a1a1a',
    surfaceDeep: '#000000',
    text: '#ffffff',
    textMuted: '#d4d4d4',
    border: '#ffffff',
    inputBg: '#0a0a0a',
    placeholder: '#a3a3a3',
    shadow: '#000000',
    shadowOpacity: 0,
    statusBar: 'light',
    weatherGradient: ['#005a00', '#003300'],
    danger: '#ff4444',
    dangerBg: '#330000',
    success: '#33ff77',
    successBg: '#003311',
    warning: '#ffcc00',
    warningBg: '#332b00',
  },
};

const FONT_SCALES = {
  small: 0.88,
  medium: 1,
  large: 1.14,
};

const DEFAULT_SETTINGS = {
  theme: 'light',
  accent: 'emerald',
  fontSize: 'medium',
  location: { state: '', district: '', lat: null, lon: null }, // empty -> ask for permission / manual picker on first run
};

const SETTINGS_KEY = 'farmervision.settings.v2';

async function loadSettings() {
  try {
    const raw = await Storage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch (e) {
    // Storage unavailable (e.g. web without wasm setup) -> fall back to defaults
  }
  return DEFAULT_SETTINGS;
}

async function saveSettings(settings) {
  try {
    await Storage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (e) {
    // Ignore persistence failures; settings still apply for this session
  }
}

// Bottom navigation destinations (icon glyphs switch between outline/filled)
const TABS = [
  { key: 'home', icon: 'home-outline', iconActive: 'home', labelKey: 'home' },
  { key: 'scanner', icon: 'scan-outline', iconActive: 'scan', labelKey: 'scanner' },
  { key: 'chat', icon: 'chatbubble-ellipses-outline', iconActive: 'chatbubble-ellipses', labelKey: 'chat' },
  { key: 'yield', icon: 'stats-chart-outline', iconActive: 'stats-chart', labelKey: 'yield' },
  { key: 'settings', icon: 'settings-outline', iconActive: 'settings', labelKey: 'settings' },
];

export default function App() {
  return (
    <SafeAreaProvider>
      <AppShell />
    </SafeAreaProvider>
  );
}

function AppShell() {
  const insets = useSafeAreaInsets();
  const { t, i18n } = useTranslation();
  const [i18nReady, setI18nReady] = useState(false);
  const [activeTab, setActiveTab] = useState('home');
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);

  // Custom font loading (waits before first paint; covers all scripts via RN fallback)
  const [fontsLoaded] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
    Inter_800ExtraBold,
  });

  // Personalization state
  const [themeMode, setThemeMode] = useState(DEFAULT_SETTINGS.theme);
  const [accentKey, setAccentKey] = useState(DEFAULT_SETTINGS.accent);
  const [fontSize, setFontSize] = useState(DEFAULT_SETTINGS.fontSize);
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  // Farmer location (drives mandi prices + yield defaults)
  const [locationInfo, setLocationInfo] = useState(DEFAULT_SETTINGS.location);
  const [states, setStates] = useState([]);
  const [districts, setDistricts] = useState([]);
  const [districtsLoading, setDistrictsLoading] = useState(false);
  const [locating, setLocating] = useState(false);
  // Combined location picker (State -> District drill-down)
  const [locModalOpen, setLocModalOpen] = useState(false);
  const [langModalOpen, setLangModalOpen] = useState(false);
  const [locStep, setLocStep] = useState('state'); // 'state' | 'district'
  const [locEditingState, setLocEditingState] = useState('');

  useEffect(() => {
    (async () => {
      const s = await loadSettings();
      setThemeMode(s.theme);
      setAccentKey(s.accent);
      setFontSize(s.fontSize);
      if (s.location) setLocationInfo(s.location);
      setSettingsLoaded(true);
    })();
  }, []);

  const updateSettings = (patch) => {
    if (patch.theme) setThemeMode(patch.theme);
    if (patch.accent) setAccentKey(patch.accent);
    if (patch.fontSize) setFontSize(patch.fontSize);
    saveSettings({
      theme: patch.theme || themeMode,
      accent: patch.accent || accentKey,
      fontSize: patch.fontSize || fontSize,
      location: patch.location || locationInfo,
    });
  };

  const setLocation = (loc) => {
    const next = {
      state: (loc.state || '').trim(),
      district: (loc.district || '').trim(),
      lat: typeof loc.lat === 'number' ? loc.lat : null,
      lon: typeof loc.lon === 'number' ? loc.lon : null,
    };
    // State changed -> drop stale districts from the current picker
    if (next.state.toLowerCase() !== (locationInfo.state || '').toLowerCase()) {
      setDistricts([]);
    }
    setLocationInfo(next);
    saveSettings({
      theme: themeMode,
      accent: accentKey,
      fontSize: fontSize,
      location: next,
    });
  };

  const accent = ACCENTS[accentKey] || ACCENTS.emerald;
  const theme = BASE_THEMES[themeMode] || BASE_THEMES.light;
  const fontScale = FONT_SCALES[fontSize] || 1;

  // Home Screen States
  const [weather, setWeather] = useState({
    temp: 31, condition: 'Sunny', location: 'Uttar Pradesh', rain: '850mm', humidity: null,
    forecast: [], source: 'static',
    feelsLike: null, maxTemp: null, minTemp: null, rainProb: null,
    windKmh: null, windGusts: null, windDirLabel: null, windDirDeg: null,
    pressure: null, dewPoint: null, cloud: null, uvIndex: null, wmoCode: null,
    sunrise: null, sunset: null, updatedAt: null
  });
  const [mandiPrices, setMandiPrices] = useState([
    { crop: 'Wheat', tag: 'MSP', price: '₹2,275/qtl', change: '+₹15' },
    { crop: 'Paddy', tag: 'MSP', price: '₹2,183/qtl', change: '+₹10' },
    { crop: 'Maize', tag: 'MSP', price: '₹2,090/qtl', change: '-₹5' },
    { crop: 'Mustard', tag: 'MSP', price: '₹5,650/qtl', change: '+₹40' }
  ]);
  const [mandiSource, setMandiSource] = useState('static');
  const [refreshing, setRefreshing] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  // Poll mandi + weather every 5 minutes so prices & conditions stay fresh in the field
  useEffect(() => {
    const pollId = setInterval(() => setReloadKey(k => k + 1), 5 * 60 * 1000);
    return () => clearInterval(pollId);
  }, []);

  // Probe the GCP AI service once at startup so Settings shows its real status
  useEffect(() => {
    if (AI_CONFIGURED) checkAiConnection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Web browsers block direct calls to the GCP AI service (no CORS headers on the
  // deployment), so on web route AI traffic through the local backend's /ai proxy.
  // Mobile keeps calling GCP directly.
  useEffect(() => {
    setAiProxyUrl(Platform.OS === 'web' ? `${apiUrl}/ai` : '');
  }, [apiUrl]);

  // --- Data helpers (shared by the poll effects and the chat live_data assembly) ---

  const rowsFromMandiData = (data) => {
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
  };

  // Fetch fresh mandi rows for the selected location; returns rows or null
  const fetchMandiRows = async (apply = true) => {
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
      const rows = rowsFromMandiData(data);
      if (apply) {
        setMandiPrices(rows);
        setMandiSource(data.source);
      }
      return rows;
    } catch (e) {
      // Server unreachable or rate-limited -> keep the static MSP fallback list
      return null;
    }
  };

  const weatherObjectFromApi = (data) => ({
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
  });

  // Fetch fresh weather snapshot (GPS coords preferred, else district city); returns object or null
  const fetchWeatherSnapshot = async (apply = true) => {
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
      const w = weatherObjectFromApi(data);
      if (apply) setWeather(w);
      return w;
    } catch (e) {
      // Server unreachable -> keep the static default weather card
      return null;
    }
  };

  // Assemble /query live_data from what /classify suggested. When the held
  // snapshot isn't live yet, refresh it from the backend first.
  const buildLiveData = async (suggestedExternal) => {
    const liveData = {};
    const want = (key) => !Array.isArray(suggestedExternal) || suggestedExternal.includes(key);

    if (want('mandi_prices')) {
      let rows = mandiSource === 'live' ? mandiPrices : null;
      if (!rows) rows = await fetchMandiRows();
      if (rows?.length) {
        liveData.mandi_prices = rows.map(p => `${p.crop}: ${p.price}`).join('; ');
      }
    }
    if (want('weather')) {
      let w = weather.source === 'live' ? weather : null;
      if (!w) w = await fetchWeatherSnapshot();
      if (w && w.temp != null) {
        liveData.weather = {
          temp_c: w.temp,
          condition: w.condition,
          humidity_pct: w.humidity != null ? parseInt(w.humidity, 10) : null,
          rain_mm: w.rain !== '—' && w.rain != null ? parseFloat(w.rain) : null,
          wind_kmh: w.windKmh,
          max_temp_c: w.maxTemp,
          min_temp_c: w.minTemp,
          forecast_3d: Array.isArray(w.forecast) && w.forecast.length ? w.forecast : undefined,
        };
      }
    }
    return Object.keys(liveData).length ? liveData : undefined;
  };

  // Live weather from backend /api/weather/current (GPS coords preferred, else district city)
  useEffect(() => {
    if (!apiUrl) return;
    let cancelled = false;
    (async () => {
      const w = await fetchWeatherSnapshot(!cancelled);
      if (!cancelled && w) setWeather(w);
    })();
    return () => { cancelled = true; };
  }, [apiUrl, locationInfo, reloadKey]);

  // Live mandi prices from backend /api/mandi/prices (falls back to static MSP list)
  useEffect(() => {
    if (!apiUrl) return;
    let cancelled = false;
    (async () => {
      const rows = await fetchMandiRows(!cancelled);
      if (!cancelled && rows?.length) setMandiPrices(rows);
    })();
    return () => { cancelled = true; };
  }, [apiUrl, locationInfo, reloadKey]);

  // Pull-to-refresh on Home re-triggers the live weather + mandi fetches
  const onRefresh = () => {
    setRefreshing(true);
    setReloadKey(k => k + 1);
    setTimeout(() => setRefreshing(false), 1200);
  };

  // Load the states list from the backend for the location selector
  const loadStates = async () => {
    if (states.length > 0 || !apiUrl) return;
    try {
      const res = await fetch(`${apiUrl}/api/mandi/states`);
      if (res.ok) {
        const data = await res.json();
        if (data.states?.length) setStates(data.states);
      }
    } catch (e) {
      // No backend -> free-text fallback not needed; manual picker will just be empty
    }
  };

  // Load a state's district pick-list (used by the combined state -> district picker)
  const loadDistrictsFor = async (state) => {
    if (!apiUrl) return [];
    setDistrictsLoading(true);
    try {
      const res = await fetch(`${apiUrl}/api/mandi/districts?state=${encodeURIComponent(state)}`);
      if (res.ok) {
        const data = await res.json();
        return data.districts || [];
      }
    } catch (e) {
      // No backend -> empty list
    } finally {
      setDistrictsLoading(false);
    }
    return [];
  };

  // Open the combined location picker (state, then district)
  const openLocationPicker = async (step = 'state') => {
    if (step === 'district' && locationInfo.state) {
      const ds = await loadDistrictsFor(locationInfo.state);
      setDistricts(ds);
      setLocEditingState(locationInfo.state);
      setLocStep('district');
    } else {
      setLocStep('state');
      loadStates();
    }
    setLocModalOpen(true);
  };

  // Pick a state in the combined picker -> advance to district step
  const chooseStateForLoc = async (st) => {
    setLocEditingState(st);
    const ds = await loadDistrictsFor(st);
    setDistricts(ds);
    setLocStep('district');
  };

  // Pick a district in the combined picker -> save the location
  const chooseDistrictForLoc = (d) => {
    setLocation({ state: locEditingState, district: d });
    setLocModalOpen(false);
    setLocStep('state');
  };

  // Auto-detect & cache the farmer's location on first launch (after settings load)
  const autoDetectOnLaunch = async () => {
    if (Platform.OS === 'web') return;
    setLocating(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        // Permission denied -> let the user pick manually
        return;
      }
      const pos = await Location.getCurrentPositionAsync({});
      const [address] = await Location.reverseGeocodeAsync({
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
      });
      const detectedDistrict = address?.district || address?.city || address?.subregion || '';
      const detectedState = address?.region || '';
      if (detectedDistrict && detectedState) {
        setLocation({
          state: detectedState,
          district: detectedDistrict,
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
        });
        showToast(`📍 ${detectedDistrict}, ${detectedState}`);
      }
    } catch (e) {
      // Ignore detection failures; manual picker is always available
    } finally {
      setLocating(false);
    }
  };

  // On first launch, if no location is set yet, ask for permission and auto-detect
  useEffect(() => {
    if (settingsLoaded && !locationInfo.state && !locationInfo.district) {
      autoDetectOnLaunch();
    }
  }, [settingsLoaded]);

  // Detect the farmer's district from device GPS (reverse geocoded)
  const detectLocation = async () => {
    if (Platform.OS === 'web') {
      Alert.alert('Not Available', 'GPS detection works on Android/iOS devices.');
      return;
    }
    setLocating(true);
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Enable location access to auto-detect your district.');
        return;
      }
      const pos = await Location.getCurrentPositionAsync({});
      const [address] = await Location.reverseGeocodeAsync({
        latitude: pos.coords.latitude,
        longitude: pos.coords.longitude,
      });
      const detectedDistrict = address?.district || address?.city || address?.subregion || '';
      const detectedState = address?.region || 'Uttar Pradesh';
      if (!detectedDistrict) {
        Alert.alert('Not Detected', 'Could not determine your district. Please select it manually.');
        return;
      }
      setLocation({
        state: detectedState,
        district: detectedDistrict,
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
      });
      showToast(`Location set: ${detectedDistrict}, ${detectedState}`);
    } catch (e) {
      Alert.alert('Location Failed', 'Could not get your location. Please select your district manually.');
    } finally {
      setLocating(false);
    }
  };

  // Pre-fill the yield estimator's district with the selected location
  useEffect(() => {
    if (locationInfo.district) {
      setYieldForm(f => ({ ...f, district: locationInfo.district.toLowerCase() }));
    }
  }, [locationInfo.district]);

  // Chat/RAG States
  // Toast helper (declared early so offline fallbacks can use it)
  const [toast, setToast] = useState(null);
  const toastTimerRef = useRef(null);
  const showToast = (msg) => {
    setToast(msg);
    clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(null), 2200);
  };

  // Full-screen weather detail view
  const [weatherDetailOpen, setWeatherDetailOpen] = useState(false);

  // Chat/RAG States
  const [chatMessages, setChatMessages] = useState([
    { id: 1, text: 'Hello! I am your FarmerVision advisor. Ask me questions about crops, fertilizer dosage, disease remedies, or policy schemes.', isUser: false }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const chatScrollRef = useRef(null);
  // One GCP AI conversation session per app run (server remembers turn context)
  const chatSessionRef = useRef(`fv-mobile-${Date.now()}`);
  // Optional leaf photo attached to a chat turn (drives the multimodal /diagnose path)
  const [chatImage, setChatImage] = useState(null); // { uri, name }

  // Real device pickers (camera app / gallery app) via expo-image-picker.
  // Shared by chat attach and the leaf scanner; caller supplies the image setter.
  const captureFromCamera = async (setImage) => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permission required', 'Allow camera access to take a leaf photo.');
      return false;
    }
    const result = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.7 });
    if (!result.canceled && result.assets?.length) {
      const a = result.assets[0];
      setImage({ uri: a.uri, name: a.fileName || `leaf_${Date.now()}.jpg` });
      return true;
    }
    return false;
  };

  const pickFromLibrary = async (setImage) => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permission required', 'Allow photo library access to choose a leaf photo.');
      return false;
    }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.7 });
    if (!result.canceled && result.assets?.length) {
      const a = result.assets[0];
      setImage({ uri: a.uri, name: a.fileName || 'leaf.jpg' });
      return true;
    }
    return false;
  };

  const chooseChatPhoto = () => pickFromLibrary(setChatImage);

  const takeChatPhoto = () => captureFromCamera(setChatImage);

  const pickChatImage = () => {
    // react-native-web's Alert supports a single button, so open the library directly there
    if (Platform.OS === 'web') { chooseChatPhoto(); return; }
    Alert.alert(
      'Attach leaf photo',
      'Add a photo so the AI can diagnose the disease alongside your question.',
      [
        { text: 'Camera', onPress: takeChatPhoto },
        { text: 'Photo Library', onPress: chooseChatPhoto },
        { text: 'Cancel', style: 'cancel' },
      ]
    );
  };

  // AI service connection state (Settings card)
  const [aiStatus, setAiStatus] = useState(null); // null | {ok, text}
  const [aiChecking, setAiChecking] = useState(false);

  const checkAiConnection = async () => {
    if (!AI_CONFIGURED) {
      setAiStatus({ ok: false, text: 'Not configured' });
      return;
    }
    setAiChecking(true);
    try {
      const h = await aiCheckHealth();
      setAiStatus({
        ok: h.status === 'ok',
        text: h.status === 'ok' ? `online (${h.points || 0} KB points, ${h.gpu_name || 'no GPU'})` : `offline: ${h.status}`,
      });
    } catch (e) {
      setAiStatus({ ok: false, text: `unreachable: ${e.message}` });
    } finally {
      setAiChecking(false);
    }
  };

  // Leaf Scanner States
  const [selectedImage, setSelectedImage] = useState(null); // Simulated or base64
  const [uploading, setUploading] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState(null);

  // Yield Estimator States
  const [yieldForm, setYieldForm] = useState({ crop: 'wheat', district: 'meerut', area: '2.5' });
  const [yieldResult, setYieldResult] = useState(null);
  const [estimating, setEstimating] = useState(false);

  // Initialize i18n (device-language detection + persisted choice) before first paint
  useEffect(() => {
    (async () => {
      await initI18n();
      setI18nReady(true);
    })();
  }, []);

  // --- ACTIONS ---

  const handleQuickQuestion = (query) => {
    setChatInput(query);
  };

  const handleSendChat = async () => {
    if (!chatInput.trim() && !chatImage) return;

    const userText = chatInput.trim();
    const attachedImage = chatImage;

    setChatMessages(prev => [...prev, { id: Date.now(), text: userText, isUser: true, image: attachedImage || undefined }]);
    setChatInput('');
    setChatImage(null);
    setIsTyping(true);

    try {
      // 1) GCP AI service — grounded RAG with guardrails + multi-turn (API_SPEC.md).
      //    Text only -> /query; photo (± question) -> /diagnose (multimodal).
      if (AI_CONFIGURED) {
        try {
          let data;
          if (attachedImage) {
            data = await aiDiagnose({
              uri: attachedImage.uri,
              name: attachedImage.name,
              question: userText || undefined,
            });
          } else {
            // Classify intent + detect which external data (mandi / weather) the answer needs
            let suggestedExternal = null;
            let aiIntent;
            try {
              const cl = await aiClassify(userText);
              if (cl && !cl.blocked) {
                aiIntent = cl.retrieval_intent || undefined;
                suggestedExternal = cl.suggested_external || [];
              }
            } catch (clErr) {
              console.warn('AI classify failed, answering without suggestion:', clErr);
            }
            const liveData = await buildLiveData(suggestedExternal);
            data = await aiAsk(userText, {
              sessionId: chatSessionRef.current,
              intent: aiIntent,
              liveData,
            });
          }
          const text = data.answer || data.message || 'No answer received from AI service.';
          setChatMessages(prev => [...prev, {
            id: Date.now(),
            text,
            isUser: false,
            sources: normalizeSources(data.sources),
            tier: data.tier,
            score: data.top_score,
          }]);
          return;
        } catch (aiErr) {
          console.warn('AI service failed, falling back to local backend:', aiErr);
        }
      }

      // 2) Fallback: local FastAPI backend proxy
      if (attachedImage) {
        const formData = new FormData();
        formData.append('file', {
          uri: attachedImage.uri,
          name: attachedImage.name,
          type: /\.png$/i.test(attachedImage.name) ? 'image/png' : 'image/jpeg',
        });
        const res = await fetch(`${apiUrl}/api/query/image`, {
          method: 'POST',
          body: formData,
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        if (res.ok) {
          const data = await res.json();
          setChatMessages(prev => [...prev, {
            id: Date.now(),
            text: data.answer || 'Diagnosis received.',
            isUser: false,
            sources: data.sources,
            tier: data.tier,
          }]);
        } else {
          setChatMessages(prev => [...prev, {
            id: Date.now(),
            text: 'Diagnosis Failed: server returned an error.',
            isUser: false,
          }]);
        }
        return;
      }

      const res = await fetch(`${apiUrl}/api/query/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userText })
      });

      if (res.ok) {
        const data = await res.json();
        setChatMessages(prev => [...prev, {
          id: Date.now(),
          text: data.answer,
          isUser: false,
          sources: data.sources,
          tier: data.tier,
          score: data.top_score
        }]);
      } else {
        const errData = await res.json();
        setChatMessages(prev => [...prev, {
          id: Date.now(),
          text: `Error: ${errData.detail || 'Server responded with error'}`,
          isUser: false
        }]);
      }
    } catch (e) {
      setChatMessages(prev => [...prev, {
        id: Date.now(),
        text: 'Network Connection Error. Please verify server URL in Settings.',
        isUser: false
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  // Leaf photo pickers — open the device's camera / gallery app directly
  const pickScannerPhotoFromCamera = () => {
    captureFromCamera((img) => { setSelectedImage(img); setDiagnosisResult(null); });
  };

  const pickScannerPhotoFromLibrary = () => {
    pickFromLibrary((img) => { setSelectedImage(img); setDiagnosisResult(null); });
  };

  const handleDiagnose = async () => {
    if (!selectedImage) return;
    setUploading(true);

    // Prepare FormData
    const formData = new FormData();
    formData.append('file', {
      uri: selectedImage.uri,
      name: selectedImage.name,
      type: 'image/jpeg'
    });

    try {
      // 1) GCP AI service — leaf diagnosis + grounded treatment (API_SPEC.md /diagnose)
      if (AI_CONFIGURED) {
        try {
          const data = await aiDiagnose({
            uri: selectedImage.uri,
            name: selectedImage.name,
            question: 'is ke liye kya karna chahiye',
          });
          const diag = data.diagnosis || {};
          const label = diag.label || '';
          setDiagnosisResult({
            detected_crop: diag.crop || (label.split('__')[0] || 'unknown'),
            detected_disease: label || 'unknown',
            answer: data.answer || `Detected: ${(diag.disease || label).replace(/_/g, ' ')} (${Math.round((diag.confidence || 0) * 100)}% confidence)`,
            confidence: diag.confidence,
            sources: normalizeSources(data.sources),
            tier: data.tier,
          });
          return;
        } catch (aiErr) {
          console.warn('AI diagnose failed, falling back to local backend:', aiErr);
        }
      }

      // 2) Fallback: local FastAPI backend proxy
      const res = await fetch(`${apiUrl}/api/query/image`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (res.ok) {
        const data = await res.json();
        setDiagnosisResult(data);
      } else {
        Alert.alert('Diagnosis Failed', 'Server returned error classification.');
      }
    } catch (e) {
      // Offline fallback simulator if server cannot be reached
      setTimeout(() => {
        const mockResponse = {
          detected_crop: selectedImage.name.includes('wheat') ? 'wheat' : 'rice',
          detected_disease: selectedImage.name.includes('wheat') ? 'wheat__yellow_rust' : 'rice__brown_spot',
          answer: selectedImage.name.includes('wheat')
            ? "Your wheat crop is diagnosed with Yellow Rust (Pila Ratua). Spray Propiconazole 25% EC @ 200ml/acre in 200L water."
            : "Your rice crop has Brown Spot (Bhura Dhabba). Apply Hexaconazole 5% EC @ 2ml/L of water.",
          sources: [
            { rank: 1, score: 0.88, source_type: 'pdf_policy', text: 'Apply propiconazole for wheat rust' }
          ]
        };
        setDiagnosisResult(mockResponse);
        showToast('Running offline mock diagnosis');
      }, 1500);
    } finally {
      setUploading(false);
    }
  };

  const handleEstimateYield = async () => {
    setEstimating(true);
    try {
      const res = await fetch(`${apiUrl}/api/query/yield`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop: yieldForm.crop,
          district: yieldForm.district,
          area_ha: parseFloat(yieldForm.area)
        })
      });

      if (res.ok) {
        const data = await res.json();
        setYieldResult(data);
      } else {
        Alert.alert('Calculation Error', 'Failed to calculate yield.');
      }
    } catch (e) {
      // Local fallback calculation
      setTimeout(() => {
        const baseYield = yieldForm.crop === 'wheat' ? 3.6 : 2.8;
        const pred_t_ha = baseYield * (yieldForm.district === 'meerut' ? 1.2 : 0.7);
        const total_yield = pred_t_ha * parseFloat(yieldForm.area);
        const total_cost = 32000 * parseFloat(yieldForm.area);
        const total_rev = total_yield * 10 * 2275;

        setYieldResult({
          predicted_yield_t_ha: pred_t_ha,
          total_yield_t: total_yield,
          economics: {
            total_cost: total_cost,
            total_revenue: total_rev,
            net_profit: total_rev - total_cost,
            roi_percent: Math.round(((total_rev - total_cost) / total_cost) * 100)
          }
        });
      }, 1000);
    } finally {
      setEstimating(false);
    }
  };

  const handleResetSettings = () => {
    setThemeMode(DEFAULT_SETTINGS.theme);
    setAccentKey(DEFAULT_SETTINGS.accent);
    setFontSize(DEFAULT_SETTINGS.fontSize);
    saveSettings(DEFAULT_SETTINGS);
  };

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return t('greetings.morning');
    if (h < 17) return t('greetings.afternoon');
    return t('greetings.evening');
  };

  const cropLabel = (slug) => {
    const key = { wheat: 'Wheat', rice: 'Paddy' }[slug] || slug;
    return t(`crops.${key}`, { defaultValue: key });
  };

  // WMO weather code -> vector icon used on the weather card and detail view
  const weatherIcon = (code) => {
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
  const renderIcon = (name, size, color, extra) =>
    name === 'windy'
      ? <MaterialCommunityIcons name="weather-windy" size={size} color={color} style={extra} />
      : <Ionicons name={name} size={size} color={color} style={extra} />;

  // Rule-based field advisory generated from live conditions (farmer + crop guidance)
  const buildAdvisory = () => {
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

  const stat = (labelKey, icon, valueRaw, unit) => (
    <View style={[s.statItem, { backgroundColor: theme.surfaceAlt, borderColor: theme.border }]}>
      <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
        {renderIcon(icon, 15, accent.softText, { marginRight: 6 })}
        <Text style={[s.statLabel, { color: theme.textMuted }]}>{t(labelKey)}</Text>
      </View>
      <Text style={[s.statValue, { color: theme.text }]}>{valueRaw != null ? `${valueRaw}${unit || ''}` : '—'}</Text>
    </View>
  );

  const fmtTime = (iso) => (iso ? String(iso).slice(11, 16) : '—');

  const fmtDay = (dateStr) => {
    if (!dateStr) return '—';
    const d = new Date(String(dateStr) + 'T00:00:00');
    return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric' }).replace(',', '');
  };

  const buildStatGrid = () => [
    stat('humidityLabel', 'water-outline', weather.humidity, ''),
    stat('rainfallLabel', 'umbrella-outline', weather.rain, ''),
    stat('windLabel', 'windy', weather.windKmh, weather.windKmh != null ? ' km/h' : ''),
    stat('windDirLabel', 'navigate-outline', weather.windDirLabel, weather.windKmh != null ? ` (${weather.windDirDeg}°)` : ''),
    stat('uvLabel', 'sunny-outline', weather.uvIndex, weather.uvIndex != null ? '' : ''),
    stat('pressureLabel', 'speedometer-outline', weather.pressure, weather.pressure != null ? ' hPa' : ''),
    stat('dewPointLabel', 'thermometer-outline', weather.dewPoint, weather.dewPoint != null ? '°C' : ''),
    stat('cloudLabel', 'cloud-outline', weather.cloud, weather.cloud != null ? '%' : ''),
  ];

  const s = createStyles(theme, accent, fontScale, themeMode);

  // Wait for i18n + fonts (cross-platform). Non-Latin scripts still render via system fallback.
  if (!i18nReady || !fontsLoaded) {
    return (
      <View style={[s.container, { alignItems: 'center', justifyContent: 'center' }]}>
        <ActivityIndicator size="large" color={accent.main} />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={[s.container, { backgroundColor: theme.bg }]}
      behavior={Platform.OS === 'ios' ? 'padding' : Platform.OS === 'android' ? 'height' : undefined}
    >
      <StatusBar style={theme.statusBar} />

      {/* Header bar (padded below the clock/battery status bar area) */}
      <LinearGradient
        colors={[theme.weatherGradient[0], theme.weatherGradient[1]]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={[s.header, { paddingTop: insets.top + 12 }]}
      >
        <View>
          <Text style={s.headerTitle}>{t('appName')}</Text>
          <Text style={s.headerSubtitle}>{t('tagline')}</Text>
        </View>
        <TouchableOpacity
          style={s.langPill}
          activeOpacity={0.7}
          onPress={() => setLangModalOpen(true)}
        >
          <Ionicons name="globe-outline" size={14} color="#fff" />
          <Text style={s.langText}>{LANG_NAMES[i18n.language] || 'English'}</Text>
          <Ionicons name="chevron-down" size={13} color="rgba(255,255,255,0.85)" />
        </TouchableOpacity>
      </LinearGradient>

      {/* Main Tab Screens Scroll area */}
      <View style={s.contentContainer}>
        {/* --- SCREEN 1: HOME --- */}
        {activeTab === 'home' && (
          <ScrollView
            contentContainerStyle={s.scrollContent}
            showsVerticalScrollIndicator={false}
            refreshControl={
              <RefreshControl
                refreshing={refreshing}
                onRefresh={onRefresh}
                tintColor={accent.main}
                colors={[accent.main]}
                progressBackgroundColor={theme.surface}
              />
            }
          >
            {/* Greeting */}
            <Text style={s.greeting}>{greeting()}</Text>

            {/* Weather & Advisory widget (tap for full weather detail view) */}
            <TouchableOpacity
              activeOpacity={0.92}
              onPress={() => setWeatherDetailOpen(true)}
            >
              <LinearGradient
                colors={theme.weatherGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={[s.card, s.weatherCard]}
              >
                <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                  <View style={{ flex: 1, paddingRight: 10 }}>
                    <Text style={s.weatherTemp}>{weather.temp}°C</Text>
                    <Text style={s.weatherDesc} numberOfLines={2}>{weather.condition} • {weather.location}</Text>
                  </View>
                  <View style={{ alignItems: 'flex-end' }}>
                    {weather.source === 'live' && (
                      <View style={[s.liveBadge, { backgroundColor: 'rgba(255,255,255,0.2)', marginLeft: 0, marginBottom: 6 }]}>
                        <Text style={[s.liveBadgeText, { color: '#fff' }]}>● Live</Text>
                      </View>
                    )}
                    <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                      <Ionicons name="water-outline" size={12} color="rgba(255,255,255,0.85)" />
                      <Text style={[s.weatherLabel, { marginLeft: 4 }]}>Rainfall</Text>
                    </View>
                    <Text style={s.weatherVal}>{weather.rain}</Text>
                    {weather.humidity && (
                      <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 4 }}>
                        <Ionicons name="water-outline" size={12} color="rgba(255,255,255,0.9)" />
                        <Text style={[s.weatherLabel, { marginLeft: 4 }]}>{weather.humidity}</Text>
                      </View>
                    )}
                  </View>
                </View>

                {/* Live readout chips */}
                <View style={s.weatherChipRow}>
                  {weather.windKmh != null && (
                    <View style={s.weatherChip}>
                      <MaterialCommunityIcons name="weather-windy" size={13} color="#fff" style={{ marginRight: 4 }} />
                      <Text style={s.weatherChipText}>{weather.windKmh} km/h</Text>
                    </View>
                  )}
                  {weather.rainProb != null && (
                    <View style={s.weatherChip}>
                      <Ionicons name="umbrella-outline" size={12} color="#fff" style={{ marginRight: 4 }} />
                      <Text style={s.weatherChipText}>{weather.rainProb}% rain</Text>
                    </View>
                  )}
                  {weather.dewPoint != null && (
                    <View style={s.weatherChip}>
                      <Ionicons name="thermometer-outline" size={12} color="#fff" style={{ marginRight: 4 }} />
                      <Text style={s.weatherChipText}>Dew {weather.dewPoint}°C</Text>
                    </View>
                  )}
                  <View style={[s.weatherChip, { marginLeft: 'auto' }]}>
                    <Text style={s.weatherChipText}>Details</Text>
                    <Ionicons name="chevron-forward" size={12} color="#fff" style={{ marginLeft: 3 }} />
                  </View>
                </View>

                {/* 3-day forecast strip */}
                {weather.forecast?.length > 0 && (
                  <View style={s.weatherForecastRow}>
                    {weather.forecast.map((f, i) => (
                      <View key={i} style={s.weatherForecastChip}>
                        <Text style={s.weatherForecastDay}>{fmtDay(f.date)}</Text>
                        <Ionicons name={weatherIcon(f.wmo_code)} size={14} color="#fff" />
                        <Text style={s.weatherForecastTemp}>
                          ↑{f.max_temp_c != null ? Math.round(f.max_temp_c) : '—'}°
                        </Text>
                        <Text style={s.weatherForecastTemp}>
                          ↓{f.min_temp_c != null ? Math.round(f.min_temp_c) : '—'}°
                        </Text>
                      </View>
                    ))}
                  </View>
                )}
                <View style={{ flexDirection: 'row', alignItems: 'flex-start', marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.2)' }}>
                  <Ionicons name="leaf-outline" size={14} color="#fff" style={{ marginTop: 2, marginRight: 6 }} />
                  <Text style={s.advisoryBanner}>
                    🌾 Advisory: Ideal conditions for Rabi crop fertilization. Monitor wheat leaves for rust flags.
                  </Text>
                </View>
              </LinearGradient>
            </TouchableOpacity>

            {/* Mandi Prices (responsive 3-column grid, page scrolls for more) */}
            <View style={s.sectionHeader}>
              <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
              <Text style={s.sectionTitle}>{t('mandi')}</Text>
              {mandiSource === 'live' && (
                <View style={[s.liveBadge, { backgroundColor: accent.soft }]}>
                  <Text style={[s.liveBadgeText, { color: accent.softText }]}>● Live</Text>
                </View>
              )}
            </View>
            <TouchableOpacity
              style={[s.mandiLocationRow, { backgroundColor: theme.surfaceAlt, borderColor: theme.border }]}
              onPress={() => openLocationPicker('state')}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center', flex: 1 }}>
                <Ionicons name="location-outline" size={15} color={accent.softText} />
                <Text style={[s.mandiLocation, { color: theme.text }]}>
                  {[locationInfo.district, locationInfo.state].filter(Boolean).join(', ') || t('setLocation')}
                </Text>
              </View>
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="create-outline" size={13} color={accent.softText} />
                <Text style={[s.mandiLocationEdit, { color: accent.softText }]}>{t('changeLocation')}</Text>
              </View>
            </TouchableOpacity>
            <View style={[s.card, { paddingBottom: 12 }]}>
              {mandiPrices.length > 0 ? (
                <View>
                  {/* 3x3 grid viewport: exactly 9 cards visible; internal scroll for the rest */}
                  <ScrollView
                    style={{ maxHeight: 3 * 108 + 2 * 10 }}
                    showsVerticalScrollIndicator={false}
                    nestedScrollEnabled
                  >
                    <View style={s.mandiGrid}>
                      {mandiPrices.map((item, idx) => (
                        <View
                          key={idx}
                          style={[s.mandiCard, { backgroundColor: theme.surfaceAlt, borderColor: theme.border }]}
                        >
                          <Text style={s.mandiCrop} numberOfLines={2}>
                            {t(`crops.${item.crop}`, { defaultValue: item.crop })}
                            {item.tag ? ` (${item.tag})` : ''}
                          </Text>
                          {item.market && (
                            <Text style={[s.mandiMarket, { color: theme.textMuted }]} numberOfLines={1}>{item.market}</Text>
                          )}
                          <Text style={s.mandiPrice} numberOfLines={1}>{item.price}</Text>
                          <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 'auto' }}>
                            {item.change !== '—' && (
                              <Ionicons
                                name={item.change.startsWith('-') ? 'trending-down-outline' : 'trending-up-outline'}
                                size={12}
                                color={item.change.startsWith('-') ? theme.danger : theme.success}
                              />
                            )}
                            <Text style={[s.mandiChange, { color: item.change === '—' ? theme.textMuted : (item.change.startsWith('-') ? theme.danger : theme.success) }]}>
                              {item.change}
                            </Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  </ScrollView>
                  {mandiPrices.length > 9 && (
                    <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginTop: 8 }}>
                      <Ionicons name="chevron-down" size={12} color={theme.textMuted} />
                      <Text style={{ color: theme.textMuted, fontSize: 11 * fontScale, fontFamily: FONT.medium, marginLeft: 4 }}>
                        {t('scrollForMore')} ({mandiPrices.length} crops)
                      </Text>
                    </View>
                  )}
                </View>
              ) : (
                <Text style={{ color: theme.textMuted, fontSize: 13 * fontScale, fontFamily: FONT.medium }}>
                  {t('noPricesHint')}
                </Text>
              )}
            </View>

            {/* Quick Actions Shortcuts */}
            <View style={s.sectionHeader}>
              <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
              <Text style={s.sectionTitle}>Advisory Pipelines</Text>
            </View>
            <View style={s.shortcutsGrid}>
              <TouchableOpacity
                style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
                onPress={() => setActiveTab('scanner')}
              >
                <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
                  <Ionicons name="scan-outline" size={24} color={accent.softText} />
                </View>
                <Text style={s.shortcutText}>{t('scanner')}</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
                onPress={() => setActiveTab('chat')}
              >
                <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
                  <Ionicons name="chatbubble-ellipses-outline" size={24} color={accent.softText} />
                </View>
                <Text style={s.shortcutText}>{t('chat')}</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
                onPress={() => setActiveTab('yield')}
              >
                <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
                  <Ionicons name="stats-chart-outline" size={24} color={accent.softText} />
                </View>
                <Text style={s.shortcutText}>{t('yield')}</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        )}

        {/* --- SCREEN 2: LEAF SCANNER --- */}
        {activeTab === 'scanner' && (
          <ScrollView contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>
            <View style={s.sectionHeader}>
              <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
              <Text style={s.sectionTitle}>{t('diagnose')}</Text>
            </View>

            <View style={[s.photoSelectorContainer, { borderColor: theme.border, backgroundColor: theme.surface }]}>
              {selectedImage ? (
                <Image source={{ uri: selectedImage.uri }} style={s.selectedLeafImage} />
              ) : (
                <View style={s.photoPlaceholder}>
                  <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft, width: 84, height: 84, borderRadius: 42 }]}>
                    <Ionicons name="leaf-outline" size={40} color={accent.softText} />
                  </View>
                  <Text style={[s.photoHint, { color: theme.textMuted }]}>Upload a photo of crop leaf</Text>
                </View>
              )}

              <View style={s.photoActionsRow}>
                <TouchableOpacity
                  style={[s.photoBtn, { backgroundColor: theme.surfaceAlt, borderColor: theme.border, borderWidth: themeMode === 'highContrast' ? 2 : 1 }]}
                  onPress={pickScannerPhotoFromCamera}
                >
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="camera-outline" size={15} color={accent.softText} style={{ marginRight: 6 }} />
                    <Text style={[s.photoBtnText, { color: theme.text }]}>{t('takePhoto')}</Text>
                  </View>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[s.photoBtn, { backgroundColor: theme.surfaceAlt, borderColor: theme.border, borderWidth: themeMode === 'highContrast' ? 2 : 1 }]}
                  onPress={pickScannerPhotoFromLibrary}
                >
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="images-outline" size={15} color={accent.softText} style={{ marginRight: 6 }} />
                    <Text style={[s.photoBtnText, { color: theme.text }]}>{t('choosePhoto')}</Text>
                  </View>
                </TouchableOpacity>
              </View>
            </View>

            {selectedImage && (
              <TouchableOpacity
                style={[s.actionBtn, uploading ? { opacity: 0.7 } : {}]}
                onPress={handleDiagnose}
                disabled={uploading}
              >
                <LinearGradient
                  colors={[accent.main, accent.strong]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={s.actionBtnGradient}
                >
                  {uploading ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <Ionicons name="scan" size={18} color="#fff" style={{ marginRight: 8 }} />
                      <Text style={s.actionBtnText}>{t('diagnoseBtn')}</Text>
                    </View>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            )}

            {/* Diagnosis results card */}
            {diagnosisResult && (
              <View style={s.card}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
                  <Ionicons name="analytics-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                  <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 0, paddingBottom: 0 }]}>{t('diagResult')}</Text>
                </View>

                {/* Crop & Disease labels */}
                <View style={s.resultBadgeRow}>
                  <View style={[s.badge, { backgroundColor: accent.soft }]}>
                    <Text style={[s.badgeText, { color: accent.softText }]}>{cropLabel(diagnosisResult.detected_crop).toUpperCase()}</Text>
                  </View>
                  <View style={[s.badge, { backgroundColor: accent.soft }]}>
                    <Text style={[s.badgeText, { color: accent.softText }]}>
                      {diagnosisResult.detected_disease?.split('__')[1]?.replace('_', ' ').toUpperCase()}
                    </Text>
                  </View>
                </View>

                {/* Warning for chemical dosage safety */}
                {diagnosisResult.answer?.includes('⚠') && (
                  <View style={[s.alertCard, { backgroundColor: theme.dangerBg, borderColor: theme.danger }]}>
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <Ionicons name="warning-outline" size={16} color={theme.danger} style={{ marginRight: 6 }} />
                      <Text style={[s.alertText, { color: theme.danger }]}>{t('alertBanned')}</Text>
                    </View>
                  </View>
                )}

                <Text style={[s.answerText, { color: theme.text }]}>{diagnosisResult.answer}</Text>
              </View>
            )}
          </ScrollView>
        )}

        {/* --- SCREEN 3: ADVISOR CHAT --- */}
        {activeTab === 'chat' && (
          <View style={{ flex: 1, backgroundColor: theme.bg }}>
            {/* Conversation Area */}
            <ScrollView
              contentContainerStyle={{ padding: 15, paddingBottom: 8 }}
              ref={chatScrollRef}
              onContentSizeChange={() => chatScrollRef.current?.scrollToEnd({ animated: true })}
              showsVerticalScrollIndicator={false}
            >
              {chatMessages.map((msg) => (
                <View
                  key={msg.id}
                  style={[s.chatRow, msg.isUser ? { justifyContent: 'flex-end' } : { justifyContent: 'flex-start' }]}
                >
                  {!msg.isUser && (
                    <LinearGradient
                      colors={[accent.main, accent.strong]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={s.botAvatar}
                    >
                      <Ionicons name="leaf" size={15} color="#fff" />
                    </LinearGradient>
                  )}
                  <View
                    style={[
                      s.chatBubble,
                      msg.isUser
                        ? [s.userBubble, { backgroundColor: accent.main }]
                        : [s.botBubble, { backgroundColor: theme.surface, borderColor: theme.border }],
                    ]}
                  >
                    {msg.image && (
                      <Image
                        source={{ uri: msg.image.uri }}
                        style={[s.chatImageThumb, { backgroundColor: theme.surfaceAlt }]}
                        resizeMode="cover"
                      />
                    )}
                    {msg.text ? (
                      <Text style={[s.chatText, { color: msg.isUser ? '#fff' : theme.text }]}>
                        {msg.text}
                      </Text>
                    ) : null}

                    {/* Citation chips */}
                    {!msg.isUser && msg.sources?.length > 0 && (
                      <View style={[s.citationRow, { borderTopColor: theme.border }]}>
                        {msg.sources.map((src, i) => (
                          <TouchableOpacity
                            key={i}
                            style={[s.citationChip, { backgroundColor: theme.surfaceAlt }]}
                            onPress={() => setSelectedCitation(src)}
                          >
                            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                              <Ionicons name="document-text-outline" size={11} color={accent.softText} style={{ marginRight: 4 }} />
                              <Text style={[s.citationText, { color: theme.textMuted }]}>[{src.rank}] {src.source_type || 'docs'}</Text>
                            </View>
                          </TouchableOpacity>
                        ))}
                      </View>
                    )}
                  </View>
                </View>
              ))}

              {isTyping && (
                <View style={[s.chatRow, { justifyContent: 'flex-start' }]}>
                  <LinearGradient
                    colors={[accent.main, accent.strong]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={s.botAvatar}
                  >
                    <Ionicons name="leaf" size={15} color="#fff" />
                  </LinearGradient>
                  <View style={[s.chatBubble, s.botBubble, { width: 64, backgroundColor: theme.surface, borderColor: theme.border }]}>
                    <ActivityIndicator size="small" color={accent.main} />
                  </View>
                </View>
              )}
            </ScrollView>

            {/* Quick action query helpers */}
            <View style={[s.quickBar, { backgroundColor: theme.surfaceAlt, borderTopColor: theme.border }]}>
              <Text style={[s.quickBarLabel, { color: theme.textMuted }]}>{t('askQuick')}</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => handleQuickQuestion(t('rustHelp'))}>
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="leaf-outline" size={13} color={accent.softText} style={{ marginRight: 5 }} />
                    <Text style={[s.quickChipText, { color: accent.softText }]}>{t('rustHelp')}</Text>
                  </View>
                </TouchableOpacity>
                <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => handleQuickQuestion(t('pmkisan'))}>
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="document-text-outline" size={13} color={accent.softText} style={{ marginRight: 5 }} />
                    <Text style={[s.quickChipText, { color: accent.softText }]}>{t('pmkisan')}</Text>
                  </View>
                </TouchableOpacity>
                <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => handleQuickQuestion(t('ureadose'))}>
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="flask-outline" size={13} color={accent.softText} style={{ marginRight: 5 }} />
                    <Text style={[s.quickChipText, { color: accent.softText }]}>{t('ureadose')}</Text>
                  </View>
                </TouchableOpacity>
              </ScrollView>
            </View>

            {/* Chat Input panel */}
            {chatImage && (
              <View style={[s.chatAttachRow, { backgroundColor: theme.surfaceAlt, borderTopColor: theme.border }]}>
                <Image source={{ uri: chatImage.uri }} style={s.chatAttachThumb} resizeMode="cover" />
                <Text numberOfLines={1} style={[s.chatAttachName, { color: theme.textMuted }]}>{chatImage.name}</Text>
                <TouchableOpacity onPress={() => setChatImage(null)} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
                  <Ionicons name="close-circle" size={20} color={theme.textMuted} />
                </TouchableOpacity>
              </View>
            )}
            <View style={[s.inputBar, { backgroundColor: theme.surface, borderTopColor: theme.border }]}>
              <TouchableOpacity
                style={[s.attachBtn, { backgroundColor: chatImage ? accent.soft : 'transparent' }]}
                onPress={pickChatImage}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              >
                <Ionicons name="image-outline" size={21} color={chatImage ? accent.softText : theme.textMuted} />
              </TouchableOpacity>
              <TextInput
                style={[s.chatTextInput, { backgroundColor: theme.inputBg, color: theme.text }]}
                placeholder={chatImage ? 'Ask about this leaf (optional)...' : "Ask about fertilizer, pesticides, crop diseases..."}
                placeholderTextColor={theme.placeholder}
                value={chatInput}
                onChangeText={setChatInput}
              />
              <TouchableOpacity style={s.sendBtn} onPress={handleSendChat}>
                <LinearGradient
                  colors={[accent.main, accent.strong]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={s.sendBtnGradient}
                >
                  <Ionicons name="send" size={18} color="#fff" />
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* --- SCREEN 4: YIELD ESTIMATOR --- */}
        {activeTab === 'yield' && (
          <ScrollView contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>
            <View style={s.sectionHeader}>
              <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
              <Text style={s.sectionTitle}>{t('calculateYield')}</Text>
            </View>

            <View style={s.card}>
              {/* Form Input fields */}
              <View style={s.formRow}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
                  <Ionicons name="leaf-outline" size={14} color={accent.softText} style={{ marginRight: 6 }} />
                  <Text style={s.formLabel}>{t('cropLabel')}</Text>
                </View>
                <TextInput
                  style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                  value={yieldForm.crop}
                  onChangeText={(val) => setYieldForm({ ...yieldForm, crop: val })}
                  placeholder="wheat / rice"
                  placeholderTextColor={theme.placeholder}
                />
              </View>

              <View style={s.formRow}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
                  <Ionicons name="location-outline" size={14} color={accent.softText} style={{ marginRight: 6 }} />
                  <Text style={s.formLabel}>{t('districtLabel')}</Text>
                </View>
                <TextInput
                  style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                  value={yieldForm.district}
                  onChangeText={(val) => setYieldForm({ ...yieldForm, district: val })}
                  placeholder="meerut / jhansi"
                  placeholderTextColor={theme.placeholder}
                />
              </View>

              <View style={s.formRow}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
                  <Ionicons name="resize-outline" size={14} color={accent.softText} style={{ marginRight: 6 }} />
                  <Text style={s.formLabel}>{t('areaLabel')}</Text>
                </View>
                <TextInput
                  style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                  value={yieldForm.area}
                  onChangeText={(val) => setYieldForm({ ...yieldForm, area: val })}
                  keyboardType="numeric"
                  placeholder="e.g. 2.5"
                  placeholderTextColor={theme.placeholder}
                />
              </View>

              <TouchableOpacity
                style={[s.actionBtn, estimating ? { opacity: 0.7 } : {}]}
                onPress={handleEstimateYield}
                disabled={estimating}
              >
                <LinearGradient
                  colors={[accent.main, accent.strong]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={s.actionBtnGradient}
                >
                  {estimating ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <Ionicons name="calculator-outline" size={18} color="#fff" style={{ marginRight: 8 }} />
                      <Text style={s.actionBtnText}>{t('calculateYield')}</Text>
                    </View>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            </View>

            {/* Projections Card */}
            {yieldResult && (
              <View style={s.card}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
                  <Ionicons name="trending-up-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                  <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 0, paddingBottom: 0 }]}>Yield Projections</Text>
                </View>

                <View style={[s.yieldGaugeContainer, { borderBottomColor: theme.border }]}>
                  <Text style={[s.yieldValText, { color: accent.main }]}>{yieldResult.total_yield_t?.toFixed(2)} t</Text>
                  <Text style={{ color: theme.textMuted, fontSize: 12 * fontScale, fontFamily: FONT.medium }}>
                    Est. Yield ({yieldResult.predicted_yield_t_ha?.toFixed(2)} tonnes/hectare)
                  </Text>
                </View>

                {yieldResult.economics && (
                  <View style={s.economicsGrid}>
                    <View style={[s.econItem, { backgroundColor: theme.inputBg, borderColor: theme.border }]}>
                      <Text style={[s.econLabel, { color: theme.textMuted }]}>{t('cost')}</Text>
                      <Text style={[s.econVal, { color: theme.danger }]}>₹{yieldResult.economics.total_cost.toLocaleString()}</Text>
                    </View>
                    <View style={[s.econItem, { backgroundColor: theme.inputBg, borderColor: theme.border }]}>
                      <Text style={[s.econLabel, { color: theme.textMuted }]}>{t('revenue')}</Text>
                      <Text style={[s.econVal, { color: theme.success }]}>₹{yieldResult.economics.total_revenue.toLocaleString()}</Text>
                    </View>
                    <View style={[s.econItem, { backgroundColor: theme.inputBg, borderColor: theme.border }]}>
                      <Text style={[s.econLabel, { color: theme.textMuted }]}>{t('netProfit')}</Text>
                      <Text style={[s.econVal, { color: accent.main, fontWeight: 'bold' }]}>
                        ₹{yieldResult.economics.net_profit.toLocaleString()}
                      </Text>
                    </View>
                    <View style={[s.econItem, { backgroundColor: theme.inputBg, borderColor: theme.border }]}>
                      <Text style={[s.econLabel, { color: theme.textMuted }]}>{t('roi')}</Text>
                      <Text style={[s.econVal, { color: theme.warning, fontWeight: 'bold' }]}>
                        {yieldResult.economics.roi_percent}%
                      </Text>
                    </View>
                  </View>
                )}
              </View>
            )}
          </ScrollView>
        )}

        {/* --- SCREEN 5: SETTINGS --- */}
        {activeTab === 'settings' && (
          <ScrollView contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>
            <View style={s.sectionHeader}>
              <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
              <Text style={s.sectionTitle}>{t('settings')}</Text>
            </View>

            {/* Appearance / Theme picker */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="contrast-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>{t('appearance')}</Text>
              </View>
              <Text style={[s.settingHint, { color: theme.textMuted }]}>{t('themeHint')}</Text>
              <Text style={[s.formLabel, { color: theme.textMuted, marginTop: 10 }]}>{t('themeMode')}</Text>
              <View style={s.themeRow}>
                {Object.values(BASE_THEMES).map((th) => {
                  const active = themeMode === Object.keys(BASE_THEMES).find(k => BASE_THEMES[k] === th);
                  return (
                    <TouchableOpacity
                      key={th.name}
                      style={[
                        s.themeChip,
                        { backgroundColor: theme.surfaceAlt, borderColor: theme.border },
                        active && { borderColor: accent.main, backgroundColor: accent.soft }
                      ]}
                      onPress={() => updateSettings({ theme: Object.keys(BASE_THEMES).find(k => BASE_THEMES[k] === th) })}
                    >
                      <View style={[s.themePreview, { backgroundColor: th.bg, borderColor: th.border }]}>
                        <View style={[s.themePreviewBar, { backgroundColor: th.surface }]}>
                          <View style={[s.themePreviewDot, { backgroundColor: th.text }]} />
                        </View>
                        <View style={[s.themePreviewLine, { backgroundColor: th.surface }]} />
                      </View>
                      <Text style={[s.themeChipText, { color: active ? accent.softText : theme.text }]}>{th.name}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>

            {/* Personalization: accent color */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="color-palette-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>{t('accentColor')}</Text>
              </View>
              <Text style={[s.settingHint, { color: theme.textMuted }]}>{t('accentHint')}</Text>
              <View style={s.accentRow}>
                {Object.entries(ACCENTS).map(([key, a]) => {
                  const active = accentKey === key;
                  return (
                    <TouchableOpacity
                      key={key}
                      style={[
                        s.swatch,
                        { backgroundColor: a.main },
                        active && { borderWidth: 3, borderColor: theme.text }
                      ]}
                      onPress={() => updateSettings({ accent: key })}
                    >
                      {active && <Ionicons name="checkmark" size={16} color="#fff" />}
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>

            {/* Personalization: font size */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="text-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>{t('fontSize')}</Text>
              </View>
              <Text style={[s.settingHint, { color: theme.textMuted }]}>{t('fontHint')}</Text>
              <View style={s.fontSizeRow}>
                {Object.keys(FONT_SCALES).map((key) => {
                  const active = fontSize === key;
                  const preview = { small: 11, medium: 14, large: 17 }[key];
                  return (
                    <TouchableOpacity
                      key={key}
                      style={[
                        s.fontSizeChip,
                        { backgroundColor: theme.surfaceAlt, borderColor: theme.border },
                        active && { borderColor: accent.main, backgroundColor: accent.soft }
                      ]}
                      onPress={() => updateSettings({ fontSize: key })}
                    >
                      <Text style={[s.fontSizeChipText, { color: active ? accent.softText : theme.text, fontSize: preview }]}>Aa</Text>
                      <Text style={[s.fontSizeChipLabel, { color: active ? accent.softText : theme.textMuted }]}>
                        {t('text' + key.charAt(0).toUpperCase() + key.slice(1))}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>

            {/* Location */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="location-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>{t('locationTitle')}</Text>
              </View>
              <Text style={[s.settingHint, { color: theme.textMuted }]}>{t('locationHint')}</Text>

              <Text style={[s.formLabel, { color: theme.textMuted, marginTop: 10 }]}>{t('stateLabel')}</Text>
              <TouchableOpacity
                style={[s.districtPicker, { backgroundColor: theme.inputBg, borderColor: theme.border }]}
                onPress={() => openLocationPicker('state')}
              >
                <Text style={[s.districtPickerText, { color: locationInfo.state ? theme.text : theme.placeholder }]}>
                  {locationInfo.state || t('selectState')}
                </Text>
                <Ionicons name="chevron-down" size={16} color={theme.textMuted} />
              </TouchableOpacity>

              <Text style={[s.formLabel, { color: theme.textMuted, marginTop: 10 }]}>{t('districtLabel')}</Text>
              <TouchableOpacity
                style={[s.districtPicker, { backgroundColor: theme.inputBg, borderColor: theme.border }]}
                onPress={() => openLocationPicker('district')}
              >
                <Text style={[s.districtPickerText, { color: locationInfo.district ? theme.text : theme.placeholder }]}>
                  {locationInfo.district ? `${locationInfo.district} (${locationInfo.state || ''})` : t('selectDistrict')}
                </Text>
                <Ionicons name="chevron-down" size={16} color={theme.textMuted} />
              </TouchableOpacity>

              <TouchableOpacity
                style={[s.locationBtn, { backgroundColor: accent.soft, borderColor: accent.main }]}
                onPress={detectLocation}
                disabled={locating}
              >
                {locating ? <ActivityIndicator color={accent.main} size="small" /> : (
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="navigate-outline" size={15} color={accent.softText} style={{ marginRight: 7 }} />
                    <Text style={[s.locationBtnText, { color: accent.softText }]}>{t('useMyLocation')}</Text>
                  </View>
                )}
              </TouchableOpacity>
            </View>

            {/* Connection */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="server-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>{t('serverUrl')}</Text>
              </View>
              <TextInput
                style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                value={apiUrl}
                onChangeText={setApiUrl}
                placeholder="http://192.168.1.100:8000"
                placeholderTextColor={theme.placeholder}
              />
              <Text style={{ fontSize: 11 * fontScale, color: theme.textMuted, marginTop: 6, fontFamily: FONT.medium }}>
                Modify to point to your FastAPI server IP on the local network (e.g. 192.168.x.x:8000).
              </Text>
            </View>

            {/* AI Service (GCP) — endpoint & key come from mobile/.env (EXPO_PUBLIC_AI_API_URL / _API_KEY) */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="cloud-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>AI Service (GCP)</Text>
                <View style={[s.aiStatusDot, { backgroundColor: !AI_CONFIGURED ? theme.placeholder : (aiStatus ? (aiStatus.ok ? '#22c55e' : '#ef4444') : '#f59e0b') }]} />
              </View>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>
                Endpoint: {AI_BASE_URL_VALUE || 'not set — add EXPO_PUBLIC_AI_API_URL to mobile/.env'}
              </Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>
                API key: {AI_CONFIGURED ? 'configured' : 'not set — add EXPO_PUBLIC_AI_API_KEY to mobile/.env'}
              </Text>
              {aiStatus && (
                <Text style={{ fontSize: 12 * fontScale, color: aiStatus.ok ? '#22c55e' : theme.danger, fontFamily: FONT.medium, marginTop: 2 }}>
                  {aiStatus.ok ? '● ' : '● '}{aiStatus.text}
                </Text>
              )}
              <TouchableOpacity
                style={[s.resetBtn, { borderColor: accent.main, marginTop: 8 }]}
                onPress={checkAiConnection}
                disabled={aiChecking}
              >
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  {aiChecking ? (
                    <ActivityIndicator size="small" color={accent.main} style={{ marginRight: 8 }} />
                  ) : (
                    <Ionicons name="pulse-outline" size={17} color={accent.main} style={{ marginRight: 8 }} />
                  )}
                  <Text style={[s.resetBtnText, { color: accent.main }]}>Check connection</Text>
                </View>
              </TouchableOpacity>
            </View>

            {/* Model info */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="hardware-chip-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>Model Mesh Metadata</Text>
              </View>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>• Text Embedder: BAAI/bge-m3 (1024-dim)</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>• Classification: ViT-Small (Fine-tuned)</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>• Yield Engine: Tabular lightGBM</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>• RAG Index Chunks: 723,439 documents</Text>
            </View>

            {/* Reset */}
            <TouchableOpacity
              style={[s.resetBtn, { borderColor: theme.danger }]}
              onPress={handleResetSettings}
            >
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="refresh-outline" size={17} color={theme.danger} style={{ marginRight: 8 }} />
                <Text style={[s.resetBtnText, { color: theme.danger }]}>{t('resetSettings')}</Text>
              </View>
            </TouchableOpacity>
          </ScrollView>
        )}
      </View>

      {/* Floating bottom navigation bar (lifted above the home-indicator/gesture bar) */}
      <View
        style={[
          s.navBar,
          {
            backgroundColor: theme.surface,
            borderColor: theme.border,
            paddingBottom: Math.max(insets.bottom, 0),
            height: 64 + Math.max(insets.bottom, 0),
          },
        ]}
      >
        {TABS.map((item) => {
          const isActive = activeTab === item.key;
          return (
            <TouchableOpacity
              key={item.key}
              style={[s.navItem, isActive && [s.navActive, { backgroundColor: accent.soft }]]}
              onPress={() => setActiveTab(item.key)}
              activeOpacity={0.7}
            >
              <Ionicons
                name={isActive ? item.iconActive : item.icon}
                size={21}
                color={isActive ? accent.softText : theme.textMuted}
              />
              <Text style={[s.navText, { color: isActive ? accent.softText : theme.textMuted }]}>
                {t(item.labelKey)}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Toast */}
      {toast && (
        <View style={[s.toast, { backgroundColor: theme.surface, borderColor: accent.main }]}>
          <Text style={[s.toastText, { color: theme.text }]}>{toast}</Text>
        </View>
      )}

      {/* RAG Source Citation Inspector modal (bottom sheet) */}
      {selectedCitation && (
        <View style={s.modalBg}>
          <View style={[s.sheet, { backgroundColor: theme.surface, paddingBottom: 34 + insets.bottom }]}>
            <View style={s.sheetHandle} />
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
              <Ionicons name="document-text-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
              <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
                Source: {selectedCitation.source_type}
              </Text>
            </View>
            <Text style={{ fontSize: 13 * fontScale, color: theme.text, lineHeight: 20, fontFamily: FONT.regular }}>
              {selectedCitation.text}
            </Text>
            <TouchableOpacity
              style={[s.modalCloseBtn, { backgroundColor: accent.main }]}
              onPress={() => setSelectedCitation(null)}
            >
              <Text style={{ color: '#fff', fontWeight: 'bold', fontFamily: FONT.bold }}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Location picker modal (State -> District drill-down) */}
      <Modal
        visible={locModalOpen}
        transparent
        animationType="slide"
        onRequestClose={() => { setLocModalOpen(false); setLocStep('state'); }}
      >
        <View style={s.modalBg}>
          <View style={[s.sheet, { backgroundColor: theme.surface, paddingBottom: 34 + insets.bottom }]}>
            <View style={s.sheetHandle} />
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
              <Ionicons name="location-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
              <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
                {locStep === 'state' ? t('selectState') : t('selectDistrict')}
              </Text>
              {locStep === 'district' && (
                <TouchableOpacity onPress={() => setLocStep('state')}>
                  <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                    <Ionicons name="chevron-back" size={14} color={accent.main} />
                    <Text style={{ color: accent.main, marginLeft: 2, fontWeight: '600', fontSize: 13 * fontScale, fontFamily: FONT.semibold }}>
                      {t('backToStates')}
                    </Text>
                  </View>
                </TouchableOpacity>
              )}
            </View>

            {locStep === 'state' ? (
              states.length > 0 ? (
                <FlatList
                  data={states}
                  keyExtractor={(item) => item}
                  style={{ maxHeight: 320 }}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      style={[s.districtRow, { borderBottomColor: theme.border }]}
                      onPress={() => chooseStateForLoc(item)}
                    >
                      <Text style={[s.districtRowText, { color: theme.text }]}>{item}</Text>
                      {(locationInfo.state || '').toLowerCase() === item.toLowerCase() && (
                        <Ionicons name="checkmark-circle" size={18} color={accent.main} />
                      )}
                    </TouchableOpacity>
                  )}
                />
              ) : (
                <Text style={{ color: theme.textMuted, paddingVertical: 12, fontSize: 13 * fontScale, fontFamily: FONT.medium }}>
                  {t('stateListUnavailable')}
                </Text>
              )
            ) : (
              districtsLoading ? (
                <ActivityIndicator color={accent.main} style={{ marginVertical: 24 }} />
              ) : districts.length > 0 ? (
                <FlatList
                  data={districts}
                  keyExtractor={(item) => item}
                  style={{ maxHeight: 320 }}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      style={[s.districtRow, { borderBottomColor: theme.border }]}
                      onPress={() => chooseDistrictForLoc(item)}
                    >
                      <Text style={[s.districtRowText, { color: theme.text }]}>{item}</Text>
                      {(locationInfo.district || '').toLowerCase() === item.toLowerCase() && (
                        <Ionicons name="checkmark-circle" size={18} color={accent.main} />
                      )}
                    </TouchableOpacity>
                  )}
                />
              ) : (
                <Text style={{ color: theme.textMuted, paddingVertical: 12, fontSize: 13 * fontScale, fontFamily: FONT.medium }}>
                  {t('districtListUnavailable')}
                </Text>
              )
            )}

            <TouchableOpacity
              style={[s.modalCloseBtn, { backgroundColor: accent.main, marginTop: 12 }]}
              onPress={() => { setLocModalOpen(false); setLocStep('state'); }}
            >
              <Text style={{ color: '#fff', fontWeight: 'bold', fontFamily: FONT.bold }}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Language picker modal */}
      <Modal
        visible={langModalOpen}
        transparent
        animationType="slide"
        onRequestClose={() => setLangModalOpen(false)}
      >
        <View style={s.modalBg}>
          <View style={[s.sheet, { backgroundColor: theme.surface, paddingBottom: 34 + insets.bottom }]}>
            <View style={s.sheetHandle} />
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
              <Ionicons name="globe-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
              <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
                {t('langLabel')}
              </Text>
            </View>
            {SUPPORTED_LANGUAGES.map((code) => (
              <TouchableOpacity
                key={code}
                style={[s.districtRow, { borderBottomColor: theme.border }]}
                onPress={() => {
                  setLanguage(code);
                  setLangModalOpen(false);
                }}
              >
                <Text style={[s.districtRowText, { color: theme.text }]}>{LANG_NAMES[code]}</Text>
                {i18n.language === code && (
                  <Ionicons name="checkmark-circle" size={18} color={accent.main} />
                )}
              </TouchableOpacity>
            ))}
            <TouchableOpacity
              style={[s.modalCloseBtn, { backgroundColor: accent.main, marginTop: 12 }]}
              onPress={() => setLangModalOpen(false)}
            >
              <Text style={{ color: '#fff', fontWeight: 'bold', fontFamily: FONT.bold }}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Full-screen Weather Detail view */}
      <Modal
        visible={weatherDetailOpen}
        animationType="slide"
        onRequestClose={() => setWeatherDetailOpen(false)}
      >
        <View style={[s.container, { backgroundColor: theme.bg }]}>
          <LinearGradient
            colors={theme.weatherGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[s.weatherDetailHeader, { paddingTop: insets.top + 10 }]}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
              <TouchableOpacity
                style={s.weatherBackBtn}
                onPress={() => setWeatherDetailOpen(false)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons name="chevron-back" size={22} color="#fff" />
              </TouchableOpacity>
              <Text style={s.weatherDetailTitle}>{t('weatherTitle')}</Text>
              <View style={{ width: 34 }} />
            </View>

            {/* Hero block */}
            <View style={{ alignItems: 'center', paddingVertical: 18 }}>
              <Ionicons name={weatherIcon(weather.wmoCode)} size={72} color="#fff" />
              <Text style={s.weatherDetailTemp}>{weather.temp}°C</Text>
              <Text style={s.weatherDetailCond}>{weather.condition}</Text>
              <Text style={s.weatherDetailLoc}>{weather.location}</Text>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginTop: 10, gap: 8 }}>
                {weather.feelsLike != null && (
                  <View style={s.weatherDetailPill}>
                    <Ionicons name="thermometer-outline" size={12} color="#fff" style={{ marginRight: 4 }} />
                    <Text style={s.weatherDetailPillText}>{t('feelsLike')} {weather.feelsLike}°C</Text>
                  </View>
                )}
                {weather.maxTemp != null && weather.minTemp != null && (
                  <View style={s.weatherDetailPill}>
                    <Text style={s.weatherDetailPillText}>↑{weather.maxTemp}° ↓{weather.minTemp}°</Text>
                  </View>
                )}
              </View>
            </View>
          </LinearGradient>

          <ScrollView contentContainerStyle={{ padding: 16, paddingBottom: 40 }} showsVerticalScrollIndicator={false}>
            {/* Live status strip */}
            <View style={[s.card, { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 12 }]}>
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <View style={[s.liveBadge, { backgroundColor: accent.soft }]}>
                  <Text style={[s.liveBadgeText, { color: accent.softText }]}>
                    {weather.source === 'live' ? '● Live' : '● Static'}
                  </Text>
                </View>
                <Text style={[s.weatherDetailUpdated, { color: theme.textMuted }]}>
                  {t('weatherUpdated')} {fmtTime(weather.updatedAt)}
                </Text>
              </View>
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="refresh-outline" size={13} color={accent.softText} style={{ marginRight: 4 }} />
                <Text style={{ color: accent.softText, fontSize: 11 * fontScale, fontFamily: FONT.semibold }}>5 min</Text>
              </View>
            </View>

            {/* Stats grid */}
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, marginTop: 4 }}>
              <View style={s.sectionHeader}>
                <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
                <Text style={s.sectionTitle}>{t('weatherConditions')}</Text>
              </View>
            </View>
            <View style={s.statGrid}>
              {buildStatGrid().map((item, i) => (
                <View key={i} style={{ width: '48%' }}>{item}</View>
              ))}
            </View>

            {/* Sunrise / sunset */}
            {(weather.sunrise || weather.sunset) && (
              <View style={[s.card, { flexDirection: 'row', justifyContent: 'space-around', paddingVertical: 14 }]}>
                <View style={{ alignItems: 'center' }}>
                  <Ionicons name="sunny-outline" size={20} color={theme.warning} />
                  <Text style={[s.statLabel, { color: theme.textMuted, marginTop: 4 }]}>{t('sunriseLabel')}</Text>
                  <Text style={[s.statValue, { color: theme.text }]}>{fmtTime(weather.sunrise)}</Text>
                </View>
                <View style={{ width: 1, backgroundColor: theme.border }} />
                <View style={{ alignItems: 'center' }}>
                  <Ionicons name="moon-outline" size={20} color="#6366f1" />
                  <Text style={[s.statLabel, { color: theme.textMuted, marginTop: 4 }]}>{t('sunsetLabel')}</Text>
                  <Text style={[s.statValue, { color: theme.text }]}>{fmtTime(weather.sunset)}</Text>
                </View>
              </View>
            )}

            {/* 3-day forecast */}
            {weather.forecast?.length > 0 && (
              <View style={s.card}>
                <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
                  <Ionicons name="calendar-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                  <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 0, paddingBottom: 0 }]}>{t('forecastTitle')}</Text>
                </View>
                {weather.forecast.map((f, i) => (
                  <View
                    key={i}
                    style={[s.forecastRow, i > 0 && { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border }]}
                  >
                    <Text style={[s.forecastDay, { color: theme.text }]}>
                      {new Date(f.date + 'T00:00:00').toLocaleDateString(undefined, { weekday: 'short', day: 'numeric' })}
                    </Text>
                    <Ionicons name={weatherIcon(f.wmo_code)} size={18} color={accent.main} />
                    <Text style={[s.forecastTemp, { color: theme.text }]}>
                      ↑{f.max_temp_c != null ? Math.round(f.max_temp_c) : '—'}°  ↓{f.min_temp_c != null ? Math.round(f.min_temp_c) : '—'}°
                    </Text>
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                      <Ionicons name="umbrella-outline" size={12} color="#3b82f6" style={{ marginRight: 4 }} />
                      <Text style={{ color: theme.textMuted, fontSize: 12 * fontScale, fontFamily: FONT.medium }}>
                        {f.rain_probability != null ? `${Math.round(f.rain_probability)}%` : '—'}
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            )}

            {/* Field advisory (farmer + crop guidance from live conditions) */}
            <View style={s.card}>
              <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
                <Ionicons name="leaf-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 0, paddingBottom: 0 }]}>{t('advisoryTitle')}</Text>
              </View>
              <Text style={[s.settingHint, { color: theme.textMuted, marginTop: 4 }]}>
                {t('advisoryHint')}
              </Text>
              <View style={{ marginTop: 8, marginBottom: 4 }}>
                {buildAdvisory().map((tip, i) => (
                  <View key={i} style={[s.tipRow, i > 0 && { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border }]}>
                    <View style={[s.tipIcon, { backgroundColor: accent.soft }]}>
                      {renderIcon(tip.icon, 15, accent.softText, undefined)}
                    </View>
                    <Text style={[s.tipText, { color: theme.text }]}>{t(tip.key, { ...tip.values })}</Text>
                  </View>
                ))}
              </View>
            </View>
          </ScrollView>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

// Build styles dynamically from the active theme so every color token adapts.
const createStyles = (theme, accent, fontScale, themeMode) => {
  const fs = fontScale;
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.bg
    },
    header: {
      paddingHorizontal: 18,
      paddingTop: 14,
      paddingBottom: 18,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    headerTitle: {
      fontSize: 22 * fs,
      fontWeight: '800',
      fontFamily: FONT.bold,
      color: '#ffffff',
      letterSpacing: -0.5
    },
    headerSubtitle: {
      fontSize: 11 * fs,
      color: 'rgba(255,255,255,0.85)',
      fontWeight: '500',
      fontFamily: FONT.medium,
      marginTop: 2
    },
    langPill: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.18)',
      borderRadius: 999,
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.35)',
      gap: 5
    },
    langText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      color: '#ffffff'
    },
    contentContainer: {
      flex: 1
    },
    scrollContent: {
      padding: 16,
      paddingBottom: 20
    },
    greeting: {
      fontSize: 15 * fs,
      fontWeight: '700',
      fontFamily: FONT.semibold,
      color: theme.text,
      marginBottom: 12
    },
    sectionHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
      marginBottom: 10,
      marginTop: 15
    },
    sectionAccent: {
      width: 4,
      height: 16,
      borderRadius: 2
    },
    sectionTitle: {
      fontSize: 16 * fs,
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      color: theme.text
    },
    liveBadge: {
      paddingVertical: 3,
      paddingHorizontal: 8,
      borderRadius: 12,
      marginLeft: 4
    },
    liveBadgeText: {
      fontSize: 10 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold
    },
    mandiMarket: {
      fontSize: 10 * fs,
      fontFamily: FONT.medium,
      marginTop: 1
    },
    mandiLocation: {
      fontSize: 11 * fs,
      marginLeft: 6,
      marginBottom: 0,
      fontWeight: '500',
      fontFamily: FONT.medium
    },
    districtPicker: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderRadius: 12,
      paddingHorizontal: 12,
      paddingVertical: 12,
      marginTop: 6
    },
    districtPickerText: {
      fontSize: 14 * fs,
      fontFamily: FONT.medium
    },
    locationBtn: {
      borderWidth: 1,
      borderRadius: 12,
      paddingVertical: 11,
      alignItems: 'center',
      marginTop: 14
    },
    locationBtnText: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    districtRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingVertical: 13,
      borderBottomWidth: StyleSheet.hairlineWidth
    },
    districtRowText: {
      fontSize: 15 * fs,
      fontFamily: FONT.medium
    },
    card: {
      backgroundColor: theme.surface,
      borderRadius: 20,
      padding: 16,
      marginBottom: 15,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderColor: theme.border,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: theme.shadowOpacity,
      shadowRadius: 12,
      elevation: 3
    },
    weatherCard: {
      borderWidth: 0,
      shadowOpacity: 0.2
    },
    weatherTemp: {
      fontSize: 34 * fs,
      fontWeight: '800',
      fontFamily: FONT.extrabold,
      color: '#fff'
    },
    weatherDesc: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 13 * fs,
      fontFamily: FONT.medium
    },
    weatherLabel: {
      color: 'rgba(255,255,255,0.8)',
      fontSize: 11 * fs,
      fontFamily: FONT.medium
    },
    weatherVal: {
      color: '#fff',
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 15 * fs
    },
    weatherForecastRow: {
      flexDirection: 'row',
      gap: 8,
      marginTop: 14
    },
    weatherForecastChip: {
      flex: 1,
      backgroundColor: 'rgba(255,255,255,0.14)',
      borderRadius: 14,
      paddingVertical: 8,
      paddingHorizontal: 6,
      alignItems: 'center'
    },
    weatherForecastDay: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 10 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      marginBottom: 3
    },
    weatherForecastTemp: {
      color: '#fff',
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    advisoryBanner: {
      flex: 1,
      color: '#fff',
      fontSize: 12 * fs,
      lineHeight: 18 * fs,
      fontFamily: FONT.medium
    },
    mandiRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: 11
    },
    mandiCrop: {
      fontWeight: '600',
      fontFamily: FONT.semibold,
      color: theme.text,
      fontSize: 12.5 * fs
    },
    mandiPrice: {
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      color: theme.text,
      fontSize: 13 * fs,
      marginTop: 4
    },
    mandiChange: {
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      marginLeft: 3
    },
    mandiLocationRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderRadius: 14,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      paddingHorizontal: 13,
      paddingVertical: 10,
      marginBottom: 12
    },
    mandiLocationEdit: {
      fontSize: 11 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      marginLeft: 4
    },
    mandiGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 10,
    },
    mandiCard: {
      width: '30%',
      height: 108,
      borderRadius: 16,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      padding: 10,
    },
    shortcutsGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginTop: 5
    },
    shortcutBtn: {
      width: '31%',
      borderRadius: 20,
      paddingVertical: 18,
      paddingHorizontal: 8,
      alignItems: 'center',
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: theme.shadowOpacity,
      shadowRadius: 6,
      elevation: 2
    },
    shortcutIconWrap: {
      width: 46,
      height: 46,
      borderRadius: 23,
      alignItems: 'center',
      justifyContent: 'center',
      marginBottom: 8
    },
    shortcutIcon: {
      fontSize: 22
    },
    shortcutText: {
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      color: theme.text,
      textAlign: 'center'
    },
    photoSelectorContainer: {
      borderRadius: 20,
      borderWidth: 2,
      borderStyle: 'dashed',
      padding: 20,
      alignItems: 'center',
      marginBottom: 16
    },
    photoPlaceholder: {
      height: 180,
      justifyContent: 'center',
      alignItems: 'center'
    },
    photoHint: {
      marginTop: 12,
      fontSize: 12 * fs,
      fontFamily: FONT.medium
    },
    selectedLeafImage: {
      width: '100%',
      height: 200,
      borderRadius: 14,
      resizeMode: 'cover'
    },
    photoActionsRow: {
      flexDirection: 'row',
      gap: 12,
      marginTop: 15
    },
    photoBtn: {
      paddingVertical: 10,
      paddingHorizontal: 16,
      borderRadius: 13
    },
    photoBtnText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    actionBtn: {
      borderRadius: 18,
      overflow: 'hidden',
      marginBottom: 16,
      shadowColor: accent.main,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.3,
      shadowRadius: 8,
      elevation: 4
    },
    actionBtnGradient: {
      padding: 15,
      alignItems: 'center',
      justifyContent: 'center'
    },
    actionBtnText: {
      color: '#fff',
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 14 * fs
    },
    cardHeader: {
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 15 * fs,
      color: theme.text,
      borderBottomWidth: 1,
      borderBottomColor: theme.border,
      paddingBottom: 10,
      marginBottom: 12
    },
    resultBadgeRow: {
      flexDirection: 'row',
      gap: 8,
      marginBottom: 12
    },
    badge: {
      paddingVertical: 5,
      paddingHorizontal: 12,
      borderRadius: 20
    },
    badgeText: {
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 11 * fs
    },
    alertCard: {
      borderWidth: 1,
      borderRadius: 12,
      padding: 10,
      marginBottom: 12
    },
    alertText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    answerText: {
      fontSize: 14 * fs,
      lineHeight: 20 * fs,
      fontFamily: FONT.regular
    },
    chatRow: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      marginBottom: 4
    },
    botAvatar: {
      width: 32,
      height: 32,
      borderRadius: 16,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 8,
      marginTop: 4
    },
    chatBubble: {
      maxWidth: '78%',
      padding: 13,
      borderRadius: 18,
      marginBottom: 12,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 2
    },
    userBubble: {
      borderBottomRightRadius: 4
    },
    botBubble: {
      borderBottomLeftRadius: 4,
      borderWidth: 1
    },
    chatText: {
      fontSize: 14 * fs,
      lineHeight: 20 * fs,
      fontFamily: FONT.regular
    },
    citationRow: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 6,
      marginTop: 8,
      borderTopWidth: StyleSheet.hairlineWidth,
      paddingTop: 8
    },
    citationChip: {
      paddingVertical: 3,
      paddingHorizontal: 9,
      borderRadius: 12
    },
    citationText: {
      fontSize: 10 * fs,
      fontWeight: '500',
      fontFamily: FONT.medium
    },
    quickBar: {
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderTopWidth: 1
    },
    quickBarLabel: {
      fontSize: 11 * fs,
      fontFamily: FONT.medium,
      marginBottom: 5
    },
    quickChip: {
      paddingHorizontal: 12,
      paddingVertical: 7,
      borderRadius: 16,
      marginRight: 8
    },
    quickChipText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    inputBar: {
      flexDirection: 'row',
      padding: 12,
      borderTopWidth: 1,
      alignItems: 'center'
    },
    attachBtn: {
      width: 38,
      height: 38,
      borderRadius: 19,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 8
    },
    chatAttachRow: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderTopWidth: 1
    },
    chatAttachThumb: {
      width: 44,
      height: 44,
      borderRadius: 8
    },
    chatAttachName: {
      flex: 1,
      fontSize: 12 * fs,
      fontFamily: FONT.medium,
      marginHorizontal: 10
    },
    chatImageThumb: {
      width: 150,
      height: 150,
      borderRadius: 12,
      marginBottom: 6
    },
    chatTextInput: {
      flex: 1,
      height: 42,
      borderRadius: 21,
      paddingHorizontal: 16,
      fontSize: 14 * fs,
      fontFamily: FONT.regular
    },
    sendBtn: {
      marginLeft: 10,
      borderRadius: 21,
      overflow: 'hidden',
      shadowColor: accent.main,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.3,
      shadowRadius: 4,
      elevation: 3
    },
    sendBtnGradient: {
      paddingVertical: 12,
      paddingHorizontal: 16
    },
    sendBtnText: {
      color: '#fff',
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 13 * fs
    },
    formRow: {
      marginBottom: 12
    },
    formLabel: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      marginBottom: 0,
      color: theme.text
    },
    formInput: {
      height: 46,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderRadius: 13,
      paddingHorizontal: 12,
      fontSize: 14 * fs,
      fontFamily: FONT.medium
    },
    yieldGaugeContainer: {
      alignItems: 'center',
      paddingVertical: 20,
      borderBottomWidth: StyleSheet.hairlineWidth,
      marginBottom: 15
    },
    yieldValText: {
      fontSize: 34 * fs,
      fontWeight: '800',
      fontFamily: FONT.extrabold
    },
    economicsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12
    },
    econItem: {
      width: '46%',
      borderRadius: 14,
      padding: 10,
      borderWidth: themeMode === 'highContrast' ? 2 : 1
    },
    econLabel: {
      fontSize: 11 * fs,
      fontFamily: FONT.medium,
      marginBottom: 4
    },
    econVal: {
      fontSize: 14 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    weatherChipRow: {
      flexDirection: 'row',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: 8,
      marginTop: 12
    },
    weatherChip: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.16)',
      borderRadius: 14,
      paddingVertical: 5,
      paddingHorizontal: 10
    },
    weatherChipText: {
      color: '#fff',
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    weatherDetailHeader: {
      paddingHorizontal: 18,
      paddingBottom: 8
    },
    weatherBackBtn: {
      width: 34,
      height: 34,
      borderRadius: 17,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: 'rgba(255,255,255,0.18)'
    },
    weatherDetailTitle: {
      fontSize: 17 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      color: '#fff',
      letterSpacing: -0.3
    },
    weatherDetailTemp: {
      fontSize: 52 * fs,
      fontWeight: '800',
      fontFamily: FONT.extrabold,
      color: '#fff',
      letterSpacing: -1
    },
    weatherDetailCond: {
      color: '#fff',
      fontSize: 15 * fs,
      fontFamily: FONT.semibold,
      marginTop: 4
    },
    weatherDetailLoc: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 13 * fs,
      fontFamily: FONT.medium,
      marginTop: 2
    },
    weatherDetailPill: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.16)',
      borderRadius: 16,
      paddingVertical: 6,
      paddingHorizontal: 12
    },
    weatherDetailPillText: {
      color: '#fff',
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    weatherDetailUpdated: {
      fontSize: 11 * fs,
      fontFamily: FONT.medium,
      marginLeft: 8
    },
    statGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12
    },
    statItem: {
      width: '100%',
      borderRadius: 16,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      padding: 12,
      marginBottom: 12
    },
    statLabel: {
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    statValue: {
      fontSize: 17 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      letterSpacing: -0.2
    },
    forecastRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingVertical: 12
    },
    forecastDay: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      width: '36%'
    },
    forecastTemp: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    tipRow: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      paddingVertical: 10
    },
    tipIcon: {
      width: 30,
      height: 30,
      borderRadius: 15,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 10,
      marginTop: 1
    },
    tipText: {
      flex: 1,
      fontSize: 13 * fs,
      lineHeight: 19 * fs,
      fontFamily: FONT.regular
    },
    navBar: {
      height: 64,
      marginHorizontal: 14,
      marginBottom: 10,
      borderRadius: 26,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      flexDirection: 'row',
      justifyContent: 'space-around',
      alignItems: 'center',
      paddingHorizontal: 8,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: theme.shadowOpacity,
      shadowRadius: 14,
      elevation: 8
    },
    navItem: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 7,
      paddingHorizontal: 12,
      borderRadius: 18,
      minWidth: 58
    },
    navActive: {
      borderRadius: 18
    },
    navIcon: {
      fontSize: 19,
      marginBottom: 2
    },
    navText: {
      fontSize: 10 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      marginTop: 2
    },
    settingsCardHeader: {
      flexDirection: 'row',
      alignItems: 'center'
    },
    settingHint: {
      fontSize: 11.5 * fs,
      lineHeight: 16 * fs,
      fontFamily: FONT.regular
    },
    themeRow: {
      flexDirection: 'row',
      gap: 10,
      marginTop: 10
    },
    themeChip: {
      flex: 1,
      borderRadius: 16,
      borderWidth: 2,
      padding: 8,
      alignItems: 'center'
    },
    themePreview: {
      width: '100%',
      height: 48,
      borderRadius: 9,
      borderWidth: 1,
      overflow: 'hidden',
      marginBottom: 6
    },
    themePreviewBar: {
      height: 14,
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 6
    },
    themePreviewDot: {
      width: 8,
      height: 8,
      borderRadius: 4
    },
    themePreviewLine: {
      flex: 1,
      marginHorizontal: 6,
      marginBottom: 6,
      borderRadius: 2
    },
    themeChipText: {
      fontSize: 11 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      textAlign: 'center'
    },
    accentRow: {
      flexDirection: 'row',
      gap: 14,
      marginTop: 14,
      justifyContent: 'center'
    },
    swatch: {
      width: 42,
      height: 42,
      borderRadius: 21,
      alignItems: 'center',
      justifyContent: 'center'
    },
    swatchCheck: {
      color: '#fff',
      fontWeight: 'bold',
      fontSize: 16
    },
    fontSizeRow: {
      flexDirection: 'row',
      gap: 10,
      marginTop: 10
    },
    fontSizeChip: {
      flex: 1,
      borderRadius: 16,
      borderWidth: 2,
      paddingVertical: 12,
      alignItems: 'center'
    },
    fontSizeChipText: {
      fontWeight: '800',
      fontFamily: FONT.extrabold,
      marginBottom: 2
    },
    fontSizeChipLabel: {
      fontSize: 10 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    resetBtn: {
      borderWidth: 1.5,
      borderRadius: 16,
      padding: 13,
      alignItems: 'center',
      marginBottom: 30
    },
    aiStatusDot: {
      width: 9,
      height: 9,
      borderRadius: 5,
      marginLeft: 'auto'
    },
    resetBtnText: {
      fontWeight: '700',
      fontFamily: FONT.bold,
      fontSize: 13 * fs
    },
    toast: {
      position: 'absolute',
      bottom: 96,
      left: 20,
      right: 20,
      borderRadius: 16,
      borderWidth: 1,
      padding: 12,
      alignItems: 'center',
      zIndex: 100,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 10,
      elevation: 6
    },
    toastText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    modalBg: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.55)',
      justifyContent: 'flex-end',
      zIndex: 9999
    },
    sheet: {
      width: '100%',
      borderTopLeftRadius: 26,
      borderTopRightRadius: 26,
      padding: 20,
      paddingBottom: 34,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: -4 },
      shadowOpacity: 0.15,
      shadowRadius: 10,
      elevation: 12
    },
    sheetHandle: {
      alignSelf: 'center',
      width: 40,
      height: 5,
      borderRadius: 3,
      backgroundColor: 'rgba(120,130,150,0.4)',
      marginBottom: 16
    },
    modalCloseBtn: {
      marginTop: 15,
      padding: 11,
      borderRadius: 14,
      alignItems: 'center'
    }
  });
};