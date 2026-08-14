import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  StyleSheet,
  Text,
  View,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Image,
  ActivityIndicator,
  Alert,
  Modal,
  FlatList,
  Platform
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';
import Storage from 'expo-sqlite/kv-store';
import * as Location from 'expo-location';
import { initI18n, setLanguage, SUPPORTED_LANGUAGES, LANG_NAMES } from './i18n';

// Default API Server. Configurable in app settings.
const DEFAULT_API_URL = 'http://127.0.0.1:8000';

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
    bg: '#f2f4f7',
    surface: '#ffffff',
    surfaceAlt: '#eef1f4',
    surfaceDeep: '#e8ebef',
    text: '#111827',
    textMuted: '#64748b',
    border: '#e2e8f0',
    inputBg: '#f4f6f8',
    placeholder: '#94a3b8',
    shadow: '#0f172a',
    shadowOpacity: 0.07,
    statusBar: 'dark',
    weatherGradient: ['#059669', '#065f46'],
    danger: '#ef4444',
    dangerBg: '#fef2f2',
    success: '#10b981',
    successBg: '#ecfdf5',
    warning: '#d97706',
    warningBg: '#fffbeb',
  },
  dark: {
    name: 'Dark',
    bg: '#0b0e14',
    surface: '#151a23',
    surfaceAlt: '#1d2531',
    surfaceDeep: '#0f141c',
    text: '#e7ecf3',
    textMuted: '#94a3b8',
    border: '#232d3b',
    inputBg: '#10151d',
    placeholder: '#64748b',
    shadow: '#000000',
    shadowOpacity: 0.35,
    statusBar: 'light',
    weatherGradient: ['#0f766e', '#134e4a'],
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

export default function App() {
  const { t, i18n } = useTranslation();
  const [i18nReady, setI18nReady] = useState(false);
  const [activeTab, setActiveTab] = useState('home');
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);

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
  const [weather, setWeather] = useState({ temp: 31, condition: 'Sunny', location: 'Uttar Pradesh', rain: '850mm', humidity: null, forecast: [], source: 'static' });
  const [mandiPrices, setMandiPrices] = useState([
    { crop: 'Wheat', tag: 'MSP', price: '₹2,275/qtl', change: '+₹15' },
    { crop: 'Paddy', tag: 'MSP', price: '₹2,183/qtl', change: '+₹10' },
    { crop: 'Maize', tag: 'MSP', price: '₹2,090/qtl', change: '-₹5' },
    { crop: 'Mustard', tag: 'MSP', price: '₹5,650/qtl', change: '+₹40' }
  ]);
  const [mandiSource, setMandiSource] = useState('static');

  // Live weather from backend /api/weather/current (GPS coords preferred, else district city)
  useEffect(() => {
    if (!apiUrl) return;
    let cancelled = false;
    (async () => {
      try {
        const params = new URLSearchParams();
        if (locationInfo.lat != null && locationInfo.lon != null) {
          params.set('lat', locationInfo.lat);
          params.set('lon', locationInfo.lon);
        } else if (locationInfo.district) {
          params.set('city', locationInfo.district);
        }
        if (!params.toString()) return;
        const res = await fetch(`${apiUrl}/api/weather/current?${params.toString()}`);
        if (!res.ok) throw new Error('bad weather response');
        const data = await res.json();
        if (cancelled || data.temperature_c == null) return;
        setWeather({
          temp: Math.round(data.temperature_c),
          condition: data.condition || '—',
          location: data.location || ([locationInfo.district, locationInfo.state].filter(Boolean).join(', ') || 'Uttar Pradesh'),
          rain: data.precipitation_mm != null ? `${data.precipitation_mm} mm` : '—',
          humidity: data.humidity != null ? `${data.humidity}%` : null,
          forecast: data.forecast || [],
          source: data.source,
        });
      } catch (e) {
        // Server unreachable -> keep the static default weather card
      }
    })();
    return () => { cancelled = true; };
  }, [apiUrl, locationInfo]);

  // Live mandi prices from backend /api/mandi/prices (falls back to static MSP list)
  useEffect(() => {
    // Until a location is set, show generic MSP reference (don't imply any district)
    if (!apiUrl || (!locationInfo.state && !locationInfo.district)) return;
    let cancelled = false;
    (async () => {
      try {
        const params = new URLSearchParams({
          state: locationInfo.state || 'Uttar Pradesh',
          district: locationInfo.district || '',
        });
        const res = await fetch(`${apiUrl}/api/mandi/prices?${params.toString()}`);
        if (!res.ok) throw new Error('bad mandi response');
        const data = await res.json();
        if (cancelled || !data.prices?.length) return;

        const rows = data.prices.map(p => ({
          crop: p.crop,
          tag: p.variety === 'MSP Reference'
            ? 'MSP'
            : (p.variety && p.variety !== '—' ? p.variety : null),
          market: p.market && p.market !== '—' ? p.market : null,
          price: p.modal_price != null
            ? `₹${Math.round(p.modal_price).toLocaleString('en-IN')}/qtl`
            : '—',
          change: p.change_per_quintal != null
            ? `${p.change_per_quintal >= 0 ? '+' : ''}₹${Math.round(p.change_per_quintal)}`
            : '—',
        }));

        // One row per crop (first = latest price) so every available crop is shown
        const seenCrops = new Set();
        const deduped = rows.filter(r => {
          const key = r.crop.toLowerCase();
          if (seenCrops.has(key)) return false;
          seenCrops.add(key);
          return true;
        });

        setMandiPrices(deduped);
        setMandiSource(data.source);
      } catch (e) {
        // Server unreachable or rate-limited -> keep the static MSP fallback list
      }
    })();
    return () => { cancelled = true; };
  }, [apiUrl, locationInfo]);

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

  // Chat/RAG States
  const [chatMessages, setChatMessages] = useState([
    { id: 1, text: 'Hello! I am your FarmerVision advisor. Ask me questions about crops, fertilizer dosage, disease remedies, or policy schemes.', isUser: false }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const chatScrollRef = useRef(null);

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
    if (!chatInput.trim()) return;

    const userText = chatInput;
    const nextMsgId = chatMessages.length + 1;
    
    setChatMessages(prev => [...prev, { id: nextMsgId, text: userText, isUser: true }]);
    setChatInput('');
    setIsTyping(true);

    try {
      const res = await fetch(`${apiUrl}/api/query/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userText })
      });

      if (res.ok) {
        const data = await res.json();
        setChatMessages(prev => [...prev, {
          id: prev.length + 1,
          text: data.answer,
          isUser: false,
          sources: data.sources,
          tier: data.tier,
          score: data.top_score
        }]);
      } else {
        const errData = await res.json();
        setChatMessages(prev => [...prev, {
          id: prev.length + 1,
          text: `Error: ${errData.detail || 'Server responded with error'}`,
          isUser: false
        }]);
      }
    } catch (e) {
      setChatMessages(prev => [...prev, {
        id: prev.length + 1,
        text: 'Network Connection Error. Please verify server URL in Settings.',
        isUser: false
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  // Simulated leaf photo loading to prevent file system errors in web environments
  const selectMockImage = (type) => {
    if (type === 'camera') {
      setSelectedImage({
        name: 'wheat_rust_infected.jpg',
        uri: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400'
      });
    } else {
      setSelectedImage({
        name: 'rice_spot_blight.jpg',
        uri: 'https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=400'
      });
    }
    setDiagnosisResult(null);
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

  const s = createStyles(theme, accent, fontScale, themeMode);

  // Wait for i18n init (device-language detection + persisted choice) before painting UI
  if (!i18nReady) {
    return (
      <View style={[s.container, { alignItems: 'center', justifyContent: 'center' }]}>
        <ActivityIndicator size="large" color={accent.main} />
      </View>
    );
  }

  return (
    <SafeAreaView style={[s.container, { backgroundColor: theme.bg }]}>
      <StatusBar style={theme.statusBar} />

      {/* Header bar */}
      <LinearGradient
        colors={[theme.weatherGradient[0], theme.weatherGradient[1]]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={s.header}
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
          <Text style={s.langText}>🌐 {LANG_NAMES[i18n.language] || 'English'}</Text>
        </TouchableOpacity>
      </LinearGradient>

      {/* Main Tab Screens Scroll area */}
      <View style={s.contentContainer}>
        {/* --- SCREEN 1: HOME --- */}
        {activeTab === 'home' && (
          <ScrollView contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>
            {/* Greeting */}
            <Text style={s.greeting}>{greeting()} 🌱</Text>

            {/* Weather & Advisory widget */}
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
                  <Text style={s.weatherLabel}>Rainfall</Text>
                  <Text style={s.weatherVal}>{weather.rain}</Text>
                  {weather.humidity && (
                    <Text style={[s.weatherLabel, { marginTop: 4 }]}>💧 {weather.humidity}</Text>
                  )}
                </View>
              </View>

              {/* 3-day forecast strip */}
              {weather.forecast?.length > 0 && (
                <View style={s.weatherForecastRow}>
                  {weather.forecast.map((f, i) => (
                    <View key={i} style={s.weatherForecastChip}>
                      <Text style={s.weatherForecastDay}>{String(f.date || '').slice(5)}</Text>
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
              <Text style={s.advisoryBanner}>
                🌾 Advisory: Ideal conditions for Rabi crop fertilization. Monitor wheat leaves for rust flags.
              </Text>
            </LinearGradient>

            {/* Mandi Prices (all crops, horizontally scrollable) */}
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
              <Text style={[s.mandiLocation, { color: theme.text }]}>
                📍 {[locationInfo.district, locationInfo.state].filter(Boolean).join(', ') || t('setLocation')}
              </Text>
              <Text style={[s.mandiLocationEdit, { color: accent.softText }]}>✏️ {t('changeLocation')}</Text>
            </TouchableOpacity>
            <View style={[s.card, { paddingBottom: 12 }]}>
              {mandiPrices.length > 0 ? (
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={s.mandiScroller}
                >
                  {mandiPrices.map((item, idx) => (
                    <View
                      key={idx}
                      style={[s.mandiCard, { backgroundColor: theme.surfaceAlt, borderColor: theme.border }]}
                    >
                      <Text style={s.mandiCrop}>
                        {t(`crops.${item.crop}`, { defaultValue: item.crop })}
                        {item.tag ? ` (${item.tag})` : ''}
                      </Text>
                      {item.market && (
                        <Text style={[s.mandiMarket, { color: theme.textMuted }]} numberOfLines={1}>{item.market}</Text>
                      )}
                      <Text style={s.mandiPrice}>{item.price}</Text>
                      <Text style={[s.mandiChange, { color: item.change === '—' ? theme.textMuted : (item.change.startsWith('-') ? theme.danger : theme.success) }]}>
                        {item.change}
                      </Text>
                    </View>
                  ))}
                </ScrollView>
              ) : (
                <Text style={{ color: theme.textMuted, fontSize: 13 * fontScale }}>
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
                  <Text style={s.shortcutIcon}>📸</Text>
                </View>
                <Text style={s.shortcutText}>{t('scanner')}</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
                onPress={() => setActiveTab('chat')}
              >
                <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
                  <Text style={s.shortcutIcon}>💬</Text>
                </View>
                <Text style={s.shortcutText}>{t('chat')}</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
                onPress={() => setActiveTab('yield')}
              >
                <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
                  <Text style={s.shortcutIcon}>📊</Text>
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
                  <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft, width: 72, height: 72, borderRadius: 36 }]}>
                    <Text style={{ fontSize: 36 }}>🍃</Text>
                  </View>
                  <Text style={[s.photoHint, { color: theme.textMuted }]}>Upload a photo of crop leaf</Text>
                </View>
              )}

              <View style={s.photoActionsRow}>
                <TouchableOpacity
                  style={[s.photoBtn, { backgroundColor: theme.surfaceAlt }]}
                  onPress={() => selectMockImage('camera')}
                >
                  <Text style={[s.photoBtnText, { color: theme.text }]}>{t('takePhoto')}</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[s.photoBtn, { backgroundColor: theme.surfaceAlt }]}
                  onPress={() => selectMockImage('gallery')}
                >
                  <Text style={[s.photoBtnText, { color: theme.text }]}>{t('choosePhoto')}</Text>
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
                    <Text style={s.actionBtnText}>{t('diagnoseBtn')}</Text>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            )}

            {/* Diagnosis results card */}
            {diagnosisResult && (
              <View style={s.card}>
                <Text style={s.cardHeader}>{t('diagResult')}</Text>
                
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
                    <Text style={[s.alertText, { color: theme.danger }]}>{t('alertBanned')}</Text>
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
              contentContainerStyle={{ padding: 15 }}
              ref={chatScrollRef}
              onContentSizeChange={() => chatScrollRef.current?.scrollToEnd({ animated: true })}
              showsVerticalScrollIndicator={false}
            >
              {chatMessages.map((msg) => (
                <View key={msg.id} style={[s.chatBubble, msg.isUser ? [s.userBubble, { backgroundColor: accent.main }] : [s.botBubble, { backgroundColor: theme.surface, borderColor: theme.border }]]}>
                  <Text style={[s.chatText, { color: msg.isUser ? '#fff' : theme.text }]}>
                    {msg.text}
                  </Text>
                  
                  {/* Citation chips */}
                  {!msg.isUser && msg.sources?.length > 0 && (
                    <View style={[s.citationRow, { borderTopColor: theme.border }]}>
                      {msg.sources.map((src, i) => (
                        <TouchableOpacity
                          key={i}
                          style={[s.citationChip, { backgroundColor: theme.surfaceAlt }]}
                          onPress={() => setSelectedCitation(src)}
                        >
                          <Text style={[s.citationText, { color: theme.textMuted }]}>[{src.rank}] {src.source_type || 'docs'}</Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  )}
                </View>
              ))}

              {isTyping && (
                <View style={[s.chatBubble, s.botBubble, { width: 60, backgroundColor: theme.surface, borderColor: theme.border }]}>
                  <ActivityIndicator size="small" color={accent.main} />
                </View>
              )}
            </ScrollView>

            {/* Quick action query helpers */}
            <View style={[s.quickBar, { backgroundColor: theme.surface, borderTopColor: theme.border }]}>
              <Text style={[s.quickBarLabel, { color: theme.textMuted }]}>{t('askQuick')}</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => handleQuickQuestion(t('rustHelp'))}>
                  <Text style={[s.quickChipText, { color: accent.softText }]}>{t('rustHelp')}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => handleQuickQuestion(t('pmkisan'))}>
                  <Text style={[s.quickChipText, { color: accent.softText }]}>{t('pmkisan')}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => handleQuickQuestion(t('ureadose'))}>
                  <Text style={[s.quickChipText, { color: accent.softText }]}>{t('ureadose')}</Text>
                </TouchableOpacity>
              </ScrollView>
            </View>

            {/* Chat Input panel */}
            <View style={[s.inputBar, { backgroundColor: theme.surface, borderTopColor: theme.border }]}>
              <TextInput
                style={[s.chatTextInput, { backgroundColor: theme.inputBg, color: theme.text }]}
                placeholder="Ask about fertilizer, pesticides, crop diseases..."
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
                  <Text style={s.sendBtnText}>{t('send')}</Text>
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
                <Text style={[s.formLabel, { color: theme.textMuted }]}>{t('cropLabel')}</Text>
                <TextInput
                  style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                  value={yieldForm.crop}
                  onChangeText={(val) => setYieldForm({ ...yieldForm, crop: val })}
                  placeholder="wheat / rice"
                  placeholderTextColor={theme.placeholder}
                />
              </View>
              
              <View style={s.formRow}>
                <Text style={[s.formLabel, { color: theme.textMuted }]}>{t('districtLabel')}</Text>
                <TextInput
                  style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                  value={yieldForm.district}
                  onChangeText={(val) => setYieldForm({ ...yieldForm, district: val })}
                  placeholder="meerut / jhansi"
                  placeholderTextColor={theme.placeholder}
                />
              </View>

              <View style={s.formRow}>
                <Text style={[s.formLabel, { color: theme.textMuted }]}>{t('areaLabel')}</Text>
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
                    <Text style={s.actionBtnText}>{t('calculateYield')}</Text>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            </View>

            {/* Projections Card */}
            {yieldResult && (
              <View style={s.card}>
                <Text style={s.cardHeader}>Yield Projections</Text>
                
                <View style={[s.yieldGaugeContainer, { borderBottomColor: theme.border }]}>
                  <Text style={[s.yieldValText, { color: accent.main }]}>{yieldResult.total_yield_t?.toFixed(2)} t</Text>
                  <Text style={{ color: theme.textMuted, fontSize: 12 * fontScale }}>
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
              <View style={s.settingsCardHeader}>
                <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 2, paddingBottom: 0 }]}>🎨 {t('appearance')}</Text>
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
              <Text style={s.cardHeader}>🌈 {t('accentColor')}</Text>
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
                      {active && <Text style={s.swatchCheck}>✓</Text>}
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>

            {/* Personalization: font size */}
            <View style={s.card}>
              <Text style={s.cardHeader}>🔠 {t('fontSize')}</Text>
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
              <Text style={s.cardHeader}>📍 {t('locationTitle')}</Text>
              <Text style={[s.settingHint, { color: theme.textMuted }]}>{t('locationHint')}</Text>

              <Text style={[s.formLabel, { color: theme.textMuted, marginTop: 10 }]}>{t('stateLabel')}</Text>
              <TouchableOpacity
                style={[s.districtPicker, { backgroundColor: theme.inputBg, borderColor: theme.border }]}
                onPress={() => openLocationPicker('state')}
              >
                <Text style={[s.districtPickerText, { color: locationInfo.state ? theme.text : theme.placeholder }]}>
                  {locationInfo.state || t('selectState')}
                </Text>
                <Text style={{ color: theme.textMuted }}>▾</Text>
              </TouchableOpacity>

              <Text style={[s.formLabel, { color: theme.textMuted, marginTop: 10 }]}>{t('districtLabel')}</Text>
              <TouchableOpacity
                style={[s.districtPicker, { backgroundColor: theme.inputBg, borderColor: theme.border }]}
                onPress={() => openLocationPicker('district')}
              >
                <Text style={[s.districtPickerText, { color: locationInfo.district ? theme.text : theme.placeholder }]}>
                  {locationInfo.district ? `${locationInfo.district} (${locationInfo.state || ''})` : t('selectDistrict')}
                </Text>
                <Text style={{ color: theme.textMuted }}>▾</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[s.locationBtn, { backgroundColor: accent.soft, borderColor: accent.main }]}
                onPress={detectLocation}
                disabled={locating}
              >
                {locating ? <ActivityIndicator color={accent.main} size="small" /> : (
                  <Text style={[s.locationBtnText, { color: accent.softText }]}>📡 {t('useMyLocation')}</Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Connection */}
            <View style={s.card}>
              <Text style={s.cardHeader}>🔌 {t('serverUrl')}</Text>
              <TextInput
                style={[s.formInput, { backgroundColor: theme.inputBg, borderColor: theme.border, color: theme.text }]}
                value={apiUrl}
                onChangeText={setApiUrl}
                placeholder="http://192.168.1.100:8000"
                placeholderTextColor={theme.placeholder}
              />
              <Text style={{ fontSize: 11 * fontScale, color: theme.textMuted, marginTop: 6 }}>
                Modify to point to your FastAPI server IP on the local network (e.g. 192.168.x.x:8000).
              </Text>
            </View>

            {/* Model info */}
            <View style={s.card}>
              <Text style={[s.cardHeader]}>🧠 Model Mesh Metadata</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted }}>• Text Embedder: BAAI/bge-m3 (1024-dim)</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted }}>• Classification: ViT-Small (Fine-tuned)</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted }}>• Yield Engine: Tabular lightGBM</Text>
              <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted }}>• RAG Index Chunks: 723,439 documents</Text>
            </View>

            {/* Reset */}
            <TouchableOpacity
              style={[s.resetBtn, { borderColor: theme.danger }]}
              onPress={handleResetSettings}
            >
              <Text style={[s.resetBtnText, { color: theme.danger }]}>↺ {t('resetSettings')}</Text>
            </TouchableOpacity>
          </ScrollView>
        )}
      </View>

      {/* Navigation bottom bar */}
      <View style={[s.navBar, { backgroundColor: theme.surface, borderTopColor: theme.border }]}>
        {[
          { key: 'home', icon: '🏠', label: t('home') },
          { key: 'scanner', icon: '📸', label: t('scanner') },
          { key: 'chat', icon: '💬', label: t('chat') },
          { key: 'yield', icon: '📊', label: t('yield') },
          { key: 'settings', icon: '⚙️', label: t('settings') },
        ].map((item) => {
          const isActive = activeTab === item.key;
          return (
            <TouchableOpacity
              key={item.key}
              style={[s.navItem, isActive && [s.navActive, { backgroundColor: accent.soft }]]}
              onPress={() => setActiveTab(item.key)}
            >
              <Text style={s.navIcon}>{item.icon}</Text>
              <Text style={[s.navText, { color: isActive ? accent.softText : theme.textMuted }]}>
                {item.label}
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

      {/* RAG Source Citation Inspector modal */}
      {selectedCitation && (
        <View style={s.modalBg}>
          <View style={[s.modalCard, { backgroundColor: theme.surface, borderColor: theme.border }]}>
            <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, marginBottom: 8 }}>
              [Citation Details] Source: {selectedCitation.source_type}
            </Text>
            <Text style={{ fontSize: 13 * fontScale, color: theme.text, lineHeight: 20 }}>
              {selectedCitation.text}
            </Text>
<TouchableOpacity
                style={[s.modalCloseBtn, { backgroundColor: accent.main }]}
                onPress={() => setSelectedCitation(null)}
              >
                <Text style={{ color: '#fff', fontWeight: 'bold' }}>Close</Text>
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
          <View style={[s.modalCard, { backgroundColor: theme.surface, borderColor: theme.border }]}>
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
              <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main }}>
                📍 {locStep === 'state' ? t('selectState') : t('selectDistrict')}
              </Text>
              {locStep === 'district' && (
                <TouchableOpacity onPress={() => setLocStep('state')}>
                  <Text style={{ color: accent.main, marginLeft: 12, fontWeight: '600', fontSize: 13 * fontScale }}>
                    ‹ {t('backToStates')}
                  </Text>
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
                        <Text style={{ color: accent.main, fontWeight: 'bold' }}>✓</Text>
                      )}
                    </TouchableOpacity>
                  )}
                />
              ) : (
                <Text style={{ color: theme.textMuted, paddingVertical: 12, fontSize: 13 * fontScale }}>
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
                        <Text style={{ color: accent.main, fontWeight: 'bold' }}>✓</Text>
                      )}
                    </TouchableOpacity>
                  )}
                />
              ) : (
                <Text style={{ color: theme.textMuted, paddingVertical: 12, fontSize: 13 * fontScale }}>
                  {t('districtListUnavailable')}
                </Text>
              )
            )}

            <TouchableOpacity
              style={[s.modalCloseBtn, { backgroundColor: accent.main, marginTop: 12 }]}
              onPress={() => { setLocModalOpen(false); setLocStep('state'); }}
            >
              <Text style={{ color: '#fff', fontWeight: 'bold' }}>Close</Text>
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
          <View style={[s.modalCard, { backgroundColor: theme.surface, borderColor: theme.border }]}>
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
              <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main }}>
                🌐 {t('langLabel')}
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
                  <Text style={{ color: accent.main, fontWeight: 'bold' }}>✓</Text>
                )}
              </TouchableOpacity>
            ))}
            <TouchableOpacity
              style={[s.modalCloseBtn, { backgroundColor: accent.main, marginTop: 12 }]}
              onPress={() => setLangModalOpen(false)}
            >
              <Text style={{ color: '#fff', fontWeight: 'bold' }}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
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
      paddingBottom: 16,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    headerTitle: {
      fontSize: 22 * fs,
      fontWeight: '800',
      color: '#ffffff',
      letterSpacing: -0.5
    },
    headerSubtitle: {
      fontSize: 11 * fs,
      color: 'rgba(255,255,255,0.85)',
      fontWeight: '500',
      marginTop: 2
    },
    langPill: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.18)',
      borderRadius: 999,
      paddingHorizontal: 12,
      paddingVertical: 7,
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.35)'
    },
    langText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      color: '#ffffff'
    },
    contentContainer: {
      flex: 1
    },
    scrollContent: {
      padding: 16,
      paddingBottom: 40
    },
    greeting: {
      fontSize: 15 * fs,
      fontWeight: '700',
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
      fontWeight: '700'
    },
    mandiMarket: {
      fontSize: 10 * fs,
      marginTop: 1
    },
    mandiLocation: {
      fontSize: 11 * fs,
      marginTop: -6,
      marginBottom: 10,
      fontWeight: '500'
    },
    districtPicker: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderWidth: 1,
      borderRadius: 10,
      paddingHorizontal: 12,
      paddingVertical: 11,
      marginTop: 6
    },
    districtPickerText: {
      fontSize: 14 * fs
    },
    locationBtn: {
      borderWidth: 1,
      borderRadius: 10,
      paddingVertical: 11,
      alignItems: 'center',
      marginTop: 14
    },
    locationBtnText: {
      fontSize: 13 * fs,
      fontWeight: '600'
    },
    districtRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingVertical: 12,
      borderBottomWidth: StyleSheet.hairlineWidth
    },
    districtRowText: {
      fontSize: 15 * fs
    },
    card: {
      backgroundColor: theme.surface,
      borderRadius: 18,
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
      color: '#fff'
    },
    weatherDesc: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 13 * fs
    },
    weatherLabel: {
      color: 'rgba(255,255,255,0.8)',
      fontSize: 11 * fs
    },
    weatherVal: {
      color: '#fff',
      fontWeight: 'bold',
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
      borderRadius: 12,
      paddingVertical: 8,
      paddingHorizontal: 6,
      alignItems: 'center'
    },
    weatherForecastDay: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 10 * fs,
      fontWeight: '700',
      marginBottom: 3
    },
    weatherForecastTemp: {
      color: '#fff',
      fontSize: 11 * fs,
      fontWeight: '600'
    },
    advisoryBanner: {
      marginTop: 12,
      paddingTop: 12,
      borderTopWidth: 1,
      borderTopColor: 'rgba(255,255,255,0.2)',
      color: '#fff',
      fontSize: 12 * fs,
      lineHeight: 18 * fs
    },
    mandiRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: 11
    },
    mandiCrop: {
      fontWeight: '600',
      color: theme.text,
      fontSize: 14 * fs
    },
    mandiPrice: {
      fontWeight: 'bold',
      color: theme.text,
      fontSize: 14 * fs
    },
    mandiChange: {
      fontSize: 11 * fs,
      fontWeight: '600'
    },
    mandiLocationRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderRadius: 12,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      paddingHorizontal: 12,
      paddingVertical: 9,
      marginBottom: 12
    },
    mandiLocationEdit: {
      fontSize: 11 * fs,
      fontWeight: '700'
    },
    mandiScroller: {
      paddingRight: 8
    },
    mandiCard: {
      width: 150,
      borderRadius: 14,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      padding: 12,
      marginRight: 10
    },
    shortcutsGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginTop: 5
    },
    shortcutBtn: {
      width: '31%',
      borderRadius: 18,
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
      color: theme.text,
      textAlign: 'center'
    },
    photoSelectorContainer: {
      borderRadius: 18,
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
      fontSize: 12 * fs
    },
    selectedLeafImage: {
      width: '100%',
      height: 200,
      borderRadius: 12,
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
      borderRadius: 12
    },
    photoBtnText: {
      fontSize: 12 * fs,
      fontWeight: '600'
    },
    actionBtn: {
      borderRadius: 16,
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
      fontSize: 14 * fs
    },
    cardHeader: {
      fontWeight: 'bold',
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
      fontSize: 11 * fs
    },
    alertCard: {
      borderWidth: 1,
      borderRadius: 10,
      padding: 10,
      marginBottom: 12
    },
    alertText: {
      fontSize: 12 * fs,
      fontWeight: '600'
    },
    answerText: {
      fontSize: 14 * fs,
      lineHeight: 20 * fs
    },
    chatBubble: {
      maxWidth: '80%',
      padding: 13,
      borderRadius: 18,
      marginBottom: 12,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 2
    },
    userBubble: {
      alignSelf: 'flex-end',
      borderBottomRightRadius: 4
    },
    botBubble: {
      alignSelf: 'flex-start',
      borderBottomLeftRadius: 4,
      borderWidth: 1
    },
    chatText: {
      fontSize: 14 * fs,
      lineHeight: 20 * fs
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
      fontWeight: '500'
    },
    quickBar: {
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderTopWidth: 1
    },
    quickBarLabel: {
      fontSize: 11 * fs,
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
      fontWeight: '600'
    },
    inputBar: {
      flexDirection: 'row',
      padding: 12,
      borderTopWidth: 1,
      alignItems: 'center'
    },
    chatTextInput: {
      flex: 1,
      height: 42,
      borderRadius: 21,
      paddingHorizontal: 16,
      fontSize: 14 * fs
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
      paddingVertical: 11,
      paddingHorizontal: 18
    },
    sendBtnText: {
      color: '#fff',
      fontWeight: 'bold',
      fontSize: 13 * fs
    },
    formRow: {
      marginBottom: 12
    },
    formLabel: {
      fontSize: 13 * fs,
      fontWeight: '600',
      marginBottom: 6
    },
    formInput: {
      height: 44,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderRadius: 12,
      paddingHorizontal: 12,
      fontSize: 14 * fs
    },
    yieldGaugeContainer: {
      alignItems: 'center',
      paddingVertical: 20,
      borderBottomWidth: StyleSheet.hairlineWidth,
      marginBottom: 15
    },
    yieldValText: {
      fontSize: 34 * fs,
      fontWeight: '800'
    },
    economicsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12
    },
    econItem: {
      width: '46%',
      borderRadius: 12,
      padding: 10,
      borderWidth: themeMode === 'highContrast' ? 2 : 1
    },
    econLabel: {
      fontSize: 11 * fs,
      marginBottom: 4
    },
    econVal: {
      fontSize: 14 * fs,
      fontWeight: '600'
    },
    navBar: {
      height: 64,
      borderTopWidth: 1,
      flexDirection: 'row',
      justifyContent: 'space-around',
      alignItems: 'center',
      paddingBottom: 6
    },
    navItem: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 6,
      paddingHorizontal: 10,
      borderRadius: 16,
      minWidth: 56
    },
    navActive: {
      borderRadius: 16
    },
    navIcon: {
      fontSize: 19,
      marginBottom: 2
    },
    navText: {
      fontSize: 10 * fs,
      fontWeight: '600'
    },
    settingsCardHeader: {
      flexDirection: 'row',
      alignItems: 'center'
    },
    settingHint: {
      fontSize: 11.5 * fs,
      lineHeight: 16 * fs
    },
    themeRow: {
      flexDirection: 'row',
      gap: 10,
      marginTop: 10
    },
    themeChip: {
      flex: 1,
      borderRadius: 14,
      borderWidth: 2,
      padding: 8,
      alignItems: 'center'
    },
    themePreview: {
      width: '100%',
      height: 48,
      borderRadius: 8,
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
      textAlign: 'center'
    },
    accentRow: {
      flexDirection: 'row',
      gap: 14,
      marginTop: 14,
      justifyContent: 'center'
    },
    swatch: {
      width: 40,
      height: 40,
      borderRadius: 20,
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
      borderRadius: 14,
      borderWidth: 2,
      paddingVertical: 12,
      alignItems: 'center'
    },
    fontSizeChipText: {
      fontWeight: '800',
      marginBottom: 2
    },
    fontSizeChipLabel: {
      fontSize: 10 * fs,
      fontWeight: '600'
    },
    resetBtn: {
      borderWidth: 1.5,
      borderRadius: 14,
      padding: 13,
      alignItems: 'center',
      marginBottom: 30
    },
    resetBtnText: {
      fontWeight: '700',
      fontSize: 13 * fs
    },
    toast: {
      position: 'absolute',
      bottom: 80,
      left: 20,
      right: 20,
      borderRadius: 14,
      borderWidth: 1,
      padding: 12,
      alignItems: 'center',
      zIndex: 100
    },
    toastText: {
      fontSize: 12 * fs,
      fontWeight: '600'
    },
    modalBg: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.6)',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 9999
    },
    modalCard: {
      width: '85%',
      borderRadius: 18,
      padding: 20,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.2,
      shadowRadius: 10,
      elevation: 6
    },
    modalCloseBtn: {
      marginTop: 15,
      padding: 11,
      borderRadius: 12,
      alignItems: 'center'
    }
  });
};
