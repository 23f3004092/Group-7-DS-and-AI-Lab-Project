import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Text,
  View,
  ActivityIndicator,
  Alert,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import * as Location from 'expo-location';
import * as ImagePicker from 'expo-image-picker';
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
  appendImage,
} from './aiClient';

// Shared modules: theme system, app config, persistence, display helpers, data services
import { FONT, ACCENTS, BASE_THEMES, FONT_SCALES, createStyles } from './src/theme';
import { DEFAULT_API_URL, DEFAULT_SETTINGS } from './src/config';
import { loadSettings, saveSettings } from './src/storage';
import {
  fetchMandiRows,
  fetchWeatherSnapshot,
  fetchYieldFact,
} from './src/services';
import Header from './src/components/Header';
import NavBar from './src/components/NavBar';
import CitationSheet from './src/components/CitationSheet';
import LocationModal from './src/components/LocationModal';
import LanguageModal from './src/components/LanguageModal';
import WeatherDetailModal from './src/components/WeatherDetailModal';
import HomeScreen from './src/screens/HomeScreen';
import ScannerScreen from './src/screens/ScannerScreen';
import ChatScreen from './src/screens/ChatScreen';
import YieldScreen from './src/screens/YieldScreen';
import SettingsScreen from './src/screens/SettingsScreen';

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
  const [mandiPrices, setMandiPrices] = useState([]);
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

  // Assemble /query live_data from what /classify suggested. When the held
  // snapshot isn't live yet, refresh it from the backend first.
  const buildLiveData = async (userText, suggestedExternal) => {
    const liveData = {};
    const want = (key) => !Array.isArray(suggestedExternal) || suggestedExternal.includes(key);

    if (want('mandi_prices')) {
      let rows = mandiSource === 'live' ? mandiPrices : null;
      if (!rows) {
        const result = await fetchMandiRows(apiUrl, locationInfo);
        if (result) rows = result.rows;
      }
      if (rows?.length) {
        liveData.mandi_prices = rows.map(p => `${p.crop}: ${p.price}`).join('; ');
      }
    }
    if (want('weather')) {
      let w = weather.source === 'live' ? weather : null;
      if (!w) w = await fetchWeatherSnapshot(apiUrl, locationInfo);
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
    if (want('yield')) {
      const fact = await fetchYieldFact(userText, apiUrl, locationInfo);
      if (fact) liveData.yield = fact;
    }
    return Object.keys(liveData).length ? liveData : undefined;
  };

  // Live weather from backend /api/weather/current (GPS coords preferred, else district city)
  useEffect(() => {
    if (!apiUrl) return;
    let cancelled = false;
    (async () => {
      const w = await fetchWeatherSnapshot(apiUrl, locationInfo);
      if (!cancelled && w) setWeather(w);
    })();
    return () => { cancelled = true; };
  }, [apiUrl, locationInfo, reloadKey]);

  // Live mandi prices from backend /api/mandi/prices (falls back to static MSP list)
  useEffect(() => {
    if (!apiUrl) return;
    let cancelled = false;
    (async () => {
      const result = await fetchMandiRows(apiUrl, locationInfo);
      if (!cancelled && result) {
        setMandiPrices(result.rows);
        setMandiSource(result.source);
      }
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
            const liveData = await buildLiveData(userText, suggestedExternal);
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
        await appendImage(formData, { uri: attachedImage.uri, name: attachedImage.name });
        const res = await fetch(`${apiUrl}/api/query/image`, {
          method: 'POST',
          body: formData,
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
        body: JSON.stringify({
          text: userText,
          state: locationInfo.state || undefined,
          district: locationInfo.district || undefined,
          lat: locationInfo.lat || undefined,
          lon: locationInfo.lon || undefined,
        })
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

    // Prepare FormData (Note: multipart = do NOT set Content-Type manually; fetch adds the boundary)
    const formData = new FormData();
    await appendImage(formData, { uri: selectedImage.uri, name: selectedImage.name, type: 'image/jpeg' });

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

      {/* App header (language pill + title) */}
      <Header
        insets={insets}
        langName={LANG_NAMES[i18n.language]}
        onOpenLang={() => setLangModalOpen(true)}
        t={t}
        s={s}
        theme={theme}
      />

      {/* Main Tab Screens */}
      <View style={s.contentContainer}>
        {activeTab === 'home' && (
          <HomeScreen
            weather={weather}
            mandiPrices={mandiPrices}
            mandiSource={mandiSource}
            locationInfo={locationInfo}
            refreshing={refreshing}
            onRefresh={onRefresh}
            onOpenLocationPicker={() => openLocationPicker('state')}
            onOpenWeatherDetail={() => setWeatherDetailOpen(true)}
            onNavigate={setActiveTab}
            t={t}
            s={s}
            theme={theme}
            accent={accent}
            fontScale={fontScale}
          />
        )}

        {activeTab === 'scanner' && (
          <ScannerScreen
            selectedImage={selectedImage}
            uploading={uploading}
            diagnosisResult={diagnosisResult}
            onTakePhoto={pickScannerPhotoFromCamera}
            onChoosePhoto={pickScannerPhotoFromLibrary}
            onDiagnose={handleDiagnose}
            t={t}
            s={s}
            theme={theme}
            accent={accent}
            themeMode={themeMode}
          />
        )}

        {activeTab === 'chat' && (
          <ChatScreen
            chatMessages={chatMessages}
            isTyping={isTyping}
            chatInput={chatInput}
            chatImage={chatImage}
            chatScrollRef={chatScrollRef}
            onInputChange={setChatInput}
            onSend={handleSendChat}
            onAttach={pickChatImage}
            onClearImage={() => setChatImage(null)}
            onQuickQuestion={handleQuickQuestion}
            onSelectCitation={setSelectedCitation}
            t={t}
            s={s}
            theme={theme}
            accent={accent}
          />
        )}

        {activeTab === 'yield' && (
          <YieldScreen
            yieldForm={yieldForm}
            onFormChange={(field, value) => setYieldForm(f => ({ ...f, [field]: value }))}
            yieldResult={yieldResult}
            estimating={estimating}
            onEstimate={handleEstimateYield}
            t={t}
            s={s}
            theme={theme}
            accent={accent}
            fontScale={fontScale}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsScreen
            themeMode={themeMode}
            accentKey={accentKey}
            fontSize={fontSize}
            locationInfo={locationInfo}
            locating={locating}
            onUpdateSettings={updateSettings}
            onOpenLocationPicker={openLocationPicker}
            onDetectLocation={detectLocation}
            apiUrl={apiUrl}
            onApiUrlChange={setApiUrl}
            aiStatus={aiStatus}
            aiChecking={aiChecking}
            onCheckAi={checkAiConnection}
            onResetSettings={handleResetSettings}
            aiConfigured={AI_CONFIGURED}
            aiBaseUrl={AI_BASE_URL_VALUE}
            t={t}
            s={s}
            theme={theme}
            accent={accent}
            fontScale={fontScale}
          />
        )}
      </View>

      {/* Floating bottom navigation bar (lifted above the home-indicator/gesture bar) */}
      <NavBar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        t={t}
        s={s}
        theme={theme}
        accent={accent}
        insets={insets}
      />

      {/* Toast */}
      {toast && (
        <View style={[s.toast, { backgroundColor: theme.surface, borderColor: accent.main }]}>
          <Text style={[s.toastText, { color: theme.text }]}>{toast}</Text>
        </View>
      )}

      {/* RAG Source Citation Inspector modal (bottom sheet) */}
      {selectedCitation && (
        <CitationSheet
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
          s={s}
          theme={theme}
          accent={accent}
          fontScale={fontScale}
          insets={insets}
        />
      )}

      {/* Location picker modal (State -> District drill-down) */}
      <LocationModal
        visible={locModalOpen}
        step={locStep}
        states={states}
        districts={districts}
        districtsLoading={districtsLoading}
        locationInfo={locationInfo}
        onChooseState={chooseStateForLoc}
        onChooseDistrict={chooseDistrictForLoc}
        onBack={() => setLocStep('state')}
        onClose={() => { setLocModalOpen(false); setLocStep('state'); }}
        t={t}
        s={s}
        theme={theme}
        accent={accent}
        fontScale={fontScale}
        insets={insets}
      />

      {/* Language picker modal */}
      <LanguageModal
        visible={langModalOpen}
        currentLang={i18n.language}
        onSelect={(code) => { setLanguage(code); setLangModalOpen(false); }}
        onClose={() => setLangModalOpen(false)}
        t={t}
        s={s}
        theme={theme}
        accent={accent}
        fontScale={fontScale}
        insets={insets}
      />

      {/* Full-screen Weather Detail view */}
      <WeatherDetailModal
        visible={weatherDetailOpen}
        weather={weather}
        onClose={() => setWeatherDetailOpen(false)}
        t={t}
        s={s}
        theme={theme}
        accent={accent}
        fontScale={fontScale}
        insets={insets}
      />
    </KeyboardAvoidingView>
  );
}
