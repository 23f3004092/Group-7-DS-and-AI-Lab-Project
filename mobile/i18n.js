import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { getLocales } from 'expo-localization';
import Storage from 'expo-sqlite/kv-store';

import en from './locales/en.json';
import hi from './locales/hi.json';
import ta from './locales/ta.json';
import te from './locales/te.json';
import kn from './locales/kn.json';
import ml from './locales/ml.json';
import mr from './locales/mr.json';
import gu from './locales/gu.json';
import bn from './locales/bn.json';
import pa from './locales/pa.json';
import or from './locales/or.json';

const LANGUAGE_KEY = 'farmervision.language';

export const SUPPORTED_LANGUAGES = ['en', 'hi', 'ta', 'te', 'kn', 'ml', 'mr', 'gu', 'bn', 'pa', 'or'];

/** Native-script display names, used in the language picker. */
export const LANG_NAMES = {
  en: 'English',
  hi: 'हिन्दी',
  ta: 'தமிழ்',
  te: 'తెలుగు',
  kn: 'ಕನ್ನಡ',
  ml: 'മലയാളം',
  mr: 'मराठी',
  gu: 'ગુજરાતી',
  bn: 'বাংলা',
  pa: 'ਪੰਜਾਬੀ',
  or: 'ଓଡ଼ିଆ',
};

/**
 * Detect the best initial language:
 * 1. A language the farmer explicitly chose (persisted in kv-store)
 * 2. The device language (e.g. a phone set to Hindi auto-starts in Hindi)
 * 3. English as a safe fallback
 */
export async function getInitialLanguage() {
  try {
    const saved = await Storage.getItem(LANGUAGE_KEY);
    if (saved && SUPPORTED_LANGUAGES.includes(saved)) return saved;
  } catch (e) {
    // Storage unavailable -> fall through to device detection
  }
  try {
    const device = getLocales?.()?.[0]?.languageCode;
    if (device && SUPPORTED_LANGUAGES.includes(device)) return device;
  } catch (e) {
    // Locale API unavailable
  }
  return 'en';
}

export async function initI18n() {
  const lng = await getInitialLanguage();
  if (!i18n.isInitialized) {
    await i18n.use(initReactI18next).init({
      resources: {
        en: { translation: en },
        hi: { translation: hi },
        ta: { translation: ta },
        te: { translation: te },
        kn: { translation: kn },
        ml: { translation: ml },
        mr: { translation: mr },
        gu: { translation: gu },
        bn: { translation: bn },
        pa: { translation: pa },
        or: { translation: or },
      },
      lng,
      fallbackLng: 'en',
      interpolation: { escapeValue: false },
    });
  }
  return i18n;
}

/** Switch language at runtime and persist the farmer's choice. */
export async function setLanguage(lng) {
  if (!SUPPORTED_LANGUAGES.includes(lng)) return;
  await i18n.changeLanguage(lng);
  try {
    await Storage.setItem(LANGUAGE_KEY, lng);
  } catch (e) {
    // Persistence unavailable -> language still applies for this session
  }
}

export default i18n;
