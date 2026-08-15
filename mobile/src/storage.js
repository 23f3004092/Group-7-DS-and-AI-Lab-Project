import Storage from 'expo-sqlite/kv-store';
import { DEFAULT_SETTINGS, SETTINGS_KEY } from './config';

export async function loadSettings() {
  try {
    const raw = await Storage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch (e) {
    // Storage unavailable (e.g. web without wasm setup) -> fall back to defaults
  }
  return DEFAULT_SETTINGS;
}

export async function saveSettings(settings) {
  try {
    await Storage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (e) {
    // Ignore persistence failures; settings still apply for this session
  }
}