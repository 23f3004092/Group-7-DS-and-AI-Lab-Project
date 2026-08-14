// Metro config for Expo web: expo-sqlite (kv-store) ships a WASM bundle loader.
// Without this, web builds fail with "Unable to resolve ./wa-sqlite/wa-sqlite.wasm".
// Docs: https://docs.expo.dev/versions/v57.0.0/sdk/sqlite/
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Support `.wasm` asset files (used by expo-sqlite's web build)
config.resolver.assetExts.push('wasm');

module.exports = config;