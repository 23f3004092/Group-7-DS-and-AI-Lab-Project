import React from 'react';
import { ActivityIndicator, ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import Ionicons from '@react-native-vector-icons/ionicons';
import { ACCENTS, BASE_THEMES, FONT_SCALES, FONT } from '../theme';

export default function SettingsScreen({
  themeMode,
  accentKey,
  fontSize,
  locationInfo,
  locating,
  onUpdateSettings,
  onOpenLocationPicker,
  onDetectLocation,
  apiUrl,
  onApiUrlChange,
  aiStatus,
  aiChecking,
  onCheckAi,
  onResetSettings,
  aiConfigured,
  aiBaseUrl,
  t,
  s,
  theme,
  accent,
  fontScale,
}) {
  return (
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
                onPress={() => onUpdateSettings({ theme: Object.keys(BASE_THEMES).find(k => BASE_THEMES[k] === th) })}
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
                onPress={() => onUpdateSettings({ accent: key })}
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
                onPress={() => onUpdateSettings({ fontSize: key })}
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
          onPress={() => onOpenLocationPicker('state')}
        >
          <Text style={[s.districtPickerText, { color: locationInfo.state ? theme.text : theme.placeholder }]}>
            {locationInfo.state || t('selectState')}
          </Text>
          <Ionicons name="chevron-down" size={16} color={theme.textMuted} />
        </TouchableOpacity>

        <Text style={[s.formLabel, { color: theme.textMuted, marginTop: 10 }]}>{t('districtLabel')}</Text>
        <TouchableOpacity
          style={[s.districtPicker, { backgroundColor: theme.inputBg, borderColor: theme.border }]}
          onPress={() => onOpenLocationPicker('district')}
        >
          <Text style={[s.districtPickerText, { color: locationInfo.district ? theme.text : theme.placeholder }]}>
            {locationInfo.district ? `${locationInfo.district} (${locationInfo.state || ''})` : t('selectDistrict')}
          </Text>
          <Ionicons name="chevron-down" size={16} color={theme.textMuted} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[s.locationBtn, { backgroundColor: accent.soft, borderColor: accent.main }]}
          onPress={onDetectLocation}
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
          onChangeText={onApiUrlChange}
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
          <View style={[s.aiStatusDot, { backgroundColor: !aiConfigured ? theme.placeholder : (aiStatus ? (aiStatus.ok ? '#22c55e' : '#ef4444') : '#f59e0b') }]} />
        </View>
        <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>
          Endpoint: {aiBaseUrl || 'not set — add EXPO_PUBLIC_AI_API_URL to mobile/.env'}
        </Text>
        <Text style={{ fontSize: 12 * fontScale, color: theme.textMuted, fontFamily: FONT.medium }}>
          API key: {aiConfigured ? 'configured' : 'not set — add EXPO_PUBLIC_AI_API_KEY to mobile/.env'}
        </Text>
        {aiStatus && (
          <Text style={{ fontSize: 12 * fontScale, color: aiStatus.ok ? '#22c55e' : theme.danger, fontFamily: FONT.medium, marginTop: 2 }}>
            {aiStatus.ok ? '● ' : '● '}{aiStatus.text}
          </Text>
        )}
        <TouchableOpacity
          style={[s.resetBtn, { borderColor: accent.main, marginTop: 8 }]}
          onPress={onCheckAi}
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
        onPress={onResetSettings}
      >
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <Ionicons name="refresh-outline" size={17} color={theme.danger} style={{ marginRight: 8 }} />
          <Text style={[s.resetBtnText, { color: theme.danger }]}>{t('resetSettings')}</Text>
        </View>
      </TouchableOpacity>
    </ScrollView>
  );
}