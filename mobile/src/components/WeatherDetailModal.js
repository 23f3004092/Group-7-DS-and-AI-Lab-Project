import React from 'react';
import { Modal, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';
import { FONT } from '../theme';
import { weatherIcon, renderIcon, fmtTime, fmtDay, buildAdvisory } from '../helpers';

const statItem = (t, s, theme, accent, labelKey, icon, valueRaw, unit) => (
  <View style={[s.statItem, { backgroundColor: theme.surfaceAlt, borderColor: theme.border }]}>
    <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 6 }}>
      {renderIcon(icon, 15, accent.softText, { marginRight: 6 })}
      <Text style={[s.statLabel, { color: theme.textMuted }]}>{t(labelKey)}</Text>
    </View>
    <Text style={[s.statValue, { color: theme.text }]}>{valueRaw != null ? `${valueRaw}${unit || ''}` : '—'}</Text>
  </View>
);

const buildStatGrid = (t, s, theme, accent, weather) => [
  statItem(t, s, theme, accent, 'humidityLabel', 'water-outline', weather.humidity, ''),
  statItem(t, s, theme, accent, 'rainfallLabel', 'umbrella-outline', weather.rain, ''),
  statItem(t, s, theme, accent, 'windLabel', 'windy', weather.windKmh, weather.windKmh != null ? ' km/h' : ''),
  statItem(t, s, theme, accent, 'windDirLabel', 'navigate-outline', weather.windDirLabel, weather.windKmh != null ? ` (${weather.windDirDeg}°)` : ''),
  statItem(t, s, theme, accent, 'uvLabel', 'sunny-outline', weather.uvIndex, weather.uvIndex != null ? '' : ''),
  statItem(t, s, theme, accent, 'pressureLabel', 'speedometer-outline', weather.pressure, weather.pressure != null ? ' hPa' : ''),
  statItem(t, s, theme, accent, 'dewPointLabel', 'thermometer-outline', weather.dewPoint, weather.dewPoint != null ? '°C' : ''),
  statItem(t, s, theme, accent, 'cloudLabel', 'cloud-outline', weather.cloud, weather.cloud != null ? '%' : ''),
];

// Full-screen weather detail view
export default function WeatherDetailModal({ visible, weather, onClose, t, s, theme, accent, fontScale, insets }) {
  return (
    <Modal
      visible={visible}
      animationType="slide"
      onRequestClose={onClose}
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
              onPress={onClose}
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
            {buildStatGrid(t, s, theme, accent, weather).map((item, i) => (
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
                    {fmtDay(f.date)}
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
              {buildAdvisory(weather).map((tip, i) => (
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
  );
}