import React from 'react';
import { RefreshControl, ScrollView, Text, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';
import MaterialCommunityIcons from '@react-native-vector-icons/material-design-icons';
import { FONT } from '../theme';
import { greeting, weatherIcon, fmtDay } from '../helpers';

export default function HomeScreen({
  weather,
  weatherAdvisory,
  mandiPrices,
  mandiSource,
  locationInfo,
  refreshing,
  onRefresh,
  onOpenLocationPicker,
  onOpenWeatherDetail,
  onNavigate,
  t,
  s,
  theme,
  accent,
  fontScale,
}) {
  return (
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
      <Text style={s.greeting}>{greeting(t)}</Text>

      {/* Weather & Advisory widget (tap for full weather detail view) */}
      <TouchableOpacity
        activeOpacity={0.92}
        onPress={onOpenWeatherDetail}
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
            <Text style={s.advisoryBanner} numberOfLines={3}>
              🌾 {weatherAdvisory?.advisory || 'Advisory: Ideal conditions for Rabi crop fertilization. Monitor wheat leaves for rust flags.'}
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
        onPress={onOpenLocationPicker}
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
          onPress={() => onNavigate('scanner')}
        >
          <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
            <Ionicons name="scan-outline" size={24} color={accent.softText} />
          </View>
          <Text style={s.shortcutText}>{t('scanner')}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
          onPress={() => onNavigate('chat')}
        >
          <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
            <Ionicons name="chatbubble-ellipses-outline" size={24} color={accent.softText} />
          </View>
          <Text style={s.shortcutText}>{t('chat')}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[s.shortcutBtn, { backgroundColor: theme.surface, borderColor: theme.border }]}
          onPress={() => onNavigate('yield')}
        >
          <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft }]}>
            <Ionicons name="stats-chart-outline" size={24} color={accent.softText} />
          </View>
          <Text style={s.shortcutText}>{t('yield')}</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}