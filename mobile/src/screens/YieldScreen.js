import React from 'react';
import { ActivityIndicator, ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';
import { FONT } from '../theme';

export default function YieldScreen({
  yieldForm,
  onFormChange,
  yieldResult,
  estimating,
  onEstimate,
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
            onChangeText={(val) => onFormChange('crop', val)}
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
            onChangeText={(val) => onFormChange('district', val)}
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
            onChangeText={(val) => onFormChange('area', val)}
            keyboardType="numeric"
            placeholder="e.g. 2.5"
            placeholderTextColor={theme.placeholder}
          />
        </View>

        <TouchableOpacity
          style={[s.actionBtn, estimating ? { opacity: 0.7 } : {}]}
          onPress={onEstimate}
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
  );
}