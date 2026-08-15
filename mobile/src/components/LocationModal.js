import React from 'react';
import { ActivityIndicator, FlatList, Modal, Text, TouchableOpacity, View } from 'react-native';
import Ionicons from '@react-native-vector-icons/ionicons';
import { FONT } from '../theme';

// Location picker modal (State -> District drill-down)
export default function LocationModal({
  visible,
  step,
  states,
  districts,
  districtsLoading,
  locationInfo,
  onChooseState,
  onChooseDistrict,
  onBack,
  onClose,
  t,
  s,
  theme,
  accent,
  fontScale,
  insets,
}) {
  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={s.modalBg}>
        <View style={[s.sheet, { backgroundColor: theme.surface, paddingBottom: 34 + insets.bottom }]}>
          <View style={s.sheetHandle} />
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 8 }}>
            <Ionicons name="location-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
            <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
              {step === 'state' ? t('selectState') : t('selectDistrict')}
            </Text>
            {step === 'district' && (
              <TouchableOpacity onPress={onBack}>
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  <Ionicons name="chevron-back" size={14} color={accent.main} />
                  <Text style={{ color: accent.main, marginLeft: 2, fontWeight: '600', fontSize: 13 * fontScale, fontFamily: FONT.semibold }}>
                    {t('backToStates')}
                  </Text>
                </View>
              </TouchableOpacity>
            )}
          </View>

          {step === 'state' ? (
            states.length > 0 ? (
              <FlatList
                data={states}
                keyExtractor={(item) => item}
                style={{ maxHeight: 320 }}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[s.districtRow, { borderBottomColor: theme.border }]}
                    onPress={() => onChooseState(item)}
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
                    onPress={() => onChooseDistrict(item)}
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
            onPress={onClose}
          >
            <Text style={{ color: '#fff', fontWeight: 'bold', fontFamily: FONT.bold }}>Close</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}