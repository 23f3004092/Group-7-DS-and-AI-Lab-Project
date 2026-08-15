import React from 'react';
import { Modal, Text, TouchableOpacity, View } from 'react-native';
import Ionicons from '@react-native-vector-icons/ionicons';
import { SUPPORTED_LANGUAGES, LANG_NAMES } from '../../i18n';
import { FONT } from '../theme';

// Language picker modal
export default function LanguageModal({ visible, currentLang, onSelect, onClose, t, s, theme, accent, fontScale, insets }) {
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
            <Ionicons name="globe-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
            <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
              {t('langLabel')}
            </Text>
          </View>
          {SUPPORTED_LANGUAGES.map((code) => (
            <TouchableOpacity
              key={code}
              style={[s.districtRow, { borderBottomColor: theme.border }]}
              onPress={() => onSelect(code)}
            >
              <Text style={[s.districtRowText, { color: theme.text }]}>{LANG_NAMES[code]}</Text>
              {currentLang === code && (
                <Ionicons name="checkmark-circle" size={18} color={accent.main} />
              )}
            </TouchableOpacity>
          ))}
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