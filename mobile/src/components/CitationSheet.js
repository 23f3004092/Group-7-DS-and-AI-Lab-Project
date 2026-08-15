import React from 'react';
import { Text, TouchableOpacity, View } from 'react-native';
import Ionicons from '@react-native-vector-icons/ionicons';
import { FONT } from '../theme';

// RAG Source Citation Inspector (bottom sheet)
export default function CitationSheet({ citation, onClose, s, theme, accent, fontScale, insets }) {
  return (
    <View style={s.modalBg}>
      <View style={[s.sheet, { backgroundColor: theme.surface, paddingBottom: 34 + insets.bottom }]}>
        <View style={s.sheetHandle} />
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
          <Ionicons name="document-text-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
          <Text style={{ fontWeight: 'bold', fontSize: 15 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
            Source: {citation.source_type}
          </Text>
        </View>
        <Text style={{ fontSize: 13 * fontScale, color: theme.text, lineHeight: 20, fontFamily: FONT.regular }}>
          {citation.text}
        </Text>
        <TouchableOpacity
          style={[s.modalCloseBtn, { backgroundColor: accent.main }]}
          onPress={onClose}
        >
          <Text style={{ color: '#fff', fontWeight: 'bold', fontFamily: FONT.bold }}>Close</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}