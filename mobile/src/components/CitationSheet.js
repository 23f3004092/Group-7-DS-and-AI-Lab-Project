import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Modal, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';
import { FONT } from '../theme';

// Full-screen RAG source page: tap a citation chip to read the full advisory content.
// When the chip carries a backend chunk id, the full text is fetched from
// /api/query/source/{id} so the page shows the complete advisory, not the preview.
export default function CitationSheet({ visible, citation, apiUrl, onClose, s, theme, accent, fontScale, insets }) {
  const [fullText, setFullText] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!visible || !citation) return;
    let cancelled = false;

    if (citation.full_text && citation.text !== citation.full_text) {
      setFullText(citation.full_text);
      return () => { cancelled = true; };
    }
    if (!citation.id || !apiUrl) {
      setFullText(null);
      return undefined;
    }

    setLoading(true);
    (async () => {
      try {
        const res = await fetch(`${apiUrl}/api/query/source/${encodeURIComponent(citation.id)}`);
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) setFullText(data.full_text || data.text || null);
        }
      } catch (e) {
        if (!cancelled) setFullText(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [visible, citation, apiUrl]);

  if (!citation) return null;

  const meta = citation.citation || {};
  const metaRows = [
    ['Type', citation.source_type],
    ['Rank', citation.rank != null ? `#${citation.rank}` : null],
    ['Score', citation.score != null ? Number(citation.score).toFixed(3) : null],
    ['Crop', meta.crop],
    ['District', meta.district],
    ['Season', meta.season],
    ['Year', meta.year],
    ['Pages', meta.pages ? meta.pages.join('-') : null],
    ['Section', meta.section],
    ['Category', meta.doc_category],
    ['Query', meta.query_type],
  ].filter(([, v]) => v != null && v !== '');

  const title = citation.name ? `Source: ${citation.name}` : 'Source Detail';
  const content = fullText || citation.text || citation.source_type || 'No additional content.';

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <View style={[s.container, { backgroundColor: theme.bg }]}>
        <LinearGradient
          colors={[accent.main, accent.strong]}
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
            <View style={{ flex: 1, marginHorizontal: 12 }}>
              <Text style={s.weatherDetailTitle} numberOfLines={1}>{title}</Text>
            </View>
            <View style={{ width: 34 }} />
          </View>
          <View style={{ paddingTop: 10 }}>
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 4 }}>
              <Ionicons name="document-text-outline" size={16} color="#fff" style={{ marginRight: 6 }} />
              <Text style={{ color: '#fff', fontSize: 13 * fontScale, fontFamily: FONT.bold }}>
                {citation.source_type || 'Advisory content'}
              </Text>
            </View>
          </View>
        </LinearGradient>

        <ScrollView contentContainerStyle={{ padding: 16, paddingBottom: 40 }} showsVerticalScrollIndicator={false}>
          {/* Metadata rows */}
          {metaRows.length > 0 && (
            <View style={[s.card, { marginBottom: 14 }]}>
              {metaRows.map(([label, value], i) => (
                <View
                  key={label}
                  style={[
                    { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 8 },
                    i > 0 && { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.border },
                  ]}
                >
                  <Text style={{ fontSize: 13 * fontScale, color: theme.textMuted, fontFamily: FONT.regular }}>
                    {label}
                  </Text>
                  <Text style={{ fontSize: 13 * fontScale, color: theme.text, fontFamily: FONT.semibold, flexShrink: 1, marginLeft: 10, textAlign: 'right' }}>
                    {value}
                  </Text>
                </View>
              ))}
            </View>
          )}

          {/* Full advisory content */}
          <View style={s.card}>
            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
              <Ionicons name="leaf-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
              <Text style={{ fontSize: 14 * fontScale, color: accent.main, fontFamily: FONT.bold }}>
                Advisory Content
              </Text>
            </View>
            {loading ? (
              <ActivityIndicator size="small" color={accent.main} />
            ) : (
              <Text style={{ fontSize: 14 * fontScale, color: theme.text, lineHeight: 22, fontFamily: FONT.regular }}>
                {content}
              </Text>
            )}
          </View>
        </ScrollView>
      </View>
    </Modal>
  );
}