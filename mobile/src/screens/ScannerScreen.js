import React from 'react';
import { ActivityIndicator, Image, ScrollView, Text, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';
import { cropLabel } from '../helpers';

export default function ScannerScreen({
  selectedImage,
  uploading,
  diagnosisResult,
  onTakePhoto,
  onChoosePhoto,
  onDiagnose,
  t,
  s,
  theme,
  accent,
  themeMode,
}) {
  return (
    <ScrollView contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>
      <View style={s.sectionHeader}>
        <View style={[s.sectionAccent, { backgroundColor: accent.main }]} />
        <Text style={s.sectionTitle}>{t('diagnose')}</Text>
      </View>

      <View style={[s.photoSelectorContainer, { borderColor: theme.border, backgroundColor: theme.surface }]}>
        {selectedImage ? (
          <Image source={{ uri: selectedImage.uri }} style={s.selectedLeafImage} />
        ) : (
          <View style={s.photoPlaceholder}>
            <View style={[s.shortcutIconWrap, { backgroundColor: accent.soft, width: 84, height: 84, borderRadius: 42 }]}>
              <Ionicons name="leaf-outline" size={40} color={accent.softText} />
            </View>
            <Text style={[s.photoHint, { color: theme.textMuted }]}>Upload a photo of crop leaf</Text>
          </View>
        )}

        <View style={s.photoActionsRow}>
          <TouchableOpacity
            style={[s.photoBtn, { backgroundColor: theme.surfaceAlt, borderColor: theme.border, borderWidth: themeMode === 'highContrast' ? 2 : 1 }]}
            onPress={onTakePhoto}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="camera-outline" size={15} color={accent.softText} style={{ marginRight: 6 }} />
              <Text style={[s.photoBtnText, { color: theme.text }]}>{t('takePhoto')}</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity
            style={[s.photoBtn, { backgroundColor: theme.surfaceAlt, borderColor: theme.border, borderWidth: themeMode === 'highContrast' ? 2 : 1 }]}
            onPress={onChoosePhoto}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="images-outline" size={15} color={accent.softText} style={{ marginRight: 6 }} />
              <Text style={[s.photoBtnText, { color: theme.text }]}>{t('choosePhoto')}</Text>
            </View>
          </TouchableOpacity>
        </View>
      </View>

      {selectedImage && (
        <TouchableOpacity
          style={[s.actionBtn, uploading ? { opacity: 0.7 } : {}]}
          onPress={onDiagnose}
          disabled={uploading}
        >
          <LinearGradient
            colors={[accent.main, accent.strong]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={s.actionBtnGradient}
          >
            {uploading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="scan" size={18} color="#fff" style={{ marginRight: 8 }} />
                <Text style={s.actionBtnText}>{t('diagnoseBtn')}</Text>
              </View>
            )}
          </LinearGradient>
        </TouchableOpacity>
      )}

      {/* Diagnosis results card */}
      {diagnosisResult && (
        <View style={s.card}>
          <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
            <Ionicons name="analytics-outline" size={16} color={accent.main} style={{ marginRight: 8 }} />
            <Text style={[s.cardHeader, { borderBottomWidth: 0, marginBottom: 0, paddingBottom: 0 }]}>{t('diagResult')}</Text>
          </View>

          {/* Crop & Disease labels */}
          <View style={s.resultBadgeRow}>
            <View style={[s.badge, { backgroundColor: accent.soft }]}>
              <Text style={[s.badgeText, { color: accent.softText }]}>{cropLabel(t, diagnosisResult.detected_crop).toUpperCase()}</Text>
            </View>
            <View style={[s.badge, { backgroundColor: accent.soft }]}>
              <Text style={[s.badgeText, { color: accent.softText }]}>
                {diagnosisResult.detected_disease?.split('__')[1]?.replace('_', ' ').toUpperCase()}
              </Text>
            </View>
          </View>

          {/* Warning for chemical dosage safety */}
          {diagnosisResult.answer?.includes('⚠') && (
            <View style={[s.alertCard, { backgroundColor: theme.dangerBg, borderColor: theme.danger }]}>
              <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                <Ionicons name="warning-outline" size={16} color={theme.danger} style={{ marginRight: 6 }} />
                <Text style={[s.alertText, { color: theme.danger }]}>{t('alertBanned')}</Text>
              </View>
            </View>
          )}

          <Text style={[s.answerText, { color: theme.text }]}>{diagnosisResult.answer}</Text>
        </View>
      )}
    </ScrollView>
  );
}