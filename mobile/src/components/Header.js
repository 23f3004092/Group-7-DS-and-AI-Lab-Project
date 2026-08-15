import React from 'react';
import { Text, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';

// App header bar (padded below the clock/battery status bar area)
export default function Header({ insets, langName, onOpenLang, t, s, theme }) {
  return (
    <LinearGradient
      colors={[theme.weatherGradient[0], theme.weatherGradient[1]]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={[s.header, { paddingTop: insets.top + 12 }]}
    >
      <View>
        <Text style={s.headerTitle}>{t('appName')}</Text>
        <Text style={s.headerSubtitle}>{t('tagline')}</Text>
      </View>
      <TouchableOpacity
        style={s.langPill}
        activeOpacity={0.7}
        onPress={onOpenLang}
      >
        <Ionicons name="globe-outline" size={14} color="#fff" />
        <Text style={s.langText}>{langName || 'English'}</Text>
        <Ionicons name="chevron-down" size={13} color="rgba(255,255,255,0.85)" />
      </TouchableOpacity>
    </LinearGradient>
  );
}