import React, { useEffect, useRef } from 'react';
import { Animated, Image, Text, View, ActivityIndicator, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { StatusBar } from 'expo-status-bar';
import { FONT } from '../theme';

export default function SplashScreen({ t, theme, fontScale, opacity, ready }) {
  const logoScale = useRef(new Animated.Value(0.7)).current;
  const logoOpacity = useRef(new Animated.Value(0)).current;
  const titleOpacity = useRef(new Animated.Value(0)).current;
  const titleY = useRef(new Animated.Value(26)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(logoOpacity, { toValue: 1, duration: 650, useNativeDriver: true }),
      Animated.spring(logoScale, { toValue: 1, friction: 7, tension: 55, useNativeDriver: true }),
    ]).start();
  }, [logoOpacity, logoScale]);

  useEffect(() => {
    if (!ready) return;
    Animated.parallel([
      Animated.timing(titleOpacity, { toValue: 1, duration: 550, useNativeDriver: true }),
      Animated.timing(titleY, { toValue: 0, duration: 550, useNativeDriver: true }),
    ]).start();
  }, [ready, titleOpacity, titleY]);

  return (
    <Animated.View style={[styles.fill, { opacity }]}>
      <StatusBar style="light" />
      <LinearGradient
        colors={theme.weatherGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.fill}
      >
        <View style={styles.center}>
          <Animated.View style={{ opacity: logoOpacity, transform: [{ scale: logoScale }] }}>
            <View style={styles.iconShell}>
              <Image source={require('../../assets/icon.png')} style={styles.icon} resizeMode="cover" />
            </View>
          </Animated.View>

          {ready && (
            <Animated.View style={{ opacity: titleOpacity, transform: [{ translateY: titleY }], alignItems: 'center' }}>
              <Text style={[styles.appName, { fontSize: 34 * fontScale }]}>{t('appName', { defaultValue: 'FarmerVision' })}</Text>
              <Text style={[styles.tagline, { fontSize: 14 * fontScale }]}>{t('tagline', { defaultValue: 'Rooted in Truth' })}</Text>
            </Animated.View>
          )}
        </View>

        <View style={styles.footer}>
          <ActivityIndicator size="small" color="rgba(255,255,255,0.9)" />
        </View>
      </LinearGradient>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  fill: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  iconShell: {
    width: 116,
    height: 116,
    borderRadius: 30,
    backgroundColor: 'rgba(255,255,255,0.95)',
    padding: 6,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.28,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 10 },
    elevation: 10,
  },
  icon: {
    width: 104,
    height: 104,
    borderRadius: 24,
  },
  appName: {
    color: '#ffffff',
    fontFamily: FONT.extrabold,
    letterSpacing: 0.5,
    marginTop: 26,
    textAlign: 'center',
  },
  tagline: {
    color: 'rgba(255,255,255,0.88)',
    fontFamily: FONT.medium,
    marginTop: 8,
    textAlign: 'center',
  },
  footer: {
    alignItems: 'center',
    paddingBottom: 56,
  },
});