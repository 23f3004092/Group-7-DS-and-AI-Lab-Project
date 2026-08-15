import { StyleSheet } from 'react-native';

// ---------- TYPOGRAPHY ----------

export const FONT = {
  regular: 'Inter_400Regular',
  medium: 'Inter_500Medium',
  semibold: 'Inter_600SemiBold',
  bold: 'Inter_700Bold',
  extrabold: 'Inter_800ExtraBold',
};

// ---------- THEME SYSTEM ----------

// Accent palettes available for personalization
export const ACCENTS = {
  emerald: {
    label: 'Emerald',
    main: '#10b981',
    strong: '#059669',
    soft: 'rgba(16, 185, 129, 0.14)',
    softText: '#059669',
  },
  sky: {
    label: 'Sky',
    main: '#0ea5e9',
    strong: '#0284c7',
    soft: 'rgba(14, 165, 233, 0.14)',
    softText: '#0284c7',
  },
  amber: {
    label: 'Amber',
    main: '#f59e0b',
    strong: '#d97706',
    soft: 'rgba(245, 158, 11, 0.16)',
    softText: '#b45309',
  },
  violet: {
    label: 'Violet',
    main: '#8b5cf6',
    strong: '#7c3aed',
    soft: 'rgba(139, 92, 246, 0.14)',
    softText: '#7c3aed',
  },
  rose: {
    label: 'Rose',
    main: '#f43f5e',
    strong: '#e11d48',
    soft: 'rgba(244, 63, 94, 0.14)',
    softText: '#e11d48',
  },
};

// Base color tokens per theme mode
export const BASE_THEMES = {
  light: {
    name: 'Light',
    bg: '#f4f6fa',
    surface: '#ffffff',
    surfaceAlt: '#eef2f8',
    surfaceDeep: '#e6ebf2',
    text: '#0f172a',
    textMuted: '#64748b',
    border: '#e7ecf3',
    inputBg: '#f1f4f9',
    placeholder: '#94a3b8',
    shadow: '#0f172a',
    shadowOpacity: 0.08,
    statusBar: 'dark',
    weatherGradient: ['#10b981', '#047857'],
    danger: '#ef4444',
    dangerBg: '#fef2f2',
    success: '#10b981',
    successBg: '#ecfdf5',
    warning: '#d97706',
    warningBg: '#fffbeb',
  },
  dark: {
    name: 'Dark',
    bg: '#0b0f17',
    surface: '#131b28',
    surfaceAlt: '#1a2333',
    surfaceDeep: '#0d1320',
    text: '#e7edf6',
    textMuted: '#8ea0bb',
    border: '#253049',
    inputBg: '#0e1420',
    placeholder: '#5c6d88',
    shadow: '#000000',
    shadowOpacity: 0.4,
    statusBar: 'light',
    weatherGradient: ['#0f9d8f', '#0b6e66'],
    danger: '#f87171',
    dangerBg: '#2a1416',
    success: '#34d399',
    successBg: '#12241d',
    warning: '#fbbf24',
    warningBg: '#2b2412',
  },
  highContrast: {
    name: 'High Contrast',
    bg: '#000000',
    surface: '#000000',
    surfaceAlt: '#1a1a1a',
    surfaceDeep: '#000000',
    text: '#ffffff',
    textMuted: '#d4d4d4',
    border: '#ffffff',
    inputBg: '#0a0a0a',
    placeholder: '#a3a3a3',
    shadow: '#000000',
    shadowOpacity: 0,
    statusBar: 'light',
    weatherGradient: ['#005a00', '#003300'],
    danger: '#ff4444',
    dangerBg: '#330000',
    success: '#33ff77',
    successBg: '#003311',
    warning: '#ffcc00',
    warningBg: '#332b00',
  },
};

export const FONT_SCALES = {
  small: 0.88,
  medium: 1,
  large: 1.14,
};

// Builds the shared StyleSheet object, derived from the active theme/accent.
export function createStyles(theme, accent, fontScale, themeMode) {
  const fs = fontScale;
  return StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.bg
    },
    header: {
      paddingHorizontal: 18,
      paddingTop: 14,
      paddingBottom: 18,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    headerTitle: {
      fontSize: 22 * fs,
      fontWeight: '800',
      fontFamily: FONT.bold,
      color: '#ffffff',
      letterSpacing: -0.5
    },
    headerSubtitle: {
      fontSize: 11 * fs,
      color: 'rgba(255,255,255,0.85)',
      fontWeight: '500',
      fontFamily: FONT.medium,
      marginTop: 2
    },
    langPill: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.18)',
      borderRadius: 999,
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderWidth: 1,
      borderColor: 'rgba(255,255,255,0.35)',
      gap: 5
    },
    langText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      color: '#ffffff'
    },
    contentContainer: {
      flex: 1
    },
    scrollContent: {
      padding: 16,
      paddingBottom: 20
    },
    greeting: {
      fontSize: 15 * fs,
      fontWeight: '700',
      fontFamily: FONT.semibold,
      color: theme.text,
      marginBottom: 12
    },
    sectionHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
      marginBottom: 10,
      marginTop: 15
    },
    sectionAccent: {
      width: 4,
      height: 16,
      borderRadius: 2
    },
    sectionTitle: {
      fontSize: 16 * fs,
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      color: theme.text
    },
    liveBadge: {
      paddingVertical: 3,
      paddingHorizontal: 8,
      borderRadius: 12,
      marginLeft: 4
    },
    liveBadgeText: {
      fontSize: 10 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold
    },
    mandiMarket: {
      fontSize: 10 * fs,
      fontFamily: FONT.medium,
      marginTop: 1
    },
    mandiLocation: {
      fontSize: 11 * fs,
      marginLeft: 6,
      marginBottom: 0,
      fontWeight: '500',
      fontFamily: FONT.medium
    },
    districtPicker: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderRadius: 12,
      paddingHorizontal: 12,
      paddingVertical: 12,
      marginTop: 6
    },
    districtPickerText: {
      fontSize: 14 * fs,
      fontFamily: FONT.medium
    },
    locationBtn: {
      borderWidth: 1,
      borderRadius: 12,
      paddingVertical: 11,
      alignItems: 'center',
      marginTop: 14
    },
    locationBtnText: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    districtRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingVertical: 13,
      borderBottomWidth: StyleSheet.hairlineWidth
    },
    districtRowText: {
      fontSize: 15 * fs,
      fontFamily: FONT.medium
    },
    card: {
      backgroundColor: theme.surface,
      borderRadius: 20,
      padding: 16,
      marginBottom: 15,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderColor: theme.border,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: theme.shadowOpacity,
      shadowRadius: 12,
      elevation: 3
    },
    weatherCard: {
      borderWidth: 0,
      shadowOpacity: 0.2
    },
    weatherTemp: {
      fontSize: 34 * fs,
      fontWeight: '800',
      fontFamily: FONT.extrabold,
      color: '#fff'
    },
    weatherDesc: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 13 * fs,
      fontFamily: FONT.medium
    },
    weatherLabel: {
      color: 'rgba(255,255,255,0.8)',
      fontSize: 11 * fs,
      fontFamily: FONT.medium
    },
    weatherVal: {
      color: '#fff',
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 15 * fs
    },
    weatherForecastRow: {
      flexDirection: 'row',
      gap: 8,
      marginTop: 14
    },
    weatherForecastChip: {
      flex: 1,
      backgroundColor: 'rgba(255,255,255,0.14)',
      borderRadius: 14,
      paddingVertical: 8,
      paddingHorizontal: 6,
      alignItems: 'center'
    },
    weatherForecastDay: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 10 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      marginBottom: 3
    },
    weatherForecastTemp: {
      color: '#fff',
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    advisoryBanner: {
      flex: 1,
      color: '#fff',
      fontSize: 12 * fs,
      lineHeight: 18 * fs,
      fontFamily: FONT.medium
    },
    mandiRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingVertical: 11
    },
    mandiCrop: {
      fontWeight: '600',
      fontFamily: FONT.semibold,
      color: theme.text,
      fontSize: 12.5 * fs
    },
    mandiPrice: {
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      color: theme.text,
      fontSize: 13 * fs,
      marginTop: 4
    },
    mandiChange: {
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      marginLeft: 3
    },
    mandiLocationRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      borderRadius: 14,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      paddingHorizontal: 13,
      paddingVertical: 10,
      marginBottom: 12
    },
    mandiLocationEdit: {
      fontSize: 11 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      marginLeft: 4
    },
    mandiGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 10,
    },
    mandiCard: {
      width: '30%',
      height: 108,
      borderRadius: 16,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      padding: 10,
    },
    shortcutsGrid: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginTop: 5
    },
    shortcutBtn: {
      width: '31%',
      borderRadius: 20,
      paddingVertical: 18,
      paddingHorizontal: 8,
      alignItems: 'center',
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: theme.shadowOpacity,
      shadowRadius: 6,
      elevation: 2
    },
    shortcutIconWrap: {
      width: 46,
      height: 46,
      borderRadius: 23,
      alignItems: 'center',
      justifyContent: 'center',
      marginBottom: 8
    },
    shortcutIcon: {
      fontSize: 22
    },
    shortcutText: {
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      color: theme.text,
      textAlign: 'center'
    },
    photoSelectorContainer: {
      borderRadius: 20,
      borderWidth: 2,
      borderStyle: 'dashed',
      padding: 20,
      alignItems: 'center',
      marginBottom: 16
    },
    photoPlaceholder: {
      height: 180,
      justifyContent: 'center',
      alignItems: 'center'
    },
    photoHint: {
      marginTop: 12,
      fontSize: 12 * fs,
      fontFamily: FONT.medium
    },
    selectedLeafImage: {
      width: '100%',
      height: 200,
      borderRadius: 14,
      resizeMode: 'cover'
    },
    photoActionsRow: {
      flexDirection: 'row',
      gap: 12,
      marginTop: 15
    },
    photoBtn: {
      paddingVertical: 10,
      paddingHorizontal: 16,
      borderRadius: 13
    },
    photoBtnText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    actionBtn: {
      borderRadius: 18,
      overflow: 'hidden',
      marginBottom: 16,
      shadowColor: accent.main,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.3,
      shadowRadius: 8,
      elevation: 4
    },
    actionBtnGradient: {
      padding: 15,
      alignItems: 'center',
      justifyContent: 'center'
    },
    actionBtnText: {
      color: '#fff',
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 14 * fs
    },
    cardHeader: {
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 15 * fs,
      color: theme.text,
      borderBottomWidth: 1,
      borderBottomColor: theme.border,
      paddingBottom: 10,
      marginBottom: 12
    },
    resultBadgeRow: {
      flexDirection: 'row',
      gap: 8,
      marginBottom: 12
    },
    badge: {
      paddingVertical: 5,
      paddingHorizontal: 12,
      borderRadius: 20
    },
    badgeText: {
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 11 * fs
    },
    alertCard: {
      borderWidth: 1,
      borderRadius: 12,
      padding: 10,
      marginBottom: 12
    },
    alertText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    answerText: {
      fontSize: 14 * fs,
      lineHeight: 20 * fs,
      fontFamily: FONT.regular
    },
    chatRow: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      marginBottom: 4
    },
    botAvatar: {
      width: 32,
      height: 32,
      borderRadius: 16,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 8,
      marginTop: 4
    },
    chatBubble: {
      maxWidth: '78%',
      padding: 13,
      borderRadius: 18,
      marginBottom: 12,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 2
    },
    userBubble: {
      borderBottomRightRadius: 4
    },
    botBubble: {
      borderBottomLeftRadius: 4,
      borderWidth: 1
    },
    chatText: {
      fontSize: 14 * fs,
      lineHeight: 20 * fs,
      fontFamily: FONT.regular
    },
    citationRow: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 6,
      marginTop: 8,
      borderTopWidth: StyleSheet.hairlineWidth,
      paddingTop: 8
    },
    citationChip: {
      paddingVertical: 3,
      paddingHorizontal: 9,
      borderRadius: 12
    },
    citationText: {
      fontSize: 10 * fs,
      fontWeight: '500',
      fontFamily: FONT.medium
    },
    quickBar: {
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderTopWidth: 1
    },
    quickBarLabel: {
      fontSize: 11 * fs,
      fontFamily: FONT.medium,
      marginBottom: 5
    },
    quickChip: {
      paddingHorizontal: 12,
      paddingVertical: 7,
      borderRadius: 16,
      marginRight: 8
    },
    quickChipText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    inputBar: {
      flexDirection: 'row',
      padding: 12,
      borderTopWidth: 1,
      alignItems: 'center'
    },
    attachBtn: {
      width: 38,
      height: 38,
      borderRadius: 19,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 8
    },
    chatAttachRow: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 12,
      paddingVertical: 8,
      borderTopWidth: 1
    },
    chatAttachThumb: {
      width: 44,
      height: 44,
      borderRadius: 8
    },
    chatAttachName: {
      flex: 1,
      fontSize: 12 * fs,
      fontFamily: FONT.medium,
      marginHorizontal: 10
    },
    chatImageThumb: {
      width: 150,
      height: 150,
      borderRadius: 12,
      marginBottom: 6
    },
    chatTextInput: {
      flex: 1,
      height: 42,
      borderRadius: 21,
      paddingHorizontal: 16,
      fontSize: 14 * fs,
      fontFamily: FONT.regular
    },
    sendBtn: {
      marginLeft: 10,
      borderRadius: 21,
      overflow: 'hidden',
      shadowColor: accent.main,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.3,
      shadowRadius: 4,
      elevation: 3
    },
    sendBtnGradient: {
      paddingVertical: 12,
      paddingHorizontal: 16
    },
    sendBtnText: {
      color: '#fff',
      fontWeight: 'bold',
      fontFamily: FONT.bold,
      fontSize: 13 * fs
    },
    formRow: {
      marginBottom: 12
    },
    formLabel: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      marginBottom: 0,
      color: theme.text
    },
    formInput: {
      height: 46,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      borderRadius: 13,
      paddingHorizontal: 12,
      fontSize: 14 * fs,
      fontFamily: FONT.medium
    },
    yieldGaugeContainer: {
      alignItems: 'center',
      paddingVertical: 20,
      borderBottomWidth: StyleSheet.hairlineWidth,
      marginBottom: 15
    },
    yieldValText: {
      fontSize: 34 * fs,
      fontWeight: '800',
      fontFamily: FONT.extrabold
    },
    economicsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12
    },
    econItem: {
      width: '46%',
      borderRadius: 14,
      padding: 10,
      borderWidth: themeMode === 'highContrast' ? 2 : 1
    },
    econLabel: {
      fontSize: 11 * fs,
      fontFamily: FONT.medium,
      marginBottom: 4
    },
    econVal: {
      fontSize: 14 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    weatherChipRow: {
      flexDirection: 'row',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: 8,
      marginTop: 12
    },
    weatherChip: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.16)',
      borderRadius: 14,
      paddingVertical: 5,
      paddingHorizontal: 10
    },
    weatherChipText: {
      color: '#fff',
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    weatherDetailHeader: {
      paddingHorizontal: 18,
      paddingBottom: 8
    },
    weatherBackBtn: {
      width: 34,
      height: 34,
      borderRadius: 17,
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: 'rgba(255,255,255,0.18)'
    },
    weatherDetailTitle: {
      fontSize: 17 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      color: '#fff',
      letterSpacing: -0.3
    },
    weatherDetailTemp: {
      fontSize: 52 * fs,
      fontWeight: '800',
      fontFamily: FONT.extrabold,
      color: '#fff',
      letterSpacing: -1
    },
    weatherDetailCond: {
      color: '#fff',
      fontSize: 15 * fs,
      fontFamily: FONT.semibold,
      marginTop: 4
    },
    weatherDetailLoc: {
      color: 'rgba(255,255,255,0.85)',
      fontSize: 13 * fs,
      fontFamily: FONT.medium,
      marginTop: 2
    },
    weatherDetailPill: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: 'rgba(255,255,255,0.16)',
      borderRadius: 16,
      paddingVertical: 6,
      paddingHorizontal: 12
    },
    weatherDetailPillText: {
      color: '#fff',
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    weatherDetailUpdated: {
      fontSize: 11 * fs,
      fontFamily: FONT.medium,
      marginLeft: 8
    },
    statGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 12
    },
    statItem: {
      width: '100%',
      borderRadius: 16,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      padding: 12,
      marginBottom: 12
    },
    statLabel: {
      fontSize: 11 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    statValue: {
      fontSize: 17 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      letterSpacing: -0.2
    },
    forecastRow: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingVertical: 12
    },
    forecastDay: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      width: '36%'
    },
    forecastTemp: {
      fontSize: 13 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    tipRow: {
      flexDirection: 'row',
      alignItems: 'flex-start',
      paddingVertical: 10
    },
    tipIcon: {
      width: 30,
      height: 30,
      borderRadius: 15,
      alignItems: 'center',
      justifyContent: 'center',
      marginRight: 10,
      marginTop: 1
    },
    tipText: {
      flex: 1,
      fontSize: 13 * fs,
      lineHeight: 19 * fs,
      fontFamily: FONT.regular
    },
    navBar: {
      height: 64,
      marginHorizontal: 14,
      marginBottom: 10,
      borderRadius: 26,
      borderWidth: themeMode === 'highContrast' ? 2 : 1,
      flexDirection: 'row',
      justifyContent: 'space-around',
      alignItems: 'center',
      paddingHorizontal: 8,
      shadowColor: theme.shadow,
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: theme.shadowOpacity,
      shadowRadius: 14,
      elevation: 8
    },
    navItem: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 7,
      paddingHorizontal: 12,
      borderRadius: 18,
      minWidth: 58
    },
    navActive: {
      borderRadius: 18
    },
    navIcon: {
      fontSize: 19,
      marginBottom: 2
    },
    navText: {
      fontSize: 10 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold,
      marginTop: 2
    },
    settingsCardHeader: {
      flexDirection: 'row',
      alignItems: 'center'
    },
    settingHint: {
      fontSize: 11.5 * fs,
      lineHeight: 16 * fs,
      fontFamily: FONT.regular
    },
    themeRow: {
      flexDirection: 'row',
      gap: 10,
      marginTop: 10
    },
    themeChip: {
      flex: 1,
      borderRadius: 16,
      borderWidth: 2,
      padding: 8,
      alignItems: 'center'
    },
    themePreview: {
      width: '100%',
      height: 48,
      borderRadius: 9,
      borderWidth: 1,
      overflow: 'hidden',
      marginBottom: 6
    },
    themePreviewBar: {
      height: 14,
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 6
    },
    themePreviewDot: {
      width: 8,
      height: 8,
      borderRadius: 4
    },
    themePreviewLine: {
      flex: 1,
      marginHorizontal: 6,
      marginBottom: 6,
      borderRadius: 2
    },
    themeChipText: {
      fontSize: 11 * fs,
      fontWeight: '700',
      fontFamily: FONT.bold,
      textAlign: 'center'
    },
    accentRow: {
      flexDirection: 'row',
      gap: 14,
      marginTop: 14,
      justifyContent: 'center'
    },
    swatch: {
      width: 42,
      height: 42,
      borderRadius: 21,
      alignItems: 'center',
      justifyContent: 'center'
    },
    swatchCheck: {
      color: '#fff',
      fontWeight: 'bold',
      fontSize: 16
    },
    fontSizeRow: {
      flexDirection: 'row',
      gap: 10,
      marginTop: 10
    },
    fontSizeChip: {
      flex: 1,
      borderRadius: 16,
      borderWidth: 2,
      paddingVertical: 12,
      alignItems: 'center'
    },
    fontSizeChipText: {
      fontWeight: '800',
      fontFamily: FONT.extrabold,
      marginBottom: 2
    },
    fontSizeChipLabel: {
      fontSize: 10 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    resetBtn: {
      borderWidth: 1.5,
      borderRadius: 16,
      padding: 13,
      alignItems: 'center',
      marginBottom: 30
    },
    aiStatusDot: {
      width: 9,
      height: 9,
      borderRadius: 5,
      marginLeft: 'auto'
    },
    resetBtnText: {
      fontWeight: '700',
      fontFamily: FONT.bold,
      fontSize: 13 * fs
    },
    toast: {
      position: 'absolute',
      bottom: 96,
      left: 20,
      right: 20,
      borderRadius: 16,
      borderWidth: 1,
      padding: 12,
      alignItems: 'center',
      zIndex: 100,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 10,
      elevation: 6
    },
    toastText: {
      fontSize: 12 * fs,
      fontWeight: '600',
      fontFamily: FONT.semibold
    },
    modalBg: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.55)',
      justifyContent: 'flex-end',
      zIndex: 9999
    },
    sheet: {
      width: '100%',
      borderTopLeftRadius: 26,
      borderTopRightRadius: 26,
      padding: 20,
      paddingBottom: 34,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: -4 },
      shadowOpacity: 0.15,
      shadowRadius: 10,
      elevation: 12
    },
    sheetHandle: {
      alignSelf: 'center',
      width: 40,
      height: 5,
      borderRadius: 3,
      backgroundColor: 'rgba(120,130,150,0.4)',
      marginBottom: 16
    },
    modalCloseBtn: {
      marginTop: 15,
      padding: 11,
      borderRadius: 14,
      alignItems: 'center'
    }
  });
}