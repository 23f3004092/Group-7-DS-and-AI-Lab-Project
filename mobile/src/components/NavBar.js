import React from 'react';
import { Text, TouchableOpacity, View } from 'react-native';
import Ionicons from '@react-native-vector-icons/ionicons';
import { TABS } from '../config';

export default function NavBar({ activeTab, setActiveTab, t, s, theme, accent, insets }) {
  return (
    <View
      style={[
        s.navBar,
        {
          backgroundColor: theme.surface,
          borderColor: theme.border,
          paddingBottom: Math.max(insets.bottom, 0),
          height: 64 + Math.max(insets.bottom, 0),
        },
      ]}
    >
      {TABS.map((item) => {
        const isActive = activeTab === item.key;
        return (
          <TouchableOpacity
            key={item.key}
            style={[s.navItem, isActive && [s.navActive, { backgroundColor: accent.soft }]]}
            onPress={() => setActiveTab(item.key)}
            activeOpacity={0.7}
          >
            <Ionicons
              name={isActive ? item.iconActive : item.icon}
              size={21}
              color={isActive ? accent.softText : theme.textMuted}
            />
            <Text style={[s.navText, { color: isActive ? accent.softText : theme.textMuted }]}>
              {t(item.labelKey)}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}