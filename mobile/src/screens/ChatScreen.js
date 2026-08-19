import React from 'react';
import { ActivityIndicator, Image, ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Ionicons from '@react-native-vector-icons/ionicons';

export default function ChatScreen({
  chatMessages,
  isTyping,
  chatInput,
  chatImage,
  chatScrollRef,
  onInputChange,
  onSend,
  onAttach,
  onClearImage,
  onQuickQuestion,
  onSelectCitation,
  t,
  s,
  theme,
  accent,
}) {
  return (
    <View style={{ flex: 1, backgroundColor: theme.bg }}>
      {/* Conversation Area */}
      <ScrollView
        contentContainerStyle={{ padding: 15, paddingBottom: 8 }}
        ref={chatScrollRef}
        onContentSizeChange={() => chatScrollRef.current?.scrollToEnd({ animated: true })}
        showsVerticalScrollIndicator={false}
      >
        {chatMessages.map((msg) => (
          <View
            key={msg.id}
            style={[s.chatRow, msg.isUser ? { justifyContent: 'flex-end' } : { justifyContent: 'flex-start' }]}
          >
            {!msg.isUser && (
              <LinearGradient
                colors={[accent.main, accent.strong]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={s.botAvatar}
              >
                <Ionicons name="leaf" size={15} color="#fff" />
              </LinearGradient>
            )}
            <View
              style={[
                s.chatBubble,
                msg.isUser
                  ? [s.userBubble, { backgroundColor: accent.main }]
                  : [s.botBubble, { backgroundColor: theme.surface, borderColor: theme.border }],
              ]}
            >
              {msg.image && (
                <Image
                  source={{ uri: msg.image.uri }}
                  style={[s.chatImageThumb, { backgroundColor: theme.surfaceAlt }]}
                  resizeMode="cover"
                />
              )}
              {msg.text ? (
                <Text style={[s.chatText, { color: msg.isUser ? '#fff' : theme.text }]}>
                  {msg.text}
                </Text>
              ) : null}

              {/* Citation chips */}
              {!msg.isUser && msg.sources?.length > 0 && (
                <View style={[s.citationRow, { borderTopColor: theme.border }]}>
                  {msg.sources.map((src, i) => (
                    <TouchableOpacity
                      key={i}
                      style={[s.citationChip, { backgroundColor: theme.surfaceAlt }]}
                      onPress={() => onSelectCitation(src)}
                    >
                      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                        <Ionicons name="document-text-outline" size={11} color={accent.softText} style={{ marginRight: 4 }} />
                        <Text numberOfLines={1} style={[s.citationText, { color: theme.textMuted }]}>[{src.rank}] {src.name || src.source_type || 'docs'}</Text>
                      </View>
                    </TouchableOpacity>
                  ))}
                </View>
              )}
            </View>
          </View>
        ))}

        {isTyping && (
          <View style={[s.chatRow, { justifyContent: 'flex-start' }]}>
            <LinearGradient
              colors={[accent.main, accent.strong]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={s.botAvatar}
            >
              <Ionicons name="leaf" size={15} color="#fff" />
            </LinearGradient>
            <View style={[s.chatBubble, s.botBubble, { width: 64, backgroundColor: theme.surface, borderColor: theme.border }]}>
              <ActivityIndicator size="small" color={accent.main} />
            </View>
          </View>
        )}
      </ScrollView>

      {/* Quick action query helpers */}
      <View style={[s.quickBar, { backgroundColor: theme.surfaceAlt, borderTopColor: theme.border }]}>
        <Text style={[s.quickBarLabel, { color: theme.textMuted }]}>{t('askQuick')}</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => onQuickQuestion(t('rustHelp'))}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="leaf-outline" size={13} color={accent.softText} style={{ marginRight: 5 }} />
              <Text style={[s.quickChipText, { color: accent.softText }]}>{t('rustHelp')}</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => onQuickQuestion(t('pmkisan'))}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="document-text-outline" size={13} color={accent.softText} style={{ marginRight: 5 }} />
              <Text style={[s.quickChipText, { color: accent.softText }]}>{t('pmkisan')}</Text>
            </View>
          </TouchableOpacity>
          <TouchableOpacity style={[s.quickChip, { backgroundColor: accent.soft }]} onPress={() => onQuickQuestion(t('ureadose'))}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <Ionicons name="flask-outline" size={13} color={accent.softText} style={{ marginRight: 5 }} />
              <Text style={[s.quickChipText, { color: accent.softText }]}>{t('ureadose')}</Text>
            </View>
          </TouchableOpacity>
        </ScrollView>
      </View>

      {/* Chat Input panel */}
      {chatImage && (
        <View style={[s.chatAttachRow, { backgroundColor: theme.surfaceAlt, borderTopColor: theme.border }]}>
          <Image source={{ uri: chatImage.uri }} style={s.chatAttachThumb} resizeMode="cover" />
          <Text numberOfLines={1} style={[s.chatAttachName, { color: theme.textMuted }]}>{chatImage.name}</Text>
          <TouchableOpacity onPress={onClearImage} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
            <Ionicons name="close-circle" size={20} color={theme.textMuted} />
          </TouchableOpacity>
        </View>
      )}
      <View style={[s.inputBar, { backgroundColor: theme.surface, borderTopColor: theme.border }]}>
        <TouchableOpacity
          style={[s.attachBtn, { backgroundColor: chatImage ? accent.soft : 'transparent' }]}
          onPress={onAttach}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Ionicons name="image-outline" size={21} color={chatImage ? accent.softText : theme.textMuted} />
        </TouchableOpacity>
        <TextInput
          style={[s.chatTextInput, { backgroundColor: theme.inputBg, color: theme.text }]}
          placeholder={chatImage ? 'Ask about this leaf (optional)...' : "Ask about fertilizer, pesticides, crop diseases..."}
          placeholderTextColor={theme.placeholder}
          value={chatInput}
          onChangeText={onInputChange}
        />
        <TouchableOpacity style={s.sendBtn} onPress={onSend}>
          <LinearGradient
            colors={[accent.main, accent.strong]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={s.sendBtnGradient}
          >
            <Ionicons name="send" size={18} color="#fff" />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
}