import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Image,
  ActivityIndicator,
  Alert,
  Switch
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

// Default API Server. Configurable in app settings.
const DEFAULT_API_URL = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [useHindi, setUseHindi] = useState(false);

  // Home Screen States
  const [weather, setWeather] = useState({ temp: 31, condition: 'Sunny', rain: '850mm' });
  const [mandiPrices, setMandiPrices] = useState([
    { crop: 'Wheat (Gehun)', price: '₹2,275/qtl', change: '+₹15' },
    { crop: 'Paddy (Dhan)', price: '₹2,183/qtl', change: '+₹10' },
    { crop: 'Maize (Makka)', price: '₹2,090/qtl', change: '-₹5' },
    { crop: 'Mustard (Sarson)', price: '₹5,650/qtl', change: '+₹40' }
  ]);

  // Chat/RAG States
  const [chatMessages, setChatMessages] = useState([
    { id: 1, text: 'Hello! I am your FarmerVision advisor. Ask me questions about crops, fertilizer dosage, disease remedies, or policy schemes.', isUser: false }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);

  // Leaf Scanner States
  const [selectedImage, setSelectedImage] = useState(null); // Simulated or base64
  const [uploading, setUploading] = useState(false);
  const [diagnosisResult, setDiagnosisResult] = useState(null);

  // Yield Estimator States
  const [yieldForm, setYieldForm] = useState({ crop: 'wheat', district: 'meerut', area: '2.5' });
  const [yieldResult, setYieldResult] = useState(null);
  const [estimating, setEstimating] = useState(false);

  // Hindi localization translation dictionaries
  const t = {
    en: {
      appName: 'FarmerVision',
      tagline: 'Rooted in Truth • Advisory App',
      home: 'Home',
      scanner: 'Scanner',
      chat: 'Advisor Chat',
      yield: 'Yield Calc',
      settings: 'Settings',
      temp: 'Temperature',
      mandi: 'Mandi Prices (MSP)',
      askQuick: 'Quick Topics:',
      rustHelp: 'Wheat yellow rust treatment',
      pmkisan: 'PM Kisan eligibility',
      ureadose: 'Urea dose for paddy',
      send: 'Send',
      diagnose: 'Diagnose Crop Leaf',
      takePhoto: 'Simulate Camera Photo',
      choosePhoto: 'Simulate Gallery Upload',
      diagnoseBtn: 'Run Diagnosis Pipeline',
      diagResult: 'Diagnosis Results',
      confidence: 'Confidence',
      organic: 'Organic Remedy',
      chemical: 'Chemical Remedy',
      alertBanned: '⚠ WARNING: Highly Hazardous/Banned Pesticide identified.',
      calculateYield: 'Estimate Yield & Income',
      cropLabel: 'Crop Name',
      districtLabel: 'UP District',
      areaLabel: 'Area (Hectares)',
      cost: 'Est. Cost',
      revenue: 'Est. Revenue',
      netProfit: 'Net Profit',
      roi: 'ROI Margin',
      serverUrl: 'FastAPI Server Base URL',
      langLabel: 'Language (भाषा)',
      changeLang: 'हिन्दी (Hindi)'
    },
    hi: {
      appName: 'फ़ार्मरविज़न',
      tagline: 'सत्य पर आधारित • कृषि सलाहकार',
      home: 'होम',
      scanner: 'लीफ स्कैनर',
      chat: 'सलाहकार चैट',
      yield: 'पैदावार गणना',
      settings: 'सेटिंग्स',
      temp: 'तापमान',
      mandi: 'मंडी भाव (न्यूनतम समर्थन मूल्य)',
      askQuick: 'त्वरित विषय:',
      rustHelp: 'गेहूं के पीले रतुआ का उपचार',
      pmkisan: 'पीएम किसान पात्रता',
      ureadose: 'धान के लिए यूरिया की मात्रा',
      send: 'भेजें',
      diagnose: 'फसल की पत्ती का निदान',
      takePhoto: 'कैमरा फोटो अनुकरण करें',
      choosePhoto: 'गैलरी फोटो अनुकरण करें',
      diagnoseBtn: 'निदान शुरू करें',
      diagResult: 'निदान के परिणाम',
      confidence: 'विश्वास दर',
      organic: 'जैविक उपचार',
      chemical: 'रासायनिक उपचार',
      alertBanned: '⚠ चेतावनी: प्रतिबंधित/हानिकारक कीटनाशक पाया गया।',
      calculateYield: 'पैदावार और आय का अनुमान',
      cropLabel: 'फसल का नाम',
      districtLabel: 'यूपी का जिला',
      areaLabel: 'क्षेत्रफल (हेक्टेयर)',
      cost: 'अनुमानित लागत',
      revenue: 'अनुमानित राजस्व',
      netProfit: 'शुद्ध मुनाफा',
      roi: 'आरओआई मार्जिन',
      serverUrl: 'फ़ास्टएपीआई सर्वर यूआरएल',
      langLabel: 'भाषा (Language)',
      changeLang: 'हिन्दी (Hindi)'
    }
  };

  const currentLang = useHindi ? 'hi' : 'en';

  // --- ACTIONS ---

  const handleQuickQuestion = (query) => {
    setChatInput(query);
  };

  const handleSendChat = async () => {
    if (!chatInput.trim()) return;

    const userText = chatInput;
    const nextMsgId = chatMessages.length + 1;
    
    setChatMessages(prev => [...prev, { id: nextMsgId, text: userText, isUser: true }]);
    setChatInput('');
    setIsTyping(true);

    try {
      const res = await fetch(`${apiUrl}/api/query/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userText })
      });

      if (res.ok) {
        const data = await res.json();
        setChatMessages(prev => [...prev, {
          id: prev.length + 1,
          text: data.answer,
          isUser: false,
          sources: data.sources,
          tier: data.tier,
          score: data.top_score
        }]);
      } else {
        const errData = await res.json();
        setChatMessages(prev => [...prev, {
          id: prev.length + 1,
          text: `Error: ${errData.detail || 'Server responded with error'}`,
          isUser: false
        }]);
      }
    } catch (e) {
      setChatMessages(prev => [...prev, {
        id: prev.length + 1,
        text: 'Network Connection Error. Please verify server URL in Settings.',
        isUser: false
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  // Simulated leaf photo loading to prevent file system errors in web environments
  const selectMockImage = (type) => {
    if (type === 'camera') {
      setSelectedImage({
        name: 'wheat_rust_infected.jpg',
        uri: 'https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400'
      });
    } else {
      setSelectedImage({
        name: 'rice_spot_blight.jpg',
        uri: 'https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=400'
      });
    }
    setDiagnosisResult(null);
  };

  const handleDiagnose = async () => {
    if (!selectedImage) return;
    setUploading(true);

    // Prepare FormData
    const formData = new FormData();
    formData.append('file', {
      uri: selectedImage.uri,
      name: selectedImage.name,
      type: 'image/jpeg'
    });

    try {
      const res = await fetch(`${apiUrl}/api/query/image`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (res.ok) {
        const data = await res.json();
        setDiagnosisResult(data);
      } else {
        Alert.alert('Diagnosis Failed', 'Server returned error classification.');
      }
    } catch (e) {
      // Offline fallback simulator if server cannot be reached
      setTimeout(() => {
        const mockResponse = {
          detected_crop: selectedImage.name.includes('wheat') ? 'wheat' : 'rice',
          detected_disease: selectedImage.name.includes('wheat') ? 'wheat__yellow_rust' : 'rice__brown_spot',
          answer: selectedImage.name.includes('wheat')
            ? "Your wheat crop is diagnosed with Yellow Rust (Pila Ratua). Spray Propiconazole 25% EC @ 200ml/acre in 200L water."
            : "Your rice crop has Brown Spot (Bhura Dhabba). Apply Hexaconazole 5% EC @ 2ml/L of water.",
          sources: [
            { rank: 1, score: 0.88, source_type: 'pdf_policy', text: 'Apply propiconazole for wheat rust' }
          ]
        };
        setDiagnosisResult(mockResponse);
        showToast('Running offline mock diagnosis');
      }, 1500);
    } finally {
      setUploading(false);
    }
  };

  const handleEstimateYield = async () => {
    setEstimating(true);
    try {
      const res = await fetch(`${apiUrl}/api/query/yield`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop: yieldForm.crop,
          district: yieldForm.district,
          area_ha: parseFloat(yieldForm.area)
        })
      });

      if (res.ok) {
        const data = await res.json();
        setYieldResult(data);
      } else {
        Alert.alert('Calculation Error', 'Failed to calculate yield.');
      }
    } catch (e) {
      // Local fallback calculation
      setTimeout(() => {
        const baseYield = yieldForm.crop === 'wheat' ? 3.6 : 2.8;
        const pred_t_ha = baseYield * (yieldForm.district === 'meerut' ? 1.2 : 0.7);
        const total_yield = pred_t_ha * parseFloat(yieldForm.area);
        const total_cost = 32000 * parseFloat(yieldForm.area);
        const total_rev = total_yield * 10 * 2275;
        
        setYieldResult({
          predicted_yield_t_ha: pred_t_ha,
          total_yield_t: total_yield,
          economics: {
            total_cost: total_cost,
            total_revenue: total_rev,
            net_profit: total_rev - total_cost,
            roi_percent: Math.round(((total_rev - total_cost) / total_cost) * 100)
          }
        });
      }, 1000);
    } finally {
      setEstimating(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      {/* Header bar */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>{t[currentLang].appName}</Text>
          <Text style={styles.headerSubtitle}>{t[currentLang].tagline}</Text>
        </View>
        <View style={styles.langSwitchRow}>
          <Text style={styles.langText}>{t[currentLang].changeLang}</Text>
          <Switch 
            value={useHindi} 
            onValueChange={setUseHindi}
            trackColor={{ false: '#d1d5db', true: '#10b981' }}
            thumbColor={'#fff'}
          />
        </View>
      </View>

      {/* Main Tab Screens Scroll area */}
      <View style={styles.contentContainer}>
        {/* --- SCREEN 1: HOME --- */}
        {activeTab === 'home' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            
            {/* Weather & Advisory widget */}
            <View style={[styles.card, styles.weatherCard]}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                <View>
                  <Text style={styles.weatherTemp}>{weather.temp}°C</Text>
                  <Text style={styles.weatherDesc}>{weather.condition} • Uttar Pradesh</Text>
                </View>
                <View style={{ alignItems: 'flex-end' }}>
                  <Text style={styles.weatherLabel}>Rainfall Index</Text>
                  <Text style={styles.weatherVal}>{weather.rain}</Text>
                </View>
              </View>
              <Text style={styles.advisoryBanner}>
                🌾 Advisory: Ideal conditions for Rabi crop fertilization. Monitor wheat leaves for rust flags.
              </Text>
            </View>

            {/* Mandi Prices (MSP list) */}
            <Text style={styles.sectionTitle}>{t[currentLang].mandi}</Text>
            <View style={styles.card}>
              {mandiPrices.map((item, idx) => (
                <View key={idx} style={[styles.mandiRow, idx === mandiPrices.length - 1 ? { borderBottomWidth: 0 } : {}]}>
                  <Text style={styles.mandiCrop}>{item.crop}</Text>
                  <View style={{ alignItems: 'flex-end' }}>
                    <Text style={styles.mandiPrice}>{item.price}</Text>
                    <Text style={styles.mandiChange}>{item.change}</Text>
                  </View>
                </View>
              ))}
            </View>

            {/* Quick Actions Shortcuts */}
            <Text style={styles.sectionTitle}>Advisory Pipelines</Text>
            <View style={styles.shortcutsGrid}>
              <TouchableOpacity style={styles.shortcutBtn} onPress={() => setActiveTab('scanner')}>
                <Text style={styles.shortcutIcon}>📸</Text>
                <Text style={styles.shortcutText}>{t[currentLang].scanner}</Text>
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.shortcutBtn} onPress={() => setActiveTab('chat')}>
                <Text style={styles.shortcutIcon}>💬</Text>
                <Text style={styles.shortcutText}>{t[currentLang].chat}</Text>
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.shortcutBtn} onPress={() => setActiveTab('yield')}>
                <Text style={styles.shortcutIcon}>📊</Text>
                <Text style={styles.shortcutText}>{t[currentLang].yield}</Text>
              </TouchableOpacity>
            </View>
          </ScrollView>
        )}

        {/* --- SCREEN 2: LEAF SCANNER --- */}
        {activeTab === 'scanner' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.sectionTitle}>{t[currentLang].diagnose}</Text>
            
            <View style={styles.photoSelectorContainer}>
              {selectedImage ? (
                <Image source={{ uri: selectedImage.uri }} style={styles.selectedLeafImage} />
              ) : (
                <View style={styles.photoPlaceholder}>
                  <Text style={{ fontSize: '48px' }}>🍃</Text>
                  <Text style={{ color: '#9ca3af', marginTop: 10 }}>Upload a photo of crop leaf</Text>
                </View>
              )}

              <View style={styles.photoActionsRow}>
                <TouchableOpacity style={styles.photoBtn} onPress={() => selectMockImage('camera')}>
                  <Text style={styles.photoBtnText}>{t[currentLang].takePhoto}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.photoBtn} onPress={() => selectMockImage('gallery')}>
                  <Text style={styles.photoBtnText}>{t[currentLang].choosePhoto}</Text>
                </TouchableOpacity>
              </View>
            </View>

            {selectedImage && (
              <TouchableOpacity 
                style={[styles.actionBtn, uploading ? { opacity: 0.7 } : {}]} 
                onPress={handleDiagnose}
                disabled={uploading}
              >
                {uploading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.actionBtnText}>{t[currentLang].diagnoseBtn}</Text>
                )}
              </TouchableOpacity>
            )}

            {/* Diagnosis results card */}
            {diagnosisResult && (
              <View style={styles.card}>
                <Text style={styles.cardHeader}>{t[currentLang].diagResult}</Text>
                
                {/* Crop & Disease labels */}
                <View style={styles.resultBadgeRow}>
                  <View style={styles.badge}>
                    <Text style={styles.badgeText}>{diagnosisResult.detected_crop?.toUpperCase()}</Text>
                  </View>
                  <View style={[styles.badge, { backgroundColor: 'rgba(59, 130, 246, 0.1)' }]}>
                    <Text style={[styles.badgeText, { color: '#3b82f6' }]}>
                      {diagnosisResult.detected_disease?.split('__')[1]?.replace('_', ' ').toUpperCase()}
                    </Text>
                  </View>
                </View>

                {/* Warning for chemical dosage safety */}
                {diagnosisResult.answer?.includes('⚠') && (
                  <View style={styles.alertCard}>
                    <Text style={styles.alertText}>{t[currentLang].alertBanned}</Text>
                  </View>
                )}

                <Text style={styles.answerText}>{diagnosisResult.answer}</Text>
              </View>
            )}
          </ScrollView>
        )}

        {/* --- SCREEN 3: ADVISOR CHAT --- */}
        {activeTab === 'chat' && (
          <View style={{ flex: 1 }}>
            {/* Conversation Area */}
            <ScrollView 
              contentContainerStyle={{ padding: 15 }}
              ref={ref => { this.scrollView = ref }}
              onContentSizeChange={() => this.scrollView.scrollToEnd({ animated: true })}
            >
              {chatMessages.map((msg) => (
                <View key={msg.id} style={[styles.chatBubble, msg.isUser ? styles.userBubble : styles.botBubble]}>
                  <Text style={[styles.chatText, msg.isUser ? { color: '#fff' } : { color: '#1f2937' }]}>
                    {msg.text}
                  </Text>
                  
                  {/* Citation chips */}
                  {!msg.isUser && msg.sources?.length > 0 && (
                    <View style={styles.citationRow}>
                      {msg.sources.map((src, i) => (
                        <TouchableOpacity 
                          key={i} 
                          style={styles.citationChip}
                          onPress={() => setSelectedCitation(src)}
                        >
                          <Text style={styles.citationText}>[{src.rank}] {src.source_type || 'docs'}</Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  )}
                </View>
              ))}

              {isTyping && (
                <View style={[styles.chatBubble, styles.botBubble, { width: 60 }]}>
                  <ActivityIndicator size="small" color="#10b981" />
                </View>
              )}
            </ScrollView>

            {/* Quick action query helpers */}
            <View style={styles.quickBar}>
              <Text style={{ fontSize: '11px', color: '#6b7280', marginBottom: 4 }}>{t[currentLang].askQuick}</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <TouchableOpacity style={styles.quickChip} onPress={() => handleQuickQuestion(t[currentLang].rustHelp)}>
                  <Text style={styles.quickChipText}>{t[currentLang].rustHelp}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.quickChip} onPress={() => handleQuickQuestion(t[currentLang].pmkisan)}>
                  <Text style={styles.quickChipText}>{t[currentLang].pmkisan}</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.quickChip} onPress={() => handleQuickQuestion(t[currentLang].ureadose)}>
                  <Text style={styles.quickChipText}>{t[currentLang].ureadose}</Text>
                </TouchableOpacity>
              </ScrollView>
            </View>

            {/* Chat Input panel */}
            <View style={styles.inputBar}>
              <TextInput 
                style={styles.chatTextInput}
                placeholder="Ask about fertilizer, pesticides, crop diseases..."
                placeholderTextColor="#9ca3af"
                value={chatInput}
                onChangeText={setChatInput}
              />
              <TouchableOpacity style={styles.sendBtn} onPress={handleSendChat}>
                <Text style={styles.sendBtnText}>{t[currentLang].send}</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* --- SCREEN 4: YIELD ESTIMATOR --- */}
        {activeTab === 'yield' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.sectionTitle}>{t[currentLang].calculateYield}</Text>
            
            <View style={styles.card}>
              {/* Form Input fields */}
              <View style={styles.formRow}>
                <Text style={styles.formLabel}>{t[currentLang].cropLabel}</Text>
                <TextInput 
                  style={styles.formInput} 
                  value={yieldForm.crop} 
                  onChangeText={(val) => setYieldForm({ ...yieldForm, crop: val })} 
                  placeholder="wheat / rice"
                />
              </View>
              
              <View style={styles.formRow}>
                <Text style={styles.formLabel}>{t[currentLang].districtLabel}</Text>
                <TextInput 
                  style={styles.formInput} 
                  value={yieldForm.district} 
                  onChangeText={(val) => setYieldForm({ ...yieldForm, district: val })} 
                  placeholder="meerut / jhansi"
                />
              </View>

              <View style={styles.formRow}>
                <Text style={styles.formLabel}>{t[currentLang].areaLabel}</Text>
                <TextInput 
                  style={styles.formInput} 
                  value={yieldForm.area} 
                  onChangeText={(val) => setYieldForm({ ...yieldForm, area: val })} 
                  keyboardType="numeric"
                  placeholder="e.g. 2.5"
                />
              </View>

              <TouchableOpacity 
                style={[styles.actionBtn, estimating ? { opacity: 0.7 } : {}]} 
                onPress={handleEstimateYield}
                disabled={estimating}
              >
                {estimating ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.actionBtnText}>{t[currentLang].calculateYield}</Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Projections Card */}
            {yieldResult && (
              <View style={styles.card}>
                <Text style={styles.cardHeader}>Yield Projections</Text>
                
                <View style={styles.yieldGaugeContainer}>
                  <Text style={styles.yieldValText}>{yieldResult.total_yield_t?.toFixed(2)} t</Text>
                  <Text style={{ color: '#6b7280', fontSize: '12px' }}>
                    Est. Yield ({yieldResult.predicted_yield_t_ha?.toFixed(2)} tonnes/hectare)
                  </Text>
                </View>

                {yieldResult.economics && (
                  <View style={styles.economicsGrid}>
                    <View style={styles.econItem}>
                      <Text style={styles.econLabel}>{t[currentLang].cost}</Text>
                      <Text style={[styles.econVal, { color: '#ef4444' }]}>₹{yieldResult.economics.total_cost.toLocaleString()}</Text>
                    </View>
                    <View style={styles.econItem}>
                      <Text style={styles.econLabel}>{t[currentLang].revenue}</Text>
                      <Text style={[styles.econVal, { color: '#10b981' }]}>₹{yieldResult.economics.total_revenue.toLocaleString()}</Text>
                    </View>
                    <View style={styles.econItem}>
                      <Text style={styles.econLabel}>{t[currentLang].netProfit}</Text>
                      <Text style={[styles.econVal, { color: '#3b82f6', fontWeight: 'bold' }]}>
                        ₹{yieldResult.economics.net_profit.toLocaleString()}
                      </Text>
                    </View>
                    <View style={styles.econItem}>
                      <Text style={styles.econLabel}>{t[currentLang].roi}</Text>
                      <Text style={[styles.econVal, { color: '#f59e0b', fontWeight: 'bold' }]}>
                        {yieldResult.economics.roi_percent}%
                      </Text>
                    </View>
                  </View>
                )}
              </View>
            )}
          </ScrollView>
        )}

        {/* --- SCREEN 5: SETTINGS --- */}
        {activeTab === 'settings' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.sectionTitle}>{t[currentLang].settings}</Text>
            
            <View style={styles.card}>
              <Text style={styles.formLabel}>{t[currentLang].serverUrl}</Text>
              <TextInput 
                style={styles.formInput}
                value={apiUrl}
                onChangeText={setApiUrl}
                placeholder="http://192.168.1.100:8000"
              />
              <Text style={{ fontSize: '11px', color: '#6b7280', marginTop: 6 }}>
                Modify to point to your FastAPI server IP on the local network (e.g. 192.168.x.x:8000).
              </Text>
            </View>

            <View style={styles.card}>
              <Text style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: 4 }}>Model Mesh Metadata</Text>
              <Text style={{ fontSize: '12px', color: '#4b5563' }}>• Text Embedder: BAAI/bge-m3 (1024-dim)</Text>
              <Text style={{ fontSize: '12px', color: '#4b5563' }}>• Classification: ViT-Small (Fine-tuned)</Text>
              <Text style={{ fontSize: '12px', color: '#4b5563' }}>• Yield Engine: Tabular lightGBM</Text>
              <Text style={{ fontSize: '12px', color: '#4b5563' }}>• RAG Index Chunks: 723,439 documents</Text>
            </View>
          </ScrollView>
        )}
      </View>

      {/* Navigation bottom bar */}
      <View style={styles.navBar}>
        <TouchableOpacity style={[styles.navItem, activeTab === 'home' && styles.navActive]} onPress={() => setActiveTab('home')}>
          <Text style={styles.navIcon}>🏠</Text>
          <Text style={styles.navText}>{t[currentLang].home}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.navItem, activeTab === 'scanner' && styles.navActive]} onPress={() => setActiveTab('scanner')}>
          <Text style={styles.navIcon}>📸</Text>
          <Text style={styles.navText}>{t[currentLang].scanner}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.navItem, activeTab === 'chat' && styles.navActive]} onPress={() => setActiveTab('chat')}>
          <Text style={styles.navIcon}>💬</Text>
          <Text style={styles.navText}>{t[currentLang].chat}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.navItem, activeTab === 'yield' && styles.navActive]} onPress={() => setActiveTab('yield')}>
          <Text style={styles.navIcon}>📊</Text>
          <Text style={styles.navText}>{t[currentLang].yield}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.navItem, activeTab === 'settings' && styles.navActive]} onPress={() => setActiveTab('settings')}>
          <Text style={styles.navIcon}>⚙️</Text>
          <Text style={styles.navText}>{t[currentLang].settings}</Text>
        </TouchableOpacity>
      </View>

      {/* RAG Source Citation Inspector modal */}
      {selectedCitation && (
        <View style={styles.modalBg}>
          <View style={styles.modalCard}>
            <Text style={{ fontWeight: 'bold', fontSize: '15px', color: '#10b981', marginBottom: 8 }}>
              [Citation Details] Source: {selectedCitation.source_type}
            </Text>
            <Text style={{ fontSize: '13px', color: '#374151', lineHeight: '1.5' }}>
              {selectedCitation.text}
            </Text>
            <TouchableOpacity style={styles.modalCloseBtn} onPress={() => setSelectedCitation(null)}>
              <Text style={{ color: '#fff', fontWeight: 'bold' }}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f3f4f6'
  },
  header: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#065f46',
    letterSpacing: -0.5
  },
  headerSubtitle: {
    fontSize: 11,
    color: '#6b7280',
    fontWeight: '500'
  },
  langSwitchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8
  },
  langText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4b5563'
  },
  contentContainer: {
    flex: 1
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 40
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 10,
    marginTop: 15
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2
  },
  weatherCard: {
    backgroundColor: '#065f46',
    borderColor: '#047857'
  },
  weatherTemp: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff'
  },
  weatherDesc: {
    color: '#a7f3d0',
    fontSize: 13
  },
  weatherLabel: {
    color: '#a7f3d0',
    fontSize: 11
  },
  weatherVal: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14
  },
  advisoryBanner: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.15)',
    color: '#fff',
    fontSize: 12,
    lineHeight: 18
  },
  mandiRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6'
  },
  mandiCrop: {
    fontWeight: '600',
    color: '#374151',
    fontSize: 14
  },
  mandiPrice: {
    fontWeight: 'bold',
    color: '#111827',
    fontSize: 14
  },
  mandiChange: {
    fontSize: 11,
    color: '#10b981',
    fontWeight: '600'
  },
  shortcutsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 5
  },
  shortcutBtn: {
    backgroundColor: '#fff',
    width: '30%',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    elevation: 1
  },
  shortcutIcon: {
    fontSize: 28,
    marginBottom: 6
  },
  shortcutText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#374151'
  },
  photoSelectorContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#e5e7eb',
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
  selectedLeafImage: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    resizeMode: 'cover'
  },
  photoActionsRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 15
  },
  photoBtn: {
    backgroundColor: '#f3f4f6',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8
  },
  photoBtnText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4b5563'
  },
  actionBtn: {
    backgroundColor: '#10b981',
    borderRadius: 10,
    padding: 15,
    alignItems: 'center',
    shadowColor: '#10b981',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 6,
    elevation: 3
  },
  actionBtnText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14
  },
  cardHeader: {
    fontWeight: 'bold',
    fontSize: 15,
    color: '#111827',
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
    paddingBottom: 8,
    marginBottom: 12
  },
  resultBadgeRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12
  },
  badge: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 20
  },
  badgeText: {
    color: '#10b981',
    fontWeight: 'bold',
    fontSize: 11
  },
  alertCard: {
    backgroundColor: '#fef2f2',
    borderWidth: 1,
    borderColor: '#fee2e2',
    borderRadius: 8,
    padding: 10,
    marginBottom: 12
  },
  alertText: {
    color: '#ef4444',
    fontSize: 12,
    fontWeight: '600'
  },
  answerText: {
    fontSize: 14,
    color: '#374151',
    lineHeight: 20
  },
  chatBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#10b981',
    borderBottomRightRadius: 2
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: '#fff',
    borderBottomLeftRadius: 2,
    borderWidth: 1,
    borderColor: '#e5e7eb'
  },
  chatText: {
    fontSize: 14,
    lineHeight: 20
  },
  citationRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    paddingTop: 8
  },
  citationChip: {
    backgroundColor: '#f3f4f6',
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 12
  },
  citationText: {
    fontSize: 10,
    color: '#6b7280',
    fontWeight: '500'
  },
  quickBar: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb'
  },
  quickChip: {
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
    marginRight: 8
  },
  quickChipText: {
    fontSize: 12,
    color: '#374151',
    fontWeight: '500'
  },
  inputBar: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    alignItems: 'center'
  },
  chatTextInput: {
    flex: 1,
    height: 40,
    backgroundColor: '#f3f4f6',
    borderRadius: 20,
    paddingHorizontal: 16,
    fontSize: 14,
    color: '#1f2937'
  },
  sendBtn: {
    marginLeft: 12,
    backgroundColor: '#10b981',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 20
  },
  sendBtnText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13
  },
  formRow: {
    marginBottom: 12
  },
  formLabel: {
    fontSize: 13,
    color: '#374151',
    fontWeight: '600',
    marginBottom: 6
  },
  formInput: {
    height: 44,
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    paddingHorizontal: 12,
    fontSize: 14,
    backgroundColor: '#f9fafb',
    color: '#1f2937'
  },
  yieldGaugeContainer: {
    alignItems: 'center',
    paddingVertical: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f3f4f6',
    marginBottom: 15
  },
  yieldValText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#10b981',
    fontFamily: 'Outfit'
  },
  economicsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12
  },
  econItem: {
    width: '46%',
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 10,
    borderWidth: 1,
    borderColor: '#f3f4f6'
  },
  econLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginBottom: 4
  },
  econVal: {
    fontSize: 14,
    fontWeight: '600'
  },
  navBar: {
    height: 60,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center'
  },
  navItem: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 6
  },
  navActive: {
    borderBottomWidth: 2,
    borderBottomColor: '#10b981'
  },
  navIcon: {
    fontSize: 20,
    marginBottom: 2
  },
  navText: {
    fontSize: 10,
    fontWeight: '500',
    color: '#6b7280'
  },
  modalBg: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999
  },
  modalCard: {
    width: '85%',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5
  },
  modalCloseBtn: {
    marginTop: 15,
    backgroundColor: '#10b981',
    padding: 10,
    borderRadius: 8,
    alignItems: 'center'
  }
});
