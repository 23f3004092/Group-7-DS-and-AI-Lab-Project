import React, { useState, useEffect } from 'react';

// Setup API Host. In dev, points to localhost:8000. In production/same-host, points to empty.
const API_BASE = 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const [config, setConfig] = useState({
    tier_grounded: 0.66,
    tier_fallback: 0.56,
    weight_pdf_policy: 2.0,
    weight_kcc_policy: 0.5,
    weight_pdf_practice: 0.5,
    weight_kcc_practice: 2.0,
    mock_models: true
  });
  const [vectorDb, setVectorDb] = useState(null);
  
  // Filtering & Pagination for logs
  const [pathwayFilter, setPathwayFilter] = useState('');
  const [blockedFilter, setBlockedFilter] = useState('');
  const [selectedLog, setSelectedLog] = useState(null);
  
  // Vector search test state
  const [vectorSearchQuery, setVectorSearchQuery] = useState('');
  const [vectorSearchResults, setVectorSearchResults] = useState(null);
  const [searchingVector, setSearchingVector] = useState(false);
  
  // System status
  const [systemHealth, setSystemHealth] = useState({ status: 'offline', components: {} });
  const [notification, setNotification] = useState(null);

  // Fetch system health and stats on load
  useEffect(() => {
    fetchHealth();
    fetchStats();
    fetchConfig();
    fetchLogs();
    fetchVectorDb();
    
    // Auto-refresh stats every 30 seconds
    const interval = setInterval(() => {
      fetchHealth();
      fetchStats();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch logs when filters change
  useEffect(() => {
    fetchLogs();
  }, [pathwayFilter, blockedFilter]);

  const showToast = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 4000);
  };

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
      } else {
        setSystemHealth({ status: 'degraded', components: {} });
      }
    } catch (e) {
      setSystemHealth({ status: 'offline', components: {} });
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.error('Error fetching stats:', e);
    }
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/config`);
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
      }
    } catch (e) {
      console.error('Error fetching config:', e);
    }
  };

  const fetchLogs = async () => {
    try {
      let url = `${API_BASE}/api/admin/logs?limit=50`;
      if (pathwayFilter) url += `&pathway=${pathwayFilter}`;
      if (blockedFilter !== '') url += `&is_blocked=${blockedFilter === 'true'}`;
      
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setLogs(data);
      }
    } catch (e) {
      console.error('Error fetching logs:', e);
    }
  };

  const fetchVectorDb = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/vectordb`);
      if (res.ok) {
        const data = await res.json();
        setVectorDb(data);
      }
    } catch (e) {
      console.error('Error fetching vector db info:', e);
    }
  };

  const saveConfig = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`${API_BASE}/api/admin/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (res.ok) {
        showToast('Dynamic configurations updated successfully!');
        fetchConfig();
      } else {
        showToast('Failed to update config.', 'error');
      }
    } catch (e) {
      showToast('API Connection Error.', 'error');
    }
  };

  const testVectorSearch = async (e) => {
    e.preventDefault();
    if (!vectorSearchQuery.trim()) return;
    setSearchingVector(true);
    try {
      const res = await fetch(`${API_BASE}/api/mcp/tools/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'search_agricultural_knowledge',
          arguments: { query: vectorSearchQuery }
        })
      });
      if (res.ok) {
        const data = await res.json();
        setVectorSearchResults(data.result);
      } else {
        setVectorSearchResults({ content: [{ text: 'Error performing vector DB search.' }] });
      }
    } catch (e) {
      setVectorSearchResults({ content: [{ text: 'Could not connect to API server.' }] });
    } finally {
      setSearchingVector(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Toast Notification */}
      {notification && (
        <div style={{
          position: 'fixed',
          top: '24px',
          right: '24px',
          padding: '14px 24px',
          borderRadius: '8px',
          zIndex: 9999,
          background: notification.type === 'error' ? '#ef4444' : '#10b981',
          color: '#fff',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)',
          fontFamily: 'var(--font-sans)',
          fontWeight: 600,
          animation: 'fadeIn 0.3s ease-out'
        }}>
          {notification.message}
        </div>
      )}

      {/* Header Navigation */}
      <header className="glass-panel" style={{
        margin: '20px',
        padding: '16px 28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderRadius: 'var(--radius-md)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, var(--primary), #059669)',
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 800,
            fontSize: '20px',
            color: '#fff',
            fontFamily: 'var(--font-mono)'
          }}>
            FV
          </div>
          <div>
            <h1 style={{ fontSize: '20px', fontWeight: 700, letterSpacing: '-0.5px' }}>
              FarmerVision
            </h1>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              ADMIN CONTROLLER v{systemHealth.version || '0.2.0'}
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '30px' }}>
          <button className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            Dashboard
          </button>
          <button className={`tab-btn ${activeTab === 'logs' ? 'active' : ''}`} onClick={() => setActiveTab('logs')}>
            Telemetry Logs
          </button>
          <button className={`tab-btn ${activeTab === 'config' ? 'active' : ''}`} onClick={() => setActiveTab('config')}>
            Config Panel
          </button>
          <button className={`tab-btn ${activeTab === 'vectordb' ? 'active' : ''}`} onClick={() => setActiveTab('vectordb')}>
            Vector DB
          </button>
        </nav>

        {/* Server status pills */}
        <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
          <div className="glass-panel" style={{
            padding: '6px 12px',
            borderRadius: '20px',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            background: 'rgba(255,255,255,0.03)'
          }}>
            <span className={`dot ${systemHealth.status === 'ok' ? 'dot-green' : systemHealth.status === 'degraded' ? 'dot-orange' : 'dot-red'}`} />
            API: <span style={{ marginLeft: '4px', fontWeight: 600, color: systemHealth.status === 'ok' ? 'var(--primary)' : 'var(--accent-red)' }}>
              {systemHealth.status.toUpperCase()}
            </span>
          </div>
          
          <div className="glass-panel" style={{
            padding: '6px 12px',
            borderRadius: '20px',
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            background: 'rgba(255,255,255,0.03)'
          }}>
            <span className={`dot ${vectorDb?.status === 'connected' ? 'dot-green' : 'dot-orange'}`} />
            VectorDB: <span style={{ marginLeft: '4px', fontWeight: 600, color: vectorDb?.status === 'connected' ? 'var(--primary)' : 'var(--accent-orange)' }}>
              {vectorDb?.mode === 'server' ? 'QDRANT' : 'LOCAL'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '0 20px 40px 20px', maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
        
        {/* TAB 1: DASHBOARD */}
        {activeTab === 'dashboard' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            {/* Top Cards Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
              gap: '20px'
            }}>
              {/* Card 1: Total Queries */}
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)', fontWeight: 500 }}>Total Queries</span>
                <span style={{ fontSize: '36px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {stats?.summary?.total_queries || 0}
                </span>
                <div style={{ fontSize: '12px', color: 'var(--primary)', fontWeight: 500 }}>
                  Active Advisory Pipeline
                </div>
              </div>
              
              {/* Card 2: Average Latency */}
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)', fontWeight: 500 }}>Avg Latency</span>
                <span style={{ fontSize: '36px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#3b82f6' }}>
                  {stats?.summary?.average_latency_ms || 0} <span style={{ fontSize: '16px' }}>ms</span>
                </span>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  Target: 200-300ms slow path
                </div>
              </div>

              {/* Card 3: Satisfaction */}
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)', fontWeight: 500 }}>Farmer Rating</span>
                <span style={{ fontSize: '36px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: '#f59e0b' }}>
                  {stats?.summary?.satisfaction_rate || 100}%
                </span>
                <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  From {stats?.summary?.total_feedback || 0} feedback responses
                </div>
              </div>

              {/* Card 4: Safety Shield */}
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <span style={{ fontSize: '14px', color: 'var(--text-secondary)', fontWeight: 500 }}>Safety Guard Blocked</span>
                <span style={{ fontSize: '36px', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-red)' }}>
                  {stats?.summary?.safety_violation_rate || 0}%
                </span>
                <div style={{ fontSize: '12px', color: 'var(--accent-red)', fontWeight: 500 }}>
                  {stats?.summary?.blocked_queries || 0} violations intercepted
                </div>
              </div>
            </div>

            {/* Charts & Visual Analytics Row */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.8fr 1.2fr', gap: '20px' }}>
              {/* Chart 1: Volume Trend over last 7 days */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>Advisory Query Traffic (Past 7 Days)</h3>
                <div style={{ height: '240px', position: 'relative', display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', paddingBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                  {stats?.volume_trend?.map((item, idx) => {
                    const maxVal = Math.max(...stats.volume_trend.map(d => d.queries), 1);
                    const pct = (item.queries / maxVal) * 180; // scale
                    return (
                      <div key={idx} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '12%' }}>
                        <div style={{
                          height: `${Math.max(pct, 8)}px`,
                          width: '100%',
                          background: 'linear-gradient(to top, rgba(16, 185, 129, 0.8), rgba(59, 130, 246, 0.8))',
                          borderRadius: '4px 4px 0 0',
                          position: 'relative',
                          display: 'flex',
                          justifyContent: 'center',
                          transition: 'height 0.5s ease-out'
                        }}>
                          <span style={{ position: 'absolute', top: '-24px', fontSize: '11px', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                            {item.queries}
                          </span>
                        </div>
                        <span style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px', whiteSpace: 'nowrap' }}>
                          {item.date.substring(5)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Chart 2: Pathway volume and latency breakdown */}
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justify: 'space-between' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '15px' }}>Advisory Pipelines</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                  {[
                    { key: 'A', name: 'Text RAG (Pathway A)', desc: 'Semantic query synthesis', color: 'var(--primary)' },
                    { key: 'B', name: 'Leaf Scanner (Pathway B)', desc: 'ViT image diagnosis + RAG', color: 'var(--secondary)' },
                    { key: 'C', name: 'Yield lightGBM (Pathway C)', desc: 'Tabular regression model', color: '#f59e0b' },
                    { key: 'AB', name: 'Multimodal (Pathway AB)', desc: 'Photo upload + question', color: '#8b5cf6' }
                  ].map((pw) => {
                    const count = stats?.pathway_counts?.[pw.key] || 0;
                    const latency = stats?.pathway_latencies?.[pw.key] || 0;
                    const total = stats?.summary?.total_queries || 1;
                    const percent = Math.round((count / total) * 100);

                    return (
                      <div key={pw.key}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '13px' }}>
                          <div>
                            <span style={{ fontWeight: 600, color: pw.color }}>{pw.key}</span> — <span style={{ color: 'var(--text-primary)' }}>{pw.name}</span>
                          </div>
                          <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{count} ({percent}%)</span>
                        </div>
                        <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', position: 'relative', overflow: 'hidden', marginBottom: '4px' }}>
                          <div style={{ width: `${percent}%`, height: '100%', backgroundColor: pw.color, borderRadius: '3px' }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)' }}>
                          <span>{pw.desc}</span>
                          <span>Avg Latency: <strong style={{ color: 'var(--text-secondary)' }}>{latency} ms</strong></span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Disease distribution and Latency charts */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '20px' }}>
              {/* Disease breakdown card */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>Top Diagnosed Crop Conditions</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {stats?.disease_breakdown?.map((item, idx) => (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                      <span style={{ fontSize: '13px', fontWeight: 500 }}>{item.name}</span>
                      <span style={{
                        background: 'rgba(16, 185, 129, 0.1)',
                        color: 'var(--primary)',
                        padding: '2px 10px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        fontWeight: 600,
                        fontFamily: 'var(--font-mono)'
                      }}>
                        {item.value} times
                      </span>
                    </div>
                  )) || <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No leaf diseases diagnosed yet.</div>}
                </div>
              </div>

              {/* Latency distribution bar chart */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '20px' }}>Inference Latency Profile</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {stats?.latency_distribution?.map((item, idx) => {
                    const total = stats?.summary?.total_queries || 1;
                    const pct = Math.round((item.count / total) * 100);
                    return (
                      <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                        <span style={{ fontSize: '13px', color: 'var(--text-secondary)', width: '90px' }}>{item.bucket}</span>
                        <div style={{ flex: 1, height: '14px', background: 'rgba(255,255,255,0.04)', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${pct}%`,
                            height: '100%',
                            background: 'linear-gradient(to right, #3b82f6, #60a5fa)',
                            borderRadius: '4px'
                          }} />
                        </div>
                        <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', width: '60px', textAlign: 'right' }}>
                          {item.count} ({pct}%)
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: TELEMETRY LOGS */}
        {activeTab === 'logs' && (
          <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 700 }}>Telemetry Logs</h2>
              
              {/* Filters */}
              <div style={{ display: 'flex', gap: '14px' }}>
                <select 
                  value={pathwayFilter} 
                  onChange={(e) => setPathwayFilter(e.target.value)}
                  style={{
                    background: 'var(--bg-main)',
                    border: '1px solid var(--border-card)',
                    color: 'var(--text-primary)',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    fontFamily: 'var(--font-sans)',
                    outline: 'none'
                  }}
                >
                  <option value="">All Pathways</option>
                  <option value="A">Pathway A (Text RAG)</option>
                  <option value="B">Pathway B (Image ViT)</option>
                  <option value="C">Pathway C (Yield lightGBM)</option>
                  <option value="AB">Pathway AB (Multimodal)</option>
                </select>

                <select 
                  value={blockedFilter} 
                  onChange={(e) => setBlockedFilter(e.target.value)}
                  style={{
                    background: 'var(--bg-main)',
                    border: '1px solid var(--border-card)',
                    color: 'var(--text-primary)',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    fontFamily: 'var(--font-sans)',
                    outline: 'none'
                  }}
                >
                  <option value="">All Safety Statuses</option>
                  <option value="false">Safe Approved</option>
                  <option value="true">Guardrail Blocked</option>
                </select>

                <button onClick={fetchLogs} className="btn-glow" style={{ padding: '8px 16px', borderRadius: '8px', fontSize: '13px' }}>
                  Refresh Logs
                </button>
              </div>
            </div>

            {/* Logs Table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '12px' }}>Timestamp</th>
                    <th style={{ padding: '12px' }}>Pathway</th>
                    <th style={{ padding: '12px' }}>Input / Query Summary</th>
                    <th style={{ padding: '12px' }}>Intents</th>
                    <th style={{ padding: '12px' }}>Latency</th>
                    <th style={{ padding: '12px' }}>Status</th>
                    <th style={{ padding: '12px' }}>Feedback</th>
                    <th style={{ padding: '12px', textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log) => (
                    <tr key={log.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', transition: 'background 0.2s' }} className="table-row-hover">
                      <td style={{ padding: '12px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </td>
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontWeight: 600,
                          fontSize: '11px',
                          backgroundColor: log.pathway === 'A' ? 'rgba(16, 185, 129, 0.15)' : log.pathway === 'B' ? 'rgba(59, 130, 246, 0.15)' : log.pathway === 'C' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(139, 92, 246, 0.15)',
                          color: log.pathway === 'A' ? 'var(--primary)' : log.pathway === 'B' ? 'var(--secondary)' : log.pathway === 'C' ? '#f59e0b' : '#a78bfa'
                        }}>
                          Pathway {log.pathway}
                        </span>
                      </td>
                      <td style={{ padding: '12px', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {log.input_text || (log.image_path ? `Uploaded Photo [${log.image_path.split('/').pop()}]` : 'N/A')}
                      </td>
                      <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                        {log.intent?.join(', ') || 'general'}
                      </td>
                      <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        {log.latency_ms} ms
                      </td>
                      <td style={{ padding: '12px' }}>
                        {log.is_blocked ? (
                          <span style={{ color: 'var(--accent-red)', fontWeight: 600, display: 'flex', alignItems: 'center' }}>
                            <span className="dot dot-red" /> Blocked
                          </span>
                        ) : log.detected_crop ? (
                          <span style={{ color: 'var(--primary)', fontWeight: 500 }}>
                            Grounded ({log.detected_crop})
                          </span>
                        ) : (
                          <span style={{ color: 'var(--text-secondary)' }}>Grounded</span>
                        )}
                      </td>
                      <td style={{ padding: '12px' }}>
                        {log.feedback_score === 1 && <span style={{ color: 'var(--primary)' }}>👍 Pos</span>}
                        {log.feedback_score === -1 && <span style={{ color: 'var(--accent-red)' }}>👎 Neg</span>}
                        {log.feedback_score === null && <span style={{ color: 'var(--text-muted)' }}>None</span>}
                      </td>
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        <button 
                          onClick={() => setSelectedLog(log)}
                          className="btn-glow" 
                          style={{
                            padding: '4px 10px',
                            borderRadius: '6px',
                            fontSize: '11px',
                            background: 'rgba(255,255,255,0.05)',
                            color: 'var(--text-primary)',
                            boxShadow: 'none',
                            border: '1px solid rgba(255,255,255,0.08)'
                          }}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  ))}
                  {logs.length === 0 && (
                    <tr>
                      <td colSpan="8" style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
                        No logs match filters or API server is offline.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 3: CONFIGURATION CONTROL */}
        {activeTab === 'config' && (
          <div style={{ maxWidth: '700px', margin: '0 auto' }}>
            <form onSubmit={saveConfig} className="glass-panel" style={{ padding: '30px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '15px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 700 }}>Orchestrator Settings</h2>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                  Calibrate RAG similarity thresholds, source priorities, and model environments.
                </p>
              </div>

              {/* Threshold 1: Grounded */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                  <label style={{ fontWeight: 600 }}>Grounded Relevance Threshold (TIER_GROUNDED)</label>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)', fontWeight: 700 }}>
                    {config.tier_grounded}
                  </span>
                </div>
                <input 
                  type="range" min="0.4" max="0.9" step="0.01" 
                  value={config.tier_grounded} 
                  onChange={(e) => setConfig({ ...config, tier_grounded: parseFloat(e.target.value) })}
                  style={{ accentColor: 'var(--primary)', cursor: 'pointer' }}
                />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Minimum cosine similarity score required to answer directly from vector DB. Default is 0.66.
                </span>
              </div>

              {/* Threshold 2: Fallback */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                  <label style={{ fontWeight: 600 }}>Fallback Warning Threshold (TIER_FALLBACK)</label>
                  <span style={{ fontFamily: 'var(--font-mono)', color: '#3b82f6', fontWeight: 700 }}>
                    {config.tier_fallback}
                  </span>
                </div>
                <input 
                  type="range" min="0.3" max="0.75" step="0.01" 
                  value={config.tier_fallback} 
                  onChange={(e) => setConfig({ ...config, tier_fallback: parseFloat(e.target.value) })}
                  style={{ accentColor: '#3b82f6', cursor: 'pointer' }}
                />
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Scores between Fallback and Grounded will trigger the 'verify with KVK' disclaimer notice. Default is 0.56.
                </span>
              </div>

              {/* Weights: PDF vs KCC */}
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '15px' }}>
                <h3 style={{ fontSize: '14px', fontWeight: 600, marginBottom: '14px' }}>RAG Source Weight Factors</h3>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  {/* Weight 1 */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <label style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>PDF Weight (Policy Intents)</label>
                    <input 
                      type="number" step="0.1" min="0.1" max="5.0"
                      value={config.weight_pdf_policy}
                      onChange={(e) => setConfig({ ...config, weight_pdf_policy: parseFloat(e.target.value) })}
                      style={{ background: 'var(--bg-main)', border: '1px solid var(--border-card)', color: '#fff', padding: '8px', borderRadius: '6px', outline: 'none' }}
                    />
                  </div>
                  {/* Weight 2 */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <label style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>KCC Weight (Policy Intents)</label>
                    <input 
                      type="number" step="0.1" min="0.1" max="5.0"
                      value={config.weight_kcc_policy}
                      onChange={(e) => setConfig({ ...config, weight_kcc_policy: parseFloat(e.target.value) })}
                      style={{ background: 'var(--bg-main)', border: '1px solid var(--border-card)', color: '#fff', padding: '8px', borderRadius: '6px', outline: 'none' }}
                    />
                  </div>
                </div>
              </div>

              {/* Cloud AI Toggle */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '20px' }}>
                <div>
                  <label style={{ fontWeight: 600, display: 'block', fontSize: '14px' }}>Bypass Cloud APIs (Offline Demo Mode)</label>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    When enabled, local pre-programmed mocks are run instead of sending requests to Gemini.
                  </span>
                </div>
                <input 
                  type="checkbox"
                  checked={config.mock_models}
                  onChange={(e) => setConfig({ ...config, mock_models: e.target.checked })}
                  style={{ width: '22px', height: '22px', accentColor: 'var(--primary)', cursor: 'pointer' }}
                />
              </div>

              <button type="submit" className="btn-glow" style={{ padding: '14px', borderRadius: '10px', fontSize: '14px', marginTop: '10px' }}>
                Save Dynamic Configurations
              </button>
            </form>
          </div>
        )}

        {/* TAB 4: VECTOR DB */}
        {activeTab === 'vectordb' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            
            {/* Collection Metadata panel */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '16px' }}>Vector Space Information</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)' }}>Active Collection</span>
                  <span style={{ fontSize: '16px', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>
                    {vectorDb?.collection || 'agri_knowledge'}
                  </span>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)' }}>Total Chunk Points</span>
                  <span style={{ fontSize: '16px', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {vectorDb?.points_count ? vectorDb.points_count.toLocaleString() : '723,439'}
                  </span>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)' }}>Vector Dimension</span>
                  <span style={{ fontSize: '16px', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                    {vectorDb?.vector_size || 1024} (bge-m3)
                  </span>
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)' }}>DB Metric</span>
                  <span style={{ fontSize: '16px', fontWeight: 600 }}>
                    Cosine Distance
                  </span>
                </div>
              </div>
            </div>

            {/* Playground semantic search test */}
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>Semantic Search Playground (RAG Debugger)</h3>
              
              <form onSubmit={testVectorSearch} style={{ display: 'flex', gap: '14px', marginBottom: '24px' }}>
                <input 
                  type="text" 
                  placeholder="Enter farmer question or search terms (e.g. wheat yellow rust dosage, PM KISAN eligibility)..."
                  value={vectorSearchQuery}
                  onChange={(e) => setVectorSearchQuery(e.target.value)}
                  style={{
                    flex: 1,
                    background: 'var(--bg-main)',
                    border: '1px solid var(--border-card)',
                    color: '#fff',
                    padding: '14px 18px',
                    borderRadius: '10px',
                    outline: 'none',
                    fontSize: '14px'
                  }}
                />
                <button type="submit" disabled={searchingVector} className="btn-glow" style={{ padding: '14px 28px', borderRadius: '10px' }}>
                  {searchingVector ? 'Searching...' : 'Test Retrieval'}
                </button>
              </form>

              {vectorSearchResults && (
                <div style={{
                  background: 'rgba(0,0,0,0.15)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '10px',
                  padding: '20px',
                  fontSize: '13px'
                }}>
                  <pre style={{
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-primary)',
                    lineHeight: '1.6'
                  }}>
                    {vectorSearchResults.content?.[0]?.text || 'No results returned.'}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer style={{
        marginTop: 'auto',
        padding: '20px',
        textAlign: 'center',
        fontSize: '12px',
        color: 'var(--text-muted)',
        borderTop: '1px solid rgba(255,255,255,0.04)'
      }}>
        FarmerVision advisory controller. Built under Capstone guidelines. Indian Agronomic data index.
      </footer>

      {/* INSPECTOR MODAL */}
      {selectedLog && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'rgba(0,0,0,0.7)',
          backdropFilter: 'blur(5px)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 9999,
          padding: '20px'
        }} onClick={() => setSelectedLog(null)}>
          <div className="glass-panel" style={{
            width: '100%',
            maxWidth: '850px',
            maxHeight: '90vh',
            overflowY: 'auto',
            background: 'var(--bg-main)',
            padding: '30px',
            borderRadius: 'var(--radius-lg)'
          }} onClick={(e) => e.stopPropagation()}>
            {/* Modal header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '15px', marginBottom: '20px' }}>
              <div>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>LOG IDENTIFIER: #{selectedLog.id}</span>
                <h3 style={{ fontSize: '18px', fontWeight: 700 }}>Telemetry Detail</h3>
              </div>
              <button 
                onClick={() => setSelectedLog(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', fontSize: '24px', cursor: 'pointer' }}
              >
                &times;
              </button>
            </div>

            {/* Modal content */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Row 1: Pathway & Latency */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)' }}>PIPELINE PATHWAY</span>
                  <span style={{ fontWeight: 700, color: 'var(--primary)' }}>Pathway {selectedLog.pathway}</span>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)' }}>EXECUTION LATENCY</span>
                  <span style={{ fontWeight: 700, color: 'var(--secondary)', fontFamily: 'var(--font-mono)' }}>{selectedLog.latency_ms} ms</span>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)' }}>DETECTED CROP</span>
                  <span style={{ fontWeight: 700 }}>{selectedLog.detected_crop ? selectedLog.detected_crop.toUpperCase() : 'NONE'}</span>
                </div>
              </div>

              {/* Leaf Image (if Pathway B/AB) */}
              {selectedLog.image_path && (
                <div>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>UPLOADED LEAF PHOTO</span>
                  <div style={{
                    width: '100%',
                    maxHeight: '280px',
                    borderRadius: '10px',
                    overflow: 'hidden',
                    background: 'rgba(0,0,0,0.2)',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    border: '1px solid rgba(255,255,255,0.08)'
                  }}>
                    <img 
                      src={`${API_BASE}${selectedLog.image_path}`} 
                      alt="Leaf disease diagnosis" 
                      style={{ maxHeight: '280px', objectFit: 'contain' }}
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                </div>
              )}

              {/* Farmer Input Text */}
              {selectedLog.input_text && (
                <div>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px', fontWeight: 600 }}>FARMER INPUT TEXT</span>
                  <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', fontSize: '14px' }}>
                    "{selectedLog.input_text}"
                  </div>
                </div>
              )}

              {/* Blocked parameters */}
              {selectedLog.is_blocked && (
                <div style={{ border: '1px solid var(--accent-red)', background: 'rgba(239, 68, 68, 0.05)', padding: '12px', borderRadius: '8px' }}>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--accent-red)', fontWeight: 700 }}>GUARDRAIL VIOLATION FLAG</span>
                  <span style={{ fontSize: '13px' }}>{selectedLog.guardrail_reason || 'Pesticide/dosage safety limit breached.'}</span>
                </div>
              )}

              {/* Retrieved Chunks */}
              {selectedLog.retrieved_chunks?.length > 0 && (
                <div>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 600 }}>RETRIEVED RAG CONTEXTS</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflowY: 'auto' }}>
                    {selectedLog.retrieved_chunks.map((chunk, idx) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.03)', fontSize: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '4px', fontSize: '10px' }}>
                          <span>[{idx+1}] Source: {chunk.source_type}</span>
                          <span>Score: {chunk.score ? chunk.score.toFixed(3) : 'N/A'}</span>
                        </div>
                        <p>{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Synthesized Response */}
              <div>
                <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px', fontWeight: 600 }}>SYNTHESIZED ADVISORY ANSWER</span>
                <div style={{ background: 'rgba(16, 185, 129, 0.04)', border: '1px solid rgba(16, 185, 129, 0.1)', padding: '16px', borderRadius: '8px', fontSize: '14px', lineHeight: '1.6', color: '#e5e7eb' }}>
                  {selectedLog.synthesis_response}
                </div>
              </div>

              {/* Feedback text */}
              {selectedLog.feedback_text && (
                <div>
                  <span style={{ display: 'block', fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px', fontWeight: 600 }}>FARMER REVIEW COMMENT</span>
                  <div style={{ background: 'rgba(245, 158, 11, 0.04)', border: '1px solid rgba(245, 158, 11, 0.15)', padding: '12px', borderRadius: '8px', fontSize: '13px', fontStyle: 'italic' }}>
                    "{selectedLog.feedback_text}"
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
