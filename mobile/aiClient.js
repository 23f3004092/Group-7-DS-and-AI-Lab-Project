// FarmerVision AI client — GCP-deployed RAG + vision service.
// Endpoint contract: API_SPEC.md at repo root (Base URL + X-API-Key).
// Base URL and API key are read from .env (EXPO_PUBLIC_AI_API_URL / EXPO_PUBLIC_AI_API_KEY).

const AI_BASE_URL = (process.env.EXPO_PUBLIC_AI_API_URL || '').replace(/\/+$/, '');
const AI_API_KEY = process.env.EXPO_PUBLIC_AI_API_KEY || '';

export const AI_CONFIGURED = Boolean(AI_BASE_URL && AI_API_KEY);
export const AI_BASE_URL_VALUE = AI_BASE_URL;

// Optional local proxy (the FastAPI backend's /ai routes). Set for web builds:
// browsers block direct calls to the GCP deployment because it sends no CORS
// headers, so web requests go through the local backend instead.
let AI_PROXY_URL = '';

export function setAiProxyUrl(url) {
  AI_PROXY_URL = url ? url.replace(/\/+$/, '') : '';
}

async function request(path, { method = 'POST', body, headers = {}, formData } = {}) {
  const opts = {
    method,
    headers: { ...(AI_PROXY_URL ? {} : { 'X-API-Key': AI_API_KEY }), ...headers },
  };
  if (formData) {
    opts.body = formData;
  } else if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const base = AI_PROXY_URL || AI_BASE_URL;
  const res = await fetch(`${base}${path}`, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.detail) detail = j.detail;
    } catch (e) { /* non-JSON error body */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// GET /health — liveness + what's loaded (no key needed by the API)
export async function checkHealth() {
  return request('/health', { method: 'GET' });
}

// POST /classify — intent, guardrail, and which external data the query needs
export async function classify(query) {
  return request('/classify', { body: { query } });
}

// POST /query — main grounded, cited answer endpoint (multi-turn + live_data)
export async function ask(query, { intent, sessionId, liveData, topK } = {}) {
  const body = { query };
  if (intent) body.intent = intent;
  if (sessionId) body.session_id = sessionId;
  if (liveData) body.live_data = liveData;
  if (topK) body.top_k = topK;
  return request('/query', { body });
}

// POST /diagnose — leaf photo (+ optional question) -> disease + grounded treatment
export async function diagnose({ uri, name = 'leaf.jpg', question } = {}) {
  const formData = new FormData();
  formData.append('file', { uri, name, type: imageMime(name) });
  if (question) formData.append('question', question);
  return request('/diagnose', { formData });
}

// POST /vision — leaf photo -> disease label + confidence only
export async function vision({ uri, name = 'leaf.jpg' } = {}) {
  const formData = new FormData();
  formData.append('file', { uri, name, type: imageMime(name) });
  return request('/vision', { formData });
}

// Maps API sources ({n, score, source_type, citation}) to the shape the chat UI
// expects ({rank, score, source_type, text}).
export function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.map((src) => {
    const c = src.citation || {};
    let text = '';
    if (c.corpus === 'pdf') {
      text = [c.file, c.pages ? `pp. ${c.pages.join('-')}` : null, c.section, c.doc_category, c.district, c.year]
        .filter(Boolean).join(', ');
    } else if (c.corpus === 'kcc') {
      text = [c.query_type, c.crop, c.district, c.season, c.year].filter(Boolean).join(', ');
    } else {
      text = src.summary || src.text || '';
    }
    return {
      rank: src.n != null ? src.n : src.rank,
      score: src.score,
      source_type: src.source_type,
      text,
    };
  });
}

function imageMime(name) {
  return /\.png$/i.test(name || '') ? 'image/png' : 'image/jpeg';
}