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

// Render's free tier sleeps after ~15 min of idle, so the first request of a
// session can fail while the instance cold-boots. Retry transient failures
// (network errors / 5xx / 429) with a short backoff so the first diagnose or
// chat message self-heals once the service is warm again. 4xx responses are
// real errors (e.g. bad key / validation) and are not retried.
export async function withAiRetry(fn, { attempts = 3, baseDelay = 5000 } = {}) {
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const status = err && err.status;
      const retriable = status === undefined || status === 0 || status >= 500 || status === 429;
      if (!retriable || i === attempts - 1) throw err;
      await new Promise((resolve) => setTimeout(resolve, baseDelay * (i + 1)));
    }
  }
  throw lastErr;
}

// GET /health — liveness + what's loaded (no key needed by the API)
export async function checkHealth() {
  return withAiRetry(() => request('/health', { method: 'GET' }), { attempts: 2, baseDelay: 3000 });
}

// POST /classify — intent, guardrail, and which external data the query needs
export async function classify(query) {
  return withAiRetry(() => request('/classify', { body: { query } }), { attempts: 2, baseDelay: 3000 });
}

// POST /query — main grounded, cited answer endpoint (multi-turn + live_data)
// include_content=true makes the AI service return the actual chunk text in
// each source, so the citation page can show the advisory content (not just
// file/page/crop metadata).
export async function ask(query, { intent, sessionId, liveData, topK } = {}) {
  const body = { query };
  if (intent) body.intent = intent;
  if (sessionId) body.session_id = sessionId;
  if (liveData) body.live_data = liveData;
  if (topK) body.top_k = topK;
  body.include_content = true;
  return withAiRetry(() => request('/query', { body }), { attempts: 4, baseDelay: 5000 });
}

// Adds an image to a multipart FormData in a platform-safe way.
// React Native needs the {uri, name, type} object; on web/browsers a plain
// object would serialize as "[object Object]", so a real File/Blob is appended.
export async function appendImage(formData, { uri, name = 'leaf.jpg', type } = {}) {
  const mime = type || imageMime(name);
  if (typeof window !== 'undefined' && typeof window.document !== 'undefined') {
    const blob = await (await fetch(uri)).blob();
    formData.append('file', blob, name);
  } else {
    formData.append('file', { uri, name, type: mime });
  }
}

// Multipart file upload that does NOT go through RN's fetch/FormData stack.
// React Native release builds (New Architecture) can fail FormData file uploads
// on-device before any bytes reach the network; expo-file-system's native
// uploader (OkHttp direct) avoids that path entirely. Web keeps fetch+FormData.
async function uploadFile(base, path, { uri, name = 'leaf.jpg', mimeType, fields, headers = {} } = {}) {
  if (typeof window !== 'undefined' && typeof window.document !== 'undefined') {
    const formData = new FormData();
    await appendImage(formData, { uri, name });
    for (const [k, v] of Object.entries(fields || {})) formData.append(k, v);
    return request(path, { formData });
  }
  // Lazy require: keeps the native module out of web bundles' runtime path.
  const { File, UploadType } = require('expo-file-system');
  const res = await new File(uri).upload(`${base}${path}`, {
    uploadType: UploadType.MULTIPART,
    fieldName: 'file',
    mimeType: mimeType || imageMime(name),
    headers,
    parameters: fields,
  });
  let data = null;
  try { data = JSON.parse(res.body); } catch (e) { /* non-JSON body */ }
  if (res.status < 200 || res.status >= 300) {
    const err = new Error((data && data.detail) || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

function aiUploadHeaders() {
  return AI_PROXY_URL ? {} : { 'X-API-Key': AI_API_KEY };
}

// POST /diagnose — leaf photo (+ optional question) -> disease + grounded treatment
export async function diagnose({ uri, name = 'leaf.jpg', question } = {}) {
  return withAiRetry(() => uploadFile(AI_PROXY_URL || AI_BASE_URL, '/diagnose', {
    uri,
    name,
    fields: question ? { question } : undefined,
    headers: aiUploadHeaders(),
  }), { attempts: 3, baseDelay: 5000 });
}

// POST /vision — leaf photo -> disease label + confidence only
export async function vision({ uri, name = 'leaf.jpg' } = {}) {
  return withAiRetry(() => uploadFile(AI_PROXY_URL || AI_BASE_URL, '/vision', {
    uri,
    name,
    headers: aiUploadHeaders(),
  }), { attempts: 4, baseDelay: 5000 });
}

// POST /api/query/vision — local backend's ViT-only fallback endpoint.
export async function backendVision(apiUrl, { uri, name = 'leaf.jpg' } = {}) {
  return uploadFile(apiUrl, '/api/query/vision', { uri, name });
}

// POST /api/query/image — local backend's chat-photo diagnosis fallback.
export async function backendImage(apiUrl, { uri, name = 'leaf.jpg' } = {}) {
  return uploadFile(apiUrl, '/api/query/image', { uri, name });
}

// Maps API sources ({n, score, source_type, citation}) to the shape the chat UI
// expects ({rank, score, source_type, text}).
export function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.map((src) => {
    const c = src.citation || {};
    let text = '';
    let name = '';
    if (c.corpus === 'pdf') {
      const file = (c.file || '').replace(/\.pdf$/i, '');
      text = [c.file, c.pages ? `pp. ${c.pages.join('-')}` : null, c.section, c.doc_category, c.district, c.year]
        .filter(Boolean).join(', ');
      name = file || src.summary || src.source_type;
    } else if (c.corpus === 'kcc') {
      text = [c.query_type, c.crop, c.district, c.season, c.year].filter(Boolean).join(', ');
      name = c.query_type || c.record || src.source_type;
    } else {
      text = src.summary || src.text || '';
      name = src.summary || src.source_type;
    }
    return {
      id: src.id,
      // GCP returns the chunk text under `content` (only when /query was sent
      // with include_content=true); carry it as full_text so the citation page
      // can show the advisory without a second fetch.
      full_text: src.full_text || src.content || '',
      rank: src.n != null ? src.n : src.rank,
      score: src.score,
      source_type: src.source_type,
      name,
      text: src.full_text || src.content || text,
      citation: c,
    };
  });
}

function imageMime(name) {
  return /\.png$/i.test(name || '') ? 'image/png' : 'image/jpeg';
}