"""Lightweight LLM adapter stub.

This adapter calls a configurable managed LLM HTTP endpoint defined by the
environment variable `LLM_API_URL` (POST with JSON {"prompt": ...}). If not
configured, falls back to a placeholder message or uses the existing
`SKIP_GENERATOR` behaviour in the main app.

Extend this module to integrate with Bedrock/Vertex/Replicate SDKs as needed.
"""
import os
import requests
from typing import Optional

LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


def call_managed_llm(prompt: str, timeout: int = 15) -> str:
    """Call the managed LLM endpoint synchronously and return text.

    Expected behavior: POST {"prompt": prompt} to `LLM_API_URL`, with optional
    `Authorization: Bearer <LLM_API_KEY>` header. Response should be JSON
    containing `text`.
    """
    if not LLM_API_URL:
        raise RuntimeError("LLM_API_URL not configured")

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {"prompt": prompt}
    resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Support multiple possible keys
    return data.get("text") or data.get("output") or data.get("result") or ""


def safe_generate(prompt: str, hits: list) -> str:
    """High-level generate wrapper used by the app.

    If managed LLM is configured, call it. Otherwise raise to let caller fall back.
    """
    # Build a compact context from hits
    ctx = "\n\n".join(f"[{i+1}] {getattr(h.payload, 'text', h.payload.get('text','') )[:300]}" for i, h in enumerate(hits))
    full_prompt = (
        "You are a helpful agricultural advisor for Indian farmers.\n"
        "Answer using ONLY the provided context. Cite sources at the end of sentences.\n\n"
        f"Context:\n{ctx}\n\nFarmer's question: {prompt}\n\nAnswer:"
    )

    return call_managed_llm(full_prompt)
