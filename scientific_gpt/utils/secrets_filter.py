"""Best-effort redaction of API-key-like tokens from error strings.

Streamlit shows tracebacks verbatim in the UI; without scrubbing, a 401/403
from a provider can leak the key value (some SDKs put the key in URLs or
echo headers in error messages).
"""
from __future__ import annotations
import re

# 24+ chars of base64-ish alphabet — covers Google AIza…, OpenAI sk-…, Anthropic
# sk-ant-…, Groq gsk_…, CORE bearer tokens, etc.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{24,}")
# Bearer / Authorization headers
_AUTH_RE = re.compile(r"(?i)(bearer|api[_-]?key|authorization)\s*[:=]\s*\S+")


def scrub(text: str) -> str:
    if not text:
        return text
    text = _AUTH_RE.sub(r"\1: ***", text)
    text = _TOKEN_RE.sub("***", text)
    return text
