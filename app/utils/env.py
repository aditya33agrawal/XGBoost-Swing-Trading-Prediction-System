"""Secrets bootstrap — fixes the bug where the app never loaded .env.

Import this module (for its side effect) before reading any secret.
`load_dotenv()` was previously never called anywhere in the repo, so
locally os.getenv("SUPABASE_URL") always returned "" even with a valid
.env file present, silently forcing JSON-fallback mode.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False


def get_secret(*keys: str, default: str = "") -> str:
    """Return the first non-empty value found across `keys`, checking
    st.secrets first (Streamlit Cloud) then os.environ (local .env), in
    the order the keys are given. Never raises."""
    if _HAS_STREAMLIT:
        try:
            for k in keys:
                v = st.secrets.get(k, "")
                if v:
                    return v
        except Exception:
            pass
    for k in keys:
        v = os.getenv(k, "")
        if v:
            return v
    return default


def get_supabase_url() -> str:
    return get_secret("SUPABASE_URL")


def get_supabase_key() -> str:
    """Prefer the write-capable secret key, fall back to publishable/legacy."""
    return get_secret("SUPABASE_SECRET_KEY", "SUPABASE_KEY", "SUPABASE_PUBLISHABLE_KEY")
