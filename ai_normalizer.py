import os
import json
import unicodedata
from typing import List, Dict, Optional

def _get_api_key() -> Optional[str]:
    """Retrieve OPENAI_API_KEY from Streamlit secrets or environment.

    Streamlit Cloud: add OPENAI_API_KEY in app settings → Secrets.
    Local dev: set OPENAI_API_KEY environment variable.
    """
    # Try Streamlit secrets first (if running inside Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY")


def suggest_mapping(headers: List[str], sample_rows: List[List[str]]) -> Optional[Dict[str, str]]:
    """
    Suggest a mapping from original headers to canonical keys using an LLM, if available.
    Canonical keys: date, description, debit, credit, amount, balance, currency.

    Returns a dict like {"Original Header": "canonical_key"} or None if unavailable.

    Configuration:
    - Set OPENAI_API_KEY in environment.
    - Optionally set OPENAI_MODEL (default: gpt-4o-mini).
    - Requires 'openai' Python package installed.
    """
    # Retrieve API key securely
    api_key = _get_api_key()
    if not api_key:
        return None

    # --- Local heuristic fallback (used if AI is unavailable or returns no mapping) ---
    def _strip_accents(s: str) -> str:
        try:
            return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        except Exception:
            return s

    HEADER_CANDIDATES = {
        "date": ["date", "txn date", "posting date", "fecha"],
        "description": [
            "description", "details", "narration", "payee", "particulars",
            "concepto", "descripcion", "descripción", "mas datos", "más datos",
        ],
        "debit": ["debit", "withdrawal", "dr", "out", "cargo", "retiro"],
        "credit": ["credit", "deposit", "cr", "in", "abono", "ingreso"],
        "amount": ["amount", "value", "amt", "importe", "valor"],
        "balance": ["balance", "closing balance", "avail balance", "saldo"],
        "currency": ["currency", "moneda", "divisa", "curr", "iso", "code"],
    }

    def _match_header(col_name: str) -> Optional[str]:
        raw = (col_name or "").strip().lower()
        name = _strip_accents(raw)
        for key, variants in HEADER_CANDIDATES.items():
            if name == key or key in name:
                return key
            for v in variants:
                vn = _strip_accents(v.lower())
                if name.startswith(vn) or vn in name:
                    return key
        return None

    def _heuristic_mapping(hdrs: List[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for h in hdrs:
            m = _match_header(str(h))
            if m:
                # Map original header to canonical key
                mapping[str(h)] = m
        return mapping

    try:
        # prefer new SDK style
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        system = (
            "You are a data normalization assistant. Map spreadsheet headers to a canonical schema. "
            "Respond ONLY with JSON that includes a 'mapping' object, where keys are original headers and "
            "values are one of: date, description, debit, credit, amount, balance, currency. "
            "If unsure, pick the closest reasonable option."
        )
        # Ensure headers are strings for JSON encoding
        headers = [str(h) for h in headers]
        user = json.dumps({
            "headers": headers,
            "sample_rows": sample_rows[:10],
            "canonical_keys": ["date", "description", "debit", "credit", "amount", "balance", "currency"],
        })

        # Try the chat completions API
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
        except Exception:
            # Fallback to responses.create if chat.completions not available
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = getattr(resp, "output_text", None) or json.dumps({"output": getattr(resp, "output", None)})

        # Robust JSON parsing
        def _try_parse(text: str) -> Optional[Dict[str, str]]:
            try:
                data = json.loads(text)
                mapping = data.get("mapping")
                if isinstance(mapping, dict) and mapping:
                    normalized = {}
                    for orig, canon in mapping.items():
                        normalized[str(orig)] = str(canon).strip().lower()
                    return normalized
            except Exception:
                return None

        parsed = _try_parse(content)
        if parsed:
            return parsed
        # Extract inner JSON block if wrapped in text
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            parsed = _try_parse(content[start:end])
            if parsed:
                return parsed
        # If AI returns nothing useful, fall back to heuristic mapping
        hmap = _heuristic_mapping(headers)
        return hmap if hmap else None
    except Exception:
        # openai not installed or other issue; gracefully fall back
        hmap = _heuristic_mapping(headers)
        return hmap if hmap else None