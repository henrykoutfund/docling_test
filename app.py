import os
import io
import json
import re
import hashlib
import tempfile
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import unicodedata

import pandas as pd
import streamlit as st
from docling.document_converter import DocumentConverter
import pdfplumber
from ai_normalizer import suggest_mapping
from agui import demo_events, build_event


def save_uploaded_file(uploaded_file) -> Tuple[str, bytes, str]:
    """Save the uploaded file to a temporary path and return (path, bytes, sha1)."""
    # Streamlit UploadedFile supports getvalue; fallback to read()
    try:
        data = uploaded_file.getvalue()
    except Exception:
        data = uploaded_file.read()
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    src_hash = hashlib.sha1(data).hexdigest()
    return path, data, src_hash


def export_docling_json(conversion_result) -> Dict[str, Any]:
    """Export Docling document to JSON dict."""
    try:
        json_str = conversion_result.document.export_to_json()
        return json.loads(json_str)
    except Exception:
        # Fallback to markdown capture if JSON export fails
        return {"error": "json_export_failed", "markdown": conversion_result.document.export_to_markdown()}


def iter_nodes(node: Dict[str, Any]):
    """Yield all nodes in a Docling JSON tree."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        children = current.get("children") or []
        # Some structures may use 'content' instead of 'children'
        content = current.get("content") or []
        for child in children:
            stack.append(child)
        for child in content:
            stack.append(child)


def table_cells_to_dataframe(attrs: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Convert Docling table attrs (cells with row/col/text) to pandas DataFrame."""
    cells = attrs.get("cells") or attrs.get("table_cells")
    if not cells:
        return None
    # Determine grid size
    max_row = 0
    max_col = 0
    for c in cells:
        max_row = max(max_row, int(c.get("row", 0)))
        max_col = max(max_col, int(c.get("col", 0)))

    n_rows = max_row + 1
    n_cols = max_col + 1
    grid: List[List[str]] = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in cells:
        r = int(c.get("row", 0))
        k = int(c.get("col", 0))
        text = c.get("text", "")
        if r < n_rows and k < n_cols:
            grid[r][k] = text

    # First row considered headers if non-empty
    headers = [h.strip() or f"col_{i}" for i, h in enumerate(grid[0])]
    df = pd.DataFrame(grid[1:], columns=headers)
    return df


def extract_tables(json_doc: Dict[str, Any]) -> List[pd.DataFrame]:
    """Extract tables from Docling JSON as DataFrames."""
    tables: List[pd.DataFrame] = []
    for node in iter_nodes(json_doc):
        if node.get("type") == "table":
            attrs = node.get("attrs", {})
            df = table_cells_to_dataframe(attrs)
            if df is not None and not df.empty:
                tables.append(df)
    return tables


def extract_tables_pdfplumber(pdf_path: str, strategy: str = "lines") -> List[pd.DataFrame]:
    """Extract tables from a PDF using pdfplumber as a fallback.
    Returns list of DataFrames (first row considered header when present).
    """
    tables_out: List[pd.DataFrame] = []
    table_settings = {}
    if strategy == "lines":
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 5,
        }
    else:
        # Text-based segmentation can work for statements without ruler lines
        table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
        }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Normalize rotation
                rotation = getattr(page, "rotation", 0) or 0
                page_to_use = page.rotate(-rotation) if rotation else page
                raw_tables = page_to_use.extract_tables(table_settings)
                for t in raw_tables or []:
                    if not t:
                        continue
                    # If header present, use it; else autogenerate
                    header = t[0] if len(t) > 1 else None
                    rows = t[1:] if header else t
                    if header and any(h and str(h).strip() for h in header):
                        cols = [str(h).strip() or f"col_{i}" for i, h in enumerate(header)]
                        df = pd.DataFrame(rows, columns=cols)
                    else:
                        df = pd.DataFrame(rows)
                    if not df.empty:
                        tables_out.append(df)
    except Exception:
        # Silently ignore fallback errors to avoid breaking primary flow
        pass

    return tables_out


def strip_accents(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s


def extract_tables_pdfplumber_by_headers(pdf_path: str) -> List[pd.DataFrame]:
    """Header-guided extraction for unruled tables.
    Looks for a line containing Spanish headers like 'Fecha', 'Concepto', 'Más datos', 'Importe', 'Saldo',
    then builds rows by assigning words to column regions.
    """
    out: List[pd.DataFrame] = []
    header_variants = [
        ["fecha"],
        ["concepto"],
        ["mas", "datos"],  # 'más datos' without accents for matching
        ["importe"],
        ["saldo"],
    ]
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(keep_blank_chars=False, use_text_flow=True, y_tolerance=2)
                # Group words by line using 'top' with tolerance
                lines: List[List[dict]] = []
                for w in sorted(words, key=lambda x: (x.get("top", 0), x.get("x0", 0))):
                    if not lines:
                        lines.append([w])
                        continue
                    last_line = lines[-1]
                    if abs(w.get("top", 0) - last_line[-1].get("top", 0)) <= 2:
                        last_line.append(w)
                    else:
                        lines.append([w])

                # Find header line
                header_line = None
                header_x_positions: List[float] = []
                for ln in lines:
                    texts = [strip_accents(str(a["text"]).lower()) for a in ln]
                    # Try to match phrases
                    found_positions = []
                    i = 0
                    while i < len(texts):
                        matched = False
                        for variant in header_variants:
                            L = len(variant)
                            chunk = texts[i:i+L]
                            if len(chunk) == L and all(variant[k] in chunk[k] for k in range(L)):
                                found_positions.append(ln[i]["x0"])  # x0 of first token in variant
                                i += L
                                matched = True
                                break
                        if not matched:
                            i += 1
                    # Heuristic: if we matched at least 3 headers on the line, treat it as header
                    if len(found_positions) >= 3:
                        header_line = ln
                        header_x_positions = sorted(found_positions)
                        break

                if not header_line:
                    continue

                # Define column boundaries from header x positions
                header_bottom = max([a.get("bottom", a.get("top", 0)) for a in header_line])
                xs = header_x_positions
                bounds: List[Tuple[float, float]] = []
                for idx, x in enumerate(xs):
                    left = xs[idx - 1] if idx > 0 else 0
                    right = xs[idx + 1] if idx < len(xs) - 1 else page.width
                    # Midpoints between adjacent header starts
                    col_left = (left + x) / 2 if idx > 0 else 0
                    col_right = (x + right) / 2 if idx < len(xs) - 1 else page.width
                    bounds.append((col_left, col_right))

                # Collect lines below header and assign words to columns
                rows: List[List[str]] = []
                for ln in lines:
                    top = ln[0].get("top", 0)
                    if top <= header_bottom + 2:
                        continue
                    cols_text = [""] * len(bounds)
                    for w in ln:
                        x = w.get("x0", 0)
                        # Find column by x position
                        for ci, (l, r) in enumerate(bounds):
                            if l <= x < r:
                                txt = w.get("text", "")
                                cols_text[ci] = (cols_text[ci] + " " + txt).strip()
                                break
                    # Skip empty lines
                    if any(t.strip() for t in cols_text):
                        rows.append(cols_text)

                # Build DataFrame with expected header names
                cols = ["Fecha", "Concepto", "Más datos", "Importe", "Saldo"][:len(bounds)]
                df = pd.DataFrame(rows, columns=cols)
                if not df.empty:
                    out.append(df)
    except Exception:
        pass
    return out


def extract_tables_from_markdown(md: str) -> List[pd.DataFrame]:
    """Parse GitHub-style pipe tables from Markdown into DataFrames.
    Looks for blocks with a header row, a separator row (---/ :---:/ etc.), then data rows.
    """
    tables: List[pd.DataFrame] = []
    if not md:
        return tables
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip().startswith("|"):
            i += 1
            continue
        header_line = lines[i].strip()
        if i + 1 >= len(lines):
            break
        sep_line = lines[i + 1].strip()
        # Validate separator: must contain only dashes/colons/pipes/spaces
        if not sep_line.startswith("|"):
            i += 1
            continue
        tokens = [t.strip() for t in sep_line.strip('|').split('|')]
        if not tokens or not all(re.fullmatch(r"[:\- ]+", t or "-") for t in tokens):
            i += 1
            continue

        # Collect data rows
        rows = []
        j = i + 2
        while j < len(lines) and lines[j].strip().startswith('|'):
            rows.append(lines[j].strip())
            j += 1

        # Build DataFrame
        def split_row(r: str) -> List[str]:
            parts = [p.strip() for p in r.strip('|').split('|')]
            return parts

        headers = split_row(header_line)
        data = [split_row(r) for r in rows]
        # Normalize width: pad/truncate rows to header length
        n = len(headers)
        norm = []
        for r in data:
            if len(r) < n:
                r = r + [""] * (n - len(r))
            elif len(r) > n:
                r = r[:n]
            norm.append(r)
        try:
            df = pd.DataFrame(norm, columns=headers)
            if not df.empty:
                tables.append(df)
        except Exception:
            pass

        # Advance
        i = j
    return tables


def parse_date(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    for fmt in ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d %b %Y", "%d %B %Y"]:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None


def normalize_amount(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse numeric amount and return (amount, currency). Handles EU formats and trailing symbols.
    Parentheses and leading '-' treated as negative.
    """
    t = (text or "").strip()
    if not t:
        return None, None

    # Detect currency anywhere (symbol or ISO code)
    currency = None
    if any(sym in t for sym in ["€", "$", "£", "₹", "¥"]):
        for sym in ["€", "$", "£", "₹", "¥"]:
            if sym in t:
                currency = currency or sym
                t = t.replace(sym, "")
    upper = t.upper()
    for code in ["USD", "EUR", "GBP", "INR", "JPY", "CHF", "AUD", "CAD"]:
        if code in upper:
            currency = currency or code
            upper = upper.replace(code, "")
            t = re.sub(code, "", t, flags=re.IGNORECASE)

    # Remove spaces and plus signs
    t = t.replace(" ", "").replace("+", "")

    # Handle negative via parentheses or leading '-'
    negative = False
    if t.startswith("(") and t.endswith(")"):
        negative = True
        t = t[1:-1]
    if t.startswith("-"):
        negative = True
        t = t[1:]

    # Normalize European formats: e.g., 1.566,52 -> 1566.52
    if "," in t and "." in t:
        t = t.replace(".", "")
        t = t.replace(",", ".")
    elif "," in t and "." not in t:
        t = t.replace(",", ".")

    # Strip any stray non-digit/non-dot characters
    t = re.sub(r"[^0-9\.]", "", t)

    try:
        if not t:
            return None, currency
        val = float(t)
        if negative:
            val = -val
        return val, currency
    except Exception:
        return None, currency


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
}


def strip_accents(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s


def match_header(col_name: str) -> Optional[str]:
    raw = (col_name or "").strip().lower()
    name = strip_accents(raw)
    for key, variants in HEADER_CANDIDATES.items():
        if name == key or key in name:
            return key
        for v in variants:
            vn = strip_accents(v.lower())
            if name.startswith(vn) or vn in name:
                return key
    return None


def choose_transaction_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Pick the table that most likely contains transactions by header keywords and column count."""
    best_df = None
    best_score = -1
    for df in tables:
        cols = list(df.columns)
        score = 0
        for c in cols:
            if match_header(str(c)):
                score += 2
        # Prefer moderate column widths typical of statements
        if 3 <= len(cols) <= 8:
            score += 1
        if score > best_score:
            best_score = score
            best_df = df
    return best_df


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a bank statement table to a structured dataset."""
    # Map columns
    col_map: Dict[str, Optional[str]] = {key: None for key in HEADER_CANDIDATES.keys()}
    for c in df.columns:
        m = match_header(str(c))
        if m and not col_map[m]:
            col_map[m] = c

    # Build unified columns
    out_rows = []
    for _, row in df.iterrows():
        d_text = str(row[col_map["date"]]) if col_map["date"] in df.columns else ""
        desc = str(row[col_map["description"]]) if col_map["description"] in df.columns else ""
        debit_text = str(row[col_map["debit"]]) if col_map["debit"] in df.columns else ""
        credit_text = str(row[col_map["credit"]]) if col_map["credit"] in df.columns else ""
        amount_text = str(row[col_map["amount"]]) if col_map["amount"] in df.columns else ""
        balance_text = str(row[col_map["balance"]]) if col_map["balance"] in df.columns else ""

        # Parse pieces
        d = parse_date(d_text)
        debit_val, cur1 = normalize_amount(debit_text)
        credit_val, cur2 = normalize_amount(credit_text)
        amount_val, cur3 = normalize_amount(amount_text)
        balance_val, cur4 = normalize_amount(balance_text)
        currency = cur1 or cur2 or cur3 or cur4

        # If no explicit debit/credit provided, infer from amount sign
        if amount_val is not None and debit_val is None and credit_val is None:
            if amount_val < 0:
                debit_val = abs(amount_val)
            else:
                credit_val = amount_val
        elif amount_val is None and (debit_val is not None or credit_val is not None):
            # Build amount from debit/credit
            if debit_val is not None:
                amount_val = -debit_val
            elif credit_val is not None:
                amount_val = credit_val

        out_rows.append({
            "date": d.isoformat() if d else d_text,
            "description": desc,
            "debit": debit_val,
            "credit": credit_val,
            "amount": amount_val,
            "balance": balance_val,
            "currency": currency,
        })

    out_df = pd.DataFrame(out_rows)
    # Sort by date when parsable
    try:
        out_df["date_parsed"] = pd.to_datetime(out_df["date"], errors="coerce")
        out_df = out_df.sort_values("date_parsed").drop(columns=["date_parsed"])  # keep original date text
    except Exception:
        pass
    return out_df


def normalize_transactions_from_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Normalize using an LLM-suggested mapping from original headers to canonical keys."""
    # Build col_map from mapping (only accept known canonical keys)
    canonical_keys = {"date", "description", "debit", "credit", "amount", "balance", "currency"}
    col_map = {k: None for k in canonical_keys}
    for original, canonical in mapping.items():
        canon = str(canonical).strip().lower()
        if canon in canonical_keys and original in df.columns and not col_map[canon]:
            col_map[canon] = original

    out_rows = []
    for _, row in df.iterrows():
        d_text = str(row[col_map["date"]]) if col_map["date"] in df.columns else ""
        desc = str(row[col_map["description"]]) if col_map["description"] in df.columns else ""
        debit_text = str(row[col_map["debit"]]) if col_map["debit"] in df.columns else ""
        credit_text = str(row[col_map["credit"]]) if col_map["credit"] in df.columns else ""
        amount_text = str(row[col_map["amount"]]) if col_map["amount"] in df.columns else ""
        balance_text = str(row[col_map["balance"]]) if col_map["balance"] in df.columns else ""

        d = parse_date(d_text)
        debit_val, cur1 = normalize_amount(debit_text)
        credit_val, cur2 = normalize_amount(credit_text)
        amount_val, cur3 = normalize_amount(amount_text)
        balance_val, cur4 = normalize_amount(balance_text)
        currency = cur1 or cur2 or cur3 or cur4

        if amount_val is not None and debit_val is None and credit_val is None:
            if amount_val < 0:
                debit_val = abs(amount_val)
            else:
                credit_val = amount_val
        elif amount_val is None and (debit_val is not None or credit_val is not None):
            if debit_val is not None:
                amount_val = -debit_val
            elif credit_val is not None:
                amount_val = credit_val

        out_rows.append({
            "date": d.isoformat() if d else d_text,
            "description": desc,
            "debit": debit_val,
            "credit": credit_val,
            "amount": amount_val,
            "balance": balance_val,
            "currency": currency,
        })

    out_df = pd.DataFrame(out_rows)
    try:
        out_df["date_parsed"] = pd.to_datetime(out_df["date"], errors="coerce")
        out_df = out_df.sort_values("date_parsed").drop(columns=["date_parsed"])
    except Exception:
        pass
    return out_df


def main():
    st.set_page_config(page_title="Docling Bank Statement Parser", page_icon="📄", layout="wide")
    st.title("Docling Bank Statement Parser")
    st.caption("Upload statements (PDF/Image), then click 'Process uploads' to start.")

    # --- Agent Events: live per-job viewer ---
    with st.sidebar:
        st.subheader("Agent Events")
        if st.session_state.get("jobs"):
            job_options = [f"{j['name']} ({j['id'][:8]})" for j in st.session_state.jobs]
            selected_label = st.selectbox("Select job", options=job_options, index=len(job_options) - 1)
            # map selection back to job
            selected_idx = job_options.index(selected_label)
            selected_job = st.session_state.jobs[selected_idx]
            st.json(selected_job.get("events") or [{"type": "message", "role": "assistant", "content": "No events yet for this job."}], expanded=False)
        else:
            st.info("Upload a document to see agent events.")

    # Session state for uploaded documents and their conversion status/results
    if "jobs" not in st.session_state:
        st.session_state.jobs = []  # list of dicts: {id, name, hash, status, outcome, rows, csv_bytes, json_bytes, markdown}

    # Upload multiple files
    uploaded_files = st.file_uploader(
        "Upload statement files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="You can add several files and they will be processed sequentially.",
    )

    # Controls
    ctrl_left, ctrl_right = st.columns([1, 1])
    with ctrl_left:
        show_debug_global = st.checkbox("Show raw Docling JSON (per-file)", value=False)
        use_pdf_fallback = st.checkbox("Use PDF fallback if no tables found", value=True,
                                       help="Enable a pdfplumber-based extractor when Docling returns no tables.")
        fallback_strategy = st.selectbox(
            "Fallback strategy",
            options=["auto", "lines", "text"],
            index=0,
            help="auto tries both; 'lines' works for ruled; 'text' for unruled."
        )
        force_fallback_only = st.checkbox("Force fallback only (skip Docling)", value=False,
                                          help="Skip Docling conversion and use fallback extractors directly for PDFs.")
        use_ai_normalization = st.checkbox(
            "Use AI normalization (LLM-assisted)", value=False,
            help="Requires OPENAI_API_KEY and 'openai' package. Falls back to rule-based if unavailable."
        )
    # Processing controls
    process_triggered = False
    with ctrl_right:
        if st.button("Clear history"):
            st.session_state.jobs = []
            st.experimental_rerun()
        process_triggered = st.button("Process uploads")

    # Process new uploads only when explicitly triggered
    if uploaded_files and not process_triggered:
        st.info("Files staged. Click 'Process uploads' to start processing.")

    if uploaded_files and process_triggered:
        files_list = list(uploaded_files)
        total = len(files_list)
        converter = DocumentConverter()
        for idx, uploaded in enumerate(files_list, start=1):
            tmp_path, data, src_hash = save_uploaded_file(uploaded)
            # Skip duplicate by hash if already processed/queued
            exists = any(j.get("hash") == src_hash for j in st.session_state.jobs)
            if exists:
                continue

            job = {
                "id": str(uuid4()),
                "name": uploaded.name,
                "hash": src_hash,
                "status": "queued",
                "outcome": None,
                "rows": None,
                "csv_bytes": None,
                "json_bytes": None,
                "raw_csv_bytes": None,
                "raw_json_bytes": None,
                "markdown": None,
                "events": [],
                "mapping": None,
            }
            st.session_state.jobs.append(job)

            try:
                # UI: show which file is being processed and what's next
                st.info(f"Processing {idx}/{total}: {uploaded.name}")
                next_names = [files_list[i].name for i in range(idx, min(idx + 2, total))]
                if next_names:
                    st.caption("Next in queue: " + ", ".join(next_names))

                job["status"] = "processing"
                ext = os.path.splitext(uploaded.name)[1].lower()

                # --- AG-UI: begin run & user intent ---
                run_id = str(uuid4())
                job["events"].append(build_event("run_start", {"run_id": run_id, "timestamp": datetime.utcnow().isoformat()}))
                job["events"].append(build_event("message", {"role": "user", "content": f"Normalize transactions from {uploaded.name}"}))

                # Force fallback-only path for PDFs
                if force_fallback_only and ext == ".pdf":
                    job["events"].append(build_event("tool_call", {"tool": "pdfplumber.extract", "args": {"strategy": fallback_strategy}}))
                    fallback_tables: List[pd.DataFrame] = []
                    if fallback_strategy == "auto":
                        tbl_lines = extract_tables_pdfplumber(tmp_path, strategy="lines")
                        tbl_text = extract_tables_pdfplumber(tmp_path, strategy="text")
                        fallback_tables = (tbl_lines or []) + (tbl_text or [])
                        if not fallback_tables:
                            fallback_tables = extract_tables_pdfplumber_by_headers(tmp_path)
                    else:
                        first = fallback_strategy
                        second = "text" if first == "lines" else "lines"
                        fallback_tables = extract_tables_pdfplumber(tmp_path, strategy=first)
                        if not fallback_tables:
                            fallback_tables = extract_tables_pdfplumber(tmp_path, strategy=second)
                        if not fallback_tables:
                            fallback_tables = extract_tables_pdfplumber_by_headers(tmp_path)

                    if not fallback_tables:
                        job["status"] = "done"
                        job["outcome"] = "no_tables"
                        job["rows"] = 0
                        job["events"].append(build_event("tool_result", {"tool": "pdfplumber.extract", "result": {"tables_found": 0}}))
                        job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "fail", "error": "no_tables"}))
                        continue
                    else:
                        tables = fallback_tables
                        json_doc = {"note": "skipped_docling"}
                        job["events"].append(build_event("tool_result", {"tool": "pdfplumber.extract", "result": {"tables_found": len(tables)}}))
                else:
                    job["events"].append(build_event("tool_call", {"tool": "docling.convert", "args": {"file": uploaded.name}}))
                    result = converter.convert(tmp_path)
                    json_doc = export_docling_json(result)
                    tables = extract_tables(json_doc)
                    job["events"].append(build_event("tool_result", {"tool": "docling.convert", "result": {"tables_found": len(tables) if tables else 0}}))
                if not tables:
                    # Try pdfplumber fallback for PDFs
                    fallback_tables: List[pd.DataFrame] = []
                    if use_pdf_fallback and ext == ".pdf":
                        if fallback_strategy == "auto":
                            tbl_lines = extract_tables_pdfplumber(tmp_path, strategy="lines")
                            tbl_text = extract_tables_pdfplumber(tmp_path, strategy="text")
                            fallback_tables = (tbl_lines or []) + (tbl_text or [])
                            if not fallback_tables:
                                fallback_tables = extract_tables_pdfplumber_by_headers(tmp_path)
                        else:
                            # Attempt selected strategy first, then the alternate if none found
                            first = fallback_strategy
                            second = "text" if first == "lines" else "lines"
                            fallback_tables = extract_tables_pdfplumber(tmp_path, strategy=first)
                            if not fallback_tables:
                                fallback_tables = extract_tables_pdfplumber(tmp_path, strategy=second)
                            if not fallback_tables:
                                # Finally, try header-guided extraction for unruled Spanish tables
                                fallback_tables = extract_tables_pdfplumber_by_headers(tmp_path)

                    if not fallback_tables:
                        # Try parsing Docling-generated markdown tables as a last-resort fallback
                        md_text = result.document.export_to_markdown()
                        md_tables = extract_tables_from_markdown(md_text)
                        if md_tables:
                            tables = md_tables
                            job["outcome"] = "success_fallback"
                            job["events"].append(build_event("progress", {"content": "Used markdown fallback.", "progress": 0.4}))
                        else:
                            # No tables anywhere; provide markdown only
                            job["status"] = "done"
                            job["outcome"] = "no_tables"
                            job["rows"] = 0
                            job["markdown"] = md_text
                            job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "fail", "error": "no_tables"}))
                            continue
                    else:
                        tables = fallback_tables
                        job["events"].append(build_event("tool_result", {"tool": "pdfplumber.extract", "result": {"tables_found": len(tables)}}))

                selected_df = choose_transaction_table(tables)
                txn_df = selected_df if selected_df is not None else tables[0]
                job["events"].append(build_event("tool_call", {"tool": "choose_table", "args": {"candidates": len(tables)}}))
                job["events"].append(build_event("tool_result", {"tool": "choose_table", "result": {"rows": int(len(txn_df)), "cols": int(len(txn_df.columns))}}))
                # Provide raw outputs before normalization
                try:
                    job["raw_csv_bytes"] = txn_df.to_csv(index=False).encode("utf-8")
                    job["raw_json_bytes"] = txn_df.to_json(orient="records").encode("utf-8")
                except Exception:
                    pass
                # Choose normalization path (AI-assisted or rule-based)
                normalized = None
                if use_ai_normalization:
                    try:
                        job["events"].append(build_event("tool_call", {"tool": "ai.suggest_mapping", "args": {"columns": list(map(str, txn_df.columns))}}))
                        mapping = suggest_mapping(list(txn_df.columns), txn_df.head(15).values.tolist())
                    except Exception:
                        mapping = None
                    if mapping:
                        normalized = normalize_transactions_from_mapping(txn_df, mapping)
                        job["mapping"] = mapping
                        job["events"].append(build_event("tool_result", {"tool": "ai.suggest_mapping", "result": {"mapping": mapping}}))
                        job["events"].append(build_event("tool_call", {"tool": "normalize.apply_mapping", "args": {"rows": int(len(txn_df))}}))
                        job["events"].append(build_event("tool_result", {"tool": "normalize.apply_mapping", "result": {"normalized_rows": int(len(normalized))}}))
                        job["outcome"] = job.get("outcome") or "success_ai"
                    else:
                        st.info("AI normalization unavailable: set OPENAI_API_KEY and install 'openai'. Using rule-based normalization.")
                        normalized = normalize_transactions(txn_df)
                        job["events"].append(build_event("message", {"role": "assistant", "content": "AI unavailable; used rule-based normalization."}))
                else:
                    normalized = normalize_transactions(txn_df)
                    job["events"].append(build_event("tool_call", {"tool": "normalize.rule_based", "args": {"rows": int(len(txn_df))}}))
                    job["events"].append(build_event("tool_result", {"tool": "normalize.rule_based", "result": {"normalized_rows": int(len(normalized))}}))
                job["status"] = "done"
                job["outcome"] = "success" if json_doc.get("error") != "json_export_failed" and extract_tables(json_doc) else (job.get("outcome") or "success_fallback")
                job["rows"] = int(len(normalized)) if len(normalized) > 0 else int(len(txn_df))
                job["csv_bytes"] = normalized.to_csv(index=False).encode("utf-8")
                job["json_bytes"] = normalized.to_json(orient="records").encode("utf-8")

                job["events"].append(build_event("message", {"role": "assistant", "content": f"Processed {uploaded.name}: {job['rows']} rows."}))
                job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "success"}))

                # Keep markdown of full document for fallback/reference, if available
                try:
                    if 'result' in locals() and result is not None:
                        job["markdown"] = result.document.export_to_markdown()
                except Exception:
                    pass

                # Optional per-file debug panel
                if show_debug_global:
                    with st.expander(f"Debug JSON: {uploaded.name}"):
                        st.code(json.dumps(json_doc)[:8000])
            except Exception as e:
                # If Docling conversion fails outright, attempt PDF fallback directly
                ext = os.path.splitext(uploaded.name)[1].lower()
                fallback_tables: List[pd.DataFrame] = []
                if use_pdf_fallback and ext == ".pdf":
                    first = fallback_strategy
                    second = "text" if first == "lines" else "lines"
                    fallback_tables = extract_tables_pdfplumber(tmp_path, strategy=first)
                    if not fallback_tables:
                        fallback_tables = extract_tables_pdfplumber(tmp_path, strategy=second)
                    if not fallback_tables:
                        fallback_tables = extract_tables_pdfplumber_by_headers(tmp_path)

                if fallback_tables:
                    try:
                        selected_df = choose_transaction_table(fallback_tables)
                        txn_df = selected_df if selected_df is not None else fallback_tables[0]
                        normalized = normalize_transactions(txn_df)
                        job["status"] = "done"
                        job["outcome"] = "success_fallback"
                        job["rows"] = int(len(normalized))
                        job["csv_bytes"] = normalized.to_csv(index=False).encode("utf-8")
                        job["json_bytes"] = normalized.to_json(orient="records").encode("utf-8")
                        st.warning(f"Docling failed for {uploaded.name}: {e}. Used pdfplumber fallback.")
                        job["events"].append(build_event("message", {"role": "assistant", "content": "Docling failed; used pdfplumber fallback."}))
                        job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "success"}))
                    except Exception as e2:
                        job["status"] = "done"
                        job["outcome"] = "fail"
                        job["rows"] = 0
                        st.warning(f"Fallback normalization failed for {uploaded.name}: {e2}")
                        job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "fail", "error": str(e2)}))
                else:
                    # Try parsing Docling markdown if available (conversion may have produced document but JSON failed)
                    try:
                        md_text = None
                        # Best-effort: if 'result' exists but json export failed, its markdown may be accessible
                        if 'result' in locals():
                            md_text = result.document.export_to_markdown()
                        if md_text:
                            md_tables = extract_tables_from_markdown(md_text)
                            if md_tables:
                                selected_df = choose_transaction_table(md_tables)
                                txn_df = selected_df if selected_df is not None else md_tables[0]
                                normalized = normalize_transactions(txn_df)
                                job["status"] = "done"
                                job["outcome"] = "success_fallback"
                                job["rows"] = int(len(normalized))
                                job["csv_bytes"] = normalized.to_csv(index=False).encode("utf-8")
                                job["json_bytes"] = normalized.to_json(orient="records").encode("utf-8")
                                st.warning(f"Docling failed for {uploaded.name}: {e}. Used markdown fallback.")
                                job["events"].append(build_event("message", {"role": "assistant", "content": "Docling failed; used markdown fallback."}))
                                job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "success"}))
                                # Do not return early; continue to render UI below
                    except Exception:
                        pass
                    job["status"] = "done"
                    job["outcome"] = "fail"
                    job["rows"] = 0
                    st.warning(f"Failed to process {uploaded.name}: {e}")
                    job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "fail", "error": str(e)}))

    # Upload history table
    st.subheader("Uploaded Documents")
    if st.session_state.jobs:
        table_df = pd.DataFrame([
            {
                "file": j.get("name"),
                "status": j.get("status"),
                "outcome": j.get("outcome"),
                "rows": j.get("rows"),
            }
            for j in st.session_state.jobs
        ])
        st.dataframe(table_df, width='stretch')

        st.markdown("**Downloads**")
        for j in st.session_state.jobs:
            dl_cols = st.columns([3, 2, 2, 2, 2, 2])
            with dl_cols[0]:
                st.write(j.get("name"))
            with dl_cols[1]:
                if j.get("csv_bytes"):
                    st.download_button(
                        label="Download CSV",
                        data=j["csv_bytes"],
                        file_name=f"{os.path.splitext(j['name'])[0]}_transactions.csv",
                        mime="text/csv",
                        key=f"csv_{j['id']}",
                    )
            with dl_cols[2]:
                if j.get("json_bytes"):
                    st.download_button(
                        label="Download JSON",
                        data=j["json_bytes"],
                        file_name=f"{os.path.splitext(j['name'])[0]}_transactions.json",
                        mime="application/json",
                        key=f"json_{j['id']}",
                    )
            with dl_cols[3]:
                if j.get("markdown"):
                    st.download_button(
                        label="Download Markdown",
                        data=(j["markdown"] or "").encode("utf-8"),
                        file_name=f"{os.path.splitext(j['name'])[0]}.md",
                        mime="text/markdown",
                        key=f"md_{j['id']}",
                    )
            with dl_cols[4]:
                if j.get("raw_csv_bytes"):
                    st.download_button(
                        label="Download Raw CSV",
                        data=j["raw_csv_bytes"],
                        file_name=f"{os.path.splitext(j['name'])[0]}_raw.csv",
                        mime="text/csv",
                        key=f"raw_csv_{j['id']}",
                    )
            with dl_cols[5]:
                if j.get("raw_json_bytes"):
                    st.download_button(
                        label="Download Raw JSON",
                        data=j["raw_json_bytes"],
                        file_name=f"{os.path.splitext(j['name'])[0]}_raw.json",
                        mime="application/json",
                        key=f"raw_json_{j['id']}",
                    )

            # --- Agent Events per job ---
            with st.expander(f"Agent Events: {j.get('name')}"):
                st.json(j.get("events") or [], expanded=False)

            # --- Mapping Editor per job ---
            with st.expander(f"Edit mapping and re-normalize: {j.get('name')}"):
                raw_df = None
                try:
                    if j.get("raw_json_bytes"):
                        raw_df = pd.read_json(io.BytesIO(j["raw_json_bytes"]))
                    elif j.get("raw_csv_bytes"):
                        raw_df = pd.read_csv(io.BytesIO(j["raw_csv_bytes"]))
                except Exception:
                    raw_df = None

                if raw_df is None:
                    st.info("No raw table available for editing.")
                else:
                    st.caption("Map original columns to canonical keys, then re-normalize.")
                    columns = list(map(str, raw_df.columns))
                    canonical_keys = ["date", "description", "debit", "credit", "amount", "balance", "currency"]
                    # Build current selection from saved mapping
                    saved_map = j.get("mapping") or {}
                    reverse = {str(v).lower(): k for k, v in saved_map.items()}
                    with st.form(key=f"map_form_{j['id']}"):
                        selections = {}
                        for key in canonical_keys:
                            default_col = reverse.get(key)
                            sel = st.selectbox(
                                f"{key}",
                                options=["(none)"] + columns,
                                index=(0 if not default_col or default_col not in columns else (columns.index(default_col) + 1)),
                                key=f"sel_{j['id']}_{key}"
                            )
                            selections[key] = None if sel == "(none)" else sel
                        submitted = st.form_submit_button("Apply mapping and re-normalize")
                        if submitted:
                            # Build mapping original->canonical
                            new_map = {}
                            for key, col in selections.items():
                                if col:
                                    new_map[col] = key
                            try:
                                new_norm = normalize_transactions_from_mapping(raw_df, new_map)
                                j["csv_bytes"] = new_norm.to_csv(index=False).encode("utf-8")
                                j["json_bytes"] = new_norm.to_json(orient="records").encode("utf-8")
                                j["rows"] = int(len(new_norm))
                                j["mapping"] = new_map
                                # Append events for manual correction
                                j["events"].append(build_event("tool_call", {"tool": "normalize.apply_manual_mapping", "args": {"rows": int(len(raw_df))}}))
                                j["events"].append(build_event("tool_result", {"tool": "normalize.apply_manual_mapping", "result": {"normalized_rows": int(len(new_norm))}}))
                                st.success("Re-normalized with manual mapping.")
                            except Exception as e3:
                                st.error(f"Failed to apply mapping: {e3}")

    else:
        st.info("No documents uploaded yet.")


if __name__ == "__main__":
    main()