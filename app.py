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
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from huggingface_hub import snapshot_download
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

@st.cache_resource
def get_converter(do_ocr: bool = True) -> DocumentConverter:
    """Create a Docling converter configured with RapidOCR ONNX models.

    Prefer packaged ONNX models from rapidocr_onnxruntime to avoid auth/network,
    and fall back to Hugging Face (public) if not found.
    """
    # Try using ONNX models bundled in rapidocr_onnxruntime
    try:
        import rapidocr_onnxruntime as rort
        pkg_dir = os.path.dirname(rort.__file__)
        models_dir = os.path.join(pkg_dir, "models")
        det_model_path = os.path.join(models_dir, "ch_PP-OCRv3_det_infer.onnx")
        rec_model_path = os.path.join(models_dir, "ch_PP-OCRv3_rec_infer.onnx")
        cls_model_path = os.path.join(models_dir, "ch_ppocr_mobile_v2.0_cls_infer.onnx")
        if all(os.path.exists(p) for p in [det_model_path, rec_model_path, cls_model_path]):
            ocr_options = RapidOcrOptions(
                det_model_path=det_model_path,
                rec_model_path=rec_model_path,
                cls_model_path=cls_model_path,
            )
            pipeline_options = PdfPipelineOptions(ocr_options=ocr_options)
            try:
                # Toggle OCR according to user guidance
                setattr(pipeline_options, "do_ocr", bool(do_ocr))
            except Exception:
                pass
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
    except Exception:
        pass

    # Fallback: download public ONNX models from Hugging Face (no auth required)
    try:
        download_path = snapshot_download(repo_id="SWHL/RapidOCR")
        det_model_path = os.path.join(download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx")
        rec_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx")
        cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
        ocr_options = RapidOcrOptions(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )
        pipeline_options = PdfPipelineOptions(ocr_options=ocr_options)
        try:
            setattr(pipeline_options, "do_ocr", bool(do_ocr))
        except Exception:
            pass
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    except Exception as e:
        st.warning(f"RapidOCR models setup failed: {e}. Using default converter; OCR may be limited.")
        return DocumentConverter()


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


def strip_accents(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    except Exception:
        return s


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
    # common formats incl. two-digit year (e.g., 01 Aug 25)
    formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None


def normalize_amount(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse numeric amount and return (amount, currency).
    Handles UK and EU formats and common currency annotations.
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

    # Decide thousands/decimal separators:
    # - UK style: 163,403.10 -> remove commas only
    # - EU style: 1.566,52 -> remove dots as thousands, convert comma to dot
    uk_match = re.fullmatch(r"[-]?\d{1,3}(,\d{3})+(\.\d+)?", t)
    eu_match = re.fullmatch(r"[-]?\d{1,3}(\.\d{3})+(,\d+)?", t)
    if uk_match:
        t = t.replace(",", "")
    elif eu_match:
        t = t.replace(".", "")
        t = t.replace(",", ".")
    else:
        # If only comma present and looks like decimal
        if "," in t and "." not in t:
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")  # safe to drop stray thousands commas

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
    # Avoid overly broad tokens like 'in'/'out'; prefer explicit phrases
    "debit": ["debit", "withdrawal", "dr", "money out", "payments", "outgoing", "cargo", "retiro"],
    "credit": ["credit", "deposit", "cr", "money in", "receipts", "incoming", "abono", "ingreso"],
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
    for key, base_variants in HEADER_CANDIDATES.items():
        # extend with user-provided hints
        user_variants = []
        try:
            if key == "credit":
                user_variants = [strip_accents(v.lower()) for v in (st.session_state.get("hints_credit") or [])]
            elif key == "debit":
                user_variants = [strip_accents(v.lower()) for v in (st.session_state.get("hints_debit") or [])]
        except Exception:
            user_variants = []
        variants = list(base_variants) + user_variants
        # direct match or clear containment of the canonical key
        if name == key or (key in name and len(key) > 2):
            return key
        for v in variants:
            vn = strip_accents(v.lower())
            # use word-boundary matching to avoid accidental matches (e.g., 'in' in 'description')
            if re.search(rf"\b{re.escape(vn)}\b", name):
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
        # Bonus if both money-in and money-out style headers detected, or balance present
        names = " ".join(map(lambda x: str(x).lower(), cols))
        if re.search(r"\bmoney\s+in\b", names) and re.search(r"\bmoney\s+out\b", names):
            score += 2
        if any(match_header(str(c)) == "balance" for c in cols):
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

    # Controls (Docling-only mode)
    ctrl_left, ctrl_right = st.columns([1, 1])
    with ctrl_left:
        show_debug_global = st.checkbox("Show raw Docling JSON (per-file)", value=False)
        ocr_enabled = st.checkbox(
            "Enable OCR (RapidOCR)", value=True,
            help="Turn on when PDFs are image-only or text is not searchable."
        )
        credit_hints_text = st.text_input(
            "Header hints for credit (comma-separated)",
            value="Money In, Receipts",
            help="Add phrases to help map credit columns (e.g., Money In)."
        )
        debit_hints_text = st.text_input(
            "Header hints for debit (comma-separated)",
            value="Money Out, Payments",
            help="Add phrases to help map debit columns (e.g., Money Out)."
        )
        try:
            st.session_state.hints_credit = [h.strip() for h in credit_hints_text.split(",") if h.strip()]
            st.session_state.hints_debit = [h.strip() for h in debit_hints_text.split(",") if h.strip()]
        except Exception:
            st.session_state.hints_credit = []
            st.session_state.hints_debit = []
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

                # Docling-only conversion
                job["events"].append(build_event("tool_call", {"tool": "docling.convert", "args": {"file": uploaded.name, "ocr": ocr_enabled}}))
                converter = get_converter(do_ocr=ocr_enabled)
                result = converter.convert(tmp_path)
                json_doc = export_docling_json(result)
                tables = extract_tables(json_doc)
                job["events"].append(build_event("tool_result", {"tool": "docling.convert", "result": {"tables_found": len(tables) if tables else 0}}))

                # Interactive guidance if no tables found
                if not tables:
                    # surface minimal markdown to let user inspect
                    try:
                        md_text = result.document.export_to_markdown()
                        job["markdown"] = md_text
                    except Exception:
                        job["markdown"] = None
                    job["status"] = "done"
                    job["outcome"] = "no_tables"
                    job["rows"] = 0
                    job["events"].append(build_event("message", {"role": "assistant", "content": "No tables detected by Docling. Try toggling OCR or adding header hints (Money In/Money Out)."}))
                    job["events"].append(build_event("run_end", {"timestamp": datetime.utcnow().isoformat(), "status": "fail", "error": "no_tables"}))
                    continue

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
                job["outcome"] = "success"
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
                # Docling-only error path: report failure and guidance
                job["status"] = "done"
                job["outcome"] = "fail"
                job["rows"] = 0
                st.warning(f"Failed to process {uploaded.name} via Docling: {e}")
                job["events"].append(build_event("message", {"role": "assistant", "content": "Docling error. Try enabling OCR or adjusting header hints."}))
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