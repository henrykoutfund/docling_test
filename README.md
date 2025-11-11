# Docling PDF to Transactions (Streamlit)

Streamlit app that converts PDFs to structured transaction tables. It uses Docling first and, when needed, falls back to `pdfplumber` (lines/text) and a last‑resort Markdown pipe‑table parser. It also provides raw table downloads (CSV/JSON) before normalization.

## Features
- Docling conversion with robust error handling.
- Fallback strategies: `lines`, `text`, and `auto` (cycles both).
- Header‑guided extractor when strategies return no tables.
- Raw CSV/JSON downloads (pre‑normalization) and normalized CSV/JSON outputs.
- UI controls: "Use PDF fallback", "Force fallback only (skip Docling)", and per‑job "Download Markdown".
- Queue display showing the currently processed file and next items.
- AG‑UI preview: sidebar demo of event‑based agent run.
 - Optional AI normalization: LLM‑assisted header mapping for better canonicalization.

## Requirements
- Python 3.11+
- See `requirements.txt` for dependencies.

## Setup (local)
1. Create and activate a virtual environment:
   - PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the app:
  ```powershell
  streamlit run app.py
  ```

### AI Normalization (LLM‑assisted)
- Enable with the checkbox "Use AI normalization (LLM‑assisted)" in the UI.
- Requires environment `OPENAI_API_KEY` and the `openai` Python package installed.
- Optional: set `OPENAI_MODEL` (default: `gpt-4o-mini`).
- If not configured, the app falls back automatically to rule‑based normalization.

Example (PowerShell):
```powershell
$env:OPENAI_API_KEY = "sk-your-key"
# optional
$env:OPENAI_MODEL = "gpt-4o-mini"
pip install openai
python -m streamlit run docling_test/app.py
```

This step asks an LLM to map source headers (e.g., "Fecha", "Concepto", "Saldo") to canonical keys: `date`, `description`, `debit`, `credit`, `amount`, `balance`, `currency`. The app then applies deterministic parsing on values (dates, amounts, negatives, currency detection) to produce a clean dataset.

### AG‑UI Integration (Preview)
- This app includes a minimal AG‑UI‑style event demo in the sidebar. Press "Run AG‑UI Demo" to see a sequence of standardized events for an agent execution rendered as JSON.
- AG‑UI is an open, event‑based protocol that standardizes agent ↔ UI interactions and works with SSE/WebSockets transports [0]. See the repo for details and quickstarts.

#### Extending to full AG‑UI
- Backend: expose an HTTP endpoint that emits AG‑UI events (e.g., SSE) and accepts AG‑UI inputs. You can wire actual agent runs to emit events like `run_start`, `message`, `tool_call`, `progress`, `tool_result`, and `run_end`.
- Frontend: embed an AG‑UI client in a web UI. In Streamlit, you can use `st.components.v1.html` to load a small JS client (from your own bundle) that subscribes to the SSE endpoint and renders events.
- Framework integrations are available (LangGraph, CrewAI, PydanticAI, LlamaIndex, Mastra, Agno, etc.) if you prefer to connect an existing agent stack [0].

> Reference: GitHub – AG‑UI Protocol: https://github.com/ag-ui-protocol/ag-ui [0]

## Push to GitHub
1. Create a new repository on GitHub (public or private). Copy the repo URL (HTTPS is simplest), e.g. `https://github.com/<your-username>/<repo>.git`.
2. In your project folder (`c:\Users\hk\Desktop\doclingtest`), initialize git and make the first commit:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: Streamlit app with fallbacks and raw exports"
   ```
3. Set `main` as the default branch and add remote:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo>.git
   ```
4. Push:
   ```powershell
   git push -u origin main
   ```

## Optional: Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Select your repo and set the entrypoint to `app.py`.
3. Ensure `requirements.txt` contains all needed packages.
4. Deploy; changes pushed to `main` will auto‑redeploy.

## Updating dependencies
- To capture your current environment:
  ```powershell
  pip freeze > requirements.txt
  ```
- Or specify only the key packages manually in `requirements.txt` and keep versions flexible.

## Notes
- `.gitignore` excludes your virtual environment, caches, and generated outputs. README.md is now tracked.
- If normalized CSV is empty but raw has rows, use Raw CSV/JSON downloads.
- For scanned/image‑only PDFs, consider adding OCR fallback (we can wire this in later).

[0] AG‑UI: the Agent‑User Interaction Protocol. Standardized event types, flexible transports, and reference implementations.