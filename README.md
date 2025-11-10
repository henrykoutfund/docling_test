# Docling PDF to Transactions (Streamlit)

Streamlit app that converts PDFs to structured transaction tables. It uses Docling first and, when needed, falls back to `pdfplumber` (lines/text) and a last‑resort Markdown pipe‑table parser. It also provides raw table downloads (CSV/JSON) before normalization.

## Features
- Docling conversion with robust error handling.
- Fallback strategies: `lines`, `text`, and `auto` (cycles both).
- Header‑guided extractor when strategies return no tables.
- Raw CSV/JSON downloads (pre‑normalization) and normalized CSV/JSON outputs.
- UI controls: "Use PDF fallback", "Force fallback only (skip Docling)", and per‑job "Download Markdown".
- Queue display showing the currently processed file and next items.

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