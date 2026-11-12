# CompositeGuard AI: AI-Powered Non-Destructive Inspection

## Quick Start (Windows PowerShell)

```powershell
:: 1. Navigate to project
cd E:\compositeguard-ai

:: 2. Activate virtual environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Start dashboard
streamlit run app.py
```

Open browser at http://localhost:8501

## ⚠️ Important Prototype Notice

This is a hackathon prototype for demonstration only. Not production-ready, not clinically validated, and must not be used for structural safety decisions. Hidden internal composite defects cannot be detected from surface photographs alone.
```