# Running CompositeGuard AI locally

Both processes must be running for the app to work: the FastAPI analysis backend on 8000 and the Vite dev server on 5173. Vite proxies `/api/*` to the backend.

## 1. Start the backend

Windows (PowerShell):
```
cd /e/compositeguard-ai
$env:PORT="8000"; $env:PYTHONUNBUFFERED="1"
Start-Process python -ArgumentList "-u","backend\serve.py" -RedirectStandardOutput logs\backend.log -RedirectStandardError logs\backend.log.err -NoNewWindow
```
Wait for the line `Uvicorn running on http://127.0.0.1:8000` in `logs/backend.log`, or check `curl http://127.0.0.1:8000/api/health` returns `{"status":"ok"}`.

Linux/macOS:
```
cd /e/compositeguard-ai   # or the project root on that machine
PORT=8000 PYTHONUNBUFFERED=1 python -u backend/serve.py > logs/backend.log 2> logs/backend.log.err &
```

## 2. Start the frontend

Windows (PowerShell):
```
cd /e/compositeguard-ai
Start-Process npm -ArgumentList "run","dev" -RedirectStandardOutput logs\vite.log -RedirectStandardError logs\vite.log.err -NoNewWindow
```
The dev server prints `Local: http://localhost:5173` shortly after start. Open that URL in a browser to use the app.

Linux/macOS:
```
cd /e/compositeguard-ai
npm run dev > logs/vite.log 2> logs/vite.log.err &
```

## 3. Verify

- `curl -s http://127.0.0.1:8000/api/health` → `{"status":"ok"}`
- `curl -s http://127.0.0.1:5173` returns the HTML shell (the dev server)
- In the browser, uploading a specimen image and clicking **Analyze Image** should produce a verdict (predicted class, confidence, RPN, risk level) and list the inspection in **Recent Inspections**.

## Notes

- The backend writes persisted images to `data/` and the inspection log to `logs/inspections.jsonl`.
- Node 20+ is required for the frontend. Python deps are listed in `requirements.txt` (the backend only needs `fastapi`, `numpy`, `scipy`, `Pillow` — the other entries are for optional model-training flows).
- The frontend uses Tailwind CSS v4 via the `@tailwindcss/vite` plugin; `npm install` (or `bun install`) must have run once before `npm run dev`. If `node_modules` is missing, run `npm install` first.
