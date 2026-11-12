@echo off
setlocal
set "APP_DIR=E:\compositeguard-ai"
set "NODE_EXE=node"

cd /d "%APP_DIR%"

echo Starting CompositeGuard AI dev server in background...
"%NODE_EXE%" --stack-size=4096 "node_modules\.bin\vite" --host 127.0.0.1 --port 5173 > "logs\vite-stdout.log" 2> "logs\vite-stderr.log" &
echo Dev server PID: %ERRORLEVEL%
echo Open http://127.0.0.1:5173
echo Vite logs -> logs\vite-stdout.log and logs\vite-stderr.log
echo.
echo Leave this window open while developing.
pause
