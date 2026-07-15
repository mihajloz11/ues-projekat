@echo off
setlocal

rem Pokreće objedinjeni Room Monitor dashboard i otvara ga u pregledniku.
rem Server koristi ruview_bridge\static\index.html, /api/state i stanje iz iot\latest_state.json.

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PORT=8765"

cd /d "%ROOT%"

start "WiFi CSI Dashboard" "%ROOT%\.venv\Scripts\python.exe" ruview_bridge\demo_server.py --port %PORT%
rem kratka pauza ostavlja serveru vrijeme da se pokrene prije otvaranja preglednika
rem ping se koristi jer timeout ne radi kada je standardni ulaz preusmjeren
ping -n 3 127.0.0.1 >nul
start "" "http://127.0.0.1:%PORT%/"

echo.
echo   Room Monitor dashboard: http://127.0.0.1:%PORT%/
echo   Server radi u posebnom prozoru; zatvaranje prozora ga zaustavlja.
echo.
