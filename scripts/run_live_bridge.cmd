@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=COM3"

cd /d "%ROOT%"
"%ROOT%\.venv\Scripts\python.exe" scripts\live_zone_inference.py ^
  --port %PORT% ^
  --baud 921600 ^
  --model models\zone_csi_mlp_fast.json ^
  --presence-model models\presence_csi_mlp_fast.json ^
  --motion-model models\motion_csi_mlp_fast.json ^
  --motion-stable-window 5 ^
  --zone-confidence-threshold 0.92 ^
  --zone-min-margin 0.25 ^
  --zone-stable-window 9 ^
  --zone-stable-votes 6 ^
  --zone-output-mode off ^
  --smooth 1 ^
  --write-every 0.25 ^
  --signal-baseline-alpha 0.02 ^
  --signal-activity-smoothing 0.35 ^
  --signal-activity-threshold 0.08 ^
  --signal-min-baseline-frames 16 ^
  --dht-stale-sec 600 ^
  --mmwave-stale-sec 90 ^
  --out iot\latest_state.json ^
  --history iot\state_history.jsonl ^
  --history-every 5 ^
  --sqlite iot\room_state.sqlite ^
  --sqlite-every 5 ^
  --mqtt-host 127.0.0.1 ^
  --mqtt-port 1883 ^
  --mqtt-topic wifi-csi/room/state ^
  --mqtt-every 2
