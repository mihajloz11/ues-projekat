@echo off
setlocal

rem MQTT na ThingSpeak cloud uploader za vježbu V5.
rem Write API ključ se čita iz iot\thingspeak_key.local.txt ili THINGSPEAK_API_KEY.
rem Za rad su potrebni MQTT broker i aktivan publisher.

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
cd /d "%ROOT%"
"%ROOT%\.venv\Scripts\python.exe" scripts\thingspeak_uploader.py %*
