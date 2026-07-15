@echo off
setlocal

rem Adresa 0.0.0.0 čini broker dostupnim ESP32-S3 pločici i drugim LAN klijentima.
rem Node-RED i Python bridge i dalje koriste 127.0.0.1.
echo Lokalni MQTT broker na 0.0.0.0:1883 dostupan je u LAN mrezi.
echo.
echo LAN IP adresa racunara za ESP32-S3 MQTT klijent:
echo   powershell -Command "(Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object {$_.PrefixOrigin -ne 'WellKnown'}).IPAddress"
echo.
npx --yes aedes-cli start --host 0.0.0.0 --port 1883
