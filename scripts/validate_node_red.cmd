@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "USERDIR=%ROOT%\iot\node_red_userdir"

cd /d "%ROOT%"
npx --yes node-red -u "%USERDIR%" --safe --settings "%USERDIR%\settings.js" --help >nul
exit /b %errorlevel%
