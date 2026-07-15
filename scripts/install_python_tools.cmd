@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VENV=%ROOT%\.venv"

if not exist "%VENV%\Scripts\python.exe" (
  python -m venv "%VENV%"
  if errorlevel 1 exit /b %errorlevel%
)

"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%
"%VENV%\Scripts\python.exe" -m pip install -r "%ROOT%\scripts\requirements-data.txt"
exit /b %errorlevel%
