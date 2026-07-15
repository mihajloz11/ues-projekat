@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "APP=%~1"
set "PORT=%~2"

if "%APP%"=="" goto :usage
if "%PORT%"=="" goto :usage

if exist "%ROOT%\tools\esp-idf\export.bat" (
  set "IDF_TOOLS_PATH=%ROOT%\tools\.espressif"
  call "%ROOT%\tools\esp-idf\export.bat"
  if errorlevel 1 exit /b %errorlevel%
) else (
  where idf.py >nul 2>nul
  if errorlevel 1 (
    echo ESP-IDF environment is not active.
    exit /b 1
  )
)

cd /d "%ROOT%\firmware\experiments\%APP%"
idf.py -p %PORT% monitor
exit /b %errorlevel%

:usage
echo Usage: scripts\monitor_experiment.cmd dht11_test COM3
exit /b 2
