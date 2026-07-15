@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "APP=%~1"
set "TARGET=%~2"
set "PORT=%~3"

if "%APP%"=="" goto :usage
if "%TARGET%"=="" goto :usage
if "%PORT%"=="" goto :usage

if not exist "%ROOT%\firmware\experiments\%APP%\CMakeLists.txt" (
  echo Missing firmware experiment: %ROOT%\firmware\experiments\%APP%
  exit /b 1
)

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
idf.py set-target %TARGET%
if errorlevel 1 exit /b %errorlevel%
idf.py -p %PORT% flash
exit /b %errorlevel%

:usage
echo Usage: scripts\flash_experiment.cmd dht11_test esp32s3 COM3
exit /b 2
