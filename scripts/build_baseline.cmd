@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "APP=%~1"
set "TARGET=%~2"

if "%APP%"=="" goto :usage
if "%TARGET%"=="" goto :usage

if not exist "%ROOT%\firmware\baseline\%APP%\CMakeLists.txt" (
  echo Missing firmware app: %ROOT%\firmware\baseline\%APP%
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

cd /d "%ROOT%\firmware\baseline\%APP%"
idf.py set-target %TARGET%
if errorlevel 1 exit /b %errorlevel%
idf.py build
exit /b %errorlevel%

:usage
echo Usage: scripts\build_baseline.cmd csi_recv esp32s3
echo Usage: scripts\build_baseline.cmd csi_send esp32
exit /b 2
