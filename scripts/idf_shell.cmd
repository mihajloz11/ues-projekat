@echo off
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
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

echo.
echo ESP-IDF shell is ready for:
echo   idf.py --version
echo   idf.py build
echo   idf.py -p COMx flash monitor
echo.
cmd /k
