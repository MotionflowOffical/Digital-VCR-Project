@echo off
setlocal
cd /d "%~dp0"
set "DIST_DIR=%CD%\dist\DigitalVCR"
set "EXE_PATH=%DIST_DIR%\DigitalVCR.exe"

if not exist .venv py -m venv .venv

taskkill /IM DigitalVCR.exe /F >nul 2>nul
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
if exist "%DIST_DIR%" ren "%DIST_DIR%" "DigitalVCR.old.%RANDOM%"

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-build.txt
python -m PyInstaller --clean --noconfirm digital_vcr.spec
if errorlevel 1 exit /b %errorlevel%

if exist "%EXE_PATH%" goto build_ok

echo Build finished without creating "%EXE_PATH%"
exit /b 1

:build_ok
echo.
echo Build complete:
echo   "%EXE_PATH%"
endlocal
