@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    py -m venv .venv
)

taskkill /IM DigitalVCR.exe /F >nul 2>nul
if exist "%CD%\dist\DigitalVCR" (
    rmdir /s /q "%CD%\dist\DigitalVCR"
    if exist "%CD%\dist\DigitalVCR" (
        ren "%CD%\dist\DigitalVCR" "DigitalVCR.old.%RANDOM%"
    )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-build.txt
python -m PyInstaller --clean --noconfirm digital_vcr.spec
if errorlevel 1 exit /b %errorlevel%

if not exist "%CD%\dist\DigitalVCR\DigitalVCR.exe" (
    echo Build finished without creating %CD%\dist\DigitalVCR\DigitalVCR.exe
    exit /b 1
)

echo.
echo Build complete:
echo   %CD%\dist\DigitalVCR\DigitalVCR.exe
endlocal
