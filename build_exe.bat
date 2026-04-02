@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
    py -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-build.txt
pyinstaller --noconfirm digital_vcr.spec

echo.
echo Build complete:
echo   %CD%\dist\DigitalVCR\DigitalVCR.exe
endlocal
