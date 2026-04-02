$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    py -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-build.txt
pyinstaller --noconfirm digital_vcr.spec

Write-Host ""
Write-Host "Build complete:"
Write-Host (Join-Path $PWD 'dist\DigitalVCR\DigitalVCR.exe')
