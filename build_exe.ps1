$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    py -m venv .venv
}

Get-Process DigitalVCR -ErrorAction SilentlyContinue | Stop-Process -Force
$distDir = Join-Path $PWD 'dist\DigitalVCR'
if (Test-Path $distDir) {
    try {
        Remove-Item -LiteralPath $distDir -Recurse -Force -ErrorAction Stop
    } catch {
        $oldDir = Join-Path $PWD ("dist\DigitalVCR.old." + (Get-Date -Format "yyyyMMddHHmmss"))
        Rename-Item -LiteralPath $distDir -NewName (Split-Path $oldDir -Leaf) -ErrorAction Stop
        Write-Warning "Old dist folder was locked, moved it to $oldDir"
    }
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-build.txt
python -m PyInstaller --clean --noconfirm digital_vcr.spec

$exePath = Join-Path $PWD 'dist\DigitalVCR\DigitalVCR.exe'
if (-not (Test-Path $exePath)) {
    throw "Build finished without creating $exePath"
}

Write-Host ""
Write-Host "Build complete:"
Write-Host $exePath
