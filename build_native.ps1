$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$manifest = Join-Path $PSScriptRoot 'native\digital_vcr_core\Cargo.toml'
$rustRoot = Join-Path $PSScriptRoot 'native\digital_vcr_core'
$outDir = Join-Path $PSScriptRoot 'vcr\native'
$outDll = Join-Path $outDir 'digital_vcr_core.dll'

if (-not (Test-Path $manifest)) {
    throw "Rust source is missing: $manifest`nExtract the repair/overwrite ZIP directly into the Digital VCR project root."
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw 'Rust toolchain not found in PATH. Install Rust with rustup, reopen PowerShell, and retry.'
}

Write-Host 'Building Digital VCR Rust native core...'
Write-Host "Manifest: $manifest"
cargo build --release --manifest-path $manifest
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item -Force (Join-Path $rustRoot 'target\release\digital_vcr_core.dll') $outDll
Write-Host "Native core installed to: $outDll"
python -c "from vcr.native_core import native_status; print(native_status())"
