@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MANIFEST=%~dp0native\digital_vcr_core\Cargo.toml"
set "RUSTROOT=%~dp0native\digital_vcr_core"
set "OUTDIR=%~dp0vcr\native"
set "OUTDLL=%OUTDIR%\digital_vcr_core.dll"

if not exist "%MANIFEST%" (
  echo.
  echo ERROR: Rust source is missing:
  echo   %MANIFEST%
  echo.
  echo The project root must contain:
  echo   native\digital_vcr_core\Cargo.toml
  echo   native\digital_vcr_core\src\lib.rs
  echo.
  echo Extract the Rust repair/overwrite ZIP directly into:
  echo   %~dp0
  echo and allow folders/files to be merged.
  echo.
  exit /b 2
)

where cargo >nul 2>nul
if errorlevel 1 (
  echo.
  echo ERROR: Rust toolchain not found in PATH.
  echo Install Rust with rustup, reopen this terminal, then run build_native.bat again.
  echo.
  exit /b 1
)

echo Building Digital VCR Rust native core...
echo Manifest: %MANIFEST%
cargo build --release --manifest-path "%MANIFEST%"
if errorlevel 1 (
  echo.
  echo ERROR: Cargo build failed.
  exit /b %errorlevel%
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
copy /Y "%RUSTROOT%\target\release\digital_vcr_core.dll" "%OUTDLL%" >nul
if errorlevel 1 (
  echo.
  echo ERROR: Built DLL could not be copied to:
  echo   %OUTDLL%
  exit /b %errorlevel%
)

echo.
echo Native core installed to:
echo   %OUTDLL%
python -c "from vcr.native_core import native_status; print(native_status())"
if errorlevel 1 (
  echo WARNING: DLL built, but Python native-status check failed.
)
endlocal
