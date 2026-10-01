@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ===============================================
echo   Digital VCR V8.1 - Windows Installer Builder
echo ===============================================
echo.

if /I "%~1"=="--skip-exe" goto find_inno

echo [1/3] Building Digital VCR executable...
call build_exe.bat
if errorlevel 1 (
  echo.
  echo ERROR: EXE build failed. Installer was not created.
  exit /b 1
)

:find_inno
echo.
echo [2/3] Locating Inno Setup 6 compiler...
set "ISCC="
for /f "delims=" %%I in ('where ISCC.exe 2^>nul') do if not defined ISCC set "ISCC=%%I"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not defined ISCC (
  echo Inno Setup 6 was not found.
  where winget >nul 2>nul
  if errorlevel 1 (
    echo Install Inno Setup 6 from https://jrsoftware.org/isdl.php and run this file again.
    exit /b 1
  )
  echo.
  choice /C YN /N /M "Install Inno Setup 6 automatically with winget? [Y/N] "
  if errorlevel 2 exit /b 1
  winget install --id JRSoftware.InnoSetup -e --source winget --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo ERROR: Inno Setup installation failed.
    exit /b 1
  )
  if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
  if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
  if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)

if not defined ISCC (
  echo ERROR: ISCC.exe still could not be found.
  exit /b 1
)

if not exist "dist\DigitalVCR\DigitalVCR.exe" (
  echo ERROR: dist\DigitalVCR\DigitalVCR.exe does not exist.
  echo Run build_installer.bat without --skip-exe first.
  exit /b 1
)

if not exist "assets\DigitalVCR.ico" (
  echo ERROR: assets\DigitalVCR.ico is missing.
  exit /b 1
)

echo.
echo [3/3] Compiling installer with Inno Setup...
"%ISCC%" /Qp "installer\DigitalVCR.iss"
if errorlevel 1 (
  echo.
  echo ERROR: Installer compilation failed.
  exit /b 1
)

echo.
echo ===============================================
echo Installer complete:
echo   "%CD%\installer\output\Digital-VCR-V8.1-Setup.exe"
echo ===============================================
endlocal
