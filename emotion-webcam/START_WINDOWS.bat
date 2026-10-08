@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_windows.ps1"
if errorlevel 1 (
  echo.
  echo Setup or application failed. Read docs\TROUBLESHOOTING.md for help.
  pause
)
endlocal
