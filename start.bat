@echo off
rem Double-clickable launcher for Windows.
rem Runs install.ps1 with the execution policy bypassed, so it works on a
rem clean machine and on files extracted from a downloaded ZIP (mark-of-the-web
rem would otherwise block the .ps1 under the default policy).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 (
    echo.
    echo Afterimage didn't start. Read the message above for what to fix.
    pause
)
