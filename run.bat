@echo off
title Afterimage
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 (
  echo.
  echo Afterimage did not start. Review the error above.
  pause
)
