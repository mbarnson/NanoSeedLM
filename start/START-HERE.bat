@echo off
rem start/START-HERE.bat [PACK [MODEL_FOLDER]] - NanoSeedLM's start on a Windows PC: start-here.ps1 beside it (no Python).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-here.ps1" %*
if errorlevel 1 pause
