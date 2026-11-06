@echo off
rem start/START-HERE.bat - NanoSeedLM's start on a Windows PC: it runs start-here.ps1 beside it (no Python needed).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-here.ps1"
if errorlevel 1 pause
