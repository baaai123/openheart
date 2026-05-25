@echo off
title OpenHeart

:: Use script directory as project root (works from any location)
set "PRJ=%~dp0"

:: Launch Python backend via WSL (minimized, log to /tmp/openheart.log)
start /b wsl bash -c "cd '%PRJ%' && python scripts/demo_full.py > /tmp/openheart.log 2>&1"

:: Launch Electron L2D + Control Panel (visible)
if exist "%PRJ%electron-l2d\package.json" (
    start "OpenHeart" /d "%PRJ%electron-l2d" cmd /k npm start
)
