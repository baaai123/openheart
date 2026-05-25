@echo off
title OpenHeart

:: Launch Python backend via WSL (minimized, log to /tmp/openheart.log)
start /min wsl bash -c "cd /home/baaai/projects/openheart && python scripts/demo_full.py > /tmp/openheart.log 2>&1"

:: Launch Electron L2D + Control Panel (visible)
if exist "D:\electron-l2d\package.json" (
    start "OpenHeart" /d "D:\electron-l2d" cmd /k npm start
)
