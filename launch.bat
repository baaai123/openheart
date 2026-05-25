@echo off
title OpenHeart

echo OpenHeart Launcher v5.x
echo ==============================
echo.

REM Start Electron L2D with built-in config panel
if exist "D:\electron-l2d\package.json" (
    echo Starting L2D + Config Panel...
    start "OpenHeart" /d "D:\electron-l2d" cmd /k npm start
) else (
    echo electron-l2d not found at D:\electron-l2d
    echo Run: npm install in that directory first
)

echo.
echo To start Python backend:
echo   wsl bash /home/baaai/projects/openheart/run_backend.sh
echo.
pause
