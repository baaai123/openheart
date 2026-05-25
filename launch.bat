@echo off
title OpenHeart
setlocal enabledelayedexpansion

echo OpenHeart Launcher v5.x
echo ==============================
echo.

REM [1] L2D
if exist "D:\electron-l2d\main.js" (
    echo [L2D] Starting from D:\electron-l2d...
    start "L2D" /d "D:\electron-l2d" cmd /k npm start
) else (
    echo [L2D] SKIP - not found
)

REM [2] Python backend
echo.
echo [PY] Starting Python backend...
echo ==============================
echo.
wsl bash /home/baaai/projects/openheart/run_backend.sh
echo.
echo Python backend exited.
pause
