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
) else if exist "%~dp0electron-l2d\main.js" (
    start "L2D" /d "%~dp0electron-l2d" cmd /k npm start
) else echo [L2D] SKIP

REM [2] Python backend
echo [PY] Starting backend...
start "Backend" cmd /k "wsl python /home/baaai/projects/openheart/scripts/demo_full.py"

REM [3] Wait for server
echo.
echo Waiting for Python server on port 9876...
set /a CNT=0
:waitloop
set /a CNT+=1
ping -n 3 127.0.0.1 >nul
wsl ss -tlnp 2>nul | find ":9876" >nul
if !errorlevel! equ 0 goto ready
echo   Retry !CNT!... (waiting for models to load)
goto waitloop

:ready
echo   Server is READY on port 9876!
echo.

REM [4] Frontend
start "" "\\wsl.localhost\Ubuntu\home\baaai\projects\openheart\frontend\index.html"

echo All services running!
pause
