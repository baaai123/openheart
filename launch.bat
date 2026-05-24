@echo off
title OpenHeart Launcher
setlocal enabledelayedexpansion

REM ============================================================
REM OpenHeart - Windows Batch Launcher
REM ============================================================

REM Fix UNC path: map to drive letter
set "ROOT=%~dp0"
pushd "%ROOT%" 2>nul
if %errorlevel% neq 0 (
    echo UNC path detected. Creating drive mapping...
    subst Z: "%ROOT%"
    Z:
    set "ROOT=Z:\"
)
cd /d "%ROOT%"

echo ========================================
echo   OpenHeart - Starting All Services
echo ========================================
echo.

REM [1/4] Start Electron L2D
echo [1/4] Starting Electron L2D...
if not exist "%ROOT%electron-l2d\node_modules" (
    echo [ERROR] electron-l2d dependencies not installed.
    echo   Run: cd electron-l2d ^&^& npm install
    pause
    exit /b 1
)
start "OpenHeart-L2D" cmd /c "cd /d %ROOT%electron-l2d && npm start"
echo  - L2D app starting...
echo.

REM [2/4] Check Redis
echo [2/4] Checking Redis...
wsl redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo  - Redis not running, starting via WSL...
    start "OpenHeart-Redis" wsl redis-server
) else (
    echo  - Redis already running.
)
echo.

REM [3/4] Wait for startup
echo [3/4] Waiting 4 seconds...
timeout /t 4 /nobreak >nul
echo.

REM [4/4] Open frontend and start backend
echo [4/4] Opening frontend + starting Python...
start "" "%ROOT%frontend\index.html"

echo ========================================
echo   Starting Python backend...
echo   Press Ctrl+C to stop.
echo ========================================
python scripts/demo_full.py

REM Cleanup drive mapping if used
subst Z: /D >nul 2>&1
if %errorlevel% neq 0 (
    echo Backend exited with error code %errorlevel%
    pause
)
