@echo off
title OpenHeart Launcher
setlocal enabledelayedexpansion

REM ============================================================
REM OpenHeart — Windows Batch Launcher
REM Starts all components of the OpenHeart virtual companion.
REM ============================================================

set "ROOT_DIR=%~dp0"

echo ========================================
echo   OpenHeart — Starting All Services
echo ========================================
echo.

REM ------ 1. Start Electron L2D App ------
echo [1/4] Starting Electron L2D app...
cd /d "%ROOT_DIR%electron-l2d"
if %errorlevel% neq 0 (
    echo [ERROR] Could not find electron-l2d directory at "%ROOT_DIR%electron-l2d"
    echo         Make sure the repository is complete.
    pause
    exit /b 1
)

start "OpenHeart-L2D" cmd /c "npm start 2>&1"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to launch npm start in electron-l2d.
    echo         Check that Node.js and electron-l2d dependencies are installed.
    pause
    exit /b 1
)
echo  - L2D app starting in new window...
echo.

REM ------ 2. Start Redis via WSL (if not running) ------
echo [2/4] Checking Redis (WSL)...
wsl redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo  - Redis not running. Starting via WSL...
    start "OpenHeart-Redis" cmd /c "wsl redis-server 2>&1"
    if !errorlevel! neq 0 (
        echo [WARNING] Failed to start redis-server via WSL.
        echo           The Python backend may fail if Redis is required.
        echo           Continuing anyway...
    ) else (
        echo  - Redis server started.
    )
) else (
    echo  - Redis is already running.
)
echo.

REM ------ 3. Wait for Electron to load ------
echo [3/4] Waiting 3 seconds for Electron to load...
timeout /t 3 /nobreak >nul
echo  - Done.
echo.

REM ------ 4. Open Frontend in Browser ------
echo [4/4] Opening frontend in browser...
start "" "%ROOT_DIR%frontend\index.html"
if %errorlevel% neq 0 (
    echo [WARNING] Could not open frontend/index.html in browser.
    echo           Open it manually: %ROOT_DIR%frontend\index.html
)
echo  - Frontend opened.
echo.

REM ------ 5. Start Python Backend ------
echo ========================================
echo   Starting Python Backend (main process)
echo ========================================
echo  - Running: python scripts/demo_full.py
echo  - Press Ctrl+C to stop the backend.
echo  - Close the L2D window to stop the avatar.
echo.

cd /d "%ROOT_DIR%"

REM Run the Python backend in the current window so logs are visible.
python scripts/demo_full.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python backend exited with code %errorlevel%.
    echo         Check scripts/demo_full.py for errors.
    echo         Make sure all Python dependencies are installed.
    pause
    exit /b %errorlevel%
)

echo.
echo Backend exited cleanly.
pause
