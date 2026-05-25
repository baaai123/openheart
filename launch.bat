@echo off
title OpenHeart
setlocal enabledelayedexpansion

REM Find root: go up to the directory containing src/
set "ROOT=%~dp0"
:findroot
if exist "%ROOT%src\" goto root_found
set "ROOT=%ROOT%..\"
goto findroot
:root_found
cd /d "%ROOT%"

echo ========================================
echo   OpenHeart Launcher
echo ========================================
echo.

REM [1] Electron L2D
echo [1/4] Electron L2D...
set "L2D=%ROOT%electron-l2d"
if exist "%L2D%\package.json" (
    if not exist "%L2D%\node_modules" (
        echo   Installing npm dependencies...
        cd /d "%L2D%"
        call npm install
        cd /d "%ROOT%"
    )
    start "L2D" cmd /c "cd /d %L2D% && npm start"
    echo   L2D starting...
) else (
    echo   [SKIP] electron-l2d not found
)
echo.

REM [2] Docker
echo [2/4] Docker services...
docker compose up -d 2>nul
if %errorlevel% equ 0 (
    echo   Backend + Redis + Frontend started
) else (
    echo   [SKIP] Docker not available - run Python directly
    start "Backend" cmd /c "cd /d %ROOT% && python scripts/demo_full.py"
)
echo.

REM [3] Wait
echo [3/4] Waiting 4s...
timeout /t 4 /nobreak >nul
echo.

REM [4] Frontend
echo [4/4] Opening frontend...
start "" "%ROOT%frontend\index.html"

echo ========================================
echo   All services started!
echo   Close this window to stop everything.
echo ========================================
pause
