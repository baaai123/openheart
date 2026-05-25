@echo off
title OpenHeart
setlocal enabledelayedexpansion

echo OpenHeart Launcher v5.x
echo ========================
echo.

REM Find root directory
pushd "%~dp0"
:findroot
if exist "src\" goto found
cd ..
goto findroot
:found
set "ROOT=%cd%"
popd
cd /d "%ROOT%"
echo Root: %ROOT%
echo.

REM [1] L2D
echo [1/4] Electron L2D...
set "L2D=%ROOT%electron-l2d"
if exist "%L2D%\package.json" (
    if not exist "%L2D%\node_modules\" (
        echo   Installing npm...
        pushd "%L2D%"
        call npm install
        popd
    )
    start "L2D" /d "%L2D%" cmd /k npm start
    echo   L2D starting.
) else (
    echo   [SKIP] not found
)
echo.

REM [2] Backend
echo [2/4] Backend...
where docker >nul 2>&1
if %errorlevel% equ 0 (
    docker compose up -d
    echo   Docker running.
) else (
    echo   Run: python scripts/demo_full.py
)
echo.

REM [3] Wait
echo [3/4] Waiting...
ping -n 5 127.0.0.1 >nul
echo.

REM [4] Frontend
echo [4/4] Frontend...
start "" "%ROOT%frontend\index.html"

echo Done!
pause
