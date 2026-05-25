@echo off
title OpenHeart

echo OpenHeart Launcher v5.x
echo.

REM Where is the Python project?
set "WSL_PROJECT=\\wsl.localhost\Ubuntu\home\baaai\projects\openheart"

REM [1] Electron L2D (from D:\electron-l2d or repo copy)
if exist "D:\electron-l2d\main.js" (
    echo [1/3] L2D from D:\electron-l2d...
    start "L2D" /d "D:\electron-l2d" cmd /k npm start
) else if exist "%~dp0electron-l2d\main.js" (
    echo [1/3] L2D from repo...
    start "L2D" /d "%~dp0electron-l2d" cmd /k npm start
) else (
    echo [1/3] L2D [SKIP] - not found
)
echo.

REM [2] Python backend via WSL
echo [2/3] Python via WSL...
wsl python %WSL_PROJECT%\scripts\demo_full.py 2>&1
echo.

echo [3/3] Frontend...
REM start "" "%WSL_PROJECT%\frontend\index.html"

echo Done!
pause
