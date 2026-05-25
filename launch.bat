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
    echo [L2D] Starting from repo...
    start "L2D" /d "%~dp0electron-l2d" cmd /k npm start
) else (
    echo [L2D] SKIP - not found
)

REM [2] Python backend
echo [PY] Starting backend (loading models)...
start "Backend" cmd /k "wsl python /home/baaai/projects/openheart/scripts/demo_full.py"

REM [3] Wait for WebSocket server
echo.
echo Waiting for Python server (port 9876)...
set /a N=0
:waitloop
ping -n 2 127.0.0.1 >nul
set /a N+=1
REM Try to detect if port 9876 is open via WSL
wsl ss -tlnp 2>nul | find "9876" >nul
if %errorlevel% equ 0 goto ready

REM Show progress bar
set "BAR="
for /l %%i in (1,1,20) do (
    if %%i leq !N! (set "BAR=!BAR!=") else (set "BAR=!BAR! ")
)
set /a P=N*5
if !P! gtr 100 set P=100
echo   [!BAR!] !P!%% 
goto waitloop

:ready
echo   [==============DONE===============] 100%%
echo   Python server is ready!

REM [4] Open frontend
start "" "\\wsl.localhost\Ubuntu\home\baaai\projects\openheart\frontend\index.html"

echo.
echo All services running!
pause
