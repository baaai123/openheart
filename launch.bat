@echo off
title OpenHeart
echo OpenHeart Launcher v5.x
echo ==============================
echo.

REM Check if VcXsrv is running
echo Checking display server...
wsl bash -c "export DISPLAY=:0 && xset q >nul 2>&1 && echo 'OK' || echo 'Start VcXsrv first'"

echo.
echo Starting Desktop UI...
wsl bash -c "export DISPLAY=:0 && source /home/baaai/miniforge3/etc/profile.d/conda.sh && conda activate cv311 && python /home/baaai/projects/openheart/frontend/desktop_ui.py"
echo.
pause
