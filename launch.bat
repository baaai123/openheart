@echo off
title OpenHeart

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
wsl bash -c "cd /home/baaai/projects/openheart export PATH=/home/baaai/miniforge3/envs/cv311/bin:$PATH && cd /home/baaai/projects/openheart && pythonexport PATH=/home/baaai/miniforge3/envs/cv311/bin:$PATH && cd /home/baaai/projects/openheart && python /home/baaai/miniforge3/envs/cv311/bin/python scripts/demo_full.py 2>&1"
echo.
echo Python backend exited.
pause
