@echo off
title OpenHeart

echo OpenHeart Launcher v5.x
echo ==============================
echo.

REM L2D
if exist "D:\electron-l2d\package.json" (
    echo [L2D] Starting...
    start "L2D" /d "D:\electron-l2d" cmd /k npm start
) else echo [L2D] SKIP - not found

REM Python backend
echo [PY] Starting backend via WSL...
wsl bash -c "source /home/baaai/miniforge3/etc/profile.d/conda.sh && conda activate cv311 && python /home/baaai/projects/openheart/scripts/demo_full.py"

echo.
echo All stopped.
pause
