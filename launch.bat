@echo off
title OpenHeart
echo OpenHeart Launcher v5.x
echo ==============================
echo.
echo Starting Desktop UI...
wsl bash -c "source /home/baaai/miniforge3/etc/profile.d/conda.sh && conda activate cv311 && python /home/baaai/projects/openheart/frontend/desktop_ui.py"
echo.
pause
