@echo off
title OpenHeart

echo OpenHeart Launcher v5.x
echo ==============================
echo.

REM Start desktop UI in a visible WSL terminal
start "OpenHeart-UI" wsl bash -c "export DISPLAY=:0 && source /home/baaai/miniforge3/etc/profile.d/conda.sh && conda activate cv311 && python /home/baaai/projects/openheart/frontend/desktop_ui.py; read -p 'Press Enter to close...'"

echo Desktop UI launched in WSL window.
echo Check the WSL window for any errors.
pause
