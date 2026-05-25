@echo off
title OpenHeart

echo OpenHeart Launcher v5.x
echo ==============================
echo.
echo Opening control panel in browser...
start "" "\\wsl.localhost\Ubuntu\home\baaai\projects\openheart\frontend\index.html"
echo.
echo After configuring settings, run the backend:
echo   wsl bash /home/baaai/projects/openheart/run_backend.sh
echo.
pause
