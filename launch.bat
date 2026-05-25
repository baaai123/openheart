@echo off
title OpenHeart

echo OpenHeart Launcher v5.x

REM Detect UNC path early
set "MYPATH=%~dp0"
echo %MYPATH% | find "\\" >nul && (
    echo UNC path detected - mapping Z: drive...
    subst Z: "%~dp0" 2>nul || (
        echo ERROR: Cannot access UNC path directly.
        echo Map a drive letter first:
        echo   net use Z: \\wsl.localhost\Ubuntu\home\baaai\projects\openheart
        echo   Z:
        echo   launch.bat
        pause
        exit
    )
    Z:
    set "ROOT=Z:\"
) || set "ROOT=%~dp0"

echo Root: %ROOT%
echo.

echo [1/3] Electron L2D...
if exist "%ROOT%electron-l2d\package.json" (
    start "L2D" /d "%ROOT%electron-l2d" cmd /k npm start
    echo   L2D starting.
) else echo   [SKIP]

echo [2/3] Backend (Python)...
echo   Run: python scripts/demo_full.py
start "Backend" cmd /k "cd /d %ROOT% && python scripts/demo_full.py"

echo [3/3] Frontend...
start "" "%ROOT%frontend\index.html"

echo Done!
pause
