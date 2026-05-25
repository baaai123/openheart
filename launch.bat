@echo off
title OpenHeart

echo OpenHeart Launcher v5.x
echo ========================
echo.

REM Find project root (look for src/ directory)
set "ROOT=%~dp0"
:findroot
if exist "%ROOT%src\" goto root_found
set "ROOT=%ROOT%..\"
goto findroot
:root_found
cd /d "%ROOT%" 2>nul || (
    echo [ERROR] Cannot access: %ROOT%
    echo If running from UNC path, map a drive letter first:
    echo   net use Z: \\wsl.localhost\Ubuntu\home\baaai\projects\openheart
    echo   Z:
    echo   launch.bat
    pause
    exit /b 1
)
echo Root: %ROOT%
echo.

REM [1] Electron L2D
echo [1/4] Electron L2D...
set "L2D=%ROOT%electron-l2d"
if exist "%L2D%\package.json" (
    if not exist "%L2D%\node_modules" (
        echo   Installing npm packages...
        cd /d "%L2D%"
        call npm install 2>&1
        cd /d "%ROOT%"
    )
    start "OpenHeart-L2D" cmd /k "cd /d %L2D% && npm start"
    echo   L2D window opened.
) else (
    echo   [SKIP] electron-l2d not found at %L2D%
)
echo.

REM [2] Docker or Python
echo [2/4] Backend...
where docker >nul 2>&1
if %errorlevel% equ 0 (
    docker compose -f "%ROOT%docker-compose.yml" up -d 2>&1
    if %errorlevel% equ 0 (
        echo   Docker containers started.
    ) else (
        echo   Docker failed. Try: docker compose up -d
    )
) else (
    echo   Docker not found - run Python directly.
    echo   Start: python scripts/demo_full.py
)
echo.

REM [3] Wait for services
echo [3/4] Waiting 4 seconds...
ping -n 5 127.0.0.1 >nul
echo.

REM [4] Open frontend
echo [4/4] Opening frontend...
start "" "%ROOT%frontend\index.html"

echo ========================================
echo   Done! Press any key to close...
echo ========================================
pause
