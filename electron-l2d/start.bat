@echo off
title OpenHeart L2D
cd /d "%~dp0"
cls

echo.
echo ============================================
echo   OpenHeart L2D - Starting Voice Engine...
echo ============================================
echo.
echo Loading models (SenseVoice + CosyVoice3 + vLLM)...
echo This may take 30-60 seconds on first launch.
echo.

:: Start WSL voice loop in a visible window
start "OpenHeart Voice Engine" wsl -d Ubuntu -- bash -i -l -c "cd /home/baaai/projects/openheart && ./run.sh"

echo Waiting for voice engine...

:: Wait for WS server port 9876 to be ready (max 120 seconds)
set /a count=0
:waitloop
    timeout /t 2 /nobreak >nul
    set /a count+=1
    
    :: Try to connect to WS server via proper HTTP check
    wsl -d Ubuntu -- bash -c "curl -s --max-time 1 http://127.0.0.1:9876 >nul 2>&1" 2>nul
    if %errorlevel% equ 0 goto ready
    
    :: Show progress dots every 5 checks (10s)
    set /a mod=count %% 5
    if %mod% equ 0 (<nul set /p=.)
    
    if %count% gtr 60 (
        echo.
        echo Timed out waiting for voice engine.
        echo Check the WSL window for errors.
        pause
        exit /b 1
    )
    goto waitloop

:ready
echo.
echo Voice engine ready! Launching L2D avatar...

:: Start Electron L2D
start "" npx electron .

echo.
echo OpenHeart L2D is running!
echo.
pause
