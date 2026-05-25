@echo off
title OpenHeart - Docker Stop
setlocal enabledelayedexpansion

REM ============================================================
REM OpenHeart — Docker Stop Script (Windows)
REM v4.5.0 — Stop and optionally clean up Docker Compose services
REM ============================================================
REM
REM Usage:
REM   docker-stop.bat              docker compose down
REM   docker-stop.bat --volumes    docker compose down -v
REM ============================================================

set "VOLUMES="

if /i "%~1"=="--volumes" (
    set "VOLUMES=-v"
    echo ========================================
    echo   OpenHeart - Docker Stop (with --volumes)
    echo   WARNING: This will DELETE all Redis data!
    echo ========================================
    echo.
    set /p "CONFIRM=Are you sure? [y/N]: "
    if /i not "!CONFIRM!"=="y" (
        echo Aborted.
        pause
        exit /b 0
    )
) else if /i "%~1"=="--help" (
    echo Usage: %~nx0 [--volumes]
    echo   --volumes  Also remove named volumes ^(Redis data, cache^)
    pause
    exit /b 0
) else if /i "%~1"=="-h" (
    echo Usage: %~nx0 [--volumes]
    pause
    exit /b 0
)

echo.
echo Stopping OpenHeart services...

docker compose down %VOLUMES%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to stop services.
    pause
    exit /b 1
)

echo  - Services stopped.
echo.
echo Done.

endlocal
