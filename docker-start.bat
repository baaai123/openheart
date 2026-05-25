@echo off
title OpenHeart - Docker Start
setlocal enabledelayedexpansion

REM ============================================================
REM OpenHeart — Docker Start Script (Windows)
REM v4.5.0 — Start infrastructure and/or app services
REM ============================================================
REM
REM Usage:
REM   docker-start.bat                        Start all services
REM   docker-start.bat --profile infra        Redis only
REM   docker-start.bat --profile app          Full stack
REM   docker-start.bat --mode mock            Mock mode
REM   docker-start.bat --vram-tier low        Force VRAM tier
REM ============================================================

REM Fix UNC path: map to drive letter
set "ROOT=%~dp0"
pushd "%ROOT%" 2>nul
if %errorlevel% neq 0 (
    echo UNC path detected. Creating drive mapping...
    subst Z: "%ROOT%"
    Z:
    set "ROOT=Z:\"
)
cd /d "%ROOT%"

REM Defaults
set "PROFILE=app"
set "MODE=real"
set "VRAM_TIER=auto"

REM Parse flags
:parse_args
if "%~1"=="" goto :parse_done
if /i "%~1"=="--profile" (
    set "PROFILE=%~2"
    if /i "!PROFILE!" neq "infra" if /i "!PROFILE!" neq "app" (
        echo [ERROR] --profile must be 'infra' or 'app'
        pause
        exit /b 1
    )
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--mode" (
    set "MODE=%~2"
    if /i "!MODE!" neq "mock" if /i "!MODE!" neq "real" (
        echo [ERROR] --mode must be 'mock' or 'real'
        pause
        exit /b 1
    )
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--vram-tier" (
    set "VRAM_TIER=%~2"
    if /i "!VRAM_TIER!" neq "auto" if /i "!VRAM_TIER!" neq "high" if /i "!VRAM_TIER!" neq "low" (
        echo [ERROR] --vram-tier must be 'auto', 'high', or 'low'
        pause
        exit /b 1
    )
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :usage
if /i "%~1"=="-h" goto :usage
echo [ERROR] Unknown option: %~1
exit /b 1
:usage
echo Usage: %~nx0 [--profile infra^|app] [--mode mock^|real] [--vram-tier auto^|high^|low]
pause
exit /b 0
:parse_done

echo ========================================
echo   OpenHeart - Docker Start
echo   Profile: %PROFILE% ^| Mode: %MODE% ^| VRAM: %VRAM_TIER%
echo ========================================
echo.

REM [1/5] Check Docker
echo [1/5] Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo   Install Docker Desktop from:
    echo   https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('docker --version') do echo  - %%i

REM Check Docker daemon
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker daemon is not running.
    echo   Start Docker Desktop first.
    pause
    exit /b 1
)
echo.

REM [2/5] Check NVIDIA/GPU (skip in mock mode)
echo [2/5] Checking GPU...
if /i "!MODE!"=="real" (
    where nvidia-smi >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [WARN] nvidia-smi not found. GPU passthrough may not work.
    ) else (
        for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name^,memory.total^,driver_version --format^=csv^,noheader 2^>nul') do echo  - GPU: %%i
    )
) else (
    echo  - Skipped (mock mode).
)
echo.

REM [3/5] Load .env
echo [3/5] Loading environment...
if exist "%ROOT%.env" (
    echo  - Found .env file.
    for /f "usebackq tokens=*" %%a in ("%ROOT%.env") do set "%%a"
    echo  - Environment loaded.
) else (
    echo  [WARN] No .env file found. Create one with DEEPSEEK_API_KEY=your_key_here
)

set "OPENHEART_MODE=%MODE%"
set "OPENHEART_VRAM_TIER=%VRAM_TIER%"
echo  - OPENHEART_MODE=!OPENHEART_MODE!
echo  - OPENHEART_VRAM_TIER=!OPENHEART_VRAM_TIER!
echo.

REM [4/5] Pull images
echo [4/5] Pulling Docker images...
if /i "!PROFILE!"=="app" (
    docker compose pull
) else (
    docker compose pull redis
)
echo.

REM [5/5] Start services
echo [5/5] Starting services...
if /i "!PROFILE!"=="infra" (
    docker compose up -d redis
) else (
    docker compose up -d
)
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose failed.
    pause
    exit /b 1
)
echo.

REM Health check loop
echo Waiting for services to become healthy...

set "MAX_RETRIES=30"
set "RETRY_INTERVAL=3"

if /i "!PROFILE!"=="infra" (
    set "SERVICE_LIST=redis"
) else (
    set "SERVICE_LIST=redis openheart frontend"
)

for %%s in (!SERVICE_LIST!) do (
    set "RETRIES=0"
    set "HEALTHY=0"
    echo  Checking %%s...
    
    :health_loop
    if !RETRIES! geq !MAX_RETRIES! goto :health_timeout
    
    REM Check container health via docker inspect
    for /f "tokens=*" %%h in ('docker inspect --format="{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}" "openheart-%%s" 2^>nul') do set "STATUS=%%h"
    
    if /i "!STATUS!"=="healthy" (
        echo   - %%s: healthy
        set "HEALTHY=1"
        goto :health_next
    )
    
    set /a "RETRIES+=1"
    timeout /t !RETRY_INTERVAL! /nobreak >nul
    goto :health_loop
    
    :health_timeout
    echo   [WARN] %%s did not become healthy within timeout.
    echo   Check logs: docker compose logs %%s
    
    :health_next
)

echo.

REM Status summary
echo ========================================
echo   OpenHeart - Status Summary
echo ========================================
docker compose ps

echo.
echo  ---
echo  OpenHeart is running.
echo.
echo   Frontend:  http://localhost:80
echo   API:       http://localhost:9876
echo   WebSocket: ws://localhost:9876
echo   Redis:     localhost:6379
echo.
echo   To stop:       docker-stop.bat
echo   To stop+wipe:  docker-stop.bat --volumes
echo   To view logs:  docker compose logs -f
echo  ---

REM Cleanup drive mapping if used
subst Z: /D >nul 2>&1

endlocal
