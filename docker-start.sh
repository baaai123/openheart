#!/usr/bin/env bash
# OpenHeart — Docker Start Script
# v4.5.0 — Start infrastructure and/or app services via Docker Compose
#
# Usage:
#   ./docker-start.sh                         # Start all services
#   ./docker-start.sh --profile infra         # Redis only
#   ./docker-start.sh --profile app           # Full stack
#   ./docker-start.sh --mode mock             # Mock mode (no GPU)
#   ./docker-start.sh --vram-tier low         # Force VRAM tier
#   ./docker-start.sh --profile app --mode real --vram-tier auto

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
ENV_FILE="${SCRIPT_DIR}/.env"

# Defaults
PROFILE="app"        # infra | app
MODE="real"          # mock | real
VRAM_TIER="auto"     # auto | high | low

# ── Color helpers ─────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
INFO="${CYAN}[INFO]${NC}"
OK="${GREEN}[OK]${NC}"
WARN="${YELLOW}[WARN]${NC}"
ERR="${RED}[ERROR]${NC}"

# ── Parse flags ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            if [[ "$2" != "infra" && "$2" != "app" ]]; then
                echo -e "${ERR} --profile must be 'infra' or 'app'"
                exit 1
            fi
            PROFILE="$2"
            shift 2
            ;;
        --mode)
            if [[ "$2" != "mock" && "$2" != "real" ]]; then
                echo -e "${ERR} --mode must be 'mock' or 'real'"
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        --vram-tier)
            if [[ "$2" != "auto" && "$2" != "high" && "$2" != "low" ]]; then
                echo -e "${ERR} --vram-tier must be 'auto', 'high', or 'low'"
                exit 1
            fi
            VRAM_TIER="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--profile infra|app] [--mode mock|real] [--vram-tier auto|high|low]"
            exit 0
            ;;
        *)
            echo -e "${ERR} Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Banner ────────────────────────────────────────────────────────────
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  OpenHeart — Docker Start${NC}"
echo -e "${CYAN}  Profile: ${PROFILE} | Mode: ${MODE} | VRAM: ${VRAM_TIER}${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ── [1/5] Check Docker ──────────────────────────────────────────────
echo -e "${INFO} [1/5] Checking Docker installation..."
if ! command -v docker &>/dev/null; then
    echo -e "${ERR} Docker is not installed."
    echo "  Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

DOCKER_VERSION=$(docker --version 2>/dev/null)
echo -e "${OK}  ${DOCKER_VERSION}"

# Check Docker daemon is running
if ! docker info &>/dev/null; then
    echo -e "${ERR} Docker daemon is not running."
    echo "  Start Docker: systemctl start docker  (or Docker Desktop)"
    exit 1
fi
echo ""

# ── [2/5] Check NVIDIA GPU (real mode only) ─────────────────────────
if [[ "$MODE" == "real" ]]; then
    echo -e "${INFO} [2/5] Checking NVIDIA GPU..."
    if ! command -v nvidia-smi &>/dev/null; then
        echo -e "${WARN} nvidia-smi not found — GPU passthrough may not work."
        echo "  Install nvidia-container-toolkit if using GPU."
    else
        NVIDIA_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$NVIDIA_INFO" ]]; then
            echo -e "${OK}  GPU: ${NVIDIA_INFO}"
        else
            echo -e "${WARN}  nvidia-smi available but no GPU detected."
        fi
    fi
    echo ""
else
    echo -e "${INFO} [2/5] Skipping GPU check (mock mode)."
    echo ""
fi

# ── [3/5] Source .env ───────────────────────────────────────────────
echo -e "${INFO} [3/5] Loading environment..."
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    echo -e "${OK}  Sourced ${ENV_FILE}"
else
    echo -e "${WARN}  No .env file found at ${ENV_FILE}"
    echo "  Create one with DEEPSEEK_API_KEY=your_key_here"
fi

# Export environment variables for docker compose
export OPENHEART_MODE="${MODE}"
export OPENHEART_VRAM_TIER="${VRAM_TIER}"
echo -e "${OK}  OPENHEART_MODE=${OPENHEART_MODE}"
echo -e "${OK}  OPENHEART_VRAM_TIER=${OPENHEART_VRAM_TIER}"
echo ""

# ── [4/5] Pull images ──────────────────────────────────────────────
echo -e "${INFO} [4/5] Pulling Docker images..."
if [[ "$PROFILE" == "app" ]]; then
    docker compose -f "$COMPOSE_FILE" pull
else
    docker compose -f "$COMPOSE_FILE" pull redis
fi
echo ""

# ── [5/5] Start services ───────────────────────────────────────────
echo -e "${INFO} [5/5] Starting services..."

if [[ "$PROFILE" == "infra" ]]; then
    docker compose -f "$COMPOSE_FILE" up -d redis
else
    docker compose -f "$COMPOSE_FILE" up -d
fi

COMPOSE_EXIT=$?
if [[ $COMPOSE_EXIT -ne 0 ]]; then
    echo -e "${ERR} Docker Compose failed (exit code ${COMPOSE_EXIT})."
    exit 1
fi
echo ""

# ── Health check loop ──────────────────────────────────────────────
echo -e "${INFO} Waiting for services to become healthy..."
if [[ "$PROFILE" == "infra" ]]; then
    SERVICES=("redis")
else
    SERVICES=("redis" "openheart" "frontend")
fi

MAX_RETRIES=30
RETRY_INTERVAL=3

for svc in "${SERVICES[@]}"; do
    echo -n "  ${svc}: "
    RETRIES=0
    while [[ $RETRIES -lt $MAX_RETRIES ]]; do
        STATUS=$(docker compose -f "$COMPOSE_FILE" ps --format json "$svc" 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        for d in data:
            if d.get('Health', '') == 'healthy':
                print('healthy')
                break
        else:
            print([d.get('Health', 'unknown') for d in data][0] if data else 'unknown')
    else:
        print(data.get('Health', 'unknown'))
except Exception:
    print('unknown')
" 2>/dev/null || echo "starting")

        if [[ "$STATUS" == "healthy" ]]; then
            echo -e "${OK}"
            break
        fi
        echo -n "."
        sleep $RETRY_INTERVAL
        RETRIES=$((RETRIES + 1))
    done

    if [[ $RETRIES -ge $MAX_RETRIES ]]; then
        echo -e " ${YELLOW}timed out${NC}"
        echo -e "${WARN}  ${svc} did not become healthy within $((MAX_RETRIES * RETRY_INTERVAL))s."
        echo "  Check logs: ./docker-logs.sh --service ${svc}"
    fi
done
echo ""

# ── Status summary ─────────────────────────────────────────────────
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  OpenHeart — Status Summary${NC}"
echo -e "${CYAN}========================================${NC}"

docker compose -f "$COMPOSE_FILE" ps

echo ""
echo -e "${CYAN}---${NC}"
echo -e "${GREEN}OpenHeart is running.${NC}"
echo ""
echo "  Frontend:  http://localhost:80"
echo "  API:       http://localhost:9876"
echo "  WebSocket: ws://localhost:9876"
echo "  Redis:     localhost:6379"
echo ""
echo "  To view logs:  ./docker-logs.sh"
echo "  To stop:       ./docker-stop.sh"
echo "  To stop + wipe volumes: ./docker-stop.sh --volumes"
echo ""
echo -e "${CYAN}---${NC}"
