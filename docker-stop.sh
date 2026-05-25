#!/usr/bin/env bash
# OpenHeart — Docker Stop Script
# v4.5.0 — Stop and optionally clean up Docker Compose services
#
# Usage:
#   ./docker-stop.sh              # docker compose down
#   ./docker-stop.sh --volumes    # docker compose down -v (wipe data)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

# Color helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
INFO="${CYAN}[INFO]${NC}"
OK="${GREEN}[OK]${NC}"
WARN="${YELLOW}[WARN]${NC}"

VOLUMES=""
CONFIRM="yes"

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --volumes|-v)
            VOLUMES="--volumes"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--volumes]"
            echo "  --volumes  Also remove named volumes (Redis data, cache)"
            exit 0
            ;;
        *)
            echo -e "${WARN} Unknown option: $1"
            shift
            ;;
    esac
done

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  OpenHeart — Docker Stop${NC}"
if [[ -n "$VOLUMES" ]]; then
    echo -e "${CYAN}  (with --volumes — data will be wiped)${NC}"
fi
echo -e "${CYAN}========================================${NC}"
echo ""

# Confirm if --volumes
if [[ -n "$VOLUMES" ]]; then
    echo -e "${WARN} This will DELETE all Redis data and caches!"
    read -r -p "Are you sure? [y/N] " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo -e "${INFO} Stopping services..."

if [[ -f "$COMPOSE_FILE" ]]; then
    # shellcheck disable=SC2086
    docker compose -f "$COMPOSE_FILE" down $VOLUMES
    echo -e "${OK} Services stopped."
else
    echo -e "${WARN} No docker-compose.yml found at ${COMPOSE_FILE}"
fi

echo ""
echo -e "${OK} Done."
