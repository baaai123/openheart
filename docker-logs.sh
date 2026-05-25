#!/usr/bin/env bash
# OpenHeart — Docker Logs Script
# v4.5.0 — Tail logs from Docker Compose services
#
# Usage:
#   ./docker-logs.sh               # Tail all services
#   ./docker-logs.sh --service redis    # Redis only
#   ./docker-logs.sh -s openheart       # OpenHeart only
#   ./docker-logs.sh -s frontend -n 100 # Last 100 lines, follow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"

SERVICE=""
FOLLOW="-f"
TAIL=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --service|-s)
            SERVICE="$2"
            shift 2
            ;;
        -n)
            TAIL="$2"
            shift 2
            ;;
        --no-follow)
            FOLLOW=""
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--service redis|openheart|frontend] [-n N] [--no-follow]"
            echo ""
            echo "  --service, -s   Service name (default: all)"
            echo "  -n              Number of recent lines to show"
            echo "  --no-follow     Print and exit (don't tail)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build args array
ARGS=()

if [[ -n "$FOLLOW" ]]; then
    ARGS+=("-f")
fi

if [[ -n "$TAIL" ]]; then
    ARGS+=("--tail" "$TAIL")
fi

if [[ -n "$SERVICE" ]]; then
    ARGS+=("$SERVICE")
fi

exec docker compose -f "$COMPOSE_FILE" logs "${ARGS[@]}"
