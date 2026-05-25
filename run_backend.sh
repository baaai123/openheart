#!/bin/bash
set -e

# Source .env if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

# Activate conda
source /home/baaai/miniforge3/etc/profile.d/conda.sh
conda activate cv311

# Start API server on port 8081 (background)
API_PID=""
python src/config/api_server.py &
API_PID=$!
echo "API server started (PID $API_PID) on port 8081"

# Cleanup: kill API server on script exit
cleanup() {
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        echo "Shutting down API server (PID $API_PID)..."
        kill "$API_PID" 2>/dev/null
        wait "$API_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# Run main demo
cd /home/baaai/projects/openheart
python scripts/demo_full.py "$@"
