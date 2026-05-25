#!/bin/bash
echo "=== OpenHeart Backend Launcher ==="

SILENT=false
NO_API_CHECK=false
_PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--silent" ]; then
        SILENT=true
    elif [ "$arg" = "--no-api-check" ]; then
        NO_API_CHECK=true
    else
        _PASSTHROUGH_ARGS+=("$arg")
    fi
done
set -- "${_PASSTHROUGH_ARGS[@]}"

# When silent, redirect all terminal output to log file
if $SILENT; then
    exec >> /tmp/openheart_backend.log 2>&1
fi

# Resolve project root: env var overrides, otherwise derive from script location
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

ENV_FILE="$PROJECT_ROOT/.env"

# Source .env
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            '#'*) continue ;;
            DEEPSEEK_*) export "$key=$value" ;;
        esac
    done < "$ENV_FILE"
fi

# [1/5] Conda environment activation
! $SILENT && echo "[1/5] Activating conda environment..."

# Detect conda base, exit with clear error if not found
_CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$_CONDA_BASE" ]; then
    echo "ERROR: Cannot find conda. Is it installed?"
    echo "Please install Miniforge or Anaconda, then create a 'cv311' environment."
    exit 1
fi

source "$_CONDA_BASE/etc/profile.d/conda.sh"
if $SILENT; then
    conda activate cv311 2>/dev/null
else
    conda activate cv311
fi

cd "$PROJECT_ROOT"

# [2/5] DeepSeek API check
if $NO_API_CHECK; then
    ! $SILENT && echo "[2/5] Skipping DeepSeek API check (--no-api-check)"
else
    ! $SILENT && echo "[2/5] Checking DeepSeek API..."
    if $SILENT; then
        python -c "
import os,sys
key = os.environ.get('DEEPSEEK_API_KEY','').strip()
if not key: sys.exit(1)
from openai import OpenAI
client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1')
r = client.chat.completions.create(model='deepseek-v4-flash', messages=[{'role':'user','content':'ping'}], max_tokens=1)
" > /dev/null 2>&1
    else
        python << 'PY_CHECK_API'
import os,sys
key = os.environ.get('DEEPSEEK_API_KEY','').strip()
print(f"Key length: {len(key)}, starts with sk-: {key.startswith('sk-')}")
if not key: sys.exit(1)
from openai import OpenAI
client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1')
r = client.chat.completions.create(model='deepseek-v4-flash', messages=[{'role':'user','content':'ping'}], max_tokens=1)
print('DEEPSEEK API OK')
PY_CHECK_API
    fi
fi

# [3/5] Start REST API server
! $SILENT && echo "[3/5] Starting REST API server on port 8081..."
python frontend/server.py > /tmp/openheart_api.log 2>&1 &
sleep 1

# Poll REST API until ready (max 30s)
! $SILENT && echo -n "[3/5] Waiting for REST API... "
_REST_READY=false
for i in $(seq 1 30); do
    if curl -s --max-time 1 http://localhost:8081/api/status > /dev/null 2>&1; then
        _REST_READY=true
        ! $SILENT && echo "OK"
        break
    fi
    ! $SILENT && echo -n "."
    sleep 1
done
if ! $_REST_READY; then
    ! $SILENT && echo "WARNING: REST API not responding"
fi

# [4/5] Start main AI backend
! $SILENT && echo "[4/5] Starting main AI backend (demo_full.py)..."
_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--text-mode" ]; then
        _ARGS+=(--voice-mode text)
    else
        _ARGS+=("$arg")
    fi
done
set -- "${_ARGS[@]}"
if $SILENT; then
    nohup python scripts/demo_full.py "$@" >> /tmp/openheart.log 2>&1 &
    disown
else
    python scripts/demo_full.py "$@"
fi

# [5/5] Done
! $SILENT && echo "[5/5] Backend launch complete!"
