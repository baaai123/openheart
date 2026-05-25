#!/bin/bash

SILENT=false
_PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--silent" ]; then
        SILENT=true
    else
        _PASSTHROUGH_ARGS+=("$arg")
    fi
done
set -- "${_PASSTHROUGH_ARGS[@]}"

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

source /home/baaai/miniforge3/etc/profile.d/conda.sh
if $SILENT; then
    conda activate cv311 2>/dev/null
else
    conda activate cv311
fi

cd "$PROJECT_ROOT"

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
    echo "=== Checking DeepSeek API ==="
    python -c "
import os,sys
key = os.environ.get('DEEPSEEK_API_KEY','').strip()
print(f'Key length: {len(key)}, starts with sk-: {key.startswith(\"sk-\")}')
if not key: sys.exit(1)
from openai import OpenAI
client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1')
r = client.chat.completions.create(model='deepseek-v4-flash', messages=[{'role':'user','content':'ping'}], max_tokens=1)
print('DEEPSEEK API OK')
"
fi

! $SILENT && echo "=== Starting REST API ==="
python frontend/server.py > /tmp/openheart_api.log 2>&1 &
sleep 1

! $SILENT && echo "=== Starting main backend ==="
_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--text-mode" ]; then
        _ARGS+=(--voice-mode text)
    else
        _ARGS+=("$arg")
    fi
done
set -- "${_ARGS[@]}"
python scripts/demo_full.py "$@" >> /tmp/openheart.log 2>&1
