#!/bin/bash

ENV_FILE="/home/baaai/projects/openheart/.env"

# Source .env
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            '#'*) continue ;;
            DEEPSEEK_*) export "$key=$value" ;;
        esac
    done < "$ENV_FILE"
fi

# Activate conda
source /home/baaai/miniforge3/etc/profile.d/conda.sh
conda activate cv311

cd /home/baaai/projects/openheart

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

echo "=== Starting REST API ==="
python frontend/server.py > /tmp/openheart_api.log 2>&1 &
sleep 1

echo "=== Starting main backend ==="
# Translate --text-mode to --voice-mode text for demo_full.py
_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--text-mode" ]; then
        _ARGS+=(--voice-mode text)
    else
        _ARGS+=("$arg")
    fi
done
set -- "${_ARGS[@]}"
python scripts/demo_full.py "$@"
