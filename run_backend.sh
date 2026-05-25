#!/bin/bash
# set -e  # v5.x: disabled - PaddleX warnings trigger exit

# Source .env if it exists (loop handles spaces in values)
if [ -f "$(dirname "$0")/.env" ]; then
    while IFS= read -r line; do
        case "$line" in
            '#'*) continue ;;
            *=*) export "$line" ;;
        esac
    done < "$(dirname "$0")/.env"
fi

# Activate conda
source /home/baaai/miniforge3/etc/profile.d/conda.sh
conda activate cv311

cd /home/baaai/projects/openheart

# Check API key FIRST
echo "=== Checking DeepSeek API ==="
python -c "
import os, sys
key = os.environ.get('DEEPSEEK_API_KEY', '')
if not key or not key.startswith('sk-'):
    print('ERROR: DEEPSEEK_API_KEY not set or invalid')
    sys.exit(1)
from openai import OpenAI
client = OpenAI(api_key=key.strip(), base_url='https://api.deepseek.com/v1')
r = client.chat.completions.create(model='deepseek-v4-flash', messages=[{'role':'user','content':'ping'}], max_tokens=1)
print('DEEPSEEK API OK')
"

# Start REST API in background
echo "=== Starting REST API on port 8081 ==="
python frontend/server.py > /tmp/openheart_api.log 2>&1 &
API_PID=$!
sleep 1

# Start main backend
echo "=== Starting main backend ==="
python scripts/demo_full.py "$@"

# Cleanup
kill $API_PID 2>/dev/null
