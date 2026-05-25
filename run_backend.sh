#!/bin/bash
set -e

# Source .env if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

# Activate conda
source /home/baaai/miniforge3/etc/profile.d/conda.sh
conda activate cv311

# Check API key FIRST - before loading any models
cd /home/baaai/projects/openheart
echo "=== Checking DeepSeek API ==="
python -c "
import os, sys
key = os.environ.get('DEEPSEEK_API_KEY', '')
if not key:
    print('ERROR: DEEPSEEK_API_KEY not set in .env file')
    print('Set it in .env or the control panel')
    sys.exit(1)
from openai import OpenAI
client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1')
try:
    r = client.chat.completions.create(model='deepseek-v4-flash', messages=[{'role':'user','content':'ping'}], max_tokens=1)
    print('DEEPSEEK API OK')
except Exception as e:
    print(f'DEEPSEEK API FAILED: {e}')
    print('Check your API key in .env or control panel')
    sys.exit(1)
"
echo

# Now start the full backend
python scripts/demo_full.py "$@"
