#!/bin/bash
# tests/run_all_contracts.sh
# Runs all contract tests for the OpenHeart project (spec v4.5.0).
# Every module must pass its corresponding contract test before being considered "done".
# Run before each commit per spec section 13 and 项目宪法 section 7.1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== OpenHeart Contract Tests v4.5.0 ==="
echo "Running all contract tests..."
echo ""

python -m pytest tests/contracts/ -v --tb=short "$@"
