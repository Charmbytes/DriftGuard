#!/usr/bin/env bash
# Start the API privately, wait until it is healthy, then serve the dashboard
# on the Space's public port.
set -e

uvicorn driftguard.api:app --host 127.0.0.1 --port 8000 &

echo "Waiting for the DriftGuard API to certify its baseline..."
for _ in $(seq 1 180); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" 2>/dev/null; then
        echo "API is up."
        break
    fi
    sleep 1
done

exec streamlit run dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
