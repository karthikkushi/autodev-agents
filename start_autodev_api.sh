#!/bin/bash
# AutoDev API startup script — launches the server (which starts the pipeline
# worker), or a one-off run of a single design.
# Usage: ./start_autodev_api.sh [--run <design.md>] [--auditor]
# The phone reaches the dashboard over Tailscale (http://<tailscale-ip>:8081).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

AUDITOR=false
DESIGN_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --auditor) AUDITOR=true; shift ;;
        --run) DESIGN_FILE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║       AutoDev API Pipeline v1.0       ║"
echo "╚═══════════════════════════════════════╝"
echo ""

if [ ! -f .env ]; then
    echo "[!] No .env found — copy .env.example to .env and add your provider keys."
    echo "    The server will still start, but pipeline runs will fail until at least"
    echo "    one provider key is set."
fi

MAIN_ARGS="--watch"
$AUDITOR && MAIN_ARGS="$MAIN_ARGS --auditor"

if [ -n "$DESIGN_FILE" ]; then
    if [ ! -f "$DESIGN_FILE" ]; then
        echo "[ERROR] Design file not found: $DESIGN_FILE"
        exit 1
    fi
    MAIN_ARGS="$DESIGN_FILE"
    $AUDITOR && MAIN_ARGS="$MAIN_ARGS --auditor"
fi

echo "[→] Starting AutoDev API (main.py $MAIN_ARGS)..."
python3 main.py $MAIN_ARGS &
PIPELINE_PID=$!
echo "[✓] Pipeline process PID: $PIPELINE_PID"

sleep 2

echo ""
echo "[✓] AutoDev API is running."
echo "    Dashboard:  http://localhost:8081/ui"
echo "    API:        http://localhost:8081"
echo "    Press Ctrl+C to stop."
echo ""

cleanup() {
    echo ""
    echo "[→] Shutting down AutoDev API..."
    [ -n "$PIPELINE_PID" ] && kill "$PIPELINE_PID" 2>/dev/null
    echo "[✓] Done."
}
trap cleanup INT TERM

wait
