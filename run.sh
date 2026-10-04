#!/usr/bin/env bash
# Start ScanSpeak locally: http://127.0.0.1:8417
# Needs: Ollama running with a tool-calling model (llama3.1:8b by default; it scored best in the benchmark),
# python deps from requirements.txt, and the TumSeg scans at ./data/TumSeg database
# (or TUMSEG_DIR=/path/to/"TumSeg database").
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8417}"
MODEL="${SCANSPEAK_MODEL:-llama3.1:8b}"
export TUMSEG_DIR="${TUMSEG_DIR:-$PWD/data/TumSeg database}"
[ -d "$TUMSEG_DIR" ] || { echo "Scans not found at $TUMSEG_DIR. Run: bash ../fauxgraft/scripts/get_data.sh data (or set TUMSEG_DIR)"; exit 1; }
curl -s localhost:11434/api/tags >/dev/null || { echo "Ollama isn't running. Open the Ollama app first."; exit 1; }
curl -s localhost:11434/api/tags | grep -q "\"$MODEL\"" || ollama pull "$MODEL"
[ -f scanspeak/data/catalog.json ] || python3 scripts/build_catalog.py "$TUMSEG_DIR"
echo "ScanSpeak: open http://127.0.0.1:$PORT  (use 127.0.0.1, not localhost)"
exec python3 -m scanspeak.server --port "$PORT"
