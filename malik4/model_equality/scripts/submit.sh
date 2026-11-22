#!/usr/bin/env bash
# Submit one CSV to the model-equality task and print the score.
#
# Validates first: a malformed file wastes a 30-minute cooldown, and the validator catches the two
# answer shapes that are mathematically guaranteed to score 0.
#
# Usage:
#   ./scripts/submit.sh out/submission_v1.csv
#   ./scripts/submit.sh out/submission_v1.csv --dry-run
#   ./scripts/submit.sh out/ladder/*.csv          # does NOT wait; use --loop for the ladder
#   ./scripts/submit.sh out/ladder/*.csv --loop   # 30 min between each
set -euo pipefail

API_KEY="${HACKATHON_API_KEY:-376ae623f4dfdf4cacd92c9864795292}"
TASK="33-model-equality"
SERVER="http://34.147.229.55"
PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

DRY=0
LOOP=0
FILES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --loop) LOOP=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) FILES+=("$arg") ;;
  esac
done

if [ ${#FILES[@]} -eq 0 ]; then
  echo "usage: $0 <file.csv> [more.csv ...] [--dry-run] [--loop]" >&2
  exit 2
fi

first=1
for file in "${FILES[@]}"; do
  path="$file"
  [ -f "$path" ] || path="$ROOT/$file"
  if [ ! -f "$path" ]; then
    echo "missing: $file" >&2
    exit 1
  fi

  echo "=== $(basename "$path") ==="
  if ! "$PY" "$HERE/validate_submission.py" "$path"; then
    echo "refusing to submit: validation failed (this would waste a cooldown)" >&2
    exit 1
  fi

  if [ "$DRY" -eq 1 ]; then
    echo "DRY RUN - not submitting"
    continue
  fi

  "$PY" - "$path" "$API_KEY" "$TASK" "$SERVER" <<'PY'
import sys
import requests
from pathlib import Path

path, api_key, task, server = sys.argv[1:5]
f = Path(path)
print(f"submitting {f.name}; the server may take up to 500 s to score...")
try:
    with f.open("rb") as fh:
        r = requests.post(f"{server}/submit/{task}",
                          headers={"X-API-Key": api_key},
                          files={"file": (f.name, fh)}, timeout=(600, 600))
except requests.RequestException as exc:
    print(f"network error: {exc}", file=sys.stderr)
    sys.exit(1)
print("HTTP", r.status_code)
try:
    print(r.json())
except ValueError:
    print(r.text[:400])
    sys.exit(1)
PY

  if [ "$LOOP" -eq 1 ] && [ $((first + 1)) -le ${#FILES[@]} ]; then
    echo "waiting 1800 s for the 30-minute cooldown..."
    sleep 1800
  fi
  first=$((first + 1))
done
