#!/bin/bash
# Monthly prospective forecast runner (schedule via cron/launchd). Repo-relative.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${MARMARA_PY:-python3}"
export MARMARA_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src"
export TMPDIR="${TMPDIR:-$ROOT/.tmp}"; export MPLCONFIGDIR="$TMPDIR/mpl"
LOG="$ROOT/results/prospective/run.log"
mkdir -p "$TMPDIR" "$ROOT/results/prospective"
echo "===== $(date -u +%FT%TZ) monthly prospective run =====" >> "$LOG"
cd "$ROOT" && "$PY" -m marmara.prospective run >> "$LOG" 2>&1
echo "----- exit $? -----" >> "$LOG"
