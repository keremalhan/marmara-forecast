#!/bin/bash
# Full reproduction of the marmara-forecast pipeline and its four gates.
# Usage:  ./run_all.sh            (uses ./.venv/bin/python if present, else python3)
# Gates (stress, ETAS, cascade, grid-leakage) are HARD STOPS — the script aborts if any fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${MARMARA_PY:-$([ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3)}"
export MARMARA_ROOT="$ROOT" PYTHONPATH="$ROOT/src"
export TMPDIR="${TMPDIR:-$ROOT/.tmp}" MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
run(){ echo; echo ">>> $* ($1 approx)"; shift; time "$PY" "$@"; }

echo "### 1. Catalog (~15 s)";            "$PY" -m marmara.catalog
echo "### GATE 1/4: stress tests (~2 s)"; "$PY" -m marmara.tests.test_stress
echo "### GATE 2/4: ETAS tests (~90 s)";  "$PY" -m marmara.tests.test_etas
echo "### 2. ETAS fit (~4 min)";          "$PY" -m marmara.etas_fit
echo "### 3. Calibrate cascade b (~2 min)"; "$PY" -m marmara.calibrate_b
echo "### 4. Feature grid (~2 min)";      "$PY" -m marmara.grid
echo "### GATE 3/4: grid leakage self-test (~30 s)"; "$PY" -m marmara.tests.test_grid_leakage
echo "### 5. Hybrid grid w/ cascade (~2 min)"; "$PY" -m marmara.grid_hybrid
echo "### GATE 4/4: cascade tests (~5 s)"; "$PY" -m marmara.tests.test_cascade
echo "### 6. Train hybrid + evaluate (~3 min)"; "$PY" -m marmara.train
echo "### 7. Wide-box y45 (~6 min)";      "$PY" -m marmara.widebox_y45
echo "### 8. Renewal priors (~2 s)";      "$PY" -m marmara.renewal
echo "### 9. Synthetic + conditional classifier (~30 min)"; "$PY" -m marmara.synthetic
echo "### 10. Live forecast maps (~1 min)"; "$PY" -m marmara.forecast
echo "### 11. Quarter/year forecast + backtest (~3 min)"; "$PY" -m marmara.forecast_horizons
echo "### 12. Final validation battery (~4 min)"; "$PY" -m marmara.validation_final
echo "### 13. M6.2 countdown study (~3 min)"; "$PY" -m marmara.m62_countdown
echo "### 14. Sequence-mode demo (~1 min)"; "$PY" -m marmara.sequence_mode
echo "### 15. Prospective forecast issue/score (~1 min)"; "$PY" -m marmara.prospective run || true
echo; echo "ALL DONE — all four gates passed. Artifacts in results/."
