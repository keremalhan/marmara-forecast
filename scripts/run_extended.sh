#!/bin/bash
# Extended-analysis pipeline: rerun order for the modern-baseline, GNSS, bootstrap and CSEP steps.
# Assumes the BASE pipeline (run_all.sh) has produced catalog.csv, grid.parquet,
# grid_hybrid.parquet, etas_params.pkl and passed its four gates. This script layers
# these steps on top and re-runs the evaluation/bootstrap/CSEP/reports.
#
# Envs:
#   .venv       pinned core (requirements.txt), everything except the Mizrahi fit
#   .venv-etas  isolated (requirements-etas.txt): lmizrahi/etas inversion only
#   (optional)  a separate environment for the pyCSEP toolkit (requirements-csep.txt),
#               used by scripts/csep/csep_run.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MARMARA_ROOT="$ROOT" PYTHONPATH="$ROOT/src"
export TMPDIR="${TMPDIR:-$ROOT/.tmp}" MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
PY="$ROOT/.venv/bin/python"; PY_ETAS="$ROOT/.venv-etas/bin/python"
run(){ echo; echo ">>> $*"; time "$@"; }

echo "### sv-ETAS (converged EM declustering)"
run "$PY" -m marmara.etas_fit_sv
echo "### independent Mizrahi inversion (isolated .venv-etas; skipped if absent)"
if [ -x "$PY_ETAS" ]; then run "$PY_ETAS" -m marmara.etas_mizrahi_invert; else echo "  (.venv-etas absent, keeping committed results/etas/etas_mizrahi_fit.json)"; fi
echo "### modern-ETAS gridded rates: sv cascade + Mizrahi first-gen intensity"
run "$PY" -m marmara.etas_rates results/etas/etas_sv_params.pkl sv_etas
run "$PY" -m marmara.etas_modern

echo "### GNSS trajectory fetch (network) + source IG gate"
"$PY" -m marmara.sources.fetch_data gnss_traj || echo "  (fetch skipped/offline; source is declared-absent)"
run "$PY" -m marmara.source_ig_test gnss_traj || true    # gate is informational; decision in gnss_traj_decision.json

echo "### grid_hybrid rebuild (adds y30/count30/lam30_sim), then train y30/y35/y45"
run "$PY" -m marmara.grid_hybrid
echo "### GATE: grid leakage self-test"; run "$PY" -m marmara.tests.test_grid_leakage
run "$PY" -m marmara.train                              # -> evaluation.{json,md}, predictions_*, hybrid_gnss

echo "### block bootstrap + machine-derived claims"
run "$PY" -m marmara.bootstrap                          # -> bootstrap_ci.{json,md}, claims.json

echo "### CSEP catalog-based consistency tests"
run "$PY" -m marmara.csep_eval                          # -> results/csep/

echo "### GATE: full test suite (incl. gnss_traj truncated self-test)"
run "$PY" -m pytest src/marmara/tests/ -x -q -p no:cacheprovider

echo; echo "DONE. See results/{claims.json, bootstrap_ci.md, evaluation.md, gnss_traj_decision.json, csep/, transfer_naf.md}"
