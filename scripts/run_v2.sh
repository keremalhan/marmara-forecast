#!/bin/bash
# v2 upgrade pipeline — the rerun order for the Phase 0-5 additions.
# Assumes the BASE pipeline (run_all.sh) has produced catalog.csv, grid.parquet,
# grid_hybrid.parquet, etas_params.pkl and passed its four gates. This script layers
# the v2 steps on top and re-runs the evaluation/bootstrap/CSEP/reports.
#
# Envs:
#   .venv       pinned core (requirements.txt) — everything except the Mizrahi fit
#   .venv-etas  isolated (requirements-etas.txt) — lmizrahi/etas inversion only
#   .venv-csep  isolated (requirements-csep.txt) — pyCSEP (optional; see requirements-csep.txt)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MARMARA_ROOT="$ROOT" PYTHONPATH="$ROOT/src"
export TMPDIR="${TMPDIR:-$ROOT/.tmp}" MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
PY="$ROOT/.venv/bin/python"; PY_ETAS="$ROOT/.venv-etas/bin/python"
run(){ echo; echo ">>> $*"; time "$@"; }

echo "### Phase 1.2 — sv-ETAS (converged EM declustering)"
run "$PY" -m marmara.etas_fit_sv
echo "### Phase 1.1 — independent Mizrahi inversion (isolated .venv-etas; skipped if absent)"
if [ -x "$PY_ETAS" ]; then run "$PY_ETAS" -m marmara.etas_mizrahi_invert; else echo "  (.venv-etas absent — keeping committed results/etas_mizrahi_fit.json)"; fi
echo "### Phase 1 — modern-ETAS gridded rates: sv cascade + Mizrahi first-gen intensity"
run "$PY" -m marmara.etas_rates results/etas_sv_params.pkl sv_etas
run "$PY" -m marmara.etas_modern

echo "### Phase 2 — GNSS v2 fetch (network) + source IG gate"
"$PY" -m marmara.sources.fetch_data gnss_v2 || echo "  (fetch skipped/offline — source is honest-absent)"
run "$PY" -m marmara.source_ig_test gnss_v2 || true    # gate is informational; decision in gnss_v2_decision.json

echo "### Phase 3 — grid_hybrid rebuild (adds y30/count30/lam30_sim), then train y30/y35/y45"
run "$PY" -m marmara.grid_hybrid
echo "### GATE: grid leakage self-test"; run "$PY" -m marmara.tests.test_grid_leakage
run "$PY" -m marmara.train                              # -> evaluation.{json,md}, predictions_*, hybrid_gnss

echo "### Phase 0 — block bootstrap + machine-derived claims"
run "$PY" -m marmara.bootstrap                          # -> bootstrap_ci.{json,md}, claims.json

echo "### Phase 4 — CSEP catalog-based consistency tests"
run "$PY" -m marmara.csep_eval                          # -> results/csep/

echo "### GATE: full test suite (incl. gnss_v2 truncated self-test)"
run "$PY" -m pytest src/marmara/tests/ -x -q -p no:cacheprovider

echo; echo "v2 DONE — see results/{claims.json, bootstrap_ci.md, evaluation.md, gnss_v2_decision.json, csep/, transfer_naf.md, PAPER_DELTAS.md}"
