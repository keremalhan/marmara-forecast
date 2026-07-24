#!/usr/bin/env bash
# Operational-b sensitivity of the ETASxML hybrid vs the physics forecasters.
# Re-runs the pipeline (grid_hybrid -> sv-ETAS rates -> train -> bootstrap) at a
# sweep of operational b, records the hybrid-vs-physics and physics-vs-physics
# verdicts from claims.json, and writes results/b_sensitivity.{json,md}. Restores
# the operational b (1.15) and its committed artifacts at the end.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src COPYFILE_DISABLE=1
PY=.venv/bin/python
REPORT=results/etas/etas_fit_report.json
BOP=1.15
SCRATCH="$(mktemp -d)"
setb(){ $PY -c "import json;p='$REPORT';r=json.load(open(p));r['operational_b_for_cascade']=float('$1');json.dump(r,open(p,'w'),indent=2)"; }
corerun(){ $PY -m marmara.grid_hybrid >/dev/null 2>&1; $PY -m marmara.etas_rates results/etas/etas_sv_params.pkl sv_etas >/dev/null 2>&1; $PY -m marmara.train >/dev/null 2>&1; $PY -m marmara.bootstrap >/dev/null 2>&1; }

echo "=== b-sensitivity sweep $(date) ==="
# current committed state is b=1.15
cp results/claims.json "$SCRATCH/claims_1.15.json"
echo "recorded 1.15 (committed)"
for b in 1.10 1.20; do
  echo ">>> sweeping b=$b @ $(date +%H:%M:%S)"; setb "$b"; corerun
  cp results/claims.json "$SCRATCH/claims_$b.json"
done
echo ">>> restoring operational b=$BOP @ $(date +%H:%M:%S)"; setb "$BOP"; corerun
$PY -m marmara.csep_eval >/dev/null 2>&1; $PY scripts/csep/csep_prep.py >/dev/null 2>&1

# assemble the table
$PY - "$SCRATCH" <<'PYEOF'
import json, sys
scratch = sys.argv[1]
bs = ["1.10", "1.15", "1.20"]
PAIRS_HP = ["hybrid_vs_cascade","hybrid_vs_sv_etas","hybrid_vs_modern_etas","hybrid_vs_firstgen_etas"]
PAIRS_PP = ["cascade_vs_sv_etas","cascade_vs_firstgen_etas","modern_etas_vs_firstgen_etas"]
def load(b):
    cl = json.load(open(f"{scratch}/claims_{b}.json"))["claims"]
    return {(c["pair"],c["target"]):c for c in cl if c["split"]=="test"}
data = {b: load(b) for b in bs}
def verdict(b,pair,t):
    c = data[b].get((pair,t));  return c["verdict"] if c else "-"
out = {"operational_b_sweep": bs, "honest_b": 1.15,
       "hybrid_vs_physics": {}, "physics_vs_physics": {}}
for t in ("y30","y35"):
    out["hybrid_vs_physics"][t] = {p:{b:verdict(b,p,t) for b in bs} for p in PAIRS_HP}
    out["physics_vs_physics"][t] = {p:{b:verdict(b,p,t) for b in bs} for p in PAIRS_PP}
json.dump(out, open("results/scoring/b_sensitivity.json","w"), indent=2)
def _flips(pairs):
    return any(len({verdict(b,p,t) for b in bs}) > 1 for t in ("y30","y35") for p in pairs)
hp_flips = _flips(PAIRS_HP)   # DYNAMIC: never let a stale hardcoded verdict typeset over its own table
_title = ("the ML hybrid's verdicts are STABLE across b (v1 instability was a supercriticality artifact)"
          if not hp_flips else "the ML hybrid is unstable, the physics is not")
_hp_head = ("## Hybrid vs physics (verdicts STABLE across b)" if not hp_flips
            else "## Hybrid vs physics (verdict FLIPS with b)")
L = [f"# Operational-b sensitivity: {_title}","",
     "Verdicts (test split) from the pre-specified block-bootstrap rule at three operational b.",
     "b=1.15 is the operational convention (paper §3; the 1.013 slope belongs to the superseded simulator, Section S10); 1.10/1.20 bracket it.","",
     _hp_head,""]
for t in ("y30","y35"):
    L += [f"### {t}/test","","| pair | b=1.10 | b=1.15 | b=1.20 |","|---|---|---|---|"]
    for p in PAIRS_HP:
        L.append("| "+p.replace("_vs_"," vs ")+" | "+" | ".join(verdict(b,p,t) for b in bs)+" |")
    L.append("")
L += ["## Physics vs physics (STABLE across b)",""]
for t in ("y30","y35"):
    L += [f"### {t}/test","","| pair | b=1.10 | b=1.15 | b=1.20 |","|---|---|---|---|"]
    for p in PAIRS_PP:
        L.append("| "+p.replace("_vs_"," vs ")+" | "+" | ".join(verdict(b,p,t) for b in bs)+" |")
    L.append("")
open("results/scoring/b_sensitivity.md","w").write("\n".join(L))
print("wrote results/b_sensitivity.{json,md}")
PYEOF
echo "=== BSENS DONE $(date) ==="
