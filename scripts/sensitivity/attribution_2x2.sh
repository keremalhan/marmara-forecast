#!/bin/bash
# 2x2 attribution (reviewer-ordered): what froze Table 2 -- the k-fix or the w-rule?
#
# The w dimension is FREE: one train+bootstrap run emits BOTH `hybrid` (1-SE gate) and
# `hybrid_naive` (naive argmax) columns. So we sweep only the k dimension (surgical n=0.95
# vs old supercritical-k, via V2_SUPERCRITICAL) x b_op {1.10,1.15,1.20}, and read the two
# w-rules off each run. Diagnostic bootstrap (B_DIAG, y30/y35 test only); the canonical
# committed claims (B=2000, all splits) are backed up and restored so this does not disturb
# the canonical state.
#
# Expectation to be TESTED (not assumed): fragility (verdict flips across b) appears in the
# supercritical-k row for BOTH w-rules, and disappears in the surgical-k row -> the bug, not
# the gate, caused the v1 fragility.
#
# Output: results/verify/attribution_2x2.md
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src MARMARA_ROOT="$(pwd)" COPYFILE_DISABLE=1
export TMPDIR="${TMPDIR:-$(pwd)/.tmp}" MPLCONFIGDIR="$TMPDIR/mpl"; mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export V2_BOOT_TARGETS=y30,y35 V2_BOOT_SPLITS=test      # diagnostic: only the Table-2 splits
PY=.venv/bin/python
REPORT=results/etas/etas_fit_report.json
B_DIAG=800
SCRATCH="$(mktemp -d)"
BACKUP="results/.canonical_backup"; mkdir -p "$BACKUP"   # persistent (survives a crash)

setb(){ $PY -c "import json;p='$REPORT';r=json.load(open(p));r['operational_b_for_cascade']=float('$1');json.dump(r,open(p,'w'),indent=2)"; }
corerun(){ $PY -m marmara.grid_hybrid >/dev/null 2>&1; \
           $PY -m marmara.etas_rates results/etas/etas_sv_params.pkl sv_etas >/dev/null 2>&1; \
           $PY -m marmara.train >/dev/null 2>&1; \
           $PY -m marmara.bootstrap "$B_DIAG" >/dev/null 2>&1; }

echo "=== 2x2 attribution sweep $(date) ==="
CANON_FILES="claims.json evaluation.json evaluation.md bootstrap_ci.json bootstrap_ci.md grid_hybrid.parquet grid_hybrid_report.json rates_sv_etas.parquet predictions_y30.parquet predictions_y35.parquet predictions_y45.parquet etas_fit_report.json"
for f in $CANON_FILES; do [ -f "results/$f" ] && cp "results/$f" "$BACKUP/$f"; done
echo "backed up canonical artifacts -> $BACKUP"

for kmode in surgical supercritical; do
  if [ "$kmode" = supercritical ]; then export V2_SUPERCRITICAL=1; else unset V2_SUPERCRITICAL; fi
  for b in 1.10 1.15 1.20; do
    echo ">>> kmode=$kmode b=$b @ $(date +%H:%M:%S)"; setb "$b"; corerun
    cp results/claims.json "$SCRATCH/claims_${kmode}_${b}.json"
  done
done
unset V2_SUPERCRITICAL

# restore canonical committed state (B=2000, surgical, b=1.15)
for f in $CANON_FILES; do [ -f "$BACKUP/$f" ] && cp "$BACKUP/$f" "results/$f"; done
echo ">>> restored canonical artifacts (B=2000 surgical b=1.15)"

$PY - "$SCRATCH" "$B_DIAG" <<'PYEOF'
import json, sys
scratch, bdiag = sys.argv[1], sys.argv[2]
bs = ["1.10","1.15","1.20"]; KM = ["supercritical","surgical"]
PHYS = ["cascade","sv_etas","firstgen_etas"]
HYB = {"1-SE":"hybrid", "naive-argmax":"hybrid_naive"}
def load(km,b):
    cl = json.load(open(f"{scratch}/claims_{km}_{b}.json"))["claims"]
    return {(c["pair"],c["target"]):c["verdict"] for c in cl if c["split"]=="test"}
data = {(km,b): load(km,b) for km in KM for b in bs}
def v(km,b,pair,t): return data[(km,b)].get((pair,t),"-")
def anyflip(km,hyb):
    return any(len({v(km,b,f"{hyb}_vs_{ph}",t) for b in bs})>1 for t in ("y30","y35") for ph in PHYS)
L=["# 2x2 attribution: what froze Table 2 - the k-fix or the w-rule?","",
   "Cell = does ANY hybrid-vs-physics verdict FLIP across b_op {1.10,1.15,1.20} on y30/y35 test?",
   f"(FRAGILE if yes.) Diagnostic bootstrap B={bdiag}; canonical claims are B=2000 (untouched).","",
   "| k-model \\ w-rule | 1-SE | naive-argmax |","|---|---|---|"]
for km in KM:
    L.append(f"| **{km}-k** | "+" | ".join("FRAGILE" if anyflip(km,h) else "stable" for h in HYB.values())+" |")
L+=["","## Detail - hybrid-vs-physics verdicts across b_op (y30/test)",""]
for km in KM:
    for lab,hyb in HYB.items():
        L+=[f"### {km}-k, {lab} (y30/test)","","| pair | 1.10 | 1.15 | 1.20 |","|---|---|---|---|"]
        for ph in PHYS:
            L.append(f"| {hyb} vs {ph} | "+" | ".join(v(km,b,f'{hyb}_vs_{ph}','y30') for b in bs)+" |")
        L.append("")
open("results/verify/attribution_2x2.md","w").write("\n".join(L))
print("=== 2x2 HEADLINE ===")
for km in KM:
    for lab,h in HYB.items():
        print(f"  {km}-k x {lab}: {'FRAGILE' if anyflip(km,h) else 'stable'}")
print("wrote results/verify/attribution_2x2.md")
PYEOF
echo "=== ATTRIBUTION DONE $(date) ==="
