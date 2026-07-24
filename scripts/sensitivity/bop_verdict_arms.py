"""Run 15 — Table 3 arms at b_op = 1.00 and 1.40 (Amendment 7, SHA-256 8e0ec88d..., hashed first).

WHY. The b_op count-calibration is unidentified: its output tracks era and threshold across
[1.00, 1.54] for reasons orthogonal to the magnitude law (Amendment 7 §1-2). b_op = 1.15 is retained
as the registered operational convention rather than recalibrated. The defence against "your b is
arbitrary" is that the paper's verdicts are paired comparisons at a common b_op and do not move
across the contested span. Table 3 currently spans 1.10-1.20; this extends it to the endpoints that
matter: 1.00 (the Md pile's answer) and 1.40 (the deconfounded magnitude law above the pile).

PREDICTION LODGED BLIND in Amendment 7: every paired hybrid-vs-physics and physics-vs-physics verdict
is unchanged at both arms. If one moves, it is reported and the conclusion revisited. No gate.

NON-DESTRUCTIVE. The b-arm path (grid_hybrid -> etas_rates -> train -> bootstrap) writes in place, so
every frozen artifact is checksummed, backed up, and restored, with the checksums re-verified at the
end. If restoration fails the script says so loudly.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/bop_verdict_arms.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from marmara.paths import RESULTS

R4 = RESULTS / "round4"
BAK = RESULTS.parent / ".tmp" / "r15_backup"
ARMS = [1.00, 1.40]
PY = str(RESULTS.parent / ".venv" / "bin" / "python")

# every artifact the b-arm path rewrites
FROZEN = [
    "etas_fit_report.json", "grid_hybrid.parquet", "grid_hybrid_report.json",
    "rates_sv_etas.parquet", "predictions_y30.parquet", "predictions_y35.parquet",
    "predictions_y45.parquet", "evaluation.json", "evaluation.md",
    "bootstrap_ci.json", "bootstrap_ci.md", "claims.json",
    "models/count30_hybrid.pkl", "models/count35_hybrid.pkl", "models/count45_hybrid.pkl",
    "models/count30_hybrid_gnss.pkl", "models/count35_hybrid_gnss.pkl",
    "models/count45_hybrid_gnss.pkl",
]
PAIRS = ["hybrid_vs_cascade", "hybrid_vs_sv_etas", "hybrid_vs_firstgen_etas",
         "hybrid_vs_modern_etas", "cascade_vs_sv_etas", "cascade_vs_firstgen_etas",
         "cascade_vs_modern_etas", "modern_etas_vs_firstgen_etas", "sv_etas_vs_firstgen_etas"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def run(mod, *extra):
    r = subprocess.run([PY, "-m", mod, *extra], cwd=str(RESULTS.parent),
                       env={"PYTHONPATH": "src", "MARMARA_ROOT": ".", "PATH": "/usr/bin:/bin",
                            "TMPDIR": str(RESULTS.parent / ".tmp"),
                            "MPLCONFIGDIR": str(RESULTS.parent / ".tmp" / "mpl")},
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    !! {mod} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", flush=True)
        raise SystemExit(f"{mod} failed")


def set_bop(b):
    p = RESULTS / "etas" / "etas_fit_report.json"
    d = json.load(open(p))
    d["operational_b_for_cascade"] = float(b)
    json.dump(d, open(p, "w"), indent=2)


def verdicts():
    c = json.load(open(RESULTS / "claims.json"))["claims"]
    out = {}
    for tgt in ("y30", "y35"):
        out[tgt] = {}
        for cl in c:
            if cl["target"] == tgt and cl["split"] == "test" and cl["pair"] in PAIRS:
                out[tgt][cl["pair"]] = {
                    "verdict": cl["verdict"],
                    "ig": cl["ig"]["point"], "ig_ci": cl["ig"]["ci95"],
                    "pr": cl["pr_auc"]["point"], "pr_ci": cl["pr_auc"]["ci95"]}
    return out


def main():
    t0 = time.time()
    BAK.mkdir(parents=True, exist_ok=True)
    print("backing up frozen artifacts ...", flush=True)
    before = {}
    for f in FROZEN:
        src = RESULTS / f
        if not src.exists():
            print(f"  (missing, skipped: {f})"); continue
        before[f] = sha(src)
        dst = BAK / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"  {len(before)} artifacts checksummed and backed up", flush=True)

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_7.md",
                           "sha256": "8e0ec88d9d32893e35d94b809daa39f00cb9335976e51d7a084c604e1246a955"},
           "why": ("b_op is unidentified; 1.15 is retained as convention. These arms show the paired "
                   "verdicts do not move across the contested span [1.00, 1.40]."),
           "prediction_lodged_blind": "every paired verdict unchanged at both arms",
           "shipped_b_op": 1.15, "arms": {}}
    out["arms"]["1.15_shipped"] = verdicts()

    try:
        for b in ARMS:
            print(f"\n=== arm b_op = {b} ===", flush=True)
            set_bop(b)
            for step in ("marmara.grid_hybrid",):
                print(f"  {step} ...", flush=True); run(step)
            print("  marmara.etas_rates (sv_etas) ...", flush=True)
            run("marmara.etas_rates", "results/etas/etas_sv_params.pkl", "sv_etas")
            print("  marmara.etas_rates (modern_etas) ...", flush=True)
            try:
                run("marmara.etas_rates", "results/etas/modern_params_cascade.pkl", "modern_etas")
            except SystemExit:
                print("    (modern_etas rates unavailable at this arm; disclosed)", flush=True)
            print("  marmara.train ...", flush=True); run("marmara.train")
            print("  marmara.bootstrap ...", flush=True); run("marmara.bootstrap")
            out["arms"][f"{b}"] = verdicts()
            print(f"  arm done ({time.time()-t0:.0f}s)", flush=True)
    finally:
        print("\nrestoring frozen artifacts ...", flush=True)
        for f in before:
            shutil.copy2(BAK / f, RESULTS / f)
        after = {f: sha(RESULTS / f) for f in before}
        bad = [f for f in before if before[f] != after[f]]
        out["restore_verified"] = (len(bad) == 0)
        out["restore_mismatches"] = bad
        print(f"  restore verified: {len(bad) == 0}" + (f"  MISMATCHES: {bad}" if bad else ""),
              flush=True)

    # compare
    diffs = []
    base = out["arms"]["1.15_shipped"]
    for b in ARMS:
        a = out["arms"].get(f"{b}", {})
        for tgt in base:
            for pair in base[tgt]:
                v0 = base[tgt][pair]["verdict"]
                v1 = a.get(tgt, {}).get(pair, {}).get("verdict")
                if v1 is not None and v1 != v0:
                    diffs.append(f"b={b} {tgt}/{pair}: {v0} -> {v1}")
    out["verdict_diffs_vs_shipped"] = diffs
    out["prediction_held"] = (len(diffs) == 0)
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r15_table3_arms.json", "w"), indent=2)

    print("\n=== VERDICTS ACROSS THE CONTESTED SPAN ===")
    for tgt in base:
        print(f"\n  {tgt}/test")
        for pair in base[tgt]:
            row = [out["arms"].get(k, {}).get(tgt, {}).get(pair, {}).get("verdict", "-")
                   for k in ("1.00", "1.15_shipped", "1.40")]
            flag = "" if len(set(x for x in row if x != "-")) == 1 else "   <-- MOVED"
            print(f"    {pair:30s} b=1.00 {row[0]:16s} b=1.15 {row[1]:16s} b=1.40 {row[2]:16s}{flag}")
    print(f"\nverdict diffs vs shipped: {diffs if diffs else 'NONE'}")
    print(f"blind prediction held: {out['prediction_held']}")
    print(f"restore verified: {out['restore_verified']}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
