"""V1 — reproduce-from-clean diff. Compare every DETERMINISTIC artifact regenerated
by the clean-FS (APFS) run against the committed versions at 1e-9. Writes
results/verify/REPRO_FAILURES.md (empty body => all reproduced).

source_ig_gnss_v2.json is EXCLUDED (its gate recomputes gnss_v2 from a fresh NGL
fetch, which is network/date-dependent, not a deterministic artifact).
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SSD = Path("/Volumes/Kerem SSD/Desktop/marmara-forecast/results")
CLEAN = Path("/Users/keremalhan/marmara-verify/results")
VER = SSD / "verify"; VER.mkdir(exist_ok=True)
TOL = 1e-9

JSON_ARTIFACTS = ["evaluation.json", "bootstrap_ci.json", "claims.json",
                  "etas_sv_fit_report.json", "grid_hybrid_report.json",
                  "csep/csep_results.json"]
PARQUET_ARTIFACTS = ["grid_hybrid.parquet", "predictions_y30.parquet",
                     "predictions_y35.parquet", "predictions_y45.parquet",
                     "rates_sv_etas.parquet", "rates_modern_etas.parquet"]


IGNORE_KEYS = {"runtime_s"}   # wall-clock timing metadata, not a scientific result


def jdiff(a, b, path=""):
    d = []
    if path.split(".")[-1] in IGNORE_KEYS:
        return d
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                d.append(f"{path}.{k}: present in only one")
            else:
                d += jdiff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            d.append(f"{path}: len {len(a)} vs {len(b)}")
        else:
            for i, (u, v) in enumerate(zip(a, b)):
                d += jdiff(u, v, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(float(a) - float(b)) > TOL * max(1.0, abs(float(a)), abs(float(b))):
            d.append(f"{path}: {a} vs {b} (|Δ|={abs(float(a)-float(b)):.3e})")
    else:
        if a != b:
            d.append(f"{path}: {a!r} vs {b!r}")
    return d


def main():
    results = []
    for rel in JSON_ARTIFACTS:
        sp, cp = SSD / rel, CLEAN / rel
        if not cp.exists():
            results.append((rel, "MISSING_IN_CLEAN", 0)); continue
        diffs = jdiff(json.load(open(sp)), json.load(open(cp)))
        results.append((rel, "OK" if not diffs else f"DIFF ({len(diffs)})", diffs[:8]))
    for rel in PARQUET_ARTIFACTS:
        sp, cp = SSD / rel, CLEAN / rel
        if not cp.exists():
            results.append((rel, "MISSING_IN_CLEAN", 0)); continue
        A, Bc = pd.read_parquet(sp), pd.read_parquet(cp)
        if len(A) != len(Bc) or list(A.columns) != list(Bc.columns):
            results.append((rel, f"SHAPE {A.shape} vs {Bc.shape}", 0)); continue
        maxdev = 0.0; bad = []
        for c in A.columns:
            if np.issubdtype(A[c].dtype, np.number):
                dev = float(np.nanmax(np.abs(A[c].to_numpy() - Bc[c].to_numpy())))
                maxdev = max(maxdev, dev)
                if dev > TOL:
                    bad.append(f"{c}:{dev:.3e}")
        results.append((rel, "OK" if not bad else f"DIFF maxdev {maxdev:.3e}", bad[:8]))

    failures = [(r, s, d) for r, s, d in results if s not in ("OK",)]
    L = ["# V1 — reproduce-from-clean (APFS) vs committed, tolerance 1e-9", ""]
    L.append(f"Compared {len(results)} deterministic artifacts; **{len(failures)} did not reproduce**.")
    L.append("(source_ig_gnss_v2.json excluded — network/date-dependent gate, not deterministic.)")
    L.append("")
    L.append("| artifact | status |")
    L.append("|---|---|")
    for r, s, _ in results:
        L.append(f"| {r} | {s} |")
    if failures:
        L.append("\n## Details of non-reproducing artifacts (treat corresponding claims VOID until fixed)")
        for r, s, d in failures:
            L.append(f"\n### {r} — {s}")
            for x in (d if isinstance(d, list) else []):
                L.append(f"  - {x}")
    else:
        L.append("\n**All deterministic artifacts reproduced bit-for-bit / within 1e-9. "
                 "No claim is VOID on reproducibility grounds.**")
    (VER / "REPRO_FAILURES.md").write_text("\n".join(L))
    print("\n".join(f"  {r:34s} {s}" for r, s, _ in results))
    print(f"\nV1: {len(failures)} failures -> results/verify/REPRO_FAILURES.md")
    sys.exit(0)


if __name__ == "__main__":
    main()
