"""Item B(2): Mizrahi 2x2 ablation {uniform bg, native bg} x {native b, b_op}.

For each config, regenerate the modern_etas (Mizrahi first-gen) forecast, swap it into
the canonical predictions, and block-bootstrap (same B/seed/mean_block/rule as claims.json)
the firstgen-vs-Mizrahi, cascade-vs-Mizrahi and hybrid-vs-Mizrahi verdicts on y30/y35 test.

STEEL-MAN rule: the abstract's separation claim (first-gen beats Mizrahi) may only claim
what holds in Mizrahi's MOST FAVORABLE configuration (native background + common b_op).
Also re-checks the lone moving Table-2 cell (hybrid beats Mizrahi at b_op).

Canonical rates_modern_etas.parquet is backed up + restored. Writes results/mizrahi_2x2.{json,md}.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.metrics import lambda_to_p, information_gain
from marmara import bootstrap as BS

OUT = RESULTS
ROOT = OUT.parent
PY = str(ROOT / ".venv" / "bin" / "python")
CONFIGS = [("uniform", "native"), ("uniform", "bop"), ("native", "native"), ("native", "bop")]
MODELS = ["modern_etas", "firstgen_etas", "cascade", "hybrid"]
PAIRS = [("firstgen_etas", "modern_etas"), ("cascade", "modern_etas"), ("hybrid", "modern_etas")]
B = 2000
MOST_FAVORABLE = "bg=native|b=bop"           # spatially-variable bg + harmonized magnitude law


def modern_pred_at_grid(grid, thr):
    lamcol = {3.0: "lam30", 3.5: "lam35"}[round(thr, 1)]
    r = (pd.read_parquet(OUT / "grid" / "rates_modern_etas.parquet", columns=["window", "ir", "ic", lamcol])
         .set_index(["window", "ir", "ic"])[lamcol])
    idx = pd.MultiIndex.from_frame(grid[["window", "ir", "ic"]])
    lam = r.reindex(idx).to_numpy(); lam = np.where(np.isfinite(lam), lam, 0.0)
    return lambda_to_p(lam)


def verdict_pairs(df):
    y = df["y"].to_numpy(float)
    rng = np.random.default_rng(BS.SEED)
    full = BS.full_metrics(df, MODELS)
    bs = BS.bootstrap_split(df, MODELS, B, rng)
    out = {}
    for a, b_ in PAIRS:
        st = BS.pair_stats(bs, full, a, b_)
        st["d_ig"]["point"] = round(float(information_gain(df[a].to_numpy(float), df[b_].to_numpy(float), y)), 4)
        out[f"{a}_vs_modern_etas"] = {"verdict": BS.verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"]),
                                      "ig_ci": st["d_ig"]["ci95"], "pr_ci": st["d_pr_auc"]["ci95"]}
    return out


def main():
    t0 = time.time()
    grid = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid); sel = m["val"] | m["test"]
    bak = OUT / ".canonical_backup"; bak.mkdir(exist_ok=True)
    shutil.copy(OUT / "grid" / "rates_modern_etas.parquet", bak / "grid" / "rates_modern_etas.parquet")
    preds = {t: pd.read_parquet(OUT / f"predictions_{t}.parquet") for t in ("y30", "y35")}

    results = {}
    for bg, bm in CONFIGS:
        env = dict(os.environ, V2_MIZ_BG=bg, V2_MIZ_B=bm, PYTHONPATH="src", MARMARA_ROOT=str(ROOT))
        rc = subprocess.run([PY, "-m", "marmara.etas_modern"], env=env, cwd=str(ROOT),
                            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"config {bg}/{bm} FAILED: {rc.stderr[-400:]}"); continue
        cfg = {}
        for target, thr in (("y30", 3.0), ("y35", 3.5)):
            pred = preds[target].copy()
            pred["modern_etas"] = modern_pred_at_grid(grid, thr)[sel]
            cfg[target] = verdict_pairs(pred[pred["split"] == "test"].reset_index(drop=True))
        results[f"bg={bg}|b={bm}"] = cfg
        print(f"  [{bg}/{bm}] y30 firstgen_vs_modern={cfg['y30']['firstgen_etas_vs_modern_etas']['verdict']} "
              f"y35={cfg['y35']['firstgen_etas_vs_modern_etas']['verdict']} "
              f"hybrid_vs_modern(y30)={cfg['y30']['hybrid_vs_modern_etas']['verdict']} ({time.time()-t0:.0f}s)",
              flush=True)

    shutil.copy(bak / "grid" / "rates_modern_etas.parquet", OUT / "grid" / "rates_modern_etas.parquet")   # restore canonical
    out = {"meta": {"B": B, "seed": BS.SEED, "most_favorable_for_mizrahi": MOST_FAVORABLE,
                    "runtime_s": round(time.time() - t0, 1)}, "configs": results}
    json.dump(out, open(OUT / "mizrahi_2x2.json", "w"), indent=2)
    _write_md(out)
    print("wrote results/mizrahi_2x2.{json,md}")


def _write_md(r):
    cfgs = list(r["configs"])
    L = ["# Item B(2): Mizrahi 2x2 ablation {uniform,native bg} x {native b,b_op}", "",
         f"Block-bootstrap (B={r['meta']['B']}, seed {r['meta']['seed']}, same rule as claims.json). "
         f"Steel-man: the separation may claim only what holds in Mizrahi's MOST FAVORABLE config "
         f"(**{r['meta']['most_favorable_for_mizrahi']}**).", "",
         "## first-gen ETAS vs Mizrahi (the abstract's separation)", "",
         "| config | y30/test | y35/test |", "|---|---|---|"]
    for c in cfgs:
        v30 = r["configs"][c]["y30"]["firstgen_etas_vs_modern_etas"]["verdict"]
        v35 = r["configs"][c]["y35"]["firstgen_etas_vs_modern_etas"]["verdict"]
        star = " **(most favorable)**" if c == r["meta"]["most_favorable_for_mizrahi"] else ""
        L.append(f"| {c}{star} | {v30} | {v35} |")
    L += ["", "## hybrid vs Mizrahi (the lone moving Table-2 cell) & cascade vs Mizrahi", "",
          "| config | hybrid_vs_modern y30 | cascade_vs_modern y30 |", "|---|---|---|"]
    for c in cfgs:
        L.append(f"| {c} | {r['configs'][c]['y30']['hybrid_vs_modern_etas']['verdict']} | "
                 f"{r['configs'][c]['y30']['cascade_vs_modern_etas']['verdict']} |")
    mf = r["configs"][r["meta"]["most_favorable_for_mizrahi"]]
    surv = mf["y30"]["firstgen_etas_vs_modern_etas"]["verdict"]
    L += ["", "## Verdict",
          f"- In Mizrahi's most favorable config ({r['meta']['most_favorable_for_mizrahi']}), "
          f"first-gen vs Mizrahi (y30) = **{surv}**.",
          "- If this is B_beats_A the separation survives ARMORED (attributable to genuine model "
          "difference, not our integration handicap); if inseparable, the abstract claim must soften "
          "to 'four physics forecasters form a single inseparable cluster'.",
          "", "**Per governance: this is an abstract-level claim. Flagging; NOT rewriting the abstract.**"]
    (OUT / "mizrahi_2x2.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
