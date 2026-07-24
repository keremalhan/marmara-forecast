"""T0-4: block-length sensitivity of the paired block bootstrap.

The paper justifies block (vs row) resampling but asserts mean block 3 without showing the
verdicts are insensitive to that choice. Re-run the two-axis verdict for the primary y30 pairs
at mean block in {2,3,5,8}. Writes results/round3/t0_block_sensitivity.json.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda
from marmara.bootstrap import stationary_window_indices, SEED

EPS = 1e-9
B = 2000
PAIRS = [("hybrid", "cascade"), ("hybrid", "sv_etas"), ("hybrid", "firstgen_etas"),
         ("firstgen_etas", "modern_etas")]
BLOCKS = [2.0, 3.0, 5.0, 8.0]


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def verdict(win, y, pa, pb, block):
    la, lb = lam(pa), lam(pb)
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    ca = np.array([np.sum(y[idx[w]] * np.log(la[idx[w]]) - la[idx[w]]) for w in wins])
    cb = np.array([np.sum(y[idx[w]] * np.log(lb[idx[w]]) - lb[idx[w]]) for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    yb = [y[idx[w]] for w in wins]; ab = [pa[idx[w]] for w in wins]; bb = [pb[idx[w]] for w in wins]
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), B, block, rng)
    ig, dpr = [], []
    for r in seqs:
        ig.append((ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1))
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            dpr.append(average_precision_score(yy, np.concatenate([ab[i] for i in r]))
                       - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    ig_ci = [float(np.percentile(ig, 2.5)), float(np.percentile(ig, 97.5))]
    dpr_ci = [float(np.percentile(dpr, 2.5)), float(np.percentile(dpr, 97.5))]
    sep = (ig_ci[0] > 0 or ig_ci[1] < 0) and (dpr_ci[0] > 0 or dpr_ci[1] < 0)
    return {"ig_ci": [round(x, 3) for x in ig_ci], "dpr_ci": [round(x, 4) for x in dpr_ci],
            "verdict": "separable" if sep else "inseparable"}


def main():
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    te = df[df.split == "test"]
    y = te.y.to_numpy(); win = te.window.to_numpy()
    out = {}
    for a, b in PAIRS:
        out[f"{a}_vs_{b}"] = {f"block_{bl:g}": verdict(win, y, te[a].to_numpy(), te[b].to_numpy(), bl)
                              for bl in BLOCKS}
    (RESULTS / "round3" / "t0_block_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({p: {bl: v["verdict"] for bl, v in d.items()} for p, d in out.items()}, indent=1))


if __name__ == "__main__":
    main()
