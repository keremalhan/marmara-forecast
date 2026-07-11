"""bootstrap.py machinery audit.
(a) resampling is over WHOLE window-ids: stationary_window_indices returns window-level
    indices; on a within-window-perfectly-correlated synthetic, the block CI is much
    WIDER than a naive row-level bootstrap (fake-tight) CI.
(b) pairing: two models with identical RANKING must give ΔPR-AUC == 0 in EVERY resample
    (only possible if both are scored on the same resampled rows).
(c) independent 20-line reference block bootstrap of the y30-test cascade-vs-hybrid
    ΔPR-AUC CI must match bootstrap_ci.json to Monte-Carlo tolerance.
Writes results/verify/bootstrap_audit.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from marmara.paths import RESULTS
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, precompute, bootstrap_split, pair_stats, full_metrics

OUT = RESULTS; VER = OUT / "verify"; VER.mkdir(exist_ok=True)


def audit_a():
    # 20 windows x 50 cells; y,p CONSTANT within a window, varying across windows.
    rng = np.random.default_rng(0)
    nwin, ncell = 20, 50
    rows = []
    for w in range(nwin):
        yv = rng.integers(0, 2)                    # window-level label
        pv = rng.random()                          # window-level score
        rows.append(pd.DataFrame({"window": w, "y": float(yv), "m": pv},
                                 index=range(ncell)))
    df = pd.concat(rows, ignore_index=True)
    # window indices in range + resample length == n_win
    idx = stationary_window_indices(nwin, 200, MEAN_BLOCK, np.random.default_rng(1))
    ok_range = bool(idx.min() >= 0 and idx.max() < nwin and idx.shape == (200, nwin))
    # block bootstrap of Brier vs row-level bootstrap of Brier
    y = df["y"].to_numpy(); p = df["m"].to_numpy(); win = df["window"].to_numpy()
    rb = np.random.default_rng(2)
    block = []
    seqs = stationary_window_indices(nwin, 1000, MEAN_BLOCK, rb)
    per_win_rows = [np.where(win == w)[0] for w in range(nwin)]
    for s in seqs:
        sel = np.concatenate([per_win_rows[w] for w in s])
        block.append(brier_score_loss(y[sel], p[sel]))
    row = []
    n = len(df)
    for _ in range(1000):
        sel = rb.integers(0, n, n)
        row.append(brier_score_loss(y[sel], p[sel]))
    w_block = float(np.percentile(block, 97.5) - np.percentile(block, 2.5))
    w_row = float(np.percentile(row, 97.5) - np.percentile(row, 2.5))
    return {"window_indices_valid": ok_range, "block_CI_width": round(w_block, 4),
            "row_CI_width": round(w_row, 4), "block_wider_than_row": bool(w_block > 3 * w_row),
            "verdict": "PASS: block respects within-window correlation (row-level would be fake-tight)"
            if (ok_range and w_block > 3 * w_row) else "FAIL"}


def audit_b():
    # two models, identical ranking (m2 = m1*0.5 + 0.01, order-preserving) -> paired ΔPR-AUC==0.
    rng = np.random.default_rng(0)
    nwin, ncell = 26, 1219
    rows = []
    for w in range(nwin):
        yv = (rng.random(ncell) < 0.02).astype(float)
        pv = rng.random(ncell)
        rows.append(pd.DataFrame({"window": w, "y": yv, "A": pv, "B": pv * 0.5 + 0.01}))
    df = pd.concat(rows, ignore_index=True)
    bs = bootstrap_split(df, ["A", "B"], 500, np.random.default_rng(42))
    full = full_metrics(df, ["A", "B"])
    st = pair_stats(bs, full, "A", "B")
    dpr = st["d_pr_auc"]
    max_abs = max(abs(dpr["ci95"][0] or 0), abs(dpr["ci95"][1] or 0), abs(dpr["point"] or 0))
    return {"delta_pr_auc_point": dpr["point"], "ci95": dpr["ci95"],
            "max_abs_delta": round(float(max_abs), 8),
            "verdict": "PASS: identical-ranking models give ΔPR-AUC==0 every resample (pairing correct)"
            if max_abs < 1e-9 else "FAIL: pairing broken (Δ != 0 for identical ranking)"}


def audit_c():
    # independent reference block bootstrap of y30-test cascade-vs-hybrid ΔPR-AUC.
    p = pd.read_parquet(OUT / "predictions_y30.parquet")
    p = p[p["split"] == "test"]
    win = p["window"].to_numpy(); y = p["y"].to_numpy()
    wins = np.sort(np.unique(win))
    per = {w: np.where(win == w)[0] for w in wins}
    a = p["cascade"].to_numpy(); h = p["hybrid"].to_numpy()
    rng = np.random.default_rng(42); pblk = 1.0 / MEAN_BLOCK; nw = len(wins)
    diffs = []
    for _ in range(2000):
        seq = []; i = int(rng.integers(nw))
        for _ in range(nw):
            seq.append(i)
            i = int(rng.integers(nw)) if rng.random() < pblk else (i + 1) % nw
        sel = np.concatenate([per[wins[j]] for j in seq])
        ys = y[sel]
        if 0 < ys.sum() < len(ys):
            diffs.append(average_precision_score(ys, a[sel]) - average_precision_score(ys, h[sel]))
    diffs = np.array(diffs)
    ref_ci = [round(float(np.percentile(diffs, 2.5)), 5), round(float(np.percentile(diffs, 97.5)), 5)]
    # committed value
    c = json.load(open(OUT / "bootstrap_ci.json"))
    committed = c["results"]["y30"]["test"]["pairs"]["cascade_vs_hybrid"]["d_pr_auc"]["ci95"] \
        if "cascade_vs_hybrid" in c["results"]["y30"]["test"]["pairs"] else \
        [-x for x in reversed(c["results"]["y30"]["test"]["pairs"]["hybrid_vs_cascade"]["d_pr_auc"]["ci95"])]
    match = all(abs(ref_ci[k] - committed[k]) < 0.02 for k in (0, 1))
    return {"reference_ci95": ref_ci, "committed_ci95": [round(x, 5) for x in committed],
            "match_within_MC_0.02": bool(match),
            "verdict": "PASS: independent reference reproduces the CI to MC tolerance" if match else "FAIL"}


def main():
    out = {"V3a_block_resampling": audit_a(), "V3b_pairing": audit_b(), "V3c_independent_reference": audit_c()}
    json.dump(out, open(VER / "bootstrap_audit.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
