#!/usr/bin/env python3
"""fig2_discriminator_reliability.png: reliability diagram of the bigger-event-ahead
discriminator on the SIMULATION-DISJOINT test split.

Rebuilds the synthetic snapshot table deterministically (same seeds as
marmara.synthetic.main), reproduces the sim-disjoint split, loads the trained
classifier from results/models/bigger_ahead.pkl (no retraining), and plots
observed frequency vs mean predicted probability in 10 quantile bins, marker
area proportional to bin count, dashed 1:1 line.

Writes results/figs/fig2_discriminator_reliability.png.
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marmara.synthetic import (DURATION, FEATS, MMAX, N_SIMS, OUT, b_positive,
                               build_dataset, simulate_catalogs)
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    with open(OUT / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    rep = json.load(open(OUT / "etas_fit_report.json"))
    b = rep["operational_b_for_cascade"]
    cat = pd.read_csv(OUT / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    ref_b = b_positive(cat[cat["datetime_utc"] < pd.Timestamp("2022-01-01")]["mag_w"].to_numpy())
    ref_b = float(ref_b) if ref_b else 1.0

    print(f"re-simulating {N_SIMS} base catalogs (deterministic, seed 2026) ...")
    sims = simulate_catalogs(params, DURATION, N_SIMS, mmax=MMAX, seed=2026)
    df = build_dataset(sims, params, b, ref_b)
    print(f"  {len(df)} snapshots")

    # identical sim-disjoint split to marmara.synthetic.main
    sim_ids = df["sim"].to_numpy()
    uniq = np.unique(sim_ids)
    np.random.default_rng(0).shuffle(uniq)
    ns = len(uniq)
    te_s = set(uniq[int(0.85 * ns):])
    te = np.where(np.isin(sim_ids, list(te_s)))[0]

    with open(OUT / "models" / "bigger_ahead.pkl", "rb") as f:
        saved = pickle.load(f)
    clf = saved["clf"]
    X = df[FEATS].to_numpy()
    y = df["label"].to_numpy()
    p = clf.predict_proba(X[te])[:, 1]
    yt = y[te]
    pr = average_precision_score(yt, p)
    roc = roc_auc_score(yt, p)
    print(f"test PR-AUC {pr:.3f}  ROC-AUC {roc:.3f}  (report: 0.119 / 0.622)")

    # 10 quantile bins on predicted probability
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    mp, of, ct = [], [], []
    for k in range(len(edges) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        mp.append(p[m].mean()); of.append(yt[m].mean()); ct.append(int(m.sum()))
    mp, of, ct = np.array(mp), np.array(of), np.array(ct)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lim = max(mp.max(), of.max()) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="0.5", lw=1, label="perfect reliability")
    ax.scatter(mp, of, s=40 + 260 * ct / ct.max(), color="#1f77b4",
               edgecolor="k", zorder=3, label="test bins (area ∝ count)")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Bigger-event-ahead discriminator, sim-disjoint test "
                 f"(PR-AUC {pr:.2f}, ROC-AUC {roc:.2f})")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    out = OUT / "figs" / "fig2_discriminator_reliability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
