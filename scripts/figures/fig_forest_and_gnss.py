"""Generate the two headline figures from committed result JSONs.

fig5_y30_forest.png  : primary-target (y30/test) forest plot of (model - hybrid) for
                        ΔPR-AUC and ΔIG with 95% block-bootstrap CIs -> visualises C2
                        (physics beats the ML hybrid). Source: results/claims.json.
fig6_gnss_placebo.png: the GNSS null -> visualises C3=VOID. Left: channel IG gain vs
                        time-shuffle / circular-shift / coverage placebo nulls
                        (results/verify/gnss_placebos.json). Right: y30 operational
                        ΔPR-AUC and ΔIG real value inside the placebo null band
                        (results/verify/gnss_y30_placebo.json).
Run:  PYTHONPATH=src .venv/bin/python scripts/figures/fig_forest_and_gnss.py
"""
from __future__ import annotations

import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marmara.paths import RESULTS

OUT = RESULTS / "figs"
OUT.mkdir(parents=True, exist_ok=True)
PHYS = {"cascade", "sv_etas", "modern_etas", "firstgen_etas"}


def y30_forest():
    claims = json.load(open(RESULTS / "claims.json"))["claims"]
    # hybrid_vs_X on y30/test -> flip to (X - hybrid): positive = X beats hybrid
    order = ["cascade", "sv_etas", "modern_etas", "firstgen_etas", "hybrid_gnss", "smoothed", "poisson"]
    rows = {}
    for c in claims:
        if c["target"] == "y30" and c["split"] == "test" and c["A"] == "hybrid" and c["B"] in order:
            rows[c["B"]] = c
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, lab in ((axes[0], "pr_auc", "Δ PR-AUC  (model − hybrid)"),
                         (axes[1], "ig", "Δ information gain, nats  (model − hybrid)")):
        ys = np.arange(len(order))[::-1]
        for y, m in zip(ys, order):
            c = rows[m]
            pt = -c[key]["point"]                      # flip A-B -> B-A = X - hybrid
            lo, hi = -c[key]["ci95"][1], -c[key]["ci95"][0]
            beats = lo > 0                              # X beats hybrid on this axis
            col = "tab:blue" if m in PHYS else "0.5"
            ax.plot([lo, hi], [y, y], color=col, lw=2.4, zorder=2)
            ax.plot(pt, y, "o", color=col, ms=7, zorder=3,
                    markeredgecolor="k" if beats else "none")
        ax.axvline(0, color="k", lw=0.9, ls="--", zorder=1)
        ax.set_yticks(ys); ax.set_yticklabels(order)
        ax.set_xlabel(lab); ax.grid(axis="x", color="0.9", zorder=0)
    axes[0].set_title("Primary target M≥3.0 / 30 d (test): physics vs the ML hybrid", loc="left", fontsize=10)
    fig.text(0.5, 0.01, "blue = ETAS physics; grey = other baselines; black-edged marker = beats hybrid "
             "(95% CI excludes 0).  Positive = model better than hybrid.", ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig5_y30_forest.png", dpi=140); plt.close(fig)
    print("wrote fig5_y30_forest.png")


def gnss_placebo():
    ch = json.load(open(RESULTS / "verify" / "gnss_placebos.json"))
    op = json.load(open(RESULTS / "verify" / "gnss_y30_placebo.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    # left: channel IG gain vs placebo nulls (IG)
    ax = axes[0]
    real = ch["real_aug_vs_base_test_ig"]
    labs, means, los, his = [], [], [], []
    for k, lab in (("time_shuffle", "time-shuffle"), ("circular_shift", "circular-shift ≥2yr")):
        d = ch[k]; labs.append(lab); means.append(d["mean_ig"])
        los.append(d["p95_range"][0]); his.append(d["p95_range"][1])
    labs.append("coverage-only"); means.append(ch["coverage_only"]["test_ig"]); los.append(np.nan); his.append(np.nan)
    ys = np.arange(len(labs))[::-1]
    for y, m, lo, hi in zip(ys, means, los, his):
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color="0.5", lw=8, alpha=0.35, solid_capstyle="butt", zorder=1)
        ax.plot(m, y, "s", color="0.35", ms=7, zorder=3)
    ax.axvline(real, color="tab:red", lw=2, zorder=4, label=f"real gain {real:+.3f}")
    ax.set_yticks(ys); ax.set_yticklabels(labs); ax.set_xlabel("Δ information gain, nats (aug − base)")
    ax.set_title("GNSS channel vs placebo nulls", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", color="0.92", zorder=0)

    # right: y30 operational real-vs-null for ΔPR and ΔIG (normalised bars)
    ax = axes[1]
    items = [("Δ PR-AUC", op["real_dPR_auc"], op["placebo_dPR_mean"], op["placebo_dPR_p95"]),
             ("Δ IG (nats)", op["real_dIG"], op["placebo_dIG_mean"], op["placebo_dIG_p95"])]
    ys = [1, 0]
    for y, (lab, real_v, pmean, p95) in zip(ys, items):
        ax.plot(p95, [y, y], color="0.5", lw=8, alpha=0.35, solid_capstyle="butt", zorder=1)
        ax.plot(pmean, y, "s", color="0.35", ms=7, zorder=3, label="placebo mean" if y == 1 else None)
        ax.plot(real_v, y, "*", color="tab:red", ms=16, zorder=4, label="real" if y == 1 else None)
        dy = -22 if y == 1 else 12                    # top label below the star, bottom label above
        ax.annotate(f"real {real_v:+.3f}\n(inside null)", (real_v, y), textcoords="offset points",
                    xytext=(-40 if y == 1 else 6, dy), fontsize=8, color="tab:red")
    ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items]); ax.set_xlim(-0.4, 1.5)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("value"); ax.set_title("y30 operational placebo: real gain is spurious", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="x", color="0.92", zorder=0)

    fig.text(0.5, 0.005, "C3 = VOID: the real GNSS 'gain' lies inside every placebo null "
             "(a time-shuffled series reproduces it).", ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig6_gnss_placebo.png", dpi=140); plt.close(fig)
    print("wrote fig6_gnss_placebo.png")


if __name__ == "__main__":
    y30_forest()
    gnss_placebo()
    print(f"figures in {OUT}")
