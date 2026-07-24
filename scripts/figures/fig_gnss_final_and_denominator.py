"""Regenerate Figure 4 (GNSS placebo, final config) and build Figure 8 (foreshock denominator).

Figure 4 replaces the stale version, whose numbers came from b_op = 1.2 with no branching rescale
and whose caption claimed "a randomized GNSS series reproduces the apparent gain" -- false at the
final configuration. The real story: the two TEMPORAL surrogates fail to void the channel; the
COVERAGE-only surrogate does.

Figure 8 (file figure8_foreshock_denominator.png; renumbered from 9 to match display order) is new: the distribution of the 30-day P(M>=6) gain at t+10min over all 673 qualifying
M>=4.0 triggers, with Kumburgaz marked at its 67th percentile.

Reads results/round4/{r1_gnss_placebo_final.json, r7_foreshock_denominator.json, r7_triggers.parquet}.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/figures/fig_gnss_final_and_denominator.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marmara.paths import RESULTS

FIGS = RESULTS.parent / "paper" / "figs"
R4 = RESULTS / "round4"
INK = "#1a1a1a"
RED = "#c0392b"
BLUE = "#2c6fbb"
GREY = "#8a8a8a"


def fig4():
    d = json.load(open(R4 / "r1_gnss_placebo_final.json"))
    t = d["targets"]["y30"]
    real_ig = t["real"]["d_ig"]["point"]
    real_ig_ci = t["real"]["d_ig"]["ci95"]
    real_pr = t["real"]["d_pr_auc"]["point"]
    real_pr_ci = t["real"]["d_pr_auc"]["ci95"]
    cov_ig = t["coverage_only"]["d_ig"]
    cov_pr = t["coverage_only"]["d_pr_auc"]

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # ---- left: information-gain axis, three surrogate nulls ----
    rows = [
        ("time-shuffle", t["time_shuffle"]["d_ig"]["null_band_95"], t["time_shuffle"]["d_ig"]["mean"]),
        ("circular-shift\n(≥ 2 yr)", t["circular_shift"]["d_ig"]["null_band_95"],
         t["circular_shift"]["d_ig"]["mean"]),
    ]
    a = ax[0]
    ypos = [2, 1]
    for (lab, band, mean), y in zip(rows, ypos):
        a.plot(band, [y, y], color=GREY, lw=7, solid_capstyle="butt", alpha=.55, zorder=1)
        a.plot([mean], [y], marker="|", ms=14, color=INK, mew=2, zorder=3)
    # guide the eye: the coverage-only value lands on top of the real one
    a.plot([real_ig, real_ig], [0, 3], color=RED, lw=1, ls=(0, (2, 3)), alpha=.5, zorder=2)
    # coverage-only is a single retrained value, not a permutation band
    a.plot([cov_ig], [0], marker="D", ms=10, color=BLUE, zorder=4)
    a.annotate(f"coverage-only  {cov_ig:+.3f}\n= 96% of the real gain,\nfrom station count alone",
               (cov_ig, 0), xytext=(12, -2), textcoords="offset points", ha="left", va="center",
               fontsize=8.5, color=BLUE, fontweight="bold")
    a.errorbar([real_ig], [3], xerr=[[real_ig - real_ig_ci[0]], [real_ig_ci[1] - real_ig]],
               fmt="o", ms=8, color=RED, ecolor=RED, elinewidth=1.6, capsize=3, zorder=5)
    a.annotate(f"real  {real_ig:+.3f}", (real_ig, 3), xytext=(8, 9),
               textcoords="offset points", fontsize=9, color=RED, fontweight="bold")
    a.axvline(0, color=INK, lw=.8, ls=":", zorder=0)
    a.set_yticks([3, 2, 1, 0])
    a.set_yticklabels(["real GNSS", "time-shuffle", "circular-shift", "coverage-only"], fontsize=9)
    a.set_ylim(-0.7, 3.7)
    a.set_xlabel("Δ information gain vs plain hybrid (nats per positive)", fontsize=9.5)
    a.set_title("Likelihood axis: the temporal surrogates do not void it —\n"
                "station coverage alone reproduces 96% of it", fontsize=9.5, loc="left")
    a.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # ---- right: ranking axis ----
    b = ax[1]
    rows_pr = [
        ("time-shuffle", t["time_shuffle"]["d_pr_auc"]["null_band_95"],
         t["time_shuffle"]["d_pr_auc"]["mean"]),
        ("circular-shift\n(≥ 2 yr)", t["circular_shift"]["d_pr_auc"]["null_band_95"],
         t["circular_shift"]["d_pr_auc"]["mean"]),
    ]
    for (lab, band, mean), y in zip(rows_pr, ypos):
        b.plot(band, [y, y], color=GREY, lw=7, solid_capstyle="butt", alpha=.55, zorder=1)
        b.plot([mean], [y], marker="|", ms=14, color=INK, mew=2, zorder=3)
    b.plot([cov_pr], [0], marker="D", ms=9, color=BLUE, zorder=4)
    b.errorbar([real_pr], [3], xerr=[[real_pr - real_pr_ci[0]], [real_pr_ci[1] - real_pr]],
               fmt="o", ms=8, color=RED, ecolor=RED, elinewidth=1.6, capsize=3, zorder=5)
    b.annotate(f"real  {real_pr:+.3f}", (real_pr, 3), xytext=(8, 9),
               textcoords="offset points", fontsize=9, color=RED, fontweight="bold")
    b.axvline(0, color=INK, lw=.8, ls=":", zorder=0)
    b.set_yticks([3, 2, 1, 0])
    b.set_yticklabels([])
    b.set_ylim(-0.7, 3.7)
    b.set_xlabel("Δ PR-AUC vs plain hybrid", fontsize=9.5)
    b.set_title("Ranking axis: no gain anywhere,\ninside every null", fontsize=9.5, loc="left")
    b.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)

    fig.suptitle("The GNSS channel is a null — and coverage, not randomization, is what voids it "
                 "(M≥3.0, test)", fontsize=10.5, x=.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(FIGS / "figure4_gnss_placebo.png", dpi=200)
    print(f"wrote {FIGS/'figure4_gnss_placebo.png'}  (real dIG {real_ig:+.4f}, coverage {cov_ig:+.4f})")


def fig8():
    d = json.load(open(R4 / "r7_foreshock_denominator.json"))
    df = pd.read_parquet(R4 / "r7_triggers.parquet")
    g = df["gain_post"].to_numpy()
    hit = df[df.hit].iloc[0]
    kg = float(hit["gain_post"])
    pct = 100.0 * (g < kg).mean()
    n_ge = int((g >= kg).sum()) - 1

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    bins = np.logspace(np.log10(g.min() * 0.95), np.log10(g.max() * 1.05), 46)
    ax.hist(g, bins=bins, color="#c8d6e5", edgecolor="#7f8c9b", lw=.5, zorder=2)
    ax.set_xscale("log")

    for thr, lab, dx, ha in ((10.0, "≥10×", 5, "left"), (40.0, "≥40×", -6, "right")):
        n = int((g >= thr).sum())
        hits = int(df[df.gain_post >= thr]["hit"].sum())
        ax.axvline(thr, color=INK, ls="--", lw=1.1, zorder=3)
        ax.annotate(f"{lab}\n{n} alarms\n{hits} hit" + ("" if hits == 1 else "s"),
                    (thr, ax.get_ylim()[1] * .92), xytext=(dx, 0), textcoords="offset points",
                    fontsize=8.5, va="top", ha=ha, color=INK)

    ax.axvline(kg, color=RED, lw=2, zorder=4)
    ax.annotate(f"Kumburgaz foreshock  {kg:.1f}×\n{pct:.0f}th percentile — "
                f"{n_ge} other triggers\nmatched or exceeded it, none\nfollowed by M≥6.\n"
                f"It falls just below the ≥40× bar\nread from it.",
                (kg, ax.get_ylim()[1] * .55), xytext=(14, 0), textcoords="offset points",
                fontsize=9, color=RED, va="center", fontweight="bold")

    med = float(np.median(g))
    ax.axvline(med, color=BLUE, lw=1.2, ls=":", zorder=3)
    ax.annotate(f"median {med:.1f}×", (med, ax.get_ylim()[1] * .07), xytext=(-6, 0),
                textcoords="offset points", fontsize=8.5, color=BLUE, ha="right")

    ax.set_xlabel("30-day P(M≥6) gain over the freeze's Poisson base, at t + 10 min "
                  "(log scale)", fontsize=9.5)
    ax.set_ylabel("number of triggers", fontsize=9.5)
    ax.set_title(f"Escalation is the norm, not the signal: all {len(df)} M≥4.0 triggers in the "
                 f"model box, 2004–2026\n"
                 f"every trigger escalates ≥5.6×; one M≥6 in 23 years",
                 fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "figure8_foreshock_denominator.png", dpi=200)
    print(f"wrote {FIGS/'figure8_foreshock_denominator.png'}  "
          f"(Kumburgaz {kg:.2f}x at p{pct:.1f}, {n_ge} matched or exceeded)")


if __name__ == "__main__":
    fig4()
    fig8()
