"""Figure 1: (a) study region map, (b) frequency-magnitude distribution.

(a) Model box (25.6-30.9E, 39.6-41.9N) and wide box (25.0-31.5E, 39.0-42.5N),
    GEM active-fault traces, the homogenized model-box catalogue (results/catalog/catalog.csv,
    31,360 events), the 2025-04-23 Mw 6.2 Kumburgaz epicentre, and the out-of-box
    2025-08-10 Sindirgi M6.1.
(b) Incremental (by source magnitude type: Md-converted vs ML/other) and cumulative
    FMD on mag_w, with the base Mc = 3.0, the max-curvature Mc = 3.65, and the
    modern-ML-population completeness 2.72 marked. Descriptive only: no GR fit drawn.

Run:  PYTHONPATH=src MARMARA_ROOT=. ./.venv/bin/python scripts/figures/fig_map_fmd.py
Writes paper/figs/figure_map_fmd.png (+ a copy under results/figs/).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from marmara.paths import ROOT, RESULTS

MODEL_BOX = dict(min_lon=25.6, max_lon=30.9, min_lat=39.6, max_lat=41.9)
WIDE_BOX = dict(min_lon=25.0, max_lon=31.5, min_lat=39.0, max_lat=42.5)

cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
faults = json.load(open(ROOT / "data" / "marmara_faults.geojson"))

fig, (axm, axf) = plt.subplots(
    1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.55, 1.0]})

# ------------------------------------------------------------------ (a) map
lon0, lon1 = WIDE_BOX["min_lon"] - 0.12, WIDE_BOX["max_lon"] + 0.12
lat0, lat1 = WIDE_BOX["min_lat"] - 0.10, WIDE_BOX["max_lat"] + 0.10

small = cat[cat.mag_w < 4.5]
mid = cat[(cat.mag_w >= 4.5) & (cat.mag_w < 5.5)]
big = cat[(cat.mag_w >= 5.5) & (cat.mag_w < 6.0)]
axm.scatter(small.longitude, small.latitude, s=1.1, c="#9db8d2", alpha=0.35,
            lw=0, rasterized=True, zorder=2)
axm.scatter(mid.longitude, mid.latitude, s=13, facecolor="none",
            edgecolor="#33526e", lw=0.7, zorder=3)
axm.scatter(big.longitude, big.latitude, s=42, facecolor="none",
            edgecolor="#1a2f42", lw=1.2, zorder=4)

for f in faults["features"]:
    xy = np.asarray(f["geometry"]["coordinates"], dtype=float)
    axm.plot(xy[:, 0], xy[:, 1], color="#a04030", lw=0.8, alpha=0.85, zorder=5)

axm.add_patch(Rectangle((MODEL_BOX["min_lon"], MODEL_BOX["min_lat"]),
                        MODEL_BOX["max_lon"] - MODEL_BOX["min_lon"],
                        MODEL_BOX["max_lat"] - MODEL_BOX["min_lat"],
                        fill=False, edgecolor="k", lw=1.4, zorder=6))
axm.add_patch(Rectangle((WIDE_BOX["min_lon"], WIDE_BOX["min_lat"]),
                        WIDE_BOX["max_lon"] - WIDE_BOX["min_lon"],
                        WIDE_BOX["max_lat"] - WIDE_BOX["min_lat"],
                        fill=False, edgecolor="0.45", lw=1.0, ls="--", zorder=6))

axm.plot(28.2307, 40.8355, marker="*", ms=17, mfc="#d62728", mec="k",
         mew=0.8, ls="none", zorder=8)
axm.annotate("2025 Mw 6.2", (28.2307, 40.8355), xytext=(27.05, 41.05),
             fontsize=8.5, zorder=9)
axm.plot(28.0455, 39.2632, marker="*", ms=11, mfc="none", mec="#d62728",
         mew=1.2, ls="none", zorder=8)
axm.annotate("2025 M6.1 Sındırgı\n(outside model box)", (28.0455, 39.2632),
             xytext=(28.28, 39.10), fontsize=7.5, color="0.25", zorder=9)
axm.plot(28.98, 41.01, marker="s", ms=4.5, color="k", ls="none", zorder=8)
axm.annotate("Istanbul", (28.98, 41.01), xytext=(29.08, 41.14), fontsize=8.5, zorder=9)

axm.set_xlim(lon0, lon1)
axm.set_ylim(lat0, lat1)
axm.set_aspect(1.0 / np.cos(np.deg2rad(40.75)))
axm.set_xlabel("Longitude (°E)")
axm.set_ylabel("Latitude (°N)")
axm.set_title("(a) study region, faults, catalogue", loc="left", fontsize=10)
axm.grid(color="0.92", lw=0.6, zorder=0)
handles = [
    Line2D([], [], color="#a04030", lw=1.2, label="GEM active faults (97 segments)"),
    Line2D([], [], marker="o", ls="none", ms=3, mfc="#9db8d2", mec="none",
           label="mag_w < 4.5"),
    Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec="#33526e",
           label="4.5 ≤ mag_w < 5.5"),
    Line2D([], [], marker="o", ls="none", ms=7, mfc="none", mec="#1a2f42",
           label="mag_w ≥ 5.5"),
    Line2D([], [], color="k", lw=1.4, label="model box (0.1° grid)"),
    Line2D([], [], color="0.45", lw=1.0, ls="--", label="wide box (training only)"),
]
axm.legend(handles=handles, loc="upper left", fontsize=6.8, framealpha=0.9,
           borderpad=0.5, handlelength=1.6)

# ------------------------------------------------------------------ (b) FMD
bins = np.arange(1.0, 6.6, 0.1)
ctr = 0.5 * (bins[:-1] + bins[1:])
is_md = cat.mag_type.str.strip().str.lower() == "md"
n_md, _ = np.histogram(cat.loc[is_md, "mag_w"], bins=bins)
n_ml, _ = np.histogram(cat.loc[~is_md, "mag_w"], bins=bins)

axf.step(ctr, n_ml, where="mid", color="#33526e", lw=1.3,
         label="incremental, ML/other → mag_w")
axf.step(ctr, n_md, where="mid", color="#c98a2b", lw=1.3,
         label="incremental, Md → mag_w")
m_sorted = np.sort(cat.mag_w.to_numpy())
cum = m_sorted.size - np.searchsorted(m_sorted, ctr, side="left")
axf.plot(ctr, cum, "o", ms=3, color="k", label="cumulative N(≥ mag_w)")

for x, ls, lab in ((2.72, ":", "ML-population completeness 2.72"),
                   (3.0, "-", "base Mc = 3.0"),
                   (3.65, "--", "max-curvature Mc = 3.65")):
    axf.axvline(x, color="0.35", ls=ls, lw=1.0)
    axf.text(x - 0.06, 6.5e4, lab, rotation=90, fontsize=6.8, color="0.25",
             ha="right", va="top")

axf.set_yscale("log")
axf.set_ylim(0.7, 8e4)
axf.set_xlim(1.0, 6.5)
axf.set_xlabel("magnitude (mag_w)")
axf.set_ylabel("count per 0.1 bin  /  cumulative")
axf.set_title("(b) frequency–magnitude distribution", loc="left", fontsize=10)
axf.grid(color="0.92", lw=0.6, zorder=0)
axf.legend(loc="lower left", fontsize=6.8, framealpha=0.9)

fig.tight_layout()
out = ROOT / "paper" / "figs" / "figure_map_fmd.png"
fig.savefig(out, dpi=200)
(RESULTS / "figs").mkdir(parents=True, exist_ok=True)
fig.savefig(RESULTS / "figs" / "figure_map_fmd.png", dpi=200)
print("wrote", out)
