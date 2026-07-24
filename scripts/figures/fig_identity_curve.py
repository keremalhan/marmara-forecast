"""Regenerate figure2_identity_curve.png — the count-scoring identity a(h)=h-1-ln h.

Cascade points (h, a(h)) are computed from the frozen predictions (Sigma_lambda/N_pos per
target); the y45 overfit hybrid is placed at its measured h=4.98 on the curve. Values:
  M>=3.0: h=2.206, a=0.415 ; M>=3.5: h=2.085, a=0.350 ; M>=4.5: h=1.122, a=0.007
  overfit y45 hybrid: h=4.98, a=2.375 (the labeled outlier / pathology of section 4).
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def a_of(h):
    return h - 1.0 - np.log(h)

# cascade, traced across targets (h from Sigma_lambda/N_pos on the test set)
CASC = [(2.206, "M≥3.0"), (2.085, "M≥3.5"), (1.122, "M≥4.5")]
HYB_Y45_H = 4.98

h = np.linspace(1.0, 5.25, 400)
fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=160)
ax.plot(h, a_of(h), color="#333333", lw=2.0, zorder=1,
        label=r"$a(h)=h-1-\ln h$")

# cascade points on the curve
cx = [c[0] for c in CASC]
cy = [a_of(c[0]) for c in CASC]
ax.scatter(cx, cy, s=70, color="#1f77b4", zorder=3, edgecolor="white", linewidth=0.8,
           label="cascade (per target)")
# hand-placed offsets so the two middle labels don't collide
OFF = {"M≥3.0": (10, 12), "M≥3.5": (14, -22), "M≥4.5": (10, 8)}
for (hh, lab), yy in zip(CASC, cy):
    ax.annotate(f"{lab}: $h$={hh:.3f}, $a$={yy:.3f}", (hh, yy),
                textcoords="offset points", xytext=OFF[lab],
                fontsize=8, color="#1f77b4")

# overfit y45 hybrid: the labeled outlier
oy = a_of(HYB_Y45_H)
ax.scatter([HYB_Y45_H], [oy], s=90, marker="X", color="#d62728", zorder=3,
           edgecolor="white", linewidth=0.8, label="overfit y45 hybrid (pathology)")
ax.annotate(f"overfit y45 hybrid\n$h$={HYB_Y45_H:.2f}, $a$={oy:.2f}", (HYB_Y45_H, oy),
            textcoords="offset points", xytext=(-96, -6), fontsize=8, color="#d62728")

ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
ax.set_xlabel(r"$h=\Sigma\lambda / N_{\mathrm{pos}}$  (expected events per occupied cell-window)")
ax.set_ylabel(r"harvestable count-scoring artifact  $a(h)$  (nats/positive)")
ax.set_xlim(0.95, 5.3)
ax.set_ylim(-0.1, 2.6)
ax.legend(loc="upper left", fontsize=8, frameon=False)
ax.grid(True, alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig("paper/figs/figure2_identity_curve.png", dpi=160)
print("wrote paper/figs/figure2_identity_curve.png")
for hh, lab in CASC:
    print(f"  {lab}: h={hh:.3f} a={a_of(hh):.4f}")
print(f"  y45 hybrid: h={HYB_Y45_H:.2f} a={a_of(HYB_Y45_H):.4f}")
