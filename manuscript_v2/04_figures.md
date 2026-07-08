# M4 — Figure manifest

v1 figures are retained; four new figures carry the v2/v3 results. All paths are
repo-relative and the files exist on disk.

## Retained from v1 (`paper/figs/`)
| fig | file | shows | section |
|---|---|---|---|
| 1 | `paper/figs/fig1_forecast_map_y35.png` | standing M≥3.5/30-d forecast map (the "where") | Results |
| 2 | `paper/figs/fig2_discriminator_reliability.png` | large-event discriminator reliability | Results |
| 3 | `paper/figs/fig3_m62_countdown.png` | 2025 Mw 6.2 information-arrival countdown | Results |
| 4 | `paper/figs/fig4_m62_forecast_map.png` | Mw 6.2 forecast map / localization | Results |

## New in v2/v3
| fig | file | shows | claim | section |
|---|---|---|---|---|
| **5** | `results/figs_v2/fig5_y30_forest.png` | primary-target forest plot | **C2** | §4.1 |
| **6** | `results/figs_v2/fig6_gnss_placebo.png` | GNSS placebo null | **C3 (void)** | §4.3 |
| **7** | `results/verify/mu_xy.png` | first-gen vs sv-ETAS μ(x,y) | **C5** | §3/§4.2 |
| **8** | `results/csep_v3/csep_v3_consistency.png` | genuine pyCSEP N/M/S/PL | **C6** | §4.4 |

## Draft captions

**Figure 5.** *Primary target (M≥3.0 in 30 days, test split): physics vs the ML hybrid.*
Paired differences (model − hybrid) in PR-AUC (left) and information gain (right) with
95% stationary-block-bootstrap confidence intervals (B=2000, whole-window resampling).
Blue = ETAS physics; grey = other baselines; black-edged markers beat the hybrid (CI
excludes 0). The four physics models beat the hybrid on both axes; on information gain the
hybrid trails every comparator, while on PR-AUC it edges the non-clustering baselines —
the mixed sign that renders those pairs formally inseparable. (`hybrid_gnss` appears to
beat the hybrid here, but Figure 6 shows this gain is spurious.)

**Figure 6.** *The GNSS channel is a null.* Left: the real information-gain from adding the
GNSS channel (+0.098 nats, red line) lies inside the null bands (grey, 95%) of both the
time-shuffle and circular-shift placebos, whose means (dark squares) straddle it; a
coverage-only surrogate reproduces none of it. Right: the y30 operational placebo — the
real ΔPR-AUC (+0.033) and ΔIG (+1.18) both fall inside the placebo null band. A
randomized GNSS series reproduces the "gain," so the channel carries no placebo-robust
signal (C3 = void).

**Figure 7.** *The first-generation ETAS background is already spatially variable.*
Background rate μ(x,y) from the first-generation fit (left) and from the EM-declustered
sv-ETAS (right). The fields nearly coincide (first-generation spatial CoV 1.27), which is
why sv-ETAS and first-generation ETAS are statistically inseparable — not EM degeneracy
but a pre-existing spatially-adaptive background (C5).

**Figure 8.** *Genuine pyCSEP catalog-based consistency (M≥3.0, test period).* Quantiles
of the number, magnitude, spatial and pseudo-likelihood tests (pyCSEP 0.8.0); green =
consistent (not rejected at α=0.05), red = rejected. Cascade and sv-ETAS are number- and
magnitude-consistent; the Mizrahi first-generation model under-counts (N and M rejected).
S and PL reject all models — an artifact of Poisson-catalogue under-dispersion, not the
spatial forecast (see §4.4). pyCSEP and in-house N/M verdicts agree for all three models.

## Regeneration
- Figures 5–6: `PYTHONPATH=src .venv/bin/python scripts/figs_v2.py` (reads
  `results/claims.json`, `results/verify/gnss_placebos.json`, `results/verify/gnss_y30_placebo.json`).
- Figure 7: produced by the Phase-V audit (`scripts/verify/v4_sv_etas_audit.py`).
- Figure 8: `venv-csep/bin/python scripts/csep_v3_run.py` (APFS pyCSEP env).
