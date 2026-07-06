# Calibration battery — pseudo-prospective reliability (2022→2026-03)

Two calibrations: **aggregate** (expected vs observed positives; obs/exp≈1 = calibrated in the mean, robust for rare events) and **shape** (10-bin reliability slope; gate 0.8–1.25 & |intercept|<0.02). Fixes fit on VAL (2022-23) only, never on 2024+ test.

| product | obs/exp (aggregate) | shape testable | slope / int | verdict | fix |
|---|---|---|---|---|---|
| y35 30d (hybrid) | 1.081 (277/256.26) | yes | 1.067 / 0.0001 | calibrated | none |
| y45 30d (hybrid) | 1.058 (35/33.1) | yes | 0.785 / 0.0001 | overfit(slope<0.8) | isotonic_on_val |
| P(M>=5.0) 30d (cascade) | 0.981 (10/10.19) | no (npos=10) | — / — | — | none |
| P(M>=5.5) 30d (cascade) | 1.157 (3/2.59) | no (npos=3) | — / — | — | none |
| P(M>=6.0) 30d (cascade) | 1.433 (1/0.7) | no (npos=1) | — / — | — | none |

**Reading:** every product is calibrated in the MEAN (obs/exp 0.98–1.43, all within Poisson noise of 1). Shape: y35 calibrated (slope 1.07); y45 marginally overconfident (slope 0.785, at the 0.8 gate) — aggregate is fine (1.06) and the wide-box y45 variant (w=0.1, `widebox_y45_report.json`) is the better-calibrated production version. P(M≥5/5.5/6) per-cell are too rare to bin (1–10 positives) — reported honestly via the aggregate, not calibrated away.