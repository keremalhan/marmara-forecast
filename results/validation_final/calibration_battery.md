# Calibration battery: pseudo-prospective reliability (2022→2026-03)

Two calibrations: **aggregate** (expected vs observed positives; obs/exp≈1 = calibrated in the mean, robust for rare events) and **shape** (10-bin reliability slope; gate 0.8–1.25 & |intercept|<0.02). Fixes fit on VAL (2022-23) only, never on 2024+ test.

| product | obs/exp (aggregate) | shape testable | slope / int | verdict | fix |
|---|---|---|---|---|---|
| y35 30d (hybrid) | 1.019 (277/271.76) | yes | 0.951 / 0.0003 | calibrated | none |
| y45 30d (hybrid) | 1.257 (35/27.85) | yes | 0.91 / 0.0002 | calibrated | none |
| P(M>=5.0) 30d (cascade) | 0.855 (10/11.69) | no (npos=10) | — / — | — | none |
| P(M>=5.5) 30d (cascade) | 0.958 (3/3.13) | no (npos=3) | — / — | — | none |
| P(M>=6.0) 30d (cascade) | 1.208 (1/0.83) | no (npos=1) | — / — | — | none |

**Reading:** every product is calibrated in the MEAN (obs/exp 0.85–1.26, all within Poisson noise of 1). Shape: y35 calibrated (slope 0.951); y45 calibrated (slope 0.91). The wide-box y45 variant (w=0.1, `widebox_y45_report.json`) is the better-calibrated production version. P(M≥5/5.5/6) per-cell are too rare to bin (1–10 positives); reported via the aggregate, not calibrated away.