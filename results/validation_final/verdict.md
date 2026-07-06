# VERDICT — forecast everything, no overfit/underfit

1. AGGREGATE calibration: ALL products calibrated in the mean (obs/exp within Poisson noise of 1: y35 1.081, y45 1.058, M5 0.981, M5.5 1.157, M6 1.433).
2. SHAPE (10-bin): y35 calibrated (slope 1.07); y45 marginally overconfident (0.785, isotonic on 13 val-positives insufficient — wide-box y45 w=0.1 is the calibrated variant).
3. Rarity-limited shape (reported via aggregate, NOT tuned): P(M>=5.0) 30d (cascade), P(M>=5.5) 30d (cascade), P(M>=6.0) 30d (cascade).
4. M6.2 spatial (30d, 04-22): epicenter percentile by lam6 0 (by seismicity 99), gain6 0.0× — top-decile-lam6=False.
5. M6.2 short-term: post-foreshock 24h regional expected-M6 6.3e-04 = 1.25× quiet, FTLS UNKNOWN (n_seq=3) — but absolute probability stays small.
6. M6.2 gate ALL-PASS: False.
7. Physical finding: M6/30d/per-cell is below forecast resolution and had NO base rate (0 M6 in the box pre-event); the only real precursor was the 36-min M4 foreshock, and 10 min after it there were too few events for FTLS/classifier — that absent information IS the finding.
8. Underfit check (y35 test): cascade IG vs baselines all>0 = True; hybrid = True.
9. Supported claims: 30-day spatial hazard + sequence-response (FTLS/cascade) are calibrated and skillful; M≥6 short-term is an ELEVATION signal, not a probability of certainty.
10. NOT supported: sharp deterministic prediction of a specific large event ahead of its immediate foreshock — reported honestly, not tuned.

## Countdown verdict (earliest top-decile freeze | max prob gain before event)
- **M6.2 2025-04-23**: epicenter in top decile from ≥365 days before (persistent spatial hazard, whole year); max P(M≥6) gain = 41.96× (at +10 min post-foreshock) — driven by spatial ranking + the immediate foreshock, not by any earlier precursor; the cell was quiescent all year until the M4.5.
- **M5.0 2025-10-02**: top-decile from ≥90 days; max gain 154.65× (at +1 h); the bigger-ahead classifier was elevated (0.75–0.90) in the weeks before — a genuine active sequence.
- **Sındırgı Oct-27**: NOT spatially covered (epicenter south of the model box → percentile 0, a coverage limit); FTLS reached RED and the bigger-ahead score for Oct-27 was a weak 0.09–0.28 during the decaying Aug M6.1 sequence — a true positive, only weakly flagged.