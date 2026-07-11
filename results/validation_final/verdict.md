# VERDICT: forecast everything, no overfit/underfit

1. AGGREGATE calibration: ALL products calibrated in the mean (obs/exp within Poisson noise of 1: y35 1.074, y45 1.327, M5 0.75, M5.5 0.836, M6 1.0).
2. SHAPE (10-bin): y35 calibrated (slope 1.026); y45 underfit(slope>1.25) (slope 1.323; wide-box y45 w=0.1 is the calibrated variant).
3. Rarity-limited shape (reported via aggregate, NOT tuned): P(M>=5.0) 30d (cascade), P(M>=5.5) 30d (cascade), P(M>=6.0) 30d (cascade).
4. M6.2 spatial (30d, 04-22): epicenter percentile by lam6 90 (by seismicity 99), gain6 8.6×, top-decile-lam6=True.
5. M6.2 short-term: post-foreshock 24h regional expected-M6 7.5e-04 = 1.0× quiet, FTLS UNKNOWN (n_seq=3), but absolute probability stays small.
6. M6.2 gate ALL-PASS: False.
7. Physical finding: M6/30d/per-cell is below forecast resolution and had NO base rate (0 M6 in the box pre-event); the only real precursor was the 36-min M4 foreshock, and 10 min after it there were too few events for FTLS/classifier. That absent information IS the finding.
8. Underfit check (y35 test): cascade IG vs baselines all>0 = True; hybrid = True.
9. Supported claims: 30-day spatial hazard + sequence-response (FTLS/cascade) are calibrated and skillful; M≥6 short-term is an ELEVATION signal, not a probability of certainty.
10. NOT supported: sharp deterministic prediction of a specific large event ahead of its immediate foreshock; reported honestly, not tuned.

## Countdown verdict (earliest top-decile freeze | max prob gain before event)
- **M6.2 2025-04-23**: epicenter in top decile from ≥365 days before (persistent spatial hazard, whole year); max P(M≥6) gain = 45.17× (at +10 min post-foreshock), driven by spatial ranking + the immediate foreshock, not by any earlier precursor; the cell was quiescent all year until the ML 4.0.
- **M5.0 2025-10-02**: top-decile from ≥90 days; max gain 165.19× (at +1 h); the bigger-ahead classifier was elevated (0.01–0.65) in the weeks before, a genuine active sequence.
- **Sındırgı Oct-27**: NOT spatially covered (epicenter south of the model box → percentile 0, a coverage limit); FTLS reached RED and the bigger-ahead score for Oct-27 was a weak 0.13 during the decaying Aug M6.1 sequence: a true positive, only weakly flagged.