# Evaluation: model vs 4 baselines

**HEADLINE (y35, M>=3.5, test): the model CLEARS the required bar; it beats plain-Poisson (+1.070), fault-proximity (+0.880) and smoothed-seismicity (+0.658 nats/event in information gain).**

But the physics-based **ETAS baseline is the single strongest predictor** on y35 test (model IG vs ETAS = -0.463; ETAS PR-AUC 0.125 >= model 0.101). The ML model beats the naive baselines but does not beat a properly-fit ETAS.

On the rarer **y45 (M>=4.5) test (only 22 positives), the ETAS and smoothed-seismicity baselines beat the model** (IG vs smoothed -0.953, vs ETAS -0.641). The ML model adds no skill over physical baselines at this magnitude.

## y35  (M>=3.5, smoothed sigma=5 km)

### val  (n=30475, positives=110)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan skill |
|---|---|---|---|---|
| model | 0.0737 | 0.8808 | 0.00341 | 0.761 |
| poisson | 0.0312 | 0.8364 | 0.00377 | 0.666 |
| fault_prox | 0.0065 | 0.6860 | 0.00365 | 0.371 |
| smoothed | 0.0206 | 0.8571 | 0.00443 | 0.712 |
| etas | 0.0683 | 0.8885 | 0.00351 | 0.774 |

**IG (model − baseline), nats/event, positive ⇒ model better:**
- vs poisson: 2.5043
- vs fault_prox: 1.7406
- vs smoothed: 2.1598
- vs etas: 0.0298

Model alert budgets (precision / recall):
- top_0.5pct: P=0.103 R=0.164 (18/175 alarmed)
- top_1pct: P=0.071 R=0.209 (23/325 alarmed)
- top_2pct: P=0.051 R=0.291 (32/625 alarmed)

### test  (n=31694, positives=167)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan skill |
|---|---|---|---|---|
| model | 0.1005 | 0.8760 | 0.00500 | 0.744 |
| poisson | 0.0323 | 0.8476 | 0.00532 | 0.681 |
| fault_prox | 0.0099 | 0.6868 | 0.00526 | 0.373 |
| smoothed | 0.0363 | 0.8715 | 0.00571 | 0.738 |
| etas | 0.1248 | 0.8928 | 0.00496 | 0.781 |

**IG (model − baseline), nats/event, positive ⇒ model better:**
- vs poisson: 1.0701
- vs fault_prox: 0.8803
- vs smoothed: 0.6578
- vs etas: -0.4630

Model alert budgets (precision / recall):
- top_0.5pct: P=0.181 R=0.198 (33/182 alarmed)
- top_1pct: P=0.124 R=0.251 (42/338 alarmed)
- top_2pct: P=0.083 R=0.323 (54/650 alarmed)

## y45  (M>=4.5, smoothed sigma=5 km)

### val  (n=30475, positives=13)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan skill |
|---|---|---|---|---|
| model | 0.0039 | 0.8772 | 0.00043 | 0.763 |
| poisson | 0.0019 | 0.6619 | 0.00043 | 0.295 |
| fault_prox | 0.0009 | 0.6654 | 0.00043 | 0.338 |
| smoothed | 0.0036 | 0.8309 | 0.00043 | 0.662 |
| etas | 0.0071 | 0.9052 | 0.00043 | 0.810 |

**IG (model − baseline), nats/event, positive ⇒ model better:**
- vs poisson: 8.1879
- vs fault_prox: 1.1550
- vs smoothed: 0.5115
- vs etas: 0.3107

Model alert budgets (precision / recall):
- top_0.5pct: P=0.017 R=0.231 (3/175 alarmed)
- top_1pct: P=0.009 R=0.231 (3/325 alarmed)
- top_2pct: P=0.008 R=0.385 (5/625 alarmed)

### test  (n=31694, positives=22)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan skill |
|---|---|---|---|---|
| model | 0.0019 | 0.8072 | 0.00069 | 0.628 |
| poisson | 0.0033 | 0.6534 | 0.00069 | 0.451 |
| fault_prox | 0.0015 | 0.7001 | 0.00069 | 0.419 |
| smoothed | 0.0118 | 0.8965 | 0.00069 | 0.793 |
| etas | 0.0668 | 0.8914 | 0.00069 | 0.782 |

**IG (model − baseline), nats/event, positive ⇒ model better:**
- vs poisson: 7.6447
- vs fault_prox: 0.2236
- vs smoothed: -0.9530
- vs etas: -0.6414

Model alert budgets (precision / recall):
- top_0.5pct: P=0.000 R=0.000 (0/182 alarmed)
- top_1pct: P=0.000 R=0.000 (0/338 alarmed)
- top_2pct: P=0.002 R=0.045 (1/650 alarmed)
