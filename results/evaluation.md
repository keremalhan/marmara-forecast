# Evaluation — hybrid (cascade x ML) vs baselines

All ranking claims must be read against results/claims.json (block-bootstrap verdicts); point differences below are NOT claims on their own.

y30: w=0.9; hybrid PR-AUC 0.146 vs cascade 0.229 vs first-gen-ETAS 0.223; IG(hybrid vs cascade)=-0.884, IG(hybrid vs first-gen-ETAS)=-1.460, IG vs smoothed -0.802 | y35: w=0.7; hybrid PR-AUC 0.117 vs cascade 0.130 vs first-gen-ETAS 0.126; IG(hybrid vs cascade)=+0.252, IG(hybrid vs first-gen-ETAS)=-0.381, IG vs smoothed +0.507 | y45: w=0.8; hybrid PR-AUC 0.033 vs cascade 0.066 vs first-gen-ETAS 0.067; IG(hybrid vs cascade)=+1.297, IG(hybrid vs first-gen-ETAS)=-3.344, IG vs smoothed -4.241

## y30 (thr 3.0, w=0.9, sigma 5km) — primary (powered)

### val (n=30475, pos=461)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.1363 | 0.8644 | 0.01429 | 0.718 |
| hybrid_gnss | 0.1316 | 0.8682 | 0.01429 | 0.725 |
| cascade | 0.1355 | 0.8634 | 0.01504 | 0.716 |
| sv_etas | 0.1390 | 0.8698 | 0.01493 | 0.728 |
| modern_etas | 0.1272 | 0.8668 | 0.01418 | 0.723 |
| firstgen_etas | 0.1529 | 0.8726 | 0.01389 | 0.734 |
| smoothed | 0.0920 | 0.8616 | 0.01733 | 0.712 |
| poisson | 0.1204 | 0.8600 | 0.01471 | 0.707 |
| fault_prox | 0.0261 | 0.6753 | 0.01511 | 0.346 |

IG(hybrid − baseline), nats/event:
- vs cascade: +0.6198
- vs poisson: +0.5123
- vs fault_prox: +0.9693
- vs smoothed: +0.9616
- vs firstgen_etas: -0.1438
- vs sv_etas: +0.5065
- vs modern_etas: -0.0362
- vs hybrid_gnss: +0.0301

### test (n=31694, pos=592)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.1458 | 0.8720 | 0.01962 | 0.730 |
| hybrid_gnss | 0.1791 | 0.8760 | 0.01695 | 0.738 |
| cascade | 0.2294 | 0.8734 | 0.01717 | 0.732 |
| sv_etas | 0.2276 | 0.8758 | 0.01716 | 0.737 |
| modern_etas | 0.2057 | 0.8780 | 0.01673 | 0.742 |
| firstgen_etas | 0.2230 | 0.8800 | 0.01638 | 0.746 |
| smoothed | 0.1247 | 0.8688 | 0.01917 | 0.724 |
| poisson | 0.1239 | 0.8558 | 0.01761 | 0.696 |
| fault_prox | 0.0358 | 0.6986 | 0.01831 | 0.386 |

IG(hybrid − baseline), nats/event:
- vs cascade: -0.8841
- vs poisson: -0.8359
- vs fault_prox: -0.4716
- vs smoothed: -0.8024
- vs firstgen_etas: -1.4598
- vs sv_etas: -0.8788
- vs modern_etas: -1.0996
- vs hybrid_gnss: -1.1766

## y35 (thr 3.5, w=0.7, sigma 5km) — powered

### val (n=30475, pos=110)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0601 | 0.8679 | 0.00354 | 0.733 |
| hybrid_gnss | 0.0737 | 0.8693 | 0.00352 | 0.736 |
| cascade | 0.0584 | 0.8496 | 0.00361 | 0.696 |
| sv_etas | 0.0600 | 0.8700 | 0.00359 | 0.737 |
| modern_etas | 0.0527 | 0.8745 | 0.00353 | 0.746 |
| firstgen_etas | 0.0666 | 0.8883 | 0.00352 | 0.774 |
| smoothed | 0.0251 | 0.8733 | 0.00411 | 0.744 |
| poisson | 0.0312 | 0.8364 | 0.00377 | 0.666 |
| fault_prox | 0.0065 | 0.6860 | 0.00365 | 0.371 |

IG(hybrid − baseline), nats/event:
- vs cascade: +0.4457
- vs poisson: +1.7993
- vs fault_prox: +1.0356
- vs smoothed: +1.2474
- vs firstgen_etas: -0.7197
- vs sv_etas: +0.5600
- vs modern_etas: -0.3837
- vs hybrid_gnss: -0.0059

### test (n=31694, pos=167)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.1174 | 0.8841 | 0.00495 | 0.761 |
| hybrid_gnss | 0.1189 | 0.8837 | 0.00502 | 0.760 |
| cascade | 0.1300 | 0.8771 | 0.00498 | 0.748 |
| sv_etas | 0.1276 | 0.8818 | 0.00498 | 0.754 |
| modern_etas | 0.1103 | 0.8945 | 0.00502 | 0.785 |
| firstgen_etas | 0.1260 | 0.8940 | 0.00497 | 0.784 |
| smoothed | 0.0440 | 0.8863 | 0.00544 | 0.768 |
| poisson | 0.0323 | 0.8476 | 0.00532 | 0.681 |
| fault_prox | 0.0099 | 0.6868 | 0.00526 | 0.373 |

IG(hybrid − baseline), nats/event:
- vs cascade: +0.2520
- vs poisson: +1.1361
- vs fault_prox: +0.9462
- vs smoothed: +0.5073
- vs firstgen_etas: -0.3809
- vs sv_etas: +0.2196
- vs modern_etas: -0.0732
- vs hybrid_gnss: +0.0485

## y45 (thr 4.5, w=0.8, sigma 5km) — unpowered — no ranking claims

### val (n=30475, pos=13)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0155 | 0.8731 | 0.00049 | 0.746 |
| hybrid_gnss | 0.0021 | 0.8730 | 0.00048 | 0.746 |
| cascade | 0.0036 | 0.7217 | 0.00043 | 0.413 |
| sv_etas | 0.0032 | 0.7015 | 0.00043 | 0.390 |
| modern_etas | 0.0041 | 0.9048 | 0.00043 | 0.809 |
| firstgen_etas | 0.0061 | 0.9069 | 0.00043 | 0.814 |
| smoothed | 0.0043 | 0.8767 | 0.00043 | 0.753 |
| poisson | 0.0019 | 0.6619 | 0.00043 | 0.295 |
| fault_prox | 0.0009 | 0.6654 | 0.00043 | 0.338 |

IG(hybrid − baseline), nats/event:
- vs cascade: +4.1411
- vs poisson: +6.8969
- vs fault_prox: -0.1359
- vs smoothed: -1.0764
- vs firstgen_etas: -0.5040
- vs sv_etas: +4.3146
- vs modern_etas: +0.6232
- vs hybrid_gnss: +0.2528

### test (n=31694, pos=22)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0334 | 0.8404 | 0.00121 | 0.680 |
| hybrid_gnss | 0.0413 | 0.8294 | 0.00107 | 0.645 |
| cascade | 0.0656 | 0.7210 | 0.00068 | 0.492 |
| sv_etas | 0.0664 | 0.7282 | 0.00068 | 0.504 |
| modern_etas | 0.0644 | 0.9054 | 0.00069 | 0.810 |
| firstgen_etas | 0.0671 | 0.8932 | 0.00069 | 0.786 |
| smoothed | 0.0103 | 0.9049 | 0.00069 | 0.809 |
| poisson | 0.0033 | 0.6534 | 0.00069 | 0.451 |
| fault_prox | 0.0015 | 0.7001 | 0.00069 | 0.419 |

IG(hybrid − baseline), nats/event:
- vs cascade: +1.2973
- vs poisson: +4.4313
- vs fault_prox: -2.9897
- vs smoothed: -4.2410
- vs firstgen_etas: -3.3441
- vs sv_etas: +1.1876
- vs modern_etas: -2.6093
- vs hybrid_gnss: +0.1655
