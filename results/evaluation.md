# Evaluation: hybrid (cascade x ML) vs baselines

All ranking claims must be read against results/claims.json (block-bootstrap verdicts); point differences below are NOT claims on their own.

y30: w=0.8; hybrid PR-AUC 0.218 vs cascade 0.228 vs first-gen-ETAS 0.223; IG(hybrid vs cascade)=+0.494, IG(hybrid vs first-gen-ETAS)=-0.116, IG vs smoothed +0.542 | y35: w=0.6; hybrid PR-AUC 0.070 vs cascade 0.128 vs first-gen-ETAS 0.126; IG(hybrid vs cascade)=-0.320, IG(hybrid vs first-gen-ETAS)=-1.048, IG vs smoothed -0.160 | y45: w=0.5; hybrid PR-AUC 0.076 vs cascade 0.069 vs first-gen-ETAS 0.067; IG(hybrid vs cascade)=-3.763, IG(hybrid vs first-gen-ETAS)=-6.505, IG vs smoothed -7.402

## y30 (thr 3.0, w=0.8, sigma 5km): primary (powered)

### val (n=30475, pos=461)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.1374 | 0.8672 | 0.01432 | 0.723 |
| hybrid_gnss | 0.1335 | 0.8714 | 0.01426 | 0.732 |
| cascade | 0.1401 | 0.8660 | 0.01510 | 0.721 |
| sv_etas | 0.1376 | 0.8686 | 0.01506 | 0.726 |
| modern_etas | 0.1272 | 0.8668 | 0.01418 | 0.723 |
| firstgen_etas | 0.1529 | 0.8726 | 0.01389 | 0.734 |
| smoothed | 0.0920 | 0.8616 | 0.01733 | 0.712 |
| poisson | 0.1204 | 0.8600 | 0.01471 | 0.707 |
| fault_prox | 0.0261 | 0.6753 | 0.01511 | 0.346 |

IG(hybrid − baseline), nats/event:
- vs cascade: +0.5941
- vs poisson: +0.4610
- vs fault_prox: +0.9179
- vs smoothed: +0.9103
- vs firstgen_etas: -0.1952
- vs sv_etas: +0.5341
- vs modern_etas: -0.0875
- vs hybrid_gnss: -0.0621

### test (n=31694, pos=592)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.2182 | 0.8801 | 0.01654 | 0.746 |
| hybrid_gnss | 0.2219 | 0.8830 | 0.01647 | 0.752 |
| cascade | 0.2283 | 0.8757 | 0.01729 | 0.737 |
| sv_etas | 0.2290 | 0.8766 | 0.01725 | 0.738 |
| modern_etas | 0.2057 | 0.8780 | 0.01673 | 0.742 |
| firstgen_etas | 0.2230 | 0.8800 | 0.01638 | 0.746 |
| smoothed | 0.1247 | 0.8688 | 0.01917 | 0.724 |
| poisson | 0.1239 | 0.8558 | 0.01761 | 0.696 |
| fault_prox | 0.0358 | 0.6986 | 0.01831 | 0.386 |

IG(hybrid − baseline), nats/event:
- vs cascade: +0.4936
- vs poisson: +0.5084
- vs fault_prox: +0.8726
- vs smoothed: +0.5419
- vs firstgen_etas: -0.1156
- vs sv_etas: +0.4988
- vs modern_etas: +0.2447
- vs hybrid_gnss: -0.0360

## y35 (thr 3.5, w=0.6, sigma 5km): powered

### val (n=30475, pos=110)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0434 | 0.8751 | 0.00360 | 0.748 |
| hybrid_gnss | 0.0461 | 0.8772 | 0.00356 | 0.752 |
| cascade | 0.0604 | 0.8677 | 0.00363 | 0.732 |
| sv_etas | 0.0590 | 0.8715 | 0.00363 | 0.741 |
| modern_etas | 0.0527 | 0.8745 | 0.00353 | 0.746 |
| firstgen_etas | 0.0666 | 0.8883 | 0.00352 | 0.774 |
| smoothed | 0.0251 | 0.8733 | 0.00411 | 0.744 |
| poisson | 0.0312 | 0.8364 | 0.00377 | 0.666 |
| fault_prox | 0.0065 | 0.6860 | 0.00365 | 0.371 |

IG(hybrid − baseline), nats/event:
- vs cascade: +0.3493
- vs poisson: +1.7576
- vs fault_prox: +0.9939
- vs smoothed: +1.2057
- vs firstgen_etas: -0.7614
- vs sv_etas: +0.5573
- vs modern_etas: -0.4253
- vs hybrid_gnss: -0.0520

### test (n=31694, pos=167)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0696 | 0.8838 | 0.00617 | 0.761 |
| hybrid_gnss | 0.0614 | 0.8799 | 0.00667 | 0.755 |
| cascade | 0.1276 | 0.8816 | 0.00502 | 0.754 |
| sv_etas | 0.1299 | 0.8802 | 0.00501 | 0.752 |
| modern_etas | 0.1103 | 0.8945 | 0.00502 | 0.785 |
| firstgen_etas | 0.1260 | 0.8940 | 0.00497 | 0.784 |
| smoothed | 0.0440 | 0.8863 | 0.00544 | 0.768 |
| poisson | 0.0323 | 0.8476 | 0.00532 | 0.681 |
| fault_prox | 0.0099 | 0.6868 | 0.00526 | 0.373 |

IG(hybrid − baseline), nats/event:
- vs cascade: -0.3202
- vs poisson: +0.4691
- vs fault_prox: +0.2793
- vs smoothed: -0.1597
- vs firstgen_etas: -1.0479
- vs sv_etas: -0.4162
- vs modern_etas: -0.7402
- vs hybrid_gnss: +1.6415

## y45 (thr 4.5, w=0.5, sigma 5km): unpowered, no ranking claims

### val (n=30475, pos=13)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0028 | 0.8235 | 0.00048 | 0.647 |
| hybrid_gnss | 0.0033 | 0.8110 | 0.00046 | 0.621 |
| cascade | 0.0039 | 0.7067 | 0.00043 | 0.406 |
| sv_etas | 0.0023 | 0.7514 | 0.00043 | 0.471 |
| modern_etas | 0.0041 | 0.9048 | 0.00043 | 0.809 |
| firstgen_etas | 0.0061 | 0.9069 | 0.00043 | 0.814 |
| smoothed | 0.0043 | 0.8767 | 0.00043 | 0.753 |
| poisson | 0.0019 | 0.6619 | 0.00043 | 0.295 |
| fault_prox | 0.0009 | 0.6654 | 0.00043 | 0.338 |

IG(hybrid − baseline), nats/event:
- vs cascade: +2.2409
- vs poisson: +4.6951
- vs fault_prox: -2.3377
- vs smoothed: -3.2782
- vs firstgen_etas: -2.7058
- vs sv_etas: +1.0746
- vs modern_etas: -1.5786
- vs hybrid_gnss: -0.1596

### test (n=31694, pos=22)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0759 | 0.8542 | 0.00102 | 0.707 |
| hybrid_gnss | 0.0466 | 0.8616 | 0.00095 | 0.720 |
| cascade | 0.0692 | 0.7837 | 0.00068 | 0.632 |
| sv_etas | 0.0664 | 0.7371 | 0.00068 | 0.513 |
| modern_etas | 0.0644 | 0.9054 | 0.00069 | 0.810 |
| firstgen_etas | 0.0671 | 0.8932 | 0.00069 | 0.786 |
| smoothed | 0.0103 | 0.9049 | 0.00069 | 0.809 |
| poisson | 0.0033 | 0.6534 | 0.00069 | 0.451 |
| fault_prox | 0.0015 | 0.7001 | 0.00069 | 0.419 |

IG(hybrid − baseline), nats/event:
- vs cascade: -3.7630
- vs poisson: +1.2699
- vs fault_prox: -6.1511
- vs smoothed: -7.4024
- vs firstgen_etas: -6.5055
- vs sv_etas: -2.4650
- vs modern_etas: -5.7707
- vs hybrid_gnss: -0.0822
