# Evaluation: hybrid (cascade x ML) vs baselines

All ranking claims must be read against results/claims.json (block-bootstrap verdicts); point differences below are NOT claims on their own.

y30: w=0.8; hybrid PR-AUC 0.226 vs cascade 0.233 vs first-gen-ETAS 0.223; IG(hybrid vs cascade)=+0.289, IG(hybrid vs first-gen-ETAS)=-0.124, IG vs smoothed +0.534 | y35: w=0.4; hybrid PR-AUC 0.130 vs cascade 0.131 vs first-gen-ETAS 0.126; IG(hybrid vs cascade)=+0.173, IG(hybrid vs first-gen-ETAS)=-0.380, IG vs smoothed +0.508 | y45: w=0.9; hybrid PR-AUC 0.008 vs cascade 0.062 vs first-gen-ETAS 0.067; IG(hybrid vs cascade)=-0.086, IG(hybrid vs first-gen-ETAS)=-4.776, IG vs smoothed -5.673

## y30 (thr 3.0, w=0.8, sigma 5km): primary (powered)

### val (n=30475, pos=461)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.1323 | 0.8694 | 0.01435 | 0.728 |
| hybrid_gnss | 0.1276 | 0.8685 | 0.01436 | 0.726 |
| cascade | 0.1352 | 0.8643 | 0.01469 | 0.718 |
| sv_etas | 0.1392 | 0.8675 | 0.01459 | 0.724 |
| modern_etas | 0.1278 | 0.8707 | 0.01416 | 0.730 |
| firstgen_etas | 0.1529 | 0.8726 | 0.01389 | 0.734 |
| smoothed | 0.0920 | 0.8616 | 0.01733 | 0.712 |
| poisson | 0.1204 | 0.8600 | 0.01471 | 0.707 |
| fault_prox | 0.0261 | 0.6753 | 0.01511 | 0.346 |

IG(hybrid − baseline), nats/event:
- vs hybrid_naive: -0.0069
- vs cascade: +0.4617
- vs poisson: +0.5262
- vs fault_prox: +0.9831
- vs smoothed: +0.9755
- vs firstgen_etas: -0.1300
- vs sv_etas: +0.3962
- vs modern_etas: -0.1008
- vs hybrid_gnss: -0.0027

### test (n=31694, pos=592)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.2258 | 0.8812 | 0.01660 | 0.748 |
| hybrid_gnss | 0.2231 | 0.8842 | 0.01655 | 0.754 |
| cascade | 0.2325 | 0.8763 | 0.01675 | 0.738 |
| sv_etas | 0.2303 | 0.8763 | 0.01675 | 0.738 |
| modern_etas | 0.2065 | 0.8807 | 0.01670 | 0.747 |
| firstgen_etas | 0.2230 | 0.8800 | 0.01638 | 0.746 |
| smoothed | 0.1247 | 0.8688 | 0.01917 | 0.724 |
| poisson | 0.1239 | 0.8558 | 0.01761 | 0.696 |
| fault_prox | 0.0358 | 0.6986 | 0.01831 | 0.386 |

IG(hybrid − baseline), nats/event:
- vs hybrid_naive: +0.0187
- vs cascade: +0.2886
- vs poisson: +0.5004
- vs fault_prox: +0.8646
- vs smoothed: +0.5339
- vs firstgen_etas: -0.1235
- vs sv_etas: +0.3539
- vs modern_etas: +0.1584
- vs hybrid_gnss: -0.0452

## y35 (thr 3.5, w=0.4, sigma 5km): powered

### val (n=30475, pos=110)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0615 | 0.8684 | 0.00354 | 0.734 |
| hybrid_gnss | 0.0677 | 0.8667 | 0.00354 | 0.729 |
| cascade | 0.0603 | 0.8597 | 0.00359 | 0.715 |
| sv_etas | 0.0602 | 0.8634 | 0.00357 | 0.723 |
| modern_etas | 0.0529 | 0.8780 | 0.00352 | 0.753 |
| firstgen_etas | 0.0666 | 0.8883 | 0.00352 | 0.774 |
| smoothed | 0.0251 | 0.8733 | 0.00411 | 0.744 |
| poisson | 0.0312 | 0.8364 | 0.00377 | 0.666 |
| fault_prox | 0.0065 | 0.6860 | 0.00365 | 0.371 |

IG(hybrid − baseline), nats/event:
- vs hybrid_naive: -0.0282
- vs cascade: +0.2100
- vs poisson: +1.8231
- vs fault_prox: +1.0594
- vs smoothed: +1.2712
- vs firstgen_etas: -0.6959
- vs sv_etas: +0.6318
- vs modern_etas: -0.4375
- vs hybrid_gnss: +0.0103

### test (n=31694, pos=167)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.1299 | 0.8842 | 0.00492 | 0.763 |
| hybrid_gnss | 0.1189 | 0.8850 | 0.00501 | 0.764 |
| cascade | 0.1307 | 0.8795 | 0.00495 | 0.753 |
| sv_etas | 0.1306 | 0.8770 | 0.00496 | 0.744 |
| modern_etas | 0.1105 | 0.8973 | 0.00502 | 0.790 |
| firstgen_etas | 0.1260 | 0.8940 | 0.00497 | 0.784 |
| smoothed | 0.0440 | 0.8863 | 0.00544 | 0.768 |
| poisson | 0.0323 | 0.8476 | 0.00532 | 0.681 |
| fault_prox | 0.0099 | 0.6868 | 0.00526 | 0.373 |

IG(hybrid − baseline), nats/event:
- vs hybrid_naive: -0.0187
- vs cascade: +0.1734
- vs poisson: +1.1370
- vs fault_prox: +0.9472
- vs smoothed: +0.5082
- vs firstgen_etas: -0.3800
- vs sv_etas: +0.1749
- vs modern_etas: -0.1508
- vs hybrid_gnss: +0.0526

## y45 (thr 4.5, w=0.9, sigma 5km): unpowered, no ranking claims

### val (n=30475, pos=13)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0055 | 0.7760 | 0.00049 | 0.544 |
| hybrid_gnss | 0.0048 | 0.7738 | 0.00049 | 0.540 |
| cascade | 0.0109 | 0.6451 | 0.00043 | 0.324 |
| sv_etas | 0.0026 | 0.7546 | 0.00043 | 0.554 |
| modern_etas | 0.0042 | 0.9097 | 0.00043 | 0.819 |
| firstgen_etas | 0.0061 | 0.9069 | 0.00043 | 0.814 |
| smoothed | 0.0043 | 0.8767 | 0.00043 | 0.753 |
| poisson | 0.0019 | 0.6619 | 0.00043 | 0.295 |
| fault_prox | 0.0009 | 0.6654 | 0.00043 | 0.338 |

IG(hybrid − baseline), nats/event:
- vs hybrid_naive: +0.0000
- vs cascade: +6.2435
- vs poisson: +6.7313
- vs fault_prox: -0.3016
- vs smoothed: -1.2421
- vs firstgen_etas: -0.6697
- vs sv_etas: +2.9604
- vs modern_etas: +0.3358
- vs hybrid_gnss: +0.0495

### test (n=31694, pos=22)
| predictor | PR-AUC | ROC-AUC | Brier | Molchan |
|---|---|---|---|---|
| hybrid | 0.0078 | 0.8000 | 0.00145 | 0.618 |
| hybrid_gnss | 0.0079 | 0.7929 | 0.00144 | 0.606 |
| cascade | 0.0622 | 0.7137 | 0.00068 | 0.464 |
| sv_etas | 0.0659 | 0.7563 | 0.00068 | 0.533 |
| modern_etas | 0.0644 | 0.9040 | 0.00069 | 0.808 |
| firstgen_etas | 0.0671 | 0.8932 | 0.00069 | 0.786 |
| smoothed | 0.0103 | 0.9049 | 0.00069 | 0.809 |
| poisson | 0.0033 | 0.6534 | 0.00069 | 0.451 |
| fault_prox | 0.0015 | 0.7001 | 0.00069 | 0.419 |

IG(hybrid − baseline), nats/event:
- vs hybrid_naive: +0.0000
- vs cascade: -0.0857
- vs poisson: +2.9997
- vs fault_prox: -4.4213
- vs smoothed: -5.6727
- vs firstgen_etas: -4.7757
- vs sv_etas: -1.4440
- vs modern_etas: -4.0869
- vs hybrid_gnss: -0.4762
