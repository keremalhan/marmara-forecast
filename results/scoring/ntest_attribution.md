# N-test attribution: forecast inflation vs observation undercount

Test period 2024-01-22..2026-03-12, 26 windows, b_op=1.15.

Reproduction check (window 232): re-run supercritical lam30=51.8 vs stored 51.8 (n=1.2119).

## Branching ratio
- fitted b=1.5424, n=0.95; at b_op unrescaled n=1.212 (supercritical); rescaled mean n=0.95.

## M>=3.0 attribution (2x2)

Forecast mean: supercritical **1499.4**, rescaled **1306.1**. Observed: raw **1383**, completeness-corrected **1428** (band [1199, 1699], Aki b=1.121).

| forecast \ observed | raw 1383 | corrected 1428 |
|---|---|---|
| supercritical 1499.4 | **FAIL** (d1 0.9989, d2 0.0012) | PASS (d1 0.969, d2 0.0328) |
| rescaled 1306.1 | **FAIL** (d1 0.0179, d2 0.9833) | **FAIL** (d1 0.0005, d2 0.9996) |

## M>=3.5 anchor (completeness unambiguous, Mc~2.72)

| forecast | observed 361 |
|---|---|
| supercritical 399.5 | PASS (d1 0.9759, d2 0.0272) |
| rescaled 348.2 | PASS (d1 0.2536, d2 0.7631) |

## Completeness diagnostic ([3.0,3.45) band)
- observed 936 vs GR-expected 980.6 (deficit 44.6).
- types [3.0,3.45): {'ML': 919, 'MW': 17}
- types [3.45,4.5): {'ML': 402, 'MW': 9}