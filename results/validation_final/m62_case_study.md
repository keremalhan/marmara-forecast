# M6.2 case study — 2025-04-23 (strictly causal, data < event)

Epicenter (28.23E, 40.84N) cell [12, 26]. There were **0 M≥6 in the model box before this event**, so an observed M6 base rate is undefined; per-cell 30-day P(M≥6) is ~0 (below resolution). We therefore rank cells by the CONTINUOUS expected-M6 rate (lam6) and report the dense M≥4.5 seismicity rank as context.

## (a) 30-day forecasts issued before the event
| issued | epicenter lam6 (exp M6) | pct by lam6 | pct by seismicity | gain6 vs uniform | regional P(M≥6) |
|---|---|---|---|---|---|
| 2025-04-01 | 0.00e+00 | 0.0 | 98.8 | 0.00× | 1.24% |
| 2025-04-22 | 0.00e+00 | 0.0 | 99.4 | 0.00× | 1.17% |

## (b) sequence replay — +10 min after the M~4 foreshock (data < 09:23)
- events in the nascent sequence at the snapshot: **3**
- FTLS color: **UNKNOWN** — only 3 events, need >=~60 for b-value
- bigger-one-ahead score: **0.01329161375808694**
- cascade 24h expected M≥6: cell lam6 **1.25e-04**, regional **6.25e-04**
- quiet-day 24h: cell lam6 0.00e+00, regional 5.00e-04 → **elevation regional 1.25×**

## (c) PASS criteria (reported as-is, nothing tuned to pass)
- epicenter top decile by lam6: **False** (by seismicity rate: True)
- gain6 > 2× uniform base: **False**
- post-foreshock 24h expected-M6 ≥ 10× quiet: **False**
- **ALL PASS: False**

## (d) secondary cases (+10 min after the trigger)
- Oct2_M5: FTLS UNKNOWN (b_ratio None), 24h P(M≥5.5) cell 0.400%, 24h regional P(M≥6) 0.200%
- Sindirgi_Aug10: FTLS UNKNOWN (b_ratio None), 24h P(M≥5.5) cell 0.000%, 24h regional P(M≥6) 0.474%