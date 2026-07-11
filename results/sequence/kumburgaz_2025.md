# Sequence report: kumburgaz_2025

**Trigger:** 2025-04-24 at (28.23, 40.84); host segment **CentralMarmara**

## FTLS traffic light: **RED**
- b_ref (background) = 1.22; b_seq (sequence) = 0.80; ratio = 0.65
- Gulia-Wiemer FTLS caveat (Mizrahi/Gulia GJI 2025 Italy test): the traffic light reached ~68% hit rate but with LOW spatial coverage; treat the color as a coarse alert, NOT a calibrated probability.

## Cascade Monte-Carlo (conditional forward simulation)
| horizon | P(M>=5.5) region | P(M>=6) region | P(M>=6) peak cell |
|---|---|---|---|
| 7d | 11.13% | 2.96% | 0.73% |
| 30d | 22.96% | 6.76% | 1.87% |

## Conditional 'bigger-one-ahead' score: **0.000**
(probability a sim event >= largest_so_far - 0.3 follows within 30 d, from the synthetic conditional classifier)

## Host-segment renewal prior (BPT): P(M~7 in 30 d) = **0.0658%**
(CentralMarmara, 260 yr since last, mean recurrence 250 yr)

## Context (Science 2025, adz0072): M>5 activity has migrated eastward toward the locked Princes Islands segment, with a quiet patch near Avcilar; the Central Marmara / Kumburgaz segment is the most overdue.