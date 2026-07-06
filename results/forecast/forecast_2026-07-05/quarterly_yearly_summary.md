# Quarterly & yearly forecast — 2026-07-05 (Marmara model box)

**Backtest-and-scale (validation gate).** Pseudo-prospective at 2022/23/24/25-01-01, 90d & 365d: predicted vs realized M>=4.5 regional counts. Aggregate reliability slope **1.21** (90d 1.11, 365d 1.23). Within 0.7-1.3 -> **no scaling applied**.

| horizon | epoch | predicted | realized | ratio |
|---|---|---|---|---|
| 90d | 2022-01-01 | 2.42 | 4 | 1.66 |
| 90d | 2023-01-01 | 2.44 | 2 | 0.82 |
| 90d | 2024-01-01 | 2.59 | 4 | 1.54 |
| 90d | 2025-01-01 | 2.47 | 1 | 0.40 |
| 365d | 2022-01-01 | 10.29 | 8 | 0.78 |
| 365d | 2023-01-01 | 10.27 | 7 | 0.68 |
| 365d | 2024-01-01 | 10.55 | 11 | 1.04 |
| 365d | 2025-01-01 | 10.37 | 25 | 2.41 |

**Honest read of the backtest:** 90d is well-calibrated. On the 365d QUIET epochs (2022, 2023) the near-critical cascade mildly OVER-predicts (ratio ~0.7-0.8 — the expected long-horizon inflation), but the 2025 epoch's realized count is huge (the unforecastable Apr-2025 M6.2 sequence, ratio 2.4) and pulls the aggregate to 1.23, inside the gate. So no global scaling is triggered, but the yearly M6 central should be read as an upper-leaning estimate for quiet periods; the b-range low end and the Poisson base rate bracket the realistic floor.

Cascade-only + renewal (the 30-day ML hybrid is not valid at these horizons). Per-sim event cap 50,000 (0% capped); Wilson 95% intervals shown; b-ensemble {1.02, 1.2, 1.54}.

## quarter 90d  (K=10000, capped 0.00%)

| M | cascade P (central) | Wilson 95% | b-range (1.54→1.02) | Poisson base |
|---|---|---|---|---|
| >=5.0 | 45.14% | 44.17–46.12% | 10.44–77.81% | 37.62% |
| >=5.5 | 14.38% | 13.71–15.08% | 1.83–38.78% | 11.82% |
| >=6.0 | 3.86% | 3.50–4.26% | 0.19–14.71% | 1.04% |

- **Renewal** P(characteristic M~7, any segment): **0.351%** (Central Marmara 0.197%).
- **Combined M>=6.8 layer** 1-(1-P_cascade_M6)(1-P_renewal): **4.20%**.

## year 365d  (K=3000, capped 0.00%)

| M | cascade P (central) | Wilson 95% | b-range (1.54→1.02) | Poisson base |
|---|---|---|---|---|
| >=5.0 | 91.80% | 90.76–92.73% | 36.77–99.87% | 85.25% |
| >=5.5 | 47.47% | 45.68–49.26% | 7.80–88.43% | 39.97% |
| >=6.0 | 15.70% | 14.44–17.05% | 1.53–49.37% | 4.16% |

- **Renewal** P(characteristic M~7, any segment): **1.420%** (Central Marmara 0.798%).
- **Combined M>=6.8 layer** 1-(1-P_cascade_M6)(1-P_renewal): **16.90%**.

## How to read this

- The **yearly number ≈ long-term hazard + a current-sequence bump** — expected behavior as the Omori/triggered signal decays over the year, not lost skill.
- **Renewal dominates at M~7** (characteristic ruptures, BPT clock); the **cascade dominates at M<=5.5** (aftershock/background productivity). M6 is the crossover — read cascade and renewal together (the combined line).
- Ranges come from the b-value ensemble (the dominant uncertainty); the central uses the calibrated b_op=1.2 (the backtest confirmed no global scaling was needed). Treat M6 as order-of-magnitude.