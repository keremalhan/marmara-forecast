# Quarterly & yearly forecast: 2026-07-05 (Marmara model box)

**Backtest-and-scale (validation gate).** Pseudo-prospective at 2022/23/24/25-01-01, 90d & 365d: predicted vs realized M>=4.5 regional counts. Aggregate reliability slope **1.20** (90d 1.07, 365d 1.23). Within 0.7-1.3 -> **no scaling applied**.

| horizon | epoch | predicted | realized | ratio |
|---|---|---|---|---|
| 90d | 2022-01-01 | 2.51 | 4 | 1.60 |
| 90d | 2023-01-01 | 2.50 | 2 | 0.80 |
| 90d | 2024-01-01 | 2.72 | 4 | 1.47 |
| 90d | 2025-01-01 | 2.55 | 1 | 0.39 |
| 365d | 2022-01-01 | 10.30 | 8 | 0.78 |
| 365d | 2023-01-01 | 10.32 | 7 | 0.68 |
| 365d | 2024-01-01 | 10.52 | 11 | 1.05 |
| 365d | 2025-01-01 | 10.46 | 25 | 2.39 |

**Backtest read:** 90d is well-calibrated. On the 365d QUIET epochs (2022, 2023) the near-critical cascade mildly OVER-predicts (ratio ~0.7-0.8, the expected long-horizon inflation), but the 2025 epoch's realized count is huge (the unforecastable Apr-2025 M6.2 sequence, ratio 2.4) and pulls the aggregate to 1.23, inside the gate. So no global scaling is triggered, but the yearly M6 central should be read as an upper-leaning estimate for quiet periods; the b-range low end and the Poisson base rate bracket the realistic floor.

Cascade-only + renewal (the 30-day ML hybrid is not valid at these horizons). Per-sim event cap 50,000 (0% capped); Wilson 95% intervals shown; b-ensemble {1.02, 1.15, 1.54}.

## quarter 90d  (K=10000, capped 0.00%)

| M | cascade P (central) | Wilson 95% | b-range (1.54→1.02) | Poisson base |
|---|---|---|---|---|
| >=5.0 | 48.74% | 47.76–49.72% | 10.44–69.41% | 37.60% |
| >=5.5 | 16.57% | 15.85–17.31% | 1.83–30.97% | 11.82% |
| >=6.0 | 4.52% | 4.13–4.94% | 0.19–10.77% | 1.04% |

- **Renewal** P(characteristic M~7, any segment): **0.351%** (Central Marmara 0.197%).
- **Combined M>=6.8 layer** 1-(1-P_cascade_M6)(1-P_renewal): **4.86%**.

## year 365d  (K=3000, capped 0.00%)

| M | cascade P (central) | Wilson 95% | b-range (1.54→1.02) | Poisson base |
|---|---|---|---|---|
| >=5.0 | 93.17% | 92.21–94.02% | 36.77–98.87% | 85.23% |
| >=5.5 | 53.07% | 51.28–54.85% | 7.80–78.00% | 39.95% |
| >=6.0 | 18.30% | 16.96–19.72% | 1.53–36.73% | 4.16% |

- **Renewal** P(characteristic M~7, any segment): **1.420%** (Central Marmara 0.798%).
- **Combined M>=6.8 layer** 1-(1-P_cascade_M6)(1-P_renewal): **19.46%**.

## How to read this

- The **yearly number ≈ long-term hazard + a current-sequence bump**, expected behavior as the Omori/triggered signal decays over the year, not lost skill.
- **Renewal dominates at M~7** (characteristic ruptures, BPT clock); the **cascade dominates at M<=5.5** (aftershock/background productivity). M6 is the crossover: read cascade and renewal together (the combined line).
- Ranges come from the b-value ensemble (the dominant uncertainty); the central uses the calibrated b_op (the backtest confirmed no global scaling was needed). Treat M6 as order-of-magnitude.