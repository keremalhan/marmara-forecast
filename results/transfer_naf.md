# NAF-wide transfer: documented blocker + within-region substitute

**Goal (design 3.3):** build a grid over the full North Anatolian Fault
(26–42°E, 38.5–42°N) with *identical* features, train the ML on NAF-excluding-Marmara
(same date splits), test on the existing Marmara grid, to test whether **positive
scarcity** is the ML's binding constraint.

## Blocker: not executable with the available data

Two independent, verified obstacles:

1. **The catalogue does not cover the NAF.** `data/koeri_events.csv` spans only
   **lon 25.0–31.5°E, lat 39.0–42.5°N** (Marmara + a thin strip east), 2003–2026.
   Events **east of the 30.9°E model box: 2,139 total, only 36 at M≥3.5.** A
   "NAF-excluding-Marmara" training region would therefore contain ≈0 target
   positives; no trainable transfer set exists. The design assumed a *new
   full-NAF fetch*; the repo ships only the Marmara KOERI bulletin, and no
   NAF-wide fetcher/source is available here.

2. **5 of the 19 features are Marmara-only geometry.** `dist_fault_km`,
   `dcfs_perm`, `dcfs_decay25` need the Marmara fault model
   (`segment_properties.json`, 97 segments, Marmara only); `strain_inv` comes from
   `marmara_strain_grid.npz` which covers only **25.6–30.9°E / 39.6–41.9°N**; and the
   `etas_rate` background (`BackgroundField`) is a Marmara-declustered KDE. Building
   the "identical features" over the NAF would require a NAF-wide fault model and a
   NAF-wide geodetic strain inversion; neither exists in the repo and both are out
   of scope for this upgrade.

Fetching a full-Turkey AFAD/KOERI bulletin **and** constructing a NAF fault +
strain model is the prerequisite; it is logged here as the blocker rather than
faked with unavailable data.

## Within-region substitute (y30): partial answer to the same question

The transfer's purpose is to test whether **positive scarcity** limits the ML. The
y30 target (M≥3.0; ~3.5× the positive *cells* of y35, 592 vs 167 on the test
split) is a within-Marmara test of exactly that: *given many more positives, does
the ML close the gap to ETAS?*

**It does not.** On y30 test (read against `claims.json` for the bootstrap
verdicts), the physics-based models lead the ML hybrid on PR-AUC:

| predictor | y30 test PR-AUC |
|---|---|
| cascade | 0.229 |
| sv_etas | 0.228 |
| firstgen_etas | 0.223 |
| modern_etas | 0.206 |
| hybrid_gnss | 0.179 |
| hybrid | 0.146 |
| smoothed | 0.125 |
| poisson | 0.124 |

With ~3.5× more positives the ML hybrid (0.15–0.18) is **still below** the pure
cascade/ETAS (0.21–0.23). This is evidence that positive scarcity is **not** the
sole binding constraint: the ETAS clustering physics captures the predictable
structure that the ML blend does not recover, even when positives are plentiful.
(A true NAF cross-region transfer would strengthen or qualify this; see the blocker
above for why it cannot be run here.)
