# v2 pre-registration — Amendment 4 (2026-07-15)

Governs alongside the base protocol (`docs/v2_preregistration.md`, SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendments 1–3.
Append-only. **Written and hashed before the analysis it governs was run.**

## Adjudicator of record (unchanged)
The pre-registered count-scored conjunctive rule remains the adjudicator of record for every
model-ranking claim; `results/claims.json` holds the verdicts. Nothing here re-adjudicates it.
Amendment 4 governs two additions that make no ranking claim: (A) a re-run of an existing
battery at the final configuration, and (B) a new descriptive analysis with no comparator.

---

## A. GNSS placebo battery — re-run at the final configuration (no new test)

**Trigger.** The §4 / Figure 4 GNSS placebo numbers were produced at commit `1205e7b`
(v3/phase-V, 2026-07-08). At that commit `results/grid_hybrid_report.json` recorded
`b_op = 1.2` over 261 windows, and `src/marmara/cascade.py` had no branching-ratio-preserving
rescale (`preserve_branching` was added later, for release 1.2.0). The stale battery also
selected the blend weight by naive argmax, whereas the published hybrid uses the pre-registered
1-SE parsimony gate (`train.select_w_1se`). The published numbers are therefore off-configuration
on three axes (b_op, branching, w-selection) at once.

**Amendment.** The battery is re-run **unchanged in protocol** at the final configuration
(`b_op = 1.15`, 262 windows, `preserve_branching=True`, 1-SE gate). Specifically:

1. `N_PERM` is held at the original per-target values (y30 = 25, y35 = 30) so the re-run is
   apples-to-apples; only the configuration changes.
2. The surrogate definitions are unchanged: time-shuffle (per cell, permute feature rows across
   windows), circular-shift (per cell, offset ≥ 24 windows = 2 yr), coverage-only (the four
   physical GNSS features replaced by a single count of stations with ≥ 5 epochs in
   [t0−365 d, t0) within 60 km). Every surrogate draw retrains in full.
3. **Identity gate (new, and binding).** The re-run's `fit_hybrid` replica must reproduce the
   frozen `results/predictions_{y30,y35}.parquet` `hybrid` and `hybrid_gnss` columns bit-for-bit
   before any placebo number is read. If it does not, the run aborts and reports nothing.
4. CIs use the paper's machinery (Politis–Romano stationary block bootstrap over window-ids,
   B = 2000, seed = 42, mean block = 3.0).
5. **The re-run replaces the §4 numbers and Figure 4 whatever it returns.** The verdict rule is
   the one already in force: the channel is genuine only if the real effect exceeds every
   surrogate null on **both** axes (ΔIG and ΔPR-AUC) *and* the coverage-only surrogate does not
   reproduce it. Any other outcome leaves the channel VOID.
6. `coverage_only.reproduces_gain` is defined only when the real ΔIG > 0; where the real effect
   is negative the coverage comparison is reported as **not meaningful**, not as a reproduction.

---

## B. Foreshock false-alarm denominator (new, descriptive)

**Trigger.** §4/§5 state that for the Mw 6.2 Kumburgaz earthquake "useful warning arrived only
36 minutes before rupture", resting on a single event: ten minutes after the lone ML 4.0
(proxy-Mw 4.5) foreshock the epicentral cell's 30-day P(M≥6) gain jumped to 40×. That sentence
reports the one time the statistic fired *before a mainshock*. It does not report how often the
same statistic fires when **no** mainshock follows. Without that denominator the 36-minute claim
is an anecdote, not a result.

**Amendment — pre-specified before running.**

1. **Qualifying trigger.** Every catalogue event with `mag_w ≥ 4.0` whose epicentre lies in the
   model box (lon 25.65–30.85, lat 39.65–41.85). Period: the **full frozen catalogue**
   (2003-01-04 → 2026-07-11). Two pre-registered filters, each reported as a count so the
   attrition is auditable:
   - **base burn-in**: the trigger needs ≥ 365 d of prior catalogue for the long-run base rate
     to be defined (`t ≥ 2004-01-04`);
   - **follow-on observability**: the trigger needs ≥ 30 d of subsequent catalogue for its
     follow-on window to be observable (`t + 30 d ≤ 2026-07-11`).
   The **primary denominator** is the trigger set passing both filters. The **2024-01-01 onward
   subset** is reported separately, as is the unfiltered count.

2. **Alarm statistic.** Identical machinery to the Kumburgaz analysis
   (`marmara.m62_countdown.Case.forecast`), reused unmodified in substance:
   - strictly causal history `< freeze` (causality asserted at every freeze);
   - `cascade_forecast(params, h, freeze, H = 30 d, K = 6000, seed = 42, b = b_op = 1.15,
     per_sim_cap = 50000)`;
   - per-cell `P6 = 1 − exp(−λ35(cell) · 10^(−b·2.5))` (the analytic rare-rate fix, §3);
   - freeze base `base6 = 1 − exp(−(reg35_30d / ncells) · 10^(−b·2.5))`, where `reg35_30d` is the
     regional M≥3.5 count to the freeze, per 30 d, spread uniformly over the box;
   - **gain = P6(cell) / base6**, the quantity §4 reports as "5–8×" and "40×".

   Two freezes per trigger:
   - `gain_pre`  at `t − 1 s` (the immediately-pre-event freeze);
   - `gain_post` at `t + 10 min` (matching the Kumburgaz +10 min freeze);
   - `jump = P6_post / P6_pre`.

3. **Cell.** Primary: the trigger's own cell. Secondary: the maximum gain over cells whose centre
   lies within 25 km of the trigger epicentre (the "25-km neighborhood").

4. **Escalation thresholds.** `gain_post ≥ 10×` and `gain_post ≥ 40×`.
   **Primary definition = the absolute post-trigger gain**, because that is the quantity the
   manuscript quotes ("the gain rose to 40×"). The **jump ratio** `jump ≥ 10× / ≥ 40×` is reported
   as a pre-specified secondary, since "gain at t+10 min vs the immediately-pre-event freeze" also
   admits that reading. Both are reported; neither is chosen after seeing the result.

5. **Follow-on (a "hit").** An event with `mag_w ≥ 6.0` occurring within **30 d** of the trigger
   **and** within the **association radius of 25 km** of the trigger epicentre.

6. **False-alarm rate.** `1 − hits/escalations` per threshold and per definition, with the
   escalation count as the denominator. Reported with exact counts, never as a rate alone.

7. **Alarm duration.** For every escalating trigger the gain is re-evaluated on a forward ladder
   {+1 h, +6 h, +1 d, +3 d, +7 d, +14 d, +30 d}; the alarm duration is the last ladder point at
   which `gain ≥ threshold`, reported as an interval-censored bracket (the ladder is coarse by
   design; we report the bracket, not a point).

8. **No ranking claim.** This analysis has no comparator and adjudicates no model. It is a
   descriptive property of the alarm statistic. It cannot promote or demote any forecaster, and
   it does not touch `claims.json`.

9. **Whatever it returns is reported.** The expectation on record before running is "many alarms,
   one hit" (the catalogue contains exactly one M≥6 in 23 years). A result showing a low
   false-alarm rate would be equally reportable and would strengthen, not weaken, the §5 claim.
