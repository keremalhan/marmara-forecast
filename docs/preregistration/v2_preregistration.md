# marmara-forecast v2 pre-registration

**Status: committed BEFORE any v2 re-scoring of the held-out test set.**

v2 changes the canonical model *after* the test set was already scored in v1. That is a
legitimate revision (the v1 cascade carried a supercriticality bug; see §1), but it is
also exactly the situation in which post-hoc choices masquerade as pre-planned ones. This
document freezes every rate-touching configuration choice and every analysis rule for v2
before the test set is re-scored. Its SHA-256 is recorded in
`results/v2_preregistration.json` and referenced from the append-only forecast log
(`results/prospective/forecast_log.jsonl`), so the commitment device and the live products
point at each other. Committed 2026-07-12.

Diagnostic artifacts this protocol is built on (pre-test, background/quiet-window only):
`results/ntest_attribution.*`, `results/ntest_residual_probe.*`, `results/em_background_probe.*`.

---

## 1. Canonical cascade configuration

The v1 cascade drew offspring magnitudes at the operational `b_op` but kept the productivity
`k` fitted at `b=1.542`, so the realized mmax-truncated branching ratio at `b_op=1.15` was
**n = 1.212 (supercritical)**, not the fitted 0.95: the cascade exploded and was bounded only
by the per-sim / max-events cap. Canonical v2 uses the **surgical** fix:

- **Real-history parents keep the fitted `k`.** Their magnitudes are observed (e.g. Kumburgaz
  Mw 6.2), so their expected direct-offspring count `kappa(m)=k*exp(alpha*(m-mc))` is the MLE
  estimate and is independent of `b`; no rescale is licensed.
- **Simulated parents (magnitudes drawn at `b_op`) are damped to `k_sim`**, holding the
  simulated sub-cascade's mmax-truncated branching ratio at the fitted **n = 0.95**.

Fixed constants: stationary first-generation ETAS background (`mu_total` + declustered KDE);
`b_op = 1.15`; `K_backtest = 500`; per-window seed `1000 + k`; `mmax = 7.6`; `per_sim_cap` and
`max_events` as in v1. Implementation: `cascade.py::simulate_window` with
`preserve_branching=True`; `preserve_branching=False` reproduces the v1 (supercritical) run
bit-for-bit for the attribution table.

**The sv-ETAS EM-declustered background is a REPORTED SENSITIVITY, not the canonical
background.** Probed 2026-07-12 (`results/em_background_probe.*`): swapping it in barely moves
the split table (quiet over-prediction 1.47 -> 1.45), because the quiet-period over-prediction
is a *temporal* non-stationarity that no stationary background (however declustered) can fix.
Stationary mu stays canonical; the EM row is a sensitivity.

## 2. Table 2 (operational-b sweep) — the headline, committed blind

Re-run the full **{1.10, 1.15, 1.20}** `b_op` sweep under the canonical (surgical, n=0.95-pinned)
cascade, reporting **both hybrids**: naive-argmax-`w` (a diagnostic retained for continuity with
v1) and the 1-SE-`w` (canonical, §3).

Committed rationale: in v1, sweeping `b_op` at fixed `k=0.579` also swept the cascade's
*supercriticality* (n rising as `b_op` falls: ~1.1 at 1.20, ~1.2 at 1.15, higher at 1.10), so the
`lam_sim` feature the hybrid ingests varied partly through cap-truncation pathology — the v1
fragility finding may be **partly bug-mediated**. The surgical fix pins n=0.95 at every `b_op`,
removing that confound. The v2 fragility verdict (softens / persists / unchanged) is **disclosed
whichever way it lands**; a softening is a cleaner mechanism story, not a loss. This outcome is
unknown at commit time, which is why it is fixed here rather than after scoring.

## 3. 1-SE parsimony gate on the hybrid weight

Hybrid weight `w` = the **smallest `w` within one bootstrap SE of the validation-Poisson-LL
argmax** (a flat validation objective therefore drives `w -> 0`, collapsing the hybrid to its
physical core, the cascade). Frozen details:

- `w`-grid unchanged from v1: {0.0, 0.1, ..., 1.0}.
- SE via the **same Politis-Romano stationary bootstrap** as the claims machinery (B, seed 42,
  mean_block 3.0) on the **validation** windows.
- Rule applied **uniformly** to y30, y35, and wide-box y45.
- Endpoint designations (y30 primary/powered; y35 powered; y45 unpowered, no ranking claims)
  and backtest `K = 500` unchanged.
- Committed before test scoring. The rule is named the **parsimony gate**: when validation
  cannot distinguish the hybrid from its physical core at one SE, ship the core.

Committed prediction (blind): the flat validation objective drives `w -> 0`, the hybrid collapses
to the cascade, and the fragility of §2 vanishes — disclosed whichever way it lands. Also
back-tested on model-box y45 (if 1-SE would have driven `w -> 0` there, it retroactively prevents
the manually-diagnosed overfit).

## 4. Completeness / observed count

Raw on-grid **M>=3.0** count is the **primary** observation. The GR-extrapolation completeness
correction (anchored on the completeness-unambiguous [3.45, 4.5) band; the 2024+ test period is
entirely ML/MW, deficit ~45, corrected ~1428 with a b-uncertainty band) is a **footnoted
sensitivity**, not the primary observation. **M>=3.5 is the completeness-unambiguous anchor.**

## 5. N-test reporting rule

Report `delta1`/`delta2` as numbers; the **time/space split table is the primary N-test result**.
The narrow total miss (surgical 1306 vs raw 1383, `delta1=0.018`) is reported as a number and
characterized by the splits — **never as "passes."** The finding is the two-sided misallocation:
quiet/background over-prediction (~1.45x) and Kumburgaz aftershock-zone under-prediction (~0.5x),
which partially cancel in the total. Its two identified causes — temporal background
non-stationarity and the aftershock productivity ceiling (the 0.95 cap lever is nearly exhausted,
n 0.95 -> 0.999 buys only +32 events) — are named as the binding constraints.

## 6. Mizrahi re-integration ladder

- **Input parity:** refit the Mizrahi inversion on the SAME homogenized `mag_w` catalogue.
- **Native background restored:** its own EM/declustering event weights piped through the
  sv-ETAS KDE machinery (Gaussian kernel, Silverman bandwidth, 5 km floor), replacing the v1
  uniform-floor substitute.
- **Magnitude law harmonized at scoring:** score at BOTH its native fitted `b` AND the common
  `b_op` (added as a Table 2 row).
- **Ablation ladder:** the 2x2 {uniform, native background} x {native b, b_op}, all through the
  shared scoring + block-bootstrap claims machinery.
- **Steel-man reporting rule:** the abstract may claim only what holds in Mizrahi's **most
  favorable** configuration. How the restored comparator is reported is fixed here, before its
  v2 scores are seen.

## 7. CSEP v2 method

Catalog-based consistency tests fed with **native clustered cascade simulations** for the cascade
and sv-ETAS (not Poisson-sampled rate fields). Poisson-sampling is retained **only** for the
first-generation intensity and disclosed as such. Magnitudes drawn from the **operational
b-ensemble** (per-member M-test), not the ad-hoc fixed `b=1.2`; spatial/temporal kernel exponents
unchanged. Both genuine pyCSEP and the in-house cross-check are reported. Frozen before the v2
S-test is seen: we predict it will independently rediscover the §5 spatial misallocation, so the
confirmatory test must not be chosen after the finding.

## 8. Claim rule — unchanged from v1

"A beats B iff the 95% percentile bootstrap CI of the paired difference excludes 0 in A's favour
for BOTH information gain (nats/event) AND PR-AUC; otherwise the pair is inseparable." Politis-
Romano stationary bootstrap over ordered window-ids, B=2000, seed=42, mean_block=3.0 (all 1219
cells per window kept together). **Not modified for v2.** Multiplicity (workstream #8) calibrates
an empirical family-wise rate on negative-control pairs; it does not alter this rule.

## 9. Pipeline transition / live products

The surgical fix shifts the analytic rare-rate map, so the live products (30-day P(M>=6), v1
central 1.61%) re-issue under v2. This is logged as a **single** pipeline-transition entry in the
append-only `results/prospective/forecast_log.jsonl`, referencing this protocol's SHA-256; prior
entries are untouched. This is the log's one sanctioned discontinuity.

---

*Execution order committed here:* pre-registration (this document, hashed) -> 1-SE gate code ->
one bundled full pipeline re-run (surgical-k + 1-SE gate; EM background a cheap sensitivity row)
-> Mizrahi ladder (§6) and the KOERI forensics note in the parallel lane -> CSEP native
catalogues (§7) once v2 rates exist -> Sindirgi wide-box -> multiplicity last.
