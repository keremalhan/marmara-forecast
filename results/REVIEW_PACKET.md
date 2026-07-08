# REVIEW PACKET — v3 verification (local only; NO remote git operations performed)

Branch: `v3-verify` (local commits only). Verification artifacts: `results/verify/`.
This packet is the human gate; nothing leaves the machine without sign-off.

---

## Section V — Independent verification verdicts

### V1–V6 audit results

| audit | verdict | evidence |
|---|---|---|
| **V1** reproduce-from-clean (fresh venv, APFS — NOT exFAT) | **PASS** — all 12 deterministic artifacts reproduce within 1e-9; only `grid_hybrid_report.runtime_s` (wall-clock) differed | `results/verify/REPRO_FAILURES.md` (0 failures) |
| **V2a** gnss_v2 truncated self-test (13 monthly cutoffs + adversarial mid-month) | **PASS** — bit-for-bit (dev 0) everywhere; caching granularity never uses epochs ≥ t0 | `results/verify/v2_truncated.json` |
| **V2b** placebo battery (time-shuffle, circular-shift, coverage) | **C3-FAIL** — time/shift placebos do NOT collapse (IG mean +0.119/+0.088 ≈ real +0.098); coverage IG ≈ 0 | `results/verify/gnss_placebos.json` |
| **V2c** IG bootstrap CI (block, window-ids, B=2000) | **C3-DOWNGRADE** — test IG +0.098, 95% CI **[−0.016, +0.207] includes 0** | `results/verify/v2_gnss_ig.json` |
| **V2d** permutation importance | gain loads on residual-rate, not the availability flag (but within noise) | `results/verify/v2_gnss_ig.json` |
| **V2e** top-20 windows | y30 gain 59.6% concentrated in one window (2024-04, 16 real events) | `results/verify/gnss_top_windows.md` |
| **V2 (y30 operational placebo)** | **SPURIOUS** — real ΔPR +0.033 inside null [−0.003,+0.063]; real ΔIG +1.18 inside null [−0.12,+1.25] | `results/verify/gnss_y30_placebo.json` |
| **V3** bootstrap machinery | **PASS** — block CI 7× wider than row (respects window correlation); pairing gives Δ≡0 for identical ranking; independent 20-line reference reproduces the y30 cascade-vs-hybrid CI exactly | `results/verify/v3_bootstrap_audit.json` |
| **V4** sv-ETAS convergence | **VERIFIED** — first-gen μ(x,y) is non-uniform (spatial CoV 1.27); EM converged (8 iters, LL stable), 5 km bandwidth floor respected, background fraction 0.34 sane → sv≈first-gen is genuine, not EM degeneracy | `results/verify/v4_sv_etas.json`, `results/verify/mu_xy.png` |
| **V5** test-touch audit | **DISCLOSE** — the base hybrid family scored test 3×, all identical-config deterministic reproductions (no test-set tuning); hyperparameters chosen on validation only | `results/verify/v5_test_touch.json`, `results/test_touch_log.json` |
| **V6** claims regeneration | **PASS** — 168 verdicts regenerated from bootstrap_ci.json, **0 mismatches** with the committed claims.json | `results/verify/v6_claims_regen.json` |

### Claim resolution (C1–C6)

| claim | verdict | basis |
|---|---|---|
| **C1** y35: cascade/hybrid/all-ETAS statistically inseparable; all beat non-clustering baselines | **VERIFIED** | V1 (bootstrap_ci reproduces 1e-9) + V3 (machinery correct) + V6 (verdicts faithful) |
| **C2** y30 (powered): physics (cascade/ETAS) decisively beats the ML hybrid | **VERIFIED** | V1 + V3 + V6 (hybrid_vs_{cascade,sv,modern,firstgen} = B_beats_A, CIs exclude 0) |
| **C3** GNSS v2 carries genuine deformation signal / resolvable at y30 | **VOID** | V2b placebos do not collapse + V2c CI includes 0 + y30 operational placebo spurious → `results/verify/gnss_verdict.md`. Replaced by a rigorous, feature-engineering-robust NULL that vindicates the v1 GNSS null. |
| **C4** independent Mizrahi ETAS is competitive, not a strawman | **VERIFIED** | V1 (evaluation reproduces: modern_etas has the highest y35 ROC-AUC 0.894 / Molchan 0.785; a factual comparison) |
| **C5** sv-ETAS ≈ first-gen because the first-gen background was already spatially variable & converged | **VERIFIED** | V4 (first-gen CoV 1.27 non-uniform; EM converged; no degeneracy) |
| **C6** CSEP: cascade/sv number+magnitude-consistent; Mizrahi first-gen under-predicts | **VERIFIED (real pyCSEP)** — genuine pyCSEP 0.8.0 N/M agree with in-house for all 3 models (Section C) | V1 + Phase C (`results/csep_v3/`, cross-check all `agree_NM:true`) |

**Gate status: no item UNRESOLVED.** C3 is VOID (resolved, with the null substituted
everywhere it appeared: PAPER_DELTAS §4, gnss_v2_decision.json, gnss_verdict.md).
Phases D / C / M may proceed.

### What the human should eyeball
- `results/verify/mu_xy.png` — first-gen vs sv-ETAS μ(x,y) (should be visibly non-uniform).
- `results/verify/gnss_placebos.json` + `gnss_y30_placebo.json` — the placebo non-collapse that voids C3.
- `results/verify/gnss_top_windows.md` — the 59.6% single-window concentration.
- `results/verify/gnss_verdict.md` — the consolidated C3 = VOID argument.
- `results/verify/REPRO_FAILURES.md` — empty (V1 clean).

---

## Section D — starving-channel outcomes

**Outcome: 2 non-catalog channels tested under the gate (gnss_v2, dense_catalog); 0
promoted. 1 channel still-absent with fetch instructions (repeaters). EPOS is
velocities-only (used for cross-validation, cannot be ingested).** "More data" does
not rescue the ML — reinforcing the y30 finding that scarcity is not the binding
constraint.

| channel | data | gate result | decision | evidence |
|---|---|---|---|---|
| **dense_catalog** (D1) | AFAD bulletin M≥1.0, 2008–2026, model box: 36,098 events → homogenized, deduped, blast-filtered, per-year Mc → **10,610** (2,276 AFAD-unique below KOERI) | leakage-pass (corr 0.091); truncated self-test **bit-for-bit**; **val IG −0.030, test IG +0.130 (95% CI [0.045, 0.229] excludes 0)** | **NOT PROMOTED** — fails the pre-registered rule (val > +0.02 AND test > 0): val IG is negative. The test IG is significant but the **val/test sign disagreement** (likely AFAD completeness non-stationarity across the val→test boundary) makes it untrustworthy. A longer validation window or per-year-Mc-normalized features could revisit. | `results/source_ig_dense_catalog.json`, `results/dense_build_report.json`, `results/verify/d1_dense_verdict.json` |
| **repeating_eq** (D2) | still-absent | not run | **STILL-ABSENT** — supplements are paywalled journal tables, not sandbox-fetchable. Exact DOIs/format in the fetch instructions; report as *absent, not null*. | `data/REPEATERS_FETCH_INSTRUCTIONS.md`, `results/source_ig_repeating_eq.json` (available:false) |
| **EPOS/MIDAS velocities** (D3) | on disk (40 INGV MIDAS records) | cross-validation only | **CANNOT INGEST** (velocities-only, no daily epochs). Used to cross-validate gnss_v2's secular fits: after removing the ~24 mm/yr IGS20-vs-Eurasia frame offset, residuals are **< 1 mm/yr** at all 4 matched stations → the gnss_v2 trajectory model is sound, so C3's null is not a secular-fitting artifact. | `results/verify/d3_epos_crossval.json` |

**D4:** no channel cleared the gate → no full model rerun required. Manuscript
source-count statement: *"Two engineered non-catalog channels (GNSS deformation,
dense sub-Mc3 microseismicity) were tested under the pre-registered gate; neither was
promoted (GNSS void per the placebo battery; dense fails the validation criterion).
A repeater channel was designed but its catalogue is absent."*

## Section C — real pyCSEP

**Outcome: genuine pyCSEP 0.8.0 was installed and run (catalog-based N/M/S/PL) and
CONFIRMS the in-house result exactly — the number- and magnitude-consistency verdicts
agree for all three models. C6 is upgraded from "VERIFIED (in-house)" to "VERIFIED
(real pyCSEP)".**

Environment: pyCSEP does NOT import on the exFAT volume (its bundled `matplotlibrc` is
corrupted byte 0xb0 by the copy-mode install). It imports and runs cleanly in a fresh
**APFS** venv (`/Users/keremalhan/venv-csep`, py3.12, pyCSEP 0.8.0). The exFAT/APFS
filesystem issue — not pyCSEP — was the v1 blocker.

Inputs: `scripts/csep_v3_prep.py` (main env) replays `marmara.csep_eval`'s EXACT
`default_rng(42)` sequence to expand each model's per-cell Poisson draws into
event-level stochastic catalogues (1000/model, events at cell centres, GR magnitudes
at bin centres), so the pyCSEP inputs are bit-identical in count/magnitude to the
in-house ones. Driver: `scripts/csep_v3_run.py` (APFS venv) builds pyCSEP
`CSEPCatalog`/`CatalogForecast` objects over the 1219-cell region and runs
`catalog_evaluations.{number,magnitude,spatial,pseudolikelihood}_test`.

Target M≥3.0 (y30), test period 2024-01-22..2026-03-12, 1383 observed events, 26 windows.

| model | N-test (pyCSEP) | M-test (pyCSEP) | S-test | PL-test | in-house N/M agree? |
|---|---|---|---|---|---|
| **cascade** | **PASS** (min δ 0.030; obs 1383 vs fcast 1457) | **PASS** (γ 0.067) | FAIL (γ 0.0) | FAIL (γ 0.0) | **YES** |
| **sv_etas** | **PASS** (min δ 0.050; obs 1383 vs fcast 1446) | **PASS** (γ 0.076) | FAIL (γ 0.0) | FAIL (γ 0.0) | **YES** |
| **modern_etas** | **FAIL** (min δ 0.0; obs 1383 vs fcast 783 — under-counts) | **FAIL** (γ 0.0) | FAIL (γ 0.0) | FAIL (γ 0.0) | **YES** |

- **N and M are the robust, citable results.** pyCSEP reproduces the in-house verdicts
  exactly: cascade and sv_etas are number- and magnitude-consistent with observed M≥3.0
  seismicity; the independent Mizrahi first-gen (`modern_etas`) under-predicts the count.
- **S-test and PL-test reject every model (γ→0)** — e.g. cascade PL obs 1718 vs simulated
  mean 165. This is the **Poisson-catalogue under-dispersion confound** (spec Phase C option
  **b**, limitation stated): the stochastic catalogues are cell-independent draws from the
  gridded rate, so the observed clustered seismicity's spatial/rate pseudo-likelihood is far
  more extreme than any realisation — a property of the catalogue approximation, not evidence
  against the spatial forecast. The preferred fix (native clustered catalogues) is left as
  future work: emitting per-event catalogues from the cascade simulator would require touching
  the hash-chained, V1-reproduced `grid_hybrid` reproduction path, which is protected.
- Evidence: `results/csep_v3/{csep_v3_results.json, csep_summary.md, csep_v3_consistency.png}`;
  inputs `results/csep_v3/inputs/`; in-house cross-check `results/csep/csep_results.json`.

### What the human should eyeball
- `results/csep_v3/csep_v3_consistency.png` — green N/M for cascade & sv_etas, red for
  modern_etas; all S/PL red (the documented confound).
- `results/csep_v3/csep_v3_results.json` → `cross_check_vs_inhouse` (all `agree_NM: true`).

## Section M — manuscript package

**Outcome: a self-contained rewrite package `manuscript_v2/` (M1–M5) that updates
`paper/paper_seismica.md` to the verified v2/v3 findings, with every ranking traced to
`claims.json`. Two headline figures generated. No prose states a ranking the bootstrap
referee does not license.**

| item | file | content |
|---|---|---|
| **M1** claims map | `manuscript_v2/01_claims_map.md` | C1–C6 + full y30/y35 verdict matrix → verdict → evidence, verbatim from `claims.json`; the "significant-but-void" GNSS trap spelled out |
| **M2** sections | `manuscript_v2/02_sections.md` | drop-in Abstract; new Methods (block bootstrap, sv-ETAS+Mizrahi, GNSS+placebo, pyCSEP); rewritten Results §4.1–4.4 |
| **M3** narrative | `manuscript_v2/03_narrative.md` | three-act arc; foreground/soften; reviewer-proofing |
| **M4** figures | `manuscript_v2/04_figures.md` | manifest: fig1–4 retained + fig5–8 new (all files on disk) |
| **M5** limitations | `manuscript_v2/05_limitations.md` | drop-in §6, 10 items ordered by claim impact |
| — index | `manuscript_v2/00_README.md` | integration map into `paper_seismica.md` + source-of-truth chain |

**Key manuscript deltas from v1**
- Primary target changed **M≥3.5 → M≥3.0** (powered; 592 vs 167 test positives).
- New headline (**C2**): at y30 the physics models (cascade/sv-ETAS/Mizrahi/first-gen) form
  an inseparable cluster that **beats the ML hybrid** on both axes (cascade PR-AUC 0.229 vs
  0.146; IG deficits 0.88–1.46 nats). v1's "cascade ranks best at y35" is demoted to
  "one of an inseparable family."
- GNSS reported as a **null** with the placebo lesson (**C3** void) — the methodological spine.
- Modern baselines (sv-ETAS, Mizrahi) added so the loss is not to a strawman (**C4/C5**).
- CSEP consistency confirmed with **genuine pyCSEP** (**C6**).

**New figures generated** (`scripts/figs_v2.py`): `results/figs_v2/fig5_y30_forest.png`
(C2 forest, physics vs hybrid) and `fig6_gnss_placebo.png` (C3-void placebo nulls). Plus
existing `results/verify/mu_xy.png` (C5) and `results/csep_v3/csep_v3_consistency.png` (C6).

### What the human should eyeball
- `manuscript_v2/00_README.md` first (integration map), then `01_claims_map.md` (the referee).
- `results/figs_v2/fig5_y30_forest.png` — the C2 result at a glance.
- `results/figs_v2/fig6_gnss_placebo.png` — why the GNSS "win" is void.

---

## FINAL GATE STATUS (v3 sign-off)

| phase | status | commit |
|---|---|---|
| **V** verification (V1–V6) | complete — no item UNRESOLVED; C3 voided | (v3/phase-V) |
| **D** starving channels | complete — 0 channels promoted; 1 absent | `07a4746` |
| **C** real pyCSEP | complete — N/M agree with in-house, all 3 models | `4f81c27` |
| **M** manuscript package | complete — `manuscript_v2/` M1–M5 + figs | (this commit) |

- **Reproduction gate (every phase):** leakage self-test PASS (all feature deviations
  0.000e+00, no corr>0.999); pytest **22 passed / 2 skipped**. exFAT AppleDouble sidecars
  (`._*`) neutralised by `conftest.py` collect_ignore (verified with a planted probe).
- **Protected paths untouched:** `results/prospective/` hash-chain, `prospective_monthly.sh`,
  KOERI corrections in `catalog.py`, `test_grid_leakage.py` assertions (only additive guards).
- **Git:** all work on branch `v3-verify`, **local commits only — no push/fetch/pull performed**
  (per instruction: working locally). Nothing has left the machine.

**This packet is the human gate. Sign off before any external release.**
