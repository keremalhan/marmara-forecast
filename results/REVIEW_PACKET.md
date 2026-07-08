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
| **C6** CSEP: cascade/sv number+magnitude-consistent; Mizrahi first-gen under-predicts | **VERIFIED (in-house)** — real-pyCSEP confirmation is Phase C | V1 (csep_results reproduces) |

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
_(pending — Phase D)_

## Section C — real pyCSEP
_(pending — Phase C)_

## Section M — manuscript package
_(pending — Phase M)_
