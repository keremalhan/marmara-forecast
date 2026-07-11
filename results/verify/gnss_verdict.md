# C3: GNSS trajectory channel: VERIFICATION VERDICT = **VOID** (rigorous null)

Claim C3 (from this work): "the trajectory-modelled, step-corrected, common-mode-
filtered gnss_traj channel carries GENUINE deformation information (+0.061 deconfounded
test IG; operationally resolvable at y30)."

**The forensic audit refutes it. The channel adds no statistically-supported skill.**

## Evidence

| audit | result | file |
|---|---|---|
| V2a truncated self-test (13 monthly cutoffs + adversarial mid-month) | **bit-for-bit** (dev 0), causality is clean, no leakage | gnss_truncated.json |
| V2c IG bootstrap CI (stationary block, window-ids, B=2000) | test IG +0.098, **95% CI [−0.016, +0.207] INCLUDES 0** | gnss_ig.json |
| V2b time-shuffle placebo (30 perms) | placebo IG mean **+0.119** ≈ real +0.098, **does NOT collapse** | gnss_placebos.json |
| V2b circular-shift placebo (≥2 yr, 30 perms) | placebo IG mean **+0.088** ≈ real, **does NOT collapse** | gnss_placebos.json |
| V2b coverage-only placebo | test IG **~0**: the gain is NOT a station-availability artifact | gnss_placebos.json |
| V2d permutation importance | loads on residual-rate, not the fallback flag, but this is within the noise | gnss_ig.json |
| V2e top-window concentration | 59.6% of the y30 gain in ONE window (2024-04) | gnss_top_windows.md |
| y30 operational placebo (25 perms) | real ΔPR +0.033 **inside** placebo null [−0.003, +0.063]; real ΔIG +1.18 **inside** null [−0.12, +1.25] → **SPURIOUS** | gnss_y30_placebo.json |

## Interpretation

The IG statistic for this channel is dominated by noise (its bootstrap CI spans
−0.016 to +0.207). Time-shuffling and circular-shifting the GNSS residuals, which
destroy any alignment with seismicity while preserving spatial support / marginal /
autocorrelation, produce IGs (and y30 ΔPR-AUC / ΔIG) whose distributions CONTAIN the
observed values. A genuine time-varying deformation signal would collapse under these
placebos; it does not. The Phase-2 "+0.061 deconfounded genuine signal" rested on a
single lucky placebo draw (0.037); over 30 draws the placebo mean matches the real IG.
The Phase-3 "resolvable at y30" result likewise lies inside its time-shuffle null.

The gain is therefore **overfitting to non-time-aligned structure**, not deformation.
Causality is clean (V2a) and it is not an availability artifact (coverage IG ~0), but
there is no statistically-supported skill.

## Consequence

- **C3 = VOID.** Everywhere it appears, replace "genuine GNSS signal / resolvable at
  y30" with: *"a properly-engineered GNSS channel (per-component trajectory model,
  step + common-mode correction) still yields NO statistically-supported skill; the
  apparent gains fail time-shuffle and circular-shift placebos and the IG's 95% CI
  includes 0. This strengthens the null to feature-engineering-robust and vindicates
  the earlier static-strain null."*
- `hybrid_gnss` is retained in the tables ONLY as a documented negative result; the
  operational forecaster should use `hybrid` (no GNSS). The physical reading stands:
  onshore GNSS resolves the offshore Main Marmara Fault too weakly to help.
- The GNSS *methods* (trajectory model, steps, common-mode, placebo battery) remain a
  contribution, as the rigorous way the null was established.
