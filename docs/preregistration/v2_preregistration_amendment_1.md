# v2 pre-registration — Amendment 1 (2026-07-13)

Governs alongside the base protocol (`docs/v2_preregistration.md`, SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`). This is an
**append**, not a rewrite. It is committed with the item-B result as its recorded
justification, exactly because the result changes a canonical comparator after test
exposure.

## Trigger

Item B (Mizrahi parity restoration) restored the independent third-party `lmizrahi/etas`
comparator's **native spatially-variable background** — its own EM per-event background
weights (independence probability = 1 − Σ_sources P_ij) piped through the sv-ETAS KDE
machinery (Gaussian kernel, Silverman bandwidth ≈ 12 km, ≥ 5 km floor; 205× spatial
contrast) — replacing the v1 uniform-floor substitute in `etas_modern.py:97`. The base
protocol (§1, §6) did not pin the `modern_etas` background, so this is not covered by it.

## Amendment

The **canonical `modern_etas` (Mizrahi first-generation) comparator now uses the NATIVE
spatially-variable background at the model's own fitted b (native b = 1.76).** The
uniform-floor version is retained as a recorded sensitivity. Canonical `results/claims.json`
is re-emitted with this comparator (the pre-amendment uniform-floor artifacts are preserved
under `results/.canonical_backup/pre_B_amendment/`).

## Recorded justification (`results/mizrahi_2x2.*`)

- **Input parity holds (B0):** the Mizrahi fit is already on the homogenized `mag_w`
  catalogue (14,031 events; mixed *origins* 7646 Md / 6237 ML but homogenized *values*).
  b = 1.76 is an **estimator** gap (`etas.estimate_beta_positive` vs our `b_positive` = 1.54
  on identical data), not a mixed-scale artifact. The M-test handicap is addressable by
  scoring at a common b.
- **2×2 ablation {uniform, native bg} × {native b, b_op}, first-gen vs Mizrahi (block
  bootstrap B=2000, seed 42, unchanged rule):**
  - native bg / native b (canonical): **A_beats_B** on both y30 and y35.
  - native bg / b_op (Mizrahi's MOST FAVORABLE, steel-man): **y30 A_beats_B**, **y35
    inseparable**.
- **The lone Table-2 cell "hybrid beats Mizrahi at b_op" is a handicap artifact:** under the
  native background it is **inseparable** in every configuration.

## Abstract-level flags (raised, NOT rewritten — per governance)

1. The abstract's separation claim ("first-generation beats the independent inversion")
   **survives full parity on the PRIMARY y30 target** (holds even in Mizrahi's most-favorable
   config) but must be **qualified on y35**, where it becomes inseparable once the native
   background is restored and b is harmonized. Recommend restricting the separation claim to
   y30 or stating the y35 softening explicitly.
2. **Under parity, the hybrid beats no physics model** (the lone hybrid > Mizrahi cell
   vanishes) — this strengthens, not weakens, the "ML adds no robust value" conclusion.

## Unchanged

The claim rule, bootstrap machinery (B/seed/mean_block), the surgical-k cascade, the 1-SE
gate, and all other base-protocol items are unchanged. Multiplicity calibration (item G) runs
against this re-emitted claims file.
