# v2 pre-registration — Amendment 7 (2026-07-16) — release 1.3.0

## What this is, and the honesty ledger inside it

A **prospectively timestamped post-review analysis amendment**, written before the analyses it
specifies are run, archived with Amendments 1–6. It does not restore an untouched test set; the
exploratory exposure disclosed in §3 stands. This is the fourth campaign against that test set.

**Not everything below is a blind prediction, and the document says which is which.** Amendment 6's
predictions were falsified and that is recorded here. The deconfounding arithmetic and the test-era
split were computed *before* this amendment was written — they are reported as established results,
not lodged as predictions. Only the Table 3 arms are genuinely prospective. Labelling settled
arithmetic as a "prediction" would be theatre, and this project has spent four campaigns removing
exactly that.

Governs alongside `docs/v2_preregistration.md` (SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendments 1–6.

---

## 1. Forensics: what was found, in order

1. **The stored b_op sweep is supercritical.** `etas_fit_report.b_calibration` (Table S6) reproduces
   exactly under `preserve_branching=False` (pred 1340.8 / 1222.5, slopes 0.9221 / 1.0131 at
   b = 1.10 / 1.15). The v1.2.0 fix modified `cascade.py` and `grid_hybrid.py` but not
   `calibrate_b.py`, so nothing forced a re-run and `grid_hybrid` read the stale value through
   `load_b_op()`. Control: at b = 1.542 ≈ `params.b` the rescale is a near-no-op (factor 0.9997) and
   the simulators agree to 9.1e-05 in slope.
2. **Scope is bounded.** The shipped `grid_hybrid.parquet` **is** `preserve_branching=True`
   (bit-exact against a pb=True re-simulation, windows 100 and 248). The simulator was never
   supercritical downstream. **Only the b_op value inherits the stale sweep.**
3. **Re-running the corrected procedure does not yield one number.** It yields
   **1.05** on the registered `[::3]` subsample and **1.00** on all 231 pre-test windows: the argmin
   moves a full grid step with the window set, against a seed sd of 0.002. The procedure is a precise
   measurement of a contaminated quantity.
4. **Amendment 6's two diagnostics, both falsified.** D1 (era-split, predicted ≈1.12–1.15) returned
   the list maximum, **1.542**; D2 (M≥4.0 threshold-split, predicted ≈1.3) returned **1.20**, an
   artifact of a candidate-list gap (the true crossing interpolates to ≈1.38). Amendment 6's decision
   rule fired on D1's 1.542 and was **not executed**: the rule's premise — that an era restriction
   isolates the magnitude law — is false.
5. **Why D1 is confounded, measured.** On modern-era windows the cascade over-predicts the
   **M≥3.0** count by 1.18×, and that quantity is *b-independent* (the branching-preserving rescale
   pins the total at n = 0.95 regardless of b; b only tilts the distribution above mc). The era
   restriction changes the rate baseline, so the sweep drives b to the grid boundary trying to
   absorb a rate error with a magnitude-law parameter. This is **§5's stationary-productivity
   failure** — the same 1.47× pre-mainshock over-prediction already in print — reaching into the
   calibration. The two defects are one defect.

## 2. The deconfounding (established, not predicted)

Dividing the ≥M slope by the b-independent ≥3.0 slope cancels the rate error — and, algebraically,
the cascade itself: `slope_M(b)/slope_3.0 = (real_M/real_3.0)·10^{b(M−3.0)}`, so the crossing is
`b = log10(real_3.0/real_M)/(M−3.0)`. **The deconfounded calibration is the catalogue's own effective
slope above the counting threshold.** Three estimates — different eras, thresholds, methods:

| estimate | b |
|---|---|
| modern-era windows, 3.0→3.5 | **1.365** |
| all pre-test windows, 3.0→4.0 | **1.405** |
| whole catalogue, 3.5→4.5 | **1.418** |

They converge on **≈1.37–1.42**. The variant table therefore decomposes rather than scatters:
**1.00 is the Md conversion pile; ≥1.54 is the rate error hitting the grid boundary; 1.37–1.42 is the
magnitude law above the artifact; 1.17 is the test era's near-threshold slope.**

**And 1.17 is itself artifacted** (also established before this was written): split at the Kumburgaz
mainshock, the test-era slope is **1.280 pre** and **1.108 post** — sequence-local incompleteness
depletes 3.0–3.5 counts after the mainshock and lowers the apparent slope. So the shipped b_op's
apparent agreement with the test era is agreement with an STAI-depleted number, reached by two bugs
cancelling (`preserve_branching=False` inflated predicted counts ~14%; `[::3]` under-realized ~11%).

## 3. Decision: no product rebuild; b_op = 1.15 retained as **convention**, not calibration

**The registered procedure is demonstrably unidentified.** Its output tracks era and threshold across
[1.00, 1.54] for reasons orthogonal to the magnitude law. When a procedure is unidentified, the
scientific output is *that demonstration*, not a fourth number drawn from it. Every single-b rebuild
would enshrine one artifact: 1.00 bakes in the pile; 1.38 tilts the simulator against a test era
whose effective slope is ~1.17, degrading the flagship y35 calibration and risking the M-test pass;
1.15 changes nothing but its justification.

Therefore: **b_op = 1.15 is retained as the registered operational configuration and the word
"calibrated" is struck.** Every evaluation of record was conducted at 1.15 before any of this was
known; those numbers are true statements about that configuration. Only the selection story was
false, and the selection story is what gets rewritten.

**Three things are forbidden here, and named so they cannot creep back in:**
* **Do not redefine the b-ensemble around b_op.** That is selecting the range after seeing the value
  — the move refused for the reliability-slope acceptance bands.
* **Do not justify 1.15 by the test-era slope.** That is test-set calibration in disguise. The
  coincidence is reported as *observed, not selected*, and flagged as agreement with an
  STAI-artifacted quantity.
* **Do not full-rebuild to tidy the story.** The story is a mixture catalogue refusing to yield one
  number; manufacturing a tidy artifact is the failure mode this paper exists to catch.

## 4. What is genuinely prospective: the Table 3 arms

**Specified now, unrun.** Extend Table 3's b arms to **b = 1.00 and b = 1.40**, bracketing the whole
contested span [1.00, 1.40] — the pile's answer and the deconfounded law. Machinery is the existing
b-arm path (`grid_hybrid` → `etas_rates` → `train` → `bootstrap`), cascade re-simulations only,
w re-selected by the registered 1-SE rule, no other hyperparameter moved. Frozen artifacts are backed
up and restored, with checksum verification.

**Prediction lodged, blind:** every paired hybrid-vs-physics and physics-vs-physics verdict is
unchanged at both new arms. The thesis is a set of paired comparisons at common b_op; it should not
move. **If a verdict does move, that is reported and the conclusion is revisited** — no gate, no
preferred outcome.

Extend the Bernoulli K = 2000 companion to the new arms if affordable; if not, say so.

## 5. The rewrite this authorizes

* **§3**: the evaluation history gains the fourth campaign. The calibration paragraph states the
  sequence — stale sweep selected 1.15 at slope 1.01; the corrected simulator returns slope 1.152
  there and argmin 1.00; deconfounding shows the argmin is the pile; **1.15 retained as convention**,
  not calibration — and states that the objective conflated the paper's own two named defects.
* **Table S6** becomes the forensic table: all variants (stale/`[::3]`, corrected-full, era-split,
  threshold-split) with one column of deconfounded values.
* **§6**: the M≥4.0 repair upgrades from "noted without commitment" to executed — it returns ≈1.38,
  repairs the pile but not the non-stationarity; the real repair remains K(**x**) and μ(**x**, t),
  now motivated from a third direction.
* **§4/§5 live products**: lead with the ensemble; the central is labelled a convention. Note that
  the deconfounded law (≈1.4) implies a *lower* defensible central M≥6 than published — which
  strengthens the responsible-communication section.
* **Blast radius, stated**: rare-event products and the magnitude law's honesty. **Not** the
  ML-versus-ETAS result, which is paired at common b_op and stable across the contested span.

## 6. Mechanics

`reproduce-all` diff target (this episode is its fourth justification); S7 changelog; Table S8
addendum; both claims files; Zenodo 1.3.0; PDF read front to back; submit.
