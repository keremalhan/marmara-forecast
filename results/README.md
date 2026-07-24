# results/ — artifact tree

Generated artifacts only: scripts write them, the claims files and the manuscript cite
them, and `scripts/release/reproduce_all.py` asserts them. The tree is organized by
pipeline stage. Where a `.md` sits next to a `.json`, the `.md` is the generated
human-readable companion of the same result. `claims.json` stays at the root: it is the registered count-scored
adjudicator of record (168 pairwise verdicts) that licenses every ranking in the paper.

Frozen paths: `round3/`, `round4/`, `round5/` are named in the hashed pre-registration
chain (`docs/preregistration/`) and by the claims files — never rename or move them or
their contents.

## Layout

| Directory | Contents | Paper |
|---|---|---|
| `catalog/` | Homogenized KOERI catalog (`catalog.csv`, wide box, report), UTC anchor cross-check, Mc = 3.65 refit deltas | §2 |
| `etas/` | In-house ETAS fit (params + report), sv-ETAS, independent `etas`-package inversion + its native background, Brownian-Passage-Time renewal report | §3, Table S11 |
| `grid/` | Feature grids (19 causal features; hybrid adds `ln_lam_sim`), per-cell intensities, per-target predictions, wide-box M≥4.5 | §3 |
| `scoring/` | Benchmark evaluation (Table 2), block-bootstrap intervals, operational-b verdict grid (Table S25), negative-control calibration, N-test decomposition | §4 |
| `channels/` | Engineered-channel information-gain tests (`source_ig_*`), GNSS trajectory features + promotion decision | Figure 5, S16 |
| `audit/` | Causality self-test output, cascade gate, prose-consistency gate output, pre-registration hash record | §3 |
| `csep/` | Catalog-based pyCSEP run (authoritative) + in-house Poisson cross-check (superseded for clustered models) | Figure 4 |
| `models/` | Trained boosters, calibrated wrappers, synthetic discriminator + its training report | §3, S14 |
| `round3/` | FROZEN — pre-registered proper-score campaign (Table S8 ledger), incl. `claims_bernoulli.json` | S3 |
| `round4/` | FROZEN — b_op forensics, post-review sensitivities, extended-b arms, `claims_sensitivities.json`, `freeze_manifest.json`, amendment hashes | S9–S10 |
| `round5/` | FROZEN — feature-ablation + grouped-PCA study | S12 |
| `validation_final/` | 2025 M6.2 countdown, case study, calibration battery, verdict list | Figure 6, S13, S15 |
| `sequence/` | Kumburgaz 2025 sequence analysis | S13 |
| `verify/` | Placebo batteries (GNSS), EPOS cross-validation, probe outputs (cap trend, mechanism split, attribution) | Figure 5 |
| `forecast/` | Issued 30-day products + quarter/year-ahead summaries | S15 |
| `prospective/` | Hash-stamped append-only forecast log + per-issue snapshots + track record | S15 |
| `figs/` | Figure sources copied into `paper/figs/` at build time | — |
