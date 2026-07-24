# scripts/: analysis and support scripts, by purpose

The reproducible pipeline itself lives in `src/marmara/` and is driven by `../run_all.sh`.
Everything here runs **on top** of that pipeline: baselines, scoring studies, sensitivity
arms, figures, verification probes, and release tooling. Unless a docstring says otherwise,
run scripts from the repository root as

```bash
PYTHONPATH=src MARMARA_ROOT=. ./.venv/bin/python scripts/<group>/<name>.py
```

## Layout

| directory | contents |
|---|---|
| `figures/` | every manuscript figure generator (map+FMD, forest, identity curve, pyCSEP/GNSS/denominator panels, discriminator reliability) |
| `csep/` | pyCSEP catalog preparation and the catalog-based N/M/S/PL consistency tests (`csep_run.py` needs the separate `.venv-csep` env), plus the hybrid-construction and in-house cross-checks |
| `scoring/` | the count/occupancy scoring campaign: the identity a(h) = h − 1 − ln h, native Monte-Carlo occupancy, Jensen debias, recalibration challengers, verdict locks |
| `ml_stage/` | analyses of the machine-learning stage itself: booster fairness and capacity, proper-score blend-weight re-selection, and the feature ablation + grouped-PCA study |
| `sensitivity/` | everything that re-runs the system under a perturbed configuration: operational-b (b_op) sweeps/arms/forensics, block length, negative controls, label floor, Mc = 3.5 arm, branching-ratio profile, early stopping, leave-Kumburgaz-out, multiplicity/BH |
| `foreshock/` | the false-alarm denominator, episode deduplication, and alarm-persistence/Clopper–Pearson accounting |
| `verify/` | one-off verification probes wired like tests (timezone anchors, leakage, GNSS placebos, N-test attribution, `prose_consistency_gate.py`, …) |
| `release/` | `reproduce_all.py` (asserts every manuscript number against its archived artifact; run before any release), the freeze manifest, and the claims-sensitivities emitter |
| `prospective/` | the monthly prospective forecast job and catalog refresh |
| `run_extended.sh` | modern-baseline layer on top of `run_all.sh` (sv-ETAS, independent inversion, bootstrap claims, in-house CSEP) |

## Two conventions

**Campaign provenance is kept, not erased.** These scripts were written in dated,
partly pre-registered analysis campaigns; docstrings keep their campaign IDs
("T0-4", "Run 15", amendment SHA-256 prefixes) because the run ledger (supplement
Table S8) and the hashed pre-registration documents refer to them. Script names
inside those hashed documents resolve via `git log --follow`.

**Output paths under `results/` are frozen.** Several scripts write to
`results/round3|round4|round5/…`. Those are *artifact* paths cited verbatim by the
manuscript, the machine-readable claims files, and the archived release, so renaming
them would desynchronize the paper from its evidence. Do not "clean them up".
