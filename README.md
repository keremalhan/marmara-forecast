# marmara-forecast

A **leakage-audited** earthquake-forecasting benchmark for the Sea of Marmara
(Türkiye): a simulation-based cascade Monte-Carlo forecaster, an ETAS×ML hybrid, a
conditional large-event discriminator, renewal priors, and an information-arrival
analysis of the 2025 Marmara events (including the 23 April 2025 Mw 6.2 Kumburgaz).

The priority throughout is to state plainly what can and cannot be forecast. Every
headline number is reproduced from an artifact under `results/`, a machine-checkable
causality gate guards against data leakage, and the negative results stay in view: the
ML stage adds no ranking skill over a well-fit ETAS, its apparent likelihood edge is a
scoring artifact with a closed form, GNSS strain adds no measurable gain, and
large-event *timing* is close to Poisson.

- **Methods & results:** [`docs/METHODS.md`](docs/METHODS.md)
- **Leakage prevention (design):** [`docs/AUDIT.md`](docs/AUDIT.md)
- **Data licensing & attribution:** [`DATA_LICENSE.md`](DATA_LICENSE.md)

## Install

The results were produced on **Python 3.13** (3.11+ also works if wheels are
available for your platform). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

The package finds its data/results relative to the repository root, so no paths need
editing. Set `PYTHONPATH` to `src` (and, if your system disk is small, point `TMPDIR`
at a roomy disk):

```bash
export PYTHONPATH="$PWD/src"
export MARMARA_ROOT="$PWD"          # optional; defaults to the repo root

# Rebuild the catalog from the bundled KOERI-derived events:
python -m marmara.catalog           # -> results/catalog/catalog.csv, catalog_report.json

# The four correctness gates (all must pass):
python -m marmara.tests.test_stress         # 13/13  Coulomb/Okada stress
python -m marmara.tests.test_etas           #  5/5   ETAS fit/simulate/causality (~90 s)
python -m marmara.tests.test_cascade        #  4/4   cascade forecaster
python -m marmara.tests.test_grid_leakage   #  truncated-catalog leakage self-test
```

To reproduce **everything** end-to-end (catalog → ETAS fit → grids → gates → hybrid
train/evaluate → synthetic → forecast → validation → M6.2 countdown), with the four
gates as hard stops:

```bash
./run_all.sh
```

`run_all.sh` uses `./.venv/bin/python` if present, else `python3`; override with
`MARMARA_PY=/path/to/python ./run_all.sh`. Expected total runtime is roughly an hour
(the synthetic stage dominates); per-stage runtimes are printed inline.

The modern-baseline and paper-support layer (sv-ETAS, the independent Mizrahi ETAS
inversion, block-bootstrap claims, CSEP consistency, the operational-b sweep, and the
headline figures) runs on top of that:

```bash
./scripts/run_extended.sh        # sv-ETAS + Mizrahi baselines, hybrid re-train,
                                 #   bootstrap claims.json, in-house CSEP
./scripts/sensitivity/b_sensitivity.sh       # operational-b sweep -> results/b_sensitivity.{json,md}
PYTHONPATH=src .venv/bin/python scripts/csep/csep_prep.py       # pyCSEP catalog inputs
.venv-csep/bin/python scripts/csep/csep_run.py                  # pyCSEP consistency tests
                                 #   (separate env: requirements-csep.txt)
PYTHONPATH=src .venv/bin/python scripts/figures/fig_discriminator_reliability.py # discriminator reliability fig
PYTHONPATH=src .venv/bin/python scripts/figures/fig_forest_and_gnss.py   # forest + GNSS-placebo figs
```

## Repository layout

```
marmara-forecast/
  README.md              this file
  LICENSE                MIT (code)
  DATA_LICENSE.md        KOERI / NGL / GEM attribution and redistribution terms
  CITATION.cff           how to cite
  requirements.txt       pinned dependencies
  run_all.sh             full reproduction with the four gates
  src/marmara/           the package (one system; see docs/METHODS.md)
    tests/               the four correctness gates
    sources/             extension-source harness (GNSS / dense catalog / repeaters)
  scripts/               analysis & support scripts, by purpose (see scripts/README.md):
                         figures/ csep/ scoring/ ml_stage/ sensitivity/ foreshock/
                         verify/ release/ prospective/ + run_extended.sh
  data/                  KOERI-derived catalog + fault/strain/GNSS inputs + fetch scripts
  results/               key result artifacts (JSON, Markdown, figures; map in results/README.md)
  docs/METHODS.md        unified methods & results
  docs/AUDIT.md          leakage-prevention design
```

## Results at a glance

- **Primary target y30 (M≥3.0/30-day), test set, 592 positives**
  (`results/scoring/evaluation.json`; verdicts in `results/claims.json`): the physics
  forecasters are mutually inseparable under the pre-specified block-bootstrap rule,
  and the ETAS×ML hybrid is inseparable from all of them on ranking (PR-AUC 0.226
  against cascade 0.233 and first-generation ETAS 0.223; ΔPR-AUC vs the cascade
  −0.007 [−0.020, +0.007]). That parity survives re-running the whole pipeline at
  b_op = 1.10, 1.15 and 1.20 (`results/scoring/b_sensitivity.md`).
- **The hybrid's likelihood edge is a scoring artifact.** Scoring a clustered forecast
  against binary occurrence inflates a rescaled forecast by up to h − 1 − ln h nats per
  positive (h ≈ 2.2), which covers the +0.289 information gain the hybrid shows against
  the cascade. Under a proper binary log-score a smaller edge survives, +0.10
  [+0.05, +0.15], and it is a dynamic occurrence recalibration resolved in active
  cells. None of it resolves at M≥3.5.
- **y45 (M≥4.5)** has 22 test positives, too few to power a comparison, and carries no
  registered verdict; the wide-box variant is kept as a calibration diagnostic only.
- **Large-event discriminator** (`results/models/synthetic_report.json`): weak on the
  simulation-disjoint split, with test PR-AUC 0.123 and ROC-AUC 0.598, roughly 2.4× the
  base rate. On the two real 2025 sequences it does not separate the sequence-preceded
  case from the cold-start one; the Foreshock Traffic-Light System does.
- **Information-arrival analysis** (`results/validation_final/m62_countdown.*`): the
  23 April Mw 6.2 epicentral cell sat in the top ~1–2% of cells all year, while the
  timing of the event itself was close to unforecastable. The lone $M_L$ 4.0 foreshock,
  36 minutes before rupture, lifted the 30-day P(M≥6) gain to 39.7× as measured ten
  minutes after it. Some 220 comparable alarms in the same catalog were followed by
  no M≥6.
- **Live 30-day P(M≥6)** at 2026-07-05 (`results/forecast/.../forecast_summary.json`):
  0.13%–3.74% across the b-ensemble, central 1.54% at b_op. That is a fraction of a
  percent per month, the same order as the Poisson base rate. No claim of an imminent
  large event.

We report probability gain, never "alarm" or "prediction."

## Citation

See [`CITATION.cff`](CITATION.cff). Authors: Basri Kerem Alhan; Kenessary Khabat.

## License

Code: MIT ([`LICENSE`](LICENSE)). Data: third-party terms, see
[`DATA_LICENSE.md`](DATA_LICENSE.md). Cite KOERI for the catalog, the Nevada
Geodetic Laboratory (Blewitt et al. 2018) for GNSS, and the GEM Global Active Faults
Database (Styron & Pagani 2020) for the fault model.
