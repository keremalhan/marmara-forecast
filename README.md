# marmara-forecast

A **leakage-audited** earthquake-forecasting benchmark for the Sea of Marmara
(Türkiye): a simulation-based cascade Monte-Carlo forecaster, an ETAS×ML hybrid, a
conditional large-event discriminator, renewal priors, and an information-arrival
analysis of the 2025 Marmara events (including the **23 April 2025 Mw 6.2 Kumburgaz**).

The defining feature of this project is honesty about what can and cannot be
forecast. Every headline number is reproduced from an artifact under `results/`, a
machine-checkable causality gate guards against data leakage, and the negative
results are kept in (the ML model does **not** beat a properly-fit ETAS on rare
targets; GNSS strain adds no measurable gain; large-event *timing* is close to Poisson).

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

# Rebuild the catalogue from the bundled KOERI-derived events:
python -m marmara.catalog           # -> results/catalog.csv, catalog_report.json

# The four correctness gates (all must pass):
python -m marmara.tests.test_stress         # 13/13  Coulomb/Okada stress
python -m marmara.tests.test_etas           #  5/5   ETAS fit/simulate/causality (~90 s)
python -m marmara.tests.test_cascade        #  4/4   cascade forecaster
python -m marmara.tests.test_grid_leakage   #  truncated-catalogue leakage self-test
```

To reproduce **everything** end-to-end (catalogue → ETAS fit → grids → gates → hybrid
train/evaluate → synthetic → forecast → validation → M6.2 countdown), with the four
gates as hard stops:

```bash
./run_all.sh
```

`run_all.sh` uses `./.venv/bin/python` if present, else `python3`; override with
`MARMARA_PY=/path/to/python ./run_all.sh`. Expected total runtime is roughly an hour
(the synthetic stage dominates); per-stage runtimes are printed inline.

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
    sources/             extension-source harness (GNSS / dense catalogue / repeaters)
  scripts/               refresh_monthly.py, prospective_monthly.sh
  data/                  KOERI-derived catalogue + fault/strain/GNSS inputs + fetch scripts
  results/               key result artifacts (JSON, Markdown, figures)
  docs/METHODS.md        unified methods & results
  docs/AUDIT.md          leakage-prevention design
```

## Honest results at a glance

- **y35 (M≥3.5), test** (`results/evaluation.json`): the cascade Monte-Carlo forecaster
  ranks best (PR-AUC 0.130) and beats first-generation ETAS (0.126), but the cascade,
  the ETAS×ML hybrid (0.117), and first-gen ETAS are within noise; **the hybrid does
  not beat a properly-fit ETAS** (IG −0.38/event). It does clear the naive baselines
  (IG vs Poisson +1.14, fault-proximity +0.95, smoothed +0.51).
- **y45 (M≥4.5):** the model-box hybrid overfits on 22 positives; the **wide-box
  remedy** (`results/widebox_y45_report.json`) regularizes it to w = 0.1 and makes IG
  and PR-AUC agree (IG vs cascade +0.735).
- **Large-event discriminator** (`results/synthetic_report.json`): ranks the escalating
  2025 Sındırgı sequence (0.277) far above the decaying Kumburgaz sequence (~0.01).
- **Information-arrival analysis** (`results/validation_final/m62_countdown.*`): applied
  to the 2025 events, it shows the 23 April Mw 6.2 fault cell was the top ~1% seismicity
  cell **all year** (99th percentile), but the specific event was near-unforecastable in
  time — the lone M4.5 foreshock lifted the 30-day P(M≥6) gain to **42×** only ten
  minutes before the mainshock.
- **Live 30-day P(M≥6)** at 2026-07-05 (`results/forecast/.../forecast_summary.json`):
  0.13%–4.56% (b-ensemble), central 1.12% — a fraction of a percent per month, the same
  order as the Poisson base rate. **No claim of an imminent large event.**

We report probability **gain**, never "alarm" or "prediction."

## Citation

See [`CITATION.cff`](CITATION.cff). Authors: Basri Kerem Alhan; Kenessary Khabat.

## License

Code: MIT ([`LICENSE`](LICENSE)). Data: third-party terms, see
[`DATA_LICENSE.md`](DATA_LICENSE.md) — cite KOERI for the catalogue, the Nevada
Geodetic Laboratory (Blewitt et al. 2018) for GNSS, and the GEM Global Active Faults
Database (Styron & Pagani 2020) for the fault model.
