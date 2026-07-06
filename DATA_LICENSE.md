# Data licensing and attribution

The code in this repository is MIT-licensed (see `LICENSE`). The datasets under
`data/` originate from third parties and carry their own terms. Please cite the
original sources in any publication.

## Earthquake catalog — `data/koeri_events.csv`
Derived from the seismicity bulletin of the **Boğaziçi University Kandilli Observatory
and Earthquake Research Institute, Regional Earthquake-Tsunami Monitoring Center
(KOERI-RETMC)**. This file is a *processed, derived* dataset (parsed, UTC-corrected,
blast-screened, and box-filtered by the scripts in `data/`); it is redistributed here
for reproducibility with attribution. Users must cite KOERI:

> Kandilli Observatory and Earthquake Research Institute (KOERI), Boğaziçi University,
> Regional Earthquake-Tsunami Monitoring Center, Istanbul, Türkiye.

Cross-checks against **AFAD** (Disaster and Emergency Management Presidency of Türkiye)
were used for the largest events. The fetch/build scripts (`data/fetch_zeqdb.py`,
`data/build_catalog.py`; monthly-preliminary feed in `scripts/refresh_monthly.py`) are
included so the dataset can be re-derived from source. Provenance and the empirical
UTC-timezone proof are documented in `data/fetch_manifest.json`.

## GNSS velocities — `data/combined_gnss_velocity_data.json`, strain field `data/marmara_strain_grid.npz`
Derived from the **Nevada Geodetic Laboratory (NGL)** GPS/GNSS solutions. Cite:

> Blewitt, G., Hammond, W. C., & Kreemer, C. (2018). Harnessing the GPS data explosion
> for interdisciplinary science. *Eos, 99*. https://doi.org/10.1029/2018EO104623

## Active fault model — `data/segment_properties.json`, `data/marmara_faults.geojson`
Derived from the **GEM Global Active Faults Database**. Cite:

> Styron, R., & Pagani, M. (2020). The GEM Global Active Faults Database.
> *Earthquake Spectra, 36*(S1), 160–180. https://doi.org/10.1177/8755293020944182

Historical fault-model attributes trace to the EMME/GEM compilations; individual
rupture source parameters used in the stress model cite the primary studies
(Barka et al., 2002; Karabulut et al., 2021; Ertuncay et al., 2025; USGS).
