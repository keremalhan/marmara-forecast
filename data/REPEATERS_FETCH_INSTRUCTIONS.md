# Repeating-earthquake catalog, fetch instructions

**Status: STILL-ABSENT.** `results/source_ig_repeating_eq.json` reports
`available: false`. The `repeating_eq` source was designed but never fed data; it is
**skipped, not tested**. The manuscript must not imply otherwise. The catalog is
NOT auto-fetchable in this sandbox (the source tables are journal supplements behind
publisher paywalls / captchas). A human should obtain one of the following and drop it
in as `data/external/marmara_repeaters.csv` with columns
`[datetime, lat, lon, family_id]`, then run `python -m marmara.source_ig_test repeating_eq`.

## Preferred sources (Main Marmara Fault repeaters / creep)

1. **Schmittbuhl, Karabulut, Lengliné & Bouchon (2016)**, "Long-lasting seismic
   repeaters in the Central Basin of the Main Marmara Fault," *Geophys. Res. Lett.* 43.
   DOI: **10.1002/2016GL070505**. → Download the Supporting Information (repeater
   family table with times, locations, family IDs).

2. **Bohnhoff, Wollin, Domigall et al. (2017)**, repeating-earthquake / aseismic-creep
   study of the Main Marmara Fault (GFZ Potsdam). Check GFZ DataServices / the paper's
   supplement for the repeater catalog. Bohnhoff et al. 2013 (*Nature Comm.*) and
   the GONAF borehole array (gfz-potsdam) are related sources.

3. **Uchida & Bürgmann (2019)** global repeater compilation (*Ann. Rev. EPS*), filter
   to the Marmara bbox [26–31°E, 39.5–41.5°N] if a machine-readable table is available.

## Expected format and gate
```
datetime,lat,lon,family_id        # one row per repeater event; family_id groups a repeating patch
2009-03-14T02:11:07,40.83,28.20,MF-017
...
```
Then: `MARMARA_ROOT=$PWD PYTHONPATH=$PWD/src .venv/bin/python -m marmara.source_ig_test repeating_eq`
Pre-registered promotion rule (same as every source): promote its 2 columns into
FEATURES iff val IG > +0.02 AND test IG > 0.

## Why it matters (do not overstate until fed)
Repeater recurrence ∝ 1/creep-rate, so a repeater catalog is a direct aseismic-creep
observable no seismicity-rate statistic captures, potentially the most interpretable
channel. But until the data is present it is **absent, not null**; report it as such.
