# Prospective monthly forecasting: track record

Each run logs a 30-day forecast (hashed, timestamped) *before* the outcome is known, then
scores past forecasts whose window has closed. This gives true out-of-sample operation at
low cost, and it is not open to the hindsight that a pseudo-prospective backtest allows.

## What runs
`python -m marmara.prospective run` (wrapped by `scripts/prospective_monthly.sh`):
1. **Refresh** the catalogue (best-effort; see the feed note below).
2. **Issue + log** a 30-day forecast at the catalogue end: regional P(M≥5/5.5/6) with the
   b-ensemble, per-segment renewal, combined M≥6.8, top cells, and the hybrid y35/y45
   per-cell grids, appended to `forecast_log.jsonl` with a stable **sha256 content hash**
   (a verifiable fingerprint of the prediction). Full grids are saved under `<t0>/grids.npz`.
   Idempotent: an unchanged catalogue end is never re-logged.
3. **Score** any forecast whose 30-day window has closed against realized events →
   `scored_log.jsonl`; the accumulation (predicted P vs realized frequency, Brier,
   per-cell Brier) is written to `track_record.md`.

## Optional scheduling (example)
On macOS, a launchd agent can run the job monthly. Example
`~/Library/LaunchAgents/com.example.marmara-monthly-forecast.plist` set to fire on the
**1st of each month, 06:00 local** (its working directory must point at the repo root):
```
launchctl list | grep marmara                                          # status
launchctl unload ~/Library/LaunchAgents/com.example.marmara-monthly-forecast.plist   # disable
bash scripts/prospective_monthly.sh                                    # run by hand
```
The bundled `forecast_log.jsonl` contains two example forecasts (t0 2026-07-05 and
2026-07-07); the first scores land ~30 days after issue.

## Data feed
The KOERI zeqdb search endpoint became a JS/AJAX app shell (unscrapable), so the feed
uses the **KOERI monthly PRELIMINARY XML lists**
(`udim.koeri.boun.edu.tr/zeqmap/xmlt/YYYYMM.xml`) via `scripts/refresh_monthly.py`:
TRT→UTC (−3 h), box filter, and a ±10 s / ±0.1° time-space novelty test (monthly rows
carry no event code) with a synthetic event code for idempotence.

**Nature of the feed (know this when reading scores):**
- Rows are **preliminary**, with a **single untyped rapid magnitude** (treated as
  `xM-unknown`, homogenized to `mag_w` like the rest of the preliminary tail).
- Scoring against preliminary magnitudes is **acceptable for M≥3.5 event counting** (y35 /
  the regional exceedance), because the count is robust to small magnitude revisions. **But
  it is threshold-sensitive for events near 3.5**: a preliminary M3.4↔3.6 can move an event
  across the boundary, so a small number of borderline y35 hits/misses may flip when reviewed
  data lands. Larger thresholds (M≥4.5/5/6) are essentially immune to this.

**Recommended maintenance (~every 2–3 months):** replace the preliminary tail with
**reviewed** data (zeqdb if it regains a scrapable endpoint, else AFAD or ISC) using the
documented 15-column schema, then re-run `python -m marmara.catalog`. This keeps the scored
track record on reviewed magnitudes. Accumulation itself only needs the catalogue to
ADVANCE, which the monthly XML already provides; the reviewed backfill is a quality upgrade,
not a prerequisite.
