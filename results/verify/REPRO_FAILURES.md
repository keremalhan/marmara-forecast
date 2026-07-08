# V1 — reproduce-from-clean (APFS) vs committed, tolerance 1e-9

Compared 12 deterministic artifacts; **0 did not reproduce**.
(source_ig_gnss_v2.json excluded — network/date-dependent gate, not deterministic.)

| artifact | status |
|---|---|
| evaluation.json | OK |
| bootstrap_ci.json | OK |
| claims.json | OK |
| etas_sv_fit_report.json | OK |
| grid_hybrid_report.json | OK |
| csep/csep_results.json | OK |
| grid_hybrid.parquet | OK |
| predictions_y30.parquet | OK |
| predictions_y35.parquet | OK |
| predictions_y45.parquet | OK |
| rates_sv_etas.parquet | OK |
| rates_modern_etas.parquet | OK |

**All deterministic artifacts reproduced bit-for-bit / within 1e-9. No claim is VOID on reproducibility grounds.**