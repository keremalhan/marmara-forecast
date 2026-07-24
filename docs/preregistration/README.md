# Pre-registration chain (hash-frozen — never edit)

The dated, hashed pre-registration and its amendment chain, cited by the manuscript
(§2, Sections S3/S5/S9/S11, Data availability). Every file's SHA-256 was recorded
before the analyses it governs were run:

| Document | Hash record |
|---|---|
| `v2_preregistration.md` | `results/v2_preregistration.json` |
| `RUNPLAN_round3.md` (round-3 battery) | `results/round3/RUNPLAN_hash.json` |
| `v2_preregistration_amendment_1..4.md` (+ 3 addendum) | `results/round3/amendment3_hash.json`, `results/round4/amendment4_hash.json` |
| `v2_analysis_amendment_5..8.md` (post-review) | `results/round4/amendment5..8_hash.json` |

These are working documents by design: pre-registration records are archived as
written, and their content is what the hashes verify. Do not edit, reformat, or
respell them — any byte change breaks the chain.

Internal paths and script names inside these documents are archived exactly as
written; current locations resolve via `git log --follow`.
