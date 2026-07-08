# manuscript_v2 — rewrite package for the Marmara forecasting paper

This directory is the **v2/v3 revision package** for
`paper/paper_seismica.md`. It does not re-typeset the whole manuscript; it supplies the
changed/added content, each block anchored to the section it replaces or follows, with
every number traced to the machine referee.

## Source-of-truth chain (do not break)
```
results/bootstrap_ci.{json,md}  →  results/claims.json  →  01_claims_map.md  →  02_sections.md (prose)
                                     (pre-registered rule; the ONLY licenser of rankings)
```
No ranking appears in the prose that `claims.json` does not license. If a number in the
prose and in `01_claims_map.md` disagree, the claims map (⇐ `claims.json`) wins.

## Contents
| file | role (spec) | what it is |
|---|---|---|
| `01_claims_map.md` | **M1** | Every claim C1–C6 + the full y30/y35 verdict matrix → verdict → evidence file, verbatim from `claims.json`. The GNSS "significant-but-void" trap is spelled out here. |
| `02_sections.md` | **M2** | Drop-in prose: new Abstract; new Methods subsections (block bootstrap, sv-ETAS + Mizrahi, GNSS + placebo, pyCSEP); rewritten Results §4.1–4.4. |
| `03_narrative.md` | **M3** | The three-act arc, what to foreground/soften, reviewer-proofing. Author guidance, not paper text. |
| `04_figures.md` | **M4** | Figure manifest: v1 figs retained + new v2/v3 figs with source paths and captions. |
| `05_limitations.md` | **M5** | Drop-in §6 Limitations and future work. |

## How to integrate into `paper/paper_seismica.md`
1. **Abstract** — replace the body with `02_sections.md ▸ Abstract`.
2. **§3 Methods** — insert the four new subsections from `02_sections.md`; replace the CSEP
   paragraph.
3. **§4 Results** — replace the opening and add §4.1–4.4 from `02_sections.md`. Demote any
   standalone "cascade ranks best" language (at y35 it is one of an inseparable family).
4. **§5 Discussion** — fold in `03_narrative.md`'s foreground points; keep the "where vs
   when" operational message.
5. **§6** — replace with `05_limitations.md`.
6. **Figures** — add fig5 (y30 forest) and fig6 (GNSS placebo) per `04_figures.md`; keep
   fig1–4.
7. Rebuild with `paper/build_pdf.sh` (or `make_seismica.py`).

## Primary-target change vs v1 (important)
v1 treated **M≥3.5** as the acceptance target. v2/v3 designates **M≥3.0 (y30)** as the
**primary powered** target (592 test positives) and M≥3.5 as a powered secondary; y45 is
unpowered. `claims.json` sets `primary:true` only on y30/test. ⚠️ `results/bootstrap_ci.md`
still shows a stale "y35/test PRIMARY" header — ignore it; numbers are unaffected.

## What changed scientifically from v1
- v1: "ML does not beat a well-fit ETAS" (inseparable at M≥3.5).
- v2/v3: on the **powered** M≥3.0 target, **physics decisively beats ML**, and the loss is
  to *modern* baselines (sv-ETAS + independent Mizrahi), not a strawman; the GNSS "win" is a
  placebo-level **null**; CSEP consistency is confirmed with **genuine pyCSEP**.

## Provenance
Produced on branch `v3-verify` (local commits only; **no remote git operations** were
performed). Post-phase gate green: leakage self-test PASS (all deviations 0), pytest
22 passed / 2 skipped. Verification verdicts in `results/REVIEW_PACKET.md` (the human gate).
