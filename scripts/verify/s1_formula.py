"""Run 6 — Supplement S1 formula verification.

S1 displays the per-unit-K branching ratio as

    n1_displayed(b) = beta/(beta-alpha) * [1 - e^{-(beta-alpha)*dm}]

but states the offspring density as the NORMALIZED truncated GR,
f(m) = beta*e^{-beta(m-mc)} / (1 - e^{-beta*dm}). Carrying that normalization through the
integral gives

    n1_correct(b) = beta/(beta-alpha) * [1 - e^{-(beta-alpha)*dm}] / [1 - e^{-beta*dm}]

i.e. the displayed expression is missing the 1 - e^{-beta*dm} denominator.

This run checks whether the omission changes any published number. It should not: the CODE
(marmara.etas_model.branching_ratio) already carries the denominator, and _sample_gr draws
from the correctly normalized truncated law, so the pipeline was always right -- the defect is
in the displayed formula only. We verify that numerically rather than assert it, and confirm
the K_sim/K ratio is unchanged (the two denominators nearly, but not exactly, cancel).

Writes results/round4/r6_s1_formula.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/s1_formula.py
"""
from __future__ import annotations

import json

import numpy as np
from dataclasses import replace

from marmara.etas_model import LN10, branching_ratio
from marmara.grid import load_params
from marmara.paths import RESULTS

MMAX = 7.6
OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)


def n1_displayed(b, alpha, mc, mmax):
    """S1 as printed: normalization denominator omitted."""
    beta = b * LN10
    dm = mmax - mc
    return beta / (beta - alpha) * (1.0 - np.exp(-(beta - alpha) * dm))


def n1_correct(b, alpha, mc, mmax):
    """S1 with the normalized truncated-GR density carried through."""
    beta = b * LN10
    dm = mmax - mc
    return (beta / (beta - alpha) * (1.0 - np.exp(-(beta - alpha) * dm))
            / (1.0 - np.exp(-beta * dm)))


def n1_quadrature(b, alpha, mc, mmax, n=4_000_001):
    """Independent check: numerically integrate exp(alpha(m-mc)) * f(m) dm over [mc, mmax]
    with the normalized truncated density. Confirms n1_correct is the right closed form."""
    beta = b * LN10
    m = np.linspace(mc, mmax, n)
    f = beta * np.exp(-beta * (m - mc)) / (1.0 - np.exp(-beta * (mmax - mc)))
    return float(np.trapezoid(np.exp(alpha * (m - mc)) * f, m))


def main():
    p = load_params()
    alpha, mc, k_fit, b_fit = p.alpha, p.mc, p.k, p.b
    dm = MMAX - mc
    n_star = branching_ratio(p, mmax=MMAX)          # fitted, mmax-truncated: 0.95

    rows = []
    for b in (b_fit, 1.15, 1.02):
        beta = b * LN10
        d = float(n1_displayed(b, alpha, mc, MMAX))
        c = float(n1_correct(b, alpha, mc, MMAX))
        q = n1_quadrature(b, alpha, mc, MMAX)
        # what the CODE computes, per unit K (k=1) -- the pipeline's ground truth
        code = branching_ratio(replace(p, b=b, k=1.0), mmax=MMAX)
        rows.append({
            "b": round(float(b), 6),
            "beta": round(float(beta), 6),
            "norm_denominator_1_minus_exp_neg_beta_dm": float(1.0 - np.exp(-beta * dm)),
            "n1_displayed_formula": round(d, 6),
            "n1_corrected_formula": round(c, 6),
            "n1_numerical_quadrature": round(q, 6),
            "n1_from_code_branching_ratio": round(float(code), 6),
            "delta_corrected_minus_displayed": float(c - d),
            "rel_delta": float((c - d) / d),
            "code_matches_corrected": bool(abs(code - c) < 1e-9),
            "quadrature_matches_corrected": bool(abs(q - c) < 1e-6),
            "rounds_to_2dp": round(c, 2), "rounds_to_3dp": round(c, 3),
        })

    n1_fit_d = float(n1_displayed(b_fit, alpha, mc, MMAX))
    n1_fit_c = float(n1_correct(b_fit, alpha, mc, MMAX))
    n1_op_d = float(n1_displayed(1.15, alpha, mc, MMAX))
    n1_op_c = float(n1_correct(1.15, alpha, mc, MMAX))

    # K_sim/K under each formula. The two normalization denominators nearly (not exactly)
    # cancel in the ratio, since both are ~1 - O(1e-6).
    ratio_d = n1_fit_d / n1_op_d
    ratio_c = n1_fit_c / n1_op_c
    k_sim_c = n_star / n1_op_c
    k_sim_d = n_star / n1_op_d

    out = {
        "params": {"alpha": alpha, "mc": mc, "mmax": MMAX, "dm": dm,
                   "k_fitted": k_fit, "b_fitted": b_fit,
                   "n_star_mmax_truncated": round(float(n_star), 6)},
        "defect": ("S1's displayed n1(b) omits the 1 - e^{-beta*dm} normalization denominator "
                   "of the truncated GR density f(m) it states one line earlier. The CODE "
                   "(etas_model.branching_ratio) and the sampler (_sample_gr) both carry it, "
                   "so no published number is affected -- the defect is typographic."),
        "per_b": rows,
        "K_sim_over_K": {
            "displayed_formula": round(float(ratio_d), 6),
            "corrected_formula": round(float(ratio_c), 6),
            "difference": float(ratio_c - ratio_d),
            "published_value": 0.78,
            "corrected_rounds_to_2dp": round(float(ratio_c), 2),
            "unchanged_at_published_precision": bool(round(float(ratio_c), 2) == 0.78),
        },
        "K_sim_absolute": {
            "corrected": round(float(k_sim_c), 6), "displayed": round(float(k_sim_d), 6),
            "published_value": 0.454,
            "corrected_rounds_to_3dp": round(float(k_sim_c), 3),
        },
        "gate": {
            "n1_b_fit_1.640_unchanged": bool(round(n1_fit_c, 3) == 1.640),
            "n1_b_op_2.092_unchanged": bool(round(n1_op_c, 3) == 2.092),
            "K_sim_over_K_0.78_unchanged": bool(round(float(ratio_c), 2) == 0.78),
        },
    }
    out["gate"]["ALL_PASS"] = bool(all(v for v in out["gate"].values()))
    json.dump(out, open(OUT / "r6_s1_formula.json", "w"), indent=2)

    print(f"alpha={alpha:.6f} mc={mc} mmax={MMAX} dm={dm} k_fit={k_fit:.6f} b_fit={b_fit:.6f}")
    print(f"n* (mmax-truncated, fitted) = {n_star:.6f}\n")
    print(f"{'b':>9} {'displayed':>11} {'corrected':>11} {'quadrature':>11} {'code':>11} "
          f"{'delta':>12} {'rel':>11}")
    for r in rows:
        print(f"{r['b']:>9.4f} {r['n1_displayed_formula']:>11.6f} {r['n1_corrected_formula']:>11.6f} "
              f"{r['n1_numerical_quadrature']:>11.6f} {r['n1_from_code_branching_ratio']:>11.6f} "
              f"{r['delta_corrected_minus_displayed']:>12.3e} {r['rel_delta']:>11.3e}")
    print(f"\nK_sim/K  displayed {ratio_d:.6f}   corrected {ratio_c:.6f}   "
          f"diff {ratio_c-ratio_d:.3e}   -> {round(ratio_c,2)} (published 0.78)")
    print(f"K_sim    corrected {k_sim_c:.6f} (published 0.454)")
    print(f"\nGATE: {json.dumps(out['gate'])}")


if __name__ == "__main__":
    main()
