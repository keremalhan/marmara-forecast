"""Item C worker: ONE constrained ETAS refit at a fixed branching ratio. One fit per process.

The fit parameterizes productivity through the branching ratio n with bounds (ln 1e-3, ln cap), and
the MLE is monotone increasing in n -- it pins at whatever cap it is given (0.95 -> 0.95;
0.999 -> 0.9990, the documented degenerate near-critical mode). Setting cap = n therefore CONSTRAINS
the fit to n exactly while re-optimizing every nuisance parameter, which is the constrained refit
item C asks for. We verify the pin rather than assume it.

Run in its own process because repeated full fits in one process exhaust memory (see
scripts/verify/mc_sensitivity.py).

Usage: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/branching_profile_worker.py <n> <out.json>
"""
from __future__ import annotations

import json
import sys

import pandas as pd

import marmara.etas_fit as EF
from marmara.etas_model import branching_ratio
from marmara.paths import RESULTS

MMAX = 7.6


def main():
    n_fix = float(sys.argv[1])
    out_path = sys.argv[2]
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat[cat["datetime_utc"] < EF.FIT_END].copy()
    params, n_tgt, n_dropped, b_val, b_aki = EF.fit_stai(cat, EF.MODEL_BOX, n_fix)
    res = getattr(params, "_fit_result")
    n_untrunc = float(branching_ratio(params))
    rec = {
        "n_fixed_cap": n_fix,
        "n_realized_untruncated": n_untrunc,
        "pinned_at_cap": bool(abs(n_untrunc - n_fix) < 1e-4),
        "n_mmax_truncated": float(branching_ratio(params, mmax=MMAX)),
        "nll": float(res.fun),
        "loglik": float(-res.fun),
        "converged": bool(res.success),
        "n_events_fit": int(n_tgt),
        "k": float(params.k), "alpha": float(params.alpha), "c": float(params.c),
        "p": float(params.p), "d": float(params.d), "q": float(params.q),
        "gamma": float(params.gamma), "mu_total": float(params.mu_total),
        "b_positive": float(b_val),
    }
    json.dump(rec, open(out_path, "w"), indent=1)
    print(f"n={n_fix}: loglik {rec['loglik']:.4f}  realized n {n_untrunc:.6f} "
          f"pinned={rec['pinned_at_cap']}  k={params.k:.4f} alpha={params.alpha:.4f} "
          f"p={params.p:.4f}  converged={rec['converged']}")


if __name__ == "__main__":
    main()
