"""Run 23 — Item 6 of Amendment 8: leave-Kumburgaz-out Bernoulli, self-contained.

Governed by docs/preregistration/v2_analysis_amendment_8.md (SHA-256
c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690), hashed 2026-07-16T11:25:21Z
BEFORE this ran. Exclusion rule fixed in the amendment as a calendar rule, before seeing which
windows fall out (they are: t0 2025-04-16, 2025-05-16, 2025-06-15, 2025-07-15; r23 lists them first).

Computes the hybrid-vs-cascade Bernoulli-native ΔIG (native MC occupancy at K=5000, add-one, top-up
construction -- the SAME construction as r16) on:
  * FULL 26 test windows
  * LEAVE-OUT: 22 windows whose 30-day target does NOT intersect 2025-04-23..2025-07-23
  * KUMBURGAZ-ONLY complement: the 4 excluded windows (kill-it pass -- if the edge lives entirely in
    the sequence, that is the finding)
Plus the per-window Bernoulli sign count on the RETAINED windows only (the count-scored 25/26 is NOT
quoted), and remaining positives per stratum (active cnt30>0 vs quiet) to distinguish
"edge vanishes" from "power vanishes".

Writes results/round4/r23_leave_kumburgaz_out.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/leave_kumburgaz_out.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.bootstrap import MEAN_BLOCK, SEED, stationary_window_indices
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda
from marmara.paths import RESULTS
from marmara.train import split_masks

EPS = 1e-9
KMAX = 5000
R4 = RESULTS / "round4"
K_LO = pd.Timestamp("2025-04-23"); K_HI = pd.Timestamp("2025-07-23")


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def bern_ig_ci(win, y, pa, pb, keep=None):
    """Bernoulli ΔIG per positive + block-bootstrap CI, restricted to windows in `keep`."""
    pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
    c = y * (np.log(pa) - np.log(pb)) + (1 - y) * (np.log(1 - pa) - np.log(1 - pb))
    wins = np.sort(np.unique(win)) if keep is None else np.sort(np.array(sorted(keep)))
    idx = {w: np.where(win == w)[0] for w in wins}
    cw = np.array([c[idx[w]].sum() for w in wins]); pos = np.array([y[idx[w]].sum() for w in wins])
    ig = cw.sum() / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    bs = [cw[r].sum() / max(pos[r].sum(), 1) for r in seqs]
    return (round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)], int(pos.sum()))


def main():
    t0 = time.time()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC
    g = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(g)
    gt = g[m["test"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    keyi = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gt.iterrows()}
    n = len(gt)
    pte = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    pte = pte[pte.split == "test"].reset_index(drop=True)
    tmp = g[m["test"]].reset_index(drop=True).assign(hyb=pte.hybrid.to_numpy())
    tmp = tmp.sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    lam_hyb = lam(tmp.hyb.to_numpy()); y = tmp.y30.to_numpy(float); wt = tmp.window.to_numpy()
    active = (tmp.cnt30.to_numpy() > 0)
    win_t0 = g.groupby("window")["t0"].first()

    # (a) exclusion listed BEFORE scoring
    wins_all = [int(x) for x in np.sort(gt.window.unique())]
    excl = [w for w in wins_all if (pd.Timestamp(win_t0.loc[w]) + pd.Timedelta(days=30) >= K_LO)
            and (pd.Timestamp(win_t0.loc[w]) <= K_HI)]
    keep = [w for w in wins_all if w not in excl]
    out = {
        "governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_8.md",
                        "sha256": "c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690"},
        "K": KMAX, "exclusion_rule": "target intersects 2025-04-23..2025-07-23",
        "excluded_t0": [str(pd.Timestamp(win_t0.loc[w]).date()) for w in excl],
        "n_windows_full": len(wins_all), "n_windows_retained": len(keep),
    }
    print(f"(a) excluded windows (before scoring): "
          f"{out['excluded_t0']}  -> retain {len(keep)}/{len(wins_all)}", flush=True)

    # occupancy reconstruction at K=5000 (same construction as r16)
    occ_c = np.zeros(n); occ_t = np.zeros(n)
    for w in wins_all:
        t0_dt = pd.Timestamp(win_t0.loc[w])
        e = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], float(G._to_days(t0_dt)),
                             G.HORIZON_D, spec.lon_c, spec.lat_c, K=KMAX, seed=1000 + w, b=1.15,
                             preserve_branching=True, return_events=True)
        sid = e["sim"]
        ic_ = np.floor((e["lon"] - round(float(spec.lon_c[0]) - 0.05, 2)) / 0.1).astype(int)
        ir_ = np.floor((e["lat"] - round(float(spec.lat_c[0]) - 0.05, 2)) / 0.1).astype(int)
        cf = ir_ * 100000 + ic_
        o = np.argsort(cf, kind="stable"); sid, cf = sid[o], cf[o]
        uq = np.unique(cf); bd = np.searchsorted(cf, uq)
        for bi, c_ in enumerate(uq):
            i = keyi.get((w, int(c_ // 100000), int(c_ % 100000)))
            if i is None:
                continue
            lo = bd[bi]; hi = bd[bi + 1] if bi + 1 < len(bd) else len(cf)
            us, cts = np.unique(sid[lo:hi], return_counts=True)
            lam_c = len(sid[lo:hi]) / KMAX; lh = lam_hyb[i]
            ratio = min(1.0, lh / lam_c) if lam_c > 0 else 0.0; dl = max(lh - lam_c, 0.0)
            occ_c[i] = len(us) / KMAX
            thin = (1 - (1 - ratio) ** cts).sum() / KMAX
            occ_t[i] = ((len(us) + (KMAX - len(us)) * (1 - np.exp(-dl))) / KMAX) if dl > 0 else thin
        print(f"  window {w} ({t0_dt.date()}) done ({time.time()-t0:.0f}s)", flush=True)
    reg = lambda o: (o * KMAX + 1) / (KMAX + 2)
    pa, pb = reg(occ_t), reg(occ_c)

    # (full / leave-out / kumburgaz-only) Bernoulli ΔIG
    full = bern_ig_ci(wt, y, pa, pb)
    lo_out = bern_ig_ci(wt, y, pa, pb, keep=keep)
    komp = bern_ig_ci(wt, y, pa, pb, keep=excl)
    out["bernoulli_dig"] = {
        "full_26w": {"ig": full[0], "ci95": full[1], "n_pos": full[2]},
        "leave_kumburgaz_out_22w": {"ig": lo_out[0], "ci95": lo_out[1], "n_pos": lo_out[2]},
        "kumburgaz_only_4w": {"ig": komp[0], "ci95": komp[1], "n_pos": komp[2]},
    }

    # (b) remaining positives per stratum -- power vs edge
    def strat(mask_windows):
        sel = np.isin(wt, list(mask_windows))
        return {"positives_total": int(y[sel].sum()),
                "positives_active_cnt30>0": int(y[sel & active].sum()),
                "positives_quiet": int(y[sel & ~active].sum())}
    out["remaining_positives"] = {"retained_22w": strat(keep), "kumburgaz_4w": strat(excl),
                                  "full_26w": strat(wins_all)}

    # (c) per-window Bernoulli sign count on RETAINED windows only
    c_all = y * (np.log(pa) - np.log(pb)) + (1 - y) * (np.log(1 - pa) - np.log(1 - pb))
    signs = []
    for w in keep:
        idx = wt == w; s = c_all[idx].sum()
        signs.append(s)
    signs = np.array(signs)
    out["per_window_sign_count_retained"] = {
        "n_windows": len(keep), "positive": int((signs > 0).sum()),
        "negative": int((signs < 0).sum()), "zero": int((signs == 0).sum()),
        "note": "per-window Bernoulli ΔIG sign on the 22 retained windows; the count-scored 25/26 is NOT used here."}

    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r23_leave_kumburgaz_out.json", "w"), indent=1, default=str)
    print("\n=== LEAVE-KUMBURGAZ-OUT BERNOULLI (hybrid vs cascade) ===")
    for k, v in out["bernoulli_dig"].items():
        print(f"  {k:24s} ΔIG {v['ig']:+.4f} {v['ci95']}  (n_pos {v['n_pos']})")
    rp = out["remaining_positives"]
    print(f"  positives: full {rp['full_26w']['positives_total']} "
          f"(active {rp['full_26w']['positives_active_cnt30>0']}); "
          f"retained {rp['retained_22w']['positives_total']} "
          f"(active {rp['retained_22w']['positives_active_cnt30>0']}); "
          f"kumburgaz {rp['kumburgaz_4w']['positives_total']} "
          f"(active {rp['kumburgaz_4w']['positives_active_cnt30>0']})")
    sc = out["per_window_sign_count_retained"]
    print(f"  per-window sign (retained 22): +{sc['positive']} / -{sc['negative']} / 0:{sc['zero']}")
    print(f"  ({out['runtime_s']}s) -> results/round4/r23_leave_kumburgaz_out.json")


if __name__ == "__main__":
    main()
