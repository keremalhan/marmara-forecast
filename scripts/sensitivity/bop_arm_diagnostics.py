"""Run 16 — arm diagnostics: h/identity per arm (item 2) + Bernoulli companion (item 4).

CONTEXT. r15 established the paired verdicts across b_op in {1.00, 1.15, 1.40} and then RESTORED
every frozen artifact, so the arm predictions and grids no longer exist on disk. They are rebuilt
here and kept under results/round4/arm_<b>/ instead of being restored away, so the diagnostics below
and any future check can read them.

ITEM 2 — is the y30 dIG drift (+0.249 -> +0.289 -> +0.301 across b) the scoring artifact?
  preserve_branching pins the expected M>=3.0 total, so h(y30) = sum(lambda)/N_pos should be nearly
  b-invariant. If h is flat, the drift is NOT "the artifact growing with h" -- an earlier gloss said
  it was -- and the cause is elsewhere (the re-selected ML stage, or the magnitude-dependent spatial
  kernel). This prints h per model per arm and runs the identity check per arm, so the mechanism is
  measured rather than asserted. Also reports per-arm w and tree count (did the 1-SE rule slam w
  anywhere, as it did in the Mc=3.5 arm?).

ITEM 4 — the Bernoulli companion at the new arms. The surviving +0.10 occurrence edge is the paper's
  only positive ML result; its b-robustness is documented across 1.10-1.20 while the thesis table now
  spans 1.00-1.40. Native Monte-Carlo occupancy at K=2000, add-one regularized, top-up construction,
  Bernoulli log-score, paired block bootstrap -- the same machinery as round-3 t6.

Lineage: every arm is preserve_branching=True; the realized branching ratio is stamped per arm.

Reads/writes only results/round4/. Does NOT touch frozen artifacts.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/bop_arm_diagnostics.py
"""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara import grid as G
from marmara.bootstrap import MEAN_BLOCK, SEED, stationary_window_indices
from marmara.cascade import cascade_forecast
from marmara.etas_model import branching_ratio
from marmara.metrics import p_to_lambda
from marmara.paths import RESULTS
from marmara.train import split_masks

EPS = 1e-9
R4 = RESULTS / "round4"
ARMS = [1.00, 1.15, 1.40]
KS = [500, 1000, 2000]
KMAX = 2000
PY = str(RESULTS.parent / ".venv" / "bin" / "python")
FROZEN = ["etas_fit_report.json", "grid_hybrid.parquet", "grid_hybrid_report.json",
          "rates_sv_etas.parquet", "predictions_y30.parquet", "predictions_y35.parquet",
          "predictions_y45.parquet", "evaluation.json", "evaluation.md",
          "models/count30_hybrid.pkl", "models/count35_hybrid.pkl", "models/count45_hybrid.pkl",
          "models/count30_hybrid_gnss.pkl", "models/count35_hybrid_gnss.pkl",
          "models/count45_hybrid_gnss.pkl"]
BAK = RESULTS.parent / ".tmp" / "r16_backup"
MODELS = ["hybrid", "cascade", "sv_etas", "firstgen_etas", "modern_etas"]


def run(mod, *extra):
    r = subprocess.run([PY, "-m", mod, *extra], cwd=str(RESULTS.parent),
                       env={"PYTHONPATH": "src", "MARMARA_ROOT": ".", "PATH": "/usr/bin:/bin",
                            "TMPDIR": str(RESULTS.parent / ".tmp"),
                            "MPLCONFIGDIR": str(RESULTS.parent / ".tmp" / "mpl")},
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{mod} failed:\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def bern_ig_ci(win, y, pa, pb):
    pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
    c = y * (np.log(pa) - np.log(pb)) + (1 - y) * (np.log(1 - pa) - np.log(1 - pb))
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    cw = np.array([c[idx[w]].sum() for w in wins]); pos = np.array([y[idx[w]].sum() for w in wins])
    ig = cw.sum() / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    bs = [cw[r].sum() / max(pos[r].sum(), 1) for r in seqs]
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4),
                                 round(float(np.percentile(bs, 97.5)), 4)]


def main():
    t0 = time.time()
    BAK.mkdir(parents=True, exist_ok=True)
    before = {}
    for f in FROZEN:
        s = RESULTS / f
        if s.exists():
            d = BAK / f; d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d); before[f] = True
    print(f"backed up {len(before)} frozen artifacts", flush=True)

    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC
    out = {"arms": {}, "note": ("arm artifacts are KEPT under results/round4/arm_<b>/ rather than "
                                "restored away; every arm is preserve_branching=True")}

    try:
        for b in ARMS:
            tag = f"{b:.2f}"
            adir = R4 / f"arm_{tag}"; adir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== arm b_op = {tag} ===", flush=True)
            rep = json.load(open(RESULTS / "etas" / "etas_fit_report.json"))
            rep["operational_b_for_cascade"] = float(b)
            json.dump(rep, open(RESULTS / "etas" / "etas_fit_report.json", "w"), indent=2)
            run("marmara.grid_hybrid")
            run("marmara.etas_rates", "results/etas/etas_sv_params.pkl", "sv_etas")
            run("marmara.train")
            for f in ("grid_hybrid.parquet", "predictions_y30.parquet", "predictions_y35.parquet",
                      "evaluation.json"):
                shutil.copy2(RESULTS / f, adir / f)
            print(f"  rebuilt + saved ({time.time()-t0:.0f}s)", flush=True)

            g = pd.read_parquet(adir / "grid" / "grid_hybrid.parquet")
            m = split_masks(g)
            ev = json.load(open(adir / "scoring" / "evaluation.json"))
            grep_ = json.load(open(RESULTS / "grid" / "grid_hybrid_report.json"))
            rec = {"b_op": b,
                   "lineage": {"preserve_branching": True,
                               "grid_report_b_op": grep_["b_op"],
                               "b_op_stamped_matches_arm": bool(abs(grep_["b_op"] - b) < 1e-9),
                               "realized_branching_mmax7.6": round(float(branching_ratio(params, mmax=7.6)), 6),
                               "note": "grid_hybrid._pb defaults True (V2_SUPERCRITICAL unset)"},
                   "ml": {}, "h": {}, "identity": {}}
            # --- item 2: w / trees / h / identity, per target ---
            for tgt in ("y30", "y35"):
                pred = pd.read_parquet(adir / f"predictions_{tgt}.parquet")
                te = pred[pred.split == "test"]
                npos = float(te.y.sum())
                h = {mm: float(lam(te[mm].to_numpy()).sum() / npos) for mm in MODELS if mm in te}
                a = {mm: h[mm] - 1 - np.log(h[mm]) for mm in h}
                rec["h"][tgt] = {mm: round(h[mm], 4) for mm in h}
                rec["identity"][tgt] = {
                    "a_h": {mm: round(a[mm], 4) for mm in a},
                    "predicted_shift_hybrid_vs_cascade": round(a["hybrid"] - a["cascade"], 4)}
                rec["ml"][tgt] = {"n_pos_test": int(npos)}
            for tgt, tt in (("y30", "count30"), ("y35", "count35")):
                mdl = pickle.load(open(RESULTS / "models" / f"{tt}_hybrid.pkl", "rb"))
                rec["ml"][tgt] = {**rec["ml"].get(tgt, {}), "w": mdl["w"],
                                  "w_argmax": mdl["w_argmax"], "n_trees": int(mdl["reg"].n_iter_)}
            print(f"  h(y30): {rec['h']['y30']}", flush=True)
            print(f"  w: y30={rec['ml']['y30']['w']} (argmax {rec['ml']['y30']['w_argmax']}, "
                  f"{rec['ml']['y30']['n_trees']} trees) | y35={rec['ml']['y35']['w']} "
                  f"(argmax {rec['ml']['y35']['w_argmax']}, {rec['ml']['y35']['n_trees']} trees)",
                  flush=True)

            # --- item 4: Bernoulli companion, native occupancy at K=2000 ---
            gt = g[m["test"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
            keyi = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gt.iterrows()}
            n = len(gt)
            pte = pd.read_parquet(adir / "grid" / "predictions_y30.parquet")
            pte = pte[pte.split == "test"].reset_index(drop=True)
            tmp = g[m["test"]].reset_index(drop=True).assign(hyb=pte.hybrid.to_numpy())
            tmp = tmp.sort_values(["window", "ir", "ic"]).reset_index(drop=True)
            lam_hyb = lam(tmp.hyb.to_numpy()); y = tmp.y30.to_numpy(float); wt = tmp.window.to_numpy()
            occ_c = np.zeros(n); occ_t = np.zeros(n)
            win_t0 = g.groupby("window")["t0"].first()
            for w in [int(x) for x in np.sort(gt.window.unique())]:
                t0_dt = pd.Timestamp(win_t0.loc[w])
                e = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt],
                                     float(G._to_days(t0_dt)), G.HORIZON_D, spec.lon_c, spec.lat_c,
                                     K=KMAX, seed=1000 + w, b=b, preserve_branching=True,
                                     return_events=True)
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
                    lam_c = len(sid[lo:hi]) / KMAX
                    lh = lam_hyb[i]; ratio = min(1.0, lh / lam_c) if lam_c > 0 else 0.0
                    dl = max(lh - lam_c, 0.0)
                    occ_c[i] = len(us) / KMAX
                    thin = (1 - (1 - ratio) ** cts).sum() / KMAX
                    occ_t[i] = ((len(us) + (KMAX - len(us)) * (1 - np.exp(-dl))) / KMAX) if dl > 0 else thin
            reg = lambda o: (o * KMAX + 1) / (KMAX + 2)
            ig, ci = bern_ig_ci(wt, y, reg(occ_t), reg(occ_c))
            dpr = float(average_precision_score(y, pte.hybrid.to_numpy())
                        - average_precision_score(y, pte.cascade.to_numpy()))
            rec["bernoulli_companion_K2000"] = {
                "hybrid_vs_cascade_ig": ig, "ci95": ci,
                "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                "intensity_dpr": round(dpr, 4),
                "occupancy_totals": {"cascade": round(float(occ_c.sum()), 1),
                                     "hybrid_topup": round(float(occ_t.sum()), 1),
                                     "n_pos": int(y.sum())}}
            print(f"  Bernoulli K=2000 hybrid-vs-cascade: {ig:+.4f} {ci} "
                  f"excl0={rec['bernoulli_companion_K2000']['excludes_zero']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            out["arms"][tag] = rec
    finally:
        for f in before:
            shutil.copy2(BAK / f, RESULTS / f)
        print("\nfrozen artifacts restored", flush=True)

    # item 2 verdict
    hs = {t: out["arms"][t]["h"]["y30"]["cascade"] for t in out["arms"]}
    spread = max(hs.values()) - min(hs.values())
    out["item2_h_flatness"] = {
        "h_cascade_y30_per_arm": hs, "spread": round(spread, 4),
        "h_is_flat": bool(spread < 0.05),
        "reading": ("h(y30) is b-invariant as S1's pinned-total implies, so the dIG drift is NOT "
                    "the artifact growing with h -- that gloss is withdrawn"
                    if spread < 0.05 else
                    "h(y30) moves with b, so S1's pinned-total story needs a caveat")}
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r16_arm_diagnostics.json", "w"), indent=2)
    print(f"\n=== ITEM 2: h(cascade, y30) per arm: {hs}  spread {spread:.4f} ===")
    print(f"  {out['item2_h_flatness']['reading']}")
    print(f"\n=== ITEM 4: Bernoulli companion across arms ===")
    for t in out["arms"]:
        r = out["arms"][t]["bernoulli_companion_K2000"]
        print(f"  b={t}: hybrid-vs-cascade {r['hybrid_vs_cascade_ig']:+.4f} {r['ci95']} "
              f"excl0={r['excludes_zero']}")
    print(f"({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
