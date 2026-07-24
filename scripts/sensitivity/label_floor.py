"""Run 10 — Label-floor robustness.

The Mc objection: the catalogue's completeness magnitude is 3.65 (catalog_report.json), while the
primary target labels occurrence at mag_w >= 3.0. If the 3.0 floor were manufacturing the verdicts
(e.g. by admitting incompletely-recorded small events whose spatial/temporal pattern flatters one
model), raising the floor should move them. A full retrain at a higher Mc is out of scope; but the
VERDICTS can be re-adjudicated cheaply by rebuilding the LABELS at a higher floor and re-scoring
the FROZEN test predictions -- the models are untouched, only what counts as a positive changes.

Floors: 3.0 (control -- must reproduce the frozen y exactly), 3.1, 3.2.
Pairs: hybrid vs cascade, cascade vs first-gen ETAS (the two the paper leans on).
Rule: the registered conjunctive rule -- A beats B iff BOTH the IG and PR-AUC 95% block-bootstrap
CIs exclude 0 in A's favour; else inseparable.

Writes results/round4/r10_label_floor.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/label_floor.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara import grid as G
from marmara.bootstrap import MEAN_BLOCK, SEED, stationary_window_indices, verdict_for
from marmara.metrics import information_gain, p_to_lambda
from marmara.paths import RESULTS
from marmara.train import split_masks

EPS = 1e-9
B_BOOT = 2000
FLOORS = [3.0, 3.1, 3.2]
PAIRS = [("hybrid", "cascade"), ("cascade", "firstgen_etas")]
OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)


def paired_ci(P_a, P_b, y, win):
    wins = np.sort(np.unique(win))
    y_by, a_by, b_by, lla, llb, pos, nrow = [], [], [], [], [], [], []
    for w in wins:
        k = win == w
        yv = y[k]
        y_by.append(yv); a_by.append(P_a[k]); b_by.append(P_b[k])
        pos.append(float(yv.sum())); nrow.append(float(k.sum()))
        for P, acc in ((P_a[k], lla), (P_b[k], llb)):
            lam = np.clip(p_to_lambda(np.clip(P, 0.0, 1.0)), EPS, None)
            acc.append(float(np.sum(yv * np.log(lam) - lam)))
    lla, llb = np.array(lla), np.array(llb)
    pos, nrow = np.array(pos), np.array(nrow)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), B_BOOT, MEAN_BLOCK, rng)
    d_ig = np.empty(B_BOOT); d_pr = np.full(B_BOOT, np.nan)
    for i in range(B_BOOT):
        s = seqs[i]
        d_ig[i] = (lla[s].sum() - llb[s].sum()) / max(pos[s].sum(), 1.0)
        if 0.0 < pos[s].sum() < nrow[s].sum():
            yy = np.concatenate([y_by[j] for j in s])
            d_pr[i] = (average_precision_score(yy, np.concatenate([a_by[j] for j in s]))
                       - average_precision_score(yy, np.concatenate([b_by[j] for j in s])))
    def ci(v):
        v = v[np.isfinite(v)]
        return [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]
    return {"d_ig": {"point": round(float(information_gain(P_a, P_b, y)), 6), "ci95": ci(d_ig)},
            "d_pr_auc": {"point": round(float(average_precision_score(y, P_a)
                                             - average_precision_score(y, P_b)), 6),
                         "ci95": ci(d_pr)}}


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet",
                           columns=["window", "t0", "ir", "ic", "y30"])
    m = split_masks(grid)
    sel = np.where(m["val"] | m["test"])[0]
    gsel = grid.iloc[sel].reset_index(drop=True)
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    assert len(gsel) == len(df) and (gsel["window"].to_numpy() == df["window"].to_numpy()).all()
    gsel = gsel.assign(split=df["split"].to_numpy())
    te_mask = (df["split"] == "test").to_numpy()
    gt = gsel[te_mask].reset_index(drop=True)
    pt = df[te_mask].reset_index(drop=True)
    win = pt["window"].to_numpy()

    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    t_days = np.asarray((cat["datetime_utc"] - G.REF) / pd.Timedelta(days=1), float)
    ir_e, ic_e = G.cell_index(cat["longitude"].to_numpy(), cat["latitude"].to_numpy())
    mag = cat["mag_w"].to_numpy()

    win_t0 = gt.groupby("window")["t0"].first()
    flat_grid = gt["ir"].to_numpy() * G.NLON + gt["ic"].to_numpy()

    out = {"floors": FLOORS, "pairs": [f"{a}_vs_{b}" for a, b in PAIRS],
           "rule": ("registered conjunctive rule: A beats B iff BOTH IG and PR-AUC 95% "
                    "block-bootstrap CIs exclude 0 in A's favour; else inseparable"),
           "note": ("models are FROZEN; only the definition of a positive changes. The 3.0 floor "
                    "is a control and must reproduce the shipped labels exactly."),
           "results": {}}

    for floor in FLOORS:
        ok = np.isfinite(mag) & (mag >= floor) & (ir_e >= 0) & (ic_e >= 0)
        te_t = t_days[ok]; te_ir = ir_e[ok]; te_ic = ic_e[ok]
        order = np.argsort(te_t, kind="stable")
        te_t, te_ir, te_ic = te_t[order], te_ir[order], te_ic[order]
        y = np.zeros(len(gt))
        for w in np.unique(win):
            t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
            lo = np.searchsorted(te_t, t0d, "left"); hi = np.searchsorted(te_t, t0d + G.HORIZON_D, "left")
            hit = set((te_ir[lo:hi] * G.NLON + te_ic[lo:hi]).tolist())
            k = np.where(win == w)[0]
            y[k] = np.isin(flat_grid[k], list(hit)).astype(float) if hit else 0.0

        rec = {"n_pos": int(y.sum()), "n_rows": int(len(y)),
               "positive_rate": round(float(y.mean()), 6)}
        if abs(floor - 3.0) < 1e-9:
            frozen = pt["y"].to_numpy(float)
            rec["control_matches_frozen_labels"] = bool((y == frozen).all())
            rec["control_n_pos_frozen"] = int(frozen.sum())
            if not rec["control_matches_frozen_labels"]:
                rec["control_n_mismatch"] = int((y != frozen).sum())

        rec["verdicts"] = {}
        for a, b in PAIRS:
            st = paired_ci(pt[a].to_numpy(float), pt[b].to_numpy(float), y, win)
            v = verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
            st["verdict"] = {"A_beats_B": f"{a} beats {b}", "B_beats_A": f"{b} beats {a}",
                             "inseparable": "inseparable"}[v]
            rec["verdicts"][f"{a}_vs_{b}"] = st
        out["results"][f"mag_w_ge_{floor}"] = rec
        print(f"floor {floor}: n_pos {rec['n_pos']}"
              + (f"  control matches frozen: {rec.get('control_matches_frozen_labels')}"
                 if "control_matches_frozen_labels" in rec else ""), flush=True)
        for k, st in rec["verdicts"].items():
            print(f"    {k:28s} dIG {st['d_ig']['point']:+.4f} {st['d_ig']['ci95']}  "
                  f"dPR {st['d_pr_auc']['point']:+.4f} {st['d_pr_auc']['ci95']}  -> {st['verdict']}")

    base = out["results"][f"mag_w_ge_3.0"]["verdicts"]
    out["gate"] = {
        "verdicts_unchanged_across_floors": bool(all(
            out["results"][f"mag_w_ge_{f}"]["verdicts"][k]["verdict"] == base[k]["verdict"]
            for f in FLOORS for k in base)),
        "control_reproduces_frozen_labels":
            out["results"]["mag_w_ge_3.0"].get("control_matches_frozen_labels"),
        "per_floor_verdicts": {f"mag_w_ge_{f}": {k: out["results"][f"mag_w_ge_{f}"]["verdicts"][k]["verdict"]
                                                 for k in base} for f in FLOORS},
    }
    json.dump(out, open(OUT / "r10_label_floor.json", "w"), indent=2)
    print(f"\nGATE: {json.dumps(out['gate'], indent=1)}")
    print("wrote results/round4/r10_label_floor.json")


if __name__ == "__main__":
    main()
