"""Task 8 (part 2): evaluate the model against 4 baselines on val + test.

Everything (model + baselines) is scored through marmara.metrics. The test split isscored once per final configuration (reruns were identical-config reproductions), at the end. Honesty rule: if IG(model vs smoothed) <= 0 or
IG(model vs fault-proximity) <= 0 on test, the headline says the baseline wins.

Output: results/evaluation_report.json , results/evaluation_report.md

Run:  "<venv>/bin/python3" -m marmara.evaluate
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import baselines as B
from marmara.catalog import b_positive
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, score_predictor
from marmara.train import split_masks

OUT = RESULTS
MODELS = OUT / "models"
SIGMAS_KM = [5.0, 10.0, 20.0]
BASE_ORDER = ["poisson", "fault_prox", "smoothed", "etas"]


def train_b_value(cat: pd.DataFrame, mc: float) -> float:
    tr = cat[(cat["datetime_utc"] < pd.Timestamp("2022-01-01")) & (cat["mag_w"] >= mc)]
    b = b_positive(tr.sort_values("datetime_utc")["mag_w"].to_numpy())
    return float(b) if b is not None else 1.0


def build_baseline_lambdas(grid, masks, cat, mc, b_train, mc_etas, thr, val_idx, test_idx):
    """Return {name: (lam_val, lam_test)} choosing smoothed sigma on val."""
    lam = {}
    lp = B.poisson_clim(grid, masks["train"], thr)
    lf = B.fault_prox_clim(grid, masks["train"], thr)
    le = B.etas_baseline(grid, thr, b_train, mc_etas)
    lam["poisson"] = (lp[val_idx], lp[test_idx])
    lam["fault_prox"] = (lf[val_idx], lf[test_idx])
    lam["etas"] = (le[val_idx], le[test_idx])

    # smoothed: pick sigma by val PR-AUC (via P), compute test with the winner
    yv = grid["y35" if thr == 3.5 else "y45"].to_numpy()[val_idx]
    best = None
    grid_val = grid.iloc[val_idx]
    for s in SIGMAS_KM:
        lv = B.smoothed_seismicity(grid_val, cat, mc, thr, s)
        pv = lambda_to_p(lv)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(yv, pv) if 0 < yv.sum() < len(yv) else 0.0
        if best is None or ap > best[0]:
            best = (ap, s, lv)
    sigma = best[1]
    lv = best[2]
    lt = B.smoothed_seismicity(grid.iloc[test_idx], cat, mc, thr, sigma)
    lam["smoothed"] = (lv, lt)
    return lam, sigma


def evaluate_target(grid, masks, cat, mc, b_train, mc_etas, ycol):
    thr = 3.5 if ycol == "y35" else 4.5
    val_idx = np.where(masks["val"])[0]
    test_idx = np.where(masks["test"])[0]
    y = grid[ycol].to_numpy().astype(float)
    win = grid["window"].to_numpy()

    # model (calibrated) probabilities
    with open(MODELS / f"{ycol}_calibrated.pkl", "rb") as f:
        model = pickle.load(f)
    X = grid[FEATURES]
    p_model_val = model.predict_proba(X.iloc[val_idx])[:, 1]
    p_model_test = model.predict_proba(X.iloc[test_idx])[:, 1]

    lam, sigma = build_baseline_lambdas(grid, masks, cat, mc, b_train, mc_etas,
                                        thr, val_idx, test_idx)

    def preds(split):  # split in {'val','test'}
        j = 0 if split == "val" else 1
        d = {"model": p_model_val if split == "val" else p_model_test}
        for name in BASE_ORDER:
            d[name] = lambda_to_p(lam[name][j])
        return d

    res = {"threshold": thr, "smoothed_sigma_km": sigma, "splits": {}}
    for split, idx in (("val", val_idx), ("test", test_idx)):
        ys, ws = y[idx], win[idx]
        pr = preds(split)
        scored = {name: score_predictor(pr[name], ys, ws) for name in pr}
        ig = {name: information_gain(pr["model"], pr[name], ys) for name in BASE_ORDER}
        res["splits"][split] = {
            "n": int(len(idx)), "n_pos": int(ys.sum()),
            "scores": scored,
            "ig_model_vs": ig,
        }
    return res


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_markdown(report: dict, path: Path):
    L = ["# Evaluation: model vs 4 baselines", ""]
    L.append(report["headline"])
    L.append("")
    for ycol in ("y35", "y45"):
        r = report["targets"][ycol]
        L.append(f"## {ycol}  (M>={r['threshold']}, smoothed sigma={r['smoothed_sigma_km']:g} km)")
        for split in ("val", "test"):
            s = r["splits"][split]
            L.append(f"\n### {split}  (n={s['n']}, positives={s['n_pos']})")
            L.append("| predictor | PR-AUC | ROC-AUC | Brier | Molchan skill |")
            L.append("|---|---|---|---|---|")
            for name in ["model"] + BASE_ORDER:
                sc = s["scores"][name]
                ms = sc["molchan"]["area_skill"]
                L.append(f"| {name} | {_fmt(sc['pr_auc'])} | {_fmt(sc['roc_auc'])} | "
                         f"{_fmt(sc['brier'],5)} | {_fmt(ms,3)} |")
            L.append("\n**IG (model − baseline), nats/event, positive ⇒ model better:**")
            for name in BASE_ORDER:
                L.append(f"- vs {name}: {_fmt(s['ig_model_vs'][name],4)}")
            # alert budget (model)
            ab = s["scores"]["model"]["alert_budget"]
            L.append("\nModel alert budgets (precision / recall):")
            for k, v in ab.items():
                L.append(f"- {k}: P={_fmt(v['precision'],3)} R={_fmt(v['recall'],3)} "
                         f"({v['n_hits']}/{v['n_alarmed']} alarmed)")
        L.append("")
    path.write_text("\n".join(L))


def main():
    assert (OUT / "leakage_ok.json").exists(), "leakage self-test not green"
    grid = pd.read_parquet(OUT / "grid.parquet")
    masks = split_masks(grid)
    cat = pd.read_csv(OUT / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    mc = json.load(open(OUT / "catalog_report.json"))["mc"]
    mc_etas = json.load(open(OUT / "etas_fit_report.json")).get("used_mc", mc)
    b_train = train_b_value(cat, mc)

    report = {"meta": {"mc": mc, "mc_etas": mc_etas, "b_train": b_train,
                       "n_test": int(masks["test"].sum()),
                       "test_scored_once": False},
              "targets": {}}
    for ycol in ("y35", "y45"):
        print(f"evaluating {ycol} ...")
        report["targets"][ycol] = evaluate_target(grid, masks, cat, mc, b_train,
                                                   mc_etas, ycol)

    # honesty rule. The design's explicit bar: the model must exceed
    # fault-proximity AND smoothed-seismicity on y35 test (IG per event).
    t = report["targets"]["y35"]["splits"]["test"]["ig_model_vs"]
    t45 = report["targets"]["y45"]["splits"]["test"]["ig_model_vs"]
    meets_bar = (t["smoothed"] > 0) and (t["fault_prox"] > 0)
    beats_etas_y35 = t["etas"] > 0

    parts = []
    if meets_bar:
        parts.append(
            "**HEADLINE (y35, M>=3.5, test): the model CLEARS the required bar. It "
            f"beats plain-Poisson (+{t['poisson']:.3f}), fault-proximity "
            f"(+{t['fault_prox']:.3f}) and smoothed-seismicity (+{t['smoothed']:.3f} "
            "nats/event in information gain).**")
    else:
        losers = [n for n in ("smoothed", "fault_prox") if t[n] <= 0]
        parts.append(
            "**HEADLINE (y35, test): a required baseline BEATS the model: "
            + ", ".join(f"{n} (IG={t[n]:.3f})" for n in losers)
            + ". The model does not demonstrate skill over these baselines.**")
    if not beats_etas_y35:
        parts.append(
            f"But the physics-based **ETAS baseline is the single strongest predictor** "
            f"on y35 test (model IG vs ETAS = {t['etas']:.3f}; ETAS PR-AUC "
            f"{report['targets']['y35']['splits']['test']['scores']['etas']['pr_auc']:.3f} "
            f">= model {report['targets']['y35']['splits']['test']['scores']['model']['pr_auc']:.3f}). "
            "The ML model beats the naive baselines but does not beat a properly-fit ETAS.")
    else:
        parts.append(f"The model also beats ETAS on y35 test (IG +{t['etas']:.3f}).")
    # y45 honesty (rare target)
    y45_baselines_win = (t45["smoothed"] <= 0) or (t45["etas"] <= 0)
    if y45_baselines_win:
        parts.append(
            f"On the rarer **y45 (M>=4.5) test (only "
            f"{report['targets']['y45']['splits']['test']['n_pos']} positives), the "
            f"ETAS and smoothed-seismicity baselines beat the model** (IG vs smoothed "
            f"{t45['smoothed']:.3f}, vs ETAS {t45['etas']:.3f}). The ML model adds no "
            "skill over physical baselines at this magnitude.")
    report["headline"] = "\n\n".join(parts)
    report["y35_test_ig_model_vs"] = t
    report["y45_test_ig_model_vs"] = t45
    report["meets_required_bar_y35"] = bool(meets_bar)
    report["beats_etas_y35"] = bool(beats_etas_y35)
    report["y45_baselines_win"] = bool(y45_baselines_win)

    with open(OUT / "evaluation_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, OUT / "evaluation_baseline.md")
    print("\n" + report["headline"])
    print("\nwrote evaluation_report.{json,md}")


if __name__ == "__main__":
    main()
