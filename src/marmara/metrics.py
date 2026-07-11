"""Shared scoring for Task 8 / Task 9. Every predictor is expressed as a
probability P over cell-windows; the Poisson rate is lambda = -ln(1-P).

All functions are pure (no fitting), so the model and all baselines are scored
through exactly the same code path.
"""
from __future__ import annotations

import numpy as np
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EPS = 1e-9


def p_to_lambda(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.0, 1.0 - EPS)
    return -np.log(1.0 - p)


def lambda_to_p(lam: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-np.clip(np.asarray(lam, float), 0.0, None))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    p = np.asarray(p, float); y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "bin": [round(float(edges[b]), 3), round(float(edges[b + 1]), 3)],
            "n": n,
            "mean_pred": float(p[m].mean()) if n else None,
            "obs_freq": float(y[m].mean()) if n else None,
        })
    return rows


def information_gain(p_model: np.ndarray, p_base: np.ndarray, y: np.ndarray) -> float:
    """Poisson log-likelihood gain per observed event (nats). n = y (0/1).
    The ln(n!) term cancels between model and baseline."""
    y = np.asarray(y, float)
    lm = np.clip(p_to_lambda(p_model), EPS, None)
    lb = np.clip(p_to_lambda(p_base), EPS, None)
    ll_m = np.sum(y * np.log(lm) - lm)
    ll_b = np.sum(y * np.log(lb) - lb)
    n = max(float(y.sum()), 1.0)
    return float((ll_m - ll_b) / n)


def molchan_skill(p: np.ndarray, y: np.ndarray) -> dict:
    """Molchan trajectory: alarmed fraction tau vs miss rate nu, as the alarm
    threshold sweeps. Area skill = 1 - 2*Area(nu vs tau); 0 = random, >0 skill."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    n = len(p); npos = float(y.sum())
    if npos == 0:
        return {"area_skill": None, "area": None}
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    hits = np.concatenate([[0.0], np.cumsum(ys)])
    tau = np.arange(n + 1) / n
    nu = 1.0 - hits / npos
    area = float(np.trapz(nu, tau))
    return {"area_skill": float(1.0 - 2.0 * area), "area": area}


def alert_budget(p: np.ndarray, y: np.ndarray, window: np.ndarray,
                 fracs=(0.005, 0.01, 0.02), ncells: int = 1219) -> dict:
    """Per window, alarm the top frac of cells; pool precision/recall."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    window = np.asarray(window)
    out = {}
    npos_total = float(y.sum())
    for f in fracs:
        k = int(np.ceil(f * ncells))
        alarmed = np.zeros(len(p), bool)
        for w in np.unique(window):
            m = np.where(window == w)[0]
            top = m[np.argsort(-p[m], kind="mergesort")[:k]]
            alarmed[top] = True
        hits = float((alarmed & (y > 0)).sum())
        n_alarm = float(alarmed.sum())
        out[f"top_{f*100:g}pct"] = {
            "cells_per_window": k,
            "precision": hits / n_alarm if n_alarm else None,
            "recall": hits / npos_total if npos_total else None,
            "n_alarmed": int(n_alarm), "n_hits": int(hits),
        }
    return out


def score_predictor(p: np.ndarray, y: np.ndarray, window: np.ndarray) -> dict:
    """Full metric bundle for one predictor (no IG; IG is relative, computed
    separately vs each baseline)."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    npos = int(y.sum())
    d = {
        "n": int(len(y)), "n_pos": npos,
        "pr_auc": None, "roc_auc": None, "brier": float(brier_score_loss(y, p)),
        "reliability": reliability_table(p, y),
        "molchan": molchan_skill(p, y),
        "alert_budget": alert_budget(p, y, window),
    }
    if 0 < npos < len(y):
        d["pr_auc"] = float(average_precision_score(y, p))
        d["roc_auc"] = float(roc_auc_score(y, p))
    return d
