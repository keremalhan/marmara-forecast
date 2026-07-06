"""Space-time ETAS (Epidemic-Type Aftershock Sequence) model for the Marmara project.

Self-contained implementation (numpy/scipy/pandas only, no network). Serves three roles:
  1. baseline probabilistic forecast (``fit_etas`` + ``expected_counts`` + ``prob_at_least_one``),
  2. causal expected-rate feature generator (``expected_counts`` is strictly causal),
  3. synthetic-catalog simulator for rare-event (M6+) training data (``simulate_catalogs``).

Model & conventions
-------------------
Conditional intensity (per day, per km^2), for events with magnitude >= mc:

    lambda(t, x, y) = mu_total * u(x, y)
                    + sum_{i: t_i < t} kappa(m_i) * g(t - t_i) * f(|r - r_i|^2; m_i)

  * Productivity (Ogata/natural-log convention):  kappa(m) = k * exp(alpha * (m - mc)).
  * Omori-Utsu temporal kernel, normalized to unit mass (requires p > 1):
        g(t) = (p - 1)/c * (1 + t/c)^(-p),  H(t) = 1 - (1 + t/c)^(1-p).
  * Isotropic spatial power-law kernel, normalized over the plane (requires q > 1):
        f(r^2; m) = (q - 1)/(pi * D) * (1 + r^2 / D)^(-q),  D = d * 10^(gamma * (m - mc)),
    with d in km^2; gamma is the (log10) magnitude scaling of the kernel area.
  * Background: mu_total [events/day, M>=mc, whole region] times a spatial density
    u(x, y) [1/km^2] normalized to integrate to 1 over the region (weighted-KDE of
    probabilistically declustered events; simplified Zhuang iteration).
  * Coordinates are projected to km with a flat-earth equirectangular projection about
    the region center; times are handled internally as float days.

Fitting = direct maximization of the standard space-time ETAS log-likelihood
(Ogata 1998; Daley & Vere-Jones):  sum_i log lambda(t_i, x_i, y_i) - integral(lambda),
with analytic gradients, L-BFGS-B on log-transformed parameters, multiple restarts,
and background-declustering iterations (unit-weight KDE -> fit -> background-
probability-weighted KDE -> refit; KDE evaluated leave-one-out at event locations).
The productivity scale is optimized as the branching ratio n = k*beta/(beta-alpha),
constrained n <= 0.95 (subcritical stationary model — a deliberate, documented
constraint required for the baseline/simulator roles).

Documented approximations
-------------------------
* Pair interactions in the likelihood are truncated at ``tau_max`` days (default 3650,
  auto-shrunk if the pair list would exceed memory limits) and ``r_max`` km (default
  150). The integral term uses the SAME truncated kernel (truncation-consistent
  likelihood), so shape parameters are essentially unbiased; the mass of triggering
  beyond the truncation window (a few % to ~20% for very heavy Omori tails) is
  absorbed into the background rate (mu biased slightly up, never down).
* Spatial kernel mass leaking outside the region is not edge-corrected (standard;
  biases k slightly downward for sources near the border).
* ``expected_counts`` is a FIRST-GENERATION (no secondary triggering) approximation,
  standard for short horizons; it underestimates rates in very active sequences by
  roughly the factor 1/(1 - n) at long horizons (n = branching ratio).
* ``expected_counts`` uses the analytic Omori integral in time and a cell-center
  approximation in space (kernel pdf at cell center times cell area), with each
  source's total distributed mass capped at 1 to avoid overshoot when a kernel is
  much narrower than a cell.
* Sub-mc events are ignored everywhere (kappa is calibrated for m >= mc sources).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from scipy.optimize import minimize

__all__ = [
    "EtasParams",
    "BackgroundField",
    "fit_etas",
    "expected_counts",
    "prob_at_least_one",
    "simulate_catalogs",
    "branching_ratio",
]

LN10 = float(np.log(10.0))
KM_PER_DEG = 111.19  # mean km per degree of latitude (flat-earth approx)


# ----------------------------------------------------------------------------- utils
def _find_col(df: pd.DataFrame, candidates) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def _extract_catalog(df: pd.DataFrame):
    """Return (times, lat, lon, mag) from a catalog-like DataFrame.

    ``times`` is either a float ndarray (days) or a datetime64 ndarray, depending on
    what the catalog provides. Column names are matched case-insensitively:
    time: datetime | origin_time | (date + time) | time ; lat: latitude|lat ;
    lon: longitude|lon ; mag: magnitude|mag|ml|mw. Depth is ignored (2-D model).
    """
    lat_c = _find_col(df, ("latitude", "lat"))
    lon_c = _find_col(df, ("longitude", "lon"))
    mag_c = _find_col(df, ("magnitude", "mag", "ml", "mw"))
    if lat_c is None or lon_c is None or mag_c is None:
        raise ValueError(f"catalog missing lat/lon/mag columns; got {list(df.columns)}")
    lat = np.asarray(df[lat_c], dtype=float)
    lon = np.asarray(df[lon_c], dtype=float)
    mag = np.asarray(df[mag_c], dtype=float)

    dt_c = _find_col(df, ("datetime", "origin_time", "date_time"))
    t_c = _find_col(df, ("time", "t"))
    d_c = _find_col(df, ("date",))
    if dt_c is not None:
        times = pd.to_datetime(df[dt_c], errors="coerce").values
    elif t_c is not None and np.issubdtype(np.asarray(df[t_c]).dtype, np.number):
        times = np.asarray(df[t_c], dtype=float)
    elif d_c is not None and t_c is not None:
        times = pd.to_datetime(
            df[d_c].astype(str).str.split(" ").str[0] + " " + df[t_c].astype(str),
            errors="coerce",
        ).values
    elif t_c is not None:
        times = pd.to_datetime(df[t_c], errors="coerce").values
    elif d_c is not None:
        times = pd.to_datetime(df[d_c], errors="coerce").values
    else:
        raise ValueError(f"catalog missing a usable time column; got {list(df.columns)}")
    return times, lat, lon, mag


def _is_timelike(x) -> bool:
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return True
    import datetime as _dt

    if isinstance(x, (_dt.date, _dt.datetime, str)):
        return True
    arr = np.asarray(x)
    return np.issubdtype(arr.dtype, np.datetime64)


def _to_days(times, ref) -> np.ndarray:
    """Convert times (datetime64 array or float-days array) to float days since ``ref``."""
    arr = np.asarray(times)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    if np.issubdtype(arr.dtype, np.datetime64):
        if not _is_timelike(ref):
            raise ValueError("catalog times are datetimes but the reference time is numeric")
        ref64 = np.datetime64(pd.Timestamp(ref))
        return (arr - ref64) / np.timedelta64(1, "s") / 86400.0
    if _is_timelike(ref):
        raise ValueError("catalog times are numeric (float days) but the reference time is a datetime")
    return arr.astype(float) - float(ref)


def _scalar_days(x, ref) -> float:
    if _is_timelike(x) != _is_timelike(ref):
        raise ValueError("mixed numeric and datetime time specifications")
    if _is_timelike(x):
        return float((pd.Timestamp(x) - pd.Timestamp(ref)).total_seconds() / 86400.0)
    return float(x) - float(ref)


def _project_km(lon, lat, ref_lon, ref_lat):
    coslat = np.cos(np.radians(ref_lat))
    x = (np.asarray(lon, float) - ref_lon) * KM_PER_DEG * coslat
    y = (np.asarray(lat, float) - ref_lat) * KM_PER_DEG
    return x, y


def _unproject_km(x, y, ref_lon, ref_lat):
    coslat = np.cos(np.radians(ref_lat))
    lon = ref_lon + np.asarray(x, float) / (KM_PER_DEG * coslat)
    lat = ref_lat + np.asarray(y, float) / KM_PER_DEG
    return lon, lat


def _region_center(region):
    return (
        0.5 * (region["min_lon"] + region["max_lon"]),
        0.5 * (region["min_lat"] + region["max_lat"]),
    )


def _region_area_km2(region) -> float:
    ref_lat = 0.5 * (region["min_lat"] + region["max_lat"])
    w = (region["max_lon"] - region["min_lon"]) * KM_PER_DEG * np.cos(np.radians(ref_lat))
    h = (region["max_lat"] - region["min_lat"]) * KM_PER_DEG
    return float(w * h)


def _aki_b(mags: np.ndarray, mc: float) -> float:
    """Aki (1965) maximum-likelihood b-value, with 0.1-bin correction if magnitudes are binned."""
    m = np.asarray(mags, float)
    m = m[m >= mc - 1e-9]
    if m.size < 10:
        return 1.0
    binned = bool(np.all(np.abs(m * 10.0 - np.round(m * 10.0)) < 1e-3))
    mref = mc - 0.05 if binned else mc
    return float(1.0 / (LN10 * max(m.mean() - mref, 1e-3)))


def branching_ratio(params: "EtasParams", mmax: float | None = None) -> float:
    """Expected number of direct offspring per event (criticality parameter n).

    n = k * beta / (beta - alpha) for an untruncated GR magnitude law (beta = b ln10);
    with ``mmax`` given, the GR law is truncated at mmax and n stays finite for any alpha.
    """
    beta = params.b * LN10
    k, alpha = params.k, params.alpha
    if mmax is None:
        if alpha >= beta:
            return float("inf")
        return float(k * beta / (beta - alpha))
    dm = mmax - params.mc
    if abs(beta - alpha) < 1e-9:
        return float(k * beta * dm / (1.0 - np.exp(-beta * dm)))
    return float(
        k * beta / (beta - alpha)
        * (1.0 - np.exp(-(beta - alpha) * dm))
        / (1.0 - np.exp(-beta * dm))
    )


# ----------------------------------------------------------------- background density
class BackgroundField:
    """Spatial background density over a region, normalized so that its integral over
    the region is 1 (units: 1/km^2). Either uniform or a weighted Gaussian KDE of
    (probabilistically declustered) epicenters, mixed with a 1% uniform floor for
    numerical robustness. Provides pdf evaluation and sampling in lon/lat.
    """

    UNIFORM_MIX = 0.01

    def __init__(self, region: dict, kind: str = "uniform", lon=None, lat=None,
                 weights=None, bw_km: float | None = None):
        self.region = dict(region)
        self.kind = kind
        self.ref_lon, self.ref_lat = _region_center(region)
        self.area_km2 = _region_area_km2(region)
        self.bw_km = bw_km
        self._Z = 1.0
        if kind == "kde":
            x, y = _project_km(np.asarray(lon, float), np.asarray(lat, float),
                               self.ref_lon, self.ref_lat)
            w = np.asarray(weights, float).clip(min=0.0)
            if w.sum() <= 0 or x.size == 0:
                self.kind = "uniform"
                return
            w = w / w.sum()
            self._x, self._y, self._w = x, y, w
            if bw_km is None:
                n_eff = w.sum() ** 2 / (w * w).sum()
                sx = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
                sy = np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
                silverman = max(n_eff, 2.0) ** (-1.0 / 6.0) * 0.5 * (sx + sy)
                self.bw_km = float(np.clip(0.5 * silverman, 1.5, 12.0))
            self._Z = self._region_mass()

    @classmethod
    def uniform(cls, region: dict) -> "BackgroundField":
        return cls(region, kind="uniform")

    @classmethod
    def from_weighted_events(cls, lon, lat, weights, region: dict,
                             bw_km: float | None = None) -> "BackgroundField":
        return cls(region, kind="kde", lon=lon, lat=lat, weights=weights, bw_km=bw_km)

    # -- internals ---------------------------------------------------------------
    def _kde_raw(self, x, y, chunk: int = 400) -> np.ndarray:
        """Unnormalized-over-region plane KDE value at km coords (1/km^2)."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        h2 = self.bw_km ** 2
        out = np.zeros(x.size, dtype=float)
        norm = 1.0 / (2.0 * np.pi * h2)
        for s in range(0, self._x.size, chunk):
            dx = x[:, None] - self._x[None, s:s + chunk]
            dy = y[:, None] - self._y[None, s:s + chunk]
            out += np.exp(-(dx * dx + dy * dy) / (2.0 * h2)) @ self._w[s:s + chunk]
        return out * norm

    def _region_mass(self, ngrid: int = 70) -> float:
        r = self.region
        gl = np.linspace(r["min_lon"], r["max_lon"], ngrid)
        gt = np.linspace(r["min_lat"], r["max_lat"], ngrid)
        LON, LAT = np.meshgrid(gl, gt)
        gx, gy = _project_km(LON.ravel(), LAT.ravel(), self.ref_lon, self.ref_lat)
        vals = self._kde_raw(gx, gy)
        da = self.area_km2 / (ngrid * ngrid)
        return float(max(vals.sum() * da, 1e-6))

    # -- public API ----------------------------------------------------------------
    def pdf_lonlat(self, lon, lat) -> np.ndarray:
        """Background density (1/km^2), normalized to integrate to 1 over the region."""
        lon = np.atleast_1d(np.asarray(lon, float))
        lat = np.atleast_1d(np.asarray(lat, float))
        unif = 1.0 / self.area_km2
        if self.kind == "uniform":
            return np.full(lon.size, unif)
        x, y = _project_km(lon, lat, self.ref_lon, self.ref_lat)
        kde = self._kde_raw(x, y) / self._Z
        m = self.UNIFORM_MIX
        return (1.0 - m) * kde + m * unif

    def pdf_at_support_loo(self) -> np.ndarray:
        """pdf at the KDE's own support points with each point's self-kernel removed
        (leave-one-out), to avoid the self-neighborhood bias when the density is
        evaluated at the very events it was built from (used during fitting)."""
        if self.kind == "uniform":
            return np.full(1, 1.0 / self.area_km2)
        kde = self._kde_raw(self._x, self._y)
        kde = np.maximum(kde - self._w / (2.0 * np.pi * self.bw_km ** 2), 0.0)
        m = self.UNIFORM_MIX
        return (1.0 - m) * kde / self._Z + m / self.area_km2

    def sample(self, n: int, rng: np.random.Generator):
        """Sample n background epicenters (lon, lat) inside the region."""
        r = self.region
        if n <= 0:
            return np.zeros(0), np.zeros(0)
        if self.kind == "uniform":
            lon = rng.uniform(r["min_lon"], r["max_lon"], n)
            lat = rng.uniform(r["min_lat"], r["max_lat"], n)
            return lon, lat
        lon = np.empty(n)
        lat = np.empty(n)
        todo = np.arange(n)
        for _ in range(100):
            m = todo.size
            if m == 0:
                break
            use_unif = rng.random(m) < self.UNIFORM_MIX
            idx = rng.choice(self._x.size, size=m, p=self._w)
            xs = self._x[idx] + rng.normal(0.0, self.bw_km, m)
            ys = self._y[idx] + rng.normal(0.0, self.bw_km, m)
            lo, la = _unproject_km(xs, ys, self.ref_lon, self.ref_lat)
            lo = np.where(use_unif, rng.uniform(r["min_lon"], r["max_lon"], m), lo)
            la = np.where(use_unif, rng.uniform(r["min_lat"], r["max_lat"], m), la)
            ok = ((lo >= r["min_lon"]) & (lo <= r["max_lon"])
                  & (la >= r["min_lat"]) & (la <= r["max_lat"]))
            lon[todo[ok]] = lo[ok]
            lat[todo[ok]] = la[ok]
            todo = todo[~ok]
        if todo.size:  # extremely rare: clamp stragglers into the region
            lon[todo] = np.clip(rng.uniform(r["min_lon"], r["max_lon"], todo.size),
                                r["min_lon"], r["max_lon"])
            lat[todo] = np.clip(rng.uniform(r["min_lat"], r["max_lat"], todo.size),
                                r["min_lat"], r["max_lat"])
        return lon, lat

    def __repr__(self):
        if self.kind == "uniform":
            return f"BackgroundField(uniform, area={self.area_km2:.0f} km^2)"
        return (f"BackgroundField(kde, n={self._x.size}, bw={self.bw_km:.1f} km, "
                f"area={self.area_km2:.0f} km^2)")


# ------------------------------------------------------------------------ parameters
@dataclass
class EtasParams:
    mu_total: float      # background events/day (M>=mc) over the region
    background_xy: object  # BackgroundField: sample/evaluate the background spatially
    k: float             # productivity at m = mc: kappa(m) = k * exp(alpha*(m-mc))
    alpha: float         # productivity exponent (natural-log convention)
    c: float             # Omori c [days]
    p: float             # Omori exponent (> 1)
    d: float             # spatial kernel area at m = mc [km^2]
    q: float             # spatial power-law exponent (> 1)
    gamma: float         # log10 magnitude scaling of d (D = d * 10^(gamma*(m-mc)); may be 0)
    mc: float            # completeness magnitude used for the fit / simulation
    b: float             # Gutenberg-Richter b-value used for simulation
    region: dict         # {'min_lat','max_lat','min_lon','max_lon'}


# ----------------------------------------------------------------- likelihood machinery
# The productivity scale is parameterized through the (untruncated) branching ratio
# n = k * beta / (beta - alpha), bounded n <= 0.95. This enforces a SUBCRITICAL,
# stationary model — required for the simulator/baseline roles — and blocks the
# well-known degenerate MLE mode on real catalogs (p -> 1 with huge k: a nearly flat
# Omori tail absorbing swarms/completeness drift, extrapolating to a supercritical
# process).
def _make_bounds(beta):
    return [
        (np.log(1e-4), np.log(200.0)),   # ln mu
        (np.log(1e-3), np.log(0.95)),    # ln n (branching ratio; subcritical)
        (0.05, min(3.2, beta - 0.02)),   # alpha (< beta so n stays finite)
        (np.log(5e-4), np.log(3.0)),     # ln c
        (np.log(0.02), np.log(1.5)),     # ln (p-1)  -> p in [1.02, 2.5]
        (np.log(1e-3), np.log(400.0)),   # ln d
        (np.log(0.08), np.log(2.5)),     # ln (q-1)  -> q in (1.08, 3.5]
        (0.0, 1.5),                      # gamma
    ]


def _theta_pack(mu, n_br, alpha, c, p, d, q, gamma):
    return np.array([np.log(mu), np.log(n_br), alpha, np.log(c),
                     np.log(p - 1.0), np.log(d), np.log(q - 1.0), gamma])


def _theta_unpack(theta, beta):
    mu = np.exp(theta[0]); n_br = np.exp(theta[1]); alpha = theta[2]
    k = n_br * (beta - alpha) / beta
    c = np.exp(theta[3]); p = 1.0 + np.exp(theta[4])
    d = np.exp(theta[5]); q = 1.0 + np.exp(theta[6]); gamma = theta[7]
    return mu, k, alpha, c, p, d, q, gamma


def _build_pairs(t, x, y, dm, tgt_idx, tau_max, r_max, max_pairs=3.0e7, min_tau=20.0):
    """Enumerate (target i, earlier source j) pairs with dt <= tau and r <= r_max.

    Returns dict(pi=target-local index, dt, r2, dm=source m - mc) and the tau used.
    tau is halved (down to min_tau days) until the raw pair count is tractable.
    Pair arrays are stored float32 (documented precision tradeoff: dt granularity
    ~15 s at 10-year lags, far below the c lower bound of ~43 s).
    """
    tau = float(tau_max)
    while True:
        lo = np.searchsorted(t, t[tgt_idx] - tau, side="left")
        cnt = (tgt_idx - lo).astype(np.int64)
        total = int(cnt.sum())
        if total <= max_pairs * 2.5 or tau <= min_tau:
            break
        tau *= 0.5
    r2max = float(r_max) ** 2
    P = {"pi": [], "dt": [], "r2": [], "dm": []}
    B = 4000
    for s in range(0, tgt_idx.size, B):
        ti = tgt_idx[s:s + B]
        li = lo[s:s + B]
        ci = cnt[s:s + B]
        tot = int(ci.sum())
        if tot == 0:
            continue
        i_loc = np.repeat(np.arange(s, s + ti.size, dtype=np.int64), ci)
        off = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(ci) - ci, ci)
        j = np.repeat(li, ci) + off
        gi = np.repeat(ti, ci)
        dt = t[gi] - t[j]
        dx = x[gi] - x[j]
        dy = y[gi] - y[j]
        r2 = dx * dx + dy * dy
        keep = r2 <= r2max
        P["pi"].append(i_loc[keep].astype(np.int32))
        P["dt"].append(dt[keep].astype(np.float32))
        P["r2"].append(r2[keep].astype(np.float32))
        P["dm"].append(dm[j[keep]].astype(np.float32))
    if P["pi"]:
        out = {k_: np.concatenate(v) for k_, v in P.items()}
    else:
        out = {"pi": np.zeros(0, np.int32), "dt": np.zeros(0, np.float32),
               "r2": np.zeros(0, np.float32), "dm": np.zeros(0, np.float32)}
    return out, tau


def _nll_grad(theta, D):
    """Negative log-likelihood and analytic gradient (both w.r.t. transformed theta)."""
    beta = D["beta"]
    mu, k, alpha, c, p, d, q, gamma = _theta_unpack(theta, beta)
    n_tgt = D["n_tgt"]
    u = D["u_tgt"]

    # --- triggering sum over pairs (float32 pair arrays; float64 accumulation)
    dmp = D["dm"]
    kap = k * np.exp(alpha * dmp)
    a = D["dt"] / np.float32(c)
    one_a = np.float32(1.0) + a
    g = (p - 1.0) / c * one_a ** np.float32(-p)
    Dk = d * np.exp(np.float32(gamma * LN10) * dmp)
    s2 = D["r2"] / Dk
    one_s = np.float32(1.0) + s2
    f = (q - 1.0) / (np.pi * Dk) * one_s ** np.float32(-q)
    w = kap * g * f
    lam = mu * u + np.bincount(D["pi"], weights=w, minlength=n_tgt)
    sln = float(np.log(lam).sum())
    wrp = (w / lam[D["pi"]]).astype(np.float64)

    # --- integral term (truncation-consistent)
    dms = D["dm_src"]
    kap_s = k * np.exp(alpha * dms)
    B1 = 1.0 + D["tau1"] / c
    B2 = 1.0 + D["tau2"] / c
    B1e = B1 ** (1.0 - p)
    B2e = B2 ** (1.0 - p)
    dH = B1e - B2e
    Dks = d * np.exp(gamma * LN10 * dms)
    G = 1.0 + D["R"] / Dks
    Ge = G ** (1.0 - q)
    F = 1.0 - Ge
    kHF = kap_s * dH * F
    integral = mu * D["T"] + float(kHF.sum())
    nll = -(sln - integral)

    # --- gradient of the log-likelihood
    grad = np.zeros(8)
    # ln mu
    grad[0] = float((mu * u / lam).sum()) - mu * D["T"]
    # ln k
    grad[1] = float(wrp.sum()) - float(kHF.sum())
    # alpha
    grad[2] = float((wrp * dmp).sum()) - float((kHF * dms).sum())
    # ln c
    pa = p * a / one_a
    dint_c = (p - 1.0) / c * (D["tau1"] * B1 ** (-p) - D["tau2"] * B2 ** (-p))
    grad[3] = float((wrp * (pa - 1.0)).sum()) - float((kap_s * F * dint_c).sum())
    # ln (p-1)
    dH_p = np.log(B2) * B2e - np.log(B1) * B1e
    grad[4] = (float((wrp * (1.0 - (p - 1.0) * np.log(one_a))).sum())
               - float((kap_s * F * dH_p).sum()) * (p - 1.0))
    # ln d
    qs = q * s2 / one_s
    dF_lnd = (1.0 - q) * G ** (-q) * (D["R"] / Dks)
    grad[5] = float((wrp * (qs - 1.0)).sum()) - float((kap_s * dH * dF_lnd).sum())
    # ln (q-1)
    dF_thq = (q - 1.0) * np.log(G) * Ge
    grad[6] = (float((wrp * (1.0 - (q - 1.0) * np.log(one_s))).sum())
               - float((kap_s * dH * dF_thq).sum()))
    # gamma
    grad[7] = (float((wrp * LN10 * dmp * (qs - 1.0)).sum())
               - float((kap_s * dH * dF_lnd * LN10 * dms).sum()))
    # chain rule for the n-parameterization: theta[1] = ln n, k = n*(beta-alpha)/beta
    # => dlnk/dln n = 1 (grad[1] already correct), dlnk/dalpha = -1/(beta-alpha)
    grad[2] -= grad[1] / (beta - alpha)
    return nll, -grad


def _fit_stage(D, theta0, maxiter=200):
    bounds = _make_bounds(D["beta"])
    lb = np.array([b_[0] for b_ in bounds]); ub = np.array([b_[1] for b_ in bounds])
    theta0 = np.clip(theta0, lb + 1e-9, ub - 1e-9)
    # tolerances sized to the float32 pair-sum noise floor (~1e-3 on NLL ~1e5);
    # tighter settings make the line search fail with an ABNORMAL status
    return minimize(_nll_grad, theta0, args=(D,), jac=True, method="L-BFGS-B",
                    bounds=bounds, options={"maxiter": maxiter, "maxfun": 4 * maxiter,
                                            "ftol": 1e-7, "gtol": 1e-4})


def _rho_background(theta, D):
    """Background probability rho_i = mu*u_i / lambda_i for every target event."""
    mu, k, alpha, c, p, d, q, gamma = _theta_unpack(theta, D["beta"])
    kap = k * np.exp(alpha * D["dm"])
    g = (p - 1.0) / c * (1.0 + D["dt"] / c) ** (-p)
    Dk = d * np.exp(gamma * LN10 * D["dm"])
    f = (q - 1.0) / (np.pi * Dk) * (1.0 + D["r2"] / Dk) ** (-q)
    trig = np.bincount(D["pi"], weights=kap * g * f, minlength=D["n_tgt"])
    bg = mu * D["u_tgt"]
    return bg / (bg + trig)


# ------------------------------------------------------------------------------ fitting
def fit_etas(catalog_df: pd.DataFrame, mc: float, fit_start, fit_end,
             aux_start=None, region: dict | None = None, *,
             tau_max_days: float = 3650.0, r_max_km: float = 150.0,
             bg_iterations: int = 2, n_restarts: int = 2,
             verbose: bool = False) -> EtasParams:
    """MLE fit of the space-time ETAS model.

    Uses events with mag >= mc and time in [fit_start, fit_end) for the likelihood;
    events in [aux_start, fit_start) act as triggering history only (aux_start=None
    means: use ALL available events before fit_start as history). Events at or after
    fit_end are dropped on entry and never looked at. fit_start/fit_end/aux_start may
    be datetimes (with a datetime catalog) or float days (with a float-time catalog).

    Approximations: pair truncation at tau_max_days / r_max_km with a truncation-
    consistent integral, and ``bg_iterations`` rounds of simplified Zhuang
    declustering (unit-weight KDE background -> fit -> weighted-KDE background ->
    refit, with leave-one-out KDE evaluation at the events themselves).
    """
    times, lat, lon, mag = _extract_catalog(catalog_df)
    t = _to_days(times, fit_start)
    T_end = _scalar_days(fit_end, fit_start)
    if T_end <= 0:
        raise ValueError("fit_end must be after fit_start")
    t_lo = -np.inf if aux_start is None else _scalar_days(aux_start, fit_start)

    ok = (np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(mag)
          & (mag >= mc - 1e-9) & (t >= t_lo) & (t < T_end))
    t, lat, lon, mag = t[ok], lat[ok], lon[ok], mag[ok]
    order = np.argsort(t, kind="stable")
    t, lat, lon, mag = t[order], lat[order], lon[order], mag[order]

    if region is None:
        pad = 0.05
        region = {"min_lat": float(lat.min() - pad), "max_lat": float(lat.max() + pad),
                  "min_lon": float(lon.min() - pad), "max_lon": float(lon.max() + pad)}
    ref_lon, ref_lat = _region_center(region)
    x, y = _project_km(lon, lat, ref_lon, ref_lat)
    dm = mag - mc

    tgt_idx = np.flatnonzero(t >= 0.0)
    n_tgt = tgt_idx.size
    if n_tgt < 50:
        raise ValueError(f"only {n_tgt} events with m>={mc} in the fit window; too few")
    b_val = _aki_b(mag[tgt_idx], mc)

    pairs, tau_used = _build_pairs(t, x, y, dm, tgt_idx, tau_max_days, r_max_km)
    D = dict(pairs)
    D["n_tgt"] = n_tgt
    D["T"] = T_end
    D["R"] = float(r_max_km) ** 2
    D["beta"] = b_val * LN10
    D["dm_src"] = dm
    D["tau2"] = np.minimum(T_end - t, tau_used)
    D["tau1"] = np.clip(-t, 0.0, tau_used)
    lon_t, lat_t = lon[tgt_idx], lat[tgt_idx]
    # stage-0 background: plain (unit-weight) KDE of the fit-window events, so the
    # stationary spatial clustering of seismicity is not forced into the triggering
    # kernel (which otherwise drives k up / p toward 1)
    bg_field = BackgroundField.from_weighted_events(lon_t, lat_t,
                                                    np.ones(n_tgt), region)
    D["u_tgt"] = bg_field.pdf_at_support_loo()
    if verbose:
        print(f"[fit_etas] n_tgt={n_tgt} n_src={t.size} pairs={D['pi'].size} "
              f"tau={tau_used:.0f}d b={b_val:.2f}")

    rate = n_tgt / T_end
    inits = [  # (mu, branching ratio n, alpha, c, p, d, q, gamma)
        _theta_pack(0.5 * rate, 0.35, 1.5, 0.01, 1.13, 1.5, 1.7, 0.4),
        _theta_pack(0.8 * rate, 0.15, 2.0, 0.03, 1.30, 5.0, 2.2, 0.1),
        _theta_pack(0.3 * rate, 0.60, 1.0, 0.005, 1.08, 0.5, 1.5, 0.8),
    ]
    best = None
    for i0 in range(max(1, min(n_restarts, len(inits)))):
        res = _fit_stage(D, inits[i0], maxiter=200)
        if verbose:
            print(f"[fit_etas] restart {i0}: nll={res.fun:.2f} nit={res.nit}")
        if best is None or res.fun < best.fun - 1e-9:
            best = res

    for it in range(max(0, bg_iterations - 1)):
        rho = _rho_background(best.x, D)
        bg_field = BackgroundField.from_weighted_events(lon_t, lat_t, rho, region)
        D["u_tgt"] = bg_field.pdf_at_support_loo()
        best = _fit_stage(D, best.x, maxiter=150)
        if verbose:
            print(f"[fit_etas] bg-iter {it + 1}: nll={best.fun:.2f} "
                  f"sum(rho)={rho.sum():.0f}")
    if not best.success:  # polish with a fresh L-BFGS-B memory from the same point
        res = _fit_stage(D, best.x, maxiter=60)
        if res.fun <= best.fun + 1e-6:
            best = res
    # final declustering pass defines the stored background field
    rho = _rho_background(best.x, D)
    bg_field = BackgroundField.from_weighted_events(lon_t, lat_t, rho, region)

    mu, k, alpha, c, p, d, q, gamma = _theta_unpack(best.x, D["beta"])
    params = EtasParams(mu_total=float(mu), background_xy=bg_field, k=float(k),
                        alpha=float(alpha), c=float(c), p=float(p), d=float(d),
                        q=float(q), gamma=float(gamma), mc=float(mc), b=float(b_val),
                        region=dict(region))
    if verbose:
        print(f"[fit_etas] mu={mu:.4f}/d k={k:.4f} alpha={alpha:.2f} c={c:.4f} "
              f"p={p:.3f} d={d:.2f} q={q:.2f} gamma={gamma:.2f} "
              f"n={branching_ratio(params):.3f} converged={best.success}")
    params._fit_result = best  # convergence info (extra attribute, not a dataclass field)
    return params


# --------------------------------------------------------------------------- forecasting
def expected_counts(params: EtasParams, history_df: pd.DataFrame, cells_df: pd.DataFrame,
                    t0, horizon_days: float = 30.0, cell_deg: float = 0.1) -> np.ndarray:
    """Expected number of M>=mc events per cell in [t0, t0 + horizon_days).

    Strictly causal: events in ``history_df`` at or after ``t0`` are filtered out
    internally, as are events below mc. First-generation approximation (offspring of
    the history only, no secondary triggering within the horizon — standard for short
    horizons; underestimates in very active sequences by up to ~1/(1-n)). The Omori
    kernel is integrated analytically over [t0, t0+horizon); the spatial kernel uses
    the cell-center approximation (pdf at center x cell area), with each source's
    total distributed mass capped at 1. Sources with negligible temporal weight
    (< 1e-9 expected events) are skipped. Includes the background term
    mu_total * u(cell) * area * horizon.
    """
    bg: BackgroundField = params.background_xy
    clon = np.asarray(cells_df["cell_lon"], float)
    clat = np.asarray(cells_df["cell_lat"], float)
    area_c = (cell_deg * KM_PER_DEG) * (cell_deg * KM_PER_DEG * np.cos(np.radians(clat)))
    out = params.mu_total * horizon_days * bg.pdf_lonlat(clon, clat) * area_c

    if history_df is None or len(history_df) == 0:
        return out
    times, lat, lon, mag = _extract_catalog(history_df)
    tau = -_to_days(times, t0)  # tau_j = t0 - t_j (days), > 0 for past events
    keep = np.isfinite(tau) & (tau > 0) & (mag >= params.mc - 1e-9)
    tau, lat, lon, mag = tau[keep], lat[keep], lon[keep], mag[keep]
    if tau.size == 0:
        return out

    c, p, q, d = params.c, params.p, params.q, params.d
    kap = params.k * np.exp(params.alpha * (mag - params.mc))
    H = lambda s: 1.0 - (1.0 + s / c) ** (1.0 - p)
    w = kap * (H(tau + horizon_days) - H(tau))
    big = w > 1e-9
    if not np.any(big):
        return out
    tau, lat, lon, mag, w = tau[big], lat[big], lon[big], mag[big], w[big]

    sx, sy = _project_km(lon, lat, bg.ref_lon, bg.ref_lat)
    cx, cy = _project_km(clon, clat, bg.ref_lon, bg.ref_lat)
    Dm = d * np.power(10.0, params.gamma * (mag - params.mc))
    chunk = max(1, int(5e6 // max(clon.size, 1)))
    for s in range(0, sx.size, chunk):
        sl = slice(s, s + chunk)
        dx = cx[None, :] - sx[sl, None]
        dy = cy[None, :] - sy[sl, None]
        r2 = dx * dx + dy * dy
        Dj = Dm[sl, None]
        M = (q - 1.0) / (np.pi * Dj) * (1.0 + r2 / Dj) ** (-q) * area_c[None, :]
        S = M.sum(axis=1)
        scale = np.where(S > 1.0, 1.0 / np.maximum(S, 1e-300), 1.0)
        out += (w[sl] * scale) @ M
    return out


def prob_at_least_one(expected: np.ndarray) -> np.ndarray:
    """Poisson probability of at least one event per cell: 1 - exp(-lambda)."""
    return 1.0 - np.exp(-np.asarray(expected, dtype=float))


# ---------------------------------------------------------------------------- simulation
def _sample_gr(n: int, rng: np.random.Generator, mc: float, b: float, mmax: float):
    """Gutenberg-Richter magnitudes truncated at mmax (inverse-CDF sampling)."""
    beta = b * LN10
    u = rng.random(n)
    return mc - np.log(1.0 - u * (1.0 - np.exp(-beta * (mmax - mc)))) / beta


def simulate_catalogs(params: EtasParams, duration_days: float, n_sims: int,
                      mmax: float = 7.6, seed: int = 0, start_time=None,
                      max_events: int = 500_000) -> list[pd.DataFrame]:
    """Full branching ETAS simulation.

    Background events: Poisson(mu_total * duration) in time, sampled spatially from
    ``params.background_xy``. Every event of every generation spawns
    Poisson(k * exp(alpha*(m - mc))) offspring with Omori-Utsu waiting times,
    power-law spatial offsets (D = d * 10^(gamma*(m-mc))), and i.i.d. truncated-GR
    magnitudes (b, mc, mmax). Offspring falling outside the region are re-sampled
    (up to 30 rounds) and finally clamped to the region boundary; offspring beyond
    ``duration_days`` are discarded (their descendants cannot occur in-window).

    Returns a list of DataFrames with columns: time (float days, or datetimes if
    ``start_time`` is given), lat, lon, mag, gen, parent_id (row index of the parent
    in the same DataFrame; -1 for background events). Each simulation uses an
    independent stream spawned from ``seed``.
    """
    reg = params.region
    bg: BackgroundField = params.background_xy
    n_tr = branching_ratio(params, mmax=mmax)
    if n_tr >= 1.0:
        warnings.warn(f"branching ratio {n_tr:.2f} >= 1 (supercritical): simulation "
                      f"will be truncated at max_events={max_events}")
    beta_c, p_c, q_c = params.c, params.p, params.q
    seeds = np.random.SeedSequence(seed).spawn(n_sims)
    cats = []
    for isim in range(n_sims):
        rng = np.random.default_rng(seeds[isim])
        nbg = rng.poisson(params.mu_total * duration_days)
        t = rng.uniform(0.0, duration_days, nbg)
        lon, lat = bg.sample(nbg, rng)
        mag = _sample_gr(nbg, rng, params.mc, params.b, mmax)
        T = [t]; LO = [lon]; LA = [lat]; MA = [mag]
        GE = [np.zeros(nbg, dtype=np.int32)]
        PA = [np.full(nbg, -1, dtype=np.int64)]
        total = nbg
        next_id = nbg
        f_t, f_lon, f_lat, f_mag = t, lon, lat, mag
        f_ids = np.arange(nbg, dtype=np.int64)
        gen = 0
        while f_t.size and total < max_events:
            gen += 1
            kap = params.k * np.exp(params.alpha * (f_mag - params.mc))
            noff = rng.poisson(kap)
            if noff.sum() == 0:
                break
            pid = np.repeat(f_ids, noff)
            pt = np.repeat(f_t, noff)
            pm = np.repeat(f_mag, noff)
            px, py = _project_km(np.repeat(f_lon, noff), np.repeat(f_lat, noff),
                                 bg.ref_lon, bg.ref_lat)
            u = rng.random(pt.size)
            dt = beta_c * (u ** (-1.0 / (p_c - 1.0)) - 1.0)
            tc = pt + dt
            keep = tc < duration_days
            if not np.any(keep):
                break
            pid, tc, pm, px, py = pid[keep], tc[keep], pm[keep], px[keep], py[keep]
            m_child = _sample_gr(tc.size, rng, params.mc, params.b, mmax)
            Dp = params.d * np.power(10.0, params.gamma * (pm - params.mc))
            lo_c = np.empty(tc.size); la_c = np.empty(tc.size)
            todo = np.arange(tc.size)
            for _ in range(30):
                if todo.size == 0:
                    break
                v = rng.random(todo.size)
                r = np.sqrt(Dp[todo] * (v ** (-1.0 / (q_c - 1.0)) - 1.0))
                th = rng.uniform(0.0, 2.0 * np.pi, todo.size)
                lo_t, la_t = _unproject_km(px[todo] + r * np.cos(th),
                                           py[todo] + r * np.sin(th),
                                           bg.ref_lon, bg.ref_lat)
                ok = ((lo_t >= reg["min_lon"]) & (lo_t <= reg["max_lon"])
                      & (la_t >= reg["min_lat"]) & (la_t <= reg["max_lat"]))
                lo_c[todo[ok]] = lo_t[ok]
                la_c[todo[ok]] = la_t[ok]
                todo = todo[~ok]
            if todo.size:  # stubborn far-field offspring: clamp to region
                lo_t, la_t = _unproject_km(px[todo], py[todo], bg.ref_lon, bg.ref_lat)
                lo_c[todo] = np.clip(lo_t, reg["min_lon"], reg["max_lon"])
                la_c[todo] = np.clip(la_t, reg["min_lat"], reg["max_lat"])
            if total + tc.size > max_events:
                tc = tc[: max_events - total]
                pid, m_child = pid[: tc.size], m_child[: tc.size]
                lo_c, la_c = lo_c[: tc.size], la_c[: tc.size]
                warnings.warn("simulate_catalogs: max_events reached, cascade truncated")
            ids = next_id + np.arange(tc.size, dtype=np.int64)
            next_id += tc.size
            total += tc.size
            T.append(tc); LO.append(lo_c); LA.append(la_c); MA.append(m_child)
            GE.append(np.full(tc.size, gen, dtype=np.int32)); PA.append(pid)
            f_t, f_lon, f_lat, f_mag, f_ids = tc, lo_c, la_c, m_child, ids
        t_all = np.concatenate(T); lon_all = np.concatenate(LO)
        lat_all = np.concatenate(LA); mag_all = np.concatenate(MA)
        gen_all = np.concatenate(GE); par_all = np.concatenate(PA)
        order = np.argsort(t_all, kind="stable")
        newpos = np.empty(t_all.size, dtype=np.int64)
        newpos[order] = np.arange(t_all.size)
        par_row = np.where(par_all >= 0, newpos[np.maximum(par_all, 0)], -1)
        time_col = t_all[order]
        if start_time is not None:
            time_col = pd.Timestamp(start_time) + pd.to_timedelta(time_col, unit="D")
        cats.append(pd.DataFrame({
            "time": time_col, "lat": lat_all[order], "lon": lon_all[order],
            "mag": mag_all[order], "gen": gen_all[order], "parent_id": par_row[order],
        }))
    return cats
