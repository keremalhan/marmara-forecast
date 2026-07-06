"""Validation tests for marmara.stress (Coulomb stress-transfer module).

Run with the project venv, either

    python -m marmara.tests.test_stress          (from the repo root), or
    python python -m marmara.tests.test_stress.py          (any cwd), or
    pytest python -m marmara.tests.test_stress.py          (if pytest is installed)

Total runtime is a few seconds (<60 s requirement).

Validation strategy
-------------------
* If a functional ``okada_wrapper`` (f2py DC3D) build is importable, the
  numpy engine is compared against it directly (5% tolerance). On this
  machine the pip build is broken (no Fortran compiler), so that test
  auto-skips.
* Independent oracle that DOES run here: ``cutde`` (Nikkhoo & Walter 2015
  triangular dislocations, half-space). A rectangle is meshed as two
  triangles with upward-pointing normals; displacements must match the
  DC3D transcription essentially to machine precision, strains (via the
  50 m finite-difference pipeline) to ~1e-4 relative, and delta-CFS to
  within 5% at 20 random points. Slip-sign conventions are pinned by a
  physical anchor (reverse slip on a 45-degree thrust uplifts the surface
  above the hanging wall), not by fitting one code to the other.
* Closed-form anchor: for a very long vertical strike-slip fault the
  surface displacement must approach the 2-D screw-dislocation solution
  u = (s/pi) * arctan(D/y).
* Physics anchors: traction-free free surface; classic right-lateral
  Coulomb lobe pattern; 180-degree rotational symmetry; Izmit-scale
  amplitudes; strict time causality; exact half-life halving; per-point
  receiver resolution onto the project's 97-segment fault model.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import sys

import numpy as np
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marmara import stress  # noqa: E402
from marmara.stress import (  # noqa: E402
    ALPHA,
    DEFAULT_RUPTURES,
    LAMBDA_PA,
    MU_PA,
    CoulombStressModel,
    dc3d_displacement,
    local_km_to_lonlat,
    nearest_segment_receivers,
)

SEGMENTS_JSON = os.path.join(_ROOT, "results", "segment_properties.json")

T0 = dt.datetime(2000, 1, 1)


def _synth_rupture(**kw):
    base = dict(name="synth", origin_time=T0, lat=40.9, lon=28.5,
                depth_top_km=0.0, strike_deg=270.0, dip_deg=90.0,
                rake_deg=180.0, length_km=60.0, width_km=15.0, slip_m=3.0)
    base.update(kw)
    return base


def _model(rupture, **kw):
    kw.setdefault("receiver_strike", rupture["strike_deg"])
    kw.setdefault("receiver_dip", rupture["dip_deg"])
    kw.setdefault("receiver_rake", rupture["rake_deg"])
    m = CoulombStressModel(ruptures=[rupture], **kw)
    m.engine = "numpy"
    return m


# ======================================================================
# (a) okada_wrapper cross-check -- runs only if the f2py build works
# ======================================================================
def test_okada_wrapper_crosscheck():
    if stress._OKADA_WRAPPER_FN is None:
        print("  [SKIP] okada_wrapper not functional in this venv "
              "(pip build lacks compiled Fortran); cutde cross-check "
              "below provides the independent oracle.")
        return
    rup = _synth_rupture(depth_top_km=2.0, dip_deg=75.0, rake_deg=170.0)
    m = _model(rup, receiver_strike=250.0, receiver_dip=70.0,
               receiver_rake=170.0)
    rng = np.random.default_rng(42)
    xe = rng.uniform(-80.0, 80.0, 20)
    yn = rng.uniform(12.0, 60.0, 20) * rng.choice([-1.0, 1.0], 20)
    lons, lats = local_km_to_lonlat(xe, yn)
    m.engine = "numpy"
    v_np = m.delta_cfs(lons, lats)
    m.engine = "okada_wrapper"
    v_ok = m.delta_cfs(lons, lats)
    scale = np.abs(v_ok).max()
    rel = np.abs(v_np - v_ok) / np.maximum(np.abs(v_ok), 0.05 * scale)
    assert rel.max() < 0.05, f"numpy vs okada_wrapper mismatch: {rel.max()}"
    print(f"  numpy vs okada_wrapper: max rel diff {rel.max():.2e} (<5%)")


# ======================================================================
# cutde (Nikkhoo & Walter TDE) cross-checks -- independent oracle
# ======================================================================
def _cutde_or_none():
    try:
        import cutde.halfspace as HS  # noqa: PLC0415
        return HS
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] cutde unavailable ({type(exc).__name__})")
        return None


def _rect_tris(L, top, W, dip_deg):
    """Rectangle (fault-local frame: x along strike, dips toward -y) as two
    triangles with UPWARD normals, as cutde requires."""
    sd, cd = math.sin(math.radians(dip_deg)), math.cos(math.radians(dip_deg))
    A = [-L / 2.0, 0.0, -top]
    B = [L / 2.0, 0.0, -top]
    C = [L / 2.0, -W * cd, -(top + W * sd)]
    D = [-L / 2.0, -W * cd, -(top + W * sd)]
    return np.array([[A, C, B], [A, D, C]])


def test_cutde_displacement_and_sign_anchor():
    HS = _cutde_or_none()
    if HS is None:
        return
    L, top, W, dip, rake = 20.0, 2.0, 10.0, 60.0, 135.0
    d1 = math.cos(math.radians(rake))
    d2 = math.sin(math.radians(rake))
    pts = np.array([
        [15.0, 5.0, -6.0], [0.0, 8.0, -4.0], [25.0, 0.1, -6.0],
        [-14.0, -9.0, -9.0], [4.0, -12.0, -3.0], [-3.0, 3.0, -11.0],
        [30.0, 20.0, -10.0], [-22.0, 14.0, -5.0],
    ])
    u1, u2, u3 = dc3d_displacement(pts[:, 0], pts[:, 1], pts[:, 2],
                                   depth_km=top, dip_deg=dip,
                                   al1=-L / 2, al2=L / 2, aw1=-W, aw2=0.0,
                                   disl1=d1, disl2=d2)
    mine = np.stack([u1, u2, u3], axis=1)
    tris = _rect_tris(L, top, W, dip)
    cu = HS.disp_free(pts, tris, np.array([[d1, d2, 0.0]] * 2), 0.25)
    diff = np.abs(cu - mine).max()
    assert diff < 1e-10 * max(1.0, np.abs(mine).max()) + 1e-12, \
        f"DC3D vs cutde displacement mismatch: {diff}"
    print(f"  DC3D vs cutde displacements: max abs diff {diff:.2e}")

    # physical sign anchor: reverse slip (rake +90) on a 45-degree thrust
    # uplifts the free surface above the hanging wall (-y side here).
    sd45 = math.sin(math.radians(45.0))
    cd45 = math.cos(math.radians(45.0))
    hw = np.array([[0.0, -(3.0 + 5.0 * sd45) * cd45 / sd45, 0.0]])  # over mid-plane
    _, _, uz_mine = dc3d_displacement(hw[:, 0], hw[:, 1], hw[:, 2],
                                      depth_km=3.0, dip_deg=45.0,
                                      al1=-10.0, al2=10.0, aw1=-10.0,
                                      aw2=0.0, disl1=0.0, disl2=1.0)
    tris45 = _rect_tris(20.0, 3.0, 10.0, 45.0)
    uz_cu = HS.disp_free(hw, tris45, np.array([[0.0, 1.0, 0.0]] * 2), 0.25)[0, 2]
    assert uz_mine[0] > 0.0 and uz_cu > 0.0, \
        f"thrust must uplift hanging wall: mine {uz_mine[0]}, cutde {uz_cu}"
    print(f"  thrust uplift anchor: uz mine {uz_mine[0]:+.4f}, "
          f"cutde {uz_cu:+.4f} (both > 0)")


def test_cutde_delta_cfs_20_points():
    """delta-CFS field vs an independently assembled cutde-strain CFS at 20
    random points -- must agree within 5% (requirement 5a, with cutde as
    the functional oracle since the okada_wrapper build is broken)."""
    HS = _cutde_or_none()
    if HS is None:
        return
    L, top, W, dip = 30.0, 1.0, 14.0, 80.0
    rake = 180.0
    rup = _synth_rupture(depth_top_km=top, dip_deg=dip, rake_deg=rake,
                         length_km=L, width_km=W, slip_m=2.0,
                         strike_deg=90.0)  # strike 90: local frame == ENU
    m = _model(rup, receiver_strike=265.0, receiver_dip=80.0,
               receiver_rake=180.0, receiver_depth_km=10.0)
    rng = np.random.default_rng(7)
    xe = rng.uniform(-70.0, 70.0, 20)
    yn = rng.uniform(8.0, 50.0, 20) * rng.choice([-1.0, 1.0], 20)
    lons, lats = local_km_to_lonlat(xe, yn)
    v_mine = m.delta_cfs(lons, lats)

    # independent oracle: cutde strain -> Hooke -> resolve (built inline)
    d1 = 0.002 * math.cos(math.radians(rake))   # slip in km
    d2 = 0.002 * math.sin(math.radians(rake))
    pts = np.column_stack([xe, yn, np.full_like(xe, -10.0)])
    es = HS.strain_free(pts, _rect_tris(L, top, W, dip),
                        np.array([[d1, d2, 0.0]] * 2), 0.25)
    eps = np.zeros((20, 3, 3))
    eps[:, 0, 0] = es[:, 0]
    eps[:, 1, 1] = es[:, 1]
    eps[:, 2, 2] = es[:, 2]
    eps[:, 0, 1] = eps[:, 1, 0] = es[:, 3]
    eps[:, 0, 2] = eps[:, 2, 0] = es[:, 4]
    eps[:, 1, 2] = eps[:, 2, 1] = es[:, 5]
    tr = np.trace(eps, axis1=1, axis2=2)
    sig = 2.0 * MU_PA * eps
    for i in range(3):
        sig[:, i, i] += LAMBDA_PA * tr
    sig /= 1e6                                     # MPa
    phi, dl, lm = map(math.radians, (265.0, 80.0, 180.0))
    s_hat = np.array([math.sin(phi), math.cos(phi), 0.0])
    d_hat = np.array([math.cos(dl) * math.cos(phi),
                      -math.cos(dl) * math.sin(phi), -math.sin(dl)])
    n_hat = np.array([math.sin(dl) * math.cos(phi),
                      -math.sin(dl) * math.sin(phi), math.cos(dl)])
    u_hat = math.cos(lm) * s_hat - math.sin(lm) * d_hat
    tvec = np.einsum("nij,j->ni", sig, n_hat)
    v_oracle = tvec @ u_hat + 0.4 * (tvec @ n_hat)

    scale = np.abs(v_oracle).max()
    rel = np.abs(v_mine - v_oracle) / np.maximum(np.abs(v_oracle),
                                                 0.05 * scale)
    assert rel.max() < 0.05, f"delta_cfs vs cutde oracle: {rel.max()}"
    print(f"  delta_cfs vs cutde oracle at 20 pts: max rel diff "
          f"{rel.max():.2e} (<5%); max |dCFS| {scale:.3f} MPa")


# ======================================================================
# closed-form and physics anchors
# ======================================================================
def test_screw_dislocation_surface_limit():
    """Long vertical strike-slip fault -> 2-D screw dislocation:
    u_strike(surface, y) = (s/pi) * arctan(D/y); for left-lateral disl1=+s
    (hanging wall = right of strike) the +y side moves in -x."""
    s, D, L = 1.0, 15.0, 4000.0
    ys = np.array([3.0, 5.0, 15.0, 45.0])
    u1, _, _ = dc3d_displacement(np.zeros_like(ys), ys, np.zeros_like(ys),
                                 depth_km=0.0, dip_deg=90.0,
                                 al1=-L / 2, al2=L / 2, aw1=-D, aw2=0.0,
                                 disl1=s, disl2=0.0)
    analytic = -(s / math.pi) * np.arctan(D / ys)
    rel = np.abs(u1 - analytic) / np.abs(analytic)
    assert rel.max() < 2e-3, f"screw-dislocation limit failed: {rel}"
    print(f"  screw-dislocation surface limit: max rel err {rel.max():.2e}")


def test_free_surface_traction():
    """sigma_xz, sigma_yz, sigma_zz must (nearly) vanish just below the
    free surface -- validates the image/B/C parts of the transcription."""
    rup = _synth_rupture(depth_top_km=2.0, dip_deg=70.0, rake_deg=160.0,
                         length_km=30.0, width_km=12.0, slip_m=2.0,
                         strike_deg=90.0)
    m = _model(rup)
    az = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    for r, tol in ((15.0, 0.10), (30.0, 0.03)):
        lons, lats = local_km_to_lonlat(r * np.sin(az), r * np.cos(az))
        sig = m.stress_tensor(lons, lats, depth_km=0.05)
        frob = np.linalg.norm(sig, axis=(1, 2))
        ratio = (np.abs(sig[:, :, 2]).max(axis=1) / frob).max()
        assert ratio < tol, f"free-surface traction ratio {ratio} at r={r}"
        print(f"  free-surface traction, ring {r:.0f} km: "
              f"max |sigma_iz|/||sigma|| = {ratio:.4f} (< {tol})")


def test_lobe_signs():
    """(b)(i) Right-lateral E-W strike-slip, parallel receiver: positive
    dCFS beyond the fault tips, negative in the fault-normal quadrants."""
    rup = _synth_rupture()          # E-W (strike 270), right-lateral, L=60
    m = _model(rup, receiver_depth_km=10.0)
    xs = np.array([40.0, -40.0, 45.0, -45.0, 0.0, 0.0, 0.0, 0.0])
    ys = np.array([0.0, 0.0, 3.0, -3.0, 15.0, -15.0, 25.0, -25.0])
    lons, lats = local_km_to_lonlat(xs, ys)
    v = m.delta_cfs(lons, lats)
    assert np.all(v[:4] > 0.0), f"beyond-tip lobes must be positive: {v[:4]}"
    assert np.all(v[4:] < 0.0), f"fault-normal lobes must be negative: {v[4:]}"
    print(f"  lobes: tips {v[:4].round(3)} MPa > 0; "
          f"normal {v[4:].round(3)} MPa < 0")


def test_rotational_symmetry():
    """(b)(ii) Vertical strike-slip source + parallel vertical receiver:
    the dCFS field is symmetric under 180-degree rotation about the fault
    center."""
    rup = _synth_rupture(length_km=120.0, width_km=17.0, slip_m=4.0)
    m = _model(rup)
    rng = np.random.default_rng(1)
    xo = rng.uniform(-80.0, 80.0, 24)
    yo = rng.uniform(5.0, 60.0, 24) * rng.choice([-1.0, 1.0], 24)
    la, lb = local_km_to_lonlat(xo, yo)
    lc, ld = local_km_to_lonlat(-xo, -yo)
    va = m.delta_cfs(la, lb)
    vb = m.delta_cfs(lc, ld)
    rel = np.abs(va - vb).max() / np.abs(va).max()
    assert rel < 1e-9, f"180-degree symmetry violated: {rel}"
    print(f"  180-degree rotational symmetry: max rel asymmetry {rel:.2e}")


def test_amplitude_izmit():
    """(b)(iii) The default Izmit Mw7.6 source produces |dCFS| of order
    0.1-3 MPa at 20-40 km from the rupture, decaying with distance."""
    izmit = [r for r in DEFAULT_RUPTURES if r["name"].startswith("Izmit1999")]
    assert 3 <= len(izmit) <= 4
    m = CoulombStressModel(ruptures=izmit)
    m.engine = "numpy"
    # probes 23-27 km from the rupture (N, S, and beyond the west tip)
    probes = [(29.30, 40.72), (30.00, 40.95), (30.45, 40.49), (30.00, 40.50)]
    lons = np.array([p[0] for p in probes])
    lats = np.array([p[1] for p in probes])
    v = m.delta_cfs(lons, lats)
    amax = np.abs(v).max()
    assert 0.1 <= amax <= 3.0, f"Izmit amplitude at 25-30 km: {v} MPa"
    assert np.all(np.abs(v) <= 3.0)
    # decay with distance (same azimuth, ~28 km -> ~78 km north)
    near = m.delta_cfs(np.array([30.0]), np.array([40.95]))
    far = m.delta_cfs(np.array([30.0]), np.array([41.40]))
    assert np.abs(far[0]) < np.abs(near[0]), (near, far)
    print(f"  Izmit amplitudes at ~25-30 km: {v.round(3)} MPa "
          f"(max {amax:.2f}); decay {near[0]:+.3f} -> {far[0]:+.3f} MPa")


# ======================================================================
# time behaviour
# ======================================================================
def test_causality():
    """(b)(iv) delta_cfs(at_time=1998) excludes Izmit; the contribution
    appears only after 1999-08-17; at_time exactly at origin -> excluded."""
    m = CoulombStressModel()          # all DEFAULT_RUPTURES
    m.engine = "numpy"
    lons = np.array([29.8, 30.2, 28.6])
    lats = np.array([40.85, 40.60, 40.80])
    izmit_origin = dt.datetime(1999, 8, 17, 0, 1, 39)

    v1998 = m.delta_cfs(lons, lats, at_time=dt.datetime(1998, 1, 1))
    assert np.all(v1998 == 0.0), f"pre-1999 field must be zero: {v1998}"

    v_at_origin = m.delta_cfs(lons, lats, at_time=izmit_origin)
    assert np.all(v_at_origin == 0.0), "strict causality: 0 AT origin time"

    v_sep99 = m.delta_cfs(lons, lats, at_time=dt.datetime(1999, 9, 1))
    assert np.any(v_sep99 != 0.0)
    izmit_only = CoulombStressModel(
        ruptures=[r for r in DEFAULT_RUPTURES
                  if r["name"].startswith("Izmit1999")])
    izmit_only.engine = "numpy"
    v_izmit = izmit_only.delta_cfs(lons, lats)
    assert np.allclose(v_sep99, v_izmit, rtol=0.0, atol=1e-12), \
        "Sep-1999 field must equal the Izmit-only field (Duzce et al. later)"

    v_2000 = m.delta_cfs(lons, lats, at_time=dt.datetime(2000, 6, 1))
    assert not np.allclose(v_2000, v_sep99), "Duzce must add after Nov 1999"

    v_all = m.delta_cfs(lons, lats)   # at_time=None -> ALL ruptures
    v_future = m.delta_cfs(lons, lats, at_time=dt.datetime(2030, 1, 1))
    assert np.allclose(v_all, v_future, rtol=0.0, atol=1e-12)
    print(f"  causality: 1998 all-zero; Sep-1999 == Izmit-only "
          f"({v_sep99.round(4)}); Duzce adds later; None == all")


def test_half_life_decay():
    """(b)(v) One half-life halves a rupture's contribution."""
    rup = dict([r for r in DEFAULT_RUPTURES
                if r["name"] == "Kumburgaz2025"][0])
    m = CoulombStressModel(ruptures=[rup])
    m.engine = "numpy"
    lons = np.array([28.5, 28.9, 28.0])
    lats = np.array([40.75, 40.85, 40.90])
    ten_years = dt.timedelta(days=3652.5)       # exactly 10 yr of 365.25 d
    t_eval = rup["origin_time"] + ten_years
    v_plain = m.delta_cfs(lons, lats, at_time=t_eval)
    v_decay = m.delta_cfs(lons, lats, at_time=t_eval, half_life_years=10.0)
    assert np.all(v_plain != 0.0)
    assert np.allclose(v_decay, 0.5 * v_plain, rtol=1e-9), \
        f"one half-life must halve: {v_decay / v_plain}"
    print(f"  half-life: ratio after one half-life = "
          f"{(v_decay / v_plain).round(6)} (= 0.5)")


# ======================================================================
# per-point receivers
# ======================================================================
def test_per_point_receivers_match_scalar():
    """Per-point receiver arrays equal to the scalar default must give
    bit-identical output; points near differently-oriented catalog
    segments must resolve onto different receivers."""
    m = CoulombStressModel()
    m.engine = "numpy"
    lons = np.array([28.4, 29.0, 29.6, 30.2])
    lats = np.array([40.85, 40.75, 40.70, 40.72])
    v_scalar = m.delta_cfs(lons, lats)
    n = lons.size
    v_arr = m.delta_cfs(lons, lats,
                        receiver_strike=np.full(n, 265.0),
                        receiver_dip=np.full(n, 80.0),
                        receiver_rake=np.full(n, 180.0))
    assert np.allclose(v_scalar, v_arr, rtol=0.0, atol=0.0), \
        "array receivers identical to scalar defaults must match exactly"
    print("  per-point receivers == scalar defaults: identical output")

    if not os.path.exists(SEGMENTS_JSON):
        print(f"  [SKIP] {SEGMENTS_JSON} not found for nearest-segment part")
        return
    # one point near the NW (Thrace) sinistral-normal segment, one in the
    # Marmara Sea near the main branch: orientations must differ.
    p_lons = np.array([26.24, 28.60])
    p_lats = np.array([41.87, 40.85])
    st, di, ra, dist = nearest_segment_receivers(p_lons, p_lats,
                                                 SEGMENTS_JSON)
    assert dist[0] < 25.0 and dist[1] < 25.0
    assert (abs(st[0] - st[1]) > 5.0 or abs(di[0] - di[1]) > 5.0
            or abs(ra[0] - ra[1]) > 5.0), \
        f"expected different receivers, got {st, di, ra}"
    v_default = m.delta_cfs(p_lons, p_lats)
    v_nearest = m.delta_cfs(p_lons, p_lats, receiver_strike=st,
                            receiver_dip=di, receiver_rake=ra)
    assert not np.allclose(v_default, v_nearest), \
        "differently-oriented receivers must change dCFS"
    print(f"  nearest-segment receivers differ: "
          f"({st[0]:.0f}/{di[0]:.0f}/{ra[0]:.0f}) vs "
          f"({st[1]:.0f}/{di[1]:.0f}/{ra[1]:.0f}); dCFS "
          f"{v_default.round(4)} -> {v_nearest.round(4)} MPa")


def test_nearest_segment_fallback():
    """Points > 25 km from every catalog segment fall back to the regional
    default receiver (265/80/180); the true distance is still returned."""
    if not os.path.exists(SEGMENTS_JSON):
        print(f"  [SKIP] {SEGMENTS_JSON} not found")
        return
    st, di, ra, dist = nearest_segment_receivers(
        np.array([34.0]), np.array([43.5]), SEGMENTS_JSON)  # Black Sea
    assert dist[0] > 25.0
    assert (st[0], di[0], ra[0]) == (265.0, 80.0, 180.0)
    print(f"  fallback: distance {dist[0]:.0f} km -> default receiver "
          f"(265/80/180)")


# ======================================================================
# (c) literature cross-check -- reported, not hard-asserted
# ======================================================================
def test_princes_islands_report():
    """(5c) Literature cross-check. Martinez-Garzon et al. (2025, Science,
    doi:10.1126/science.adz0072) report that the 2019 Mw5.8 + 2025 Mw6.2
    events raised Coulomb stress by up to ~0.2 MPa (2 bar) on adjacent MMF
    sections, particularly at the WESTERN edge of the Princes' Islands
    segment (~28.6-28.7 E). Stein & Sevilgen (Temblor, Apr 2025) report the
    same pattern qualitatively ("significantly closer to failure").
    We print our model's numbers there (no hard assert)."""
    m = CoulombStressModel(ruptures=[r for r in DEFAULT_RUPTURES
                                     if r["name"] in ("Silivri2019",
                                                      "Kumburgaz2025")])
    m.engine = "numpy"
    lons = np.array([28.55, 28.65, 28.70, 28.90, 29.00, 29.10])
    lats = np.array([40.86, 40.84, 40.85, 40.85, 40.85, 40.84])
    v = m.delta_cfs(lons, lats)
    print("  Princes' Islands segment dCFS from 2019+2025 events (MPa):")
    for lo, la, val in zip(lons, lats, v):
        print(f"    ({lo:.2f}E, {la:.2f}N): {val:+.4f}")
    print(f"    western-edge max {v[:3].max():+.3f} MPa vs literature "
          f"'up to ~0.2 MPa' (Martinez-Garzon et al. 2025 Science; same "
          f"order of magnitude expected)")
    assert np.all(np.isfinite(v))


# ======================================================================
# runner
# ======================================================================
def main():
    tests = [
        ("okada_wrapper cross-check (5a)", test_okada_wrapper_crosscheck),
        ("cutde displacement + sign anchor", test_cutde_displacement_and_sign_anchor),
        ("cutde delta-CFS 20 points (5a oracle)", test_cutde_delta_cfs_20_points),
        ("screw-dislocation surface limit", test_screw_dislocation_surface_limit),
        ("free-surface traction", test_free_surface_traction),
        ("lobe signs (5b-i)", test_lobe_signs),
        ("rotational symmetry (5b-ii)", test_rotational_symmetry),
        ("Izmit amplitude (5b-iii)", test_amplitude_izmit),
        ("causality (5b-iv)", test_causality),
        ("half-life decay (5b-v)", test_half_life_decay),
        ("per-point receivers", test_per_point_receivers_match_scalar),
        ("nearest-segment fallback", test_nearest_segment_fallback),
        ("Princes' Islands cross-check (5c)", test_princes_islands_report),
    ]
    failed = []
    for label, fn in tests:
        print(f"[TEST] {label}")
        try:
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed.append(label)
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(label)
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print("=" * 60)
    if failed:
        print(f"{len(failed)} test(s) FAILED: {failed}")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
