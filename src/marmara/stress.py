"""Coulomb stress-transfer (delta-CFS) module for the Marmara region.

Computes the static Coulomb failure stress change

    dCFS = d_tau_rake + mu' * d_sigma_n      [MPa, tension-positive sigma_n]

at depth in a uniform elastic half-space (lambda = mu = 30 GPa, Poisson
ratio 0.25), produced by a set of rectangular fault ruptures, resolved onto
a receiver fault plane (per-point receiver orientations supported).

Implementation route
--------------------
Internal (at-depth) displacements are computed with a pure-numpy, vectorized
transcription of Y. Okada's DC3D code:

    Okada, Y. (1992), "Internal deformation due to shear and tensile faults
    in a half-space", Bull. Seismol. Soc. Am. 82(2), 1018-1040.
    (surface-only precursor: Okada, Y. (1985), BSSA 75, 1135-1154)

The transcription (parts A/B/C, Chinnery corner differences, image source)
follows the reference Fortran DC3D.f (Okada, coded Sep. 1991, rev. May 2002)
verbatim; strike-slip and dip-slip elementary dislocations are implemented
(tensile opening is not). Strains are obtained by central finite differences
of the internal displacement field with step ``FD_STEP_KM`` = 0.05 km (50 m),
then converted to stress via Hooke's law. If a *functional* build of
``okada_wrapper`` (f2py DC3D bindings, github.com/cutde-org/okada_wrapper)
is importable, it is auto-detected and used instead (analytic gradients, no
finite differences); otherwise the numpy engine is used. The engine in use
is exposed as ``CoulombStressModel.engine`` and may be overridden by
assigning ``model.engine = "numpy"``.

Conventions and units
---------------------
* Local coordinates: flat-earth (transverse-Mercator-style) approximation
  about (40.9 N, 28.5 E); x = East [km], y = North [km], z = Up [km]
  (observation depths are z < 0). 1 deg lat = 111.195 km; 1 deg lon =
  111.195 * cos(40.9 deg) km. Adequate for the ~300 km Marmara aperture.
* Fault angles in degrees, Aki & Richards (1980) convention: strike
  clockwise from North with the fault dipping to the RIGHT of strike;
  rake = angle of hanging-wall slip measured CCW from strike in the fault
  plane (0 = left-lateral, 180/-180 = right-lateral, +90 = reverse).
* Rupture geometry: each rupture dict gives the center of the TOP edge of
  the rectangle (lon, lat), its depth ``depth_top_km``, and along-strike
  length / down-dip width in km; uniform slip in meters.
* Stress in MPa; sigma_n is tension-positive, so unclamping (positive
  d_sigma_n) promotes failure: dCFS = tau_rake + friction * sigma_n.
* Effective friction coefficient mu' = 0.4 by default (King, Stein & Lin,
  1994, BSSA 84, 935-953).

Sources for DEFAULT_RUPTURES (see each dict's ``reference`` field)
------------------------------------------------------------------
* 1999-08-17 00:01:39 UTC Izmit (Kocaeli) Mw7.6 (USGS usp0009d4z),
  4-segment right-lateral model: surface-rupture segmentation and slip of
  Barka et al. (2002), BSSA 92(1), 43-60, doi:10.1785/0120000841; InSAR/
  geodetic model of Cakir et al. (2003), GJI 155(1), 93-110 (near-vertical
  segments, ~18 km locking depth, coseismic Mo ~1.9e20 Nm); GPS model of
  Reilinger et al. (2000), Science 289, 1519-1524. Segment-average slips
  integrated from the published surface-slip curves; summed moment of the
  4 rectangles = 1.8e20 Nm (Mw 7.5). The low-slip (<=1.5 m) submarine
  Yalova-Hersek western extension of Cakir et al. is NOT included (kept to
  the 3-4 published high-slip segments; see Known limitations).
* 1999-11-12 16:57:20 UTC Duzce Mw7.2 (usp0009hev): Utkucu et al. (2003),
  GJI 153(1), 229-241 (strike 262, dip 65N, 40x20 km, Mo 5.5e19 Nm);
  Umutlu et al. (2004), Tectonophysics 391, 315-324 (264/64/-172);
  Burgmann et al. (2002), BSSA 92(1), 161-171; Konca et al. (2010),
  BSSA 100(1), 267-288.
* 2019-09-26 10:59:25 UTC Silivri (Marmara Sea) Mw5.8 (us70005lkl):
  Karabulut et al. (2021), GJI 224(1), 377-388 (preferred plane
  281/60/165 +-10 deg, centroid 11.6 km, ~8 km bilateral rupture at
  11-13 km depth on a N-dipping oblique fault just north of the Main
  Marmara Fault); slip 1.0 m from Mo ~6.3e17 Nm (Mw 5.8) with mu = 30 GPa.
  (Agency solutions -- USGS Mww 5.7, GFZ, AFAD -- and Irmak et al. 2021
  give oblique thrust/strike-slip alternatives; no vertical pure-SS
  solution was published.)
* 2025-04-23 09:49:11 UTC Sea of Marmara / Kumburgaz segment Mw6.2
  (us7000pufs): USGS W-phase MT (NP1 82.6/76.6/178, Mo 2.89e18 Nm,
  centroid 11.5 km); AFAD report (84/81/171, hypocenter 40.86N 28.244E
  6.9 km); Ertuncay et al. (2025), Seismica 4(2), doi:10.26443/
  seismica.v4i2.1773; Martinez-Garzon et al. (2025), Science,
  doi:10.1126/science.adz0072 (~10-20 km rupture of the central-MMF
  Kumburgaz segment, propagating eastward); Eken et al. (2025),
  J. Seismology 29, 1727-1755. Slip 0.40 m = Mo/(mu*L*W); no surface
  rupture, top of faulting inferred at ~5 km.
* Coulomb cross-check (test 5c): Martinez-Garzon et al. (2025, Science,
  adz0072) report the 2019+2025 events raised Coulomb stress by up to
  ~0.2 MPa (2 bar) at the WESTERN edge of the Princes' Islands segment
  (~28.6-28.7 E). Stein & Sevilgen (Temblor, Apr 2025,
  temblor.net/earthquake-insights/...-16755/) report the same pattern
  qualitatively ("brought significantly closer to failure").

Time behaviour
--------------
``delta_cfs`` is time-causal: a rupture contributes only if its
``origin_time`` < ``at_time`` (strict). If ``half_life_years`` is given,
each rupture's contribution is scaled by 2**(-dt/half_life) with dt in
years at ``at_time`` -- i.e. one half-life halves the contribution (this is
the "half-life" reading of the requested exp decay; exp(-dt ln2 / T_half)).

Known limitations
-----------------
* Uniform-slip rectangles (no tapered slip); half-space (no layering,
  no topography); static elastic only (no viscoelastic relaxation --
  the optional half-life factor is a phenomenological proxy).
* The Izmit model keeps the 3-4 published high-slip segments; the low-slip
  (<=1.5 m, 10-15 km deep) submarine Yalova-Hersek western extension of
  Cakir et al. (2003) is omitted, which slightly UNDERestimates the 1999
  stress step in the easternmost Marmara Sea (Yalova-Cinarcik area).
* Points lying exactly ON a source rectangle (or its edges) are singular;
  the displacement field is discontinuous across the rupture surface, so
  dCFS within ~1 FD step (50 m) of a source plane is unreliable.
* Receiver depth must be >= 0.05 km so the vertical FD stencil stays in
  the half-space.
* Tensile (opening) sources are not implemented.

This module is self-contained (numpy only) and imports nothing from the
legacy project.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import math as _math

import numpy as _np

__all__ = [
    "DEFAULT_RUPTURES",
    "CoulombStressModel",
    "nearest_segment_receivers",
    "lonlat_to_local_km",
    "local_km_to_lonlat",
]

# ----------------------------------------------------------------------
# Elastic medium (uniform half-space)
# ----------------------------------------------------------------------
LAMBDA_PA = 30.0e9  #: Lame lambda [Pa]
MU_PA = 30.0e9      #: shear modulus [Pa]
#: Okada's medium constant alpha = (lambda + mu) / (lambda + 2 mu)  (= 2/3)
ALPHA = (LAMBDA_PA + MU_PA) / (LAMBDA_PA + 2.0 * MU_PA)
POISSON = LAMBDA_PA / (2.0 * (LAMBDA_PA + MU_PA))  # = 0.25

#: finite-difference step for displacement gradients [km] (50 m)
FD_STEP_KM = 0.05

# ----------------------------------------------------------------------
# Local flat-earth projection about (40.9 N, 28.5 E)
# ----------------------------------------------------------------------
REF_LON = 28.5
REF_LAT = 40.9
_KM_PER_DEG = 6371.0 * _math.pi / 180.0            # 111.195 km / deg latitude
_KM_PER_DEG_LON = _KM_PER_DEG * _math.cos(_math.radians(REF_LAT))

_EPS = 1.0e-6                    # DC3D internal epsilon [km]
_SEC_PER_YEAR = 365.25 * 86400.0


def lonlat_to_local_km(lons, lats):
    """Convert lon/lat [deg] to local (x_east, y_north) [km] about the
    reference (28.5 E, 40.9 N), cos-latitude flat-earth approximation."""
    lons = _np.asarray(lons, dtype=float)
    lats = _np.asarray(lats, dtype=float)
    return (lons - REF_LON) * _KM_PER_DEG_LON, (lats - REF_LAT) * _KM_PER_DEG


def local_km_to_lonlat(x_km, y_km):
    """Inverse of :func:`lonlat_to_local_km` (x east [km], y north [km])."""
    x_km = _np.asarray(x_km, dtype=float)
    y_km = _np.asarray(y_km, dtype=float)
    return REF_LON + x_km / _KM_PER_DEG_LON, REF_LAT + y_km / _KM_PER_DEG


# ======================================================================
# Okada (1992) DC3D internal displacement -- pure-numpy transcription
# ======================================================================
# Verbatim transcription (displacement parts only, i.e. U(1..3) of the
# reference code) of subroutines DCCON0/DCCON2/UA/UB/UC and the real +
# image source assembly of subroutine DC3D from DC3D.f (Y. Okada,
# Sep. 1991 / May 2002).  x is along strike, y is horizontal left of
# strike (the fault dips toward -y), z is up; the fault plane spans
# al1 <= xi <= al2 along strike and aw1 <= eta <= aw2 up-dip, with the
# reference point (xi = eta = 0) at depth `depth` (>= 0, km).
# disl1 = strike-slip (positive: hanging wall moves +strike, i.e.
# left-lateral), disl2 = dip-slip (positive: hanging wall moves up-dip,
# i.e. reverse).  Lengths in km; displacement in units of dislocation.


def _dip_trig(dip_deg):
    """sin/cos of dip with DC3D's DCCON0 rounding (|cos| < 1e-6 -> exactly
    vertical)."""
    sd = _math.sin(_math.radians(dip_deg))
    cd = _math.cos(_math.radians(dip_deg))
    if abs(cd) < _EPS:
        cd = 0.0
        sd = 1.0 if sd > 0.0 else -1.0
    return sd, cd


def _round0(a):
    return _np.where(_np.abs(a) < _EPS, 0.0, a)


def _dccon2(xi, et, q, sd, cd):
    """Station geometry constants for a finite source (subroutine DCCON2),
    vectorized. The R+xi / R+eta singular branches (KXI/KET flags in the
    Fortran) are handled pointwise with np.where."""
    xi2 = xi * xi
    et2 = et * et
    q2 = q * q
    r2 = xi2 + et2 + q2
    r = _np.sqrt(r2)
    r3 = r * r2
    y = et * cd + q * sd
    d = et * sd - q * cd

    q_safe = _np.where(q == 0.0, 1.0, q)
    tt = _np.where(q == 0.0, 0.0, _np.arctan(xi * et / (q_safe * r)))

    rxi = r + xi
    kxi = rxi < _EPS                      # only possible for xi < 0
    rxi_safe = _np.where(kxi, 1.0, rxi)
    rmxi_safe = _np.where(kxi, r - xi, 1.0)
    alx = _np.where(kxi, -_np.log(rmxi_safe), _np.log(rxi_safe))
    x11 = _np.where(kxi, 0.0, 1.0 / (r * rxi_safe))
    x32 = _np.where(kxi, 0.0, (r + rxi_safe) * x11 * x11 / r)

    ret = r + et
    ket = ret < _EPS                      # only possible for et < 0
    ret_safe = _np.where(ket, 1.0, ret)
    rmet_safe = _np.where(ket, r - et, 1.0)
    ale = _np.where(ket, -_np.log(rmet_safe), _np.log(ret_safe))
    y11 = _np.where(ket, 0.0, 1.0 / (r * ret_safe))
    y32 = _np.where(ket, 0.0, (r + ret_safe) * y11 * y11 / r)

    return {
        "xi2": xi2, "q2": q2, "r": r, "r2": r2, "r3": r3,
        "y": y, "d": d, "tt": tt,
        "alx": alx, "ale": ale,
        "x11": x11, "y11": y11, "x32": x32, "y32": y32,
    }


def _ua(xi, et, q, c2, sd, cd, alp1, alp2, disl1, disl2):
    """Part-A (infinite-medium) displacement, subroutine UA, U(1..3)."""
    two_pi = 2.0 * _math.pi
    r = c2["r"]
    xy = xi * c2["y11"]
    qx = q * c2["x11"]
    qy = q * c2["y11"]
    u1 = _np.zeros_like(r)
    u2 = _np.zeros_like(r)
    u3 = _np.zeros_like(r)
    if disl1 != 0.0:
        du1 = c2["tt"] / 2.0 + alp2 * xi * qy
        du2 = alp2 * q / r
        du3 = alp1 * c2["ale"] - alp2 * q * qy
        f = disl1 / two_pi
        u1 += f * du1
        u2 += f * du2
        u3 += f * du3
    if disl2 != 0.0:
        du1 = alp2 * q / r
        du2 = c2["tt"] / 2.0 + alp2 * et * qx
        du3 = alp1 * c2["alx"] - alp2 * q * qx
        f = disl2 / two_pi
        u1 += f * du1
        u2 += f * du2
        u3 += f * du3
    return u1, u2, u3


def _ub(xi, et, q, c2, sd, cd, alp3, disl1, disl2):
    """Part-B (free-surface correction) displacement, subroutine UB,
    U(1..3)."""
    two_pi = 2.0 * _math.pi
    r = c2["r"]
    d = c2["d"]
    y = c2["y"]
    ale = c2["ale"]
    x11 = c2["x11"]
    y11 = c2["y11"]
    xi2 = c2["xi2"]
    q2 = c2["q2"]
    sdcd = sd * cd

    rd = r + d
    d11 = 1.0 / (r * rd)
    aj2 = xi * y / rd * d11
    aj5 = -(d + y * y / rd) * d11
    if cd != 0.0:
        cdcd = cd * cd
        xx = _np.sqrt(xi2 + q2)
        xi_safe = _np.where(xi == 0.0, 1.0, xi)
        ai4 = _np.where(
            xi == 0.0,
            0.0,
            (1.0 / cdcd)
            * (xi / rd * sdcd
               + 2.0 * _np.arctan((et * (xx + q * cd) + xx * (r + xx) * sd)
                                  / (xi_safe * (r + xx) * cd))),
        )
        ai3 = (y * cd / rd - ale + sd * _np.log(rd)) / cdcd
        ak1 = xi * (d11 - y11 * sd) / cd
        ak3 = (q * y11 - y * d11) / cd
        aj3 = (ak1 - aj2 * sd) / cd
        aj6 = (ak3 - aj5 * sd) / cd
    else:
        rd2 = rd * rd
        ai3 = (et / rd + y * q / rd2 - ale) / 2.0
        ai4 = xi * y / rd2 / 2.0
        ak1 = xi * q / rd * d11
        ak3 = sd / rd * (xi2 * d11 - 1.0)
        aj3 = -xi / rd2 * (q2 * d11 - 0.5)
        aj6 = -y / rd2 * (xi2 * d11 - 0.5)

    ai1 = -xi / rd * cd - ai4 * sd
    ai2 = _np.log(rd) + ai3 * sd

    qx = q * x11
    qy = q * y11
    u1 = _np.zeros_like(r)
    u2 = _np.zeros_like(r)
    u3 = _np.zeros_like(r)
    if disl1 != 0.0:
        du1 = -xi * qy - c2["tt"] - alp3 * ai1 * sd
        du2 = -q / r + alp3 * y / rd * sd
        du3 = q * qy - alp3 * ai2 * sd
        f = disl1 / two_pi
        u1 += f * du1
        u2 += f * du2
        u3 += f * du3
    if disl2 != 0.0:
        du1 = -q / r + alp3 * ai3 * sdcd
        du2 = -et * qx - c2["tt"] - alp3 * xi / rd * sdcd
        du3 = q * qx + alp3 * ai4 * sdcd
        f = disl2 / two_pi
        u1 += f * du1
        u2 += f * du2
        u3 += f * du3
    return u1, u2, u3


def _uc(xi, et, q, z, c2, sd, cd, alp4, alp5, disl1, disl2):
    """Part-C (depth-multiplied) displacement, subroutine UC, U(1..3).
    ``z`` is the observation depth coordinate (<= 0)."""
    two_pi = 2.0 * _math.pi
    r = c2["r"]
    r3 = c2["r3"]
    d = c2["d"]
    y = c2["y"]
    x11 = c2["x11"]
    y11 = c2["y11"]
    x32 = c2["x32"]
    y32 = c2["y32"]
    xi2 = c2["xi2"]
    q2 = c2["q2"]

    c = d + z
    h = q * cd - z
    z32 = sd / r3 - h * y32
    xy = xi * y11
    qy = q * y11

    u1 = _np.zeros_like(r)
    u2 = _np.zeros_like(r)
    u3 = _np.zeros_like(r)
    if disl1 != 0.0:
        du1 = alp4 * xy * cd - alp5 * xi * q * z32
        du2 = alp4 * (cd / r + 2.0 * qy * sd) - alp5 * c * q / r3
        du3 = alp4 * qy * cd - alp5 * (c * et / r3 - z * y11 + xi2 * z32)
        f = disl1 / two_pi
        u1 += f * du1
        u2 += f * du2
        u3 += f * du3
    if disl2 != 0.0:
        du1 = alp4 * cd / r - qy * sd - alp5 * c * q / r3
        du2 = alp4 * y * x11 - alp5 * c * et * q * x32
        du3 = -d * x11 - xy * sd - alp5 * c * (x11 - q2 * x32)
        f = disl2 / two_pi
        u1 += f * du1
        u2 += f * du2
        u3 += f * du3
    return u1, u2, u3


def dc3d_displacement(x, y, z, depth_km, dip_deg, al1, al2, aw1, aw2,
                      disl1, disl2, alpha=ALPHA):
    """Internal displacement of a buried finite rectangular fault in a
    half-space (Okada 1992, DC3D), vectorized over observation points.

    Parameters
    ----------
    x, y, z : arrays [km]; fault-local frame (x along strike, y horizontal
        left of strike -- the fault dips toward -y, z up; z <= 0).
    depth_km : depth of the fault reference point (>= 0) [km].
    dip_deg : dip angle [deg].
    al1, al2, aw1, aw2 : fault extent along strike / up-dip [km].
    disl1, disl2 : strike-slip / dip-slip dislocation (any length unit;
        output displacement has the same unit).

    Returns
    -------
    (u1, u2, u3) : displacement components in the fault-local frame,
        same unit as the dislocations.
    """
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)
    z = _np.asarray(z, dtype=float)
    x, y, z = _np.broadcast_arrays(x, y, z)
    if _np.any(z > 1e-9):
        raise ValueError("dc3d_displacement: observation z must be <= 0")
    z = _np.minimum(z, 0.0)

    sd, cd = _dip_trig(dip_deg)
    alp1 = (1.0 - alpha) / 2.0
    alp2 = alpha / 2.0
    alp3 = (1.0 - alpha) / alpha
    alp4 = 1.0 - alpha
    alp5 = alpha

    xi_pair = [_round0(x - al1), _round0(x - al2)]

    u1 = _np.zeros_like(x)
    u2 = _np.zeros_like(x)
    u3 = _np.zeros_like(x)

    # ----- real-source contribution -----
    d = depth_km + z
    p = y * cd + d * sd
    q = _round0(y * sd - d * cd)
    et_pair = [_round0(p - aw1), _round0(p - aw2)]
    for j in (0, 1):
        for k in (0, 1):
            sign = 1.0 if j == k else -1.0
            c2 = _dccon2(xi_pair[j], et_pair[k], q, sd, cd)
            a1, a2, a3 = _ua(xi_pair[j], et_pair[k], q, c2, sd, cd,
                             alp1, alp2, disl1, disl2)
            u1 += sign * (-a1)
            u2 += sign * (-a2 * cd + a3 * sd)
            u3 += sign * (-a2 * sd - a3 * cd)

    # ----- image-source contribution -----
    d = depth_km - z
    p = y * cd + d * sd
    q = _round0(y * sd - d * cd)
    et_pair = [_round0(p - aw1), _round0(p - aw2)]
    for j in (0, 1):
        for k in (0, 1):
            sign = 1.0 if j == k else -1.0
            c2 = _dccon2(xi_pair[j], et_pair[k], q, sd, cd)
            a1, a2, a3 = _ua(xi_pair[j], et_pair[k], q, c2, sd, cd,
                             alp1, alp2, disl1, disl2)
            b1, b2, b3 = _ub(xi_pair[j], et_pair[k], q, c2, sd, cd,
                             alp3, disl1, disl2)
            c1, c2_, c3 = _uc(xi_pair[j], et_pair[k], q, z, c2, sd, cd,
                              alp4, alp5, disl1, disl2)
            du1 = a1 + b1 + z * c1
            du2 = (a2 + b2 + z * c2_) * cd - (a3 + b3 + z * c3) * sd
            du3 = (a2 + b2 - z * c2_) * sd + (a3 + b3 - z * c3) * cd
            u1 += sign * du1
            u2 += sign * du2
            u3 += sign * du3
    return u1, u2, u3


# ----------------------------------------------------------------------
# Optional okada_wrapper engine (auto-detected; usually absent/broken
# because the f2py build needs a Fortran compiler).
# ----------------------------------------------------------------------
def _detect_okada_wrapper():
    try:
        from okada_wrapper import dc3dwrapper  # noqa: PLC0415
        ok, u, g = dc3dwrapper(2.0 / 3.0, [10.0, 5.0, -10.0], 5.0, 90.0,
                               [-20.0, 20.0], [-15.0, 0.0], [1.0, 0.0, 0.0])
        if ok == 0 and _np.all(_np.isfinite(u)) and _np.all(_np.isfinite(g)):
            return dc3dwrapper
    except Exception:
        pass
    return None


_OKADA_WRAPPER_FN = _detect_okada_wrapper()


# ======================================================================
# Default rupture set (see module docstring for full citations)
# ======================================================================
def _utc(*args):
    return _dt.datetime(*args)


#: Rectangular uniform-slip sources. lat/lon = center of the TOP edge of
#: each rectangle; depth_top_km = depth of that edge; angles in degrees
#: (Aki & Richards); length/width in km; slip in m.
DEFAULT_RUPTURES: list[dict] = [
    # ---- 1999-08-17 Izmit (Kocaeli) Mw 7.6 -- 4-segment approximation
    # (Barka et al. 2002 segmentation; Cakir et al. 2003 vertical geometry,
    # ~18 km locking depth; summed Mo = 1.8e20 Nm = Mw 7.5).
    dict(name="Izmit1999_Golcuk", origin_time=_utc(1999, 8, 17, 0, 1, 39),
         lat=40.72, lon=29.75, depth_top_km=0.0, strike_deg=270.0,
         dip_deg=90.0, rake_deg=180.0, length_km=25.0, width_km=18.0,
         slip_m=4.0,
         reference=("Karamursel-Golcuk segment: Barka et al. 2002 BSSA "
                    "92(1):43-60 (max surface slip 4.7-5.2 m at Golcuk), "
                    "https://doi.org/10.1785/0120000841; Cakir et al. 2003 "
                    "GJI 155(1):93-110 (slip to ~18-20 km depth here), "
                    "https://doi.org/10.1046/j.1365-246X.2003.02001.x")),
    dict(name="Izmit1999_Izmit_Sapanca", origin_time=_utc(1999, 8, 17, 0, 1, 39),
         lat=40.72, lon=30.10, depth_top_km=0.0, strike_deg=264.0,
         dip_deg=90.0, rake_deg=180.0, length_km=26.0, width_km=18.0,
         slip_m=2.75,
         reference=("Izmit-Sapanca segment (trend N80-86E): Barka et al. "
                    "2002 BSSA 92(1):43-60 (avg ~2.5-3 m, max 3.5 m), "
                    "https://doi.org/10.1785/0120000841; Cakir et al. 2003 "
                    "GJI 155:93 (model III: dip 85N, rake 176; vertical "
                    "pure-RL fits comparably and is used here)")),
    dict(name="Izmit1999_Sakarya", origin_time=_utc(1999, 8, 17, 0, 1, 39),
         lat=40.70, lon=30.45, depth_top_km=0.0, strike_deg=278.0,
         dip_deg=90.0, rake_deg=180.0, length_km=33.0, width_km=18.0,
         slip_m=4.0,
         reference=("Sakarya (Sapanca-Akyazi/Arifiye) segment, trend "
                    "N75-85W: Barka et al. 2002 BSSA 92(1):43-60 (max "
                    "5.2 m at Arifiye); Langridge et al. 2002 BSSA "
                    "92(1):107-125; https://doi.org/10.1785/0120000841")),
    dict(name="Izmit1999_Karadere", origin_time=_utc(1999, 8, 17, 0, 1, 39),
         lat=40.74, lon=30.80, depth_top_km=0.0, strike_deg=245.0,
         dip_deg=90.0, rake_deg=180.0, length_km=27.0, width_km=18.0,
         slip_m=1.25,
         reference=("Karadere segment (trend N65E): Barka et al. 2002 BSSA "
                    "92(1):43-60 (max ~1.5 m); Hartleb et al. 2002 BSSA "
                    "92(1):67-78; https://doi.org/10.1785/0120000841")),
    # ---- 1999-11-12 Duzce Mw 7.2 (Mo-consistent slip: 3e10*40e3*20e3*2.4
    #      = 5.8e19 Nm ~ published Mo 5.0-5.5e19)
    dict(name="Duzce1999", origin_time=_utc(1999, 11, 12, 16, 57, 20),
         lat=40.77, lon=31.20, depth_top_km=0.0, strike_deg=264.0,
         dip_deg=65.0, rake_deg=-172.0, length_km=40.0, width_km=20.0,
         slip_m=2.4,
         reference=("Utkucu et al. 2003 GJI 153(1):229-241 (262/65N, "
                    "40x20 km, Mo 5.5e19, peak slip 5.96 m), "
                    "https://doi.org/10.1046/j.1365-246X.2003.01904.x; "
                    "Umutlu et al. 2004 Tectonophysics 391:315 "
                    "(264/64/-172); Burgmann et al. 2002 BSSA 92(1):161 "
                    "(N-dipping), https://doi.org/10.1785/0120000834; "
                    "Konca et al. 2010 BSSA 100(1):267-288")),
    # ---- 2019-09-26 Silivri (Marmara Sea) Mw 5.8 (Mo-consistent slip:
    #      3e10*8e3*2.5e3*1.0 = 6.0e17 Nm ~ Mw 5.8)
    dict(name="Silivri2019", origin_time=_utc(2019, 9, 26, 10, 59, 25),
         lat=40.88, lon=28.20, depth_top_km=11.0, strike_deg=281.0,
         dip_deg=60.0, rake_deg=165.0, length_km=8.0, width_km=2.5,
         slip_m=1.0,
         reference=("Karabulut et al. 2021 GJI 224(1):377-388 (preferred "
                    "plane 281/60/165 +-10, centroid 11.6 km, ~8 km "
                    "bilateral rupture at 11-13 km depth, N-dipping fault "
                    "north of the MMF), "
                    "https://doi.org/10.1093/gji/ggaa469; slip from "
                    "Mo~6.3e17 Nm (Mw 5.8) with mu=30 GPa. USGS us70005lkl")),
    # ---- 2025-04-23 Sea of Marmara (Kumburgaz segment) Mw 6.2
    #      (slip = Mo/(mu*L*W) = 2.89e18/(3e10*20e3*12e3) = 0.40 m)
    dict(name="Kumburgaz2025", origin_time=_utc(2025, 4, 23, 9, 49, 11),
         lat=40.84, lon=28.30, depth_top_km=5.0, strike_deg=84.0,
         dip_deg=81.0, rake_deg=171.0, length_km=20.0, width_km=12.0,
         slip_m=0.40,
         reference=("USGS us7000pufs W-phase MT (NP1 82.6/76.6/178, Mo "
                    "2.89e18 Nm, centroid 11.5 km), "
                    "https://earthquake.usgs.gov/earthquakes/eventpage/"
                    "us7000pufs; AFAD 2025 report (84/81/171); Ertuncay "
                    "et al. 2025 Seismica 4(2) doi:10.26443/seismica."
                    "v4i2.1773; Martinez-Garzon et al. 2025 Science "
                    "doi:10.1126/science.adz0072 (~10-20 km eastward "
                    "rupture of the Kumburgaz segment; center of the "
                    "rectangle shifted ~10 km east of the 28.19E "
                    "epicenter accordingly); Eken et al. 2025 J Seismol "
                    "29:1727-1755. No surface rupture: top at ~5 km.")),
]


_REQUIRED_KEYS = ("name", "origin_time", "lat", "lon", "depth_top_km",
                  "strike_deg", "dip_deg", "rake_deg", "length_km",
                  "width_km", "slip_m")


# ======================================================================
# Coulomb stress model
# ======================================================================
class CoulombStressModel:
    """Static Coulomb stress-transfer model (rectangular sources, uniform
    elastic half-space, Okada 1992 internal deformation).

    Parameters
    ----------
    ruptures : list of dict, optional
        Rupture dictionaries (see ``DEFAULT_RUPTURES`` for schema).
        Defaults to ``DEFAULT_RUPTURES``.
    friction : float
        Effective friction coefficient mu' (default 0.4).
    receiver_strike, receiver_dip, receiver_rake : float [deg]
        Default receiver fault plane (Aki & Richards convention).
        Defaults 265/80/180 = the Marmara main-branch right-lateral fault.
    receiver_depth_km : float [km]
        Depth at which dCFS is evaluated (default 10 km; must be >= 0.05).
    """

    def __init__(self, ruptures=None, friction=0.4, receiver_strike=265.0,
                 receiver_dip=80.0, receiver_rake=180.0,
                 receiver_depth_km=10.0):
        self.ruptures = [dict(r) for r in
                         (DEFAULT_RUPTURES if ruptures is None else ruptures)]
        for r in self.ruptures:
            missing = [k for k in _REQUIRED_KEYS if k not in r]
            if missing:
                raise ValueError(f"rupture {r.get('name', '?')!r} missing "
                                 f"keys: {missing}")
        if receiver_depth_km < FD_STEP_KM:
            raise ValueError("receiver_depth_km must be >= 0.05 km")
        self.friction = float(friction)
        self.receiver_strike = float(receiver_strike)
        self.receiver_dip = float(receiver_dip)
        self.receiver_rake = float(receiver_rake)
        self.receiver_depth_km = float(receiver_depth_km)
        #: "okada_wrapper" if a functional f2py DC3D build was detected,
        #: else "numpy". May be overridden by assignment.
        self.engine = "okada_wrapper" if _OKADA_WRAPPER_FN else "numpy"

    # ------------------------------------------------------------------
    def _rupture_weights(self, at_time, half_life_years):
        """Causal (and optionally decayed) weight per rupture."""
        if half_life_years is not None and at_time is None:
            raise ValueError("half_life_years requires at_time")
        weights = []
        for rup in self.ruptures:
            if at_time is None:
                weights.append(1.0)
                continue
            t0 = rup["origin_time"]
            if not t0 < at_time:          # strict causality
                weights.append(0.0)
                continue
            if half_life_years is None:
                weights.append(1.0)
            else:
                dt_years = (at_time - t0).total_seconds() / _SEC_PER_YEAR
                weights.append(0.5 ** (dt_years / float(half_life_years)))
        return weights

    # ------------------------------------------------------------------
    def _sum_displacement_enu(self, xe, yn, z, weighted_ruptures):
        """Weighted sum of rupture displacement fields at local points
        (xe, yn east/north [km], z up [km] <= 0). Slip is converted to km
        so displacements are in km and FD gradients are dimensionless."""
        ue = _np.zeros_like(xe)
        un = _np.zeros_like(xe)
        uz = _np.zeros_like(xe)
        for rup, w in weighted_ruptures:
            phi = _math.radians(rup["strike_deg"])
            sphi, cphi = _math.sin(phi), _math.cos(phi)
            x0, y0 = lonlat_to_local_km(rup["lon"], rup["lat"])
            dx = xe - x0
            dy = yn - y0
            x_loc = dx * sphi + dy * cphi
            y_loc = -dx * cphi + dy * sphi
            slip_km = rup["slip_m"] / 1000.0
            rake = _math.radians(rup["rake_deg"])
            disl1 = slip_km * _math.cos(rake)
            disl2 = slip_km * _math.sin(rake)
            L = rup["length_km"]
            W = rup["width_km"]
            u1, u2, u3 = dc3d_displacement(
                x_loc, y_loc, z, depth_km=rup["depth_top_km"],
                dip_deg=rup["dip_deg"], al1=-L / 2.0, al2=L / 2.0,
                aw1=-W, aw2=0.0, disl1=disl1, disl2=disl2)
            ue += w * (u1 * sphi - u2 * cphi)
            un += w * (u1 * cphi + u2 * sphi)
            uz += w * u3
        return ue, un, uz

    # ------------------------------------------------------------------
    def _gradient_enu_numpy(self, xe, yn, z, weighted_ruptures):
        """Displacement gradient G[i, j] = du_i/dx_j (dimensionless) by
        central finite differences (step FD_STEP_KM) of the summed field."""
        n = xe.shape[0]
        G = _np.zeros((n, 3, 3))
        h = FD_STEP_KM
        for axis in range(3):
            ex = h if axis == 0 else 0.0
            ey = h if axis == 1 else 0.0
            ez = h if axis == 2 else 0.0
            up = self._sum_displacement_enu(xe + ex, yn + ey, z + ez,
                                            weighted_ruptures)
            um = self._sum_displacement_enu(xe - ex, yn - ey, z - ez,
                                            weighted_ruptures)
            for i in range(3):
                G[:, i, axis] = (up[i] - um[i]) / (2.0 * h)
        return G

    # ------------------------------------------------------------------
    def _gradient_enu_okada_wrapper(self, xe, yn, z, weighted_ruptures):
        """Same gradient via okada_wrapper's analytic DC3D derivatives
        (loops over points; only used when the f2py build is functional)."""
        n = xe.shape[0]
        G = _np.zeros((n, 3, 3))
        for rup, w in weighted_ruptures:
            phi = _math.radians(rup["strike_deg"])
            sphi, cphi = _math.sin(phi), _math.cos(phi)
            R = _np.array([[sphi, -cphi, 0.0],
                           [cphi, sphi, 0.0],
                           [0.0, 0.0, 1.0]])   # local -> ENU
            x0, y0 = lonlat_to_local_km(rup["lon"], rup["lat"])
            x_loc = (xe - x0) * sphi + (yn - y0) * cphi
            y_loc = -(xe - x0) * cphi + (yn - y0) * sphi
            slip_km = rup["slip_m"] / 1000.0
            rake = _math.radians(rup["rake_deg"])
            disl = [slip_km * _math.cos(rake), slip_km * _math.sin(rake), 0.0]
            L, W = rup["length_km"], rup["width_km"]
            for i in range(n):
                ok, _u, g = _OKADA_WRAPPER_FN(
                    ALPHA, [x_loc[i], y_loc[i], z[i]], rup["depth_top_km"],
                    rup["dip_deg"], [-L / 2.0, L / 2.0], [-W, 0.0], disl)
                if ok != 0:
                    continue          # singular point: skip contribution
                G[i] += w * (R @ _np.asarray(g) @ R.T)
        return G

    # ------------------------------------------------------------------
    def stress_tensor(self, lons, lats, at_time=None, half_life_years=None,
                      depth_km=None):
        """Summed stress-change tensor [MPa] at points, ENU basis,
        tension-positive. Shape (n, 3, 3)."""
        lons = _np.atleast_1d(_np.asarray(lons, dtype=float))
        lats = _np.atleast_1d(_np.asarray(lats, dtype=float))
        if lons.shape != lats.shape:
            raise ValueError("lons and lats must have the same shape")
        depth = self.receiver_depth_km if depth_km is None else float(depth_km)
        if depth < FD_STEP_KM:
            raise ValueError("evaluation depth must be >= 0.05 km")
        xe, yn = lonlat_to_local_km(lons, lats)
        z = _np.full_like(xe, -depth)

        weights = self._rupture_weights(at_time, half_life_years)
        active = [(r, w) for r, w in zip(self.ruptures, weights) if w > 0.0]
        n = xe.shape[0]
        if not active:
            return _np.zeros((n, 3, 3))
        if self.engine == "okada_wrapper" and _OKADA_WRAPPER_FN is not None:
            G = self._gradient_enu_okada_wrapper(xe, yn, z, active)
        else:
            G = self._gradient_enu_numpy(xe, yn, z, active)
        eps = 0.5 * (G + _np.transpose(G, (0, 2, 1)))
        tr = _np.trace(eps, axis1=1, axis2=2)
        sigma = 2.0 * MU_PA * eps
        sigma[:, 0, 0] += LAMBDA_PA * tr
        sigma[:, 1, 1] += LAMBDA_PA * tr
        sigma[:, 2, 2] += LAMBDA_PA * tr
        return sigma / 1.0e6              # Pa -> MPa

    # ------------------------------------------------------------------
    @staticmethod
    def _receiver_vectors(strike_deg, dip_deg, rake_deg, n):
        """Unit vectors (ENU) of the receiver plane, broadcast to (n, 3):
        returns (slip_dir u_hat, normal n_hat). Normal points from footwall
        to hanging wall; A&R angle conventions."""
        phi = _np.radians(_np.broadcast_to(
            _np.asarray(strike_deg, dtype=float), (n,)).copy())
        dlt = _np.radians(_np.broadcast_to(
            _np.asarray(dip_deg, dtype=float), (n,)).copy())
        lam = _np.radians(_np.broadcast_to(
            _np.asarray(rake_deg, dtype=float), (n,)).copy())
        sphi, cphi = _np.sin(phi), _np.cos(phi)
        sdlt, cdlt = _np.sin(dlt), _np.cos(dlt)
        s_hat = _np.stack([sphi, cphi, _np.zeros(n)], axis=1)
        d_hat = _np.stack([cdlt * cphi, -cdlt * sphi, -sdlt], axis=1)
        n_hat = _np.stack([sdlt * cphi, -sdlt * sphi, cdlt], axis=1)
        u_hat = (_np.cos(lam)[:, None] * s_hat
                 - _np.sin(lam)[:, None] * d_hat)
        return u_hat, n_hat

    # ------------------------------------------------------------------
    def delta_cfs(self, lons, lats, at_time=None, half_life_years=None,
                  receiver_strike=None, receiver_dip=None,
                  receiver_rake=None):
        """Sum dCFS (MPa) at points, from ruptures with origin_time <
        at_time only (ALL ruptures if at_time is None).

        If ``half_life_years`` is given (requires ``at_time``), each
        rupture's contribution is scaled by 2**(-dt/half_life) with dt in
        years at ``at_time`` -- one half-life halves the contribution.
        Time-causal: a rupture contributes exactly 0 at or before its
        origin_time.

        ``receiver_strike``/``receiver_dip``/``receiver_rake`` [deg] may be
        scalars or arrays of len(lons) (per-point receiver planes, e.g.
        from :func:`nearest_segment_receivers`); ``None`` uses the model's
        defaults. dCFS = tau_rake + friction * sigma_n (tension-positive).

        Returns ``np.ndarray`` of shape ``(len(lons),)`` in MPa.
        """
        lons = _np.atleast_1d(_np.asarray(lons, dtype=float))
        lats = _np.atleast_1d(_np.asarray(lats, dtype=float))
        sigma = self.stress_tensor(lons, lats, at_time=at_time,
                                   half_life_years=half_life_years)
        n = sigma.shape[0]
        rs = self.receiver_strike if receiver_strike is None else receiver_strike
        rd = self.receiver_dip if receiver_dip is None else receiver_dip
        rr = self.receiver_rake if receiver_rake is None else receiver_rake
        u_hat, n_hat = self._receiver_vectors(rs, rd, rr, n)
        traction = _np.einsum("nij,nj->ni", sigma, n_hat)
        tau_rake = _np.einsum("ni,ni->n", traction, u_hat)
        sigma_n = _np.einsum("ni,ni->n", traction, n_hat)
        return tau_rake + self.friction * sigma_n


# ======================================================================
# Per-point receivers from the project's fault-segment catalog
# ======================================================================
def nearest_segment_receivers(lons, lats, segments_path,
                              max_distance_km=25.0,
                              default=(265.0, 80.0, 180.0)):
    """Receiver orientation (strike, dip, rake [deg]) of the NEAREST fault
    segment for each observation point.

    Reads a segment-properties JSON (e.g. the project's
    ``results/segment_properties.json``: a mapping whose values each carry
    ``strike``, ``dip``, ``rake`` [deg] and a ``coords`` lon/lat polyline).
    Distance is the 2-D distance (local flat-earth km) from the point to
    the segment's surface-trace polyline.

    Points farther than ``max_distance_km`` (default 25 km) from every
    segment FALL BACK to the regional default receiver
    (strike 265, dip 80, rake 180) -- the returned distance is still the
    true distance to the nearest segment.

    Returns
    -------
    (strike, dip, rake, distance_km) : four ``np.ndarray`` of len(lons).
    """
    lons = _np.atleast_1d(_np.asarray(lons, dtype=float))
    lats = _np.atleast_1d(_np.asarray(lats, dtype=float))
    px, py = lonlat_to_local_km(lons, lats)
    npts = px.shape[0]

    with open(segments_path) as f:
        raw = _json.load(f)
    seg_iter = raw.values() if isinstance(raw, dict) else raw

    best_d = _np.full(npts, _np.inf)
    best_strike = _np.full(npts, default[0])
    best_dip = _np.full(npts, default[1])
    best_rake = _np.full(npts, default[2])

    for seg in seg_iter:
        try:
            strike = float(seg["strike"])
            dip = float(seg["dip"])
            rake = float(seg["rake"])
            coords = seg["coords"]
        except (KeyError, TypeError, ValueError):
            continue                      # defensively skip malformed rows
        if not coords or len(coords) < 2:
            continue
        c = _np.asarray(coords, dtype=float)
        sx, sy = lonlat_to_local_km(c[:, 0], c[:, 1])
        # min distance from each point to each polyline edge
        d_seg = _np.full(npts, _np.inf)
        for a in range(len(sx) - 1):
            ax, ay = sx[a], sy[a]
            wx, wy = sx[a + 1] - ax, sy[a + 1] - ay
            ww = wx * wx + wy * wy
            if ww == 0.0:
                t = _np.zeros(npts)
            else:
                t = _np.clip(((px - ax) * wx + (py - ay) * wy) / ww, 0.0, 1.0)
            dx = px - (ax + t * wx)
            dy = py - (ay + t * wy)
            d_seg = _np.minimum(d_seg, _np.hypot(dx, dy))
        closer = d_seg < best_d
        best_d = _np.where(closer, d_seg, best_d)
        best_strike = _np.where(closer, strike, best_strike)
        best_dip = _np.where(closer, dip, best_dip)
        best_rake = _np.where(closer, rake, best_rake)

    far = best_d > max_distance_km
    best_strike[far] = default[0]
    best_dip[far] = default[1]
    best_rake[far] = default[2]
    return best_strike, best_dip, best_rake, best_d
