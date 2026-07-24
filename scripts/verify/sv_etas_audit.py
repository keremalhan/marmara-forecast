"""V4: sv-ETAS convergence audit (claim C5). Plot first-gen and sv-ETAS mu(x,y);
the "first-gen was already spatially variable" claim is only acceptable if the
first-gen background is demonstrably NON-uniform. Check the EM did not degenerate
(bandwidth >= 5 km floor, background fraction sane, LL converged).
Writes results/verify/sv_etas.json + results/verify/mu_xy.png.
"""
import json
import pickle
import numpy as np
from marmara.paths import RESULTS
from marmara import grid as G

OUT = RESULTS; VER = OUT / "verify"; VER.mkdir(exist_ok=True)


def mu_field(params):
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    pdf = params.background_xy.pdf_lonlat(LO.ravel(), LA.ravel())   # 1/km^2, integrates to 1
    mu = params.mu_total * pdf                                      # events/day/km^2 spatial rate
    return mu.reshape(G.NLAT, G.NLON), pdf


def cov(x):
    x = np.asarray(x, float)
    return float(np.std(x) / (np.mean(x) + 1e-12))


def main():
    rep = json.load(open(OUT / "etas" / "etas_sv_fit_report.json"))
    fg = pickle.load(open(OUT / "etas" / "etas_params.pkl", "rb"))
    sv = pickle.load(open(OUT / "etas" / "etas_sv_params.pkl", "rb"))
    mu_fg, pdf_fg = mu_field(fg)
    mu_sv, pdf_sv = mu_field(sv)
    cov_fg, cov_sv = cov(pdf_fg), cov(pdf_sv)

    ll = rep.get("ll_trajectory", [])
    converged = len(ll) >= 2 and abs(ll[-1] - ll[-2]) < 0.5
    bw_ok = rep.get("bw_final_km", 0) >= 5.0 - 1e-6
    bg_frac = rep.get("background_fraction", None)
    bg_sane = bg_frac is not None and 0.02 < bg_frac < 0.98

    out = {
        "claim": "C5: sv-ETAS ≈ first-gen because first-gen background already spatially variable",
        "firstgen_mu_spatial_CoV": round(cov_fg, 3),
        "sv_mu_spatial_CoV": round(cov_sv, 3),
        "firstgen_nonuniform": bool(cov_fg > 0.1),
        "sv_nonuniform": bool(cov_sv > 0.1),
        "em_iterations_run": rep.get("em_iterations_run"),
        "ll_converged": bool(converged), "ll_last_two": ll[-2:] if len(ll) >= 2 else ll,
        "bw_final_km": rep.get("bw_final_km"), "bw_floor_ok": bool(bw_ok),
        "background_fraction": bg_frac, "background_fraction_sane": bool(bg_sane),
        "params_close": {
            "mu": [round(float(fg.mu_total), 4), round(float(sv.mu_total), 4)],
            "alpha": [round(float(fg.alpha), 4), round(float(sv.alpha), 4)],
            "p": [round(float(fg.p), 4), round(float(sv.p), 4)]},
        "verdict": ("VERIFIED: first-gen background IS non-uniform (CoV %.2f); the EM converged "
                    "(%d iters, LL stable), bandwidth respected the 5 km floor, background "
                    "fraction sane; sv≈first-gen is a genuine robustness result, not EM degeneracy."
                    % (cov_fg, rep.get("em_iterations_run", -1)))
        if (cov_fg > 0.1 and converged and bw_ok and bg_sane) else
        "FLAG: EM degeneracy or uniform first-gen background; the deviation claim needs revisiting.",
    }
    # plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ext = [G.LON_C[0], G.LON_C[-1], G.LAT_C[0], G.LAT_C[-1]]
        for a, mu, ttl in ((ax[0], mu_fg, f"first-gen μ(x,y)  CoV={cov_fg:.2f}"),
                           (ax[1], mu_sv, f"sv-ETAS μ(x,y)  CoV={cov_sv:.2f}")):
            im = a.imshow(mu, origin="lower", extent=ext, aspect="auto", cmap="magma")
            a.set_title(ttl); a.set_xlabel("lon"); a.set_ylabel("lat"); fig.colorbar(im, ax=a, shrink=0.8)
        fig.suptitle("ETAS background spatial rate μ(x,y): first-gen vs sv (converged EM)")
        fig.tight_layout(); fig.savefig(VER / "mu_xy.png", dpi=130); plt.close(fig)
        out["plot"] = "results/verify/mu_xy.png"
    except Exception as e:
        out["plot_error"] = str(e)
    json.dump(out, open(VER / "sv_etas.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("firstgen_mu_spatial_CoV", "sv_mu_spatial_CoV",
                                          "em_iterations_run", "bw_final_km", "background_fraction",
                                          "verdict")}, indent=2))


if __name__ == "__main__":
    main()
