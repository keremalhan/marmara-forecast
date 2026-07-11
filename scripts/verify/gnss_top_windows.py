"""event-level sanity: the top-20 windows where gnss_traj most increased the
predicted rate (hybrid_gnss vs hybrid), with what actually happened (observed M>=3.0
events in those windows). The gain must trace to physically sensible episodes, not
one freak window. Writes results/verify/gnss_top_windows.md.
"""
import numpy as np
import pandas as pd
from marmara.paths import RESULTS
from marmara import grid as G

OUT = RESULTS; VER = OUT / "verify"; VER.mkdir(exist_ok=True)


def main():
    p = pd.read_parquet(OUT / "predictions_y30.parquet")   # has hybrid, hybrid_gnss per row+window
    if "hybrid_gnss" not in p.columns:
        (VER / "gnss_top_windows.md").write_text("hybrid_gnss column absent. GNSS not promoted.")
        return
    p = p[p["split"] == "test"].copy()
    p["delta"] = p["hybrid_gnss"] - p["hybrid"]
    # per-window mean increase in predicted P
    perwin = p.groupby(["window", "t0"]).agg(
        mean_delta=("delta", "mean"), max_delta=("delta", "max"),
        y30_pos=("y", "sum"), n=("y", "size")).reset_index().sort_values("mean_delta", ascending=False)
    top = perwin.head(20)

    # observed M>=3.0 events per window (for context)
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    L = ["# top-20 windows where gnss_traj most raised the predicted rate (y30 test)",
         "", "Per-window mean increase in predicted P (hybrid_gnss − hybrid) on the M≥3.0 test "
         "windows, with the observed M≥3.0 count and the largest event in each 30-day window.",
         "", "| rank | window | t0 | mean ΔP | max ΔP | obs M≥3.0 | max mag | note |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(top.itertuples(), 1):
        t0 = pd.Timestamp(r.t0)
        ev = cat[(cat["datetime_utc"] >= t0) & (cat["datetime_utc"] < t0 + pd.Timedelta(days=30))
                 & (cat["mag_w"] >= 3.0)]
        mx = float(ev["mag_w"].max()) if len(ev) else 0.0
        note = "active sequence" if len(ev) >= 5 else ("isolated" if len(ev) else "quiet")
        L.append(f"| {i} | {int(r.window)} | {t0.date()} | {r.mean_delta:+.4f} | {r.max_delta:+.3f} | "
                 f"{int(r.y30_pos)} | {mx:.1f} | {note} |")
    # aggregate sanity: is the gain concentrated in one window?
    share = float(top['mean_delta'].iloc[0] / max(perwin['mean_delta'].clip(lower=0).sum(), 1e-9))
    L += ["", f"Top-window share of total positive mean-ΔP across test windows: {share:.1%} "
          f"({'CONCENTRATED: inspect' if share > 0.5 else 'spread across multiple episodes: OK'})."]
    (VER / "gnss_top_windows.md").write_text("\n".join(L))
    print(f"wrote gnss_top_windows.md; top-window share {share:.1%}")
    print(top[["window", "t0", "mean_delta", "y30_pos"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
