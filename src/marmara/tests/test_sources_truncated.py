"""Truncated self-test for registered FeatureSources (Phase 2+ leakage gate).

Ground rule: every FeatureSource MUST implement recompute_truncated() honestly and
pass the truncated-catalogue self-test. For an AVAILABLE source and a window with
start t0, recompute_truncated(cells_w, cutoff=t0) must equal add_columns(cells_w)
BIT-FOR-BIT — truncating the external data to < t0 is a no-op precisely because
every feature for window t0 uses only data with time < t0. A source whose data is
not on disk is skipped (honest-absent), not failed.

pytest-native (no-arg / parametrized, assert-based). Runs in `pytest src/marmara/tests`.
"""
import numpy as np
import pandas as pd
import pytest

from marmara.paths import RESULTS
from marmara.source_ig_test import SOURCES


def _grid():
    return pd.read_parquet(RESULTS / "grid_hybrid.parquet",
                           columns=["window", "t0", "cell_lon", "cell_lat", "ir", "ic"])


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_source_recompute_truncated_bitforbit(name):
    src = SOURCES[name]
    ok, why = src.available()
    if not ok:
        pytest.skip(f"{name}: data absent ({why})")
    g = _grid()
    wins = np.unique(np.linspace(g.window.min(), g.window.max(), 4).astype(int))
    for w in wins:
        cw = g[g.window == w].copy()
        t0 = pd.Timestamp(cw["t0"].iloc[0])
        a = src.add_columns(cw).to_numpy()
        b = src.recompute_truncated(cw, t0).to_numpy()
        assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b)), (
            f"{name} window {w} (t0={t0.date()}): recompute_truncated != add_columns "
            f"(max|dev|={np.nanmax(np.abs(a - b)):.3e}) — causality/leakage violation")
