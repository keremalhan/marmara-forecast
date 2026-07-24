"""Multi-source feature-extension harness: base contract.

A FeatureSource turns some external dataset into extra columns on the existing
leakage-gated (cell, 30-day-window) grid, so its information gain can be tested
with the same harness that already scores the model. The contract:

  * CAUSAL: add_columns(cells) may use ONLY external records with time < t0 for
    each row's window start t0. recompute_truncated() lets the harness verify this
    by rebuilding a row from a copy of the external data truncated to < t0.
  * SELF-DESCRIBING: data_spec says exactly what file/format/provenance is needed,
    so a future session drops the data in and re-runs, no guessing.
  * DECLARED-ABSENT: available() returns (False, reason) when the data is not on disk;
    the harness then SKIPS the source. It never fabricates columns.

Nothing here touches the validated core code or grid_hybrid.parquet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from marmara.paths import RESULTS, DATA_EXTERNAL as DATA

OUT = RESULTS


class FeatureSource:
    name: str = "base"
    columns: list[str] = []
    data_spec: dict = {}

    def available(self) -> tuple[bool, str]:
        """Return (True, '') if the required data is on disk, else (False, reason)."""
        raise NotImplementedError

    def add_columns(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame (same row order as `cells`) with `self.columns`,
        each strictly causal (row uses only external data with time < row.t0).
        `cells` has at least: window, t0 (datetime64), cell_lon, cell_lat, ir, ic."""
        raise NotImplementedError

    def recompute_truncated(self, cells: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Recompute `columns` using external data truncated to < cutoff. Used by the
        leakage self-test: for rows whose t0 == cutoff, this must equal add_columns.
        Default: sources that only ever read < t0 can reuse add_columns unchanged."""
        return self.add_columns(cells)


def causal_corr_check(new_cols: pd.DataFrame, targets: pd.DataFrame,
                      thresh: float = 0.999) -> dict:
    """A new column must not be a near-copy of a target (that would be leakage)."""
    out = {}
    for c in new_cols.columns:
        x = new_cols[c].to_numpy(float)
        row = {}
        for t in targets.columns:
            y = targets[t].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)         # drop NaN pairs
            if m.sum() < 2 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                r = 0.0
            else:
                r = float(abs(np.corrcoef(x[m], y[m])[0, 1]))
            row[t] = r
        out[c] = row
    max_r = max((max(v.values()) for v in out.values()), default=0.0)
    return {"per_column": out, "max_abs_corr": max_r, "passes": bool(max_r <= thresh)}
