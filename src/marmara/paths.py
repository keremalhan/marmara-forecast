"""Repository paths — all relative to the repo root, no absolute paths anywhere.
Override with the MARMARA_ROOT environment variable; else inferred from this file.
"""
import os
from pathlib import Path

ROOT = Path(os.environ["MARMARA_ROOT"]) if os.environ.get("MARMARA_ROOT") \
    else Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
DATA_EXTERNAL = DATA / "external"
SEG_PATH = str(DATA / "segment_properties.json")
STRAIN_NPZ = DATA / "marmara_strain_grid.npz"
KOERI_CSV = DATA / "koeri_events.csv"
CATALOG = RESULTS / "catalog.csv"
CATALOG_WIDE = RESULTS / "catalog_widebox.csv"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
