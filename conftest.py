"""Repo-root pytest config (v3).

The project lives on an exFAT volume, where macOS scatters AppleDouble metadata
sidecars (`._*`). Files like `src/marmara/._source_ig_test.py` match pytest's
`*_test.py` collection glob and crash collection with
`ModuleNotFoundError: No module named 'marmara.'`. These are NOT source files;
exclude them from collection. This adds no path and weakens no test assertion —
it only stops pytest from trying to import filesystem junk.
"""
collect_ignore_glob = ["**/._*", "._*", "*/._*"]
