"""Repo-root pytest configuration.

Ignore macOS metadata sidecar files (`._*`) so pytest does not try to import them as
test modules: on some filesystems macOS writes an `._<name>.py` next to each source
file, and those match the `*_test.py` collection glob and crash collection.
"""
collect_ignore_glob = ["**/._*", "._*", "*/._*"]
