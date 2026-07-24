"""Verify the pre-registration chain: SHA-256 of every docs/preregistration document
against its recorded pre-execution hash, and document the one known discrepancy
(RUNPLAN_round3.md, edited after hashing; no unedited copy survives) so the archive
states it machine-checkably instead of leaving it for a referee to find.

Writes results/audit/preregistration_chain.json; reproduce-all asserts its outcome.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/preregistration_chain.py
"""
import hashlib
import json

from marmara.paths import RESULTS, ROOT

DOCS = ROOT / "docs" / "preregistration"

RECORDS = {
    "v2_preregistration.md": (RESULTS / "audit" / "v2_preregistration.json", "protocol_sha256"),
    "RUNPLAN_round3.md": (RESULTS / "round3" / "RUNPLAN_hash.json", "sha256"),
    "v2_preregistration_amendment_3.md": (RESULTS / "round3" / "amendment3_hash.json", "sha256"),
    "v2_preregistration_amendment_4.md": (RESULTS / "round4" / "amendment4_hash.json", "sha256"),
    "v2_analysis_amendment_5.md": (RESULTS / "round4" / "amendment5_hash.json", "sha256"),
    "v2_analysis_amendment_6.md": (RESULTS / "round4" / "amendment6_hash.json", "sha256"),
    "v2_analysis_amendment_7.md": (RESULTS / "round4" / "amendment7_hash.json", "sha256"),
    "v2_analysis_amendment_8.md": (RESULTS / "round4" / "amendment8_hash.json", "sha256"),
}
NO_RECORD = ["v2_preregistration_amendment_2.md", "v2_preregistration_amendment_3_addendum.md"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    files = {}
    # amendment 1's record lives in the prospective log
    a1_rec = None
    for line in open(RESULTS / "prospective" / "forecast_log.jsonl"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("event", "").startswith("amendment_1"):
            a1_rec = d.get("amendment_sha256")
    files["v2_preregistration_amendment_1.md"] = {
        "sha256": sha(DOCS / "v2_preregistration_amendment_1.md"),
        "record": "results/prospective/forecast_log.jsonl", "recorded": a1_rec}
    for name, (rec_path, key) in RECORDS.items():
        files[name] = {"sha256": sha(DOCS / name),
                       "record": str(rec_path.relative_to(ROOT)),
                       "recorded": json.load(open(rec_path)).get(key)}
    for name in NO_RECORD:
        files[name] = {"sha256": sha(DOCS / name), "record": None, "recorded": None,
                       "note": "dated document; no pre-execution hash was recorded"}
    for f in files.values():
        f["verifies"] = (f["sha256"] == f["recorded"]) if f["recorded"] else None
    run = files["RUNPLAN_round3.md"]
    out = {
        "files": files,
        "n_with_record": sum(1 for f in files.values() if f["recorded"]),
        "n_verified": sum(1 for f in files.values() if f["verifies"]),
        "n_no_record": sum(1 for f in files.values() if not f["recorded"]),
        "runplan_discrepancy": {
            "recorded": run["recorded"], "actual": run["sha256"], "documented": True,
            "note": ("edited after hashing at an unreconstructable point before the first "
                     "commit tracking the file; no unedited copy survives in the working "
                     "repositories or the deposited 1.1.0 archive. Table S8 mirrors the "
                     "pre-registered lines and is the operative registration record.")},
    }
    (RESULTS / "audit" / "preregistration_chain.json").write_text(json.dumps(out, indent=1))
    print(f"chain: {out['n_verified']}/{out['n_with_record']} recorded hashes verify, "
          f"{out['n_no_record']} without records; RUNPLAN discrepancy documented")


if __name__ == "__main__":
    main()
