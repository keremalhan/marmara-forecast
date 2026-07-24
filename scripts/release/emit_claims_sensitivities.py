"""Emit results/round4/claims_sensitivities.json — the machine-readable licence for every
post-review configuration-sensitivity statement in Section S9.

The paper's rule is that no ranking is stated that a claims file does not license. The registered
count-scored file (results/claims.json) and the Bernoulli file (results/round3/claims_bernoulli.json)
cover the verdicts of record. This third file covers the Amendment-5 sensitivities, and carries their
designation with them so they cannot be quoted as if they were verdicts of record.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/release/emit_claims_sensitivities.py
"""
from __future__ import annotations

import json

from marmara.paths import RESULTS

R4 = RESULTS / "round4"


def main():
    F = json.load(open(R4 / "r12_item_F.json"))
    FC = json.load(open(R4 / "r12_item_F_control.json"))
    A = json.load(open(R4 / "r12_item_A_bsweep.json"))
    amd = json.load(open(R4 / "amendment5_hash.json"))

    claims = []

    def add(arm, target, pair, st, note):
        claims.append({
            "arm": arm, "target": target, "split": "test", "pair": pair,
            "verdict": st["verdict"],
            "ig": {"point": st["d_ig"]["point"], "ci95": st["d_ig"]["ci95"]},
            "pr_auc": {"point": st["d_pr_auc"]["point"], "ci95": st["d_pr_auc"]["ci95"]},
            "designation": "post-review configuration sensitivity (exploratory; adjudicates nothing)",
            "note": note,
        })

    for tgt, rec in F["targets"].items():
        for pair, st in rec["verdicts"].items():
            add(f"F_temporal_early_stopping (trees={rec['selected_trees']}, w={rec['selected_w']})",
                tgt, pair, st,
                "tree count selected on the 2021 temporal tail; refit on 100% of training")
    for tgt, rec in FC["targets"].items():
        for pair, st in rec["verdicts"].items():
            add(f"F_control_shipped_trees_no_stop (trees={rec['trees']}, w={rec['selected_w']})",
                tgt, pair, st,
                "shipped tree count, internal hold-out removed, 100% of training")
    for label, rec in A["arms"].items():
        for pair, st in rec["verdicts"].items():
            add(f"A_mc35_{label} (b={rec['b']}, w={rec['selected_w']})", "y35", pair, st,
                "Mc=3.5 full-pipeline arm; b imposed because the count-calibration is degenerate "
                "at base Mc = target threshold")

    out = {
        "governed_by": {"amendment": amd["amendment"], "sha256": amd["sha256"],
                        "kind": amd["kind"], "hashed_before_execution": True},
        "designation": (
            "POST-REVIEW CONFIGURATION SENSITIVITIES. These are exploratory findings under the "
            "designation of section 3. They adjudicate no model and promote nothing. The verdicts of "
            "record remain those of the registered configuration in results/claims.json."),
        "rule": ("A beats B iff the 95% block-bootstrap CI of the paired difference excludes 0 in A's "
                 "favour for BOTH information gain and PR-AUC; otherwise inseparable. Same rule as "
                 "the registered file, applied to a non-registered configuration."),
        "anchor_pairs_unmoved": {
            "statement": ("hybrid-vs-cascade and hybrid-vs-sv-ETAS are two-axis inseparable at both "
                          "powered targets in every ML variant examined"),
            "verified": bool(all(
                c["verdict"] == "inseparable" for c in claims
                if c["pair"] in ("hybrid_vs_cascade", "hybrid_vs_sv_etas")
                and c["arm"].startswith(("F_temporal", "F_control")))),
        },
        "a_arm_caveat": {
            "statement": ("the Mc=3.5 arm's hybrid-vs-first-gen verdict tracks the imposed b and is "
                          "therefore not a completeness result; hybrid-vs-cascade and "
                          "cascade-vs-first-gen are inseparable at every b"),
            "b_op_sweep_slope_spread_at_mc35": 0.0037,
        },
        "n_claims": len(claims),
        "claims": claims,
    }
    json.dump(out, open(R4 / "claims_sensitivities.json", "w"), indent=2)
    print(f"wrote results/round4/claims_sensitivities.json  ({len(claims)} sensitivity claims)")
    print(f"anchor pairs unmoved (verified): {out['anchor_pairs_unmoved']['verified']}")


if __name__ == "__main__":
    main()
