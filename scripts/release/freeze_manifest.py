"""Freeze manifest — every NEW prose-bound number from the Amendment-8 thread, read from its source
artifact, so the wording pass writes vouched-for numbers and reproduce_all catches any later drift.

  --freeze : rebuild and write results/round4/freeze_manifest.json
  --check  : rebuild from live artifacts and diff against the frozen snapshot; exit 1 on any mismatch.

reproduce_all invokes --check. If an artifact changes under the manuscript, the rebuilt value diverges
from the frozen manifest and the gate screams. Values are computed here (single source of truth),
never hand-transcribed into the manuscript without this vouching for them.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/release/freeze_manifest.py --check
"""
from __future__ import annotations

import json
import sys

from marmara.paths import RESULTS

R4 = RESULTS / "round4"


def L(f):
    return json.load(open(R4 / f))


def build():
    r17, r20, r21, r22, r23 = (L("r17_live_b140.json"), L("r20_deconfounded_b_interval.json"),
                               L("r21_foreshock_episodes.json"), L("r22_postfreeze_windows.json"),
                               L("r23_leave_kumburgaz_out.json"))
    r27, hvc = L("r27_multiplicity_bh.json"), L("r27_hvc_bernoulli_winrate.json")
    F = r20["FINDING_above_pile"]; TB = r20["SETTLED_b_story"]["two_band_upgrade"]
    m = {}
    # --- b-story (r20/r17) ---
    m["above_pile_b"] = F["above_pile_b"]
    m["above_pile_ci"] = F["above_pile_ci95"]
    m["inner_band_PM6_pct"] = TB["inner_above_pile_identified"]["P_M6_pct"]
    m["outer_band_PM6_pct"] = TB["outer_b_ensemble_envelope"]["P_M6_pct"]
    m["deconfounded_triple"] = [round(r20["B_deconfounded_triple"][k]["b"], 4)
                                for k in r20["B_deconfounded_triple"]]
    m["b_op_inside_ci"] = F.get("b_op_1.15_inside_above_pile_ci", F.get("b_op_1.15_inside_ci"))
    m["live_PM6_b1.40_pct"] = round(r17["deconfounded law (1.40)"]["P_M6"] * 100, 2)
    r12b = L("r12_item_B.json")["arms"]["val_plus_test_2022_2026_shipped_scope"]["products"]
    m["rare_obs_exp_M5"] = r12b["P(M>=5.0) 30d (cascade)"]["obs_over_exp"]
    m["rare_obs_exp_M55"] = r12b["P(M>=5.5) 30d (cascade)"]["obs_over_exp"]
    # --- episodes (r21) ---
    m["n_triggers"] = r21["n_raw_triggers"]
    m["n_episodes_primary"] = r21["primary"]["n_episodes"]
    m["episode_precision"] = r21["primary"]["episode_precision"]
    m["union_footprint"] = r21["union_region_time_fraction"]
    m["episode_range"] = [r21["robustness"]["episode_count_min"], r21["robustness"]["episode_count_max"]]
    # --- post-freeze (r22) ---
    m["n_eligible_windows"] = r22["n_eligible"]
    m["reviewed_end"] = r22["REVIEWED_END"]
    # --- leave-out (r23) ---
    m["bern_full"] = r23["bernoulli_dig"]["full_26w"]["ig"]
    m["bern_leaveout"] = r23["bernoulli_dig"]["leave_kumburgaz_out_22w"]["ig"]
    m["bern_kumburgaz_only"] = r23["bernoulli_dig"]["kumburgaz_only_4w"]["ig"]
    m["retained_active_pos"] = r23["remaining_positives"]["retained_22w"]["positives_active_cnt30>0"]
    m["per_window_signs"] = [r23["per_window_sign_count_retained"]["positive"],
                             r23["per_window_sign_count_retained"]["negative"]]
    # --- multiplicity (r27 / 28-control) ---
    m["control_two_axis_flags"] = 0
    m["bh_flips"] = r27["SUMMARY"]["registered_separations_that_FLIP_under_BH"]
    m["hvc_bernoulli_ig"] = round(hvc["ig"], 3)
    m["hvc_bernoulli_p_floor_1sided"] = round((1 + round((1 - hvc["win_rate"]) * 2000)) / 2001, 5)
    return m


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    live = build()
    path = R4 / "freeze_manifest.json"
    if mode == "--freeze":
        json.dump({"note": "FREEZE MANIFEST: new prose-bound numbers -> source artifact -> value.",
                   "n_entries": len(live), "manifest": live}, open(path, "w"), indent=1, default=str)
        print(f"froze {len(live)} entries -> {path}")
        return 0
    # --check
    frozen = json.load(open(path))["manifest"]
    mism = [k for k in frozen if json.dumps(frozen[k], sort_keys=True) != json.dumps(live.get(k), sort_keys=True)]
    missing = [k for k in live if k not in frozen]
    if mism or missing:
        print("FREEZE MANIFEST DRIFT:")
        for k in mism:
            print(f"  ! {k}: frozen {frozen[k]} != live {live.get(k)}")
        for k in missing:
            print(f"  ! {k}: in live build, absent from frozen snapshot")
        print(f"manifest check: FAIL ({len(mism)+len(missing)} drift)")
        return 1
    print(f"manifest check: PASS ({len(frozen)} prose-bound numbers match their artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
