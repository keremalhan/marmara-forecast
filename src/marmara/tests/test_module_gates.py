"""Additive pytest wrappers for the two module-style gates.

test_cascade.py and test_grid_leakage.py are gates run in the repo as
``python -m marmara.tests.<name>`` (their assertions live behind main()/sys.exit,
not pytest-collectable ``test_*`` functions). These wrappers invoke each gate's
main() and assert it succeeds, so a single `pytest src/marmara/tests/ -x` run gates
the FULL suite (test_etas + test_stress natively + cascade + leakage here). This
file only ADDS coverage; it does not modify or weaken the underlying gates.

Runtime note: the cascade gate runs a 50-catalogue reliability simulation
(~1-2 min); this is the same work run_all.sh's GATE 4/4 does.
"""
from marmara.tests import test_cascade, test_grid_leakage


def test_grid_leakage_gate():
    """GATE 3/4: causal-truncation self-test on the feature grid (writes leakage_ok.json)."""
    assert test_grid_leakage.main() == 0, "grid leakage self-test failed"


def test_cascade_gate():
    """GATE 4/4: cascade causality / reliability / cascade>=first-gen / anisotropy."""
    assert test_cascade.main() == 0, "cascade gate failed"
