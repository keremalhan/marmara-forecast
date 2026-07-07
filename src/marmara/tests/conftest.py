"""pytest configuration for the marmara test suite.

Two of the test modules are *module-style* gates, not pytest-native, and cannot be
collected by pytest directly:

  * test_cascade.py      — its ``test_*`` functions take positional args
                           (params, spec) and RETURN a bool; its main() calls them
                           and sys.exit()s on the result (see run_all.sh GATE 4/4).
  * test_grid_leakage.py — has no ``test_*`` functions at all; it is a self-test
                           driven entirely by main() (GATE 3/4).

Collecting them with pytest would either raise "fixture not found" or silently
ignore their bool return (pytest does not fail on a returned False). So we exclude
them from direct collection here and instead gate them through their main() in the
additive wrapper test_module_gates.py — which makes `pytest src/marmara/tests/ -x`
a COMPLETE gate (test_etas + test_stress natively, cascade + leakage via wrappers)
without modifying or weakening any existing test.

test_etas.py and test_stress.py are pytest-native (no-arg, assert-based) and run
directly.
"""
collect_ignore = ["test_cascade.py", "test_grid_leakage.py"]
